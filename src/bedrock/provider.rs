//! Concrete [`ChatProvider`] for Amazon Bedrock — the COMPOSITION layer.
//!
//! Everything the gateway needs to talk to Bedrock Converse already exists as
//! small, pure, individually-tested pieces (Wave 3):
//! - [`crate::bedrock::translate`] — OpenAI → Converse `ConverseArgs` (messages,
//!   system, inferenceConfig, additionalModelRequestFields, the reasoning/tool
//!   seam via [`ConverseExtras`]).
//! - [`crate::bedrock::reasoning`] — `reasoning_effort` → reasoning fields +
//!   `maxTokens`/`topP` side-signals.
//! - [`crate::bedrock::tools`] — tool spec conversion, `tool_choice` mapping
//!   (with the de-hardcoded `skip_tool_choice`), tool-result normalization, and
//!   the placeholder-toolConfig safety net.
//! - [`crate::bedrock::cache`] — `extra_body.prompt_caching` extraction + system
//!   / message `cachePoint` decoration.
//! - [`crate::bedrock::client`] — shared clients + per-request region override.
//! - [`crate::bedrock::response`] / [`crate::bedrock::stream`] — Converse output
//!   → OpenAI `ChatResponse` / `ChatStream`.
//!
//! This module does NOT re-implement any of that. [`BedrockChatProvider`]
//! *orchestrates* them in the exact order the legacy Python `_parse_request` /
//! `chat` / `chat_stream` did, then performs the one piece of genuinely new
//! integration work: mapping the parity-critical `ConverseArgs` JSON slots into
//! the typed aws-sdk Converse builder, and mapping the typed SDK output back
//! into the [`serde_json::Value`] shape [`crate::bedrock::response::from_converse_output`]
//! consumes.
//!
//! ## JSON ↔ SDK bridge (the real integration work)
//!
//! [`ConverseArgs`] models each Converse slot as a `serde_json::Value` so the
//! pure translation stays decoupled from the SDK. The aws-sdk Converse builder,
//! however, wants typed `Vec<Message>` / `Vec<SystemContentBlock>` /
//! `InferenceConfiguration` / `ToolConfiguration` / `Document`. Rather than make
//! `translate` produce SDK types (a much larger, parity-risky change), this
//! module owns a focused, well-tested JSON→SDK adapter:
//! - [`json_to_document`] converts any `serde_json::Value` into the
//!   `aws_smithy_types::Document` the SDK uses for free-form fields
//!   (`additionalModelRequestFields`, `toolUse.input`, `toolResult` JSON,
//!   `inputSchema.json`).
//! - [`build_sdk_messages`] / [`build_sdk_system`] / [`build_sdk_inference_config`] /
//!   [`build_sdk_tool_config`] build the typed request pieces from the JSON
//!   shapes this codebase itself produces in `translate` / `tools` / `cache`.
//!
//! The shapes consumed here are exactly the shapes those modules emit (the
//! Bedrock Converse wire names), so the adapter is total over the gateway's own
//! output and rejects anything malformed with an [`AppError`].
//!
//! ## llama `tool_choice` skip
//!
//! Bedrock rejects `toolChoice` for some models (the legacy Python special-cased
//! `model.startswith("meta.llama3-1-")`). To keep model knowledge out of code,
//! [`BedrockChatProvider`] computes `skip_tool_choice` at THIS composition layer
//! via a single documented predicate ([`skip_tool_choice_for`]) that inspects
//! the resolved foundation model id. This is the one and only place a model-id
//! shape is matched, and it is isolated + documented here per the task contract.
//! (If a capability flag is later added to `config/models.toml`, this predicate
//! becomes a one-line `caps.has(...)` call with no other changes.)

use std::sync::Arc;

use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, CachePointBlock, CachePointType, ContentBlock, ConversationRole,
    ImageBlock, ImageFormat, ImageSource, InferenceConfiguration, JsonSchemaDefinition, Message,
    OutputConfig, OutputFormat, OutputFormatStructure, OutputFormatType, SpecificToolChoice,
    SystemContentBlock, Tool, ToolChoice, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_smithy_types::{Blob, Document, Number};
use serde_json::{Map, Value};

use aws_sdk_bedrockruntime::operation::converse::{ConverseError, ConverseOutput};
use aws_sdk_bedrockruntime::operation::converse_stream::{
    ConverseStreamError, ConverseStreamOutput,
};

use crate::bedrock::cache::PromptCachingControl;
use crate::bedrock::cache_support::{send_with_cache_strip_retry, CacheSupportRegistry, SendError};
use crate::bedrock::capabilities::normalize_for_match;
use crate::bedrock::client::{region_config_override, BedrockClients};
use crate::bedrock::translate::{to_converse_args, ConverseArgs, ConverseExtras, ImageResolver};
use crate::bedrock::{cache, reasoning, response, stream, tools};
use crate::config::{AppSettings, RegionRoutingConfig};
use crate::domain::{
    ChatProvider, ChatStream, ModelCapabilities, NormalizedChatRequest, RouteOverride,
};
use crate::error::AppError;
use crate::openai::schema::{ChatRequest, ChatResponse};

// ===========================================================================
// JSON → aws-sdk Document
// ===========================================================================

/// Convert a [`serde_json::Value`] into an [`aws_smithy_types::Document`].
///
/// `Document` is the SDK's free-form value type used for
/// `additionalModelRequestFields`, `toolUse.input`, `toolResult` JSON content,
/// and `inputSchema.json`. The mapping is total and lossless for JSON:
/// objects/arrays recurse; integers preferentially map to `PosInt`/`NegInt`
/// (so they don't become floats); other numbers map to `Float`.
#[must_use]
pub fn json_to_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(b) => Document::Bool(*b),
        Value::String(s) => Document::String(s.clone()),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(Number::NegInt(i))
            } else {
                // f64 is the only remaining JSON number representation.
                Document::Number(Number::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        Value::Array(arr) => Document::Array(arr.iter().map(json_to_document).collect()),
        Value::Object(obj) => Document::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect(),
        ),
    }
}

/// Convert an [`aws_smithy_types::Document`] back into a [`serde_json::Value`].
///
/// The inverse of [`json_to_document`], used by [`converse_output_to_json`] to
/// lower the SDK's typed `toolUse.input` (a `Document`) back into the JSON the
/// pure [`response::from_converse_output`] mapper consumes.
#[must_use]
pub fn document_to_json(doc: &Document) -> Value {
    match doc {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::String(s) => Value::String(s.clone()),
        Document::Number(n) => match n {
            Number::PosInt(u) => Value::Number((*u).into()),
            Number::NegInt(i) => Value::Number((*i).into()),
            Number::Float(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        },
        Document::Array(arr) => Value::Array(arr.iter().map(document_to_json).collect()),
        Document::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect(),
        ),
    }
}

// ===========================================================================
// ConverseArgs JSON slots → typed SDK request pieces
// ===========================================================================

/// Read a required string field from a JSON object, erroring on absence.
fn require_str<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<&'a str, AppError> {
    obj.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::Internal(format!("converse arg missing string field `{key}`")))
}

/// Build the SDK `ImageBlock` from a translate `{"format","source":{"bytes":[...]}}`.
fn build_image_block(image: &Value) -> Result<ImageBlock, AppError> {
    let format = image
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::Internal("image block missing `format`".to_string()))?;
    let bytes = image
        .get("source")
        .and_then(|s| s.get("bytes"))
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Internal("image block missing `source.bytes`".to_string()))?;
    let raw: Vec<u8> = bytes
        .iter()
        .map(|b| b.as_u64().unwrap_or(0) as u8)
        .collect();
    ImageBlock::builder()
        .format(ImageFormat::from(format))
        .source(ImageSource::Bytes(Blob::new(raw)))
        .build()
        .map_err(|e| AppError::Internal(format!("invalid image block: {e}")))
}

/// Build an SDK `ToolUseBlock` from a translate `toolUse` JSON object.
fn build_tool_use_block(tu: &Value) -> Result<ToolUseBlock, AppError> {
    let obj = tu
        .as_object()
        .ok_or_else(|| AppError::Internal("toolUse must be an object".to_string()))?;
    let id = require_str(obj, "toolUseId")?;
    let name = require_str(obj, "name")?;
    let input = obj.get("input").cloned().unwrap_or(Value::Null);
    ToolUseBlock::builder()
        .tool_use_id(id)
        .name(name)
        .input(json_to_document(&input))
        .build()
        .map_err(|e| AppError::Internal(format!("invalid toolUse block: {e}")))
}

fn empty_object_tool_result_content_block() -> ToolResultContentBlock {
    ToolResultContentBlock::Json(json_to_document(&serde_json::json!({})))
}

/// Build an SDK `ToolResultBlock` from a translate `toolResult` JSON object.
///
/// The translate/tools layers always emit `content` as a list of `{"text":...}`
/// blocks; anything else is mapped through the SDK's JSON content variant so no
/// gateway-produced shape is ever lost.
fn build_tool_result_block(tr: &Value) -> Result<ToolResultBlock, AppError> {
    let obj = tr
        .as_object()
        .ok_or_else(|| AppError::Internal("toolResult must be an object".to_string()))?;
    let id = require_str(obj, "toolUseId")?;
    let mut content: Vec<ToolResultContentBlock> = Vec::new();
    if let Some(blocks) = obj.get("content").and_then(Value::as_array) {
        for block in blocks {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                if text.is_empty() {
                    content.push(empty_object_tool_result_content_block());
                } else {
                    content.push(ToolResultContentBlock::Text(text.to_string()));
                }
            } else {
                content.push(ToolResultContentBlock::Json(json_to_document(block)));
            }
        }
    }
    if content.is_empty() {
        content.push(empty_object_tool_result_content_block());
    }
    ToolResultBlock::builder()
        .tool_use_id(id)
        .set_content(Some(content))
        .build()
        .map_err(|e| AppError::Internal(format!("invalid toolResult block: {e}")))
}

/// Convert one translate content block (`{"text"|"image"|"toolUse"|"toolResult"|"cachePoint": ...}`)
/// into an SDK [`ContentBlock`].
fn build_content_block(block: &Value) -> Result<ContentBlock, AppError> {
    let obj = block
        .as_object()
        .ok_or_else(|| AppError::Internal("content block must be an object".to_string()))?;
    if let Some(text) = obj.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return Err(AppError::Internal(
                "empty text content block reached SDK bridge".to_string(),
            ));
        }
        return Ok(ContentBlock::Text(text.to_string()));
    }
    if let Some(image) = obj.get("image") {
        return Ok(ContentBlock::Image(build_image_block(image)?));
    }
    if let Some(tu) = obj.get("toolUse") {
        return Ok(ContentBlock::ToolUse(build_tool_use_block(tu)?));
    }
    if let Some(tr) = obj.get("toolResult") {
        return Ok(ContentBlock::ToolResult(build_tool_result_block(tr)?));
    }
    if obj.contains_key("cachePoint") {
        return Ok(ContentBlock::CachePoint(
            CachePointBlock::builder()
                .r#type(CachePointType::Default)
                .build()
                .map_err(|e| AppError::Internal(format!("invalid cachePoint: {e}")))?,
        ));
    }
    Err(AppError::Internal(format!(
        "unrecognized content block shape: {block}"
    )))
}

fn is_empty_text_block(block: &Value) -> bool {
    block.get("text").and_then(Value::as_str) == Some("")
}

/// Build the typed SDK `messages` from the `ConverseArgs::messages` JSON array.
///
/// `pub(crate)` so the Responses provider ([`crate::bedrock::responses_provider`])
/// can reuse the same JSON→SDK bridge for its own Converse call.
pub(crate) fn build_sdk_messages(messages: &Value) -> Result<Vec<Message>, AppError> {
    let turns = messages
        .as_array()
        .ok_or_else(|| AppError::Internal("converse messages must be an array".to_string()))?;
    let mut out = Vec::with_capacity(turns.len());
    for turn in turns {
        let role = turn
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| AppError::Internal("message turn missing `role`".to_string()))?;
        let role = match role {
            "user" => ConversationRole::User,
            "assistant" => ConversationRole::Assistant,
            other => ConversationRole::from(other),
        };
        let content_json = turn
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                AppError::Internal("message turn missing `content` array".to_string())
            })?;
        let mut content = Vec::with_capacity(content_json.len());
        for block in content_json {
            if is_empty_text_block(block) {
                continue;
            }
            content.push(build_content_block(block)?);
        }
        if content.is_empty() {
            return Err(AppError::Internal(
                "message turn content contained no SDK content blocks".to_string(),
            ));
        }
        let message = Message::builder()
            .role(role)
            .set_content(Some(content))
            .build()
            .map_err(|e| AppError::Internal(format!("invalid message: {e}")))?;
        out.push(message);
    }
    Ok(out)
}

/// Build the typed SDK `system` blocks from the `ConverseArgs::system` JSON
/// array (entries are `{"text": ...}` or `{"cachePoint": ...}`). `pub(crate)`
/// for reuse by the Responses provider.
pub(crate) fn build_sdk_system(system: &Value) -> Result<Vec<SystemContentBlock>, AppError> {
    let blocks = system
        .as_array()
        .ok_or_else(|| AppError::Internal("converse system must be an array".to_string()))?;
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            if !text.is_empty() {
                out.push(SystemContentBlock::Text(text.to_string()));
            }
        } else if block.get("cachePoint").is_some() {
            out.push(SystemContentBlock::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .map_err(|e| AppError::Internal(format!("invalid cachePoint: {e}")))?,
            ));
        } else {
            return Err(AppError::Internal(format!(
                "unrecognized system block shape: {block}"
            )));
        }
    }
    Ok(out)
}

/// Build the SDK `InferenceConfiguration` from the `ConverseArgs::inference_config`
/// JSON object (`maxTokens`/`temperature`/`topP`/`stopSequences`). `pub(crate)`
/// for reuse by the Responses provider.
pub(crate) fn build_sdk_inference_config(cfg: &Value) -> InferenceConfiguration {
    let mut builder = InferenceConfiguration::builder();
    if let Some(max_tokens) = cfg.get("maxTokens").and_then(Value::as_i64) {
        builder = builder.max_tokens(max_tokens as i32);
    }
    if let Some(temp) = cfg.get("temperature").and_then(Value::as_f64) {
        builder = builder.temperature(temp as f32);
    }
    if let Some(top_p) = cfg.get("topP").and_then(Value::as_f64) {
        builder = builder.top_p(top_p as f32);
    }
    if let Some(stops) = cfg.get("stopSequences").and_then(Value::as_array) {
        let seqs: Vec<String> = stops
            .iter()
            .filter_map(|s| s.as_str().map(str::to_string))
            .collect();
        builder = builder.set_stop_sequences(Some(seqs));
    }
    builder.build()
}

/// Build the SDK `ToolConfiguration` from the `ConverseArgs::tool_config` JSON
/// object (`{"tools": [{"toolSpec": ...}], "toolChoice"?: ...}`). `pub(crate)`
/// for reuse by the Responses provider.
pub(crate) fn build_sdk_tool_config(tc: &Value) -> Result<ToolConfiguration, AppError> {
    let obj = tc
        .as_object()
        .ok_or_else(|| AppError::Internal("toolConfig must be an object".to_string()))?;
    let tools_json = obj
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Internal("toolConfig missing `tools` array".to_string()))?;

    let mut tools = Vec::with_capacity(tools_json.len());
    for tool in tools_json {
        // Check for cachePoint block first (before looking for toolSpec).
        if tool.get("cachePoint").is_some() {
            tools.push(Tool::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .map_err(|e| AppError::Internal(format!("invalid cachePoint: {e}")))?,
            ));
            continue;
        }

        let spec = tool
            .get("toolSpec")
            .ok_or_else(|| AppError::Internal("tool missing `toolSpec`".to_string()))?;
        let spec_obj = spec
            .as_object()
            .ok_or_else(|| AppError::Internal("toolSpec must be an object".to_string()))?;
        let name = require_str(spec_obj, "name")?;
        let schema_json = spec_obj
            .get("inputSchema")
            .and_then(|s| s.get("json"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
        let mut builder = ToolSpecification::builder()
            .name(name)
            .input_schema(ToolInputSchema::Json(json_to_document(&schema_json)));
        if let Some(desc) = spec_obj.get("description").and_then(Value::as_str) {
            if !desc.trim().is_empty() {
                builder = builder.description(desc);
            }
        }
        let tool_spec = builder
            .build()
            .map_err(|e| AppError::Internal(format!("invalid toolSpec: {e}")))?;
        tools.push(Tool::ToolSpec(tool_spec));
    }

    let mut config_builder = ToolConfiguration::builder().set_tools(Some(tools));

    if let Some(choice) = obj.get("toolChoice") {
        config_builder = config_builder.tool_choice(build_sdk_tool_choice(choice)?);
    }

    config_builder
        .build()
        .map_err(|e| AppError::Internal(format!("invalid toolConfig: {e}")))
}

/// Map a translate `toolChoice` JSON value onto the SDK `ToolChoice` enum.
fn build_sdk_tool_choice(choice: &Value) -> Result<ToolChoice, AppError> {
    if choice.get("any").is_some() {
        return Ok(ToolChoice::Any(AnyToolChoice::builder().build()));
    }
    if choice.get("auto").is_some() {
        return Ok(ToolChoice::Auto(AutoToolChoice::builder().build()));
    }
    if let Some(tool) = choice.get("tool") {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
        let specific = SpecificToolChoice::builder()
            .name(name)
            .build()
            .map_err(|e| AppError::Internal(format!("invalid specific toolChoice: {e}")))?;
        return Ok(ToolChoice::Tool(specific));
    }
    Err(AppError::Internal(format!(
        "unrecognized toolChoice shape: {choice}"
    )))
}

/// Build the SDK `OutputConfig` from the `ConverseArgs::output_config` JSON
/// (`{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":<string>,"name"?:...}}}}`).
/// The `schema` value is already a STRING (stringified in translation); it is
/// forwarded verbatim into `JsonSchemaDefinition.schema`.
pub(crate) fn build_sdk_output_config(oc: &Value) -> Result<OutputConfig, AppError> {
    let tf = oc
        .get("textFormat")
        .and_then(Value::as_object)
        .ok_or_else(|| AppError::Internal("outputConfig missing `textFormat`".to_string()))?;

    let js = tf
        .get("structure")
        .and_then(|s| s.get("jsonSchema"))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            AppError::Internal("outputConfig missing `structure.jsonSchema`".to_string())
        })?;

    let schema = js
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::Internal("jsonSchema.schema must be a string".to_string()))?;

    let mut def = JsonSchemaDefinition::builder().schema(schema);
    if let Some(name) = js.get("name").and_then(Value::as_str) {
        def = def.name(name);
    }
    if let Some(desc) = js.get("description").and_then(Value::as_str) {
        def = def.description(desc);
    }
    let def = def
        .build()
        .map_err(|e| AppError::Internal(format!("invalid jsonSchema: {e}")))?;

    let format = OutputFormat::builder()
        .r#type(OutputFormatType::JsonSchema)
        .structure(OutputFormatStructure::JsonSchema(def))
        .build()
        .map_err(|e| AppError::Internal(format!("invalid outputFormat: {e}")))?;

    Ok(OutputConfig::builder().text_format(format).build())
}

// ===========================================================================
// SDK ConverseOutput → response.rs JSON shape
// ===========================================================================

/// Lower the SDK `ConverseOutput` operation result into the
/// [`serde_json::Value`] shape [`response::from_converse_output`] consumes
/// (`{"output":{"message":{"content":[...]}}, "stopReason":..., "usage":{...}}`).
///
/// This is the response half of the JSON↔SDK bridge. It keeps the parity-tested
/// mapping in `response.rs` (which is pure over JSON) authoritative, while this
/// adapter only re-projects the typed SDK content blocks / usage / stopReason.
/// `pub(crate)` so the Responses provider reuses the identical output lowering.
pub(crate) fn converse_output_to_json(
    out: &aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
) -> Value {
    let mut content_blocks: Vec<Value> = Vec::new();
    if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(message)) = out.output() {
        for block in message.content() {
            match block {
                ContentBlock::Text(text) => {
                    content_blocks.push(serde_json::json!({ "text": text }));
                }
                ContentBlock::ToolUse(tu) => {
                    content_blocks.push(serde_json::json!({
                        "toolUse": {
                            "toolUseId": tu.tool_use_id(),
                            "name": tu.name(),
                            "input": document_to_json(tu.input()),
                        }
                    }));
                }
                ContentBlock::ReasoningContent(rc) => {
                    if let Ok(rt) = rc.as_reasoning_text() {
                        content_blocks.push(serde_json::json!({
                            "reasoningContent": { "reasoningText": { "text": rt.text() } }
                        }));
                    }
                }
                // Other block kinds are not produced for chat completions.
                _ => {}
            }
        }
    }

    let mut usage_obj = Map::new();
    if let Some(usage) = out.usage() {
        usage_obj.insert("inputTokens".to_string(), Value::from(usage.input_tokens()));
        usage_obj.insert(
            "outputTokens".to_string(),
            Value::from(usage.output_tokens()),
        );
        usage_obj.insert("totalTokens".to_string(), Value::from(usage.total_tokens()));
        if let Some(cr) = usage.cache_read_input_tokens() {
            usage_obj.insert("cacheReadInputTokens".to_string(), Value::from(cr));
        }
        if let Some(cw) = usage.cache_write_input_tokens() {
            usage_obj.insert("cacheWriteInputTokens".to_string(), Value::from(cw));
        }
    }

    serde_json::json!({
        "output": { "message": { "role": "assistant", "content": content_blocks } },
        "stopReason": out.stop_reason().as_str(),
        "usage": Value::Object(usage_obj),
    })
}

// ===========================================================================
// llama tool_choice skip (the ONE documented model-shape check)
// ===========================================================================

/// Decide whether `toolChoice` must be omitted for a resolved foundation model.
///
/// Bedrock rejects `toolChoice` for the Llama 3.1 family (the legacy Python
/// special-cased `model.startswith("meta.llama3-1-")`, bedrock.py:1194). Per the
/// task contract, the model-id shape check belongs at THIS composition layer and
/// is the only place a model id is inspected by shape. If a `skip_tool_choice`
/// capability is later added to `config/models.toml`, replace this body with a
/// single `caps.has(model, Capability::SkipToolChoice)` call.
#[must_use]
pub fn skip_tool_choice_for(resolved_model: &str) -> bool {
    let lower = resolved_model.to_lowercase();
    // Match the substring (not just prefix) so inference-profile-prefixed ids
    // like `us.meta.llama3-1-...` are also covered.
    lower.contains("meta.llama3-1-")
}

// ===========================================================================
// BedrockChatProvider
// ===========================================================================

/// Concrete [`ChatProvider`] backed by Amazon Bedrock Converse.
///
/// Holds the shared [`BedrockClients`], the config-driven capability resolver,
/// the region-routing table, an [`ImageResolver`] for multimodal URL fetching,
/// and the loaded [`AppSettings`] (for the global prompt-caching default). It is
/// cheap to clone (every field is `Arc`-wrapped or itself cheaply clonable) so a
/// single instance can be shared across the whole server via `Arc<dyn ChatProvider>`.
#[derive(Clone)]
pub struct BedrockChatProvider {
    clients: BedrockClients,
    caps: Arc<dyn ModelCapabilities>,
    regions: Arc<RegionRoutingConfig>,
    image_resolver: Arc<dyn ImageResolver>,
    settings: Arc<AppSettings>,
    /// Shared negative cache of foundation ids that reject prompt caching.
    /// Consulted by the read-gate in [`Self::assemble`] and updated by the
    /// strip-and-retry safety net at the send points; the same `Arc` is shared
    /// with the Responses provider via `build_app_state`.
    cache_support: Arc<CacheSupportRegistry>,
}

impl BedrockChatProvider {
    /// Construct a provider from its collaborators.
    pub fn new(
        clients: BedrockClients,
        caps: Arc<dyn ModelCapabilities>,
        regions: Arc<RegionRoutingConfig>,
        image_resolver: Arc<dyn ImageResolver>,
        settings: Arc<AppSettings>,
        cache_support: Arc<CacheSupportRegistry>,
    ) -> Self {
        Self {
            clients,
            caps,
            regions,
            image_resolver,
            settings,
            cache_support,
        }
    }

    /// Compose all Wave-3 pieces into a fully-assembled [`ConverseArgs`].
    ///
    /// This mirrors the legacy Python `_parse_request` ordering exactly:
    /// 1. reasoning ([`reasoning::build_reasoning_config`]),
    /// 2. tool config ([`tools`]) — convert + tool_choice (with the
    ///    de-hardcoded `skip_tool_choice`), normalize tool-result turns,
    /// 3. prompt-caching control extraction + strip from `extra_body`,
    /// 4. `translate::to_converse_args` (messages/system/inferenceConfig/
    ///    additionalModelRequestFields), passing reasoning+tools through
    ///    [`ConverseExtras`],
    /// 5. apply the reasoning `maxTokens`/`topP` side-signals,
    /// 6. decorate system + messages with `cachePoint`s ([`cache`]),
    /// 7. normalize tool-result turns + inject placeholder toolConfig when the
    ///    history carries tool content but no tools were defined.
    ///
    /// Returns the assembled [`ConverseArgs`] (still JSON-slot shaped) paired
    /// with a `cache_points_injected` flag — `true` when at least one
    /// `cachePoint` block was placed across the tools/system/messages zones. The
    /// per-request region override is resolved separately by
    /// [`Self::resolve_region`].
    ///
    /// `force_caching_off` is the cache safety net's strip path: when `true` the
    /// per-zone enablement is forced off so no `cachePoint` is injected and
    /// `cache_points_injected` is returned `false`. This is the SAME no-cachePoint
    /// assembly the master-switch-off path already produces (the strip is a
    /// re-assemble, never a surgical edit of a built SDK struct).
    async fn assemble(
        &self,
        req: &NormalizedChatRequest,
        force_caching_off: bool,
    ) -> Result<(ConverseArgs, bool), AppError> {
        let caps = self.caps.as_ref();
        let resolved = &req.resolved_model;
        // Clone the request so we can strip the prompt_caching control field
        // from extra_body before translation forwards extra_body to Bedrock.
        let mut chat: ChatRequest = req.request.clone();

        // --- 1. Reasoning ----------------------------------------------------
        let reasoning_outcome = match chat.reasoning_effort {
            Some(effort) => reasoning::build_reasoning_config(
                resolved,
                effort,
                chat.max_tokens,
                chat.max_completion_tokens,
                caps,
            ),
            None => reasoning::ReasoningOutcome::default(),
        };

        // --- 2. Tool config (pre-translate) ----------------------------------
        // Build a toolConfig from request tools + tool_choice. The llama skip is
        // computed HERE (composition layer) from the resolved model id.
        let skip_tool_choice = skip_tool_choice_for(resolved);
        let tool_config = match &chat.tools {
            Some(tools) if !tools.is_empty() => Some(tools::build_tool_config(
                tools,
                Some(&chat.tool_choice),
                skip_tool_choice,
            )?),
            _ => None,
        };

        // --- 3. Prompt-caching control: parse + strip from extra_body --------
        let caching = PromptCachingControl::extract_and_strip(&mut chat.extra_body);

        // --- 4. Translate (with reasoning + tool seam) -----------------------
        let extras = ConverseExtras {
            reasoning_fields: if reasoning_outcome.additional_model_request_fields.is_empty() {
                None
            } else {
                Some(Value::Object(
                    reasoning_outcome.additional_model_request_fields.clone(),
                ))
            },
            tool_config,
        };
        let mut args = to_converse_args(&chat, caps, self.image_resolver.as_ref(), &extras).await?;
        if args.tool_config.is_none() {
            args.tool_config = tools::synthesize_tool_config_from_messages(&args.messages);
        }

        // --- 5. Apply reasoning side-signals to inferenceConfig --------------
        if let Value::Object(cfg) = &mut args.inference_config {
            if let Some(max_tokens) = reasoning_outcome.max_tokens {
                cfg.insert("maxTokens".to_string(), Value::from(max_tokens));
            }
            if reasoning_outcome.drop_top_p {
                cfg.remove("topP");
            }
        }

        // --- 6. Prompt caching decoration ------------------------------------
        //
        // Cache zones are decorated in the Bedrock-canonical order
        // tools → system → messages, sharing ONE running checkpoint budget so
        // the GRAND total across all three zones never exceeds the configured
        // ceiling (`max_cache_checkpoints`, with the DEFAULT fallback below).
        // Tools consume first (most stable / highest hit-rate prefix), then
        // system, then messages with whatever budget remains.
        //
        // Byte-stability discipline: the cached prefix region (toolConfig.tools
        // → system → the last eligible user turn) must serialize deterministically
        // for cache hits to land — earlier zones are the cache prefix of later
        // zones, so any change to tools invalidates the system+messages cache and
        // any change to system invalidates the messages cache. The decorators
        // operate on ordered JSON arrays (no unordered HashMap in the prefix), so
        // the placement is fixed and reproducible.
        let global_default = self.settings.enable_prompt_caching;
        // Read-gate: a model already memoized as caching-unsupported (or an
        // explicit strip-retry) forces all zones off, so this re-assembly never
        // re-injects the cachePoints that were just rejected.
        let caching_off = force_caching_off
            || self
                .cache_support
                .is_unsupported(&normalize_for_match(resolved));
        // Tools reuse the system enablement decision: there is no separate
        // `prompt_caching.tools` request flag, so tools follow the same
        // global/system master switch.
        let tools_enabled = !caching_off && caching.system_enabled(global_default);
        let system_enabled = !caching_off && caching.system_enabled(global_default);
        let messages_enabled = !caching_off && caching.messages_enabled(global_default);
        // Bedrock per-request cachePoint ceiling used when a model declares no
        // explicit limit in config/models.toml.
        const DEFAULT_MAX_CACHE_CHECKPOINTS: u32 = 4;
        let max_checkpoints = Some(
            caps.max_cache_checkpoints(resolved)
                .unwrap_or(DEFAULT_MAX_CACHE_CHECKPOINTS),
        );

        // Resolve the UNIFORM per-request cache TTL: per-request
        // `extra_body.prompt_caching.ttl` wins over the settings default; a `1h`
        // request on a model lacking `Capability::CacheTtl1h` is silently
        // downgraded to `5m` with a metadata-only WARN (no content).
        let resolved_ttl = cache::resolve_cache_ttl(
            caching.ttl.as_deref(),
            &self.settings.prompt_cache_ttl,
            resolved,
            caps,
        );
        if resolved_ttl.downgraded {
            tracing::warn!(
                request_id = %req.request_id,
                model = %resolved,
                requested_ttl = %resolved_ttl.requested,
                effective_ttl = %resolved_ttl.effective,
                "prompt-cache 1h TTL not supported by model; downgraded to 5m"
            );
        }
        let ttl = Some(resolved_ttl.effective.as_str());

        // Running grand total of cachePoints placed across all zones. assemble()
        // owns this; each zone consumes from it and must not push it over
        // `max_checkpoints`.
        let mut used_checkpoints: u32 = 0;

        // Zone 1 — tools: inject a cachePoint into the toolConfig.tools array
        // tail, gated by the shared budget. The toolConfig was built pre-translate
        // (step 2) and round-trips through ConverseExtras into args.tool_config.
        if let Some(Value::Object(tc)) = args.tool_config.as_mut() {
            if let Some(tools_val) = tc.remove("tools") {
                let decorated_tools = cache::decorate_tools(
                    tools_val,
                    resolved,
                    caps,
                    tools_enabled,
                    used_checkpoints,
                    max_checkpoints,
                    ttl,
                );
                used_checkpoints += count_cache_points(&decorated_tools);
                tc.insert("tools".to_string(), decorated_tools);
            }
        }

        // Zone 2 — system.
        let decorated_system = cache::decorate_system_blocks(
            std::mem::replace(&mut args.system, Value::Null),
            resolved,
            caps,
            system_enabled,
            ttl,
        );
        used_checkpoints += count_cache_points(&decorated_system);
        args.system = decorated_system;

        // Zone 3 — messages, capped by whatever budget remains.
        let decorated_messages = cache::decorate_messages(
            std::mem::replace(&mut args.messages, Value::Null),
            resolved,
            caps,
            messages_enabled,
            used_checkpoints,
            max_checkpoints,
            ttl,
        );
        used_checkpoints += count_cache_points(&decorated_messages);
        args.messages = decorated_messages;

        // Whether ANY cachePoint landed across all three zones — threaded out
        // for the cache safety net (a later step strips + retries when an
        // unsupported model rejects these).
        let cache_points_injected = used_checkpoints > 0;

        // --- 7. Tool-result normalization + placeholder safety net -----------
        if let Value::Array(turns) = &args.messages {
            let normalized = tools::normalize_tool_result_turns(turns);
            args.messages = Value::Array(normalized);
        }
        if let Value::Array(turns) = &args.messages {
            args.tool_config =
                tools::inject_placeholder_tool_config(turns, args.tool_config.take());
        }

        Ok((args, cache_points_injected))
    }

    /// Resolve a per-request region override from the region-routing table.
    ///
    /// `None` means "use the home-region client and the original model id". When
    /// `Some`, the returned [`RouteOverride`] carries both the target region and
    /// the rewritten model id to send (the SDK call applies both).
    fn resolve_region(&self, model: &str) -> Option<RouteOverride> {
        self.regions.route_for(model)
    }

    /// Build the typed SDK `converse` call from assembled [`ConverseArgs`] and
    /// send it (applying the per-request region override at the call site).
    ///
    /// Returns the raw service error in a [`SendError`] so the shared cache
    /// safety net can inspect `.code()`/`.message()` before mapping. JSON→SDK
    /// build failures surface as [`SendError::App`] (never a cache rejection).
    async fn send_converse(
        &self,
        args: &ConverseArgs,
    ) -> Result<ConverseOutput, SendError<ConverseError>> {
        let route = self.resolve_region(&args.model_id);
        let model_id = route
            .as_ref()
            .map(|r| r.rewritten_model_id.clone())
            .unwrap_or_else(|| args.model_id.clone());

        let messages = build_sdk_messages(&args.messages).map_err(SendError::App)?;
        let system = build_sdk_system(&args.system).map_err(SendError::App)?;
        let inference_config = build_sdk_inference_config(&args.inference_config);

        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock converse"
        );

        let mut call = self
            .clients
            .runtime
            .converse()
            .model_id(&model_id)
            .set_messages(Some(messages))
            .set_system(Some(system))
            .inference_config(inference_config);

        if let Some(fields) = &args.additional_model_request_fields {
            call = call.additional_model_request_fields(json_to_document(fields));
        }
        if let Some(tc) = &args.tool_config {
            call = call.tool_config(build_sdk_tool_config(tc).map_err(SendError::App)?);
        }
        if let Some(oc) = &args.output_config {
            call = call.output_config(build_sdk_output_config(oc).map_err(SendError::App)?);
        }

        if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        } else {
            call.send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        }
    }

    /// Build the typed SDK `converse_stream` call from assembled
    /// [`ConverseArgs`] and send it. Mirrors [`Self::send_converse`]; the
    /// rejection surfaces at `.send()` BEFORE any stream event (confirmed live),
    /// so the strip-and-retry safety net is identical to the non-stream path.
    async fn send_converse_stream(
        &self,
        args: &ConverseArgs,
    ) -> Result<ConverseStreamOutput, SendError<ConverseStreamError>> {
        let route = self.resolve_region(&args.model_id);
        let model_id = route
            .as_ref()
            .map(|r| r.rewritten_model_id.clone())
            .unwrap_or_else(|| args.model_id.clone());

        let messages = build_sdk_messages(&args.messages).map_err(SendError::App)?;
        let system = build_sdk_system(&args.system).map_err(SendError::App)?;
        let inference_config = build_sdk_inference_config(&args.inference_config);

        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock converse_stream"
        );

        let mut call = self
            .clients
            .runtime
            .converse_stream()
            .model_id(&model_id)
            .set_messages(Some(messages))
            .set_system(Some(system))
            .inference_config(inference_config);

        if let Some(fields) = &args.additional_model_request_fields {
            call = call.additional_model_request_fields(json_to_document(fields));
        }
        if let Some(tc) = &args.tool_config {
            call = call.tool_config(build_sdk_tool_config(tc).map_err(SendError::App)?);
        }
        if let Some(oc) = &args.output_config {
            call = call.output_config(build_sdk_output_config(oc).map_err(SendError::App)?);
        }

        if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        } else {
            call.send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        }
    }
}

/// Count the top-level `cachePoint` blocks in a decorated array (the
/// `toolConfig.tools` array or the `system` array). Used to advance the shared
/// running checkpoint budget so tools + system + messages together never exceed
/// the configured ceiling. `pub(crate)` for reuse by the Responses provider's
/// own cachePoint budget assembly.
pub(crate) fn count_cache_points(blocks: &Value) -> u32 {
    let Value::Array(blocks) = blocks else {
        return 0;
    };
    blocks
        .iter()
        .map(|b| match b {
            Value::Object(o) if o.contains_key("cachePoint") => 1,
            // A message turn carries its cachePoint nested inside `content`
            // rather than as a top-level array entry; recurse so the running
            // budget and the injected-flag count messages zones correctly.
            _ => b.get("content").map(count_cache_points).unwrap_or(0),
        })
        .sum()
}

#[async_trait::async_trait]
impl ChatProvider for BedrockChatProvider {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        let (args, cache_points_injected) = self.assemble(req, false).await?;
        let normalized = normalize_for_match(&req.resolved_model);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            cache_points_injected,
            || self.send_converse(&args),
            || async {
                let (retry_args, _) = self.assemble(req, true).await.map_err(SendError::App)?;
                self.send_converse(&retry_args).await
            },
        )
        .await?;

        // Lower the typed output into the JSON shape response.rs maps, then run
        // the parity-tested pure mapper.
        let output_json = converse_output_to_json(&output);
        let message_id = format!("chatcmpl-{}", req_id());
        response::from_converse_output(&output_json, &req.request.model, &message_id)
    }

    async fn chat_stream(&self, req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        let (args, cache_points_injected) = self.assemble(req, false).await?;
        let normalized = normalize_for_match(&req.resolved_model);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            cache_points_injected,
            || self.send_converse_stream(&args),
            || async {
                let (retry_args, _) = self.assemble(req, true).await.map_err(SendError::App)?;
                self.send_converse_stream(&retry_args).await
            },
        )
        .await?;

        // include_usage gate from stream_options (defaults to true when the
        // caller asked to stream usage; absent stream_options ⇒ no usage chunk).
        let include_usage = req
            .request
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        let message_id = format!("chatcmpl-{}", req_id());

        Ok(stream::converse_stream_to_openai(
            output,
            req.request.model.clone(),
            message_id,
            include_usage,
            req.request_id.clone(),
            req.received_at,
        ))
    }
}

/// Generate a short, unique-ish request id suffix for the response/stream `id`.
///
/// Uses the current Unix time in nanoseconds; collision-free enough for an id
/// label (the legacy Python used a uuid4 hex — this is a dependency-free
/// equivalent that keeps the `chatcmpl-` prefix shape).
fn req_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod tests;
