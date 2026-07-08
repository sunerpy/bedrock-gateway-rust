//! OpenAI → Bedrock Converse request translation (pure functions).
//!
//! This module is the highest-risk PARITY surface: it ports the Python request
//! builder in `.legacy-python/src/api/models/bedrock.py` into a pure, testable
//! translation that produces Bedrock Converse `args` (modeled here as
//! [`ConverseArgs`], a typed holder of `serde_json::Value` slots).
//!
//! Ported functions (with provenance line ranges):
//! - `_parse_request`        (bedrock.py:1106-1296) → [`to_converse_args`]
//! - `_parse_messages`       (bedrock.py:774-1104)  → [`parse_messages`]
//! - `_reframe_multi_payloard` same-role merge/split (912-973, 1725-1756)
//! - `no_assistant_prefill`  continuation (bedrock.py:977-990)
//! - `_parse_system_prompts` (bedrock.py:709-744)   → [`parse_system_prompts`]
//! - `_parse_content_parts`  (bedrock.py:1621-1658) → [`parse_content_parts`]
//! - `_parse_image`          (bedrock.py:1594-1619) → [`parse_image_data_uri`]
//!   / [`ImageResolver`]
//! - `_collect_passthrough_fields` (bedrock.py:1691-1695)
//! - context_1m injection    (bedrock.py:1248-1259)
//! - topP conflict           (bedrock.py:1134-1138)
//!
//! ## Purity & the network seam
//!
//! Translation never calls Bedrock. The one place the legacy code reaches the
//! network is image fetching for non-data-URI image URLs (`requests.get`,
//! bedrock.py:1609). That is isolated behind the [`ImageResolver`] trait so the
//! core translation is synchronous and fully testable offline: `data:` URIs are
//! decoded inline (base64), and remote URLs are delegated to the injected
//! resolver. The async, network-backed resolver is [`ReqwestImageResolver`].
//!
//! ## De-hardcoding
//!
//! No model-id literals appear here. All model knowledge flows through
//! [`ModelCapabilities`] (capability flags, beta headers). Image-modality
//! support is supplied by the caller via [`ImageResolver::supports_image`],
//! mirroring the Python `is_supported_modality` gate (bedrock.py:1641-1645).
//!
//! ## Integration seam for reasoning (task 16) and tools (task 17)
//!
//! [`to_converse_args`] owns messages, system prompts, inference config,
//! multimodal content, and the additional-fields passthrough/context-1m
//! assembly. It deliberately does **not** build the `reasoning_config` /
//! `thinking` additional fields (task 16) nor the `toolConfig` (task 17).
//! Instead it exposes them as already-built inputs via [`ConverseExtras`]:
//! a follow-up wires `crate::bedrock::reasoning` and `crate::bedrock::tools`
//! into those slots. Reasoning fields are merged into
//! `additional_model_request_fields` (so the context-1m / `thinking` topP-drop
//! rules still apply), and the tool config is placed verbatim into
//! [`ConverseArgs::tool_config`]. This keeps the parity-critical message/system
//! logic in one place without duplicating reasoning/tool logic here.

use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::domain::{Capability, ModelCapabilities};
use crate::error::AppError;
use crate::openai::schema::{
    ChatRequest, ContentInput, ContentPart, Message, StringOrVec, ToolContentInput,
};

/// The fully-built Converse request arguments (pure translation output).
///
/// Mirrors the Python `args` dict assembled by `_parse_request`
/// (bedrock.py:1146-1296). Each field is a JSON slot so this stays decoupled
/// from the AWS SDK builder types; the provider layer maps these into the SDK
/// request.
#[derive(Debug, Clone, PartialEq)]
pub struct ConverseArgs {
    /// `modelId` — the original request model id (Converse resolves profiles).
    pub model_id: String,
    /// `messages` — Bedrock user/assistant turns (a JSON array).
    pub messages: Value,
    /// `system` — Bedrock system blocks (a JSON array; may be empty).
    pub system: Value,
    /// `inferenceConfig` — maxTokens/temperature/topP/stopSequences.
    pub inference_config: Value,
    /// `additionalModelRequestFields` — assembled passthrough + context-1m +
    /// (later) reasoning fields. `None` when there is nothing to send.
    pub additional_model_request_fields: Option<Value>,
    /// `toolConfig` slot — populated by the tool task (17). Translation leaves
    /// this `None`; the seam owner fills it.
    pub tool_config: Option<Value>,
}

impl ConverseArgs {
    /// Render the args as a single `serde_json::Value` object using Bedrock
    /// Converse key names. Convenient for golden tests and the provider layer.
    pub fn to_value(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("modelId".to_string(), Value::String(self.model_id.clone()));
        obj.insert("messages".to_string(), self.messages.clone());
        obj.insert("system".to_string(), self.system.clone());
        obj.insert("inferenceConfig".to_string(), self.inference_config.clone());
        if let Some(fields) = &self.additional_model_request_fields {
            obj.insert("additionalModelRequestFields".to_string(), fields.clone());
        }
        if let Some(tc) = &self.tool_config {
            obj.insert("toolConfig".to_string(), tc.clone());
        }
        Value::Object(obj)
    }
}

/// Pre-built pieces contributed by the reasoning (task 16) and tool (task 17)
/// tasks. Translation composes these without re-deriving them.
///
/// This is the explicit integration seam: callers that have run reasoning/tool
/// normalization pass the results in here. The default ([`ConverseExtras::default`])
/// contributes nothing, which is exactly the behavior for a plain chat request.
#[derive(Debug, Clone, Default)]
pub struct ConverseExtras {
    /// Additional model request fields produced by reasoning normalization
    /// (e.g. `{"thinking": {...}}` or `{"reasoning_config": {...}}`). Merged
    /// into `additionalModelRequestFields`. If this contains a `thinking`
    /// key, `topP` is dropped from inference config (bedrock.py:1267-1268).
    pub reasoning_fields: Option<Value>,
    /// The fully-built `toolConfig` object (task 17). Placed verbatim into
    /// [`ConverseArgs::tool_config`].
    pub tool_config: Option<Value>,
}

/// Resolves an image URL into raw bytes + a Bedrock image `format`.
///
/// This is the network seam (`_parse_image`, bedrock.py:1594-1619) and the
/// modality gate (`is_supported_modality`, bedrock.py:1641-1645). `data:` URIs
/// are decoded by the translation itself; only remote URLs reach
/// [`ImageResolver::fetch`]. Implementors also declare whether a model accepts
/// image content so tests can exercise the 400 path offline.
#[async_trait::async_trait]
pub trait ImageResolver: Send + Sync {
    /// Does `model_id` accept image content parts? Mirrors the IMAGE-modality
    /// check the Python code performs before parsing any image
    /// (bedrock.py:1641-1645).
    fn supports_image(&self, model_id: &str) -> bool;

    /// Fetch a remote (non-`data:`) image URL, returning `(bytes, format)`
    /// where `format` is the Bedrock image format (e.g. `"jpeg"`).
    async fn fetch(&self, url: &str) -> Result<(Vec<u8>, String), AppError>;
}

/// Decoded inline image: raw bytes plus the Bedrock `format` (the part after
/// `image/`, e.g. `"png"`).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    /// Raw image bytes.
    pub bytes: Vec<u8>,
    /// Bedrock image format (`content_type[6:]`, bedrock.py:1650).
    pub format: String,
}

/// Decode a `data:image/<fmt>;base64,<payload>` URI into bytes + format.
///
/// Returns `Ok(None)` when `url` is not a `data:` image URI (the caller then
/// delegates to an [`ImageResolver`]). Mirrors bedrock.py:1600-1606.
///
/// # Errors
/// Returns [`AppError::BadRequest`] if the payload is not valid base64.
pub fn parse_image_data_uri(url: &str) -> Result<Option<DecodedImage>, AppError> {
    // Pattern: ^data:(image/[a-z]*);base64,\s*   (bedrock.py:1600)
    let rest = match url.strip_prefix("data:") {
        Some(r) => r,
        None => return Ok(None),
    };
    let (mime, payload) = match rest.split_once(";base64,") {
        Some(parts) => parts,
        None => return Ok(None),
    };
    if !mime.starts_with("image/") {
        return Ok(None);
    }
    // `\s*` after the comma: strip leading ASCII whitespace from the payload.
    let payload = payload.trim_start();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| AppError::BadRequest(format!("invalid base64 image data: {e}")))?;
    // format = content_type[6:] — the substring after "image/" (bedrock.py:1650).
    let format = mime["image/".len()..].to_string();
    Ok(Some(DecodedImage { bytes, format }))
}

fn push_non_empty_text_block(blocks: &mut Vec<Value>, text: &str) {
    if !text.is_empty() {
        blocks.push(json!({ "text": text }));
    }
}

/// Build the Bedrock `system` blocks from system/developer messages.
///
/// Ports `_parse_system_prompts` (bedrock.py:709-744) for the translation
/// scope: every `system`/`developer` message contributes a `{"text": ...}`
/// block. Prompt-cache `cachePoint` insertion (bedrock.py:736-772) is a
/// separate task (20) and is intentionally omitted here.
///
/// # Errors
/// Returns [`AppError::BadRequest`] if a system/developer message content is
/// not a string (Python raises `TypeError`, bedrock.py:729-730, surfaced as a
/// 400 to the client).
pub fn parse_system_prompts(req: &ChatRequest) -> Result<Value, AppError> {
    let mut blocks: Vec<Value> = Vec::new();
    for message in &req.messages {
        match message {
            Message::System { content, .. } | Message::Developer { content, .. } => {
                push_non_empty_text_block(&mut blocks, content);
            }
            _ => continue,
        }
    }
    Ok(Value::Array(blocks))
}

/// Parse a user/assistant message's content into Bedrock content blocks.
///
/// Ports `_parse_content_parts` (bedrock.py:1621-1658). A string body becomes a
/// single `{"text": ...}` block; a parts list maps text → `{"text"}` and image
/// → `{"image": {...}}`. Image handling enforces the IMAGE-modality gate and
/// decodes `data:` URIs inline; remote URLs are fetched via `resolver`.
///
/// # Errors
/// - [`AppError::BadRequest`] if an image appears for a model without IMAGE
///   modality (bedrock.py:1641-1645).
/// - Propagates resolver/base64 errors.
async fn parse_content_parts(
    content: &ContentInput,
    model_id: &str,
    resolver: &dyn ImageResolver,
) -> Result<Vec<Value>, AppError> {
    match content {
        ContentInput::Text(text) => {
            let mut blocks = Vec::new();
            push_non_empty_text_block(&mut blocks, text);
            Ok(blocks)
        }
        ContentInput::Parts(parts) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ContentPart::Text(t) => push_non_empty_text_block(&mut blocks, &t.text),
                    ContentPart::Image(img) => {
                        if !resolver.supports_image(model_id) {
                            return Err(AppError::BadRequest(format!(
                                "Multimodal message is currently not supported by {model_id}"
                            )));
                        }
                        let url = &img.image_url.url;
                        let (bytes, format) = match parse_image_data_uri(url)? {
                            Some(d) => (d.bytes, d.format),
                            None => resolver.fetch(url).await?,
                        };
                        blocks.push(json!({
                            "image": {
                                "format": format,
                                "source": { "bytes": Value::Array(
                                    bytes.into_iter().map(|b| json!(b)).collect()
                                ) },
                            }
                        }));
                    }
                }
            }
            Ok(blocks)
        }
    }
}

/// Does an assistant message have non-empty text content? (bedrock.py:800-806)
fn assistant_has_text(content: &Option<ContentInput>) -> bool {
    match content {
        Some(ContentInput::Text(s)) => !s.trim().is_empty(),
        Some(ContentInput::Parts(p)) => !p.is_empty(),
        None => false,
    }
}

/// Extract text from a tool message's content (`_extract_tool_content`,
/// bedrock.py:861-910). String content is returned as-is; a parts list joins
/// each item's `text` (or the JSON of the item) with newlines.
fn extract_tool_content(content: &ToolContentInput) -> String {
    match content {
        ToolContentInput::Text(s) => s.clone(),
        ToolContentInput::Parts(items) => {
            let mut text_parts: Vec<String> = Vec::with_capacity(items.len());
            for item in items {
                if let Some(obj) = item.as_object() {
                    if let Some(t) = obj.get("text") {
                        if let Some(s) = t.as_str() {
                            // Pretty-print embedded JSON objects (bedrock.py:882-891).
                            let trimmed = s.trim();
                            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                                match serde_json::from_str::<Value>(s) {
                                    Ok(parsed) => text_parts.push(
                                        serde_json::to_string_pretty(&parsed)
                                            .unwrap_or_else(|_| s.to_string()),
                                    ),
                                    Err(_) => text_parts.push(s.to_string()),
                                }
                            } else {
                                text_parts.push(s.to_string());
                            }
                        } else {
                            text_parts.push(t.to_string());
                        }
                    } else {
                        text_parts.push(
                            serde_json::to_string_pretty(item).unwrap_or_else(|_| item.to_string()),
                        );
                    }
                } else if let Some(s) = item.as_str() {
                    text_parts.push(s.to_string());
                } else {
                    text_parts.push(item.to_string());
                }
            }
            text_parts.join("\n")
        }
    }
}

/// An intermediate Bedrock message before same-role reframing: a role and its
/// content blocks. Mirrors the dicts produced by `_parse_messages`
/// (bedrock.py:786-858).
struct IntermediateMessage {
    role: String,
    content: Vec<Value>,
}

/// Build the per-message intermediate list (pre-reframe).
///
/// Ports the body of `_parse_messages` (bedrock.py:786-858): user → content
/// parts; assistant → text parts + `toolUse` blocks; tool → a user turn with a
/// single `toolResult` block. System/developer messages are skipped (handled by
/// [`parse_system_prompts`]).
async fn build_intermediate_messages(
    req: &ChatRequest,
    resolver: &dyn ImageResolver,
) -> Result<Vec<IntermediateMessage>, AppError> {
    let mut out: Vec<IntermediateMessage> = Vec::new();
    for message in &req.messages {
        match message {
            Message::User { content, .. } => {
                let blocks = parse_content_parts(content, &req.model, resolver).await?;
                if blocks.is_empty() {
                    return Err(AppError::BadRequest(
                        "message content must contain at least one non-empty content block"
                            .to_string(),
                    ));
                }
                out.push(IntermediateMessage {
                    role: "user".to_string(),
                    content: blocks,
                });
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_content: Vec<Value> = Vec::new();
                if assistant_has_text(content) {
                    if let Some(c) = content {
                        let blocks = parse_content_parts(c, &req.model, resolver).await?;
                        assistant_content.extend(blocks);
                    }
                }
                if let Some(calls) = tool_calls {
                    for call in calls {
                        // Tool-call arguments are a JSON string (bedrock.py:815).
                        let input: Value =
                            serde_json::from_str(&call.function.arguments).map_err(|e| {
                                AppError::BadRequest(format!(
                                    "invalid tool_call arguments JSON: {e}"
                                ))
                            })?;
                        assistant_content.push(json!({
                            "toolUse": {
                                "toolUseId": call.id,
                                "name": call.function.name,
                                "input": input,
                            }
                        }));
                    }
                }
                // Only add the message if it has content (bedrock.py:827).
                if !assistant_content.is_empty() {
                    out.push(IntermediateMessage {
                        role: "assistant".to_string(),
                        content: assistant_content,
                    });
                }
            }
            Message::Tool {
                content,
                tool_call_id,
            } => {
                let tool_content = extract_tool_content(content);
                out.push(IntermediateMessage {
                    role: "user".to_string(),
                    content: vec![json!({
                        "toolResult": {
                            "toolUseId": tool_call_id,
                            "content": [{ "text": tool_content }],
                        }
                    })],
                });
            }
            // System / developer handled by parse_system_prompts.
            Message::System { .. } | Message::Developer { .. } => continue,
        }
    }
    Ok(out)
}

/// Does a content block list contain a block with the given key
/// (`_content_contains_block`, bedrock.py:1718-1723)?
fn content_contains_block(content: &[Value], key: &str) -> bool {
    content
        .iter()
        .any(|b| b.as_object().is_some_and(|o| o.contains_key(key)))
}

/// Decide whether to split rather than merge two same-role content groups.
///
/// Ports `_should_split_same_role_merge` (bedrock.py:1725-1756): merge
/// contiguous tool-only turns; split tool-only from normal content.
fn should_split_same_role_merge(role: &str, current: &[Value], next: &[Value]) -> bool {
    if role == "user" {
        let cur_tr = content_contains_block(current, "toolResult");
        let next_tr = content_contains_block(next, "toolResult");
        if cur_tr && next_tr {
            return false;
        }
        if cur_tr != next_tr {
            return true;
        }
    }
    if role == "assistant" {
        let cur_tu = content_contains_block(current, "toolUse");
        let next_tu = content_contains_block(next, "toolUse");
        if cur_tu && next_tu {
            return false;
        }
        if cur_tu != next_tu {
            return true;
        }
    }
    false
}

/// Reframe per-message intermediates into Converse turns, merging contiguous
/// same-role messages and splitting tool-only from normal content.
///
/// Ports `_reframe_multi_payloard` (bedrock.py:912-973) plus the
/// `no_assistant_prefill` continuation (bedrock.py:977-990). Prompt-cache
/// checkpoint insertion (bedrock.py:996-1035) and `toolResult` normalization
/// (bedrock.py:1038-1104) are separate tasks (20/17) and omitted here.
fn reframe_messages(
    intermediates: Vec<IntermediateMessage>,
    req: &ChatRequest,
    caps: &dyn ModelCapabilities,
) -> Value {
    let mut reformatted: Vec<(String, Vec<Value>)> = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_content: Vec<Value> = Vec::new();

    for msg in intermediates {
        let next_role = msg.role;
        let next_content = msg.content;

        if Some(&next_role) != current_role.as_ref() {
            if !current_content.is_empty() {
                reformatted.push((
                    current_role.clone().unwrap_or_default(),
                    std::mem::take(&mut current_content),
                ));
            }
            current_role = Some(next_role.clone());
            current_content = Vec::new();
        }

        let should_split = !current_content.is_empty()
            && current_role.as_deref() == Some(next_role.as_str())
            && should_split_same_role_merge(&next_role, &current_content, &next_content);

        if should_split {
            reformatted.push((
                current_role.clone().unwrap_or_default(),
                std::mem::replace(&mut current_content, next_content),
            ));
        } else {
            current_content.extend(next_content);
        }
    }

    if !current_content.is_empty() {
        reformatted.push((current_role.unwrap_or_default(), current_content));
    }

    // no_assistant_prefill: if the conversation ends on an assistant turn and
    // the model can't prefill, append a user continuation (bedrock.py:977-990).
    if let Some((last_role, _)) = reformatted.last() {
        if last_role == "assistant" && caps.has(&req.model, Capability::NoAssistantPrefill) {
            reformatted.push((
                "user".to_string(),
                vec![json!({
                    "text": "Please continue your response from where you left off."
                })],
            ));
        }
    }

    Value::Array(
        reformatted
            .into_iter()
            .map(|(role, content)| json!({ "role": role, "content": content }))
            .collect(),
    )
}

/// Parse the OpenAI messages into Bedrock Converse `messages`.
///
/// Ports `_parse_messages` (bedrock.py:774-1104) for the translation scope:
/// per-message conversion + same-role reframing + `no_assistant_prefill`. See
/// [`build_intermediate_messages`] and [`reframe_messages`].
///
/// # Errors
/// Propagates image-modality / decode / tool-argument errors.
pub async fn parse_messages(
    req: &ChatRequest,
    caps: &dyn ModelCapabilities,
    resolver: &dyn ImageResolver,
) -> Result<Value, AppError> {
    let intermediates = build_intermediate_messages(req, resolver).await?;
    Ok(reframe_messages(intermediates, req, caps))
}

/// Top-level fields forwarded to Bedrock `additionalModelRequestFields`.
///
/// Allowlist (not blocklist) mirroring `_BEDROCK_PASSTHROUGH_FIELDS`
/// (bedrock.py:165-167) so unknown OpenAI-standard fields never leak into
/// Bedrock and trigger a `ValidationException`.
const BEDROCK_PASSTHROUGH_FIELDS: [&str; 3] = ["thinking", "output_config", "anthropic_beta"];

/// Collect the controlled passthrough fields from the request's captured
/// `extra` map (`_collect_passthrough_fields`, bedrock.py:1691-1695): only the
/// allowlisted keys, and only when their value is non-null.
fn collect_passthrough_fields(req: &ChatRequest) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, v) in &req.extra {
        if BEDROCK_PASSTHROUGH_FIELDS.contains(&k.as_str()) && !v.is_null() {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// Build the `inferenceConfig` object (bedrock.py:1116-1144).
///
/// `maxTokens` always; `temperature`/`topP` only when set; drop `topP` when
/// both are present and the model has the temperature/topP conflict capability
/// (bedrock.py:1134-1138); drop BOTH `temperature` and `topP` when the model
/// has [`Capability::DropSamplingParams`] (Opus 4.7+, Sonnet 5, Fable/Mythos 5
/// reject non-default sampling params with HTTP 400); `stop` (string or list)
/// → `stopSequences`.
fn build_inference_config(req: &ChatRequest, caps: &dyn ModelCapabilities) -> Value {
    let mut cfg = Map::new();

    // maxTokens always present (bedrock.py:1117-1119).
    let max_tokens = req.max_tokens.unwrap_or(2048);
    cfg.insert("maxTokens".to_string(), json!(max_tokens));

    if let Some(t) = req.temperature {
        cfg.insert("temperature".to_string(), json!(t));
    }
    if let Some(p) = req.top_p {
        cfg.insert("topP".to_string(), json!(p));
    }

    // Drop topP on conflict when both present (bedrock.py:1134-1138).
    if cfg.contains_key("temperature")
        && cfg.contains_key("topP")
        && caps.has(&req.model, Capability::TemperatureToppConflict)
    {
        cfg.remove("topP");
    }

    // Models that deprecate all sampling parameters (Claude Opus 4.7+, Sonnet 5,
    // Fable/Mythos 5) return HTTP 400 for any non-default temperature/topP, so
    // strip both unconditionally.
    if caps.has(&req.model, Capability::DropSamplingParams) {
        cfg.remove("temperature");
        cfg.remove("topP");
    }

    if let Some(stop) = &req.stop {
        let seqs = match stop {
            StringOrVec::String(s) if s.trim().is_empty() => Vec::new(),
            StringOrVec::String(s) => vec![Value::String(s.clone())],
            StringOrVec::Vec(v) => v
                .iter()
                .filter(|s| !s.trim().is_empty())
                .cloned()
                .map(Value::String)
                .collect(),
        };
        if !seqs.is_empty() {
            cfg.insert("stopSequences".to_string(), Value::Array(seqs));
        }
    }

    Value::Object(cfg)
}

/// Translate an OpenAI [`ChatRequest`] into Bedrock Converse [`ConverseArgs`].
///
/// This is the pure entrypoint (no Bedrock client calls). It ports
/// `_parse_request` (bedrock.py:1106-1296) for messages, system prompts,
/// inference config, multimodal content, the controlled passthrough allowlist,
/// and context-1m beta-header auto-injection. Reasoning (task 16) and tool
/// (task 17) pieces are supplied pre-built via `extras` — see [`ConverseExtras`].
///
/// Image fetching for remote (`http(s)://`) URLs goes through `resolver`; this
/// function is otherwise synchronous in spirit (only awaits the resolver).
///
/// # Errors
/// Returns [`AppError::BadRequest`] for non-string system content, images on a
/// non-IMAGE model, invalid base64 image data, or invalid tool-call arguments;
/// propagates resolver errors.
pub async fn to_converse_args(
    req: &ChatRequest,
    caps: &dyn ModelCapabilities,
    resolver: &dyn ImageResolver,
    extras: &ConverseExtras,
) -> Result<ConverseArgs, AppError> {
    let messages = parse_messages(req, caps, resolver).await?;
    let system = parse_system_prompts(req)?;
    let mut inference_config = build_inference_config(req, caps);

    // Assemble additionalModelRequestFields (bedrock.py:1233-1263).
    let mut additional = Map::new();

    // Reasoning-task contribution (seam): merge its fields first so the
    // context-1m / thinking rules below see them (bedrock.py:1234-1236).
    if let Some(Value::Object(rf)) = &extras.reasoning_fields {
        for (k, v) in rf {
            additional.insert(k.clone(), v.clone());
        }
    }

    // extra_body, minus our control field `prompt_caching` (bedrock.py:1238-1241).
    if let Some(Value::Object(extra_body)) = &req.extra_body {
        for (k, v) in extra_body {
            if k != "prompt_caching" {
                additional.insert(k.clone(), v.clone());
            }
        }
    }

    // Controlled top-level passthrough allowlist (bedrock.py:1243-1245).
    for (k, v) in collect_passthrough_fields(req) {
        additional.insert(k, v);
    }

    // Auto-enable the 1M-context beta header for capable models if the caller
    // didn't already set it (bedrock.py:1247-1259). The header value is the
    // model's configured beta header(s).
    if caps.has(&req.model, Capability::Context1mBeta) {
        for header in caps.beta_headers(&req.model) {
            merge_anthropic_beta(&mut additional, &header);
        }
    }

    // Extended thinking doesn't support both temperature and topP — drop topP
    // when `thinking` is present (bedrock.py:1265-1268).
    if additional.contains_key("thinking") {
        if let Value::Object(cfg) = &mut inference_config {
            cfg.remove("topP");
        }
    }

    let additional_model_request_fields = if additional.is_empty() {
        None
    } else {
        Some(Value::Object(additional))
    };

    Ok(ConverseArgs {
        model_id: req.model.clone(),
        messages,
        system,
        inference_config,
        additional_model_request_fields,
        // Tool config is the task-17 seam; translation never builds it.
        tool_config: extras.tool_config.clone(),
    })
}

/// Merge a single beta header into `additional["anthropic_beta"]`, matching the
/// Python list/str/absent handling (bedrock.py:1249-1259). The result is always
/// a list and never duplicates `header`.
fn merge_anthropic_beta(additional: &mut Map<String, Value>, header: &str) {
    match additional.get("anthropic_beta") {
        None => {
            additional.insert(
                "anthropic_beta".to_string(),
                Value::Array(vec![Value::String(header.to_string())]),
            );
        }
        Some(Value::String(existing)) => {
            let existing = existing.clone();
            let list = if existing == header {
                vec![Value::String(existing)]
            } else {
                vec![Value::String(existing), Value::String(header.to_string())]
            };
            additional.insert("anthropic_beta".to_string(), Value::Array(list));
        }
        Some(Value::Array(existing)) => {
            let mut list = existing.clone();
            let already = list.iter().any(|v| v.as_str() == Some(header));
            if !already {
                list.push(Value::String(header.to_string()));
            }
            additional.insert("anthropic_beta".to_string(), Value::Array(list));
        }
        Some(_) => {
            // Non-list/str value — replace with the header list (defensive).
            additional.insert(
                "anthropic_beta".to_string(),
                Value::Array(vec![Value::String(header.to_string())]),
            );
        }
    }
}

/// Network-backed [`ImageResolver`] using `reqwest` with a 30s timeout
/// (`_parse_image` remote branch, bedrock.py:1608-1619).
///
/// `supports_image` is delegated to a caller-provided predicate so this type
/// stays free of model knowledge. Construct via [`ReqwestImageResolver::new`].
pub struct ReqwestImageResolver {
    client: reqwest::Client,
    supports_image: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl std::fmt::Debug for ReqwestImageResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReqwestImageResolver")
            .finish_non_exhaustive()
    }
}

impl ReqwestImageResolver {
    /// Construct a resolver with a 30s-timeout client and an image-modality
    /// predicate (typically `|m| caps.is_supported_modality(m, "IMAGE")` once a
    /// modality accessor exists; until then the wiring layer supplies it).
    pub fn new(supports_image: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            supports_image: Box::new(supports_image),
        }
    }
}

#[async_trait::async_trait]
impl ImageResolver for ReqwestImageResolver {
    fn supports_image(&self, model_id: &str) -> bool {
        (self.supports_image)(model_id)
    }

    async fn fetch(&self, url: &str) -> Result<(Vec<u8>, String), AppError> {
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| AppError::Internal(format!("Unable to access the image url: {e}")))?;
        if !resp.status().is_success() {
            return Err(AppError::Internal(
                "Unable to access the image url".to_string(),
            ));
        }
        // Content-Type → format; default to image/jpeg when absent/non-image
        // (bedrock.py:1612-1614).
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .unwrap_or_default();
        let content_type = if content_type.starts_with("image") {
            content_type
        } else {
            "image/jpeg".to_string()
        };
        let format = content_type
            .strip_prefix("image/")
            .unwrap_or("jpeg")
            .to_string();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| AppError::Internal(format!("Unable to read image bytes: {e}")))?
            .to_vec();
        Ok((bytes, format))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bedrock::capabilities::ConfigModelCapabilities;
    use crate::config::ModelCapabilityConfig;
    use crate::openai::schema::{
        ContentInput, ContentPart, ImageContent, ImageUrl, Message, TextContent,
    };
    use std::collections::HashMap;

    const MODELS_TOML: &str = "config/models.toml";

    fn caps() -> ConfigModelCapabilities {
        let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load models.toml");
        ConfigModelCapabilities::new(config)
    }

    /// A test resolver that never hits the network. `supports_image` is a flag;
    /// `fetch` returns canned bytes so the remote path is exercised offline.
    struct TestResolver {
        image_ok: bool,
        canned: Option<(Vec<u8>, String)>,
    }

    #[async_trait::async_trait]
    impl ImageResolver for TestResolver {
        fn supports_image(&self, _model_id: &str) -> bool {
            self.image_ok
        }
        async fn fetch(&self, _url: &str) -> Result<(Vec<u8>, String), AppError> {
            self.canned
                .clone()
                .ok_or_else(|| AppError::Internal("no canned image".to_string()))
        }
    }

    fn resolver(image_ok: bool) -> TestResolver {
        TestResolver {
            image_ok,
            canned: None,
        }
    }

    fn base_request(model: &str, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            messages,
            model: model.to_string(),
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            user: None,
            max_tokens: Some(2048),
            max_completion_tokens: None,
            reasoning_effort: None,
            n: None,
            tools: None,
            tool_choice: Default::default(),
            stop: None,
            extra_body: None,
            extra: HashMap::new(),
        }
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            name: None,
            content: ContentInput::Text(text.to_string()),
        }
    }

    #[tokio::test]
    async fn text_message_translation() {
        let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("Hello!")]);
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");

        assert_eq!(args.model_id, "anthropic.claude-3-sonnet-v1:0");
        let msgs = args.messages.as_array().expect("messages array");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello!");
        assert_eq!(args.inference_config["maxTokens"], 2048);
        // No optional params set.
        assert!(args.inference_config.get("temperature").is_none());
        assert!(args.inference_config.get("topP").is_none());
        assert!(args.additional_model_request_fields.is_none());
        assert!(args.tool_config.is_none());
    }

    #[tokio::test]
    async fn system_and_developer_become_system_blocks() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![
                Message::System {
                    name: None,
                    content: "You are helpful.".to_string(),
                },
                Message::Developer {
                    name: None,
                    content: "Be terse.".to_string(),
                },
                user_text("Hi"),
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");

        let sys = args.system.as_array().expect("system array");
        assert_eq!(sys.len(), 2);
        assert_eq!(sys[0]["text"], "You are helpful.");
        assert_eq!(sys[1]["text"], "Be terse.");
        // System/developer messages do NOT appear in messages.
        let msgs = args.messages.as_array().expect("messages array");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[tokio::test]
    async fn empty_system_text_is_skipped() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![
                Message::System {
                    name: None,
                    content: "".to_string(),
                },
                Message::Developer {
                    name: None,
                    content: "Be terse.".to_string(),
                },
                user_text("Hi"),
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");

        let sys = args.system.as_array().expect("system array");
        assert_eq!(sys.len(), 1);
        assert_eq!(sys[0]["text"], "Be terse.");
    }

    #[tokio::test]
    async fn user_empty_text_is_bad_request() {
        let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("")]);
        let c = caps();
        let r = resolver(false);
        let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect_err("empty user text must reject");

        match err {
            AppError::BadRequest(message) => {
                assert!(message.contains("message content must contain at least"))
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mixed_empty_text_parts_keep_only_non_empty_text() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::User {
                name: None,
                content: ContentInput::Parts(vec![
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "".to_string(),
                    }),
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "keep me".to_string(),
                    }),
                ]),
            }],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");

        let content = args.messages[0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "keep me");
    }

    #[tokio::test]
    async fn non_string_system_content_is_rejected() {
        // The schema types System.content as String, so a JSON array fails to
        // deserialize into a SystemMessage at the wire boundary — proving the
        // parity guarantee (non-string system content never reaches Bedrock).
        let raw = r#"{
            "model": "anthropic.claude-3-sonnet-v1:0",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "x"}]},
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let parsed: Result<ChatRequest, _> = serde_json::from_str(raw);
        assert!(
            parsed.is_err(),
            "non-string system content must not deserialize"
        );
    }

    #[tokio::test]
    async fn stop_string_becomes_singleton_sequence() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::String("STOP".to_string()));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let seqs = args.inference_config["stopSequences"]
            .as_array()
            .expect("stopSequences array");
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0], "STOP");
    }

    #[tokio::test]
    async fn blank_stop_list_is_omitted() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::Vec(vec!["\n".to_string()]));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        assert!(args.inference_config.get("stopSequences").is_none());
    }

    #[tokio::test]
    async fn blank_stop_string_is_omitted() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::String("\n".to_string()));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        assert!(args.inference_config.get("stopSequences").is_none());
    }

    #[tokio::test]
    async fn blank_stop_list_entries_are_filtered() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::Vec(vec![
            "".to_string(),
            "\n".to_string(),
            "END".to_string(),
        ]));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let seqs = args.inference_config["stopSequences"]
            .as_array()
            .expect("array");
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0], "END");
    }

    #[tokio::test]
    async fn non_blank_stop_list_preserves_order() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::Vec(vec!["a".to_string(), "b".to_string()]));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let seqs = args.inference_config["stopSequences"]
            .as_array()
            .expect("array");
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0], "a");
        assert_eq!(seqs[1], "b");
    }

    #[tokio::test]
    async fn stop_list_passes_through() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.stop = Some(StringOrVec::Vec(vec!["a".to_string(), "b".to_string()]));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let seqs = args.inference_config["stopSequences"]
            .as_array()
            .expect("array");
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[1], "b");
    }

    #[tokio::test]
    async fn topp_conflict_drops_topp_when_temperature_present() {
        // claude-sonnet-4-5 has temperature_topp_conflict.
        let mut req = base_request(
            "global.anthropic.claude-sonnet-4-5-20250101-v1:0",
            vec![user_text("hi")],
        );
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let temp = args.inference_config["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 1e-6, "temperature ~0.7, got {temp}");
        assert!(
            args.inference_config.get("topP").is_none(),
            "topP must be dropped on conflict"
        );
    }

    #[tokio::test]
    async fn topp_kept_when_no_conflict() {
        // A model WITHOUT the conflict keeps both.
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let temp = args.inference_config["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 1e-6, "temperature ~0.7, got {temp}");
        let topp = args.inference_config["topP"].as_f64().unwrap();
        assert!((topp - 0.9).abs() < 1e-6, "topP ~0.9, got {topp}");
    }

    #[tokio::test]
    async fn drop_sampling_params_strips_both_temperature_and_topp() {
        // Opus 4.7+ / Sonnet 5 / Fable/Mythos 5 reject any non-default sampling
        // param with HTTP 400, so drop_sampling_params must strip BOTH.
        for model in [
            "us.anthropic.claude-opus-4-7",
            "anthropic.claude-opus-4-8",
            "claude-mythos-5",
            "claude-fable-5",
        ] {
            let mut req = base_request(model, vec![user_text("hi")]);
            req.temperature = Some(0.7);
            req.top_p = Some(0.9);
            let c = caps();
            let r = resolver(false);
            let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
                .await
                .expect("translate");
            assert!(
                args.inference_config.get("temperature").is_none(),
                "{model}: temperature must be dropped"
            );
            assert!(
                args.inference_config.get("topP").is_none(),
                "{model}: topP must be dropped"
            );
        }
    }

    #[tokio::test]
    async fn drop_sampling_params_strips_temperature_even_without_topp() {
        // Regression for issue #248: a lone temperature (no top_p) still 400s on
        // Opus 4.7, so it must be stripped even when top_p is absent.
        let mut req = base_request("us.anthropic.claude-opus-4-7", vec![user_text("hi")]);
        req.temperature = Some(0.7);
        req.top_p = None;
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        assert!(
            args.inference_config.get("temperature").is_none(),
            "lone temperature must be dropped on drop_sampling_params models"
        );
    }

    #[tokio::test]
    async fn opus_4_6_keeps_sampling_params() {
        // Opus 4.6 predates the deprecation and still accepts temperature/top_p —
        // it must NOT carry drop_sampling_params.
        let mut req = base_request("us.anthropic.claude-opus-4-6", vec![user_text("hi")]);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let temp = args.inference_config["temperature"].as_f64().unwrap();
        assert!(
            (temp - 0.7).abs() < 1e-6,
            "4.6 keeps temperature, got {temp}"
        );
        let topp = args.inference_config["topP"].as_f64().unwrap();
        assert!((topp - 0.9).abs() < 1e-6, "4.6 keeps topP, got {topp}");
    }

    #[tokio::test]
    async fn data_uri_image_decodes_to_image_block() {
        // "hi" base64 = "aGk=".
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::User {
                name: None,
                content: ContentInput::Parts(vec![
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "look".to_string(),
                    }),
                    ContentPart::Image(ImageContent {
                        r#type: "image_url".to_string(),
                        image_url: ImageUrl {
                            url: "data:image/png;base64,aGk=".to_string(),
                            detail: "auto".to_string(),
                        },
                    }),
                ]),
            }],
        );
        let c = caps();
        let r = resolver(true);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let content = args.messages[0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "look");
        assert_eq!(content[1]["image"]["format"], "png");
        let bytes = content[1]["image"]["source"]["bytes"]
            .as_array()
            .expect("bytes array");
        // "hi" = [104, 105].
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 104);
        assert_eq!(bytes[1], 105);
    }

    #[tokio::test]
    async fn image_to_non_image_model_is_bad_request() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::User {
                name: None,
                content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                    r#type: "image_url".to_string(),
                    image_url: ImageUrl {
                        url: "data:image/png;base64,aGk=".to_string(),
                        detail: "auto".to_string(),
                    },
                })]),
            }],
        );
        let c = caps();
        let r = resolver(false); // model lacks IMAGE modality
        let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect_err("must reject image on non-IMAGE model");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn remote_image_uses_resolver() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::User {
                name: None,
                content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                    r#type: "image_url".to_string(),
                    image_url: ImageUrl {
                        url: "https://example.com/cat.jpg".to_string(),
                        detail: "auto".to_string(),
                    },
                })]),
            }],
        );
        let c = caps();
        let r = TestResolver {
            image_ok: true,
            canned: Some((vec![1, 2, 3], "jpeg".to_string())),
        };
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let content = args.messages[0]["content"].as_array().expect("content");
        assert_eq!(content[0]["image"]["format"], "jpeg");
        let bytes = content[0]["image"]["source"]["bytes"]
            .as_array()
            .expect("bytes");
        assert_eq!(bytes.len(), 3);
    }

    #[tokio::test]
    async fn invalid_base64_image_is_bad_request() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::User {
                name: None,
                content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                    r#type: "image_url".to_string(),
                    image_url: ImageUrl {
                        url: "data:image/png;base64,!!!notbase64!!!".to_string(),
                        detail: "auto".to_string(),
                    },
                })]),
            }],
        );
        let c = caps();
        let r = resolver(true);
        let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect_err("invalid base64");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn contiguous_same_role_users_merge() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![user_text("Hello"), user_text("Who are you?")],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        // Two user messages merge into ONE user turn with two text blocks.
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        let content = msgs[0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Hello");
        assert_eq!(content[1]["text"], "Who are you?");
    }

    #[tokio::test]
    async fn no_assistant_prefill_appends_continuation() {
        // claude-opus-4-8 has no_assistant_prefill.
        let req = base_request(
            "global.anthropic.claude-opus-4-8-20251101-v1:0",
            vec![
                user_text("hi"),
                Message::Assistant {
                    name: None,
                    content: Some(ContentInput::Text("partial".to_string())),
                    tool_calls: None,
                },
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        // user, assistant, then an appended user continuation.
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(
            msgs[2]["content"][0]["text"],
            "Please continue your response from where you left off."
        );
    }

    #[tokio::test]
    async fn no_continuation_when_model_supports_prefill() {
        // A plain model (no no_assistant_prefill) ending on assistant: no append.
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![
                user_text("hi"),
                Message::Assistant {
                    name: None,
                    content: Some(ContentInput::Text("partial".to_string())),
                    tool_calls: None,
                },
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[tokio::test]
    async fn passthrough_allowlist_only_forwards_whitelisted_fields() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.extra
            .insert("thinking".to_string(), json!({"type": "x"}));
        req.extra
            .insert("anthropic_beta".to_string(), json!(["foo"]));
        // Not whitelisted — must NOT pass through.
        req.extra
            .insert("prompt_cache_key".to_string(), json!("secret"));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let fields = args
            .additional_model_request_fields
            .expect("additional fields present");
        assert!(fields.get("thinking").is_some());
        assert!(fields.get("anthropic_beta").is_some());
        assert!(
            fields.get("prompt_cache_key").is_none(),
            "non-whitelisted field leaked to Bedrock"
        );
    }

    #[tokio::test]
    async fn extra_body_passes_through_minus_prompt_caching() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.extra_body = Some(json!({
            "thinking": {"type": "enabled"},
            "prompt_caching": {"system": true}
        }));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let fields = args
            .additional_model_request_fields
            .expect("fields present");
        assert!(fields.get("thinking").is_some());
        assert!(
            fields.get("prompt_caching").is_none(),
            "prompt_caching is a control field, must not reach Bedrock"
        );
    }

    #[tokio::test]
    async fn thinking_field_drops_topp() {
        let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        req.top_p = Some(0.9);
        req.extra_body = Some(json!({ "thinking": {"type": "enabled"} }));
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        assert!(
            args.inference_config.get("topP").is_none(),
            "topP must be dropped when thinking is present"
        );
    }

    #[tokio::test]
    async fn context_1m_beta_auto_injected_from_config_header() {
        // claude-sonnet-4-6 has context_1m_beta. The config model entry does not
        // set per-model beta_headers, so beta_headers() is empty and nothing is
        // injected — this asserts the de-hardcoded behavior: injection sources
        // ONLY from caps.beta_headers(), never a literal. We additionally verify
        // the merge helper directly below.
        let req = base_request(
            "global.anthropic.claude-sonnet-4-6-20250601-v1:0",
            vec![user_text("hi")],
        );
        let c = caps();
        assert!(c.has(&req.model, Capability::Context1mBeta));
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        // beta_headers is empty in the shipped config, so no anthropic_beta key.
        let injected = args
            .additional_model_request_fields
            .as_ref()
            .and_then(|f| f.get("anthropic_beta"))
            .is_some();
        assert!(
            !injected,
            "no per-model beta header configured, so nothing injected"
        );
    }

    #[test]
    fn merge_anthropic_beta_handles_absent_string_and_list() {
        // Absent → singleton list.
        let mut m = Map::new();
        merge_anthropic_beta(&mut m, "ctx-1m");
        assert_eq!(m["anthropic_beta"], json!(["ctx-1m"]));

        // String (different) → both, in order.
        let mut m = Map::new();
        m.insert("anthropic_beta".to_string(), json!("existing"));
        merge_anthropic_beta(&mut m, "ctx-1m");
        assert_eq!(m["anthropic_beta"], json!(["existing", "ctx-1m"]));

        // String (same) → single.
        let mut m = Map::new();
        m.insert("anthropic_beta".to_string(), json!("ctx-1m"));
        merge_anthropic_beta(&mut m, "ctx-1m");
        assert_eq!(m["anthropic_beta"], json!(["ctx-1m"]));

        // List without header → appended.
        let mut m = Map::new();
        m.insert("anthropic_beta".to_string(), json!(["a"]));
        merge_anthropic_beta(&mut m, "ctx-1m");
        assert_eq!(m["anthropic_beta"], json!(["a", "ctx-1m"]));

        // List with header → unchanged (no dup).
        let mut m = Map::new();
        m.insert("anthropic_beta".to_string(), json!(["ctx-1m", "b"]));
        merge_anthropic_beta(&mut m, "ctx-1m");
        assert_eq!(m["anthropic_beta"], json!(["ctx-1m", "b"]));
    }

    #[tokio::test]
    async fn reasoning_seam_fields_merge_into_additional() {
        let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
        let c = caps();
        let r = resolver(false);
        let extras = ConverseExtras {
            reasoning_fields: Some(json!({"reasoning_config": {"type": "enabled"}})),
            tool_config: Some(json!({"tools": []})),
        };
        let args = to_converse_args(&req, &c, &r, &extras)
            .await
            .expect("translate");
        let fields = args
            .additional_model_request_fields
            .expect("fields present");
        assert!(fields.get("reasoning_config").is_some());
        // Tool config is placed verbatim into the slot (task-17 seam).
        assert_eq!(args.tool_config, Some(json!({"tools": []})));
    }

    #[tokio::test]
    async fn assistant_tool_calls_become_tool_use_blocks() {
        use crate::openai::schema::{ResponseFunction, ToolCall};
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![
                user_text("call a tool"),
                Message::Assistant {
                    name: None,
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        index: None,
                        id: Some("call_1".to_string()),
                        r#type: "function".to_string(),
                        function: ResponseFunction {
                            name: Some("get_weather".to_string()),
                            arguments: r#"{"city":"SF"}"#.to_string(),
                        },
                    }]),
                },
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        let assistant = &msgs[1];
        assert_eq!(assistant["role"], "assistant");
        let tu = &assistant["content"][0]["toolUse"];
        assert_eq!(tu["toolUseId"], "call_1");
        assert_eq!(tu["name"], "get_weather");
        assert_eq!(tu["input"]["city"], "SF");
    }

    #[tokio::test]
    async fn tool_message_becomes_user_tool_result() {
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![Message::Tool {
                content: ToolContentInput::Text("72F sunny".to_string()),
                tool_call_id: "call_1".to_string(),
            }],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        assert_eq!(msgs[0]["role"], "user");
        let tr = &msgs[0]["content"][0]["toolResult"];
        assert_eq!(tr["toolUseId"], "call_1");
        assert_eq!(tr["content"][0]["text"], "72F sunny");
    }

    #[tokio::test]
    async fn tool_result_and_text_user_turns_split() {
        // A toolResult user turn followed by a normal-text user turn must NOT
        // merge (bedrock.py:1740-1742).
        let req = base_request(
            "anthropic.claude-3-sonnet-v1:0",
            vec![
                Message::Tool {
                    content: ToolContentInput::Text("result".to_string()),
                    tool_call_id: "call_1".to_string(),
                },
                user_text("now do this"),
            ],
        );
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages");
        // Two separate user turns (split, not merged).
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0]["content"][0].get("toolResult").is_some());
        assert_eq!(msgs[1]["content"][0]["text"], "now do this");
    }

    #[test]
    fn parse_image_data_uri_rejects_non_data() {
        assert_eq!(parse_image_data_uri("https://x/y.png").expect("ok"), None);
    }
}
