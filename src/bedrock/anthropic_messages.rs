//! Anthropic Messages bodies for the Responses **tool-search lane**
//! (`InvokeModel` / `InvokeModelWithResponseStream`).
//!
//! Converse has no field for deferred tools (`defer_loading`) or for search
//! results (`tool_reference`), so a `/responses` request that uses
//! client-executed tool search on a `tool_search` model is sent to InvokeModel
//! instead. This module keeps that lane a thin shape change around the
//! existing Converse machinery:
//!
//! - request: the Converse-shaped JSON that
//!   [`crate::bedrock::responses_provider`] already assembles (messages, system,
//!   inference config, additional fields, tool config, cache points) becomes a
//!   Messages body ([`build_messages_body`]); the tools loaded by
//!   `tool_search_output` items are appended as deferred tools;
//! - non-stream response: the Messages response becomes Converse-output JSON
//!   ([`messages_response_to_converse_output`]) for the existing Responses
//!   mapper;
//! - stream: each Messages stream event becomes the equivalent SDK
//!   `ConverseStreamOutput` event ([`MessagesStreamAdapter`]) for the existing
//!   Responses stream state machine.
//!
//! Pure: no I/O, no model names. Protocol constants only.

use std::collections::HashSet;

use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ContentBlockStopEvent, ConversationRole, ConverseStreamMetadataEvent, ConverseStreamOutput,
    MessageStartEvent, MessageStopEvent, ReasoningContentBlockDelta, StopReason, TokenUsage,
    ToolUseBlockDelta, ToolUseBlockStart,
};
use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::bedrock::responses_translate::SearchedTool;
use crate::error::AppError;

/// `anthropic_version` for Bedrock InvokeModel Messages bodies (the value the
/// AWS Claude Messages parameter page requires).
pub const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Per-request ceiling on `defer_loading: true` tools (Claude tool-search
/// limits). Checked before the call so the 400 names the gateway's count.
pub const MAX_DEFERRED_TOOLS: usize = 10_000;

/// Maximum tool-name length on the Messages API (`^[a-zA-Z0-9_-]{1,128}$`).
const MAX_TOOL_NAME_LEN: usize = 128;

/// The Converse-shaped request slots plus the searched tools, borrowed from the
/// provider's assembly.
pub(crate) struct MessagesBodyInput<'a> {
    pub messages: &'a Value,
    pub system: &'a Value,
    pub inference_config: &'a Value,
    pub additional_fields: Option<&'a Value>,
    pub tool_config: Option<&'a Value>,
    pub output_config: Option<&'a Value>,
    /// Tools loaded by `tool_search_output`, declared with `defer_loading`.
    pub searched_tools: &'a [SearchedTool],
    /// `max_tokens` used when the inference config carries none.
    pub default_max_tokens: Option<u32>,
}

/// Build the InvokeModel Messages body.
///
/// Ordering of `tools`: the request's tools in request order (namespaces
/// flattened in place, `tool_search` where the client put it), then every
/// searched tool with `defer_loading: true` and no `cache_control`. Cache
/// points become `cache_control` on the block they follow, so the tools
/// breakpoint lands on the last non-deferred tool.
///
/// # Errors
/// [`AppError::BadRequest`] when no `max_tokens` is available, a tool name is
/// not valid upstream, a searched tool collides with a declared tool, there
/// are more than [`MAX_DEFERRED_TOOLS`] searched tools, every tool would be
/// deferred, or a content block has no Messages equivalent.
pub(crate) fn build_messages_body(input: &MessagesBodyInput<'_>) -> Result<Value, AppError> {
    let mut body = Map::new();
    body.insert("anthropic_version".to_string(), json!(ANTHROPIC_VERSION));

    let max_tokens = input
        .inference_config
        .get("maxTokens")
        .and_then(Value::as_u64)
        .or(input.default_max_tokens.map(u64::from))
        .ok_or_else(|| {
            AppError::BadRequest("max_output_tokens is required for this request".to_string())
        })?;
    body.insert("max_tokens".to_string(), json!(max_tokens));

    let system = convert_blocks(input.system.as_array().map_or(&[], Vec::as_slice))?;
    if !system.is_empty() {
        body.insert("system".to_string(), Value::Array(system));
    }

    let mut messages = Vec::new();
    for turn in input.messages.as_array().into_iter().flatten() {
        let role = turn.get("role").cloned().unwrap_or(Value::Null);
        let content = convert_blocks(
            turn.get("content")
                .and_then(Value::as_array)
                .map_or(&[], Vec::as_slice),
        )?;
        messages.push(json!({ "role": role, "content": content }));
    }
    body.insert("messages".to_string(), Value::Array(messages));

    for (from, to) in [
        ("temperature", "temperature"),
        ("topP", "top_p"),
        ("stopSequences", "stop_sequences"),
    ] {
        if let Some(value) = input.inference_config.get(from) {
            body.insert(to.to_string(), value.clone());
        }
    }

    let (mut tools, tool_choice) = match input.tool_config {
        Some(config) => convert_tool_config(config)?,
        None => (Vec::new(), None),
    };
    let mut declared: HashSet<String> = HashSet::new();
    for tool in &tools {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
        validate_tool_name(name)?;
        declared.insert(name.to_string());
    }
    if input.searched_tools.len() > MAX_DEFERRED_TOOLS {
        return Err(AppError::BadRequest(format!(
            "tool_search_output items load {} tools; at most {MAX_DEFERRED_TOOLS} deferred tools are allowed per request",
            input.searched_tools.len()
        )));
    }
    if !input.searched_tools.is_empty() && declared.is_empty() {
        return Err(AppError::BadRequest(
            "tools loaded by tool_search_output need at least one tool that is not deferred (send the tool_search tool)"
                .to_string(),
        ));
    }
    for searched in input.searched_tools {
        validate_tool_name(&searched.bedrock_name)?;
        if declared.contains(&searched.bedrock_name) {
            return Err(AppError::BadRequest(format!(
                "tool '{}' is both declared in tools and loaded by tool_search_output",
                searched.bedrock_name
            )));
        }
        let mut tool = Map::new();
        tool.insert("name".to_string(), json!(searched.bedrock_name));
        if let Some(description) = &searched.description {
            tool.insert("description".to_string(), json!(description));
        }
        tool.insert("input_schema".to_string(), searched.input_schema.clone());
        tool.insert("defer_loading".to_string(), Value::Bool(true));
        tools.push(Value::Object(tool));
    }
    if !tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(tools));
        if let Some(choice) = tool_choice {
            body.insert("tool_choice".to_string(), choice);
        }
    }

    if let Some(Value::Object(fields)) = input.additional_fields {
        for (key, value) in fields {
            match key.as_str() {
                // Converse's Claude `reasoning_config` is the Messages `thinking`
                // object (`{type: "enabled", budget_tokens}`).
                "reasoning_config" if value.is_object() => {
                    body.insert("thinking".to_string(), value.clone());
                }
                _ => {
                    body.insert(key.clone(), value.clone());
                }
            }
        }
    }

    if let Some(format) = input
        .output_config
        .map(output_format)
        .transpose()?
        .flatten()
    {
        let config = body
            .entry("output_config".to_string())
            .or_insert_with(|| json!({}));
        if let Value::Object(config) = config {
            config.insert("format".to_string(), format);
        }
    }

    Ok(Value::Object(body))
}

/// Converse content blocks → Messages content blocks. A `cachePoint` becomes
/// `cache_control` on the block before it.
fn convert_blocks(blocks: &[Value]) -> Result<Vec<Value>, AppError> {
    let mut out: Vec<Value> = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let Some(point) = block.get("cachePoint") {
            attach_cache_control(&mut out, point);
            continue;
        }
        out.push(convert_block(block)?);
    }
    Ok(out)
}

fn attach_cache_control(out: &mut [Value], point: &Value) {
    let Some(Value::Object(last)) = out.last_mut() else {
        return;
    };
    // Thinking blocks cannot carry cache_control; the gateway never places a
    // point after one, so skipping is purely defensive.
    if matches!(
        last.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    ) {
        return;
    }
    let mut control = Map::new();
    control.insert("type".to_string(), json!("ephemeral"));
    if let Some(ttl) = point.get("ttl").and_then(Value::as_str) {
        control.insert("ttl".to_string(), json!(ttl));
    }
    last.insert("cache_control".to_string(), Value::Object(control));
}

fn convert_block(block: &Value) -> Result<Value, AppError> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        return Ok(json!({ "type": "text", "text": text }));
    }
    if let Some(image) = block.get("image") {
        return image_block(image);
    }
    if let Some(tool_use) = block.get("toolUse") {
        return Ok(json!({
            "type": "tool_use",
            "id": tool_use.get("toolUseId").cloned().unwrap_or(Value::Null),
            "name": tool_use.get("name").cloned().unwrap_or(Value::Null),
            "input": tool_use.get("input").cloned().unwrap_or_else(|| json!({})),
        }));
    }
    if let Some(result) = block.get("toolResult") {
        return tool_result_block(result);
    }
    if let Some(reasoning) = block.get("reasoningContent") {
        return reasoning_block(reasoning);
    }
    Err(AppError::BadRequest(
        "a content block in this request has no Anthropic Messages equivalent".to_string(),
    ))
}

fn image_block(image: &Value) -> Result<Value, AppError> {
    let format = image
        .get("format")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::BadRequest("image block is missing its format".to_string()))?;
    let source = image.get("source").and_then(|s| s.get("bytes"));
    let data = match source {
        Some(Value::Array(bytes)) => {
            let raw: Vec<u8> = bytes
                .iter()
                .map(|b| b.as_u64().and_then(|b| u8::try_from(b).ok()))
                .collect::<Option<_>>()
                .ok_or_else(|| AppError::BadRequest("image bytes are malformed".to_string()))?;
            base64::engine::general_purpose::STANDARD.encode(raw)
        }
        Some(Value::String(encoded)) => encoded.clone(),
        _ => {
            return Err(AppError::BadRequest(
                "image block is missing its bytes".to_string(),
            ))
        }
    };
    Ok(json!({
        "type": "image",
        "source": { "type": "base64", "media_type": format!("image/{format}"), "data": data },
    }))
}

fn tool_result_block(result: &Value) -> Result<Value, AppError> {
    let mut content = Vec::new();
    for part in result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(reference) = part.get("toolReference") {
            content.push(json!({
                "type": "tool_reference",
                "tool_name": reference.get("toolName").cloned().unwrap_or(Value::Null),
            }));
        } else if let Some(text) = part.get("text").and_then(Value::as_str) {
            // Messages rejects empty text blocks; an empty tool output simply
            // has no content.
            if !text.is_empty() {
                content.push(json!({ "type": "text", "text": text }));
            }
        } else if let Some(image) = part.get("image") {
            content.push(image_block(image)?);
        } else if let Some(document) = part.get("json") {
            content.push(json!({ "type": "text", "text": document.to_string() }));
        } else {
            return Err(AppError::BadRequest(
                "a tool result block in this request has no Anthropic Messages equivalent"
                    .to_string(),
            ));
        }
    }
    let mut block = Map::new();
    block.insert("type".to_string(), json!("tool_result"));
    block.insert(
        "tool_use_id".to_string(),
        result.get("toolUseId").cloned().unwrap_or(Value::Null),
    );
    if !content.is_empty() {
        block.insert("content".to_string(), Value::Array(content));
    }
    if result.get("status").and_then(Value::as_str) == Some("error") {
        block.insert("is_error".to_string(), Value::Bool(true));
    }
    Ok(Value::Object(block))
}

/// Replayed reasoning (the `bedrock-reasoning-v1` envelope's Converse blocks) →
/// `thinking` / `redacted_thinking`.
///
/// A redacted block's envelope string is the standard base64 of the Converse
/// `redactedContent` blob, and Converse carries the Messages `data` string as
/// that blob's UTF-8 bytes (live, 2026-11-13). So `data` is the envelope
/// string base64-decoded, and this lane writes envelopes the same way
/// ([`messages_response_to_converse_output`], [`MessagesStreamAdapter`]): a
/// session may move between the Converse path and this one.
fn reasoning_block(reasoning: &Value) -> Result<Value, AppError> {
    if let Some(text) = reasoning.get("reasoningText") {
        let signature = text
            .get("signature")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                AppError::BadRequest("replayed reasoning has no signature".to_string())
            })?;
        return Ok(json!({
            "type": "thinking",
            "thinking": text.get("text").and_then(Value::as_str).unwrap_or_default(),
            "signature": signature,
        }));
    }
    if let Some(encoded) = reasoning.get("redactedContent").and_then(Value::as_str) {
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .ok_or_else(|| {
                AppError::BadRequest("replayed redacted reasoning is malformed".to_string())
            })?;
        return Ok(json!({ "type": "redacted_thinking", "data": data }));
    }
    Err(AppError::BadRequest(
        "replayed reasoning block is malformed".to_string(),
    ))
}

/// Converse `toolConfig` → Messages `tools` + `tool_choice`.
fn convert_tool_config(config: &Value) -> Result<(Vec<Value>, Option<Value>), AppError> {
    let mut tools: Vec<Value> = Vec::new();
    for entry in config
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(point) = entry.get("cachePoint") {
            attach_cache_control(&mut tools, point);
            continue;
        }
        let Some(spec) = entry.get("toolSpec") else {
            continue;
        };
        let mut tool = Map::new();
        tool.insert(
            "name".to_string(),
            spec.get("name").cloned().unwrap_or(Value::Null),
        );
        if let Some(description) = spec.get("description").and_then(Value::as_str) {
            tool.insert("description".to_string(), json!(description));
        }
        tool.insert(
            "input_schema".to_string(),
            spec.get("inputSchema")
                .and_then(|schema| schema.get("json"))
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object" })),
        );
        tools.push(Value::Object(tool));
    }
    let choice = config.get("toolChoice").and_then(|choice| {
        if choice.get("auto").is_some() {
            Some(json!({ "type": "auto" }))
        } else if choice.get("any").is_some() {
            Some(json!({ "type": "any" }))
        } else if choice.get("none").is_some() {
            Some(json!({ "type": "none" }))
        } else {
            choice
                .get("tool")
                .and_then(|tool| tool.get("name"))
                .map(|name| json!({ "type": "tool", "name": name }))
        }
    });
    Ok((tools, choice))
}

fn validate_tool_name(name: &str) -> Result<(), AppError> {
    let valid = !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LEN
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if valid {
        Ok(())
    } else {
        Err(AppError::BadRequest(format!(
            "tool name '{name}' is not valid upstream: names must match ^[a-zA-Z0-9_-]{{1,128}}$"
        )))
    }
}

/// Converse `outputConfig.textFormat` (schema as a JSON string) → Messages
/// `output_config.format` (schema as an object).
fn output_format(config: &Value) -> Result<Option<Value>, AppError> {
    let Some(schema) = config
        .pointer("/textFormat/structure/jsonSchema/schema")
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let schema: Value = serde_json::from_str(schema)
        .map_err(|e| AppError::Internal(format!("structured-output schema is not JSON: {e}")))?;
    Ok(Some(json!({ "type": "json_schema", "schema": schema })))
}

/// Messages `stop_reason` → the Converse value the Responses mappers read.
/// `refusal` (a classifier stop) is Converse's `content_filtered`: the
/// response is incomplete and no tool call becomes executable. The others
/// share Converse's spelling.
#[must_use]
pub fn map_stop_reason(reason: &str) -> &str {
    match reason {
        "refusal" => "content_filtered",
        other => other,
    }
}

/// Non-stream InvokeModel Messages response → Converse-output JSON
/// (`output.message.content`, `stopReason`, `usage`).
///
/// # Errors
/// [`AppError::Internal`] when the body has no `content` array.
pub fn messages_response_to_converse_output(body: &Value) -> Result<Value, AppError> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            AppError::Internal("InvokeModel response has no content array".to_string())
        })?;
    let blocks: Vec<Value> = content.iter().filter_map(response_block).collect();
    let stop_reason = body
        .get("stop_reason")
        .and_then(Value::as_str)
        .map_or("end_turn", map_stop_reason);
    let mut usage = UsageAccum::default();
    if let Some(u) = body.get("usage") {
        usage.merge(u);
    }
    Ok(json!({
        "output": { "message": { "role": "assistant", "content": blocks } },
        "stopReason": stop_reason,
        "usage": usage.to_converse_json(),
    }))
}

fn response_block(block: &Value) -> Option<Value> {
    let text = |key: &str| block.get(key).and_then(Value::as_str).unwrap_or_default();
    match block.get("type").and_then(Value::as_str)? {
        "text" => Some(json!({ "text": text("text") })),
        "thinking" => Some(json!({
            "reasoningContent": { "reasoningText": {
                "text": text("thinking"),
                "signature": block.get("signature").cloned().unwrap_or(Value::Null),
            } }
        })),
        // Same blob encoding as Converse: base64 of the `data` string's bytes.
        "redacted_thinking" => Some(json!({
            "reasoningContent": { "redactedContent":
                base64::engine::general_purpose::STANDARD.encode(text("data").as_bytes()) }
        })),
        "tool_use" => Some(json!({
            "toolUse": {
                "toolUseId": text("id"),
                "name": text("name"),
                "input": block.get("input").cloned().unwrap_or_else(|| json!({})),
            }
        })),
        _ => None,
    }
}

/// Messages usage, accumulated across `message_start` and `message_delta`
/// (whose counts are cumulative, so the last value wins).
#[derive(Debug, Default, Clone, Copy)]
struct UsageAccum {
    input: i32,
    output: i32,
    cache_read: Option<i32>,
    cache_write: Option<i32>,
}

impl UsageAccum {
    fn merge(&mut self, usage: &Value) {
        let read = |key: &str| {
            usage
                .get(key)
                .and_then(Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
        };
        if let Some(v) = read("input_tokens") {
            self.input = v;
        }
        if let Some(v) = read("output_tokens") {
            self.output = v;
        }
        if let Some(v) = read("cache_read_input_tokens") {
            self.cache_read = Some(v);
        }
        if let Some(v) = read("cache_creation_input_tokens") {
            self.cache_write = Some(v);
        }
    }

    fn total(&self) -> i32 {
        self.input
            .saturating_add(self.output)
            .saturating_add(self.cache_read.unwrap_or(0))
            .saturating_add(self.cache_write.unwrap_or(0))
    }

    /// Converse usage JSON: `inputTokens` is the uncached part, as in
    /// Messages, so the shared token formula applies unchanged.
    fn to_converse_json(self) -> Value {
        let mut map = Map::new();
        map.insert("inputTokens".to_string(), json!(self.input));
        map.insert("outputTokens".to_string(), json!(self.output));
        map.insert("totalTokens".to_string(), json!(self.total()));
        if let Some(read) = self.cache_read {
            map.insert("cacheReadInputTokens".to_string(), json!(read));
        }
        if let Some(write) = self.cache_write {
            map.insert("cacheWriteInputTokens".to_string(), json!(write));
        }
        Value::Object(map)
    }

    fn to_token_usage(self) -> Result<TokenUsage, AppError> {
        TokenUsage::builder()
            .input_tokens(self.input)
            .output_tokens(self.output)
            .total_tokens(self.total())
            .set_cache_read_input_tokens(self.cache_read)
            .set_cache_write_input_tokens(self.cache_write)
            .build()
            .map_err(sdk_build_error)
    }
}

fn sdk_build_error(e: impl std::fmt::Display) -> AppError {
    AppError::Internal(format!("failed to build stream event: {e}"))
}

/// Converts InvokeModelWithResponseStream Messages events into the SDK
/// `ConverseStreamOutput` events the Responses stream state machine consumes.
///
/// `message_stop` yields `MessageStop` (mapped stop reason) followed by
/// `Metadata` (usage), which is where the state machine finalizes. `ping` and
/// unknown event or block types are ignored.
#[derive(Debug, Default)]
pub struct MessagesStreamAdapter {
    stop_reason: Option<String>,
    usage: UsageAccum,
}

impl MessagesStreamAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Map one `chunk.bytes` payload (one Messages stream event as JSON).
    ///
    /// # Errors
    /// The upstream error for an `error` event, or
    /// [`AppError::UpstreamBedrock`] for a payload that is not JSON.
    pub fn map_chunk(&mut self, bytes: &[u8]) -> Result<Vec<ConverseStreamOutput>, AppError> {
        let event: Value = serde_json::from_slice(bytes).map_err(|_| {
            AppError::UpstreamBedrock("Upstream returned a malformed stream event.".to_string())
        })?;
        self.map_event(&event)
    }

    /// Map one parsed Messages stream event.
    ///
    /// # Errors
    /// As [`Self::map_chunk`].
    pub fn map_event(&mut self, event: &Value) -> Result<Vec<ConverseStreamOutput>, AppError> {
        let index = event
            .get("index")
            .and_then(Value::as_i64)
            .and_then(|i| i32::try_from(i).ok())
            .unwrap_or(0);
        let str_of = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_string);
        let mut out = Vec::new();
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                if let Some(usage) = event.pointer("/message/usage") {
                    self.usage.merge(usage);
                }
                out.push(ConverseStreamOutput::MessageStart(
                    MessageStartEvent::builder()
                        .role(ConversationRole::Assistant)
                        .build()
                        .map_err(sdk_build_error)?,
                ));
            }
            "content_block_start" => {
                let block = event.get("content_block").cloned().unwrap_or(Value::Null);
                match block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                {
                    "tool_use" => {
                        let start = ToolUseBlockStart::builder()
                            .tool_use_id(str_of(&block, "id").unwrap_or_default())
                            .name(str_of(&block, "name").unwrap_or_default())
                            .build()
                            .map_err(sdk_build_error)?;
                        out.push(ConverseStreamOutput::ContentBlockStart(
                            ContentBlockStartEvent::builder()
                                .content_block_index(index)
                                .start(ContentBlockStart::ToolUse(start))
                                .build()
                                .map_err(sdk_build_error)?,
                        ));
                    }
                    "text" => {
                        if let Some(text) = str_of(&block, "text").filter(|t| !t.is_empty()) {
                            out.push(delta(index, ContentBlockDelta::Text(text))?);
                        }
                    }
                    "thinking" => {
                        if let Some(text) = str_of(&block, "thinking").filter(|t| !t.is_empty()) {
                            out.push(delta(
                                index,
                                ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Text(text),
                                ),
                            )?);
                        }
                        if let Some(sig) = str_of(&block, "signature").filter(|s| !s.is_empty()) {
                            out.push(delta(
                                index,
                                ContentBlockDelta::ReasoningContent(
                                    ReasoningContentBlockDelta::Signature(sig),
                                ),
                            )?);
                        }
                    }
                    "redacted_thinking" => {
                        // Converse's blob for a redacted block is the `data`
                        // string's UTF-8 bytes; the stream state base64-encodes
                        // them into the envelope like a ConverseStream block.
                        let bytes = str_of(&block, "data").unwrap_or_default().into_bytes();
                        out.push(delta(
                            index,
                            ContentBlockDelta::ReasoningContent(
                                ReasoningContentBlockDelta::RedactedContent(Blob::new(bytes)),
                            ),
                        )?);
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let d = event.get("delta").cloned().unwrap_or(Value::Null);
                let mapped = match d.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "text_delta" => str_of(&d, "text").map(ContentBlockDelta::Text),
                    "thinking_delta" => str_of(&d, "thinking").map(|t| {
                        ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(t))
                    }),
                    "signature_delta" => str_of(&d, "signature").map(|s| {
                        ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                            s,
                        ))
                    }),
                    "input_json_delta" => str_of(&d, "partial_json")
                        .map(|json| {
                            ToolUseBlockDelta::builder()
                                .input(json)
                                .build()
                                .map(ContentBlockDelta::ToolUse)
                                .map_err(sdk_build_error)
                        })
                        .transpose()?,
                    _ => None,
                };
                if let Some(mapped) = mapped {
                    out.push(delta(index, mapped)?);
                }
            }
            "content_block_stop" => {
                out.push(ConverseStreamOutput::ContentBlockStop(
                    ContentBlockStopEvent::builder()
                        .content_block_index(index)
                        .build()
                        .map_err(sdk_build_error)?,
                ));
            }
            "message_delta" => {
                if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(usage) = event.get("usage") {
                    self.usage.merge(usage);
                }
            }
            "message_stop" => {
                let reason = self
                    .stop_reason
                    .as_deref()
                    .map_or("end_turn", map_stop_reason);
                out.push(ConverseStreamOutput::MessageStop(
                    MessageStopEvent::builder()
                        .stop_reason(StopReason::from(reason))
                        .build()
                        .map_err(sdk_build_error)?,
                ));
                out.push(ConverseStreamOutput::Metadata(
                    ConverseStreamMetadataEvent::builder()
                        .usage(self.usage.to_token_usage()?)
                        .build(),
                ));
            }
            "error" => return Err(stream_error(event.get("error"))),
            _ => {}
        }
        Ok(out)
    }
}

fn delta(index: i32, delta: ContentBlockDelta) -> Result<ConverseStreamOutput, AppError> {
    Ok(ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(delta)
            .build()
            .map_err(sdk_build_error)?,
    ))
}

/// A Messages `error` stream event → the gateway error class it stands for.
fn stream_error(error: Option<&Value>) -> AppError {
    let message = error
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Upstream stream error.")
        .to_string();
    match error.and_then(|e| e.get("type")).and_then(Value::as_str) {
        Some("rate_limit_error") => AppError::Throttled(message),
        Some("invalid_request_error") => AppError::BadRequest(message),
        _ => AppError::UpstreamBedrock(message),
    }
}

#[cfg(test)]
#[path = "anthropic_messages_tests.rs"]
mod tests;
