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

use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::bedrock::capsule::{decode_capsule, is_capsule, CapsuleRuntime, DecodedCapsule};
use crate::domain::{Capability, ModelCapabilities};
use crate::error::AppError;
use crate::openai::schema::{
    ChatRequest, ContentInput, ContentPart, Message, ResponseFormat, StringOrVec,
    SystemContentInput, ToolContentInput,
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
    /// `outputConfig` slot — native structured output (`textFormat`). `None`
    /// unless the request carried a non-passthrough `response_format`.
    pub output_config: Option<Value>,
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
        if let Some(oc) = &self.output_config {
            obj.insert("outputConfig".to_string(), oc.clone());
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
#[derive(Clone, Default)]
pub struct ConverseExtras {
    /// Additional model request fields produced by reasoning normalization
    /// (e.g. `{"thinking": {...}}` or `{"reasoning_config": {...}}`). Merged
    /// into `additionalModelRequestFields`. If this contains a `thinking`
    /// key, `topP` is dropped from inference config (bedrock.py:1267-1268).
    pub reasoning_fields: Option<Value>,
    /// The fully-built `toolConfig` object (task 17). Placed verbatim into
    /// [`ConverseArgs::tool_config`].
    pub tool_config: Option<Value>,
    /// Chat reasoning capsule decoder state. Decoding is independent of the
    /// encoder feature flag and applies only to reserved `brtc_v1.` ids.
    pub capsule: Option<Arc<CapsuleRuntime>>,
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

/// Flatten a system/developer message's content into a single text string.
///
/// A [`SystemContentInput::Text`] passes through unchanged; a
/// [`SystemContentInput::Parts`] joins each text part's `.text` with `"\n"`.
///
/// # Errors
/// Returns [`AppError::BadRequest`] if any part is a non-text (image) part.
fn flatten_system_content(content: &SystemContentInput) -> Result<String, AppError> {
    match content {
        SystemContentInput::Text(text) => Ok(text.clone()),
        SystemContentInput::Parts(parts) => {
            let mut texts = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ContentPart::Text(text_content) => texts.push(text_content.text.as_str()),
                    ContentPart::Image(_) => {
                        return Err(AppError::BadRequest(
                            "system/developer message does not accept non-text content parts"
                                .to_string(),
                        ));
                    }
                }
            }
            Ok(texts.join("\n"))
        }
    }
}

/// Build the Bedrock `system` blocks from system/developer messages.
///
/// Ports `_parse_system_prompts` (bedrock.py:709-744) for the translation
/// scope: every `system`/`developer` message contributes a `{"text": ...}`
/// block. String content passes through; a text-part array is flattened with a
/// `"\n"` join. Prompt-cache `cachePoint` insertion (bedrock.py:736-772) is a
/// separate task (20) and is intentionally omitted here.
///
/// # Errors
/// Returns [`AppError::BadRequest`] if a system/developer array contains a
/// non-text (image) content part.
pub fn parse_system_prompts(req: &ChatRequest) -> Result<Value, AppError> {
    let mut blocks: Vec<Value> = Vec::new();
    for message in &req.messages {
        match message {
            Message::System { content, .. } | Message::Developer { content, .. } => {
                let text = flatten_system_content(content)?;
                push_non_empty_text_block(&mut blocks, &text);
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

fn decode_continuation_capsule(
    candidate: &str,
    runtime: Option<&CapsuleRuntime>,
) -> Result<DecodedCapsule, AppError> {
    let runtime = runtime.ok_or_else(|| {
        AppError::BadRequest("reasoning capsule decoder is unavailable".to_string())
    })?;
    decode_capsule(candidate, &runtime.keyring)
}

fn strip_replayed_reasoning(
    content: &Option<ContentInput>,
    reasoning_blocks: &[Value],
) -> Result<Option<ContentInput>, AppError> {
    let mut reasoning_text = String::new();
    let mut has_reasoning_text = false;
    for block in reasoning_blocks {
        if let Some(text) = block
            .get("reasoningText")
            .and_then(|reasoning| reasoning.get("text"))
            .and_then(Value::as_str)
        {
            has_reasoning_text = true;
            reasoning_text.push_str(text);
        }
    }

    if !has_reasoning_text || content.is_none() {
        return Ok(content.clone());
    }

    let expected_prefix = format!("<think>{reasoning_text}</think>");
    match content {
        Some(ContentInput::Text(text)) => text
            .strip_prefix(&expected_prefix)
            .map(|rest| Some(ContentInput::Text(rest.to_string())))
            .ok_or_else(|| {
                AppError::BadRequest(
                    "assistant reasoning prefix does not match the reasoning capsule".to_string(),
                )
            }),
        Some(ContentInput::Parts(parts)) => {
            let mut stripped = parts.clone();
            let leading_text = match stripped.first_mut() {
                Some(ContentPart::Text(text)) => text,
                Some(ContentPart::Image(_)) | None => {
                    return Err(AppError::BadRequest(
                        "assistant content parts have no leading text for reasoning replay"
                            .to_string(),
                    ));
                }
            };
            leading_text.text = leading_text
                .text
                .strip_prefix(&expected_prefix)
                .map(str::to_string)
                .ok_or_else(|| {
                    AppError::BadRequest(
                        "assistant reasoning prefix does not match the reasoning capsule"
                            .to_string(),
                    )
                })?;
            Ok(Some(ContentInput::Parts(stripped)))
        }
        None => Ok(None),
    }
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
    capsule: Option<&CapsuleRuntime>,
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
                let mut shared_reasoning: Option<Vec<Value>> = None;
                let mut tool_use_blocks = Vec::new();
                if let Some(calls) = tool_calls {
                    for call in calls {
                        let tool_use_id = match call.id.as_deref() {
                            Some(id) if is_capsule(id) => {
                                let decoded = decode_continuation_capsule(id, capsule)?;
                                if shared_reasoning
                                    .as_ref()
                                    .is_some_and(|blocks| blocks != &decoded.reasoning_blocks)
                                {
                                    return Err(AppError::BadRequest(
                                        "parallel reasoning capsules carry different reasoning blocks"
                                            .to_string(),
                                    ));
                                }
                                if shared_reasoning.is_none() {
                                    shared_reasoning = Some(decoded.reasoning_blocks);
                                }
                                Some(decoded.tool_use_id)
                            }
                            _ => call.id.clone(),
                        };
                        // Tool-call arguments are a JSON string (bedrock.py:815).
                        let input: Value =
                            serde_json::from_str(&call.function.arguments).map_err(|e| {
                                AppError::BadRequest(format!(
                                    "invalid tool_call arguments JSON: {e}"
                                ))
                            })?;
                        tool_use_blocks.push(json!({
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": call.function.name,
                                "input": input,
                            }
                        }));
                    }
                }

                let replayed_content = if let Some(reasoning_blocks) = shared_reasoning {
                    let stripped_content = strip_replayed_reasoning(content, &reasoning_blocks)?;
                    assistant_content.extend(
                        reasoning_blocks
                            .into_iter()
                            .map(|block| json!({"reasoningContent": block})),
                    );
                    Some(stripped_content)
                } else {
                    None
                };
                let content = replayed_content.as_ref().unwrap_or(content);
                if assistant_has_text(content) {
                    if let Some(content) = content {
                        assistant_content
                            .extend(parse_content_parts(content, &req.model, resolver).await?);
                    }
                }
                assistant_content.extend(tool_use_blocks);
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
                let decoded_tool_call_id = if is_capsule(tool_call_id) {
                    decode_continuation_capsule(tool_call_id, capsule)?.tool_use_id
                } else {
                    tool_call_id.clone()
                };
                out.push(IntermediateMessage {
                    role: "user".to_string(),
                    content: vec![json!({
                        "toolResult": {
                            "toolUseId": decoded_tool_call_id,
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
    capsule: Option<&CapsuleRuntime>,
) -> Result<Value, AppError> {
    let intermediates = build_intermediate_messages(req, resolver, capsule).await?;
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

    // The current OpenAI field wins over the deprecated legacy field. If both
    // are absent, do not invent a low gateway-side truncation limit.
    if let Some(max_tokens) = req.max_completion_tokens.or(req.max_tokens) {
        cfg.insert("maxTokens".to_string(), json!(max_tokens));
    }

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
    let messages = parse_messages(req, caps, resolver, extras.capsule.as_deref()).await?;
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

    let output_config = build_output_config(req, caps)?;

    Ok(ConverseArgs {
        model_id: req.model.clone(),
        messages,
        system,
        inference_config,
        additional_model_request_fields,
        // Tool config is the task-17 seam; translation never builds it.
        tool_config: extras.tool_config.clone(),
        output_config,
    })
}

/// Build the Bedrock `outputConfig` from an OpenAI `response_format`.
///
/// Returns `None` for an absent or `text` (passthrough) format — byte-stable
/// with the pre-feature behavior. For `json_object` / `json_schema`, gates on
/// [`Capability::StructuredOutput`] (clean 400 when unsupported) and emits a
/// native `outputConfig.textFormat` with the JSON schema STRINGIFIED into the
/// `jsonSchema.schema` string slot (grammar-constrained decoding — not
/// tool-coercion).
fn build_output_config(
    req: &ChatRequest,
    caps: &dyn ModelCapabilities,
) -> Result<Option<Value>, AppError> {
    let format = match &req.response_format {
        None | Some(ResponseFormat::Text) => return Ok(None),
        Some(other) => other,
    };

    if !caps.has(&req.model, Capability::StructuredOutput) {
        return Err(AppError::BadRequest(format!(
            "model `{}` does not support response_format (structured output)",
            req.model
        )));
    }

    let (schema, name) = match format {
        ResponseFormat::Text => unreachable!("text handled above"),
        ResponseFormat::JsonObject => (
            json!({ "type": "object", "additionalProperties": false }),
            None,
        ),
        ResponseFormat::JsonSchema { json_schema } => (
            json_schema
                .schema
                .clone()
                .unwrap_or_else(|| json!({ "type": "object" })),
            json_schema.name.clone(),
        ),
    };

    let schema_string = serde_json::to_string(&schema).map_err(|e| {
        AppError::Internal(format!("failed to stringify response_format schema: {e}"))
    })?;

    let mut json_schema = Map::new();
    json_schema.insert("schema".to_string(), Value::String(schema_string));
    if let Some(name) = name {
        json_schema.insert("name".to_string(), Value::String(name));
    }

    Ok(Some(json!({
        "textFormat": {
            "type": "json_schema",
            "structure": { "jsonSchema": Value::Object(json_schema) }
        }
    })))
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
#[path = "translate_tests.rs"]
mod tests;
