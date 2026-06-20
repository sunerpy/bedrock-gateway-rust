//! OpenAI **Responses API** → Bedrock Converse input translation (pure).
//!
//! This module parses a [`ResponsesRequest`] into the SAME intermediate shape
//! the chat translation produces — a Bedrock Converse `messages` turn array plus
//! a `system` block array — so the existing Converse call path
//! ([`crate::bedrock::translate`]) can consume it unchanged. Nothing here touches
//! the network or the AWS SDK except via the injected
//! [`ImageResolver`](crate::bedrock::translate::ImageResolver) seam (for remote
//! image URLs); `data:` URIs are decoded inline.
//!
//! ## What is translated
//!
//! - `input: String`            → a single Bedrock `user` turn (one text block).
//! - `input: [item, ...]`       → ordered Bedrock turns (see below).
//! - `instructions` (top-level) → a Bedrock `system` block, **prepended** before
//!   any `system`/`developer` message-item system blocks.
//! - `message` items            → role mapping (`user` → a Bedrock `user` turn;
//!   `assistant` → a Bedrock `assistant` turn — codex replays the prior
//!   assistant turn as an input message on every multi-turn request;
//!   `system` / `developer` → a Bedrock `system` block, NOT a turn). Content
//!   parts map `input_text` and `output_text` → `{text}` (codex echoes the
//!   prior assistant text as an `output_text` part), `input_image` → an image
//!   block (reusing the chat multimodal image handling), and `input_file` is
//!   currently rejected as unsupported (no Bedrock document-block mapping wired
//!   yet).
//! - `function_call` items      → a Bedrock assistant `toolUse` turn (reusing the
//!   Bedrock-side `toolUse` shape from [`crate::bedrock::tools`]).
//! - `function_call_output`     → a Bedrock user `toolResult` turn (reusing
//!   [`crate::bedrock::tools::tool_message_to_tool_result_turn`]).
//! - `reasoning` items          → **DROPPED** (Strategy A). Bedrock has no
//!   equivalent for `encrypted_content`; there is no signature to replay. See
//!   the inline comment at the drop site.
//!
//! Contiguous same-role turns are merged / split with the SAME rule the chat
//! path uses ([`crate::bedrock::tools::should_split_same_role_merge`]): tool-only
//! turns merge with each other but split from normal content.
//!
//! ## Request-level reasoning vs reasoning INPUT items
//!
//! The request-level `reasoning { effort }` field is a SEPARATE concept from the
//! dropped reasoning input items: it drives the Bedrock thinking budget via
//! [`crate::bedrock::reasoning::build_reasoning_config`], identically to chat.
//! [`reasoning_outcome`] exposes that mapping for the provider seam (T13); the
//! input-item parser itself never emits a Bedrock reasoning block.
//!
//! ## Stateless rejection matrix (codex-safety-critical)
//!
//! - `store` (any value)            → **accept & IGNORE** (codex sends
//!   `store:false`; a 400 would break codex).
//! - `previous_response_id`         → **accept & IGNORE** (lenient).
//! - `include` (e.g.
//!   `["reasoning.encrypted_content"]`) → **accept & IGNORE** (emit nothing
//!   extra; Strategy A).
//! - built-in server tools in `tools[]` (any tool type other than `function`,
//!   i.e. `file_search` / `web_search` / `code_interpreter` / `mcp` /
//!   `computer` / `image_generation`) → **400** ([`AppError::BadRequest`]).
//!   Because [`ResponsesTool`] only models the `function` variant, an unknown
//!   tool type fails deserialization at the wire boundary; this parser
//!   additionally guards by re-checking any captured `extra` tool entries.
//! - unsatisfiable `text.format` (a malformed / unsupported structured-output
//!   schema) → **400**. A well-formed `json_schema` that can pass through to
//!   Bedrock is accepted.

use serde_json::{json, Value};

use crate::bedrock::tools::{should_split_same_role_merge, tool_message_to_tool_result_turn};
use crate::bedrock::translate::{parse_image_data_uri, ImageResolver};
use crate::domain::ModelCapabilities;
use crate::error::AppError;
use crate::openai::responses_schema::{
    FunctionCallOutputValue, ResponseContentPart, ResponseInputItem, ResponsesContent,
    ResponsesInput, ResponsesRequest, ResponsesRole, ResponsesTool,
};
use crate::openai::schema::ReasoningEffort;

/// The parsed Responses input in the shape the Bedrock Converse call consumes.
///
/// Both fields are `serde_json::Value` arrays using Bedrock Converse key names,
/// mirroring [`crate::bedrock::translate::ConverseArgs::messages`] /
/// [`crate::bedrock::translate::ConverseArgs::system`] exactly, so a provider can
/// drop them straight into a Converse request alongside the inference config the
/// rest of the request produces.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponsesConverseInput {
    /// Bedrock `messages` — an array of `{"role", "content"}` turns.
    pub messages: Value,
    /// Bedrock `system` — an array of `{"text"}` blocks (may be empty).
    pub system: Value,
}

/// A per-item intermediate before same-role reframing: a Bedrock role plus its
/// content blocks. Mirrors the chat path's `IntermediateMessage`.
struct Turn {
    role: String,
    content: Vec<Value>,
}

/// Translate a [`ResponsesRequest`] into Bedrock Converse `messages` + `system`.
///
/// Runs the full stateless rejection matrix first (so a malformed request is
/// rejected before any work), then parses `input` (string or item array) into
/// Bedrock turns, prepends `instructions` to the system blocks, and merges
/// contiguous same-role turns with the shared chat reframing rule.
///
/// `model_id` is the resolved model id used for the image-modality gate and for
/// any downstream Bedrock call; `resolver` decodes/fetches images; `caps` is
/// accepted for parity with the chat translate signature (reserved for future
/// capability-driven branching — currently unused in the input parse itself).
///
/// # Errors
/// Returns [`AppError::BadRequest`] for a built-in server tool, an unsatisfiable
/// `text.format`, an `input_file` content part, or image-modality/decode
/// failures; propagates resolver errors.
pub async fn to_responses_converse_input(
    req: &ResponsesRequest,
    model_id: &str,
    resolver: &dyn ImageResolver,
    caps: &dyn ModelCapabilities,
) -> Result<ResponsesConverseInput, AppError> {
    // `caps` is part of the parity signature; the input parse does not branch on
    // capabilities today (kept for future use — e.g. per-model file handling).
    let _ = caps;

    // 1) Stateless rejection matrix. `store` / `previous_response_id` / `include`
    //    are intentionally NOT inspected here: they are accepted and ignored.
    reject_builtin_tools(req)?;
    reject_unsatisfiable_text_format(req)?;

    // 2) System blocks: `instructions` first (prepended), then any system /
    //    developer message-item blocks, in input order.
    let mut system_blocks: Vec<Value> = Vec::new();
    if let Some(instructions) = &req.instructions {
        system_blocks.push(json!({ "text": instructions }));
    }

    // 3) Parse input into per-item turns (system/developer items contribute to
    //    `system_blocks` instead of producing a turn).
    let turns = match &req.input {
        ResponsesInput::Text(text) => {
            vec![Turn {
                role: "user".to_string(),
                content: vec![json!({ "text": text })],
            }]
        }
        ResponsesInput::Items(items) => {
            parse_input_items(items, model_id, resolver, &mut system_blocks).await?
        }
    };

    Ok(ResponsesConverseInput {
        messages: reframe_turns(turns),
        system: Value::Array(system_blocks),
    })
}

/// Parse the array-form `input` items into Bedrock turns, routing
/// system/developer message items into `system_blocks`.
async fn parse_input_items(
    items: &[ResponseInputItem],
    model_id: &str,
    resolver: &dyn ImageResolver,
    system_blocks: &mut Vec<Value>,
) -> Result<Vec<Turn>, AppError> {
    let mut turns: Vec<Turn> = Vec::new();
    for item in items {
        match item {
            ResponseInputItem::Message { role, content } => match role {
                // user → a Bedrock user turn.
                ResponsesRole::User => {
                    let blocks = parse_message_content(content, model_id, resolver).await?;
                    turns.push(Turn {
                        role: "user".to_string(),
                        content: blocks,
                    });
                }
                // assistant → a Bedrock assistant turn. codex replays the prior
                // assistant turn as an input message (its text arriving as an
                // `output_text` content part); same-role reframing merges it.
                ResponsesRole::Assistant => {
                    let blocks = parse_message_content(content, model_id, resolver).await?;
                    turns.push(Turn {
                        role: "assistant".to_string(),
                        content: blocks,
                    });
                }
                // system / developer → Bedrock system blocks (NOT a turn).
                ResponsesRole::System | ResponsesRole::Developer => {
                    push_system_blocks(content, system_blocks)?;
                }
            },
            // function_call → assistant toolUse turn (reusing the Bedrock-side
            // toolUse block shape). `arguments` is a JSON string parsed into the
            // `input` object, matching the chat path (tools.rs).
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let input: Value = serde_json::from_str(arguments).map_err(|e| {
                    AppError::BadRequest(format!("invalid function_call arguments JSON: {e}"))
                })?;
                turns.push(Turn {
                    role: "assistant".to_string(),
                    content: vec![json!({
                        "toolUse": {
                            "toolUseId": call_id,
                            "name": name,
                            "input": input,
                        }
                    })],
                });
            }
            // function_call_output → user toolResult turn (reusing tools.rs).
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                let content =
                    parse_function_call_output(call_id, output, model_id, resolver).await?;
                turns.push(Turn {
                    role: "user".to_string(),
                    content,
                });
            }
            // item_reference → DROP. The gateway is stateless and cannot resolve
            // OpenAI-hosted stored items; accepting and ignoring is safer than
            // failing opencode continuation payloads that contain references.
            ResponseInputItem::ItemReference { .. } => continue,
            ResponseInputItem::Other { item_type, .. } => {
                return Err(AppError::BadRequest(format!(
                    "Responses input item type '{item_type}' is not supported"
                )));
            }
            // reasoning → DROP (Strategy A). Bedrock has no equivalent for the
            // Responses `encrypted_content` blob and there is no thinking
            // signature to replay, so we emit nothing for reasoning input items.
            // (Pre-staged fallback: if a future live tool-follow-up test shows a
            // Bedrock ValidationException on a missing reasoning signature, this
            // is the escalation point — see notepad. For now: drop.)
            ResponseInputItem::Reasoning { .. } => continue,
        }
    }
    Ok(turns)
}

async fn parse_function_call_output(
    call_id: &str,
    output: &FunctionCallOutputValue,
    model_id: &str,
    resolver: &dyn ImageResolver,
) -> Result<Vec<Value>, AppError> {
    match output {
        FunctionCallOutputValue::Text(text) => {
            let turn = tool_message_to_tool_result_turn(call_id, text);
            Ok(turn
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default())
        }
        FunctionCallOutputValue::Parts(parts) => {
            let content = ResponsesContent::Parts(parts.clone());
            let blocks = parse_message_content(&content, model_id, resolver).await?;
            Ok(vec![json!({
                "toolResult": {
                    "toolUseId": call_id,
                    "content": blocks,
                }
            })])
        }
    }
}

/// Append system/developer message content as `{"text": ...}` system blocks.
///
/// A string body becomes one block; a parts body contributes one block per
/// `input_text` / `output_text` part. Images/files are not valid in a system
/// block.
fn push_system_blocks(
    content: &ResponsesContent,
    system_blocks: &mut Vec<Value>,
) -> Result<(), AppError> {
    match content {
        ResponsesContent::Text(text) => {
            system_blocks.push(json!({ "text": text }));
        }
        ResponsesContent::Parts(parts) => {
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text } => {
                        system_blocks.push(json!({ "text": text }));
                    }
                    ResponseContentPart::InputImage { .. } => {
                        return Err(AppError::BadRequest(
                            "image content is not supported in a system/developer message"
                                .to_string(),
                        ));
                    }
                    ResponseContentPart::InputFile { .. } => {
                        return Err(AppError::BadRequest(
                            "file content is not supported in a system/developer message"
                                .to_string(),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Parse a user/assistant message's content into Bedrock content blocks.
///
/// Mirrors the chat path's `parse_content_parts`: a string body → one `{text}`
/// block; a parts body maps `input_text` / `output_text` → `{text}` and
/// `input_image` → an image block (decoding `data:` URIs inline, delegating
/// remote URLs to `resolver`, and enforcing the IMAGE-modality gate).
/// `input_file` is rejected as unsupported.
async fn parse_message_content(
    content: &ResponsesContent,
    model_id: &str,
    resolver: &dyn ImageResolver,
) -> Result<Vec<Value>, AppError> {
    match content {
        ResponsesContent::Text(text) => Ok(vec![json!({ "text": text })]),
        ResponsesContent::Parts(parts) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text } => {
                        blocks.push(json!({ "text": text }));
                    }
                    ResponseContentPart::InputImage { image_url, .. } => {
                        if !resolver.supports_image(model_id) {
                            return Err(AppError::BadRequest(format!(
                                "Multimodal message is currently not supported by {model_id}"
                            )));
                        }
                        // Reuse the chat image handling: decode data: URIs inline,
                        // delegate remote URLs to the resolver.
                        let (bytes, format) = match parse_image_data_uri(image_url)? {
                            Some(d) => (d.bytes, d.format),
                            None => resolver.fetch(image_url).await?,
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
                    ResponseContentPart::InputFile { .. } => {
                        return Err(AppError::BadRequest(
                            "input_file content parts are not supported".to_string(),
                        ));
                    }
                }
            }
            Ok(blocks)
        }
    }
}

/// Reframe per-item turns into Converse turns, merging contiguous same-role
/// turns and splitting tool-only content from normal content.
///
/// Uses the SAME split rule as the chat path
/// ([`crate::bedrock::tools::should_split_same_role_merge`]) so tool-only turns
/// merge with each other but split from text turns.
fn reframe_turns(turns: Vec<Turn>) -> Value {
    let mut reformatted: Vec<(String, Vec<Value>)> = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_content: Vec<Value> = Vec::new();

    for turn in turns {
        let next_role = turn.role;
        let next_content = turn.content;

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

    Value::Array(
        reformatted
            .into_iter()
            .map(|(role, content)| json!({ "role": role, "content": content }))
            .collect(),
    )
}

/// Reject any built-in server tool in `tools[]` (a 400).
///
/// [`ResponsesTool`] only models the `function` variant, so a built-in tool type
/// (`web_search`, `file_search`, `code_interpreter`, `mcp`, `computer`,
/// `image_generation`, …) fails to deserialize at the wire boundary. This guard
/// additionally rejects any tool-shaped entry captured in `extra` (defensive —
/// in case a future schema change loosens deserialization). The modeled
/// `function` tools are always accepted.
fn reject_builtin_tools(req: &ResponsesRequest) -> Result<(), AppError> {
    // Modeled tools are all `function` by construction; nothing to reject.
    if let Some(tools) = &req.tools {
        for tool in tools {
            match tool {
                ResponsesTool::Function { .. } => {}
            }
        }
    }

    // Defensive: a built-in tool array smuggled through `extra["tools"]`.
    if let Some(Value::Array(extra_tools)) = req.extra.get("tools") {
        for tool in extra_tools {
            if let Some(ty) = tool.get("type").and_then(Value::as_str) {
                if ty != "function" {
                    return Err(AppError::BadRequest(format!(
                        "built-in server tool '{ty}' is not supported; only function tools are accepted"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Reject an unsatisfiable `text.format` (a 400).
///
/// A well-formed structured-output format that can pass through to Bedrock is
/// accepted; only a malformed / unsupported `text.format` is rejected. The rule:
/// - `text.format` absent → OK.
/// - `format` is not a JSON object → malformed → 400.
/// - `format.type == "text"` → OK (plain text; nothing to honor).
/// - `format.type == "json_object"` → OK (free-form JSON; no schema to satisfy).
/// - `format.type == "json_schema"` → OK only if it carries a non-null `schema`
///   object we can pass through; a missing/empty/non-object schema is
///   unsatisfiable → 400.
/// - any other / missing `type` → unsupported → 400.
fn reject_unsatisfiable_text_format(req: &ResponsesRequest) -> Result<(), AppError> {
    let Some(text) = &req.text else {
        return Ok(());
    };
    let Some(format) = &text.format else {
        return Ok(());
    };

    let obj = format
        .as_object()
        .ok_or_else(|| AppError::BadRequest("text.format must be an object".to_string()))?;

    let ty = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
        AppError::BadRequest("text.format.type is missing or not a string".to_string())
    })?;

    match ty {
        "text" | "json_object" => Ok(()),
        "json_schema" => {
            // A satisfiable json_schema must carry a non-empty schema object.
            // The schema may live directly under `schema` or nested under a
            // `json_schema` wrapper, mirroring the two OpenAI shapes.
            let schema = obj
                .get("schema")
                .or_else(|| obj.get("json_schema").and_then(|w| w.get("schema")));
            match schema {
                Some(Value::Object(map)) if !map.is_empty() => Ok(()),
                _ => Err(AppError::BadRequest(
                    "text.format json_schema is malformed: missing or empty 'schema' object"
                        .to_string(),
                )),
            }
        }
        other => Err(AppError::BadRequest(format!(
            "unsupported text.format.type '{other}'"
        ))),
    }
}

/// Map the request-level `reasoning { effort }` to a Bedrock thinking budget,
/// reusing the chat reasoning mapping verbatim.
///
/// This is the request-level reasoning config (distinct from the dropped
/// reasoning INPUT items). It is exposed for the provider seam (T13) so the
/// Responses path drives the Bedrock thinking budget identically to chat. When
/// `reasoning.effort` is absent, reasoning injection is skipped entirely
/// (empty outcome) — matching the chat path's `Some/None` branch.
///
/// `max_output_tokens` plays the role of the chat `max_completion_tokens`
/// (Responses has no separate `max_tokens`).
#[must_use]
pub fn reasoning_outcome(
    req: &ResponsesRequest,
    model_id: &str,
    caps: &dyn ModelCapabilities,
) -> crate::bedrock::reasoning::ReasoningOutcome {
    // Mirror chat (provider.rs): a missing `reasoning.effort` must SKIP
    // injection, not default to `None`. Passing `None` into adaptive_thinking
    // emits `output_config.effort = "none"`, which Bedrock rejects with 400
    // (valid: low/medium/high/xhigh/max). Do not collapse this back to
    // `unwrap_or(None)` + unconditional build.
    match req.reasoning.as_ref().and_then(|r| r.effort.as_deref()) {
        Some(effort_str) => crate::bedrock::reasoning::build_reasoning_config(
            model_id,
            parse_effort(effort_str),
            req.max_output_tokens,
            req.max_output_tokens,
            caps,
        ),
        None => crate::bedrock::reasoning::ReasoningOutcome::default(),
    }
}

/// Parse an effort string into a [`ReasoningEffort`], defaulting unknown values
/// to [`ReasoningEffort::Medium`] (lenient — never a 400 on effort).
fn parse_effort(s: &str) -> ReasoningEffort {
    match s {
        "none" => ReasoningEffort::None,
        "minimal" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" => ReasoningEffort::Xhigh,
        "max" => ReasoningEffort::Max,
        _ => ReasoningEffort::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bedrock::capabilities::ConfigModelCapabilities;
    use crate::config::ModelCapabilityConfig;
    use serde_json::json;

    const MODELS_TOML: &str = "config/models.toml";

    fn caps() -> ConfigModelCapabilities {
        let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load models.toml");
        ConfigModelCapabilities::new(config)
    }

    /// A test resolver that never hits the network; `supports_image` is a flag.
    struct TestResolver {
        image_ok: bool,
    }

    #[async_trait::async_trait]
    impl ImageResolver for TestResolver {
        fn supports_image(&self, _model_id: &str) -> bool {
            self.image_ok
        }
        async fn fetch(&self, _url: &str) -> Result<(Vec<u8>, String), AppError> {
            Ok((vec![1, 2, 3], "jpeg".to_string()))
        }
    }

    fn resolver(image_ok: bool) -> TestResolver {
        TestResolver { image_ok }
    }

    /// Build a request from a JSON value (exercises the real serde boundary).
    fn req_from(value: Value) -> ResponsesRequest {
        serde_json::from_value(value).expect("deserialize ResponsesRequest")
    }

    async fn parse(req: &ResponsesRequest) -> Result<ResponsesConverseInput, AppError> {
        let c = caps();
        let r = resolver(true);
        to_responses_converse_input(req, "anthropic.claude-3-sonnet-v1:0", &r, &c).await
    }

    // -- Test 1: string input → one user message; item array incl toolResult --

    #[tokio::test]
    async fn string_input_becomes_single_user_message() {
        let req = req_from(json!({ "model": "m", "input": "Hello, world" }));
        let out = parse(&req).await.expect("parse");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello, world");
        // No instructions → empty system.
        assert_eq!(out.system.as_array().expect("system").len(), 0);
    }

    #[tokio::test]
    async fn item_array_with_function_call_output_yields_tool_result() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "run the tool" },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}" },
                { "type": "function_call_output", "call_id": "c1", "output": "42" }
            ]
        }));
        let out = parse(&req).await.expect("parse");
        let msgs = out.messages.as_array().expect("messages");
        // user turn, assistant toolUse turn, user toolResult turn.
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "run the tool");

        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["toolUse"]["toolUseId"], "c1");
        assert_eq!(msgs[1]["content"][0]["toolUse"]["name"], "f");
        assert_eq!(msgs[1]["content"][0]["toolUse"]["input"]["x"], 1);

        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["toolResult"]["toolUseId"], "c1");
        assert_eq!(
            msgs[2]["content"][0]["toolResult"]["content"][0]["text"],
            "42"
        );
    }

    #[tokio::test]
    async fn structured_function_call_output_yields_ordered_tool_result_content() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "function_call", "call_id": "c1", "name": "screenshot", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": [
                    { "type": "input_text", "text": "Image read successfully" },
                    { "type": "input_image", "image_url": "data:image/png;base64,AAECAw==" }
                ]}
            ]
        }));
        let out = parse(&req).await.expect("parse structured tool output");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 2);
        let tool_result = &msgs[1]["content"][0]["toolResult"];
        assert_eq!(tool_result["toolUseId"], "c1");
        assert_eq!(tool_result["content"][0]["text"], "Image read successfully");
        assert_eq!(tool_result["content"][1]["image"]["format"], "png");
    }

    #[tokio::test]
    async fn item_reference_is_accepted_and_ignored_for_stateless_gateway() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "item_reference", "id": "rs_stored_1" },
                { "role": "user", "content": "continue" }
            ]
        }));
        let out = parse(&req).await.expect("item_reference must not reject");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"][0]["text"], "continue");
    }

    // -- Test 2: instructions present → a Bedrock system block ----------------

    #[tokio::test]
    async fn instructions_become_prepended_system_block() {
        let req = req_from(json!({
            "model": "m",
            "instructions": "You are terse.",
            "input": [
                { "type": "message", "role": "system", "content": "Extra context." },
                { "type": "message", "role": "user", "content": "hi" }
            ]
        }));
        let out = parse(&req).await.expect("parse");
        let sys = out.system.as_array().expect("system");
        // instructions FIRST, then the system message item.
        assert_eq!(sys.len(), 2);
        assert_eq!(sys[0]["text"], "You are terse.");
        assert_eq!(sys[1]["text"], "Extra context.");
        // The user message is the only turn; system items are not turns.
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[tokio::test]
    async fn developer_role_maps_to_system_block() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "role": "developer", "content": "be helpful" },
                { "role": "user", "content": "hi" }
            ]
        }));
        let out = parse(&req).await.expect("parse");
        let sys = out.system.as_array().expect("system");
        assert_eq!(sys.len(), 1);
        assert_eq!(sys[0]["text"], "be helpful");
    }

    // -- Test 3: store + previous_response_id + include → accepted & IGNORED --

    #[tokio::test]
    async fn store_previous_id_and_include_are_accepted_and_ignored() {
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "store": true,
            "previous_response_id": "resp_prev_123",
            "include": ["reasoning.encrypted_content"]
        }));
        // Parse must SUCCEED (no 400) and emit nothing extra.
        let out = parse(&req)
            .await
            .expect("store/prev_id/include must be ignored, not rejected");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"][0]["text"], "hi");
        // No system blocks were synthesized from include/encrypted_content.
        assert_eq!(out.system.as_array().expect("system").len(), 0);
    }

    #[tokio::test]
    async fn store_false_is_accepted() {
        // codex sends store:false — must NOT 400.
        let req = req_from(json!({ "model": "m", "input": "hi", "store": false }));
        assert!(parse(&req).await.is_ok());
    }

    // -- Test 4: built-in tool → 400; malformed text.format → 400 -------------

    #[tokio::test]
    async fn builtin_server_tool_is_bad_request() {
        // A web_search tool type fails to deserialize into ResponsesTool, which
        // surfaces as a 400 at the wire boundary. Verify the request does not
        // deserialize into a function tool.
        let raw = json!({
            "model": "m",
            "input": "hi",
            "tools": [{ "type": "web_search" }]
        });
        let parsed: Result<ResponsesRequest, _> = serde_json::from_value(raw);
        assert!(
            parsed.is_err(),
            "built-in server tool type must not deserialize as a function tool"
        );

        // And the defensive guard rejects a built-in tool smuggled via `extra`.
        let mut req = req_from(json!({ "model": "m", "input": "hi" }));
        req.extra
            .insert("tools".to_string(), json!([{ "type": "code_interpreter" }]));
        let err = parse(&req)
            .await
            .expect_err("built-in tool must be rejected");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn function_tool_is_accepted() {
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "tools": [{ "type": "function", "name": "f", "parameters": {"type": "object"} }]
        }));
        assert!(parse(&req).await.is_ok());
    }

    #[tokio::test]
    async fn malformed_text_format_is_bad_request() {
        // json_schema with no schema object → unsatisfiable → 400.
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "text": { "format": { "type": "json_schema" } }
        }));
        let err = parse(&req)
            .await
            .expect_err("malformed text.format must 400");
        assert!(matches!(err, AppError::BadRequest(_)));

        // Non-object format → 400.
        let req2 = req_from(json!({
            "model": "m",
            "input": "hi",
            "text": { "format": "not an object" }
        }));
        assert!(matches!(
            parse(&req2).await.expect_err("non-object format must 400"),
            AppError::BadRequest(_)
        ));

        // Unknown type → 400.
        let req3 = req_from(json!({
            "model": "m",
            "input": "hi",
            "text": { "format": { "type": "weird" } }
        }));
        assert!(matches!(
            parse(&req3)
                .await
                .expect_err("unknown format type must 400"),
            AppError::BadRequest(_)
        ));
    }

    #[tokio::test]
    async fn wellformed_text_format_is_accepted() {
        // A well-formed json_schema passes through.
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "text": { "format": {
                "type": "json_schema",
                "name": "out",
                "schema": { "type": "object", "properties": {} }
            } }
        }));
        assert!(parse(&req).await.is_ok());

        // Plain text / json_object also accepted.
        let req_text = req_from(json!({
            "model": "m", "input": "hi", "text": { "format": { "type": "text" } }
        }));
        assert!(parse(&req_text).await.is_ok());
        let req_obj = req_from(json!({
            "model": "m", "input": "hi", "text": { "format": { "type": "json_object" } }
        }));
        assert!(parse(&req_obj).await.is_ok());
    }

    // -- Test 5: reasoning input item is DROPPED ------------------------------

    #[tokio::test]
    async fn reasoning_input_item_is_dropped() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "hi" },
                { "type": "reasoning", "id": "r1", "encrypted_content": "OPAQUE_BLOB", "summary": ["s"] }
            ]
        }));
        let out = parse(&req)
            .await
            .expect("parse succeeds despite reasoning item");
        let msgs = out.messages.as_array().expect("messages");
        // Only the user turn survives; the reasoning item is dropped entirely.
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "hi");
        // No Bedrock reasoning/thinking block was emitted anywhere.
        let serialized = serde_json::to_string(&out.messages).unwrap();
        assert!(!serialized.contains("reasoning"));
        assert!(!serialized.contains("thinking"));
        assert!(!serialized.contains("OPAQUE_BLOB"));
    }

    // -- Multimodal image reuse + same-role merge ----------------------------

    #[tokio::test]
    async fn input_image_part_decodes_via_shared_handling() {
        // "hi" base64 = "aGk=".
        let req = req_from(json!({
            "model": "m",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "look" },
                    { "type": "input_image", "image_url": "data:image/png;base64,aGk=" }
                ]
            }]
        }));
        let out = parse(&req).await.expect("parse");
        let content = out.messages[0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "look");
        assert_eq!(content[1]["image"]["format"], "png");
        let bytes = content[1]["image"]["source"]["bytes"]
            .as_array()
            .expect("bytes");
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 104);
    }

    #[tokio::test]
    async fn image_on_non_image_model_is_bad_request() {
        let req = req_from(json!({
            "model": "m",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_image", "image_url": "data:image/png;base64,aGk=" }
                ]
            }]
        }));
        let c = caps();
        let r = resolver(false); // no IMAGE modality
        let err = to_responses_converse_input(&req, "m", &r, &c)
            .await
            .expect_err("image on non-image model must 400");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[tokio::test]
    async fn contiguous_user_turns_merge() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "role": "user", "content": "Hello" },
                { "role": "user", "content": "Who are you?" }
            ]
        }));
        let out = parse(&req).await.expect("parse");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 1);
        let content = msgs[0]["content"].as_array().expect("content");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["text"], "Hello");
        assert_eq!(content[1]["text"], "Who are you?");
    }

    #[tokio::test]
    async fn assistant_message_with_output_text_becomes_assistant_turn() {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "first" },
                { "type": "message", "role": "assistant", "content": [
                    { "type": "output_text", "text": "prior reply" }
                ]}
            ]
        }));
        let out = parse(&req).await.expect("parse");
        let msgs = out.messages.as_array().expect("messages");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["text"], "first");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["text"], "prior reply");
    }

    #[tokio::test]
    async fn codex_multiturn_replay_shape_parses_with_assistant_turn() {
        let req = req_from(json!({
            "model": "qwen.qwen3-235b-a22b-2507-v1:0",
            "instructions": "You are a coding agent.",
            "input": [
                { "type": "message", "role": "developer", "content": [
                    { "type": "input_text", "text": "permissions" }
                ]},
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "agents instructions" }
                ]},
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "environment_context" }
                ]},
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "Run the shell command: echo T15TOOLOK" }
                ]},
                { "type": "message", "role": "assistant", "content": [
                    { "type": "output_text", "text": "I'm about to run the shell command." }
                ]},
                { "type": "function_call", "name": "exec_command",
                  "arguments": "{\"cmd\": \"echo T15TOOLOK\"}",
                  "call_id": "tooluse_LwpzalXzdnMiABFHMBTqYF" },
                { "type": "function_call_output",
                  "call_id": "tooluse_LwpzalXzdnMiABFHMBTqYF",
                  "output": "T15TOOLOK\n" }
            ],
            "store": false,
            "stream": true
        }));
        let out = parse(&req)
            .await
            .expect("codex multi-turn shape must parse");

        let sys = out.system.as_array().expect("system");
        assert_eq!(sys[0]["text"], "You are a coding agent.");
        assert_eq!(sys[1]["text"], "permissions");

        let msgs = out.messages.as_array().expect("messages");
        let assistant_turns: Vec<&Value> =
            msgs.iter().filter(|m| m["role"] == "assistant").collect();
        let echoed = assistant_turns
            .iter()
            .any(|m| m["content"][0]["text"] == "I'm about to run the shell command.");
        assert!(
            echoed,
            "expected a Bedrock assistant turn carrying the echoed output_text"
        );
        let has_tool_use = assistant_turns
            .iter()
            .any(|m| m["content"][0].get("toolUse").is_some());
        assert!(has_tool_use, "expected the function_call assistant turn");
    }

    #[tokio::test]
    async fn input_file_part_is_bad_request() {
        let req = req_from(json!({
            "model": "m",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_file", "file_id": "f1" }
                ]
            }]
        }));
        let err = parse(&req).await.expect_err("input_file unsupported");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    // -- reasoning_outcome (request-level reasoning, reused from chat) ---------

    #[tokio::test]
    async fn reasoning_outcome_none_for_plain_model() {
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "reasoning": { "effort": "high" }
        }));
        let c = caps();
        // A non-reasoning model yields an empty outcome regardless of effort.
        let outcome = reasoning_outcome(&req, "anthropic.claude-3-sonnet-v1:0", &c);
        assert!(outcome.is_empty());
    }

    #[tokio::test]
    async fn reasoning_outcome_skips_injection_when_effort_absent_adaptive() {
        let req = req_from(json!({ "model": "m", "input": "hi" }));
        let c = caps();
        let outcome = reasoning_outcome(&req, "claude-fable-5", &c);
        assert!(outcome.is_empty());
        assert!(!outcome
            .additional_model_request_fields
            .contains_key("output_config"));
    }

    #[tokio::test]
    async fn reasoning_outcome_injects_output_config_when_effort_high_adaptive() {
        let req = req_from(json!({
            "model": "m",
            "input": "hi",
            "reasoning": { "effort": "high" }
        }));
        let c = caps();
        let outcome = reasoning_outcome(&req, "claude-fable-5", &c);
        assert!(!outcome.is_empty());
        assert_eq!(
            outcome.additional_model_request_fields["output_config"]["effort"],
            "high"
        );
    }

    #[tokio::test]
    async fn reasoning_outcome_aligns_with_chat_on_missing_effort() {
        use crate::bedrock::reasoning::{build_reasoning_config, ReasoningOutcome};
        let req = req_from(json!({ "model": "m", "input": "hi" }));
        let c = caps();
        let responses_outcome = reasoning_outcome(&req, "claude-fable-5", &c);
        // Chat path on `None` effort returns `ReasoningOutcome::default()` and
        // never calls `build_reasoning_config`; both surfaces must match.
        assert_eq!(responses_outcome, ReasoningOutcome::default());
        let unconditional =
            build_reasoning_config("claude-fable-5", ReasoningEffort::None, None, None, &c);
        assert_ne!(responses_outcome, unconditional);
    }

    #[test]
    fn parse_effort_covers_known_and_unknown() {
        assert_eq!(parse_effort("none"), ReasoningEffort::None);
        assert_eq!(parse_effort("low"), ReasoningEffort::Low);
        assert_eq!(parse_effort("max"), ReasoningEffort::Max);
        // Unknown defaults to Medium (lenient, never an error).
        assert_eq!(parse_effort("bogus"), ReasoningEffort::Medium);
    }

    #[tokio::test]
    async fn extra_map_is_not_a_tools_smuggle_false_positive() {
        // A function tool in `extra["tools"]` (not built-in) must NOT be rejected.
        let mut req = req_from(json!({ "model": "m", "input": "hi" }));
        req.extra
            .insert("tools".to_string(), json!([{ "type": "function" }]));
        assert!(parse(&req).await.is_ok());
    }
}
