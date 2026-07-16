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
//! - `reasoning` items          → Bedrock `reasoningContent`, decoded from the
//!   gateway's opaque `encrypted_content` envelope with text/signature intact.
//!
//! Contiguous same-role turns are merged / split with the SAME rule the chat
//! path uses ([`crate::bedrock::tools::should_split_same_role_merge`]): tool-only
//! turns merge with each other but split from normal content.
//!
//! ## Request-level reasoning vs reasoning INPUT items
//!
//! The request-level `reasoning { effort }` field is separate from replayed
//! reasoning input items: it drives the Bedrock thinking budget via
//! [`crate::bedrock::reasoning::build_reasoning_config`], identically to chat.
//! [`reasoning_outcome`] exposes that mapping for the provider seam.
//!
//! ## Stateless compatibility matrix (codex-safety-critical)
//!
//! - `store` (any value)            → **accept & IGNORE** (codex sends
//!   `store:false`; a 400 would break codex).
//! - `previous_response_id` / `item_reference` → **accept & IGNORE** because
//!   this backend is stateless. Clients must replay full history for context.
//! - `include: ["reasoning.encrypted_content"]` is accepted; signed reasoning
//!   output carries the replay envelope.
//! - tools in `tools[]` → function/custom/namespace and client-executed shell /
//!   apply-patch tools are translated reversibly. OpenAI-hosted and unknown
//!   tools with no Converse equivalent are silently omitted so bundled tools
//!   cannot abort otherwise valid client-executed tool calls.
//! - unsatisfiable `text.format` (a malformed / unsupported structured-output
//!   schema) → **400**. A well-formed `json_schema` that can pass through to
//!   Bedrock is accepted.

use std::collections::HashSet;

use base64::Engine as _;
use serde_json::{json, Value};

use crate::bedrock::tools::{
    convert_tool_spec, should_split_same_role_merge, tool_message_to_tool_result_turn,
};
use crate::bedrock::translate::{parse_image_data_uri, ImageResolver};
use crate::domain::ModelCapabilities;
use crate::error::AppError;
use crate::openai::responses_schema::{
    FunctionCallOutputValue, ResponseContentPart, ResponseInputItem, ResponsesContent,
    ResponsesInput, ResponsesNamespaceInner, ResponsesRequest, ResponsesRole, ResponsesTool,
};
use crate::openai::schema::{Function, ReasoningEffort};

/// Delimiter joining a namespace name to an inner tool name when a `namespace`
/// tool is flattened into the Bedrock `toolConfig` (`{ns}__{fn}`). A
/// protocol-shaping constant, not model knowledge: it keeps tools from
/// different namespaces from colliding and is echoed back verbatim by the
/// client on the stateless round-trip.
pub const NAMESPACE_DELIMITER: &str = "__";
const REASONING_ENVELOPE_PREFIX: &str = "bedrock-reasoning-v1:";

/// Encode Bedrock reasoning blocks into the standard opaque Responses
/// `encrypted_content` slot. The provider-issued signature inside the payload
/// remains the authority: Bedrock rejects any altered text/signature pair when
/// the client replays it.
pub(crate) fn encode_reasoning_envelope(content: &[Value]) -> Option<String> {
    let blocks: Vec<Value> = content
        .iter()
        .filter_map(|block| {
            let reasoning = block.get("reasoningContent")?;
            let signed_text = reasoning
                .get("reasoningText")
                .and_then(|text| text.get("signature"))
                .and_then(Value::as_str)
                .is_some();
            let redacted = reasoning
                .get("redactedContent")
                .and_then(Value::as_str)
                .is_some();
            (signed_text || redacted).then(|| reasoning.clone())
        })
        .collect();
    if blocks.is_empty() {
        return None;
    }
    let bytes = serde_json::to_vec(&json!({ "version": 1, "blocks": blocks })).ok()?;
    Some(format!(
        "{REASONING_ENVELOPE_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    ))
}

fn decode_reasoning_envelope(envelope: &str) -> Result<Vec<Value>, AppError> {
    let encoded = envelope
        .strip_prefix(REASONING_ENVELOPE_PREFIX)
        .ok_or_else(|| {
            AppError::BadRequest("unsupported reasoning.encrypted_content envelope".to_string())
        })?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| AppError::BadRequest("malformed reasoning.encrypted_content".to_string()))?;
    let payload: Value = serde_json::from_slice(&bytes)
        .map_err(|_| AppError::BadRequest("malformed reasoning.encrypted_content".to_string()))?;
    if payload.get("version").and_then(Value::as_u64) != Some(1) {
        return Err(AppError::BadRequest(
            "unsupported reasoning.encrypted_content envelope version".to_string(),
        ));
    }
    let blocks = payload
        .get("blocks")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::BadRequest("reasoning envelope has no blocks".to_string()))?;
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        let valid_text = block
            .get("reasoningText")
            .and_then(Value::as_object)
            .is_some_and(|text| {
                text.get("text").and_then(Value::as_str).is_some()
                    && text.get("signature").and_then(Value::as_str).is_some()
            });
        let valid_redacted = block
            .get("redactedContent")
            .and_then(Value::as_str)
            .is_some();
        if !valid_text && !valid_redacted {
            return Err(AppError::BadRequest(
                "reasoning envelope contains an invalid block".to_string(),
            ));
        }
        out.push(json!({ "reasoningContent": block }));
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResponsesToolKind {
    Function,
    Custom,
    LocalShell,
    Shell,
    ApplyPatch,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesToolBinding {
    pub bedrock_name: String,
    pub client_name: String,
    pub namespace: Option<String>,
    pub kind: ResponsesToolKind,
}

/// Per-request reversible mapping between Bedrock's function-only tool model
/// and the richer Responses tool/item kinds.
#[derive(Debug, Clone, Default)]
pub(crate) struct ResponsesToolRegistry {
    bindings: Vec<ResponsesToolBinding>,
}

impl ResponsesToolRegistry {
    pub fn resolve(&self, bedrock_name: &str) -> Option<&ResponsesToolBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.bedrock_name == bedrock_name)
    }

    pub fn bedrock_name_for(&self, client_name: &str) -> Option<&str> {
        self.bindings
            .iter()
            .find(|binding| binding.client_name == client_name)
            .map(|binding| binding.bedrock_name.as_str())
    }
}

/// Build Bedrock tool specifications plus the reversible response mapping.
pub(crate) fn build_responses_tools(
    req: &ResponsesRequest,
) -> Result<(Vec<Value>, ResponsesToolRegistry), AppError> {
    let Some(tools) = &req.tools else {
        return Ok((Vec::new(), ResponsesToolRegistry::default()));
    };
    let mut specs = Vec::new();
    let mut bindings = Vec::new();
    let mut names = HashSet::new();

    let mut push = |bedrock_name: String,
                    client_name: String,
                    namespace: Option<String>,
                    kind: ResponsesToolKind,
                    description: Option<&str>,
                    parameters: Option<Value>|
     -> Result<(), AppError> {
        if !names.insert(bedrock_name.clone()) {
            return Err(AppError::BadRequest(format!(
                "duplicate Responses tool name after translation: '{bedrock_name}'"
            )));
        }
        specs.push(function_tool_spec(&bedrock_name, description, &parameters));
        bindings.push(ResponsesToolBinding {
            bedrock_name,
            client_name,
            namespace,
            kind,
        });
        Ok(())
    };

    for tool in tools {
        match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
                ..
            } => push(
                name.clone(),
                name.clone(),
                None,
                ResponsesToolKind::Function,
                description.as_deref(),
                parameters.clone(),
            )?,
            ResponsesTool::Custom {
                name, description, ..
            } => push(
                name.clone(),
                name.clone(),
                None,
                ResponsesToolKind::Custom,
                description.as_deref(),
                Some(custom_input_schema()),
            )?,
            ResponsesTool::Namespace {
                name: namespace,
                tools: inner,
                ..
            } => {
                for item in inner {
                    match item {
                        ResponsesNamespaceInner::Function {
                            name,
                            description,
                            parameters,
                            ..
                        } => push(
                            format!("{namespace}{NAMESPACE_DELIMITER}{name}"),
                            name.clone(),
                            Some(namespace.clone()),
                            ResponsesToolKind::Function,
                            description.as_deref(),
                            parameters.clone(),
                        )?,
                        ResponsesNamespaceInner::Custom {
                            name, description, ..
                        } => push(
                            format!("{namespace}{NAMESPACE_DELIMITER}{name}"),
                            name.clone(),
                            Some(namespace.clone()),
                            ResponsesToolKind::Custom,
                            description.as_deref(),
                            Some(custom_input_schema()),
                        )?,
                    }
                }
            }
            ResponsesTool::LocalShell { .. } => push(
                "local_shell".to_string(),
                "local_shell".to_string(),
                None,
                ResponsesToolKind::LocalShell,
                Some("Run a command in the client's local shell."),
                Some(json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",
                            "properties": {
                                "type": { "type": "string", "enum": ["exec"] },
                                "command": { "type": "array", "items": { "type": "string" } },
                                "timeout_ms": { "type": "number" },
                                "user": { "type": "string" },
                                "working_directory": { "type": "string" },
                                "env": { "type": "object", "additionalProperties": { "type": "string" } }
                            },
                            "required": ["type", "command"]
                        }
                    },
                    "required": ["action"]
                })),
            )?,
            ResponsesTool::Shell { .. } => push(
                "shell".to_string(),
                "shell".to_string(),
                None,
                ResponsesToolKind::Shell,
                Some("Run one or more shell commands in the client environment."),
                Some(json!({
                    "type": "object",
                    "properties": {
                        "commands": { "type": "array", "items": { "type": "string" } },
                        "timeout_ms": { "type": "number" },
                        "max_output_length": { "type": "number" }
                    },
                    "required": ["commands"],
                    "additionalProperties": false
                })),
            )?,
            ResponsesTool::ApplyPatch { .. } => push(
                "apply_patch".to_string(),
                "apply_patch".to_string(),
                None,
                ResponsesToolKind::ApplyPatch,
                Some("Apply a source-code patch in the client workspace."),
                Some(json!({
                    "type": "object",
                    "properties": {
                        "operation": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": { "type": "string", "enum": ["create_file"] },
                                        "path": { "type": "string" },
                                        "diff": { "type": "string" }
                                    },
                                    "required": ["type", "path", "diff"],
                                    "additionalProperties": false
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": { "type": "string", "enum": ["delete_file"] },
                                        "path": { "type": "string" }
                                    },
                                    "required": ["type", "path"],
                                    "additionalProperties": false
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "type": { "type": "string", "enum": ["update_file"] },
                                        "path": { "type": "string" },
                                        "diff": { "type": "string" }
                                    },
                                    "required": ["type", "path", "diff"],
                                    "additionalProperties": false
                                }
                            ]
                        }
                    },
                    "required": ["operation"],
                    "additionalProperties": false
                })),
            )?,
            ResponsesTool::WebSearch { .. }
            | ResponsesTool::WebSearchPreview { .. }
            | ResponsesTool::FileSearch { .. }
            | ResponsesTool::CodeInterpreter { .. }
            | ResponsesTool::ToolSearch { .. }
            | ResponsesTool::Mcp { .. }
            | ResponsesTool::Computer { .. }
            | ResponsesTool::ImageGeneration { .. }
            | ResponsesTool::Unknown => {}
        }
    }

    Ok((specs, ResponsesToolRegistry { bindings }))
}

fn custom_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "input": { "type": "string" } },
        "required": ["input"],
        "additionalProperties": false
    })
}

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

fn push_non_empty_text_block(blocks: &mut Vec<Value>, text: &str) {
    if !text.is_empty() {
        blocks.push(json!({ "text": text }));
    }
}

/// Translate a [`ResponsesRequest`] into Bedrock Converse `messages` + `system`.
///
/// Runs request validation first, then parses `input` (string or item array) into
/// Bedrock turns, prepends `instructions` to the system blocks, and merges
/// contiguous same-role turns with the shared chat reframing rule.
///
/// `model_id` is the resolved model id used for the image-modality gate and for
/// any downstream Bedrock call; `resolver` decodes/fetches images; `caps` is
/// accepted for parity with the chat translate signature (reserved for future
/// capability-driven branching — currently unused in the input parse itself).
///
/// # Errors
/// Returns [`AppError::BadRequest`] for an unsatisfiable `text.format`, an
/// `input_file` content part, or image-modality/decode failures; propagates
/// resolver errors.
pub async fn to_responses_converse_input(
    req: &ResponsesRequest,
    model_id: &str,
    resolver: &dyn ImageResolver,
    caps: &dyn ModelCapabilities,
) -> Result<ResponsesConverseInput, AppError> {
    // `caps` is part of the parity signature; the input parse does not branch on
    // capabilities today (kept for future use — e.g. per-model file handling).
    let _ = caps;

    // 1) Request validation. Tool availability is checked when the provider
    //    builds the reversible tool registry.
    reject_unsatisfiable_text_format(req)?;

    // 2) System blocks: `instructions` first (prepended), then any system /
    //    developer message-item blocks, in input order.
    let mut system_blocks: Vec<Value> = Vec::new();
    if let Some(instructions) = &req.instructions {
        push_non_empty_text_block(&mut system_blocks, instructions);
    }

    // 3) Parse input into per-item turns (system/developer items contribute to
    //    `system_blocks` instead of producing a turn).
    let turns = match &req.input {
        ResponsesInput::Text(text) => {
            if text.is_empty() {
                return Err(AppError::BadRequest(
                    "input must contain non-empty text".to_string(),
                ));
            }
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
                    if blocks.is_empty() {
                        return Err(AppError::BadRequest(
                            "message content must contain at least one non-empty content block"
                                .to_string(),
                        ));
                    }
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
                    if blocks.is_empty() {
                        continue;
                    }
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
            //
            // Namespace names remain client-visible as `(namespace, name)` and
            // are joined only for Bedrock's flat tool namespace.
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
                namespace,
            } => {
                let input: Value = serde_json::from_str(arguments).map_err(|e| {
                    AppError::BadRequest(format!("invalid function_call arguments JSON: {e}"))
                })?;
                let bedrock_name = namespace.as_ref().map_or_else(
                    || name.clone(),
                    |ns| format!("{ns}{NAMESPACE_DELIMITER}{name}"),
                );
                turns.push(Turn {
                    role: "assistant".to_string(),
                    content: vec![json!({
                        "toolUse": {
                            "toolUseId": call_id,
                            "name": bedrock_name,
                            "input": input,
                        }
                    })],
                });
            }
            // function_call_output → user toolResult turn (reusing tools.rs).
            // `call_id` is passed through UNCHANGED — same round-trip invariant
            // as function_call above: it correlates to the (possibly prefixed)
            // toolUseId the client already received, so it must echo back
            // verbatim. Do not rewrite it.
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
            ResponseInputItem::ItemReference { .. } => {}
            ResponseInputItem::Other { item_type, fields } => match item_type.as_str() {
                "custom_tool_call" | "local_shell_call" | "shell_call" | "apply_patch_call" => {
                    let call_id =
                        fields
                            .get("call_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                AppError::BadRequest(format!(
                                    "{item_type} input item is missing call_id"
                                ))
                            })?;
                    let name = match item_type.as_str() {
                        "custom_tool_call" => {
                            fields.get("name").and_then(Value::as_str).ok_or_else(|| {
                                AppError::BadRequest(
                                    "custom_tool_call input item is missing name".to_string(),
                                )
                            })?
                        }
                        "local_shell_call" => "local_shell",
                        "shell_call" => "shell",
                        "apply_patch_call" => "apply_patch",
                        _ => unreachable!(),
                    };
                    let input = match item_type.as_str() {
                        "custom_tool_call" => json!({
                            "input": fields.get("input").cloned().unwrap_or(Value::String(String::new()))
                        }),
                        "local_shell_call" => json!({
                            "action": fields.get("action").cloned().unwrap_or_else(|| json!({}))
                        }),
                        "shell_call" => fields.get("action").cloned().unwrap_or_else(|| json!({})),
                        "apply_patch_call" => json!({
                            "operation": fields.get("operation").cloned().unwrap_or_else(|| json!({}))
                        }),
                        _ => unreachable!(),
                    };
                    turns.push(Turn {
                        role: "assistant".to_string(),
                        content: vec![json!({
                            "toolUse": { "toolUseId": call_id, "name": name, "input": input }
                        })],
                    });
                }
                "custom_tool_call_output"
                | "local_shell_call_output"
                | "shell_call_output"
                | "apply_patch_call_output" => {
                    let call_id =
                        fields
                            .get("call_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                AppError::BadRequest(format!(
                                    "{item_type} input item is missing call_id"
                                ))
                            })?;
                    let output = fields.get("output").cloned().unwrap_or(Value::Null);
                    turns.push(Turn {
                        role: "user".to_string(),
                        content: vec![json!({
                            "toolResult": {
                                "toolUseId": call_id,
                                "content": [{ "text": output.as_str().map_or_else(|| output.to_string(), str::to_string) }]
                            }
                        })],
                    });
                }
                _ => {
                    return Err(AppError::BadRequest(format!(
                        "Responses input item type '{item_type}' is not supported"
                    )));
                }
            },
            // Replay signed reasoning exactly; Bedrock validates the provider
            // signature against the unmodified text on the continuation call.
            ResponseInputItem::Reasoning {
                encrypted_content, ..
            } => {
                let envelope = encrypted_content.as_deref().ok_or_else(|| {
                    AppError::BadRequest(
                        "reasoning continuation requires reasoning.encrypted_content when store is false"
                            .to_string(),
                    )
                })?;
                turns.push(Turn {
                    role: "assistant".to_string(),
                    content: decode_reasoning_envelope(envelope)?,
                });
            }
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
            push_non_empty_text_block(system_blocks, text);
        }
        ResponsesContent::Parts(parts) => {
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text } => {
                        push_non_empty_text_block(system_blocks, text);
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
        ResponsesContent::Text(text) => {
            let mut blocks = Vec::new();
            push_non_empty_text_block(&mut blocks, text);
            Ok(blocks)
        }
        ResponsesContent::Parts(parts) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ResponseContentPart::InputText { text }
                    | ResponseContentPart::OutputText { text } => {
                        push_non_empty_text_block(&mut blocks, text);
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

/// Test helper that returns only representable tool specs. Runtime assembly
/// uses [`build_responses_tools`] and silently omits unsupported hosted tools.
///
/// - `function` → one `toolSpec` keeping its bare name.
/// - `custom`   → one `toolSpec` (name + description; the `format` grammar has
///   no Bedrock slot and is dropped).
/// - `namespace` → FLATTENED: one `toolSpec` per inner tool, with each inner
///   name prefixed `{namespace_name}__{inner_name}` (see [`NAMESPACE_DELIMITER`])
///   so tools from different namespaces never collide. A nested `custom` is
///   flattened the same way as a nested `function`.
///
/// Hosted tools are filtered here solely so translation-shape property tests
/// can inspect the supported subset.
#[cfg(test)]
#[must_use]
pub fn build_responses_tool_specs(req: &ResponsesRequest) -> Vec<Value> {
    let mut supported = req.clone();
    if let Some(tools) = supported.tools.as_mut() {
        tools.retain(|tool| {
            matches!(
                tool,
                ResponsesTool::Function { .. }
                    | ResponsesTool::Custom { .. }
                    | ResponsesTool::Namespace { .. }
                    | ResponsesTool::LocalShell { .. }
                    | ResponsesTool::Shell { .. }
                    | ResponsesTool::ApplyPatch { .. }
            )
        });
    }
    build_responses_tools(&supported)
        .map(|(specs, _)| specs)
        .unwrap_or_default()
}

/// Shape one Bedrock `toolSpec` from a (possibly prefixed) name + optional
/// description + optional JSON-schema parameters, reusing the chat path's
/// [`convert_tool_spec`] so the Bedrock toolSpec shaping is defined in exactly
/// one place. A missing `parameters` defaults to an empty object schema (the
/// same default the chat path applies for parameter-less tools).
fn function_tool_spec(name: &str, description: Option<&str>, parameters: &Option<Value>) -> Value {
    let func = Function {
        name: name.to_string(),
        description: description.map(str::to_string),
        parameters: parameters
            .clone()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
    };
    convert_tool_spec(&func)
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
#[path = "responses_translate_tests.rs"]
mod tests;
