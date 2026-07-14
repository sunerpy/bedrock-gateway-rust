//! Tool-calling normalization state machine (pure functions).
//!
//! Ports the tool-related portions of the Python request builder in
//! `.legacy-python/src/api/models/bedrock.py` into pure, testable functions.
//! Nothing here touches the network or the AWS SDK — every function maps
//! OpenAI tool concepts onto Bedrock Converse `toolConfig` / `toolUse` /
//! `toolResult` JSON and normalizes multi-turn tool histories.
//!
//! Ported functions (with provenance line ranges):
//! - `_convert_tool_spec`              (bedrock.py:1668-1677) → [`convert_tool_spec`]
//! - tool_choice mapping + llama skip  (bedrock.py:1190-1207) → [`build_tool_config`]
//! - placeholder toolConfig + safety net (bedrock.py:1208-1232, 1270-1294)
//!   → [`inject_placeholder_tool_config`]
//! - `_normalize_tool_result_turns`    (bedrock.py:1038-1104) → [`normalize_tool_result_turns`]
//! - same-role merge/split             (bedrock.py:1725-1756) → [`should_split_same_role_merge`]
//! - assistant tool_calls → toolUse    (bedrock.py:795-833)   → [`assistant_tool_calls_to_tool_use`]
//! - tool-role msg → toolResult turn   (bedrock.py:834-854)   → [`tool_message_to_tool_result_turn`]
//! - `_messages_contain_tool_content`  (bedrock.py:1697-1716) → [`messages_contain_tool_content`]
//!
//! ## De-hardcoding the llama skip
//!
//! The Python code special-cases `chat_request.model.startswith("meta.llama3-1-")`
//! when deciding whether to emit `toolChoice` (bedrock.py:1194). To keep this
//! module free of model-id literals, [`build_tool_config`] takes a
//! `skip_tool_choice: bool` parameter that the *caller* computes (e.g. via a
//! model capability). When `true`, `toolChoice` is omitted entirely.
//!
//! ## The dead validator
//!
//! `_validate_tool_use_result_sequence` (bedrock.py:1803-1838) raises 400s for
//! malformed toolUse/toolResult pairings, but it is **not** wired into the
//! normal request flow in the Python code. We deliberately do not implement an
//! active equivalent; [`normalize_tool_result_turns`] is the production path and
//! silently drops stale/duplicate results instead of erroring.

use serde_json::{json, Map, Value};

use crate::error::AppError;
use crate::openai::schema::{Function, Tool, ToolCall, ToolChoice};

/// The description used by the placeholder tool injected when a conversation
/// carries tool content but defines no tools (bedrock.py:1220, 1284).
const PLACEHOLDER_DESCRIPTION: &str =
    "Placeholder tool to satisfy Bedrock API requirements for multi-turn tool conversations";

/// Convert an OpenAI [`Function`] declaration into a Bedrock `toolSpec` block.
///
/// Ports `_convert_tool_spec` (bedrock.py:1668-1677):
/// ```text
/// {"toolSpec": {"name", "description", "inputSchema": {"json": <parameters>}}}
/// ```
/// `description` is emitted as JSON `null` when absent (mirroring Python passing
/// `func.description` through verbatim, which is `None` for tools without one).
#[must_use]
pub fn convert_tool_spec(func: &Function) -> Value {
    let description = match &func.description {
        Some(d) => Value::String(d.clone()),
        None => Value::Null,
    };
    json!({
        "toolSpec": {
            "name": func.name,
            "description": description,
            "inputSchema": {
                "json": func.parameters,
            },
        }
    })
}

/// Map an OpenAI `tool_choice` onto a Bedrock `toolChoice` value.
///
/// Ports the tool_choice branch of `_parse_request` (bedrock.py:1195-1206):
/// - `"required"` → `{"any": {}}`
/// - any other string (incl. `"auto"`) → `{"auto": {}}`
/// - an object → `{"tool": {"name": <function.name>}}`
///
/// # Errors
/// Returns [`AppError::BadRequest`] when an object `tool_choice` is missing its
/// `function` key (bedrock.py:1204-1205).
fn map_tool_choice(tool_choice: &ToolChoice) -> Result<Value, AppError> {
    match tool_choice {
        ToolChoice::String(s) => {
            if s == "required" {
                Ok(json!({ "any": {} }))
            } else {
                // "auto" (default) and any other string → auto.
                Ok(json!({ "auto": {} }))
            }
        }
        ToolChoice::Object(obj) => {
            let function = obj.get("function").ok_or_else(|| {
                AppError::BadRequest(
                    "tool_choice must contain 'function' key when specifying a specific tool"
                        .to_string(),
                )
            })?;
            // Python: chat_request.tool_choice["function"].get("name", "").
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(json!({ "tool": { "name": name } }))
        }
    }
}

/// Build the Bedrock `toolConfig` object from request tools + tool_choice.
///
/// Ports bedrock.py:1191-1207. Every tool becomes a `toolSpec` via
/// [`convert_tool_spec`]. `tool_choice` is mapped via [`map_tool_choice`] and
/// attached as `toolChoice` — unless `skip_tool_choice` is `true`, in which case
/// the `toolChoice` key is omitted (the de-hardcoded llama-skip; see the module
/// docs).
///
/// `tool_choice` is an [`Option`] so callers can distinguish "no tool_choice
/// supplied" (Python `if chat_request.tool_choice:` is falsy) from an explicit
/// one. When `None`, no `toolChoice` is emitted regardless of `skip_tool_choice`.
///
/// # Errors
/// Propagates the missing-`function` error from [`map_tool_choice`].
pub fn build_tool_config(
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
    skip_tool_choice: bool,
) -> Result<Value, AppError> {
    let specs: Vec<Value> = tools
        .iter()
        .map(|t| convert_tool_spec(&t.function))
        .collect();
    let mut config = Map::new();
    config.insert("tools".to_string(), Value::Array(specs));

    if let Some(choice) = tool_choice {
        if !skip_tool_choice {
            config.insert("toolChoice".to_string(), map_tool_choice(choice)?);
        }
    }

    Ok(Value::Object(config))
}

/// Convert an assistant message's `tool_calls` into Bedrock `toolUse` blocks.
///
/// Ports bedrock.py:812-824: each tool call becomes
/// `{"toolUse": {"toolUseId": id, "name": fn.name, "input": json.parse(args)}}`.
/// The OpenAI `function.arguments` field is a JSON *string* that is parsed into
/// the `input` object (bedrock.py:815).
///
/// # Errors
/// Returns [`AppError::BadRequest`] if a tool call's `arguments` is not valid
/// JSON.
pub fn assistant_tool_calls_to_tool_use(tool_calls: &[ToolCall]) -> Result<Vec<Value>, AppError> {
    let mut blocks = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        let input: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|e| AppError::BadRequest(format!("invalid tool_call arguments JSON: {e}")))?;
        let id = call.id.clone().unwrap_or_default();
        let name = call.function.name.clone().unwrap_or_default();
        blocks.push(json!({
            "toolUse": {
                "toolUseId": id,
                "name": name,
                "input": input,
            }
        }));
    }
    Ok(blocks)
}

/// Convert an OpenAI tool-role message into a Bedrock user turn carrying a
/// single `toolResult` block.
///
/// Ports bedrock.py:834-854: Bedrock has no `tool` role, so a tool result is
/// expressed as a `user` message whose content is one
/// `{"toolResult": {"toolUseId": <tool_call_id>, "content": [{"text": <text>}]}}`
/// block.
#[must_use]
pub fn tool_message_to_tool_result_turn(tool_call_id: &str, text_content: &str) -> Value {
    json!({
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": [{ "text": text_content }],
                }
            }
        ]
    })
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
/// Ports `_should_split_same_role_merge` (bedrock.py:1725-1756):
/// - user role: merge contiguous `toolResult` turns; split when one side has a
///   `toolResult` and the other does not (don't mix tool results with text).
/// - assistant role: merge contiguous `toolUse` turns; split when one side has a
///   `toolUse` and the other does not (don't mix toolUse with normal content).
///
/// Returns `false` (merge) for every other situation.
#[must_use]
pub fn should_split_same_role_merge(role: &str, current: &[Value], next: &[Value]) -> bool {
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

/// Do any messages carry `toolUse` or `toolResult` content blocks?
///
/// Ports `_messages_contain_tool_content` (bedrock.py:1697-1716). Bedrock
/// requires a `toolConfig` to be present whenever tool content appears in the
/// history, even if no new tools are defined.
#[must_use]
pub fn messages_contain_tool_content(messages: &[Value]) -> bool {
    for message in messages {
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in content {
            if let Some(obj) = block.as_object() {
                if obj.contains_key("toolUse") || obj.contains_key("toolResult") {
                    return true;
                }
            }
        }
    }
    false
}

#[must_use]
pub(crate) fn synthesize_tool_config_from_messages(messages: &Value) -> Option<Value> {
    let Value::Array(turns) = messages else {
        return None;
    };

    let mut names: Vec<&str> = Vec::new();
    for turn in turns {
        let Some(content) = turn.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in content {
            let Some(name) = block
                .get("toolUse")
                .and_then(|tool_use| tool_use.get("name"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }

    if names.is_empty() {
        return None;
    }

    let tools: Vec<Value> = names
        .into_iter()
        .map(|name| {
            json!({
                "toolSpec": {
                    "name": name,
                    "description": "(tool definition omitted on continuation turn)",
                    "inputSchema": {
                        "json": {
                            "type": "object"
                        }
                    },
                }
            })
        })
        .collect();

    Some(json!({ "tools": tools }))
}

/// Normalize `toolResult` turns to avoid Bedrock count mismatches.
///
/// Ports `_normalize_tool_result_turns` (bedrock.py:1038-1104). For a `user`
/// turn that immediately follows an `assistant` `toolUse` turn:
/// - build the set of `toolUseId`s the assistant turn produced
///   (`expected_id_set`),
/// - keep only `toolResult` blocks whose `toolUseId` is in that set,
/// - drop blocks with a missing/empty `toolUseId`,
/// - dedupe repeated `toolUseId`s within the same user turn.
///
/// Non-user turns, user turns without any `toolResult`, the first message, and
/// user turns whose predecessor is not an assistant `toolUse` turn (empty
/// expected set) are passed through unchanged.
#[must_use]
pub fn normalize_tool_result_turns(messages: &[Value]) -> Vec<Value> {
    let mut normalized: Vec<Value> = Vec::with_capacity(messages.len());

    for (idx, message) in messages.iter().enumerate() {
        let role = message.get("role").and_then(Value::as_str);
        let content = message.get("content").and_then(Value::as_array);

        // Only user turns with a list content are candidates.
        let (role, content) = match (role, content) {
            (Some("user"), Some(c)) => ("user", c),
            _ => {
                normalized.push(message.clone());
                continue;
            }
        };

        if !content_contains_block(content, "toolResult") {
            normalized.push(message.clone());
            continue;
        }

        if idx == 0 {
            normalized.push(message.clone());
            continue;
        }

        // Predecessor must be an assistant turn with a list content.
        let prev = &messages[idx - 1];
        let prev_role = prev.get("role").and_then(Value::as_str);
        let prev_content = prev.get("content").and_then(Value::as_array);
        let prev_content = match (prev_role, prev_content) {
            (Some("assistant"), Some(c)) => c,
            _ => {
                normalized.push(message.clone());
                continue;
            }
        };

        // Collect the toolUseIds the assistant turn produced.
        let mut expected_id_set: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for block in prev_content {
            if let Some(tool_use) = block.get("toolUse") {
                if let Some(id) = tool_use.get("toolUseId").and_then(Value::as_str) {
                    if !id.is_empty() {
                        expected_id_set.insert(id.to_string());
                    }
                }
            }
        }

        if expected_id_set.is_empty() {
            normalized.push(message.clone());
            continue;
        }

        // Filter: keep non-toolResult blocks; keep toolResult blocks whose id is
        // expected and not yet seen.
        let mut filtered: Vec<Value> = Vec::with_capacity(content.len());
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for block in content {
            let tool_result = block.get("toolResult");
            let Some(tool_result) = tool_result else {
                // Not a toolResult block — keep as-is.
                filtered.push(block.clone());
                continue;
            };
            let tool_use_id = tool_result.get("toolUseId").and_then(Value::as_str);
            let Some(tool_use_id) = tool_use_id else {
                // Missing toolUseId → drop.
                continue;
            };
            if tool_use_id.is_empty() {
                continue;
            }
            if !expected_id_set.contains(tool_use_id) {
                continue;
            }
            if seen.contains(tool_use_id) {
                continue;
            }
            seen.insert(tool_use_id.to_string());
            filtered.push(block.clone());
        }

        normalized.push(json!({ "role": role, "content": filtered }));
    }

    normalized
}

/// Build the placeholder `toolConfig` object injected when a conversation
/// carries tool content but defines no tools (bedrock.py:1215-1230, 1279-1294).
#[must_use]
pub fn placeholder_tool_config() -> Value {
    json!({
        "tools": [
            {
                "toolSpec": {
                    "name": "_placeholder",
                    "description": PLACEHOLDER_DESCRIPTION,
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {},
                        }
                    },
                }
            }
        ]
    })
}

/// Ensure a usable `toolConfig` exists whenever the conversation history carries
/// tool content.
///
/// Ports the placeholder injection (bedrock.py:1208-1232) **and** the final
/// safety net (bedrock.py:1270-1294) as a single idempotent operation:
///
/// - If `messages` contain no `toolUse`/`toolResult` blocks, returns
///   `existing_tool_config` unchanged (no injection needed).
/// - Otherwise, if `existing_tool_config` is already a valid config (an object
///   with a non-empty `tools` array), it is kept.
/// - Otherwise (no config, or a config with a missing/empty `tools` array), the
///   [`placeholder_tool_config`] is returned.
///
/// This collapses the two Python checkpoints (the `else`-branch placeholder and
/// the trailing safety net) into one function the caller invokes once after all
/// other `toolConfig` assembly.
#[must_use]
pub fn inject_placeholder_tool_config(
    messages: &[Value],
    existing_tool_config: Option<Value>,
) -> Option<Value> {
    if !messages_contain_tool_content(messages) {
        return existing_tool_config;
    }

    // Tool content present: a valid config (object with non-empty tools array)
    // is kept; anything else is replaced with the placeholder.
    let is_valid = existing_tool_config
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|o| o.get("tools"))
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());

    if is_valid {
        existing_tool_config
    } else {
        Some(placeholder_tool_config())
    }
}

#[cfg(test)]
#[path = "tools_tests.rs"]
mod tests;
