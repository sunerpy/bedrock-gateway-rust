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
mod tests {
    use super::*;
    use crate::openai::schema::ResponseFunction;

    fn func(name: &str, description: Option<&str>, parameters: Value) -> Function {
        Function {
            name: name.to_string(),
            description: description.map(str::to_string),
            parameters,
        }
    }

    fn tool(name: &str) -> Tool {
        Tool {
            r#type: "function".to_string(),
            function: func(
                name,
                Some("desc"),
                json!({ "type": "object", "properties": {} }),
            ),
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            index: None,
            id: Some(id.to_string()),
            r#type: "function".to_string(),
            function: ResponseFunction {
                name: Some(name.to_string()),
                arguments: arguments.to_string(),
            },
        }
    }

    // ----- convert_tool_spec -------------------------------------------------

    #[test]
    fn convert_tool_spec_shapes_toolspec() {
        let f = func(
            "get_weather",
            Some("Get the weather"),
            json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        );
        let spec = convert_tool_spec(&f);
        assert_eq!(spec["toolSpec"]["name"], "get_weather");
        assert_eq!(spec["toolSpec"]["description"], "Get the weather");
        assert_eq!(
            spec["toolSpec"]["inputSchema"]["json"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn convert_tool_spec_null_description_when_absent() {
        let f = func("noop", None, json!({ "type": "object" }));
        let spec = convert_tool_spec(&f);
        assert!(spec["toolSpec"]["description"].is_null());
    }

    // ----- build_tool_config: tool_choice variants --------------------------

    #[test]
    fn tool_choice_required_maps_to_any() {
        let tools = vec![tool("t1")];
        let choice = ToolChoice::String("required".to_string());
        let cfg = build_tool_config(&tools, Some(&choice), false).expect("config");
        assert_eq!(cfg["toolChoice"], json!({ "any": {} }));
        assert_eq!(cfg["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn tool_choice_auto_maps_to_auto() {
        let tools = vec![tool("t1")];
        let choice = ToolChoice::String("auto".to_string());
        let cfg = build_tool_config(&tools, Some(&choice), false).expect("config");
        assert_eq!(cfg["toolChoice"], json!({ "auto": {} }));
    }

    #[test]
    fn tool_choice_other_string_maps_to_auto() {
        // Any non-"required" string falls back to auto (bedrock.py:1200-1201).
        let tools = vec![tool("t1")];
        let choice = ToolChoice::String("none".to_string());
        let cfg = build_tool_config(&tools, Some(&choice), false).expect("config");
        assert_eq!(cfg["toolChoice"], json!({ "auto": {} }));
    }

    #[test]
    fn tool_choice_specific_object_maps_to_tool_name() {
        let tools = vec![tool("t1")];
        let choice = ToolChoice::Object(json!({
            "type": "function",
            "function": { "name": "get_weather" }
        }));
        let cfg = build_tool_config(&tools, Some(&choice), false).expect("config");
        assert_eq!(
            cfg["toolChoice"],
            json!({ "tool": { "name": "get_weather" } })
        );
    }

    #[test]
    fn tool_choice_object_missing_function_errors() {
        let tools = vec![tool("t1")];
        let choice = ToolChoice::Object(json!({ "type": "function" }));
        let err = build_tool_config(&tools, Some(&choice), false)
            .expect_err("missing function must error");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[test]
    fn tool_choice_object_missing_name_defaults_empty() {
        // Python uses .get("name", "") — an empty name, not an error.
        let tools = vec![tool("t1")];
        let choice = ToolChoice::Object(json!({ "function": {} }));
        let cfg = build_tool_config(&tools, Some(&choice), false).expect("config");
        assert_eq!(cfg["toolChoice"], json!({ "tool": { "name": "" } }));
    }

    #[test]
    fn skip_tool_choice_omits_tool_choice() {
        // The de-hardcoded llama-skip: caller passes skip=true → no toolChoice.
        let tools = vec![tool("t1")];
        let choice = ToolChoice::String("required".to_string());
        let cfg = build_tool_config(&tools, Some(&choice), true).expect("config");
        assert!(
            cfg.get("toolChoice").is_none(),
            "toolChoice must be omitted when skip_tool_choice is true"
        );
        // tools still present.
        assert_eq!(cfg["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn no_tool_choice_omits_tool_choice() {
        let tools = vec![tool("t1")];
        let cfg = build_tool_config(&tools, None, false).expect("config");
        assert!(cfg.get("toolChoice").is_none());
    }

    // ----- assistant tool_calls → toolUse -----------------------------------

    #[test]
    fn assistant_tool_calls_to_tool_use_parses_arguments() {
        let calls = vec![tool_call("call_1", "get_weather", r#"{"city":"Paris"}"#)];
        let blocks = assistant_tool_calls_to_tool_use(&calls).expect("blocks");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["toolUse"]["toolUseId"], "call_1");
        assert_eq!(blocks[0]["toolUse"]["name"], "get_weather");
        assert_eq!(blocks[0]["toolUse"]["input"]["city"], "Paris");
    }

    #[test]
    fn assistant_tool_calls_invalid_arguments_errors() {
        let calls = vec![tool_call("call_1", "x", "not json")];
        let err = assistant_tool_calls_to_tool_use(&calls).expect_err("invalid args must error");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    // ----- tool message → toolResult turn -----------------------------------

    #[test]
    fn tool_message_becomes_user_tool_result_turn() {
        let turn = tool_message_to_tool_result_turn("call_1", "result text");
        assert_eq!(turn["role"], "user");
        assert_eq!(turn["content"][0]["toolResult"]["toolUseId"], "call_1");
        assert_eq!(
            turn["content"][0]["toolResult"]["content"][0]["text"],
            "result text"
        );
    }

    // ----- same-role merge/split --------------------------------------------

    #[test]
    fn merge_contiguous_tool_result_users() {
        let a = vec![json!({ "toolResult": { "toolUseId": "1" } })];
        let b = vec![json!({ "toolResult": { "toolUseId": "2" } })];
        assert!(!should_split_same_role_merge("user", &a, &b));
    }

    #[test]
    fn split_tool_result_from_text_user() {
        let tr = vec![json!({ "toolResult": { "toolUseId": "1" } })];
        let text = vec![json!({ "text": "hi" })];
        assert!(should_split_same_role_merge("user", &tr, &text));
        assert!(should_split_same_role_merge("user", &text, &tr));
    }

    #[test]
    fn merge_plain_text_users() {
        let a = vec![json!({ "text": "a" })];
        let b = vec![json!({ "text": "b" })];
        assert!(!should_split_same_role_merge("user", &a, &b));
    }

    #[test]
    fn merge_contiguous_tool_use_assistants() {
        let a = vec![json!({ "toolUse": { "toolUseId": "1" } })];
        let b = vec![json!({ "toolUse": { "toolUseId": "2" } })];
        assert!(!should_split_same_role_merge("assistant", &a, &b));
    }

    #[test]
    fn split_tool_use_from_text_assistant() {
        let tu = vec![json!({ "toolUse": { "toolUseId": "1" } })];
        let text = vec![json!({ "text": "thinking" })];
        assert!(should_split_same_role_merge("assistant", &tu, &text));
        assert!(should_split_same_role_merge("assistant", &text, &tu));
    }

    // ----- messages_contain_tool_content ------------------------------------

    #[test]
    fn detects_tool_use_and_tool_result() {
        let msgs = vec![json!({
            "role": "assistant",
            "content": [{ "toolUse": { "toolUseId": "1", "name": "x", "input": {} } }]
        })];
        assert!(messages_contain_tool_content(&msgs));

        let msgs2 = vec![json!({
            "role": "user",
            "content": [{ "toolResult": { "toolUseId": "1", "content": [] } }]
        })];
        assert!(messages_contain_tool_content(&msgs2));
    }

    #[test]
    fn no_tool_content_for_plain_text() {
        let msgs = vec![json!({ "role": "user", "content": [{ "text": "hi" }] })];
        assert!(!messages_contain_tool_content(&msgs));
    }

    // ----- synthesize toolConfig from replayed toolUse -------------------------

    #[test]
    fn synthesize_tool_config_from_messages_builds_specs() {
        let messages = json!([
            {
                "role": "assistant",
                "content": [
                    { "toolUse": { "toolUseId": "call_1", "name": "get_weather", "input": {} } },
                    { "toolUse": { "toolUseId": "call_2", "name": "lookup_user", "input": {} } }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    { "toolUse": { "toolUseId": "call_3", "name": "get_weather", "input": {} } }
                ]
            }
        ]);

        let config = synthesize_tool_config_from_messages(&messages).expect("toolConfig");
        let tools = config["tools"].as_array().expect("tools array");

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["toolSpec"]["name"], "get_weather");
        assert_eq!(tools[1]["toolSpec"]["name"], "lookup_user");
        for tool in tools {
            let spec = &tool["toolSpec"];
            assert!(!spec["description"].as_str().unwrap_or_default().is_empty());
            assert_eq!(spec["inputSchema"]["json"]["type"], "object");
        }
    }

    #[test]
    fn synthesize_tool_config_none_without_tooluse() {
        let messages = json!([
            { "role": "user", "content": [{ "text": "hi" }] },
            {
                "role": "user",
                "content": [
                    { "toolResult": { "toolUseId": "call_1", "content": [{ "text": "ok" }] } }
                ]
            }
        ]);

        assert!(synthesize_tool_config_from_messages(&messages).is_none());
    }

    #[test]
    fn synthesize_tool_config_none_for_non_array_messages() {
        assert!(synthesize_tool_config_from_messages(&json!({})).is_none());
    }

    // ----- normalize_tool_result_turns --------------------------------------

    fn assistant_tool_use(ids: &[&str]) -> Value {
        let blocks: Vec<Value> = ids
            .iter()
            .map(|id| json!({ "toolUse": { "toolUseId": id, "name": "f", "input": {} } }))
            .collect();
        json!({ "role": "assistant", "content": blocks })
    }

    fn user_tool_results(ids: &[&str]) -> Value {
        let blocks: Vec<Value> = ids
            .iter()
            .map(|id| json!({ "toolResult": { "toolUseId": id, "content": [{ "text": "r" }] } }))
            .collect();
        json!({ "role": "user", "content": blocks })
    }

    #[test]
    fn normalize_drops_unknown_tool_result_ids() {
        let msgs = vec![
            assistant_tool_use(&["a", "b"]),
            user_tool_results(&["a", "b", "c"]), // "c" is stale/unknown
        ];
        let out = normalize_tool_result_turns(&msgs);
        let results = out[1]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results
            .iter()
            .map(|b| b["toolResult"]["toolUseId"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(!ids.contains(&"c"));
    }

    #[test]
    fn normalize_dedupes_repeated_ids() {
        let msgs = vec![
            assistant_tool_use(&["a"]),
            user_tool_results(&["a", "a", "a"]),
        ];
        let out = normalize_tool_result_turns(&msgs);
        let results = out[1]["content"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["toolResult"]["toolUseId"], "a");
    }

    #[test]
    fn normalize_drops_missing_tool_use_id() {
        let msgs = vec![
            assistant_tool_use(&["a"]),
            json!({
                "role": "user",
                "content": [
                    { "toolResult": { "content": [{ "text": "no id" }] } },
                    { "toolResult": { "toolUseId": "a", "content": [{ "text": "ok" }] } }
                ]
            }),
        ];
        let out = normalize_tool_result_turns(&msgs);
        let results = out[1]["content"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["toolResult"]["toolUseId"], "a");
    }

    #[test]
    fn normalize_keeps_non_tool_result_blocks_in_turn() {
        let msgs = vec![
            assistant_tool_use(&["a"]),
            json!({
                "role": "user",
                "content": [
                    { "toolResult": { "toolUseId": "a", "content": [{ "text": "ok" }] } },
                    { "text": "and a note" }
                ]
            }),
        ];
        let out = normalize_tool_result_turns(&msgs);
        let results = out[1]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[1]["text"], "and a note");
    }

    #[test]
    fn normalize_passes_through_first_message() {
        // A user toolResult at index 0 has no predecessor → untouched.
        let msgs = vec![user_tool_results(&["x"])];
        let out = normalize_tool_result_turns(&msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn normalize_passes_through_when_prev_not_assistant_tool_use() {
        // Predecessor is a plain user turn → empty expected set → untouched.
        let msgs = vec![
            json!({ "role": "user", "content": [{ "text": "hi" }] }),
            user_tool_results(&["x"]),
        ];
        let out = normalize_tool_result_turns(&msgs);
        // The second turn is preceded by a user (not assistant), so it's kept.
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn normalize_leaves_plain_turns_untouched() {
        let msgs = vec![
            json!({ "role": "user", "content": [{ "text": "hi" }] }),
            json!({ "role": "assistant", "content": [{ "text": "hello" }] }),
        ];
        let out = normalize_tool_result_turns(&msgs);
        assert_eq!(out, msgs);
    }

    // ----- placeholder injection --------------------------------------------

    #[test]
    fn placeholder_injected_when_tool_content_but_no_config() {
        let msgs = vec![assistant_tool_use(&["a"]), user_tool_results(&["a"])];
        let out = inject_placeholder_tool_config(&msgs, None).expect("placeholder injected");
        let tools = out["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["toolSpec"]["name"], "_placeholder");
        assert_eq!(
            tools[0]["toolSpec"]["inputSchema"]["json"]["type"],
            "object"
        );
    }

    #[test]
    fn no_placeholder_without_tool_content() {
        let msgs = vec![json!({ "role": "user", "content": [{ "text": "hi" }] })];
        assert!(inject_placeholder_tool_config(&msgs, None).is_none());
    }

    #[test]
    fn existing_valid_config_kept() {
        let msgs = vec![assistant_tool_use(&["a"]), user_tool_results(&["a"])];
        let existing = build_tool_config(&[tool("real")], None, false).unwrap();
        let out = inject_placeholder_tool_config(&msgs, Some(existing)).expect("kept");
        // Real tool config preserved, not replaced by placeholder.
        assert_eq!(out["tools"][0]["toolSpec"]["name"], "real");
    }

    #[test]
    fn safety_net_replaces_empty_tools_config() {
        // Config exists but with an empty tools array → placeholder kicks in.
        let msgs = vec![assistant_tool_use(&["a"]), user_tool_results(&["a"])];
        let empty = json!({ "tools": [] });
        let out = inject_placeholder_tool_config(&msgs, Some(empty)).expect("replaced");
        assert_eq!(out["tools"][0]["toolSpec"]["name"], "_placeholder");
    }

    #[test]
    fn safety_net_replaces_non_object_config() {
        let msgs = vec![assistant_tool_use(&["a"]), user_tool_results(&["a"])];
        let bogus = json!("not an object");
        let out = inject_placeholder_tool_config(&msgs, Some(bogus)).expect("replaced");
        assert_eq!(out["tools"][0]["toolSpec"]["name"], "_placeholder");
    }
}
