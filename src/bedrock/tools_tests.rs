//! Unit tests for [`crate::bedrock::tools`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec).
//!
//! The source file declares this via a `#[path]` mod tests, so the
//! top-level `use super::*;` resolves to the implementation module.

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
    let err =
        build_tool_config(&tools, Some(&choice), false).expect_err("missing function must error");
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

#[test]
fn empty_tool_message_preserves_tool_result_turn() {
    let turn = tool_message_to_tool_result_turn("call_1", "");
    assert_eq!(turn["role"], "user");
    assert_eq!(turn["content"][0]["toolResult"]["toolUseId"], "call_1");
    assert_eq!(turn["content"][0]["toolResult"]["content"][0]["text"], "");
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

// ---- Property: tool-name round-trip invariants -------------------------
//
// Feature: test-coverage-codecov, Property: namespace-prefix-roundtrip
// (see `.kiro/specs/test-coverage-codecov/design.md`, tools.rs row).
//
// The `{ns}__{fn}` flattening itself lives in `responses_translate.rs`
// (`build_responses_tool_specs` / `NAMESPACE_DELIMITER`), but the stateless
// round-trip contract — a prefixed name echoed back by the client must survive
// UNCHANGED through the tools.rs pipeline — is exercised here against the
// `tools.rs` functions that carry it: a prefixed name replayed as an assistant
// `toolUse` is preserved verbatim by `assistant_tool_calls_to_tool_use` and by
// `synthesize_tool_config_from_messages` (the prefix is never stripped).
//
// Validates: Requirements 1.2

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Feature: test-coverage-codecov, Property: namespace-prefix-roundtrip.
    ///
    /// For ANY namespace/inner name pair, the flattened `{ns}__{fn}` tool name
    /// round-trips UNCHANGED through the tools.rs continuation pipeline:
    /// assistant `tool_calls` → `toolUse` block name, then a synthesized
    /// `toolConfig` `toolSpec` name — both preserve the prefixed name verbatim.
    #[test]
    fn prop_namespace_prefixed_name_roundtrips_through_tool_pipeline(
        ns in "[a-z][a-z0-9]{0,10}",
        fname in "[a-z][a-z0-9]{0,10}",
    ) {
        let prefixed = format!("{ns}__{fname}");

        // 1) assistant tool_calls → toolUse: name preserved verbatim.
        let calls = vec![tool_call("call_1", &prefixed, "{}")];
        let blocks = assistant_tool_calls_to_tool_use(&calls).expect("blocks");
        prop_assert_eq!(
            blocks[0]["toolUse"]["name"].as_str().expect("name"),
            prefixed.as_str()
        );

        // 2) synthesize_tool_config_from_messages echoes it unchanged (prefix
        //    never stripped — the stateless round-trip invariant).
        let messages = json!([{ "role": "assistant", "content": blocks }]);
        let config = synthesize_tool_config_from_messages(&messages).expect("config");
        let specs = config["tools"].as_array().expect("tools array");
        prop_assert_eq!(specs.len(), 1);
        prop_assert_eq!(
            specs[0]["toolSpec"]["name"].as_str().expect("spec name"),
            prefixed.as_str()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Feature: test-coverage-codecov, Property: toolConfig name preservation.
    ///
    /// `build_tool_config` emits exactly one `toolSpec` per input tool, in the
    /// same order, each keeping its bare name.
    #[test]
    fn prop_build_tool_config_preserves_names_in_order(
        names in prop::collection::vec("[a-z][a-z0-9]{0,8}", 1..6),
    ) {
        let tools: Vec<Tool> = names.iter().map(|n| tool(n)).collect();
        let cfg = build_tool_config(&tools, None, false).expect("config");
        let specs = cfg["tools"].as_array().expect("tools array");
        prop_assert_eq!(specs.len(), names.len());
        for (spec, name) in specs.iter().zip(names.iter()) {
            prop_assert_eq!(spec["toolSpec"]["name"].as_str().expect("name"), name.as_str());
        }
    }
}
