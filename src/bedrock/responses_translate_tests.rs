//! Unit and property tests for the Responses → Converse input translation.
//!
//! Relocated out of `responses_translate.rs` into a sibling `#[path]` module for
//! code organization (see the `test-coverage-codecov` spec). Behavior is
//! unchanged; `use super::*;` still resolves to the implementation module.

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
async fn empty_string_input_is_bad_request() {
    let req = req_from(json!({ "model": "m", "input": "" }));
    let err = parse(&req)
        .await
        .expect_err("empty string input must reject");

    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("input must contain non-empty text"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn user_message_with_only_empty_text_is_bad_request() {
    let req = req_from(json!({
        "model": "m",
        "input": [{ "type": "message", "role": "user", "content": "" }]
    }));
    let err = parse(&req)
        .await
        .expect_err("empty user message must reject");

    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("message content must contain at least"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn mixed_empty_text_parts_keep_only_non_empty_text() {
    let req = req_from(json!({
        "model": "m",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                { "type": "input_text", "text": "" },
                { "type": "input_text", "text": "keep me" }
            ]
        }]
    }));
    let out = parse(&req).await.expect("parse");
    let content = out.messages[0]["content"].as_array().expect("content");

    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["text"], "keep me");
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
async fn empty_instructions_are_skipped() {
    let req = req_from(json!({
        "model": "m",
        "instructions": "",
        "input": "hi"
    }));
    let out = parse(&req).await.expect("parse");

    assert_eq!(out.system.as_array().expect("system").len(), 0);
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

// -- Test 4: hosted tools are DROPPED (not 400); malformed text.format → 400

#[tokio::test]
async fn hosted_server_tool_is_dropped_not_rejected() {
    // A web_search tool now deserializes to ResponsesTool::Unknown and is
    // silently dropped — the request must parse (no 400) so codex sessions
    // that bundle a hosted tool survive.
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [{ "type": "web_search" }]
    }));
    let out = parse(&req)
        .await
        .expect("hosted tool must be dropped, not rejected");
    let msgs = out.messages.as_array().expect("messages");
    assert_eq!(msgs.len(), 1);
    // No toolSpec produced from a lone hosted tool.
    assert!(build_responses_tool_specs(&req).is_empty());

    // A hosted tool smuggled via `extra["tools"]` is also ignored now (no
    // 400): the defensive guard was removed along with the rejection path.
    let mut req2 = req_from(json!({ "model": "m", "input": "hi" }));
    req2.extra
        .insert("tools".to_string(), json!([{ "type": "code_interpreter" }]));
    assert!(
        parse(&req2).await.is_ok(),
        "extra-smuggled hosted tool must not 400"
    );
}

#[tokio::test]
async fn function_tool_is_accepted() {
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [{ "type": "function", "name": "f", "parameters": {"type": "object"} }]
    }));
    assert!(parse(&req).await.is_ok());
    // A top-level function keeps its BARE name in the toolSpec.
    let specs = build_responses_tool_specs(&req);
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "f");
}

#[tokio::test]
async fn namespace_tool_is_flattened_with_prefixed_names_and_hosted_dropped() {
    // tools: [namespace{multi_agent_v1, tools:[function spawn_agent]}, web_search]
    // → exactly ONE toolSpec named "multi_agent_v1__spawn_agent"; web_search
    // is dropped (not present, no error).
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "agent tools",
                "tools": [
                    { "type": "function", "name": "spawn_agent",
                      "description": "spawn", "strict": false,
                      "parameters": { "type": "object", "properties": {} } }
                ]
            },
            { "type": "web_search" }
        ]
    }));
    // Parse must succeed (no 400 on the hosted web_search).
    assert!(parse(&req).await.is_ok());

    let specs = build_responses_tool_specs(&req);
    assert_eq!(
        specs.len(),
        1,
        "exactly one flattened toolSpec, web_search dropped"
    );
    assert_eq!(specs[0]["toolSpec"]["name"], "multi_agent_v1__spawn_agent");
    assert_eq!(specs[0]["toolSpec"]["description"], "spawn");
    assert!(specs[0]["toolSpec"]["inputSchema"]["json"].is_object());
}

#[tokio::test]
async fn multiple_namespaces_do_not_collide() {
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [
            { "type": "namespace", "name": "ns_a", "description": "a",
              "tools": [{ "type": "function", "name": "run" }] },
            { "type": "namespace", "name": "ns_b", "description": "b",
              "tools": [{ "type": "function", "name": "run" }] }
        ]
    }));
    let specs = build_responses_tool_specs(&req);
    let names: Vec<&str> = specs
        .iter()
        .map(|s| s["toolSpec"]["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"ns_a__run"));
    assert!(names.contains(&"ns_b__run"));
}

#[tokio::test]
async fn custom_tool_becomes_toolspec() {
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [{ "type": "custom", "name": "c", "description": "free-form" }]
    }));
    let specs = build_responses_tool_specs(&req);
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "c");
    assert_eq!(specs[0]["toolSpec"]["description"], "free-form");
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
async fn assistant_replay_empty_output_text_is_skipped() {
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "user", "content": "first" },
            { "type": "message", "role": "assistant", "content": [
                { "type": "output_text", "text": "" }
            ]}
        ]
    }));
    let out = parse(&req).await.expect("parse");
    let msgs = out.messages.as_array().expect("messages");

    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"][0]["text"], "first");
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
    let assistant_turns: Vec<&Value> = msgs.iter().filter(|m| m["role"] == "assistant").collect();
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

#[tokio::test]
async fn empty_function_call_output_reaches_sdk_as_json_empty_object() {
    use aws_sdk_bedrockruntime::types::{ContentBlock, ToolResultContentBlock};

    use crate::bedrock::provider::{build_sdk_messages, document_to_json};

    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "" }
        ]
    }));
    let out = parse(&req).await.expect("parse empty tool output");
    let msgs = out.messages.as_array().expect("messages");

    assert_eq!(msgs[1]["content"][0]["toolResult"]["toolUseId"], "c1");
    assert_eq!(
        msgs[1]["content"][0]["toolResult"]["content"][0]["text"],
        ""
    );

    let sdk = build_sdk_messages(&out.messages).expect("sdk messages");
    match &sdk[1].content()[0] {
        ContentBlock::ToolResult(result) => match &result.content()[0] {
            ToolResultContentBlock::Json(doc) => assert_eq!(document_to_json(doc), json!({})),
            other => panic!("expected JSON empty object, got {other:?}"),
        },
        other => panic!("expected toolResult, got {other:?}"),
    }
}

// -- Additional branch coverage: error paths & edge cases -----------------

#[tokio::test]
async fn function_call_with_invalid_arguments_json_is_bad_request() {
    // `arguments` is a JSON string; a non-JSON value must 400, not panic.
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "not json" }
        ]
    }));
    let err = parse(&req)
        .await
        .expect_err("invalid function_call arguments must 400");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("invalid function_call arguments JSON"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn unsupported_input_item_type_is_bad_request() {
    // An item carrying an unrecognized `type` (no `role`) deserializes to
    // `ResponseInputItem::Other` and is rejected with a clear 400.
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "totally_unknown_item", "foo": "bar" }
        ]
    }));
    let err = parse(&req)
        .await
        .expect_err("unknown input item type must 400");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("totally_unknown_item"));
            assert!(message.contains("is not supported"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn image_in_system_message_is_bad_request() {
    // An `input_image` part is not valid inside a system/developer message.
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "system", "content": [
                { "type": "input_image", "image_url": "data:image/png;base64,aGk=" }
            ]}
        ]
    }));
    let err = parse(&req)
        .await
        .expect_err("image in system message must 400");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("image content is not supported in a system"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn file_in_developer_message_is_bad_request() {
    // An `input_file` part is not valid inside a system/developer message.
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "developer", "content": [
                { "type": "input_file", "file_id": "f1" }
            ]}
        ]
    }));
    let err = parse(&req)
        .await
        .expect_err("file in developer message must 400");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("file content is not supported in a system"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn remote_image_url_is_fetched_via_resolver() {
    // A non-`data:` image URL is delegated to the resolver's `fetch` (the
    // offline TestResolver returns fixed bytes), covering the remote-image
    // branch that `data:`-URI tests never reach.
    let req = req_from(json!({
        "model": "m",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                { "type": "input_image", "image_url": "https://example.com/pic.jpg" }
            ]
        }]
    }));
    let out = parse(&req).await.expect("parse");
    let content = out.messages[0]["content"].as_array().expect("content");
    assert_eq!(content.len(), 1);
    // TestResolver::fetch returns (vec![1, 2, 3], "jpeg").
    assert_eq!(content[0]["image"]["format"], "jpeg");
    let bytes = content[0]["image"]["source"]["bytes"]
        .as_array()
        .expect("bytes");
    assert_eq!(bytes.len(), 3);
    assert_eq!(bytes[0], 1);
}

#[tokio::test]
async fn namespace_inner_custom_tool_is_flattened_with_prefix() {
    // A namespace holding a `custom` inner tool flattens to a single prefixed
    // toolSpec, covering the `ResponsesNamespaceInner::Custom` arm.
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [
            {
                "type": "namespace",
                "name": "ns_c",
                "description": "container",
                "tools": [
                    { "type": "custom", "name": "freeform", "description": "cd" }
                ]
            }
        ]
    }));
    assert!(parse(&req).await.is_ok());
    let specs = build_responses_tool_specs(&req);
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "ns_c__freeform");
    assert_eq!(specs[0]["toolSpec"]["description"], "cd");
    // A custom inner tool carries no parameters → the default empty-object
    // schema is applied.
    assert!(specs[0]["toolSpec"]["inputSchema"]["json"].is_object());
}

#[tokio::test]
async fn text_without_format_is_accepted() {
    // `text` present but with no `format` → the format gate returns early Ok.
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "text": { "verbosity": "low" }
    }));
    assert!(parse(&req).await.is_ok());
}

#[tokio::test]
async fn text_format_without_type_is_bad_request() {
    // A `format` object missing the `type` string is malformed → 400.
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "text": { "format": { "name": "no_type_here" } }
    }));
    let err = parse(&req).await.expect_err("format without type must 400");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("text.format.type is missing"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

/// Property tests for the `namespace` → `{ns}__{fn}` flattening in
/// [`build_responses_tool_specs`].
///
/// Feature: test-coverage-codecov, Property: namespace-flatten
/// (see `.kiro/specs/test-coverage-codecov/design.md`).
///
/// Validates: Requirements 1.2
mod prop_tests {
    use super::super::*;
    use proptest::prelude::*;
    use serde_json::json;

    /// A generator of DISTINCT, delimiter-free lowercase identifiers. Excluding
    /// `_` from the alphabet guarantees no generated name can itself contain the
    /// `{ns}__{fn}` join delimiter, so the resulting prefixed names are
    /// unambiguous and any collision would be a real bug, not a generator
    /// artifact.
    fn ident() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9]{0,7}"
    }

    /// Build a `ResponsesRequest` whose `tools` array is a set of namespace
    /// tools, each holding a set of inner `function` tools, from a
    /// `(ns_name, [inner_name, ...])` spec.
    fn request_with_namespaces(specs: &[(String, Vec<String>)]) -> ResponsesRequest {
        let tools: Vec<Value> = specs
            .iter()
            .map(|(ns, inner)| {
                let inner_tools: Vec<Value> = inner
                    .iter()
                    .map(|name| json!({ "type": "function", "name": name }))
                    .collect();
                json!({
                    "type": "namespace",
                    "name": ns,
                    "description": "ns",
                    "tools": inner_tools,
                })
            })
            .collect();
        serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "tools": tools,
        }))
        .expect("deserialize ResponsesRequest")
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov, Property: namespace-flatten.
        ///
        /// For any set of namespaces (distinct names) each holding a set of
        /// inner function tools (names distinct within a namespace):
        /// - the flattened toolSpec count equals the total inner tool count
        ///   (one toolSpec per inner tool — nothing dropped, nothing added),
        /// - every toolSpec name equals `{ns}{NAMESPACE_DELIMITER}{fn}`, and
        /// - all toolSpec names are unique (namespaces never collide).
        #[test]
        fn prop_namespace_flatten_prefixes_and_never_collides(
            raw in prop::collection::vec(
                (ident(), prop::collection::vec(ident(), 0..4)),
                0..5,
            ),
        ) {
            // Dedup namespace names, and dedup inner names within each namespace,
            // so the input is a well-formed "set of sets" (the flattening
            // contract's precondition).
            let mut seen_ns = std::collections::HashSet::new();
            let mut specs: Vec<(String, Vec<String>)> = Vec::new();
            for (ns, inner) in raw {
                if !seen_ns.insert(ns.clone()) {
                    continue;
                }
                let mut seen_fn = std::collections::HashSet::new();
                let deduped: Vec<String> =
                    inner.into_iter().filter(|f| seen_fn.insert(f.clone())).collect();
                specs.push((ns, deduped));
            }

            let req = request_with_namespaces(&specs);
            let out = build_responses_tool_specs(&req);

            // Expected prefixed names, in flattening order.
            let expected: Vec<String> = specs
                .iter()
                .flat_map(|(ns, inner)| {
                    inner
                        .iter()
                        .map(move |f| format!("{ns}{NAMESPACE_DELIMITER}{f}"))
                })
                .collect();

            // (1) one toolSpec per inner tool.
            prop_assert_eq!(out.len(), expected.len());

            // (2) each name is the `{ns}__{fn}` prefixed form, in order.
            let got: Vec<String> = out
                .iter()
                .map(|s| s["toolSpec"]["name"].as_str().unwrap().to_string())
                .collect();
            prop_assert_eq!(&got, &expected);

            // (3) no collisions across namespaces.
            let unique: std::collections::HashSet<&String> = got.iter().collect();
            prop_assert_eq!(unique.len(), got.len());

            // Every prefixed name embeds the delimiter exactly where expected.
            for name in &got {
                prop_assert!(name.contains(NAMESPACE_DELIMITER));
            }
        }
    }
}
