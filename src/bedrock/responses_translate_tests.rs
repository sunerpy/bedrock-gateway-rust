//! Unit and property tests for the Responses → Converse input translation.
//!
//! Relocated out of `responses_translate.rs` into a sibling `#[path]` module for
//! code organization (see the sidecar-test convention in `AGENTS.md`). Behavior is
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
async fn item_reference_is_accepted_and_ignored_by_stateless_gateway() {
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "item_reference", "id": "rs_stored_1" },
            { "role": "user", "content": "continue" }
        ]
    }));
    let out = parse(&req).await.expect("stateless reference is ignored");
    let messages = out.messages.as_array().expect("messages");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"][0]["text"], "continue");
}

// -- Replayed web_search_call items (a hosted search mantle ran) are skipped --

/// A `web_search_call` as zuno replays it after a search the GPT (mantle)
/// backend ran: the action with its query, and no results.
fn replayed_web_search_call(id: &str, action: Value) -> Value {
    json!({ "type": "web_search_call", "id": id, "status": "completed", "action": action })
}

/// The same `input` without the items `skipped` matches.
fn input_without(input: &Value, skipped: impl Fn(&Value) -> bool) -> Value {
    input
        .as_array()
        .expect("input items")
        .iter()
        .filter(|item| !skipped(item))
        .cloned()
        .collect()
}

/// The same `input` with its `web_search_call` items removed.
fn without_web_search_calls(input: &Value) -> Value {
    input_without(input, |item| item["type"] == "web_search_call")
}

fn roles(out: &ResponsesConverseInput) -> Vec<&str> {
    out.messages
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| message["role"].as_str().expect("role"))
        .collect()
}

#[tokio::test]
async fn replayed_web_search_call_is_skipped_on_both_lanes() {
    let input = json!([
        { "type": "message", "role": "user", "content": "What changed in Rust 1.98?" },
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "Rust 1.98 stabilized two APIs." }
        ]},
        replayed_web_search_call("ws_1", json!({
            "type": "search",
            "query": "Rust 1.98 release notes",
            "queries": ["Rust 1.98 release notes"]
        })),
        { "type": "message", "role": "user", "content": "Which ones?" }
    ]);
    let without = req_from(json!({ "model": "m", "input": without_web_search_calls(&input) }));
    let with = req_from(json!({ "model": "m", "input": input }));
    let c = caps();
    let r = resolver(true);

    for tool_search in [false, true] {
        let model = "anthropic.claude-3-sonnet-v1:0";
        let out = to_responses_converse_input_with(&with, model, &r, &c, tool_search)
            .await
            .unwrap_or_else(|e| panic!("tool_search={tool_search}: {e:?}"));
        let expected = to_responses_converse_input_with(&without, model, &r, &c, tool_search)
            .await
            .expect("parse without the search");

        assert_eq!(out, expected, "tool_search={tool_search}");
        assert_eq!(roles(&out), ["user", "assistant", "user"]);
        assert_eq!(
            out.messages[1]["content"],
            json!([{ "text": "Rust 1.98 stabilized two APIs." }])
        );
        assert_eq!(
            out.messages[2]["content"],
            json!([{ "text": "Which ones?" }])
        );
    }
}

#[tokio::test]
async fn assistant_turns_around_a_skipped_web_search_call_merge() {
    // GPT's commentary, two searches, then its answer: with the searches
    // skipped, the two assistant texts are contiguous and merge into one
    // assistant message.
    let input = json!([
        { "type": "message", "role": "user", "content": "What changed in Rust 1.98?" },
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "Let me check the release notes." }
        ]},
        replayed_web_search_call("ws_1", json!({
            "type": "search",
            "query": "Rust 1.98",
            "queries": ["Rust 1.98", "Rust 1.98 release notes"]
        })),
        replayed_web_search_call("ws_2", json!({
            "type": "open_page",
            "url": "https://example.com/rust-1-98"
        })),
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "Rust 1.98 stabilized two APIs." }
        ]},
        { "type": "message", "role": "user", "content": "Thanks." }
    ]);
    let expected = parse(&req_from(
        json!({ "model": "m", "input": without_web_search_calls(&input) }),
    ))
    .await
    .expect("parse without the searches");
    let out = parse(&req_from(json!({ "model": "m", "input": input })))
        .await
        .expect("web_search_call items are skipped");

    assert_eq!(out, expected);
    assert_eq!(roles(&out), ["user", "assistant", "user"]);
    assert_eq!(
        out.messages[1]["content"],
        json!([
            { "text": "Let me check the release notes." },
            { "text": "Rust 1.98 stabilized two APIs." }
        ])
    );
}

#[tokio::test]
async fn skipped_web_search_call_keeps_reasoning_replay_boundaries() {
    // A search between a signed reasoning item and its text must not split
    // them into two assistant messages, and a search before the next
    // reasoning item must not merge that reasoning into the previous message.
    let first = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "reasoningText": { "text": "search first", "signature": "sig-1" }
        }
    })])
    .expect("first envelope");
    let second = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "reasoningText": { "text": "answer now", "signature": "sig-2" }
        }
    })])
    .expect("second envelope");
    let input = json!([
        { "type": "message", "role": "user", "content": "start" },
        { "type": "reasoning", "id": "rs_1", "encrypted_content": first },
        replayed_web_search_call("ws_1", json!({
            "type": "search",
            "query": "q",
            "queries": ["q"]
        })),
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "first response" }
        ]},
        replayed_web_search_call("ws_2", json!({
            "type": "find_in_page",
            "url": "https://example.com/notes",
            "pattern": "stabilized"
        })),
        { "type": "reasoning", "id": "rs_2", "encrypted_content": second },
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "second response" }
        ]},
        { "type": "message", "role": "user", "content": "continue" }
    ]);
    let expected = parse(&req_from(
        json!({ "model": "m", "input": without_web_search_calls(&input) }),
    ))
    .await
    .expect("parse without the searches");
    let out = parse(&req_from(json!({ "model": "m", "input": input })))
        .await
        .expect("web_search_call items are skipped");

    assert_eq!(out, expected);
    assert_eq!(roles(&out), ["user", "assistant", "assistant", "user"]);
    let first_turn = out.messages[1]["content"].as_array().expect("first turn");
    assert_eq!(first_turn.len(), 2);
    assert_eq!(
        first_turn[0]["reasoningContent"]["reasoningText"]["signature"],
        "sig-1"
    );
    assert_eq!(first_turn[1]["text"], "first response");
    let second_turn = out.messages[2]["content"].as_array().expect("second turn");
    assert_eq!(second_turn.len(), 2);
    assert_eq!(
        second_turn[0]["reasoningContent"]["reasoningText"]["signature"],
        "sig-2"
    );
    assert_eq!(second_turn[1]["text"], "second response");
}

// -- A conversation that switches between Claude and GPT (mantle) ---------

/// A reasoning item from a GPT (mantle) turn: OpenAI's own opaque
/// `encrypted_content`, which is not the gateway's envelope.
fn gpt_reasoning(id: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [{ "type": "summary_text", "text": "Looked it up." }],
        "encrypted_content": "gAAAAABpZXhhbXBsZS1vcGVuYWktY2lwaGVydGV4dA=="
    })
}

/// An `image_generation_call` as zuno replays it after the hosted image tool
/// ran on a GPT turn.
fn replayed_image_generation_call(id: &str) -> Value {
    json!({
        "type": "image_generation_call",
        "id": id,
        "status": "completed",
        "revised_prompt": "A gray tabby cat",
        "result": "aGk="
    })
}

#[tokio::test]
async fn alternating_claude_and_gpt_turns_replay_only_gateway_reasoning() {
    let claude_reasoning = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "reasoningText": { "text": "read a.rs first", "signature": "sig-claude" }
        }
    })])
    .expect("envelope");
    let input = json!([
        // Claude turn.
        { "type": "message", "role": "user", "content": "Find the bug in a.rs." },
        { "type": "reasoning", "id": "rs_claude_1", "encrypted_content": claude_reasoning },
        { "type": "function_call", "call_id": "tooluse_1", "name": "read_file",
          "arguments": "{\"path\":\"a.rs\"}" },
        { "type": "function_call_output", "call_id": "tooluse_1", "output": "panic!()" },
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "main panics." }
        ]},
        // GPT turn: OpenAI reasoning, a search, a tool call, an image.
        { "type": "message", "role": "user", "content": "Find the fix." },
        gpt_reasoning("rs_gpt_1"),
        replayed_web_search_call("ws_1", json!({
            "type": "search",
            "query": "rust panic fix",
            "queries": ["rust panic fix"]
        })),
        { "type": "function_call", "call_id": "call_2", "name": "read_file",
          "arguments": "{\"path\":\"b.rs\"}" },
        { "type": "function_call_output", "call_id": "call_2", "output": "fn fix() {}" },
        // zuno sends `encrypted_content: null` when the turn had none.
        { "type": "reasoning", "id": "rs_gpt_2", "summary": [], "encrypted_content": null },
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "Call fix() instead." }
        ]},
        replayed_image_generation_call("ig_1"),
        // Back on Claude.
        { "type": "message", "role": "user", "content": "Apply it." }
    ]);
    let without = req_from(json!({
        "model": "m",
        "input": input_without(&input, |item| {
            item["id"].as_str().is_some_and(|id| {
                id.starts_with("rs_gpt") || id.starts_with("ws_") || id.starts_with("ig_")
            })
        })
    }));
    let with = req_from(json!({ "model": "m", "input": input }));
    let c = caps();
    let r = resolver(true);

    for tool_search in [false, true] {
        let model = "anthropic.claude-3-sonnet-v1:0";
        let out = to_responses_converse_input_with(&with, model, &r, &c, tool_search)
            .await
            .unwrap_or_else(|e| panic!("tool_search={tool_search}: {e:?}"));
        let expected = to_responses_converse_input_with(&without, model, &r, &c, tool_search)
            .await
            .expect("parse without the GPT-only items");

        assert_eq!(out, expected, "tool_search={tool_search}");
        assert_eq!(
            roles(&out),
            [
                "user",
                "assistant",
                "user",
                "assistant",
                "user",
                "assistant",
                "user",
                "assistant",
                "user"
            ]
        );
        let claude_turn = out.messages[1]["content"].as_array().expect("Claude turn");
        assert_eq!(
            claude_turn[0]["reasoningContent"]["reasoningText"]["signature"],
            "sig-claude"
        );
        assert_eq!(claude_turn[1]["toolUse"]["toolUseId"], "tooluse_1");
        assert_eq!(
            out.messages[5]["content"],
            json!([{ "toolUse": {
                "toolUseId": "call_2", "name": "read_file", "input": { "path": "b.rs" }
            }}])
        );
        assert_eq!(
            out.messages[7]["content"],
            json!([{ "text": "Call fix() instead." }])
        );
    }
}

#[tokio::test]
async fn reasoning_input_without_encrypted_content_is_skipped() {
    // Only the gateway's envelope can be replayed; an item without one is
    // another provider's reasoning (or carries nothing Bedrock can verify).
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "user", "content": "hi" },
            { "type": "reasoning" },
            { "type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": null }
        ]
    }));

    let out = parse(&req)
        .await
        .expect("reasoning without an envelope is skipped");
    assert_eq!(
        out.messages,
        json!([{ "role": "user", "content": [{ "text": "hi" }] }])
    );
}

#[tokio::test]
async fn malformed_gateway_reasoning_envelope_is_still_rejected() {
    fn gateway_envelope(payload: &[u8]) -> String {
        format!(
            "bedrock-reasoning-v1:{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, payload)
        )
    }
    let cases = [
        (
            "bedrock-reasoning-v1:not base64!".to_string(),
            "malformed reasoning.encrypted_content",
        ),
        (
            gateway_envelope(b"not json"),
            "malformed reasoning.encrypted_content",
        ),
        (
            gateway_envelope(br#"{"version":3,"blocks":[]}"#),
            "unsupported reasoning.encrypted_content envelope version",
        ),
        // Version 2 is the web search carrier (plan C2): a v2 payload without
        // the carrier fields is still a 400.
        (
            gateway_envelope(br#"{"version":2,"blocks":[]}"#),
            "malformed web search carrier",
        ),
        (
            gateway_envelope(br#"{"version":1}"#),
            "reasoning envelope has no blocks",
        ),
        (
            gateway_envelope(br#"{"version":1,"blocks":[{"reasoningText":{"text":"t"}}]}"#),
            "reasoning envelope contains an invalid block",
        ),
    ];

    for (envelope, expected) in cases {
        let req = req_from(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "hi" },
                { "type": "reasoning", "id": "rs_1", "encrypted_content": envelope }
            ]
        }));
        match parse(&req).await {
            Err(AppError::BadRequest(message)) => assert_eq!(message, expected),
            other => panic!("{expected}: expected BadRequest, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn replayed_image_generation_call_is_skipped() {
    let input = json!([
        { "type": "message", "role": "user", "content": "Draw a cat." },
        replayed_image_generation_call("ig_1"),
        { "type": "message", "role": "assistant", "content": [
            { "type": "output_text", "text": "Here is the cat." }
        ]},
        { "type": "message", "role": "user", "content": "Make it gray." }
    ]);
    let expected = parse(&req_from(json!({
        "model": "m",
        "input": input_without(&input, |item| item["type"] == "image_generation_call")
    })))
    .await
    .expect("parse without the image");
    let out = parse(&req_from(json!({ "model": "m", "input": input })))
        .await
        .expect("image_generation_call is skipped");

    assert_eq!(out, expected);
    assert_eq!(roles(&out), ["user", "assistant", "user"]);
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
async fn previous_response_id_is_accepted_and_ignored_by_stateless_backend() {
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "store": true,
        "previous_response_id": "resp_prev_123",
        "include": ["reasoning.encrypted_content"]
    }));
    assert!(parse(&req).await.is_ok());
}

#[tokio::test]
async fn store_false_is_accepted() {
    // codex sends store:false — must NOT 400.
    let req = req_from(json!({ "model": "m", "input": "hi", "store": false }));
    assert!(parse(&req).await.is_ok());
}

// -- Test 4: hosted tools are DROPPED (not 400); malformed text.format → 400

#[tokio::test]
async fn hosted_server_tool_is_silently_dropped_for_converse() {
    let req = req_from(json!({
        "model": "m",
        "input": "hi",
        "tools": [{ "type": "web_search" }]
    }));
    assert!(
        parse(&req).await.is_ok(),
        "input translation remains independent"
    );
    let (specs, registry) = build_responses_tools(&req).expect("hosted tool is ignored");
    assert!(specs.is_empty());
    assert!(registry.resolve("web_search").is_none());

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

// -- Test 5: signed reasoning input is replayed ---------------------------

#[tokio::test]
async fn reasoning_input_item_reconstructs_signed_bedrock_block() {
    let envelope = encode_reasoning_envelope(&[json!({
        "reasoningContent": { "reasoningText": { "text": "secret", "signature": "sig" } }
    })])
    .expect("envelope");
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "user", "content": "hi" },
            { "type": "reasoning", "id": "r1", "encrypted_content": envelope, "summary": ["s"] }
        ]
    }));
    let out = parse(&req).await.expect("signed reasoning replay parses");
    let msgs = out.messages.as_array().expect("messages");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"][0]["text"], "hi");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(
        msgs[1]["content"][0]["reasoningContent"]["reasoningText"]["text"],
        "secret"
    );
    assert_eq!(
        msgs[1]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
        "sig"
    );
}

#[tokio::test]
async fn signed_reasoning_text_and_tool_calls_stay_in_one_assistant_turn() {
    let envelope = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "reasoningText": { "text": "secret", "signature": "sig" }
        }
    })])
    .expect("envelope");
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "user", "content": "inspect both files" },
            {
                "type": "reasoning",
                "id": "rs_response_1",
                "encrypted_content": envelope
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "I will inspect both." }]
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{\"path\":\"a.rs\"}"
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": "{\"path\":\"b.rs\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "a"
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "b"
            }
        ]
    }));

    let out = parse(&req).await.expect("signed tool replay parses");
    let messages = out.messages.as_array().expect("messages");

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "assistant");
    let assistant = messages[1]["content"]
        .as_array()
        .expect("assistant content");
    assert_eq!(assistant.len(), 4);
    assert_eq!(
        assistant[0]["reasoningContent"]["reasoningText"]["signature"],
        "sig"
    );
    assert_eq!(assistant[1]["text"], "I will inspect both.");
    assert_eq!(assistant[2]["toolUse"]["toolUseId"], "call_1");
    assert_eq!(assistant[3]["toolUse"]["toolUseId"], "call_2");

    assert_eq!(messages[2]["role"], "user");
    let results = messages[2]["content"].as_array().expect("tool results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["toolResult"]["toolUseId"], "call_1");
    assert_eq!(results[1]["toolResult"]["toolUseId"], "call_2");
}

#[tokio::test]
async fn each_signed_reasoning_item_starts_a_new_assistant_turn() {
    let first = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "reasoningText": { "text": "first thought", "signature": "sig-1" }
        }
    })])
    .expect("first envelope");
    let second = encode_reasoning_envelope(&[json!({
        "reasoningContent": {
            "redactedContent": "opaque-redacted-thinking"
        }
    })])
    .expect("second envelope");
    let req = req_from(json!({
        "model": "m",
        "input": [
            { "type": "message", "role": "user", "content": "start" },
            { "type": "reasoning", "id": "rs_1", "encrypted_content": first },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "first response" }]
            },
            { "type": "reasoning", "id": "rs_2", "encrypted_content": second },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "second response" }]
            },
            { "type": "message", "role": "user", "content": "continue" }
        ]
    }));

    let out = parse(&req).await.expect("multiple signed responses parse");
    let messages = out.messages.as_array().expect("messages");

    assert_eq!(messages.len(), 4);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][1]["text"], "first response");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(
        messages[2]["content"][0]["reasoningContent"]["redactedContent"],
        "opaque-redacted-thinking"
    );
    assert_eq!(messages[2]["content"][1]["text"], "second response");
    assert_eq!(messages[3]["role"], "user");
}

#[tokio::test]
async fn reasoning_input_without_id_replays_identically_to_id_bearing_item() {
    let envelope = encode_reasoning_envelope(&[json!({
        "reasoningContent": { "reasoningText": { "text": "secret", "signature": "sig" } }
    })])
    .expect("envelope");
    let with_id = req_from(json!({
        "model": "m",
        "input": [{ "type": "reasoning", "id": "r1", "encrypted_content": envelope }]
    }));
    let without_id = req_from(json!({
        "model": "m",
        "input": [{ "type": "reasoning", "encrypted_content": envelope }]
    }));

    let with_id_output = parse(&with_id).await.expect("id-bearing replay parses");
    let without_id_output = parse(&without_id).await.expect("id-less replay parses");
    assert_eq!(without_id_output, with_id_output);
}

#[tokio::test]
async fn malformed_reasoning_envelope_is_rejected_locally() {
    let req = req_from(json!({
        "model": "m",
        "input": [{
            "type": "reasoning",
            "id": "r1",
            "encrypted_content": "bedrock-reasoning-v1:not-base64"
        }]
    }));
    let err = parse(&req)
        .await
        .expect_err("malformed envelope must reject");
    assert!(err
        .to_string()
        .contains("malformed reasoning.encrypted_content"));
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
async fn no_assistant_prefill_appends_continuation_for_responses() {
    for model in [
        "us.anthropic.claude-opus-5",
        "global.anthropic.claude-opus-5",
        "global.anthropic.claude-fable-5-1",
        "us.anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5-5",
        "global.anthropic.claude-opus-5-5",
    ] {
        let req = req_from(json!({
            "model": model,
            "input": [
                { "type": "message", "role": "user", "content": "first" },
                { "type": "message", "role": "assistant", "content": [
                    { "type": "output_text", "text": "partial reply" }
                ]}
            ]
        }));
        let c = caps();
        let r = resolver(true);
        let out = to_responses_converse_input(&req, model, &r, &c)
            .await
            .expect("parse");
        let msgs = out.messages.as_array().expect("messages");

        assert_eq!(msgs.len(), 3, "{model}");
        assert_eq!(msgs[1]["role"], "assistant", "{model}");
        assert_eq!(msgs[2]["role"], "user", "{model}");
        assert_eq!(
            msgs[2]["content"][0]["text"], "Please continue your response from where you left off.",
            "{model}"
        );
    }
}

#[tokio::test]
async fn no_assistant_prefill_leaves_other_endings_unchanged() {
    let cases = [
        (
            "us.anthropic.claude-opus-5",
            json!([
                { "type": "message", "role": "user", "content": "current request" }
            ]),
            "user",
        ),
        (
            // A model without `no_assistant_prefill` keeps the assistant ending.
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            json!([
                { "type": "message", "role": "user", "content": "first" },
                { "type": "message", "role": "assistant", "content": [
                    { "type": "output_text", "text": "prior reply" }
                ]}
            ]),
            "assistant",
        ),
    ];

    for (model, input, expected_last_role) in cases {
        let req = req_from(json!({ "model": model, "input": input }));
        let c = caps();
        let r = resolver(true);
        let out = to_responses_converse_input(&req, model, &r, &c)
            .await
            .expect("parse");
        let msgs = out.messages.as_array().expect("messages");

        assert_eq!(msgs.last().unwrap()["role"], expected_last_role, "{model}");
        assert_eq!(msgs.len(), input.as_array().unwrap().len(), "{model}");
    }
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
            { "type": "some_future_thing", "foo": "bar" }
        ]
    }));
    let err = parse(&req)
        .await
        .expect_err("unknown input item type must 400");
    match err {
        AppError::BadRequest(message) => {
            assert_eq!(
                message,
                "Responses input item type 'some_future_thing' is not supported"
            );
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
/// Property: namespace-flatten.
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

        /// Property: namespace-flatten.
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

fn format_req(model: &str, format: Value) -> ResponsesRequest {
    req_from(json!({ "model": model, "input": "hi", "text": { "format": format } }))
}

fn strict_capital_format() -> Value {
    json!({
        "type": "json_schema",
        "name": "capital",
        "strict": true,
        "schema": {
            "type": "object",
            "properties": { "capital": { "type": "string" } },
            "required": ["capital"],
            "additionalProperties": false
        }
    })
}

#[test]
fn text_format_json_schema_maps_to_output_config_with_strict_field() {
    let req = format_req("gpt-6.1-sol", strict_capital_format());
    let out = responses_output_format(&req, "global.openai.gpt-6.1-sol", &caps())
        .expect("supported")
        .expect("structured output requested");

    let js = &out.output_config["textFormat"]["structure"]["jsonSchema"];
    assert_eq!(out.output_config["textFormat"]["type"], "json_schema");
    assert_eq!(js["name"], "capital");
    let schema: Value = serde_json::from_str(js["schema"].as_str().expect("stringified")).unwrap();
    assert_eq!(schema["required"], json!(["capital"]));
    assert!(
        out.strict_field,
        "OpenAI-text-format model takes strict as a field"
    );
}

#[test]
fn text_format_nested_json_schema_wrapper_is_read() {
    let req = format_req(
        "gpt-6-sol",
        json!({ "type": "json_schema", "json_schema": {
            "name": "cap", "strict": true, "schema": { "type": "object", "properties": {} }
        }}),
    );
    let out = responses_output_format(&req, "global.openai.gpt-6-sol", &caps())
        .expect("supported")
        .expect("requested");
    assert_eq!(
        out.output_config["textFormat"]["structure"]["jsonSchema"]["name"],
        "cap"
    );
    assert!(out.strict_field);
}

#[test]
fn text_format_json_object_gets_default_name_and_no_strict() {
    let req = format_req("gpt-6.1-sol", json!({ "type": "json_object" }));
    let out = responses_output_format(&req, "global.openai.gpt-6.1-sol", &caps())
        .expect("supported")
        .expect("requested");
    assert_eq!(
        out.output_config["textFormat"]["structure"]["jsonSchema"]["name"],
        "response"
    );
    assert!(!out.strict_field);
}

#[test]
fn text_format_on_claude_keeps_chat_semantics() {
    // Supported Claude: outputConfig, but never the OpenAI strict field.
    let req = format_req(
        "global.anthropic.claude-sonnet-4-6",
        strict_capital_format(),
    );
    let out = responses_output_format(&req, "global.anthropic.claude-sonnet-4-6", &caps())
        .expect("supported")
        .expect("requested");
    assert!(!out.strict_field);

    // Claude 5 generation: forwarded like every other model, with the
    // default name and still no OpenAI strict field.
    let req = format_req("us.anthropic.claude-opus-5", strict_capital_format());
    let out = responses_output_format(&req, "us.anthropic.claude-opus-5", &caps())
        .expect("forwarded")
        .expect("requested");
    assert_eq!(out.output_config["textFormat"]["type"], "json_schema");
    assert!(!out.strict_field);
}

#[test]
fn text_format_is_forwarded_whatever_the_registry_says() {
    // The gateway no longer decides which models take structured output;
    // Bedrock answers for each model. A model the registry has no entry for,
    // or one Bedrock is known to refuse, still gets its outputConfig sent.
    for model in [
        "global.anthropic.claude-sonnet-5-5",
        "global.anthropic.claude-haiku-5-5",
        "global.anthropic.claude-fable-5",
        "us.amazon.nova-pro-v1:0",
        "acme.unlisted-model-v1:0",
    ] {
        let req = format_req(model, json!({ "type": "json_object" }));
        let out = responses_output_format(&req, model, &caps())
            .unwrap_or_else(|e| panic!("{model}: {e:?}"))
            .unwrap_or_else(|| panic!("{model}: not requested"));
        assert_eq!(
            out.output_config["textFormat"]["structure"]["jsonSchema"]["name"], "response",
            "{model}"
        );
    }
}

#[test]
fn text_format_text_or_absent_requests_nothing() {
    let c = caps();
    let req = format_req(
        "global.anthropic.claude-sonnet-5-5",
        json!({ "type": "text" }),
    );
    assert_eq!(
        responses_output_format(&req, "global.anthropic.claude-sonnet-5-5", &c).unwrap(),
        None
    );
    let req = req_from(json!({ "model": "global.anthropic.claude-sonnet-5-5", "input": "hi" }));
    assert_eq!(
        responses_output_format(&req, "global.anthropic.claude-sonnet-5-5", &c).unwrap(),
        None
    );
}

// ---------------------------------------------------------------------------
// Gateway-executed web search (plan C2): envelopes, carriers, replay
// ---------------------------------------------------------------------------

const WS_MODEL: &str = "global.anthropic.claude-sonnet-5-5";

fn signed(text: &str, signature: &str) -> Value {
    json!({ "reasoningContent": { "reasoningText": { "text": text, "signature": signature } } })
}

fn decode_payload(envelope: &str) -> Value {
    let encoded = envelope
        .strip_prefix(REASONING_ENVELOPE_PREFIX)
        .expect("gateway prefix");
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .expect("base64url");
    serde_json::from_slice(&bytes).expect("json payload")
}

fn ws_call(id: &str, query: &str) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": "web_search", "input": { "query": query } } })
}

fn ws_result(id: &str, payload: &str) -> Value {
    json!({ "toolResult": { "toolUseId": id, "content": [{ "text": payload }] } })
}

fn reasoning_item(id: &str, envelope: &str) -> Value {
    json!({ "type": "reasoning", "id": id, "summary": [], "encrypted_content": envelope })
}

async fn parse_with(req: &ResponsesRequest, opts: TranslateOptions) -> Result<Value, AppError> {
    let c = caps();
    let parsed = parse_responses_input(req, WS_MODEL, &resolver(false), opts).await?;
    Ok(finish_messages(&parsed, WS_MODEL, &c))
}

const DECLARED: TranslateOptions = TranslateOptions {
    tool_search: false,
    web_search: true,
};

/// The envelope of an undeclared request is today's string: same payload
/// bytes, same base64url, no `ws` key.
#[test]
fn unmarked_envelope_bytes_unchanged() {
    let content = [
        signed("Let me think.", "sig-1"),
        json!({ "reasoningContent": { "redactedContent": "cmVkYWN0ZWQ=" } }),
        json!({ "text": "not reasoning" }),
        json!({ "reasoningContent": { "reasoningText": { "text": "unsigned" } } }),
    ];
    let envelope = encode_reasoning_envelope(&content).expect("envelope");
    assert_eq!(
        envelope,
        "bedrock-reasoning-v1:eyJibG9ja3MiOlt7InJlYXNvbmluZ1RleHQiOnsic2lnbmF0dXJlIjoic2lnLTEiLCJ0\
         ZXh0IjoiTGV0IG1lIHRoaW5rLiJ9fSx7InJlZGFjdGVkQ29udGVudCI6ImNtVmtZV04wWldRPSJ9XSwidmVyc2lv\
         biI6MX0"
    );
    assert_eq!(
        String::from_utf8(
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(envelope.trim_start_matches(REASONING_ENVELOPE_PREFIX))
                .unwrap()
        )
        .unwrap(),
        r#"{"blocks":[{"reasoningText":{"signature":"sig-1","text":"Let me think."}},{"redactedContent":"cmVkYWN0ZWQ="}],"version":1}"#
    );
    assert_eq!(
        encode_reasoning_envelope_with(&content, None),
        Some(envelope)
    );
    assert_eq!(encode_reasoning_envelope(&[json!({ "text": "x" })]), None);
}

#[test]
fn marked_envelope_round_trip() {
    let content = [signed("Plan.", "sig-m")];
    let envelope = encode_reasoning_envelope_with(&content, Some(WEB_SEARCH_MARK)).unwrap();
    let payload = decode_payload(&envelope);
    assert_eq!(
        payload,
        json!({ "blocks": [content[0]["reasoningContent"]], "version": 1, "ws": 1 })
    );
    assert_eq!(
        decode_gateway_envelope(envelope.trim_start_matches(REASONING_ENVELOPE_PREFIX)).unwrap(),
        GatewayEnvelope::Reasoning {
            blocks: content.to_vec(),
            marked: true
        }
    );
    let plain = encode_reasoning_envelope(&content).unwrap();
    assert_eq!(
        decode_gateway_envelope(plain.trim_start_matches(REASONING_ENVELOPE_PREFIX)).unwrap(),
        GatewayEnvelope::Reasoning {
            blocks: content.to_vec(),
            marked: false
        }
    );
}

#[test]
fn unknown_marker_is_400() {
    for ws in [
        json!(2),
        json!(0),
        json!("1"),
        json!(1.0),
        json!(null),
        json!(true),
    ] {
        let payload =
            json!({ "blocks": [signed("t", "s")["reasoningContent"]], "version": 1, "ws": ws });
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        match decode_gateway_envelope(&encoded) {
            Err(AppError::BadRequest(message)) => {
                assert_eq!(message, "unsupported web search marker", "{ws}")
            }
            other => panic!("{ws}: {other:?}"),
        }
    }
}

#[test]
fn carrier_round_trip() {
    let uses = [
        ws_call("toolu_1", "aws news"),
        ws_call("toolu_2", "bedrock"),
    ];
    let results = [
        ws_result(
            "toolu_1",
            r#"{"query":"aws news","results":[],"source":"s","trust":"t","type":"web_search_results"}"#,
        ),
        ws_result(
            "toolu_2",
            r#"{"error_code":"too_many_requests","query":"bedrock","type":"web_search_error"}"#,
        ),
    ];
    let carrier = encode_web_search_carrier(&uses, &results, 3);
    assert!(is_gateway_reasoning_envelope(&carrier));
    let payload = decode_payload(&carrier);
    assert_eq!(
        payload,
        json!({ "tool_results": results, "tool_uses": uses, "used": 3, "version": 2, "ws": 1 })
    );
    // Keys are written in sorted order, so the bytes are deterministic.
    let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(carrier.trim_start_matches(REASONING_ENVELOPE_PREFIX))
        .unwrap();
    assert!(String::from_utf8(raw)
        .unwrap()
        .starts_with(r#"{"tool_results":[{"toolResult":{"content":[{"text":"#));
    assert_eq!(encode_web_search_carrier(&uses, &results, 3), carrier);
    assert_eq!(
        decode_gateway_envelope(carrier.trim_start_matches(REASONING_ENVELOPE_PREFIX)).unwrap(),
        GatewayEnvelope::WebSearchStep {
            tool_uses: uses.to_vec(),
            tool_results: results.to_vec(),
            used: 3
        }
    );
}

#[test]
fn malformed_carriers_are_400() {
    let good_use = ws_call("toolu_1", "q");
    let good_result = ws_result("toolu_1", "{\"type\":\"web_search_error\"}");
    let base = json!({
        "tool_results": [good_result], "tool_uses": [good_use], "used": 1, "version": 2, "ws": 1
    });
    let with = |pointer: &str, value: Value| {
        let mut payload = base.clone();
        *payload.pointer_mut(pointer).unwrap() = value;
        payload
    };
    let without = |key: &str| {
        let mut payload = base.clone();
        payload.as_object_mut().unwrap().remove(key);
        payload
    };
    let cases = [
        ("ws missing", without("ws")),
        ("ws not 1", with("/ws", json!(2))),
        ("tool_uses missing", without("tool_uses")),
        ("tool_uses empty", with("/tool_uses", json!([])).clone()),
        ("tool_results missing", without("tool_results")),
        (
            "length mismatch",
            with("/tool_results", json!([good_result, good_result])),
        ),
        (
            "not a toolUse",
            with("/tool_uses/0", json!({ "text": "x" })),
        ),
        (
            "wrong name",
            with("/tool_uses/0/toolUse/name", json!("lookup")),
        ),
        (
            "input not an object",
            with("/tool_uses/0/toolUse/input", json!("q")),
        ),
        (
            "empty id",
            with("/tool_uses/0/toolUse/toolUseId", json!("")),
        ),
        (
            "id not a string",
            with("/tool_uses/0/toolUse/toolUseId", json!(7)),
        ),
        (
            "result id differs",
            with("/tool_results/0/toolResult/toolUseId", json!("toolu_2")),
        ),
        (
            "not a toolResult",
            with("/tool_results/0", json!({ "text": "x" })),
        ),
        (
            "two blocks",
            with(
                "/tool_results/0/toolResult/content",
                json!([{ "text": "a" }, { "text": "b" }]),
            ),
        ),
        (
            "no blocks",
            with("/tool_results/0/toolResult/content", json!([])),
        ),
        (
            "empty text",
            with("/tool_results/0/toolResult/content/0/text", json!("")),
        ),
        (
            "not a text block",
            with(
                "/tool_results/0/toolResult/content/0",
                json!({ "json": {} }),
            ),
        ),
        ("used negative", with("/used", json!(-1))),
        ("used not an integer", with("/used", json!("1"))),
        (
            "used too large",
            with("/used", json!(u64::from(u32::MAX) + 1)),
        ),
        ("used missing", without("used")),
    ];
    // The base itself decodes.
    let encode = |payload: &Value| {
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(payload).unwrap())
    };
    assert!(decode_gateway_envelope(&encode(&base)).is_ok());
    for (name, payload) in cases {
        match decode_gateway_envelope(&encode(&payload)) {
            Err(AppError::BadRequest(message)) => {
                assert_eq!(message, "malformed web search carrier", "{name}")
            }
            other => panic!("{name}: {other:?}"),
        }
    }
}

/// A declared request drops replayed gateway reasoning without the mark and
/// keeps marked reasoning; an undeclared request keeps both. Another
/// provider's reasoning is skipped either way.
#[tokio::test]
async fn declared_drops_unmarked_reasoning_only() {
    let unmarked = encode_reasoning_envelope(&[signed("old", "sig-old")]).unwrap();
    let marked =
        encode_reasoning_envelope_with(&[signed("new", "sig-new")], Some(WEB_SEARCH_MARK)).unwrap();
    let req = req_from(json!({
        "model": WS_MODEL,
        "input": [
            { "type": "message", "role": "user", "content": "q1" },
            reasoning_item("rs_old", &unmarked),
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "a1" }] },
            { "type": "message", "role": "user", "content": "q2" },
            reasoning_item("rs_new", &marked),
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "a2" }] },
            { "type": "reasoning", "id": "rs_gpt", "summary": [], "encrypted_content": "gAAAA==" },
            { "type": "message", "role": "user", "content": "q3" }
        ]
    }));

    let declared = parse_with(&req, DECLARED).await.unwrap();
    assert_eq!(
        declared,
        json!([
            { "role": "user", "content": [{ "text": "q1" }] },
            { "role": "assistant", "content": [{ "text": "a1" }] },
            { "role": "user", "content": [{ "text": "q2" }] },
            { "role": "assistant", "content": [signed("new", "sig-new"), { "text": "a2" }] },
            { "role": "user", "content": [{ "text": "q3" }] }
        ])
    );

    let undeclared = parse_with(&req, TranslateOptions::default()).await.unwrap();
    assert_eq!(
        undeclared,
        json!([
            { "role": "user", "content": [{ "text": "q1" }] },
            { "role": "assistant", "content": [signed("old", "sig-old"), { "text": "a1" }] },
            { "role": "user", "content": [{ "text": "q2" }] },
            { "role": "assistant", "content": [signed("new", "sig-new"), { "text": "a2" }] },
            { "role": "user", "content": [{ "text": "q3" }] }
        ])
    );
}

/// A search step replays as one assistant message (`[reasoning, text,
/// toolUse…]`) and one user message of results; the gateway's own
/// `web_search_call` items add nothing.
#[tokio::test]
async fn search_step_replays_as_one_assistant_message() {
    let reasoning =
        encode_reasoning_envelope_with(&[signed("search first", "sig-0")], Some(WEB_SEARCH_MARK))
            .unwrap();
    let carrier = encode_web_search_carrier(
        &[ws_call("toolu_a", "one"), ws_call("toolu_b", "two")],
        &[
            ws_result("toolu_a", "{\"r\":1}"),
            ws_result("toolu_b", "{\"r\":2}"),
        ],
        2,
    );
    let req = req_from(json!({
        "model": WS_MODEL,
        "input": [
            { "type": "message", "role": "user", "content": "q" },
            reasoning_item("rs_S", &reasoning),
            { "type": "message", "id": "msg_S", "role": "assistant",
              "content": [{ "type": "output_text", "text": "Searching." }] },
            { "type": "web_search_call", "id": "ws_bgw_toolu_a", "status": "completed",
              "action": { "type": "search", "query": "one", "queries": ["one"] } },
            { "type": "web_search_call", "id": "ws_bgw_toolu_b", "status": "failed",
              "action": { "type": "search", "query": "two", "queries": ["two"] } },
            reasoning_item("rs_S_ws0", &carrier),
        ]
    }));
    let opts = DECLARED;
    let parsed = parse_responses_input(&req, WS_MODEL, &resolver(false), opts)
        .await
        .unwrap();
    assert_eq!(
        parsed.carrier_ids,
        ["toolu_a".to_string(), "toolu_b".to_string()]
            .into_iter()
            .collect::<HashSet<_>>()
    );
    assert_eq!(
        finish_messages(&parsed, WS_MODEL, &caps()),
        json!([
            { "role": "user", "content": [{ "text": "q" }] },
            { "role": "assistant", "content": [
                signed("search first", "sig-0"),
                { "text": "Searching." },
                ws_call("toolu_a", "one"),
                ws_call("toolu_b", "two")
            ] },
            { "role": "user", "content": [
                ws_result("toolu_a", "{\"r\":1}"),
                ws_result("toolu_b", "{\"r\":2}")
            ] }
        ])
    );
}

/// A mixed step: the carrier's results come first in the input, the client's
/// result after them, and the user message is put in the assistant message's
/// call order (client call first).
#[tokio::test]
async fn mixed_step_tool_results_follow_tool_use_order() {
    let reasoning =
        encode_reasoning_envelope_with(&[signed("both", "sig-m")], Some(WEB_SEARCH_MARK)).unwrap();
    let carrier = encode_web_search_carrier(
        &[ws_call("toolu_ws", "news")],
        &[ws_result("toolu_ws", "{\"r\":1}")],
        1,
    );
    let req = req_from(json!({
        "model": WS_MODEL,
        "tools": [{ "type": "function", "name": "lookup_ticket", "parameters": { "type": "object" } }],
        "input": [
            { "type": "message", "role": "user", "content": "q" },
            reasoning_item("rs_S", &reasoning),
            { "type": "web_search_call", "id": "ws_bgw_toolu_ws", "status": "completed",
              "action": { "type": "search", "query": "news", "queries": ["news"] } },
            { "type": "function_call", "id": "fc_toolu_fn", "call_id": "toolu_fn",
              "name": "lookup_ticket", "arguments": "{\"id\":\"T-1\"}" },
            reasoning_item("rs_S_ws0", &carrier),
            { "type": "function_call_output", "call_id": "toolu_fn", "output": "ticket T-1" }
        ]
    }));
    let messages = parse_with(&req, DECLARED).await.unwrap();
    assert_eq!(
        messages,
        json!([
            { "role": "user", "content": [{ "text": "q" }] },
            { "role": "assistant", "content": [
                signed("both", "sig-m"),
                { "toolUse": { "toolUseId": "toolu_fn", "name": "lookup_ticket", "input": { "id": "T-1" } } },
                ws_call("toolu_ws", "news")
            ] },
            { "role": "user", "content": [
                { "toolResult": { "toolUseId": "toolu_fn", "content": [{ "text": "ticket T-1" }] } },
                ws_result("toolu_ws", "{\"r\":1}")
            ] }
        ])
    );
}

/// Without carriers no user message is reordered, even when the client sent
/// parallel results in a different order than the calls.
#[tokio::test]
async fn alignment_never_touches_histories_without_carriers() {
    let input = json!([
        { "type": "message", "role": "user", "content": "q" },
        { "type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}" },
        { "type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}" },
        { "type": "function_call_output", "call_id": "c2", "output": "two" },
        { "type": "function_call_output", "call_id": "c1", "output": "one" }
    ]);
    let req = req_from(json!({ "model": WS_MODEL, "input": input }));
    for opts in [TranslateOptions::default(), DECLARED] {
        let messages = parse_with(&req, opts).await.unwrap();
        assert_eq!(messages[2]["content"][0]["toolResult"]["toolUseId"], "c2");
        assert_eq!(messages[2]["content"][1]["toolResult"]["toolUseId"], "c1");
    }
    let mut messages = parse_with(&req, DECLARED).await.unwrap();
    let before = messages.clone();
    align_carrier_tool_results(&mut messages, &HashSet::new());
    assert_eq!(messages, before);
    align_carrier_tool_results(&mut messages, &["other".to_string()].into_iter().collect());
    assert_eq!(messages, before);
}

fn tool_names(specs: &[Value]) -> Vec<&str> {
    specs
        .iter()
        .map(|spec| spec["toolSpec"]["name"].as_str().unwrap())
        .collect()
}

/// The synthetic tool is appended after every client tool (the tool-search
/// tool included) and only when declared; `tools: None` declares it alone.
#[test]
fn synthetic_tool_last_and_only_when_declared() {
    let req = req_from(json!({
        "model": WS_MODEL,
        "input": "hi",
        "tools": [
            { "type": "function", "name": "get_goal", "parameters": { "type": "object" } },
            { "type": "web_search", "external_web_access": false },
            { "type": "namespace", "name": "mcp__ze", "description": "d",
              "tools": [{ "type": "function", "name": "alpha" }] },
            { "type": "tool_search", "execution": "client", "description": "search tools",
              "parameters": { "type": "object", "properties": {} } }
        ]
    }));
    let (specs, registry) = build_responses_tools_with(&req, true, true).unwrap();
    assert_eq!(
        tool_names(&specs),
        vec!["get_goal", "mcp__ze__alpha", "tool_search", "web_search"]
    );
    assert_eq!(
        specs[3],
        json!({ "toolSpec": {
            "name": "web_search",
            "description": WEB_SEARCH_TOOL_DESCRIPTION,
            "inputSchema": { "json": web_search_tool_schema() }
        } })
    );
    assert_eq!(
        registry.resolve("web_search").unwrap().kind,
        ResponsesToolKind::WebSearch
    );

    let (specs, _) = build_responses_tools_with(&req, false, true).unwrap();
    assert_eq!(
        tool_names(&specs),
        vec!["get_goal", "mcp__ze__alpha", "web_search"]
    );

    let (specs, registry) = build_responses_tools_with(&req, true, false).unwrap();
    assert_eq!(
        tool_names(&specs),
        vec!["get_goal", "mcp__ze__alpha", "tool_search"]
    );
    assert!(registry.resolve("web_search").is_none());

    let no_tools = req_from(json!({ "model": WS_MODEL, "input": "hi" }));
    let (specs, _) = build_responses_tools_with(&no_tools, false, true).unwrap();
    assert_eq!(tool_names(&specs), vec!["web_search"]);
    let (specs, registry) = build_responses_tools_with(&no_tools, false, false).unwrap();
    assert!(specs.is_empty());
    assert!(registry.resolve("web_search").is_none());
}

/// A client tool named `web_search` is a 400 only in a declared request;
/// otherwise it is an ordinary client function.
#[test]
fn client_web_search_name_rejected_only_when_declared() {
    let req = req_from(json!({
        "model": WS_MODEL,
        "input": "hi",
        "tools": [{ "type": "function", "name": "web_search", "parameters": { "type": "object" } }]
    }));
    let err = build_responses_tools_with(&req, false, true).unwrap_err();
    assert!(matches!(&err, AppError::BadRequest(_)));
    assert_eq!(
        err.to_string(),
        "bad request: tool name 'web_search' is reserved for the gateway's web search on this \
         request (X-Weniq-Web-Search)"
    );
    let (specs, registry) = build_responses_tools_with(&req, false, false).unwrap();
    assert_eq!(tool_names(&specs), vec!["web_search"]);
    assert_eq!(
        registry.resolve("web_search").unwrap().kind,
        ResponsesToolKind::Function
    );
}

/// The split into parse + finish is the old translation: on every existing
/// shape (text, tools, reasoning, prefill, images) both give the same value.
#[tokio::test]
async fn parse_then_finish_equals_existing_translation() {
    let unmarked = encode_reasoning_envelope(&[signed("t", "s")]).unwrap();
    let requests = [
        json!({ "model": WS_MODEL, "input": "plain" }),
        json!({ "model": WS_MODEL, "instructions": "be brief", "input": [
            { "type": "message", "role": "developer", "content": "dev" },
            { "type": "message", "role": "user", "content": [
                { "type": "input_text", "text": "look" },
                { "type": "input_image", "image_url": "https://example.com/a.jpg" }
            ] },
            reasoning_item("rs_1", &unmarked),
            { "type": "function_call", "call_id": "c", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c", "output": "ok" },
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "done" }] }
        ] }),
    ];
    let c = caps();
    let r = resolver(true);
    for raw in requests {
        let req = req_from(raw);
        for tool_search in [false, true] {
            let old = to_responses_converse_input_with(&req, WS_MODEL, &r, &c, tool_search)
                .await
                .unwrap();
            let opts = TranslateOptions {
                tool_search,
                web_search: false,
            };
            let parsed = parse_responses_input(&req, WS_MODEL, &r, opts)
                .await
                .unwrap();
            assert_eq!(finish_messages(&parsed, WS_MODEL, &c), old.messages);
            assert_eq!(Value::Array(parsed.system), old.system);
        }
    }
}

/// The history scan: marks, carrier payloads in order, and the answer's
/// search count, which a new user message resets.
#[test]
fn carrier_used_round_trips_through_the_history_scan() {
    let marked =
        encode_reasoning_envelope_with(&[signed("m", "s")], Some(WEB_SEARCH_MARK)).unwrap();
    let carrier_1 =
        encode_web_search_carrier(&[ws_call("t1", "a")], &[ws_result("t1", "{\"p\":1}")], 2);
    let carrier_2 = encode_web_search_carrier(
        &[ws_call("t2", "b"), ws_call("t3", "c")],
        &[ws_result("t2", "{\"p\":2}"), ws_result("t3", "{\"p\":3}")],
        3,
    );
    let items = |tail: Vec<Value>| {
        let mut input = vec![
            json!({ "type": "message", "role": "user", "content": "q" }),
            reasoning_item("rs_1", &carrier_1),
            json!({ "type": "function_call_output", "call_id": "x", "output": "y" }),
            reasoning_item("rs_2", &carrier_2),
        ];
        input.extend(tail);
        req_from(json!({ "model": WS_MODEL, "input": input }))
    };
    let scan = scan_web_search_history(&items(vec![json!({
        "type": "function_call_output", "call_id": "z", "output": "w"
    })]))
    .unwrap();
    assert!(scan.marked);
    assert_eq!(
        scan.carrier_payloads,
        vec!["{\"p\":1}", "{\"p\":2}", "{\"p\":3}"]
    );
    assert_eq!(scan.used_in_answer, 3);

    let scan = scan_web_search_history(&items(vec![
        json!({ "type": "message", "role": "user", "content": "next question" }),
    ]))
    .unwrap();
    assert_eq!(scan.used_in_answer, 0);
    assert!(scan.marked);

    let only_marked = req_from(json!({ "model": WS_MODEL, "input": [
        { "type": "message", "role": "user", "content": "q" },
        reasoning_item("rs_1", &marked)
    ] }));
    let scan = scan_web_search_history(&only_marked).unwrap();
    assert_eq!(
        scan,
        HistoryScan {
            marked: true,
            carrier_payloads: Vec::new(),
            used_in_answer: 0,
            unmarked_tool_loop: false,
        }
    );

    let unmarked_only = req_from(json!({ "model": WS_MODEL, "input": [
        reasoning_item("rs_1", &encode_reasoning_envelope(&[signed("m", "s")]).unwrap()),
        { "type": "reasoning", "summary": [], "encrypted_content": "gAAAA==" }
    ] }));
    assert_eq!(
        scan_web_search_history(&unmarked_only).unwrap(),
        HistoryScan::default()
    );
    assert_eq!(
        scan_web_search_history(&req_from(json!({ "model": WS_MODEL, "input": "hi" }))).unwrap(),
        HistoryScan::default()
    );

    let broken = req_from(json!({ "model": WS_MODEL, "input": [
        reasoning_item("rs_1", "bedrock-reasoning-v1:!!")
    ] }));
    assert!(matches!(
        scan_web_search_history(&broken),
        Err(AppError::BadRequest(_))
    ));
}

/// The scan knows whether the request continues a tool loop that holds
/// gateway thinking without the mark: the last input item that becomes a
/// Bedrock turn is a tool output, and unmarked gateway reasoning comes after
/// the last user message. System and developer messages, skipped items and
/// gateway reasoning do not end the loop.
#[test]
fn history_scan_sees_an_unmarked_tool_loop() {
    let unmarked = encode_reasoning_envelope(&[signed("m", "s")]).unwrap();
    let thinking = reasoning_item("rs_0", &unmarked);
    let in_loop = |tail: Vec<Value>| {
        let mut input = vec![json!({ "type": "message", "role": "user", "content": "q" })];
        input.extend(tail);
        scan_web_search_history(&req_from(json!({ "model": WS_MODEL, "input": input })))
            .unwrap()
            .unmarked_tool_loop
    };
    let call = json!({ "type": "function_call", "call_id": "c", "name": "f", "arguments": "{}" });
    let output = json!({ "type": "function_call_output", "call_id": "c", "output": "o" });
    assert!(!in_loop(Vec::new()));
    assert!(in_loop(vec![
        thinking.clone(),
        call.clone(),
        output.clone()
    ]));
    assert!(in_loop(vec![
        thinking.clone(),
        call.clone(),
        output.clone(),
        json!({ "type": "message", "role": "developer", "content": "note" }),
        json!({ "type": "web_search_call", "id": "ws_1", "status": "completed",
                "action": { "type": "search", "query": "q" } }),
        reasoning_item("rs_1", &unmarked),
        json!({ "type": "item_reference", "id": "x" }),
    ]));
    for (kind, output_kind) in [
        ("custom_tool_call", "custom_tool_call_output"),
        ("shell_call", "shell_call_output"),
        ("apply_patch_call", "apply_patch_call_output"),
        ("local_shell_call", "local_shell_call_output"),
        ("tool_search_call", "tool_search_output"),
    ] {
        assert!(
            in_loop(vec![
                thinking.clone(),
                json!({ "type": kind, "call_id": "c" }),
                json!({ "type": output_kind, "call_id": "c" }),
            ]),
            "{output_kind}"
        );
        assert!(
            !in_loop(vec![
                thinking.clone(),
                json!({ "type": kind, "call_id": "c" })
            ]),
            "{kind}"
        );
    }
    assert!(!in_loop(vec![thinking.clone(), call.clone()]));
    assert!(!in_loop(vec![
        thinking.clone(),
        call.clone(),
        output.clone(),
        json!({ "type": "message", "role": "assistant",
                "content": [{ "type": "output_text", "text": "done" }] }),
    ]));
    assert!(!in_loop(vec![
        thinking.clone(),
        output.clone(),
        json!({ "type": "message", "role": "user", "content": "next" }),
    ]));
    // A loop without gateway thinking loses nothing to a first declaration,
    // and neither does one whose unmarked thinking precedes the last user
    // message.
    assert!(!in_loop(vec![call.clone(), output.clone()]));
    assert!(!in_loop(vec![
        thinking,
        json!({ "type": "message", "role": "user", "content": "next" }),
        call,
        output,
    ]));
}

/// A gateway `web_search_call` after the newest carrier (a step whose
/// carrier never reached the client) counts toward the answer's searches;
/// one before a carrier is covered by that carrier's `used`, and a user
/// message starts a new answer. Other providers' search items do not count.
#[test]
fn history_scan_counts_searches_without_a_carrier() {
    let carrier = encode_web_search_carrier(&[ws_call("t1", "a")], &[ws_result("t1", "{}")], 3);
    let call = |id: &str| {
        json!({ "type": "web_search_call", "id": id, "status": "completed",
                "action": { "type": "search", "query": "q" } })
    };
    let used = |tail: Vec<Value>| {
        let mut input = vec![json!({ "type": "message", "role": "user", "content": "q" })];
        input.extend(tail);
        scan_web_search_history(&req_from(json!({ "model": WS_MODEL, "input": input })))
            .unwrap()
            .used_in_answer
    };
    assert_eq!(used(vec![call("ws_bgw_a"), call("ws_bgw_b")]), 2);
    assert_eq!(
        used(vec![call("ws_bgw_t1"), reasoning_item("rs_1", &carrier)]),
        3
    );
    assert_eq!(
        used(vec![
            call("ws_bgw_t1"),
            reasoning_item("rs_1", &carrier),
            call("ws_bgw_x")
        ]),
        4
    );
    assert_eq!(used(vec![call("ws_67f0"), call("ws_bgw_a")]), 1);
    assert_eq!(
        used(vec![
            reasoning_item("rs_1", &carrier),
            call("ws_bgw_x"),
            json!({ "type": "message", "role": "user", "content": "next" }),
        ]),
        0
    );
}

/// Pinning the no-prefill continuation turn into the parsed input changes
/// nothing about the messages it finishes into; it only keeps the turn in
/// place once more turns follow. Models that accept prefill, and inputs that
/// end with a user turn, get no turn.
#[tokio::test]
async fn pinned_continuation_turn_finishes_into_the_same_messages() {
    let c = caps();
    let marked =
        encode_reasoning_envelope_with(&[signed("m", "s")], Some(WEB_SEARCH_MARK)).unwrap();
    let ending_on_assistant = req_from(json!({ "model": WS_MODEL, "input": [
        { "type": "message", "role": "user", "content": "q" },
        reasoning_item("rs_1", &marked)
    ] }));
    let parsed = parse_responses_input(&ending_on_assistant, WS_MODEL, &resolver(false), DECLARED)
        .await
        .unwrap();
    let mut pinned = parsed.clone();
    pinned.pin_no_prefill_continuation(WS_MODEL, &c);
    assert_eq!(pinned.turns.len(), parsed.turns.len() + 1);
    assert_eq!(
        finish_messages(&pinned, WS_MODEL, &c),
        finish_messages(&parsed, WS_MODEL, &c)
    );
    // A model that accepts prefill gets no turn.
    let prefill_model = "anthropic.claude-3-sonnet-v1:0";
    let mut unchanged = parsed.clone();
    unchanged.pin_no_prefill_continuation(prefill_model, &c);
    assert_eq!(unchanged, parsed);
    // An input that ends with a user turn gets no turn.
    let ending_on_user = req_from(json!({ "model": WS_MODEL, "input": "hi" }));
    let parsed = parse_responses_input(&ending_on_user, WS_MODEL, &resolver(false), DECLARED)
        .await
        .unwrap();
    let mut unchanged = parsed.clone();
    unchanged.pin_no_prefill_continuation(WS_MODEL, &c);
    assert_eq!(unchanged, parsed);
}

mod carrier_props {
    use super::*;
    use proptest::prelude::*;

    fn text() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 _\\-\\u{4e00}-\\u{4e10}\"\\\\{}]{1,40}"
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Any carrier the gateway writes decodes to the same blocks and count.
        #[test]
        fn carrier_codec_round_trips(
            calls in prop::collection::vec(("[a-zA-Z0-9_]{1,24}", text(), text()), 1..4),
            used in any::<u32>(),
        ) {
            let mut seen = HashSet::new();
            let calls: Vec<_> = calls.into_iter().filter(|(id, _, _)| seen.insert(id.clone())).collect();
            let uses: Vec<Value> = calls
                .iter()
                .map(|(id, query, _)| web_search_tool_use_block(id, &json!({ "query": query })))
                .collect();
            let results: Vec<Value> = calls
                .iter()
                .map(|(id, _, payload)| tool_result_block(id, payload))
                .collect();
            let carrier = encode_web_search_carrier(&uses, &results, used);
            let decoded = decode_gateway_envelope(
                carrier.strip_prefix(REASONING_ENVELOPE_PREFIX).unwrap(),
            );
            prop_assert_eq!(
                decoded.unwrap(),
                GatewayEnvelope::WebSearchStep { tool_uses: uses, tool_results: results, used }
            );
        }
    }
}
