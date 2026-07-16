//! Unit tests for the non-streaming Converse → Responses mapper.
//!
//! Relocated out of `responses_response.rs` into a sibling `#[path]` module for
//! code organization (see the `test-coverage-codecov` spec). Behavior is
//! unchanged; `use super::*;` still resolves to the implementation module.

use super::*;
use crate::openai::responses_schema::ResponsesInput;
use std::collections::HashMap;

fn req() -> ResponsesRequest {
    ResponsesRequest {
        model: "incoming".to_string(),
        input: ResponsesInput::Text("hi".to_string()),
        instructions: Some("be terse".to_string()),
        tools: None,
        tool_choice: None,
        temperature: Some(0.5),
        top_p: Some(0.9),
        max_output_tokens: Some(256),
        stream: None,
        reasoning: None,
        text: None,
        include: None,
        metadata: None,
        parallel_tool_calls: Some(true),
        store: None,
        previous_response_id: None,
        extra: HashMap::new(),
    }
}

#[test]
fn custom_and_client_shell_tools_preserve_responses_item_types() {
    let cases = [
        (
            json!({"type":"custom","name":"bash"}),
            "bash",
            json!({"input":"pwd"}),
            "custom_tool_call",
            "input",
        ),
        (
            json!({"type":"local_shell"}),
            "local_shell",
            json!({"action":{"type":"exec","command":["pwd"]}}),
            "local_shell_call",
            "action",
        ),
        (
            json!({"type":"shell"}),
            "shell",
            json!({"commands":["pwd"]}),
            "shell_call",
            "action",
        ),
        (
            json!({"type":"apply_patch"}),
            "apply_patch",
            json!({"operation":{"type":"create_file","path":"a"}}),
            "apply_patch_call",
            "operation",
        ),
    ];
    for (tool, bedrock_name, input, expected_type, payload_key) in cases {
        let mut request = req();
        request.tools = Some(vec![serde_json::from_value(tool).expect("tool")]);
        let (_, registry) = crate::bedrock::responses_translate::build_responses_tools(&request)
            .expect("build tools");
        let output = json!({
            "output": { "message": { "role": "assistant", "content": [{
                "toolUse": { "toolUseId": "call_1", "name": bedrock_name, "input": input }
            }] } },
            "stopReason": "tool_use",
            "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
        });
        let response = from_converse_output_to_responses_with_tools(
            &output,
            &request,
            "m",
            "resp_tools",
            &registry,
        )
        .expect("map");
        match &response.output[0] {
            ResponseOutputItem::Other { item_type, fields } => {
                assert_eq!(item_type, expected_type);
                assert_eq!(fields["call_id"], "call_1");
                assert!(fields.contains_key(payload_key));
                if matches!(expected_type, "shell_call" | "apply_patch_call") {
                    assert_eq!(fields["status"], "completed");
                } else {
                    assert!(!fields.contains_key("status"));
                }
                if expected_type != "custom_tool_call" {
                    assert!(!fields.contains_key("name"));
                }
            }
            other => panic!("expected typed client tool item, got {other:?}"),
        }
    }
}

#[test]
fn signed_reasoning_emits_replayable_encrypted_content() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [{
            "reasoningContent": { "reasoningText": { "text": "think", "signature": "sig" } }
        }] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
    });
    let response = from_converse_output_to_responses(&output, &req(), "m", "resp_reason")
        .expect("map reasoning");
    match &response.output[0] {
        ResponseOutputItem::Reasoning {
            encrypted_content, ..
        } => {
            assert!(encrypted_content
                .as_deref()
                .is_some_and(|value| value.starts_with("bedrock-reasoning-v1:")));
        }
        other => panic!("expected reasoning, got {other:?}"),
    }
}

// -- Test 1: text output → message item with output_text; no <think> ------

#[test]
fn text_output_becomes_message_with_output_text() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [{ "text": "Hello" }] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 8, "outputTokens": 2, "totalTokens": 10 }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_abc").expect("map text");

    assert_eq!(resp.id, "resp_abc");
    assert_eq!(resp.object, "response");
    assert_eq!(resp.status, "completed");
    assert_eq!(resp.model, "m");
    // Exactly one message item with one output_text part.
    assert_eq!(resp.output.len(), 1);
    match &resp.output[0] {
        ResponseOutputItem::Message { role, content, .. } => {
            assert_eq!(role, "assistant");
            assert_eq!(content.len(), 1);
            match &content[0] {
                OutputContentPart::OutputText { text, .. } => assert_eq!(text, "Hello"),
                other => panic!("expected output_text part, got {other:?}"),
            }
        }
        other => panic!("expected message item, got {other:?}"),
    }

    // Echoed request params.
    assert_eq!(resp.instructions.as_deref(), Some("be terse"));
    assert_eq!(resp.temperature, Some(0.5));
    assert_eq!(resp.top_p, Some(0.9));
    assert_eq!(resp.max_output_tokens, Some(256));
    assert_eq!(resp.parallel_tool_calls, Some(true));

    // No <think>, no top-level output_text key, id matches ^resp_.
    let v = serde_json::to_value(&resp).expect("serialize");
    let s = serde_json::to_string(&resp).expect("string");
    assert!(!s.contains("<think>"), "<think> leaked: {s}");
    assert!(
        v.get("output_text").is_none(),
        "output_text wire key present"
    );
    assert!(resp.id.starts_with("resp_"));
}

/// REGRESSION (@ai-sdk/openai non-stream `generateText`): `output_text`
/// MUST serialize `annotations` as a present array (`[]` when empty).
/// ai-sdk declares `annotations: z.array(...)` without `.nullish()`, so an
/// absent field fails validation ("Invalid JSON response").
#[test]
fn output_text_serializes_annotations_as_empty_array() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [{ "text": "pong" }] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_ann").expect("map text");
    let v = serde_json::to_value(&resp).expect("serialize");

    let part = &v["output"][0]["content"][0];
    assert_eq!(part["type"], "output_text");
    assert_eq!(part["text"], "pong");
    assert!(
        part.get("annotations").is_some(),
        "annotations key absent — ai-sdk would reject: {part}"
    );
    assert_eq!(
        part["annotations"],
        json!([]),
        "annotations must serialize as an empty array: {part}"
    );
    assert!(
        part["annotations"].is_array(),
        "annotations must be an array"
    );
}

// -- Test 2: tool_use → function_call item with call_id/name/arguments ----

#[test]
fn tool_use_output_becomes_function_call_items() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "toolUse": {
                "toolUseId": "call-1",
                "name": "get_weather",
                "input": { "city": "Paris", "unit": "c" }
            }}
        ] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 12, "outputTokens": 6, "totalTokens": 18 }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_x").expect("map tool_use");

    assert_eq!(resp.output.len(), 1);
    match &resp.output[0] {
        ResponseOutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(call_id, "call-1");
            assert_eq!(name, "get_weather");
            let args: Value = serde_json::from_str(arguments).expect("args json");
            assert_eq!(args, json!({ "city": "Paris", "unit": "c" }));
        }
        other => panic!("expected function_call item, got {other:?}"),
    }
}

// -- Test 3: reasoning present → reasoning item FIRST, then message --------

#[test]
fn reasoning_emits_reasoning_item_first_then_message() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "reasoningContent": { "reasoningText": { "text": "think step by step" } } },
            { "text": "The answer is 4." }
        ] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 10, "outputTokens": 5, "totalTokens": 15 }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_r").expect("map reasoning");

    // reasoning FIRST, then the message.
    assert_eq!(resp.output.len(), 2);
    match &resp.output[0] {
        ResponseOutputItem::Reasoning { summary, .. } => {
            let summary = summary.as_ref().expect("summary present");
            assert_eq!(summary[0]["text"], "think step by step");
        }
        other => panic!("expected reasoning item first, got {other:?}"),
    }
    match &resp.output[1] {
        ResponseOutputItem::Message { content, .. } => match &content[0] {
            OutputContentPart::OutputText { text, .. } => {
                assert_eq!(text, "The answer is 4.");
                // Reasoning is NOT rendered inline as <think>.
                assert!(!text.contains("<think>"));
            }
            other => panic!("expected output_text part, got {other:?}"),
        },
        other => panic!("expected message item second, got {other:?}"),
    }

    // No <think> anywhere on the wire.
    let s = serde_json::to_string(&resp).expect("serialize");
    assert!(!s.contains("<think>"), "<think> leaked: {s}");
}

// -- Test 4: usage via shared helper → Responses field names --------------

#[test]
fn usage_uses_shared_helper_and_responses_field_names() {
    // input=9, cacheRead=4, cacheWrite=7, output=5.
    // Shared helper: prompt = 9+4+7 = 20; total = 20+5 = 25; cached = 4.
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": { "reasoningText": { "text": "deliberation" } } },
            { "text": "ok" }
        ] } },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 9,
            "outputTokens": 5,
            "totalTokens": 20,
            "cacheReadInputTokens": 4,
            "cacheWriteInputTokens": 7
        }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_u").expect("map usage");

    // Matches compute_token_usage exactly.
    let expected = compute_token_usage(9, 5, 4, 7);
    assert_eq!(resp.usage.input_tokens, expected.prompt_tokens);
    assert_eq!(resp.usage.output_tokens, expected.completion_tokens);
    assert_eq!(resp.usage.total_tokens, expected.total_tokens);
    assert_eq!(
        resp.usage
            .input_tokens_details
            .as_ref()
            .expect("input details")
            .cached_tokens,
        expected.cached_tokens
    );
    // reasoning_tokens = estimate of the reasoning text.
    let rt = resp
        .usage
        .output_tokens_details
        .as_ref()
        .expect("output details")
        .reasoning_tokens;
    assert_eq!(rt, estimate_reasoning_tokens("deliberation") as i32);
    assert!(rt > 0);

    // Responses field names on the wire — NOT chat names.
    let v = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(v["usage"]["input_tokens"], expected.prompt_tokens);
    assert_eq!(v["usage"]["output_tokens"], expected.completion_tokens);
    assert_eq!(v["usage"]["total_tokens"], expected.total_tokens);
    assert!(v["usage"].get("prompt_tokens").is_none());
    assert!(v["usage"].get("completion_tokens").is_none());
    assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 4);
    assert!(v["usage"]["output_tokens_details"]["reasoning_tokens"]
        .as_i64()
        .is_some());
}

// -- status mapping for incomplete stop reasons ---------------------------

#[test]
fn max_tokens_maps_to_incomplete_max_output_tokens() {
    let output = json!({
        "output": { "message": { "content": [{ "text": "truncated" }] } },
        "stopReason": "max_tokens",
        "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
    });
    let resp = from_converse_output_to_responses(&output, &req(), "m", "resp_t").expect("map");
    assert_eq!(resp.status, "incomplete");
    assert_eq!(
        resp.incomplete_details.expect("details")["reason"],
        "max_output_tokens"
    );
}

#[test]
fn max_tokens_with_tooluse_emits_function_call_item() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "toolUse": {
                "toolUseId": "call-mt",
                "name": "get_weather",
                "input": { "city": "Paris" }
            }}
        ] } },
        "stopReason": "max_tokens",
        "usage": { "inputTokens": 12, "outputTokens": 6, "totalTokens": 18 }
    });
    let resp = from_converse_output_to_responses(&output, &req(), "m", "resp_mt").expect("map");

    let fc = resp
        .output
        .iter()
        .find(|item| matches!(item, ResponseOutputItem::FunctionCall { .. }))
        .expect("a function_call output item on a max_tokens truncation");
    match fc {
        ResponseOutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(call_id, "call-mt");
            assert_eq!(name, "get_weather");
            let args: Value = serde_json::from_str(arguments).expect("args json");
            assert_eq!(args, json!({ "city": "Paris" }));
        }
        other => panic!("expected function_call item, got {other:?}"),
    }
    // status/incomplete_details for max_tokens unchanged.
    assert_eq!(resp.status, "incomplete");
    assert_eq!(
        resp.incomplete_details.expect("details")["reason"],
        "max_output_tokens"
    );
}

#[test]
fn content_filtered_maps_to_incomplete_content_filter() {
    let output = json!({
        "output": { "message": { "content": [{ "text": "" }] } },
        "stopReason": "content_filtered",
        "usage": { "inputTokens": 1, "outputTokens": 0, "totalTokens": 1 }
    });
    let resp = from_converse_output_to_responses(&output, &req(), "m", "resp_c").expect("map");
    assert_eq!(resp.status, "incomplete");
    assert_eq!(
        resp.incomplete_details.expect("details")["reason"],
        "content_filter"
    );
}

#[test]
fn missing_content_array_errors() {
    let output = json!({ "output": { "message": {} }, "stopReason": "end_turn" });
    let err = from_converse_output_to_responses(&output, &req(), "m", "resp_e")
        .expect_err("should error");
    assert!(matches!(err, AppError::Internal(_)));
}

#[test]
fn output_text_part_always_serializes_annotations_as_empty_array() {
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [{ "text": "pong" }] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
    });
    let resp =
        from_converse_output_to_responses(&output, &req(), "m", "resp_ann").expect("map text");
    let v = serde_json::to_value(&resp).expect("serialize");
    let part = &v["output"][0]["content"][0];
    assert_eq!(part["type"], "output_text");
    assert_eq!(part["text"], "pong");
    assert!(
        part.get("annotations").is_some(),
        "annotations key must always be present: {part}"
    );
    assert_eq!(
        part["annotations"],
        json!([]),
        "empty annotations must serialize as []: {part}"
    );
    let s = serde_json::to_string(&resp).expect("string");
    assert!(
        s.contains("\"annotations\":[]"),
        "serialized JSON must contain annotations:[] : {s}"
    );
}
