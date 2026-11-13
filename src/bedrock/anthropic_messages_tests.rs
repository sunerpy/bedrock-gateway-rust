//! Unit tests for [`crate::bedrock::anthropic_messages`] (sidecar, see `AGENTS.md`).

use super::*;
use crate::bedrock::responses_translate::ResponsesToolKind;

fn searched(name: &str) -> SearchedTool {
    SearchedTool {
        bedrock_name: name.to_string(),
        client_name: name.to_string(),
        namespace: None,
        kind: ResponsesToolKind::Function,
        description: None,
        input_schema: json!({ "type": "object", "properties": {} }),
    }
}

fn input<'a>(
    messages: &'a Value,
    system: &'a Value,
    inference: &'a Value,
    tool_config: Option<&'a Value>,
    searched_tools: &'a [SearchedTool],
) -> MessagesBodyInput<'a> {
    MessagesBodyInput {
        messages,
        system,
        inference_config: inference,
        additional_fields: None,
        tool_config,
        output_config: None,
        searched_tools,
        default_max_tokens: None,
    }
}

#[test]
fn stop_reason_maps_refusal_only() {
    assert_eq!(map_stop_reason("refusal"), "content_filtered");
    for same in [
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "model_context_window_exceeded",
    ] {
        assert_eq!(map_stop_reason(same), same);
    }
}

#[test]
fn cache_points_become_cache_control_on_the_previous_block() {
    let messages = json!([{ "role": "user", "content": [
        { "text": "hello" }, { "cachePoint": { "type": "default", "ttl": "1h" } }
    ]}]);
    let system = json!([{ "text": "sys" }, { "cachePoint": { "type": "default" } }]);
    let inference = json!({ "maxTokens": 100, "temperature": 0.5, "topP": 0.9 });
    let body = build_messages_body(&input(&messages, &system, &inference, None, &[])).unwrap();
    assert_eq!(
        body["system"],
        json!([{ "type": "text", "text": "sys", "cache_control": { "type": "ephemeral" } }])
    );
    assert_eq!(
        body["messages"][0]["content"],
        json!([{ "type": "text", "text": "hello",
                 "cache_control": { "type": "ephemeral", "ttl": "1h" } }])
    );
    assert_eq!(body["max_tokens"], 100);
    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["top_p"], 0.9);
    assert!(body.get("tools").is_none());
}

#[test]
fn blocks_convert_to_messages_shapes() {
    let messages = json!([
        { "role": "user", "content": [
            { "text": "look" },
            { "image": { "format": "png", "source": { "bytes": [1, 2, 3] } } }
        ]},
        { "role": "assistant", "content": [
            { "reasoningContent": { "reasoningText": { "text": "t", "signature": "s" } } },
            { "reasoningContent": { "redactedContent": "UkVEQUNURUQ=" } },
            { "toolUse": { "toolUseId": "c1", "name": "f", "input": { "a": 1 } } }
        ]},
        { "role": "user", "content": [
            { "toolResult": { "toolUseId": "c1", "status": "error", "content": [
                { "text": "" }, { "text": "boom" }, { "json": { "k": 1 } }
            ] } }
        ]}
    ]);
    let tools = json!({ "tools": [{ "toolSpec": { "name": "f", "inputSchema": { "json": { "type": "object" } } } }] });
    let body = build_messages_body(&input(
        &messages,
        &json!([]),
        &json!({ "maxTokens": 9 }),
        Some(&tools),
        &[],
    ))
    .unwrap();
    assert!(body.get("system").is_none());
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AQID" } })
    );
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            { "type": "thinking", "thinking": "t", "signature": "s" },
            { "type": "redacted_thinking", "data": "REDACTED" },
            { "type": "tool_use", "id": "c1", "name": "f", "input": { "a": 1 } }
        ])
    );
    assert_eq!(
        body["messages"][2]["content"][0],
        json!({ "type": "tool_result", "tool_use_id": "c1", "is_error": true, "content": [
            { "type": "text", "text": "boom" }, { "type": "text", "text": "{\"k\":1}" }
        ] })
    );
    assert_eq!(
        body["tools"],
        json!([{ "name": "f", "input_schema": { "type": "object" } }])
    );
}

#[test]
fn reasoning_and_structured_output_fields_reach_the_top_level() {
    let fields = json!({
        "reasoning_config": { "type": "enabled", "budget_tokens": 2048 },
        "output_config": { "effort": "high" }
    });
    let output = json!({ "textFormat": { "type": "json_schema", "structure": { "jsonSchema": {
        "schema": "{\"type\":\"object\"}", "name": "response" } } } });
    let messages = json!([{ "role": "user", "content": [{ "text": "x" }] }]);
    let (system, inference) = (json!([]), json!({}));
    let mut body_input = input(&messages, &system, &inference, None, &[]);
    body_input.additional_fields = Some(&fields);
    body_input.output_config = Some(&output);
    body_input.default_max_tokens = Some(4096);
    let body = build_messages_body(&body_input).unwrap();
    assert_eq!(
        body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 2048 })
    );
    assert!(body.get("reasoning_config").is_none());
    assert_eq!(
        body["output_config"],
        json!({ "effort": "high", "format": { "type": "json_schema", "schema": { "type": "object" } } })
    );
    assert_eq!(body["max_tokens"], 4096);
}

#[test]
fn missing_max_tokens_is_a_bad_request() {
    let messages = json!([]);
    let err =
        build_messages_body(&input(&messages, &json!([]), &json!({}), None, &[])).unwrap_err();
    assert!(matches!(err, AppError::BadRequest(_)));
}

#[test]
fn deferred_tools_follow_declared_tools_without_cache_control() {
    let messages = json!([]);
    let tools = json!({
        "tools": [
            { "toolSpec": { "name": "a", "description": "A", "inputSchema": { "json": { "type": "object" } } } },
            { "cachePoint": { "type": "default", "ttl": "5m" } }
        ],
        "toolChoice": { "auto": {} }
    });
    let deferred = [searched("ns__b")];
    let body = build_messages_body(&input(
        &messages,
        &json!([]),
        &json!({ "maxTokens": 1 }),
        Some(&tools),
        &deferred,
    ))
    .unwrap();
    assert_eq!(
        body["tools"],
        json!([
            { "name": "a", "description": "A", "input_schema": { "type": "object" },
              "cache_control": { "type": "ephemeral", "ttl": "5m" } },
            { "name": "ns__b", "input_schema": { "type": "object", "properties": {} }, "defer_loading": true }
        ])
    );
    assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
}

#[test]
fn deferred_tool_limits_are_checked_before_the_call() {
    let messages = json!([]);
    let inference = json!({ "maxTokens": 1 });
    // Every tool deferred.
    let err = build_messages_body(&input(
        &messages,
        &json!([]),
        &inference,
        None,
        &[searched("x")],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("at least one tool"));

    // More than MAX_DEFERRED_TOOLS.
    let tools =
        json!({ "tools": [{ "toolSpec": { "name": "a", "inputSchema": { "json": {} } } }] });
    let many: Vec<SearchedTool> = (0..=MAX_DEFERRED_TOOLS)
        .map(|i| searched(&format!("t{i}")))
        .collect();
    let err = build_messages_body(&input(
        &messages,
        &json!([]),
        &inference,
        Some(&tools),
        &many,
    ))
    .unwrap_err();
    assert!(err.to_string().contains("10000"));

    // Invalid and overlong names.
    for name in ["has space", "dot.ted", &"x".repeat(129)] {
        let tools =
            json!({ "tools": [{ "toolSpec": { "name": name, "inputSchema": { "json": {} } } }] });
        assert!(
            build_messages_body(&input(&messages, &json!([]), &inference, Some(&tools), &[]))
                .is_err()
        );
    }
}

#[test]
fn unsupported_blocks_are_rejected() {
    let messages = json!([{ "role": "user", "content": [{ "document": {} }] }]);
    assert!(build_messages_body(&input(
        &messages,
        &json!([]),
        &json!({ "maxTokens": 1 }),
        None,
        &[]
    ))
    .is_err());
}

#[test]
fn non_stream_response_maps_to_converse_output() {
    let converse = messages_response_to_converse_output(&json!({
        "id": "msg_1", "type": "message", "role": "assistant",
        "content": [
            { "type": "thinking", "thinking": "hmm", "signature": "sig" },
            { "type": "redacted_thinking", "data": "QUJD" },
            { "type": "text", "text": "hi" },
            { "type": "tool_use", "id": "toolu_1", "name": "tool_search", "input": { "query": "q" } },
            { "type": "server_tool_use", "id": "srv", "name": "x", "input": {} }
        ],
        "stop_reason": "refusal",
        "usage": { "input_tokens": 5, "output_tokens": 7,
                   "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20 }
    }))
    .unwrap();
    assert_eq!(converse["stopReason"], "content_filtered");
    assert_eq!(
        converse["output"]["message"]["content"],
        json!([
            { "reasoningContent": { "reasoningText": { "text": "hmm", "signature": "sig" } } },
            { "reasoningContent": { "redactedContent": "UVVKRA==" } },
            { "text": "hi" },
            { "toolUse": { "toolUseId": "toolu_1", "name": "tool_search", "input": { "query": "q" } } }
        ])
    );
    assert_eq!(
        converse["usage"],
        json!({ "inputTokens": 5, "outputTokens": 7, "totalTokens": 132,
                "cacheReadInputTokens": 100, "cacheWriteInputTokens": 20 })
    );
    assert!(messages_response_to_converse_output(&json!({})).is_err());
}

#[test]
fn stream_adapter_maps_events_and_ignores_pings() {
    let mut adapter = MessagesStreamAdapter::new();
    assert!(adapter.map_chunk(br#"{"type":"ping"}"#).unwrap().is_empty());
    assert!(adapter
        .map_chunk(br#"{"type":"future_event"}"#)
        .unwrap()
        .is_empty());
    assert!(adapter.map_chunk(b"not json").is_err());

    let start = adapter
        .map_chunk(br#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"n","input":{}}}"#)
        .unwrap();
    match &start[..] {
        [ConverseStreamOutput::ContentBlockStart(ev)] => {
            assert_eq!(ev.content_block_index(), 2);
            let tool = ev.start().unwrap().as_tool_use().unwrap();
            assert_eq!(tool.tool_use_id(), "t1");
            assert_eq!(tool.name(), "n");
        }
        other => panic!("unexpected {other:?}"),
    }

    let redacted = adapter
        .map_chunk(br#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"QUJD"}}"#)
        .unwrap();
    match &redacted[..] {
        [ConverseStreamOutput::ContentBlockDelta(ev)] => match ev.delta().unwrap() {
            ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::RedactedContent(
                blob,
            )) => {
                assert_eq!(blob.as_ref(), b"QUJD");
            }
            other => panic!("unexpected {other:?}"),
        },
        other => panic!("unexpected {other:?}"),
    }

    adapter
        .map_chunk(br#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#)
        .unwrap();
    let stop = adapter.map_chunk(br#"{"type":"message_stop"}"#).unwrap();
    match &stop[..] {
        [ConverseStreamOutput::MessageStop(ev), ConverseStreamOutput::Metadata(meta)] => {
            assert_eq!(ev.stop_reason().as_str(), "tool_use");
            assert_eq!(meta.usage().unwrap().output_tokens(), 3);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn stream_error_events_map_to_gateway_errors() {
    let mut adapter = MessagesStreamAdapter::new();
    let rate = adapter
        .map_chunk(br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#)
        .unwrap_err();
    assert!(matches!(rate, AppError::Throttled(_)));
    let invalid = adapter
        .map_chunk(br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#)
        .unwrap_err();
    assert!(matches!(invalid, AppError::BadRequest(_)));
}

#[test]
fn redacted_reasoning_round_trips_in_converse_encoding() {
    // Converse stores Messages `data` as the blob of its UTF-8 bytes; the
    // envelope holds that blob in standard base64.
    let data = "CAQSlwYKEggTEAEYAjgBQgh0aGlua2luZxIMJbzbC+ICLtHrU1Mq";
    let converse = messages_response_to_converse_output(&json!({
        "content": [{ "type": "redacted_thinking", "data": data }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    }))
    .unwrap();
    let block = &converse["output"]["message"]["content"][0];
    let encoded = block["reasoningContent"]["redactedContent"]
        .as_str()
        .unwrap();
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap(),
        data.as_bytes()
    );
    let messages = json!([{ "role": "assistant", "content": [block.clone()] }]);
    let body = build_messages_body(&input(
        &messages,
        &json!([]),
        &json!({ "maxTokens": 1 }),
        None,
        &[],
    ))
    .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({ "type": "redacted_thinking", "data": data })
    );

    let bad = json!([{ "role": "assistant", "content": [
        { "reasoningContent": { "redactedContent": "not base64!" } }
    ]}]);
    assert!(build_messages_body(&input(
        &bad,
        &json!([]),
        &json!({ "maxTokens": 1 }),
        None,
        &[]
    ))
    .is_err());
}
