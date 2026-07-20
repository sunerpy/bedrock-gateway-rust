use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use futures::stream::{self, StreamExt};
use serde_json::json;

use super::*;
use crate::bedrock::capsule::{decode_responses_capsule, is_responses_capsule, CapsuleKeyring};
use crate::domain::{RawResponsesStream, ResponsesStream};
use crate::openai::responses_schema::{
    InputTokensDetails, OutputTokensDetails, ResponsesResponse, ResponsesUsage,
};
use crate::openai::schema::{Function, ReasoningEffort, StreamOptions, Tool};

fn capsule_runtime(enabled: bool) -> Arc<CapsuleRuntime> {
    Arc::new(CapsuleRuntime {
        keyring: CapsuleKeyring::new(
            HashMap::from([("current".to_string(), b"responses-test-key".to_vec())]),
            Some("current".to_string()),
        ),
        encoder_enabled: enabled,
    })
}

fn request(messages: Vec<Message>) -> ChatRequest {
    ChatRequest {
        messages,
        model: "gpt-test".to_string(),
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
        stream_options: None,
        temperature: None,
        top_p: None,
        user: None,
        max_tokens: None,
        max_completion_tokens: Some(2048),
        reasoning_effort: Some(ReasoningEffort::High),
        n: None,
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }),
            },
        }]),
        tool_choice: ToolChoice::String("auto".to_string()),
        stop: None,
        response_format: None,
        extra_body: None,
        extra: HashMap::new(),
    }
}

fn reasoning_item() -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "check weather"}],
        "encrypted_content": "opaque-provider-state"
    })
}

fn responses_response_with_tool() -> ResponsesResponse {
    ResponsesResponse {
        id: "resp_1".to_string(),
        object: "response".to_string(),
        created_at: 1,
        status: "completed".to_string(),
        output: vec![
            serde_json::from_value(reasoning_item()).expect("reasoning item"),
            ResponseOutputItem::FunctionCall {
                id: Some("fc_call_1".to_string()),
                call_id: "call_1".to_string(),
                name: "get_weather".to_string(),
                namespace: None,
                arguments: "{\"city\":\"Paris\"}".to_string(),
                status: None,
            },
        ],
        usage: ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 2 }),
            output_tokens: 20,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 12,
            }),
            total_tokens: 30,
        },
        model: "gpt-test".to_string(),
        instructions: None,
        temperature: None,
        top_p: None,
        tool_choice: None,
        tools: None,
        max_output_tokens: Some(2048),
        parallel_tool_calls: None,
        error: None,
        incomplete_details: None,
    }
}

#[test]
fn chat_request_maps_reasoning_tools_and_stateless_controls() {
    let req = request(vec![Message::User {
        name: None,
        content: ContentInput::Text("weather?".to_string()),
    }]);
    let mapped =
        chat_request_to_responses(&req, &capsule_runtime(true), false).expect("request maps");

    assert_eq!(mapped.store, Some(false));
    assert_eq!(mapped.stream, Some(false));
    assert_eq!(mapped.max_output_tokens, Some(2048));
    assert_eq!(
        mapped.include.as_deref(),
        Some(["reasoning.encrypted_content".to_string()].as_slice())
    );
    assert_eq!(
        mapped
            .reasoning
            .as_ref()
            .and_then(|value| value.effort.as_deref()),
        Some("high")
    );
    assert_eq!(
        mapped
            .reasoning
            .as_ref()
            .and_then(|value| value.summary.as_deref()),
        Some("auto")
    );
    let body = serde_json::to_value(mapped).expect("serializes");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert!(body["tools"][0].get("function").is_none());
}

#[test]
fn generated_chat_completion_ids_are_unique() {
    let first = new_chat_completion_id();
    let second = new_chat_completion_id();

    assert!(first.starts_with("chatcmpl-"));
    assert!(second.starts_with("chatcmpl-"));
    assert_ne!(first, second);
}

#[test]
fn assistant_text_parts_are_flattened_while_user_parts_remain_typed() {
    let assistant: Message = serde_json::from_value(json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "first", "annotations": []},
            {"type": "text", "text": " second"}
        ]
    }))
    .expect("assistant message parses");
    let user: Message = serde_json::from_value(json!({
        "role": "user",
        "content": [{"type": "text", "text": "continue"}]
    }))
    .expect("user message parses");

    let mapped = chat_request_to_responses(
        &request(vec![assistant, user]),
        &capsule_runtime(true),
        false,
    )
    .expect("request maps");
    let body = serde_json::to_value(mapped).expect("serializes");

    assert_eq!(body["input"][0]["role"], "assistant");
    assert_eq!(body["input"][0]["content"], "first second");
    assert_eq!(body["input"][1]["role"], "user");
    assert_eq!(body["input"][1]["content"][0]["type"], "input_text");
}

#[test]
fn nonstream_tool_response_uses_replayable_capsule_and_reasoning_usage() {
    let runtime = capsule_runtime(true);
    let response =
        responses_to_chat(responses_response_with_tool(), &runtime).expect("response maps");
    let message = &response.choices[0].message;

    assert_eq!(
        message.content.as_deref(),
        Some("<think>check weather</think>")
    );
    assert_eq!(
        response.choices[0].finish_reason.as_deref(),
        Some("tool_calls")
    );
    assert_eq!(
        response
            .usage
            .completion_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens),
        Some(12)
    );
    let id = message.tool_calls.as_ref().expect("tool calls")[0]
        .id
        .as_deref()
        .expect("id");
    assert!(is_responses_capsule(id));
    let decoded = decode_responses_capsule(id, &runtime.keyring).expect("capsule decodes");
    assert_eq!(decoded.call_id, "call_1");
    assert_eq!(decoded.reasoning_items, vec![reasoning_item()]);
}

#[test]
fn nonstream_unknown_call_output_fails_instead_of_returning_stop() {
    let mut response = responses_response_with_tool();
    response.output = vec![ResponseOutputItem::Other {
        item_type: "web_search_call".to_string(),
        fields: HashMap::new(),
    }];

    let error = responses_to_chat(response, &capsule_runtime(true))
        .expect_err("an unsupported call item must fail closed");

    assert!(matches!(error, AppError::UpstreamBedrock(_)));
    assert!(error.to_string().contains("web_search_call"));
}

#[test]
fn continuation_replays_reasoning_and_restores_original_call_id() {
    let runtime = capsule_runtime(true);
    let id = encode_responses_capsule("call_1", &[reasoning_item()], &runtime.keyring)
        .expect("capsule encodes");
    let req = request(vec![
        Message::Assistant {
            name: None,
            content: Some(ContentInput::Text(
                "<think>check weather</think>".to_string(),
            )),
            tool_calls: Some(vec![ToolCall {
                index: Some(0),
                id: Some(id.clone()),
                r#type: "function".to_string(),
                function: ResponseFunction {
                    name: Some("get_weather".to_string()),
                    arguments: "{\"city\":\"Paris\"}".to_string(),
                },
            }]),
        },
        Message::Tool {
            content: ToolContentInput::Text("sunny".to_string()),
            tool_call_id: id,
        },
    ]);

    let mapped = chat_request_to_responses(&req, &runtime, false).expect("continuation maps");
    let ResponsesInput::Items(items) = mapped.input else {
        panic!("expected item input");
    };
    assert!(matches!(items[0], ResponseInputItem::Reasoning { .. }));
    assert!(matches!(
        &items[1],
        ResponseInputItem::FunctionCall { call_id, .. } if call_id == "call_1"
    ));
    assert!(matches!(
        &items[2],
        ResponseInputItem::FunctionCallOutput { call_id, .. } if call_id == "call_1"
    ));
}

#[test]
fn continuation_deduplicates_reasoning_ids_across_tool_turns() {
    let runtime = capsule_runtime(true);
    let reasoning = reasoning_item();
    let first =
        encode_responses_capsule("call_1", std::slice::from_ref(&reasoning), &runtime.keyring)
            .expect("first capsule encodes");
    let second =
        encode_responses_capsule("call_2", std::slice::from_ref(&reasoning), &runtime.keyring)
            .expect("second capsule encodes");
    let assistant = |id: String| Message::Assistant {
        name: None,
        content: None,
        tool_calls: Some(vec![ToolCall {
            index: Some(0),
            id: Some(id),
            r#type: "function".to_string(),
            function: ResponseFunction {
                name: Some("get_weather".to_string()),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
        }]),
    };
    let tool = |id: String| Message::Tool {
        content: ToolContentInput::Text("sunny".to_string()),
        tool_call_id: id,
    };
    let req = request(vec![
        assistant(first.clone()),
        tool(first),
        assistant(second.clone()),
        tool(second),
    ]);

    let mapped = chat_request_to_responses(&req, &runtime, false).expect("continuation maps");
    let ResponsesInput::Items(items) = mapped.input else {
        panic!("expected item input");
    };
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, ResponseInputItem::Reasoning { .. }))
            .count(),
        1
    );
    assert!(matches!(
        &items[1],
        ResponseInputItem::FunctionCall { call_id, .. } if call_id == "call_1"
    ));
    assert!(matches!(
        &items[2],
        ResponseInputItem::FunctionCallOutput { call_id, .. } if call_id == "call_1"
    ));
    assert!(matches!(
        &items[3],
        ResponseInputItem::FunctionCall { call_id, .. } if call_id == "call_2"
    ));
    assert!(matches!(
        &items[4],
        ResponseInputItem::FunctionCallOutput { call_id, .. } if call_id == "call_2"
    ));
}

#[test]
fn continuation_rejects_conflicting_duplicate_reasoning_ids() {
    let runtime = capsule_runtime(true);
    let first_reasoning = reasoning_item();
    let mut conflicting_reasoning = first_reasoning.clone();
    conflicting_reasoning["encrypted_content"] = json!("different-ciphertext");
    let first = encode_responses_capsule(
        "call_1",
        std::slice::from_ref(&first_reasoning),
        &runtime.keyring,
    )
    .expect("first capsule encodes");
    let second = encode_responses_capsule(
        "call_2",
        std::slice::from_ref(&conflicting_reasoning),
        &runtime.keyring,
    )
    .expect("second capsule encodes");
    let assistant = |id: String| Message::Assistant {
        name: None,
        content: None,
        tool_calls: Some(vec![ToolCall {
            index: Some(0),
            id: Some(id),
            r#type: "function".to_string(),
            function: ResponseFunction {
                name: Some("get_weather".to_string()),
                arguments: "{}".to_string(),
            },
        }]),
    };
    let req = request(vec![assistant(first), assistant(second)]);

    let err = chat_request_to_responses(&req, &runtime, false)
        .expect_err("conflicting duplicate must fail closed");
    assert!(err
        .to_string()
        .contains("duplicate Responses reasoning item id has conflicting payloads"));
}

#[test]
fn sse_decoder_accepts_arbitrary_byte_boundaries_and_multiline_data() {
    let wire = b"event: ignored\r\ndata: {\"type\":\"one\",\r\ndata: \"value\":1}\r\n\r\ndata: {\"type\":\"two\"}\n\n";
    let mut decoder = SseDecoder::default();
    let mut events = Vec::new();
    for byte in wire {
        events.extend(decoder.push(&[*byte]).expect("byte accepted"));
    }
    events.extend(decoder.finish().expect("decoder finishes"));
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["type"], "one");
    assert_eq!(events[0]["value"], 1);
    assert_eq!(events[1]["type"], "two");
}

#[test]
fn stream_unknown_call_output_fails_before_a_stop_chunk() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1"}
        }),
        json!({
            "type": "response.completed",
            "response": {
                "output": [{"type": "web_search_call", "id": "ws_1"}]
            }
        }),
    ];

    for event in events {
        let mut state = ResponsesChatStreamState::new(
            Arc::from("req-test"),
            "chatcmpl-test".to_string(),
            "gpt-test".to_string(),
            false,
            capsule_runtime(true),
        );
        let error = state
            .map_event(&event)
            .expect_err("an unsupported call item must fail closed");

        assert!(matches!(error, AppError::UpstreamBedrock(_)));
        assert!(error.to_string().contains("web_search_call"));
        assert!(!state.terminal_seen);
        assert_eq!(state.finish_reason, None);
        assert!(state.unknown_output_item_types.contains("web_search_call"));
    }
}

#[test]
fn stream_diagnostics_track_item_types_terminal_event_and_visible_bytes() {
    let mut state = ResponsesChatStreamState::new(
        Arc::from("req-test"),
        "chatcmpl-test".to_string(),
        "gpt-test".to_string(),
        false,
        capsule_runtime(true),
    );

    state
        .map_event(&json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message"}
        }))
        .expect("message item accepted");
    state
        .map_event(&json!({
            "type": "response.output_text.delta",
            "delta": "hello"
        }))
        .expect("text delta accepted");
    let terminal = state
        .map_event(&json!({
            "type": "response.completed",
            "response": {
                "output": [{"type": "message"}]
            }
        }))
        .expect("terminal event accepted");

    assert_eq!(state.visible_text_bytes, 5);
    assert_eq!(state.terminal_event, Some("response.completed"));
    assert_eq!(
        state.output_item_types,
        BTreeSet::from(["message".to_string()])
    );
    assert!(state.unknown_output_item_types.is_empty());
    assert!(terminal.iter().any(|chunk| {
        chunk
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
            == Some("stop")
    }));
    state.adapter_finished = true;
}

struct MockResponsesProvider {
    response: ResponsesResponse,
    raw: Mutex<Option<Vec<Bytes>>>,
}

#[async_trait::async_trait]
impl ResponsesProvider for MockResponsesProvider {
    async fn respond(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Ok(self.response.clone())
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        Err(AppError::Internal("typed stream not expected".to_string()))
    }

    async fn respond_raw_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<Option<RawResponsesStream>, AppError> {
        Ok(self
            .raw
            .lock()
            .expect("raw lock")
            .take()
            .map(|chunks| stream::iter(chunks.into_iter().map(Ok)).boxed()))
    }
}

struct FailingRawResponsesProvider {
    typed_stream_called: AtomicBool,
}

#[async_trait::async_trait]
impl ResponsesProvider for FailingRawResponsesProvider {
    async fn respond(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Err(AppError::Internal(
            "non-stream path not expected".to_string(),
        ))
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        self.typed_stream_called.store(true, Ordering::SeqCst);
        Err(AppError::Internal(
            "typed stream must not be retried".to_string(),
        ))
    }

    async fn respond_raw_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<Option<RawResponsesStream>, AppError> {
        Err(AppError::UpstreamBedrock(
            "original raw stream open failure".to_string(),
        ))
    }
}

#[tokio::test]
async fn raw_open_error_is_preserved_without_typed_retry() {
    let backend = Arc::new(FailingRawResponsesProvider {
        typed_stream_called: AtomicBool::new(false),
    });
    let provider = ResponsesChatProvider::new(backend.clone(), capsule_runtime(true));
    let mut req = request(vec![Message::User {
        name: None,
        content: ContentInput::Text("continue".to_string()),
    }]);
    req.stream = Some(true);
    let normalized = NormalizedChatRequest {
        request: req,
        resolved_model: "openai.gpt-test".to_string(),
        request_id: Arc::from("req-raw-open-failure"),
        received_at: Instant::now(),
        raw_body: Bytes::new(),
    };

    let error = match provider.chat_stream(&normalized).await {
        Err(error) => error,
        Ok(_) => panic!("raw open failure must be returned"),
    };

    assert!(matches!(error, AppError::UpstreamBedrock(_)));
    assert!(error
        .to_string()
        .contains("original raw stream open failure"));
    assert!(!backend.typed_stream_called.load(Ordering::SeqCst));
}

fn sse_event(value: Value) -> String {
    format!("data: {value}\n\n")
}

#[tokio::test]
async fn stream_closes_reasoning_before_tool_id_and_emits_metadata_once() {
    let mut wire = String::new();
    wire.push_str(&sse_event(json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": reasoning_item()
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.reasoning_summary_text.delta",
        "delta": "check weather"
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.reasoning_summary_text.done",
        "text": "check weather"
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "id": "fc_call_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": ""
        }
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.function_call_arguments.delta",
        "output_index": 1,
        "delta": "{\"city\":\"Paris\"}"
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.function_call_arguments.done",
        "output_index": 1,
        "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}"
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "id": "fc_call_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}",
            "status": "completed"
        }
    })));
    wire.push_str(&sse_event(json!({
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 12}
            }
        }
    })));

    let chunks = wire
        .as_bytes()
        .chunks(7)
        .map(Bytes::copy_from_slice)
        .collect::<Vec<_>>();
    let runtime = capsule_runtime(true);
    let provider = ResponsesChatProvider::new(
        Arc::new(MockResponsesProvider {
            response: responses_response_with_tool(),
            raw: Mutex::new(Some(chunks)),
        }),
        Arc::clone(&runtime),
    );
    let mut req = request(vec![Message::User {
        name: None,
        content: ContentInput::Text("weather?".to_string()),
    }]);
    req.stream = Some(true);
    req.stream_options = Some(StreamOptions {
        include_usage: true,
    });
    let normalized = NormalizedChatRequest {
        request: req,
        resolved_model: "openai.gpt-test".to_string(),
        request_id: Arc::from("req-test"),
        received_at: Instant::now(),
        raw_body: Bytes::new(),
    };

    let chunks = provider
        .chat_stream(&normalized)
        .await
        .expect("stream starts")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream succeeds");

    let completion_ids = chunks
        .iter()
        .map(|chunk| chunk.id.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(
        completion_ids.len(),
        1,
        "all chunks in one completion must share one id"
    );
    assert_ne!(
        completion_ids.into_iter().next(),
        Some("chatcmpl-req-test"),
        "the client correlation id must not become the completion id"
    );

    let mut content = String::new();
    let mut metadata_ids = Vec::new();
    let mut arguments = String::new();
    let mut finish = None;
    for chunk in &chunks {
        for choice in &chunk.choices {
            if let Some(part) = &choice.delta.content {
                content.push_str(part);
            }
            if let Some(calls) = &choice.delta.tool_calls {
                for call in calls {
                    if let Some(id) = &call.id {
                        metadata_ids.push(id.clone());
                        assert_eq!(content, "<think>check weather</think>");
                    }
                    arguments.push_str(&call.function.arguments);
                }
            }
            if choice.finish_reason.is_some() {
                finish = choice.finish_reason.clone();
            }
        }
    }

    assert_eq!(metadata_ids.len(), 1);
    assert!(is_responses_capsule(&metadata_ids[0]));
    assert_eq!(arguments, "{\"city\":\"Paris\"}");
    assert_eq!(finish.as_deref(), Some("tool_calls"));
    assert!(chunks.iter().any(|chunk| {
        chunk
            .usage
            .as_ref()
            .and_then(|usage| usage.completion_tokens_details.as_ref())
            .is_some_and(|details| details.reasoning_tokens == 12)
    }));
}

#[test]
fn tool_response_with_reasoning_fails_closed_when_capsules_are_disabled() {
    let error = responses_to_chat(responses_response_with_tool(), &capsule_runtime(false))
        .expect_err("must fail closed");
    assert!(matches!(error, AppError::Internal(_)));
}
