//! Unit tests for the streaming Converse → Responses lifecycle state machine.
//!
//! Relocated out of `responses_stream.rs` into a sibling `#[path]` module for
//! code organization (see the `test-coverage-codecov` spec). Behavior is
//! unchanged; `use super::*;` still resolves to the implementation module.

use super::*;
use crate::openai::responses_schema::ResponsesInput;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent, ConversationRole,
    ConverseStreamMetadataEvent, MessageStartEvent, MessageStopEvent, StopReason, TokenUsage,
    ToolUseBlockDelta, ToolUseBlockStart,
};
use std::collections::HashMap;

const MODEL: &str = "anthropic.claude-3-sonnet-20240229-v1:0";
const RID: &str = "resp_test";

fn request() -> ResponsesRequest {
    ResponsesRequest {
        model: "incoming".to_string(),
        input: ResponsesInput::Text("hi".to_string()),
        instructions: None,
        tools: None,
        tool_choice: None,
        temperature: None,
        top_p: None,
        max_output_tokens: None,
        stream: Some(true),
        reasoning: None,
        text: None,
        include: None,
        metadata: None,
        parallel_tool_calls: None,
        store: None,
        previous_response_id: None,
        extra: HashMap::new(),
    }
}

fn state() -> ResponsesStreamState {
    ResponsesStreamState::new(
        MODEL.to_string(),
        RID.to_string(),
        request(),
        Arc::from("req-test"),
        Instant::now(),
    )
}

// -- SDK event builders --------------------------------------------------

fn ev_message_start() -> ConverseStreamOutput {
    ConverseStreamOutput::MessageStart(
        MessageStartEvent::builder()
            .role(ConversationRole::Assistant)
            .build()
            .unwrap(),
    )
}

fn ev_text(text: &str, block: i32) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::Text(text.to_string()))
            .content_block_index(block)
            .build()
            .unwrap(),
    )
}

fn ev_reasoning_text(text: &str, block: i32) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Text(text.to_string()),
            ))
            .content_block_index(block)
            .build()
            .unwrap(),
    )
}

fn ev_tool_start(block: i32, id: &str, name: &str) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockStart(
        ContentBlockStartEvent::builder()
            .start(ContentBlockStart::ToolUse(
                ToolUseBlockStart::builder()
                    .tool_use_id(id)
                    .name(name)
                    .build()
                    .unwrap(),
            ))
            .content_block_index(block)
            .build()
            .unwrap(),
    )
}

fn ev_tool_input(block: i32, input: &str) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ToolUse(
                ToolUseBlockDelta::builder().input(input).build().unwrap(),
            ))
            .content_block_index(block)
            .build()
            .unwrap(),
    )
}

fn ev_block_stop(block: i32) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockStop(
        ContentBlockStopEvent::builder()
            .content_block_index(block)
            .build()
            .unwrap(),
    )
}

fn ev_message_stop(reason: StopReason) -> ConverseStreamOutput {
    ConverseStreamOutput::MessageStop(
        MessageStopEvent::builder()
            .stop_reason(reason)
            .build()
            .unwrap(),
    )
}

fn ev_metadata(input: i32, output: i32, total: i32) -> ConverseStreamOutput {
    let usage = TokenUsage::builder()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(total)
        .build()
        .unwrap();
    ConverseStreamOutput::Metadata(ConverseStreamMetadataEvent::builder().usage(usage).build())
}

/// Drive a synthetic event vector through the pure mapper + `finish`,
/// returning the full ordered event sequence (mirrors the async wrapper
/// without a live receiver).
fn drive(events: &[ConverseStreamOutput]) -> Vec<ResponseStreamEvent> {
    let mut st = state();
    let mut all = Vec::new();
    for e in events {
        all.extend(st.map_event(e));
    }
    all.extend(st.finish());
    all
}

fn drive_with_tool(tool: Value, events: &[ConverseStreamOutput]) -> Vec<ResponseStreamEvent> {
    let mut req = request();
    req.tools = Some(vec![serde_json::from_value(tool).expect("tool schema")]);
    let (_, registry) =
        crate::bedrock::responses_translate::build_responses_tools(&req).expect("tool registry");
    let mut st = ResponsesStreamState::new_with_tools(
        MODEL.to_string(),
        RID.to_string(),
        req,
        Arc::from("req-test"),
        Instant::now(),
        registry,
    );
    let mut all = Vec::new();
    for event in events {
        all.extend(st.map_event(event));
    }
    all.extend(st.finish());
    all
}

/// The `type` tag of an event (via serde) — for order assertions.
fn type_of(ev: &ResponseStreamEvent) -> String {
    serde_json::to_value(ev).unwrap()["type"]
        .as_str()
        .unwrap()
        .to_string()
}

fn seq_of(ev: &ResponseStreamEvent) -> u64 {
    serde_json::to_value(ev).unwrap()["sequence_number"]
        .as_u64()
        .unwrap()
}

/// Assert sequence numbers are monotonic from 0 with no gaps.
fn assert_monotonic_from_zero(events: &[ResponseStreamEvent]) {
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(seq_of(ev), i as u64, "sequence_number not monotonic at {i}");
    }
}

/// Assert no Chat-style `[DONE]` sentinel appears in a Responses stream.
fn assert_no_done_no_argdelta(events: &[ResponseStreamEvent]) {
    for ev in events {
        let s = serde_json::to_string(ev).unwrap();
        assert!(!s.contains("[DONE]"), "[DONE] sentinel leaked: {s}");
    }
}

// -- Test 1: text stream → exact event order -----------------------------

#[test]
fn text_stream_emits_exact_event_order() {
    let events = drive(&[
        ev_message_start(),
        ev_text("Hel", 0),
        ev_text("lo", 0),
        ev_block_stop(0),
        ev_message_stop(StopReason::EndTurn),
        ev_metadata(8, 2, 10),
    ]);

    let types: Vec<String> = events.iter().map(type_of).collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "text stream lifecycle order mismatch"
    );

    // sequence_number monotonic from 0.
    assert_monotonic_from_zero(&events);
    // NO [DONE], NO arg-delta.
    assert_no_done_no_argdelta(&events);

    // The two text deltas carry the streamed fragments.
    match &events[4] {
        ResponseStreamEvent::OutputTextDelta { delta, .. } => assert_eq!(delta, "Hel"),
        other => panic!("expected output_text.delta, got {other:?}"),
    }
    match &events[5] {
        ResponseStreamEvent::OutputTextDelta { delta, .. } => assert_eq!(delta, "lo"),
        other => panic!("expected output_text.delta, got {other:?}"),
    }
    // output_text.done carries the coalesced text.
    match &events[6] {
        ResponseStreamEvent::OutputTextDone { text, .. } => assert_eq!(text, "Hello"),
        other => panic!("expected output_text.done, got {other:?}"),
    }

    // Final completed carries a resp_ id + usage.
    match events.last().expect("completed") {
        ResponseStreamEvent::Completed { response, .. } => {
            assert!(response.id.starts_with("resp_"));
            assert_eq!(response.status, "completed");
            // usage filled from metadata: input=8,out=2,total=10.
            assert_eq!(response.usage.input_tokens, 8);
            assert_eq!(response.usage.output_tokens, 2);
            assert_eq!(response.usage.total_tokens, 10);
            // The message item carries the full text.
            match &response.output[0] {
                ResponseOutputItem::Message { content, .. } => match &content[0] {
                    OutputContentPart::OutputText { text, .. } => assert_eq!(text, "Hello"),
                    other => panic!("expected output_text part, got {other:?}"),
                },
                other => panic!("expected message item, got {other:?}"),
            }
        }
        other => panic!("expected response.completed last, got {other:?}"),
    }
}

// -- Test 2: tool-use stream → arguments delta/done + item done ----------

#[test]
fn tool_use_stream_emits_function_call_item_add_and_done() {
    let events = drive(&[
        ev_message_start(),
        ev_tool_start(0, "call-1", "get_weather"),
        ev_tool_input(0, "{\"city\":"),
        ev_tool_input(0, "\"Paris\"}"),
        ev_block_stop(0),
        ev_message_stop(StopReason::ToolUse),
        ev_metadata(12, 6, 18),
    ]);

    let types: Vec<String> = events.iter().map(type_of).collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ],
        "tool-use lifecycle order mismatch"
    );

    assert_monotonic_from_zero(&events);
    assert_no_done_no_argdelta(&events);

    // output_item.added carries a function_call item with the call_id/name
    // and status "in_progress".
    match &events[2] {
        ResponseStreamEvent::OutputItemAdded { item, .. } => match item {
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                status,
                ..
            } => {
                assert_eq!(call_id, "call-1");
                assert_eq!(name, "get_weather");
                assert_eq!(status.as_deref(), None);
            }
            other => panic!("expected function_call item, got {other:?}"),
        },
        other => panic!("expected output_item.added, got {other:?}"),
    }

    // output_item.done carries the COMPLETE, JSON-parseable arguments and the
    // status "completed" that @ai-sdk/openai's stream schema REQUIRES.
    match &events[6] {
        ResponseStreamEvent::OutputItemDone { item, .. } => match item {
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                status,
                ..
            } => {
                assert_eq!(call_id, "call-1");
                assert_eq!(name, "get_weather");
                assert_eq!(status.as_deref(), Some("completed"));
                let parsed: Value =
                    serde_json::from_str(arguments).expect("arguments must be JSON-parseable");
                assert_eq!(parsed, json!({ "city": "Paris" }));
            }
            other => panic!("expected function_call item, got {other:?}"),
        },
        other => panic!("expected output_item.done, got {other:?}"),
    }

    // The serialized output_item.done JSON must literally carry
    // "status":"completed" (ai-sdk validates the wire string, not the enum).
    let done_json = serde_json::to_value(&events[6]).unwrap();
    assert_eq!(done_json["item"]["status"], "completed");

    // The final completed Response carries the function_call output item.
    match events.last().unwrap() {
        ResponseStreamEvent::Completed { response, .. } => match &response.output[0] {
            ResponseOutputItem::FunctionCall {
                call_id, arguments, ..
            } => {
                assert_eq!(call_id, "call-1");
                let parsed: Value = serde_json::from_str(arguments).unwrap();
                assert_eq!(parsed, json!({ "city": "Paris" }));
            }
            other => panic!("expected function_call in final output, got {other:?}"),
        },
        other => panic!("expected completed, got {other:?}"),
    }
}

/// REGRESSION (@ai-sdk/openai streaming tool calls): the streamed
/// `output_item.done` function_call item MUST serialize `"status":"completed"`
/// (ai-sdk's done-chunk schema pins `status: z.literal("completed")`), while
/// the `output_item.added` function_call item MUST NOT carry a `status` field
/// (ai-sdk's added-chunk schema has none). Without the done-side status the
/// chunk degrades to `unknown_chunk` and the tool call is never reconstructed.
#[test]
fn tool_use_stream_done_item_serializes_status_completed() {
    let events = drive(&[
        ev_message_start(),
        ev_tool_start(0, "call-1", "get_weather"),
        ev_tool_input(0, "{\"city\":\"Tokyo\"}"),
        ev_block_stop(0),
        ev_message_stop(StopReason::ToolUse),
        ev_metadata(12, 6, 18),
    ]);

    let added = events
        .iter()
        .find(|e| matches!(e, ResponseStreamEvent::OutputItemAdded { .. }))
        .expect("an output_item.added event");
    let added_item = &serde_json::to_value(added).unwrap()["item"];
    assert_eq!(added_item["type"], "function_call");
    assert!(
        added_item.get("status").is_none(),
        "output_item.added function_call MUST NOT carry status: {added_item}"
    );

    let done = events
        .iter()
        .find(|e| matches!(e, ResponseStreamEvent::OutputItemDone { item, .. } if matches!(item, ResponseOutputItem::FunctionCall { .. })))
        .expect("an output_item.done function_call event");
    let done_item = &serde_json::to_value(done).unwrap()["item"];
    assert_eq!(done_item["type"], "function_call");
    assert_eq!(
        done_item["status"], "completed",
        "output_item.done function_call MUST serialize status:completed: {done_item}"
    );
    // The other fields remain unchanged.
    assert_eq!(done_item["call_id"], "call-1");
    assert_eq!(done_item["name"], "get_weather");
    assert_eq!(done_item["id"], "fc_call-1");
    assert_eq!(done_item["arguments"], "{\"city\":\"Tokyo\"}");

    // The final completed Response's function_call output item is the
    // non-stream assembly and MUST stay status-free (unchanged contract).
    let completed = events.last().unwrap();
    let completed_item = &serde_json::to_value(completed).unwrap()["response"]["output"][0];
    assert_eq!(completed_item["type"], "function_call");
    assert!(
        completed_item.get("status").is_none(),
        "completed function_call output item MUST NOT carry status: {completed_item}"
    );
}

#[test]
fn client_tool_added_events_wait_for_required_payloads() {
    let cases = [
        (
            json!({"type":"local_shell"}),
            "local_shell",
            json!({"action":{"type":"exec","command":["pwd"]}}),
            "local_shell_call",
            "action",
            json!({"type":"exec","command":["pwd"]}),
        ),
        (
            json!({"type":"shell"}),
            "shell",
            json!({"commands":["pwd"],"timeout_ms":1000}),
            "shell_call",
            "action",
            json!({"commands":["pwd"],"timeout_ms":1000}),
        ),
        (
            json!({"type":"apply_patch"}),
            "apply_patch",
            json!({"operation":{"type":"update_file","path":"src/lib.rs","diff":"@@"}}),
            "apply_patch_call",
            "operation",
            json!({"type":"update_file","path":"src/lib.rs","diff":"@@"}),
        ),
    ];

    for (tool, name, input, item_type, payload_key, expected_payload) in cases {
        let encoded = serde_json::to_string(&input).unwrap();
        let events = drive_with_tool(
            tool,
            &[
                ev_message_start(),
                ev_tool_start(0, "call-client", name),
                ev_tool_input(0, &encoded),
                ev_block_stop(0),
                ev_message_stop(StopReason::ToolUse),
                ev_metadata(8, 4, 12),
            ],
        );
        let added = events
            .iter()
            .find(|event| matches!(event, ResponseStreamEvent::OutputItemAdded { .. }))
            .expect("delayed output_item.added");
        let added = serde_json::to_value(added).unwrap();
        assert_eq!(added["item"]["type"], item_type);
        assert_eq!(added["item"][payload_key], expected_payload);
        assert!(added["item"].get("name").is_none());

        let done = events
            .iter()
            .find(|event| matches!(event, ResponseStreamEvent::OutputItemDone { .. }))
            .expect("output_item.done");
        let done = serde_json::to_value(done).unwrap();
        assert_eq!(done["item"]["type"], item_type);
        assert_eq!(done["item"][payload_key], expected_payload);
    }
}

#[test]
fn reasoning_stream_emits_reasoning_item_before_message() {
    let events = drive(&[
        ev_message_start(),
        ev_reasoning_text("let me ", 0),
        ev_reasoning_text("think", 0),
        ev_text("The answer is 4.", 1),
        ev_message_stop(StopReason::EndTurn),
        ev_metadata(10, 5, 15),
    ]);

    let types: Vec<String> = events.iter().map(type_of).collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            // reasoning item FIRST.
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            // then the message item.
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "reasoning lifecycle order mismatch"
    );

    assert_monotonic_from_zero(&events);
    assert_no_done_no_argdelta(&events);

    // The reasoning item is emitted via reasoning_text.delta — NOT <think>.
    for ev in &events {
        let s = serde_json::to_string(ev).unwrap();
        assert!(!s.contains("<think>"), "<think> leaked into reasoning: {s}");
    }

    // First output_item.added is a reasoning item.
    match &events[2] {
        ResponseStreamEvent::OutputItemAdded { item, .. } => {
            assert!(matches!(item, ResponseOutputItem::Reasoning { .. }));
        }
        other => panic!("expected reasoning output_item.added, got {other:?}"),
    }
    // Reasoning text delta carries the first fragment.
    match &events[4] {
        ResponseStreamEvent::ReasoningTextDelta { delta, .. } => assert_eq!(delta, "let me "),
        other => panic!("expected reasoning_text.delta, got {other:?}"),
    }
    // The paired summary delta carries the SAME fragment.
    match &events[5] {
        ResponseStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
            assert_eq!(delta, "let me ")
        }
        other => panic!("expected reasoning_summary_text.delta, got {other:?}"),
    }
    // reasoning_text.done carries the coalesced reasoning text.
    match &events[8] {
        ResponseStreamEvent::ReasoningTextDone { text, .. } => assert_eq!(text, "let me think"),
        other => panic!("expected reasoning_text.done, got {other:?}"),
    }

    // The final completed Response has reasoning FIRST, then the message.
    match events.last().unwrap() {
        ResponseStreamEvent::Completed { response, .. } => {
            assert_eq!(response.output.len(), 2);
            assert!(matches!(
                response.output[0],
                ResponseOutputItem::Reasoning { .. }
            ));
            assert!(matches!(
                response.output[1],
                ResponseOutputItem::Message { .. }
            ));
        }
        other => panic!("expected completed, got {other:?}"),
    }
}

// -- dual reasoning emission: text family + ai-sdk summary family --------

/// A reasoning stream emits BOTH reasoning families so opencode (which keys
/// off `reasoning_text.*`) AND the vercel ai-sdk Responses parser (which
/// keys off `reasoning_summary_text.*`) each render reasoning live. Every
/// `reasoning_text.delta` is paired with a `reasoning_summary_text.delta`
/// carrying the identical fragment, all sharing one `item_id`, with
/// strictly monotonic `sequence_number`s across the combined set.
#[test]
fn reasoning_emits_both_text_and_summary_families() {
    let events = drive(&[
        ev_message_start(),
        ev_reasoning_text("alpha", 0),
        ev_reasoning_text("beta", 0),
        ev_message_stop(StopReason::EndTurn),
        ev_metadata(4, 2, 6),
    ]);

    let mut text_deltas: Vec<String> = Vec::new();
    let mut summary_deltas: Vec<String> = Vec::new();
    let mut reasoning_item_ids: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut saw_summary_part_added = false;
    let mut saw_summary_part_done = false;
    let mut saw_summary_text_done = false;

    for ev in &events {
        match ev {
            ResponseStreamEvent::ReasoningTextDelta { delta, item_id, .. } => {
                text_deltas.push(delta.clone());
                reasoning_item_ids.insert(item_id.clone());
            }
            ResponseStreamEvent::ReasoningSummaryTextDelta {
                delta,
                item_id,
                summary_index,
                ..
            } => {
                summary_deltas.push(delta.clone());
                reasoning_item_ids.insert(item_id.clone());
                assert_eq!(*summary_index, 0, "summary_index must be 0");
            }
            ResponseStreamEvent::ReasoningSummaryPartAdded {
                item_id,
                summary_index,
                ..
            } => {
                saw_summary_part_added = true;
                reasoning_item_ids.insert(item_id.clone());
                assert_eq!(*summary_index, 0);
            }
            ResponseStreamEvent::ReasoningSummaryPartDone {
                item_id,
                summary_index,
                ..
            } => {
                saw_summary_part_done = true;
                reasoning_item_ids.insert(item_id.clone());
                assert_eq!(*summary_index, 0);
            }
            ResponseStreamEvent::ReasoningSummaryTextDone { text, item_id, .. } => {
                saw_summary_text_done = true;
                reasoning_item_ids.insert(item_id.clone());
                assert_eq!(text, "alphabeta");
            }
            _ => {}
        }
    }

    assert_eq!(text_deltas, vec!["alpha", "beta"]);
    assert_eq!(
        summary_deltas, text_deltas,
        "summary deltas must mirror the text deltas exactly"
    );
    assert!(
        saw_summary_part_added,
        "missing reasoning_summary_part.added"
    );
    assert!(saw_summary_part_done, "missing reasoning_summary_part.done");
    assert!(saw_summary_text_done, "missing reasoning_summary_text.done");
    assert_eq!(
        reasoning_item_ids.len(),
        1,
        "both reasoning families must share one item_id, got {reasoning_item_ids:?}"
    );

    assert_monotonic_from_zero(&events);
    assert_no_done_no_argdelta(&events);
}

// -- mid-stream error → response.failed, no completed --------------------

#[test]
fn fail_emits_response_failed_and_is_terminal() {
    let mut st = state();
    st.map_event(&ev_message_start());
    let failed = st.fail(&AppError::Internal("boom".to_string()));
    assert_eq!(failed.len(), 1);
    match &failed[0] {
        ResponseStreamEvent::Failed { response, .. } => {
            assert_eq!(response.status, "failed");
            assert!(response.error.is_some());
        }
        other => panic!("expected response.failed, got {other:?}"),
    }
    // After failure, finish() is a no-op (no spurious completed).
    assert!(st.finish().is_empty());
}

/// The error code in a `response.failed` event is mapped by the shared
/// [`crate::error::responses_error`] mapper.
fn failed_error_code(st: &mut ResponsesStreamState, err: &AppError) -> String {
    let events = st.fail(err);
    assert_eq!(events.len(), 1);
    match &events[0] {
        ResponseStreamEvent::Failed { response, .. } => response
            .error
            .as_ref()
            .and_then(|e| e["code"].as_str())
            .expect("response.error.code present")
            .to_string(),
        other => panic!("expected response.failed, got {other:?}"),
    }
}

/// A simulated Bedrock `ThrottlingException` produces a `response.failed`
/// event carrying `rate_limit_exceeded` (MUST DO scenario 1).
#[test]
fn fail_throttling_carries_rate_limit_exceeded() {
    let mut st = state();
    st.map_event(&ev_message_start());
    let err = from_bedrock_sdk_error(&meta("ThrottlingException", "slow down"));
    assert_eq!(failed_error_code(&mut st, &err), "rate_limit_exceeded");
}

/// A simulated `ValidationException` maps to `context_length_exceeded`
/// inside the `response.failed` event (MUST DO scenario 2).
#[test]
fn fail_validation_carries_context_length_exceeded() {
    let mut st = state();
    st.map_event(&ev_message_start());
    let err = from_bedrock_sdk_error(&meta("ValidationException", "too long"));
    assert_eq!(failed_error_code(&mut st, &err), "context_length_exceeded");
}

/// A simulated 5xx / `ServiceUnavailableException` maps to
/// `server_is_overloaded` inside the `response.failed` event (MUST DO
/// scenario 2).
#[test]
fn fail_service_unavailable_carries_server_is_overloaded() {
    let mut st = state();
    st.map_event(&ev_message_start());
    let err = from_bedrock_sdk_error(&meta("ServiceUnavailableException", "down"));
    assert_eq!(failed_error_code(&mut st, &err), "server_is_overloaded");
}

/// MUST DO scenario 4: a mid-stream error emits exactly one
/// `response.failed` event and the stream terminates cleanly (the
/// connection is NOT dropped — `finish()` afterwards is a no-op rather than
/// silently truncating).
#[test]
fn mid_stream_error_emits_failed_not_dropped_stream() {
    let mut st = state();
    st.map_event(&ev_message_start());
    st.map_event(&ev_text("partial", 0));
    let err = from_bedrock_sdk_error(&meta("InternalServerException", "kaboom"));
    let events = st.fail(&err);
    assert_eq!(events.len(), 1, "exactly one response.failed, not a drop");
    assert!(matches!(events[0], ResponseStreamEvent::Failed { .. }));
    assert!(
        st.finish().is_empty(),
        "no spurious completed after failure"
    );
}

/// Build a Bedrock-style [`ErrorMetadata`] for the failure-mapping tests.
fn meta(code: &str, message: &str) -> aws_smithy_types::error::metadata::ErrorMetadata {
    aws_smithy_types::error::metadata::ErrorMetadata::builder()
        .code(code)
        .message(message)
        .build()
}

// -- content-filter stop reason → incomplete status ----------------------

/// A `content_filtered` stop reason maps the terminal Response to
/// `status == "incomplete"` with `incomplete_details.reason ==
/// "content_filter"` (reusing the non-stream mapper). The stream still
/// terminates on `response.completed` — there is NO separate refusal stream
/// event from the Bedrock state machine.
#[test]
fn content_filter_stop_reason_yields_incomplete_event() {
    let events = drive(&[
        ev_message_start(),
        ev_text("partial", 0),
        ev_message_stop(StopReason::ContentFiltered),
        ev_metadata(5, 1, 6),
    ]);

    // Terminates on response.completed (the lifecycle envelope), not a
    // refusal event.
    match events.last().expect("completed") {
        ResponseStreamEvent::Incomplete { response, .. } => {
            assert_eq!(response.status, "incomplete");
            let details = response
                .incomplete_details
                .as_ref()
                .expect("incomplete_details present");
            assert_eq!(details["reason"], "content_filter");
        }
        other => panic!("expected response.incomplete last, got {other:?}"),
    }
    assert_no_done_no_argdelta(&events);
}

// -- refusal stream events are NEVER emitted by the state machine ---------

/// NEGATIVE LOCK: the Bedrock state machine NEVER emits
/// `response.refusal.delta` / `response.refusal.done` events. The refusal
/// stream-event variants exist in the schema for SDK/client compatibility
/// only (a content filter surfaces as an `incomplete` completed Response,
/// not a refusal stream event). This guards against a future change that
/// would start emitting refusal events and break the codex/ai-sdk contract.
#[test]
fn state_machine_never_emits_refusal_stream_events() {
    let cases: Vec<Vec<ConverseStreamOutput>> = vec![
        // text path
        vec![
            ev_message_start(),
            ev_text("hi", 0),
            ev_message_stop(StopReason::EndTurn),
            ev_metadata(1, 1, 2),
        ],
        // content-filtered path
        vec![
            ev_message_start(),
            ev_text("x", 0),
            ev_message_stop(StopReason::ContentFiltered),
            ev_metadata(1, 1, 2),
        ],
        // tool path
        vec![
            ev_message_start(),
            ev_tool_start(0, "c1", "f"),
            ev_tool_input(0, "{}"),
            ev_block_stop(0),
            ev_message_stop(StopReason::ToolUse),
            ev_metadata(1, 1, 2),
        ],
    ];
    for events in cases {
        let driven = drive(&events);
        for ev in &driven {
            let s = serde_json::to_string(ev).unwrap();
            assert!(
                !s.contains("response.refusal"),
                "state machine unexpectedly emitted a refusal event: {s}"
            );
        }
    }
}

// -- T3 regression: typed-stream lifecycle locked for the T5 seam refactor

/// CHARACTERIZATION LOCK (T3): pins TODAY's typed Converse→Responses SSE
/// streaming lifecycle so the later raw-bytes seam refactor (T5) cannot
/// silently regress it. This drives the CURRENT
/// [`ResponsesStreamState`] / [`converse_stream_to_openai_responses`] path
/// (via the same `drive()` harness the async wrapper mirrors) over a
/// representative reasoning+text+tool response and asserts, using the
/// [`ResponseStreamEvent::event_type`] wire tags:
///
/// 1. the exact ordered `event_type()` sequence,
/// 2. the terminal event is `response.completed`, and
/// 3. ZERO `[DONE]` sentinels appear anywhere (the Responses surface, unlike
///    chat, never emits `[DONE]` — AGENTS.md §9).
///
/// If T5 reorders, drops, or appends any lifecycle event — or leaks a
/// `[DONE]` — this test fails, flagging a behavior change for review.
#[test]
fn converse_responses_stream_lifecycle_locked() {
    // A representative response exercising reasoning, then text, then a
    // tool call — the full lifecycle fan-out in one stream.
    let events = drive(&[
        ev_message_start(),
        ev_reasoning_text("thinking", 0),
        ev_text("Answer", 1),
        ev_block_stop(1),
        ev_tool_start(2, "call-1", "get_weather"),
        ev_tool_input(2, "{\"city\":\"Paris\"}"),
        ev_block_stop(2),
        ev_message_stop(StopReason::ToolUse),
        ev_metadata(20, 10, 30),
    ]);

    // (a) The exact ordered wire-type sequence, read straight off
    // ResponseStreamEvent::event_type() (the tag the SSE frame carries).
    let tags: Vec<&str> = events.iter().map(ResponseStreamEvent::event_type).collect();
    assert_eq!(
        tags,
        vec![
            "response.created",
            "response.in_progress",
            // reasoning item lifecycle (structured, NOT <think>).
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.done",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            // message item lifecycle.
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            // function_call item: add + argument lifecycle + done.
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            // terminal envelope.
            "response.completed",
        ],
        "typed Responses stream lifecycle order changed — T5 seam refactor must preserve this"
    );

    // (b) The terminal event MUST be response.completed.
    assert_eq!(
        events
            .last()
            .map(ResponseStreamEvent::event_type)
            .expect("at least one event"),
        "response.completed",
        "Responses stream MUST terminate on response.completed"
    );
    assert!(
        matches!(events.last(), Some(ResponseStreamEvent::Completed { .. })),
        "terminal event must be the Completed variant"
    );

    // (c) ZERO [DONE] sentinels anywhere — neither in the wire tags nor in
    // any serialized event payload (the Responses surface never emits one).
    assert!(
        !tags.contains(&"[DONE]"),
        "no event_type may be the [DONE] sentinel"
    );
    for ev in &events {
        let s = serde_json::to_string(ev).expect("event serializes");
        assert!(
            !s.contains("[DONE]"),
            "[DONE] sentinel leaked into the Responses stream: {s}"
        );
    }

    // Sanity: the lock guards a single terminal completion, not many.
    let completed = tags.iter().filter(|t| **t == "response.completed").count();
    assert_eq!(completed, 1, "exactly one response.completed expected");
}

// -- #8: open tool block at truncation is flushed through finish() -------

/// A tool block that receives input but no `contentBlockStop` before a
/// truncation must never be marked completed or become executable.
#[test]
fn tool_block_open_at_stop_remains_incomplete() {
    let events = drive(&[
        ev_message_start(),
        ev_tool_start(0, "call-open", "get_weather"),
        ev_tool_input(0, "{\"city\":\"Paris\"}"),
        ev_message_stop(StopReason::MaxTokens),
        ev_metadata(12, 6, 18),
    ]);

    let types: Vec<String> = events.iter().map(type_of).collect();
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.incomplete",
        ],
        "truncated tool block must not emit output_item.done"
    );

    assert_monotonic_from_zero(&events);
    assert_no_done_no_argdelta(&events);

    assert!(!events
        .iter()
        .any(|event| matches!(event, ResponseStreamEvent::OutputItemDone { .. })));

    // The completed Response carries the tool call and the incomplete
    // status from the max_tokens truncation.
    match events.last().unwrap() {
        ResponseStreamEvent::Incomplete { response, .. } => {
            assert_eq!(response.status, "incomplete");
            assert_eq!(
                response.incomplete_details.as_ref().expect("details")["reason"],
                "max_output_tokens"
            );
            match &response.output[0] {
                ResponseOutputItem::FunctionCall { call_id, .. } => {
                    assert_eq!(call_id, "call-open");
                }
                other => panic!("expected function_call in final output, got {other:?}"),
            }
        }
        other => panic!("expected incomplete, got {other:?}"),
    }
}

// -- finish() is idempotent ----------------------------------------------

#[test]
fn finish_is_idempotent() {
    let mut st = state();
    st.map_event(&ev_message_start());
    let first = st.finish();
    assert!(!first.is_empty());
    assert!(st.finish().is_empty(), "second finish must be empty");
}
