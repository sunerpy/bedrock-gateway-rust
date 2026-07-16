//! Unit and property-based tests for the streaming state machine.
//!
//! Relocated out of `stream.rs` for code organization (see the
//! `test-coverage-codecov` spec, task 5.2). The original inline `#[test]`
//! functions are preserved verbatim as FLAT functions here; `use super::*;`
//! resolves to the implementation `stream` module exactly as the inline
//! `mod tests` did before the move. New augmenting unit tests and a
//! property-based `prop_tests` submodule are appended below.
//!
//! Golden streaming parity (Bedrock event stream → OpenAI SSE chunks, driven
//! through the REAL `StreamState::map_event` + `assert_stream_eq`) lives in the
//! offline corpus at `tests/golden/corpus.rs` (`stream_*` cases). The async
//! wrapper [`converse_stream_to_openai`] is intentionally not unit-tested: its
//! `EventReceiver` cannot be constructed outside the SDK crate (its `new` is
//! `pub(crate)`), so the entire parity surface is exercised through the pure
//! `map_event` state machine instead — see the module docs on `stream.rs`.
//!
//! All tests run OFFLINE with no AWS credentials and no `sleep`/wait: the state
//! machine is driven with in-memory SDK events, never timers.

use super::*;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent, ConversationRole,
    MessageStartEvent, MessageStopEvent, ReasoningContentBlockDelta, StopReason, TokenUsage,
    ToolUseBlockDelta, ToolUseBlockStart,
};

const MODEL: &str = "anthropic.claude-3-sonnet-20240229-v1:0";
const MID: &str = "chatcmpl-test";
const RID: &str = "req-test";

fn t0() -> Instant {
    Instant::now()
}

// -- builders for SDK events ---------------------------------------------

fn ev_message_start(role: ConversationRole) -> ConverseStreamOutput {
    ConverseStreamOutput::MessageStart(MessageStartEvent::builder().role(role).build().unwrap())
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

fn ev_reasoning_signature(sig: &str, block: i32) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::Signature(sig.to_string()),
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

fn ev_metadata(
    input: i32,
    output: i32,
    total: i32,
    cache_read: Option<i32>,
    cache_write: Option<i32>,
) -> ConverseStreamOutput {
    let mut usage = TokenUsage::builder()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(total);
    if let Some(r) = cache_read {
        usage = usage.cache_read_input_tokens(r);
    }
    if let Some(w) = cache_write {
        usage = usage.cache_write_input_tokens(w);
    }
    ConverseStreamOutput::Metadata(
        aws_sdk_bedrockruntime::types::ConverseStreamMetadataEvent::builder()
            .usage(usage.build().unwrap())
            .build(),
    )
}

// -- messageStart → role delta -------------------------------------------

#[test]
fn message_start_emits_role_delta() {
    let mut st = StreamState::new();
    let chunk = st
        .map_event(
            &ev_message_start(ConversationRole::Assistant),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("role chunk");
    assert_eq!(chunk.object, "chat.completion.chunk");
    assert_eq!(chunk.id, MID);
    assert_eq!(chunk.model, MODEL);
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some(""));
    assert!(chunk.choices[0].finish_reason.is_none());
    assert!(chunk.usage.is_none());
}

// -- plain text stream ----------------------------------------------------

#[test]
fn text_delta_emits_content() {
    let mut st = StreamState::new();
    let chunk = st
        .map_event(&ev_text("Hello", 0), MODEL, MID, true, RID, t0())
        .expect("text");
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
    assert!(chunk.choices[0].delta.role.is_none());
    // No <think> open, so no closing tag is prepended.
    assert!(!chunk.choices[0]
        .delta
        .content
        .as_deref()
        .unwrap()
        .contains("</think>"));
}

// -- reasoning open/close <think> transitions ----------------------------

#[test]
fn reasoning_opens_think_then_text_closes_it() {
    let mut st = StreamState::new();

    // First reasoning delta opens <think>.
    let c1 = st
        .map_event(
            &ev_reasoning_text("let me think", 0),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("r1");
    assert_eq!(
        c1.choices[0].delta.content.as_deref(),
        Some("<think>let me think")
    );
    assert!(st.think_emitted);

    // Subsequent reasoning delta does NOT re-open.
    let c2 = st
        .map_event(&ev_reasoning_text(" more", 0), MODEL, MID, true, RID, t0())
        .expect("r2");
    assert_eq!(c2.choices[0].delta.content.as_deref(), Some(" more"));
    assert!(st.think_emitted);

    // Transition to regular text prepends </think>.
    let c3 = st
        .map_event(&ev_text("the answer", 0), MODEL, MID, true, RID, t0())
        .expect("t");
    assert_eq!(
        c3.choices[0].delta.content.as_deref(),
        Some("</think>the answer")
    );
    assert!(!st.think_emitted);
}

#[test]
fn signature_closes_open_think() {
    let mut st = StreamState::new();
    st.map_event(&ev_reasoning_text("hmm", 0), MODEL, MID, true, RID, t0())
        .expect("open");
    assert!(st.think_emitted);
    let c = st
        .map_event(
            &ev_reasoning_signature("sig-abc", 0),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("sig closes");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("</think>"));
    assert!(!st.think_emitted);
}

#[test]
fn signature_without_open_think_is_skipped() {
    let mut st = StreamState::new();
    // No <think> open → signature yields nothing.
    let out = st.map_event(
        &ev_reasoning_signature("sig", 0),
        MODEL,
        MID,
        true,
        RID,
        t0(),
    );
    assert!(out.is_none());
    assert!(!st.think_emitted);
}

#[test]
fn message_stop_closes_open_think_first() {
    let mut st = StreamState::new();
    st.map_event(
        &ev_reasoning_text("thinking", 0),
        MODEL,
        MID,
        true,
        RID,
        t0(),
    )
    .expect("open");
    assert!(st.think_emitted);
    // messageStop while think open closes the tag and retains the terminal reason.
    let c = st
        .map_event(
            &ev_message_stop(StopReason::EndTurn),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("close");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("</think>"));
    assert_eq!(c.choices[0].finish_reason.as_deref(), Some("stop"));
    assert!(!st.think_emitted);
}

// -- messageStop finish_reason -------------------------------------------

#[test]
fn message_stop_emits_finish_reason() {
    let mut st = StreamState::new();
    let c = st
        .map_event(
            &ev_message_stop(StopReason::EndTurn),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("stop");
    assert_eq!(c.choices[0].finish_reason.as_deref(), Some("stop"));
    // delta is empty.
    assert!(c.choices[0].delta.content.is_none());
    assert!(c.choices[0].delta.role.is_none());
}

#[test]
fn message_stop_tool_use_maps_to_tool_calls() {
    let mut st = StreamState::new();
    let c = st
        .map_event(
            &ev_message_stop(StopReason::ToolUse),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("stop");
    assert_eq!(c.choices[0].finish_reason.as_deref(), Some("tool_calls"));
}

#[test]
fn message_stop_max_tokens_maps_to_length() {
    let mut st = StreamState::new();
    let c = st
        .map_event(
            &ev_message_stop(StopReason::MaxTokens),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("stop");
    assert_eq!(c.choices[0].finish_reason.as_deref(), Some("length"));
}

// -- #8: open tool block at truncation is not dropped --------------------

#[test]
fn tool_block_open_at_stop_is_flushed() {
    let mut st = StreamState::new();
    let mut chunks = Vec::new();
    // contentBlockStart(toolUse) → contentBlockDelta(input) → messageStop
    // (max_tokens) with NO contentBlockStop. Each tool piece is emitted as
    // its own chunk at arrival, so a missing contentBlockStop drops nothing.
    let events = vec![
        ev_tool_start(0, "call-open", "get_weather"),
        ev_tool_input(0, "{\"city\":\"Paris\"}"),
        ev_message_stop(StopReason::MaxTokens),
    ];
    for e in &events {
        if let Some(c) = st.map_event(e, MODEL, MID, true, RID, t0()) {
            chunks.push(c);
        }
    }

    // The tool-call delta chunk(s) carry the id/name and the accumulated
    // arguments.
    let tool_chunks: Vec<&ChatStreamResponse> = chunks
        .iter()
        .filter(|c| {
            c.choices
                .first()
                .is_some_and(|ch| ch.delta.tool_calls.is_some())
        })
        .collect();
    assert!(
        !tool_chunks.is_empty(),
        "at least one tool-call delta chunk must be emitted"
    );
    let start = tool_chunks[0].choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(start[0].id.as_deref(), Some("call-open"));
    assert_eq!(start[0].function.name.as_deref(), Some("get_weather"));
    let frag = tool_chunks[1].choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(frag[0].function.arguments, "{\"city\":\"Paris\"}");

    // The finish chunk carries finish_reason "length" (max_tokens).
    let finish = chunks
        .iter()
        .find(|c| {
            c.choices
                .first()
                .and_then(|ch| ch.finish_reason.as_deref())
                .is_some()
        })
        .expect("a finish chunk");
    assert_eq!(finish.choices[0].finish_reason.as_deref(), Some("length"));
}

// -- content block stop is inert -----------------------------------------

#[test]
fn content_block_stop_yields_nothing() {
    let mut st = StreamState::new();
    assert!(st
        .map_event(&ev_block_stop(0), MODEL, MID, true, RID, t0())
        .is_none());
}

// -- tool-call streaming (start + delta) ---------------------------------

#[test]
fn tool_call_start_then_input_fragments() {
    let mut st = StreamState::new();

    // contentBlockStart → initial tool_call delta with id/name, empty args.
    let start = st
        .map_event(
            &ev_tool_start(1, "tool-xyz", "get_weather"),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("start");
    let calls = start.choices[0]
        .delta
        .tool_calls
        .as_ref()
        .expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].index, Some(0)); // first synthesized index is 0
    assert_eq!(calls[0].id.as_deref(), Some("tool-xyz"));
    assert_eq!(calls[0].r#type, "function");
    assert_eq!(calls[0].function.name.as_deref(), Some("get_weather"));
    assert_eq!(calls[0].function.arguments, "");

    // input fragment → arguments delta, reuses id/name from the block.
    let frag = st
        .map_event(&ev_tool_input(1, "{\"city\":"), MODEL, MID, true, RID, t0())
        .expect("frag1");
    let calls = frag.choices[0].delta.tool_calls.as_ref().expect("tc");
    assert_eq!(calls[0].index, Some(0));
    assert_eq!(calls[0].id.as_deref(), Some("tool-xyz"));
    assert_eq!(calls[0].function.name.as_deref(), Some("get_weather"));
    assert_eq!(calls[0].function.arguments, "{\"city\":");

    let frag2 = st
        .map_event(&ev_tool_input(1, "\"Paris\"}"), MODEL, MID, true, RID, t0())
        .expect("frag2");
    let calls = frag2.choices[0].delta.tool_calls.as_ref().expect("tc");
    assert_eq!(calls[0].function.arguments, "\"Paris\"}");
}

#[test]
fn tool_call_indices_are_contiguous_across_blocks() {
    let mut st = StreamState::new();
    // Bedrock uses block indices 3 and 5; OpenAI indices must be 0 and 1.
    let a = st
        .map_event(&ev_tool_start(3, "id-a", "fa"), MODEL, MID, true, RID, t0())
        .expect("a");
    let b = st
        .map_event(&ev_tool_start(5, "id-b", "fb"), MODEL, MID, true, RID, t0())
        .expect("b");
    assert_eq!(
        a.choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
        Some(0)
    );
    assert_eq!(
        b.choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
        Some(1)
    );

    // A later input fragment on block 3 keeps index 0.
    let frag = st
        .map_event(&ev_tool_input(3, "{}"), MODEL, MID, true, RID, t0())
        .expect("frag");
    assert_eq!(
        frag.choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
        Some(0)
    );
}

#[test]
fn tool_input_without_prior_start_has_no_id_or_name() {
    let mut st = StreamState::new();
    // No contentBlockStart was seen for this block.
    let frag = st
        .map_event(&ev_tool_input(7, "{\"a\":1}"), MODEL, MID, true, RID, t0())
        .expect("frag");
    let calls = frag.choices[0].delta.tool_calls.as_ref().expect("tc");
    assert_eq!(calls[0].index, Some(0)); // freshly allocated contiguous index
    assert!(calls[0].id.is_none());
    assert!(calls[0].function.name.is_none());
    assert_eq!(calls[0].function.arguments, "{\"a\":1}");
}

// -- metadata usage chunk: include_usage on/off --------------------------

#[test]
fn metadata_usage_chunk_with_include_usage() {
    let mut st = StreamState::new();
    // Rebuild-from-parts: input=9, cacheRead=4, cacheWrite=7, output=5.
    // prompt = 9+4+7 = 20; total = 20+5 = 25.
    let c = st
        .map_event(
            &ev_metadata(9, 5, 20, Some(4), Some(7)),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("usage chunk");
    // Empty choices for the usage chunk (per OpenAI doc).
    assert!(c.choices.is_empty());
    let usage = c.usage.as_ref().expect("usage present");
    assert_eq!(usage.prompt_tokens, 20);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 25);
    // cached_tokens reflects READ only.
    let details = usage.prompt_tokens_details.as_ref().expect("cache details");
    assert_eq!(details.cached_tokens, 4);
    // cacheWrite never surfaces.
    let json = serde_json::to_string(&c).expect("serialize");
    assert!(!json.to_lowercase().contains("cache_write"));
}

#[test]
fn metadata_usage_chunk_suppressed_without_include_usage() {
    let mut st = StreamState::new();
    let out = st.map_event(
        &ev_metadata(9, 5, 20, None, None),
        MODEL,
        MID,
        false,
        RID,
        t0(),
    );
    assert!(out.is_none(), "no usage chunk when include_usage is false");
}

#[test]
fn metadata_no_cache_omits_prompt_details() {
    let mut st = StreamState::new();
    let c = st
        .map_event(
            &ev_metadata(8, 2, 10, None, None),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("usage");
    let usage = c.usage.as_ref().expect("usage");
    assert!(usage.prompt_tokens_details.is_none());
    // No reasoning seen → no completion details.
    assert!(usage.completion_tokens_details.is_none());
}

#[test]
fn reasoning_tokens_patched_into_usage_chunk() {
    let mut st = StreamState::new();
    // Reasoning deltas accumulate tokens.
    st.map_event(
        &ev_reasoning_text("a long chain of thought reasoning step", 0),
        MODEL,
        MID,
        true,
        RID,
        t0(),
    );
    st.map_event(
        &ev_reasoning_text(" continuing the analysis carefully", 0),
        MODEL,
        MID,
        true,
        RID,
        t0(),
    );
    assert!(st.reasoning_tokens > 0);
    let captured = st.reasoning_tokens;

    let c = st
        .map_event(
            &ev_metadata(10, 6, 30, None, None),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("usage");
    let details = c
        .usage
        .as_ref()
        .unwrap()
        .completion_tokens_details
        .as_ref()
        .expect("reasoning patched");
    assert_eq!(details.reasoning_tokens, captured as i32);
    assert_eq!(details.audio_tokens, 0);
}

// -- full sequence: serialized wire shape parity -------------------------

#[test]
fn full_text_sequence_serializes_clean() {
    let mut st = StreamState::new();
    let events = vec![
        ev_message_start(ConversationRole::Assistant),
        ev_text("Hi", 0),
        ev_block_stop(0),
        ev_message_stop(StopReason::EndTurn),
        ev_metadata(8, 2, 10, None, None),
    ];
    let mut chunks = Vec::new();
    for e in &events {
        if let Some(c) = st.map_event(e, MODEL, MID, true, RID, t0()) {
            chunks.push(c);
        }
    }
    // role, content, finish_reason, usage = 4 chunks (block_stop is inert).
    assert_eq!(chunks.len(), 4);
    assert_eq!(
        chunks[0].choices[0].delta.role.as_deref(),
        Some("assistant")
    );
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    assert!(chunks[3].choices.is_empty());
    assert!(chunks[3].usage.is_some());

    // Every chunk is a valid chat.completion.chunk with no reasoning_content leak.
    for c in &chunks {
        let json = serde_json::to_string(c).expect("serialize");
        assert!(json.contains("chat.completion.chunk"));
        assert!(!json.contains("reasoning_content"));
    }
}

#[test]
fn full_reasoning_sequence_think_wrapped() {
    let mut st = StreamState::new();
    let events = vec![
        ev_message_start(ConversationRole::Assistant),
        ev_reasoning_text("step1", 0),
        ev_reasoning_text(" step2", 0),
        ev_text("final answer", 1),
        ev_message_stop(StopReason::EndTurn),
    ];
    let mut content = String::new();
    for e in &events {
        if let Some(c) = st.map_event(e, MODEL, MID, true, RID, t0()) {
            if let Some(text) = &c.choices.first().and_then(|ch| ch.delta.content.clone()) {
                content.push_str(text);
            }
        }
    }
    // Reasoning is wrapped and closed before final text.
    assert_eq!(content, "<think>step1 step2</think>final answer");
}

// ===========================================================================
// Augmenting unit tests (task 5.2) — edge cases in the pure state machine.
// ===========================================================================

#[test]
fn new_and_default_are_equivalent_fresh_state() {
    let a = StreamState::new();
    let b = StreamState::default();
    assert_eq!(a.think_emitted, b.think_emitted);
    assert_eq!(a.next_tool_index, b.next_tool_index);
    assert_eq!(a.reasoning_tokens, b.reasoning_tokens);
    assert!(a.tool_meta.is_empty() && b.tool_meta.is_empty());
}

#[test]
fn redacted_reasoning_delta_yields_nothing_and_leaves_think_closed() {
    let mut st = StreamState::new();
    let ev = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ReasoningContent(
                ReasoningContentBlockDelta::RedactedContent(aws_smithy_types::Blob::new(
                    b"secret".to_vec(),
                )),
            ))
            .content_block_index(0)
            .build()
            .unwrap(),
    );
    // redactedContent is not rendered; no <think> is opened.
    assert!(st.map_event(&ev, MODEL, MID, true, RID, t0()).is_none());
    assert!(!st.think_emitted);
}

#[test]
fn tool_result_content_delta_is_inert() {
    let mut st = StreamState::new();
    // A toolResult delta has no OpenAI mapping in the stream → None.
    let ev = ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .delta(ContentBlockDelta::ToolResult(Vec::new()))
            .content_block_index(0)
            .build()
            .unwrap(),
    );
    assert!(st.map_event(&ev, MODEL, MID, true, RID, t0()).is_none());
}

#[test]
fn empty_reasoning_text_still_opens_think() {
    let mut st = StreamState::new();
    // Empty reasoning text estimates 0 tokens but still opens the <think> tag.
    let c = st
        .map_event(&ev_reasoning_text("", 0), MODEL, MID, true, RID, t0())
        .expect("opens think");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("<think>"));
    assert!(st.think_emitted);
    assert_eq!(st.reasoning_tokens, 0);
}

#[test]
fn metadata_cache_write_only_still_emits_prompt_details_with_zero_cached() {
    let mut st = StreamState::new();
    // cacheRead absent, cacheWrite present → cached_tokens is 0 but the
    // prompt_tokens_details block is still emitted (cache_write > 0 branch).
    let c = st
        .map_event(
            &ev_metadata(10, 3, 60, None, Some(50)),
            MODEL,
            MID,
            true,
            RID,
            t0(),
        )
        .expect("usage");
    let usage = c.usage.as_ref().expect("usage");
    // prompt = input(10) + cacheRead(0) + cacheWrite(50) = 60; total = 63.
    assert_eq!(usage.prompt_tokens, 60);
    assert_eq!(usage.total_tokens, 63);
    let details = usage
        .prompt_tokens_details
        .as_ref()
        .expect("details present via cache_write branch");
    assert_eq!(details.cached_tokens, 0);
    // cacheWrite is folded into prompt/total but never a distinct wire field.
    let json = serde_json::to_string(&c).expect("serialize");
    assert!(!json.to_lowercase().contains("cache_write"));
}

#[test]
fn signature_before_any_reasoning_is_inert_then_text_is_plain() {
    let mut st = StreamState::new();
    // Signature with no open <think> → nothing.
    assert!(st
        .map_event(&ev_reasoning_signature("s", 0), MODEL, MID, true, RID, t0())
        .is_none());
    // A following plain-text delta is emitted verbatim (no stray </think>).
    let c = st
        .map_event(&ev_text("plain", 0), MODEL, MID, true, RID, t0())
        .expect("text");
    assert_eq!(c.choices[0].delta.content.as_deref(), Some("plain"));
}

// ===========================================================================
// Property-based tests (task 5.2) — universal invariants of the state machine.
//
// Feature: test-coverage-codecov. These support the Requirement 1.2 coverage
// goal for `bedrock/stream.rs`; they are coverage-supporting properties of the
// pure state machine, not one of the design's numbered Correctness Properties
// (1–4). All run offline with in-memory SDK events and no `sleep`.
//
// Validates: Requirements 1.2
// ===========================================================================
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    /// A compact, generatable stand-in for the reasoning/text event alphabet.
    #[derive(Debug, Clone)]
    enum Ev {
        Reasoning(String),
        Text(String),
        Signature,
    }

    /// Text/reasoning payloads drawn from an alphabet that CANNOT contain the
    /// literal `<think>` / `</think>` tags, so the only tags in the emitted
    /// content come from the state machine itself.
    fn safe_text() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9 ]{0,24}"
    }

    fn ev_strategy() -> impl Strategy<Value = Ev> {
        prop_oneof![
            safe_text().prop_map(Ev::Reasoning),
            safe_text().prop_map(Ev::Text),
            Just(Ev::Signature),
        ]
    }

    fn to_output(ev: &Ev) -> ConverseStreamOutput {
        match ev {
            Ev::Reasoning(s) => ev_reasoning_text(s, 0),
            Ev::Text(s) => ev_text(s, 0),
            Ev::Signature => ev_reasoning_signature("sig", 0),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov — `<think>` tags are always balanced.
        ///
        /// For ANY sequence of reasoning/text/signature deltas terminated by a
        /// `messageStop`, the concatenation of every emitted `content` string
        /// contains equal numbers of `<think>` and `</think>` markers, the
        /// closes never run ahead of the opens at any prefix, and the machine
        /// ends with no `<think>` left open.
        #[test]
        fn prop_think_tags_are_balanced(seq in prop::collection::vec(ev_strategy(), 0..24)) {
            let mut st = StreamState::new();
            let mut content = String::new();
            for ev in &seq {
                if let Some(c) =
                    st.map_event(&to_output(ev), MODEL, MID, true, RID, t0())
                {
                    if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.clone()) {
                        content.push_str(&text);
                    }
                }
            }
            // Terminal messageStop closes any still-open <think>.
            if let Some(c) =
                st.map_event(&ev_message_stop(StopReason::EndTurn), MODEL, MID, true, RID, t0())
            {
                if let Some(text) = c.choices.first().and_then(|ch| ch.delta.content.clone()) {
                    content.push_str(&text);
                }
            }

            let opens = content.matches("<think>").count();
            let closes = content.matches("</think>").count();
            prop_assert_eq!(opens, closes, "unbalanced think tags in {:?}", content);
            prop_assert!(!st.think_emitted, "think left open after messageStop");

            // Prefix well-formedness: at no point does a close precede its open.
            // Scan marker-by-marker, always advancing to the EARLIEST of the
            // next open / next close (note `</think>` also contains `think` but
            // NOT the literal `<think>` substring, so the two never alias).
            let mut depth: i32 = 0;
            let mut rest = content.as_str();
            loop {
                let open_at = rest.find("<think>");
                let close_at = rest.find("</think>");
                let (is_close, pos) = match (open_at, close_at) {
                    (Some(o), Some(c)) if c < o => (true, c),
                    (Some(o), _) => (false, o),
                    (None, Some(c)) => (true, c),
                    (None, None) => break,
                };
                if is_close {
                    depth -= 1;
                    prop_assert!(depth >= 0, "close before open in {:?}", content);
                    rest = &rest[pos + "</think>".len()..];
                } else {
                    depth += 1;
                    rest = &rest[pos + "<think>".len()..];
                }
            }
            prop_assert_eq!(depth, 0);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov — plain text passes through verbatim.
        ///
        /// On a FRESH state (no open `<think>`), a single text delta is emitted
        /// unchanged: same content, no role, no tool_calls, no finish_reason.
        #[test]
        fn prop_text_passthrough_on_fresh_state(text in safe_text()) {
            let mut st = StreamState::new();
            let c = st
                .map_event(&ev_text(&text, 0), MODEL, MID, true, RID, t0())
                .expect("text chunk");
            prop_assert_eq!(c.choices[0].delta.content.as_deref(), Some(text.as_str()));
            prop_assert!(c.choices[0].delta.role.is_none());
            prop_assert!(c.choices[0].delta.tool_calls.is_none());
            prop_assert!(c.choices[0].finish_reason.is_none());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov — synthesized tool indices are a
        /// contiguous 0..n in first-seen order.
        ///
        /// For ANY sequence of `contentBlockStart(toolUse)` events on arbitrary
        /// Bedrock block indices, the OpenAI tool-call `index` assigned to the
        /// k-th DISTINCT block equals k, and a repeated block reuses its index.
        #[test]
        fn prop_tool_indices_contiguous_in_first_seen_order(
            blocks in prop::collection::vec(0i32..12, 1..16)
        ) {
            let mut st = StreamState::new();

            // Expected index = position of first occurrence in the dedup order.
            let mut first_seen: Vec<i32> = Vec::new();
            for &b in &blocks {
                if !first_seen.contains(&b) {
                    first_seen.push(b);
                }
            }

            for (call_no, &block) in blocks.iter().enumerate() {
                let id = format!("id-{call_no}");
                let name = format!("fn-{call_no}");
                let chunk = st
                    .map_event(&ev_tool_start(block, &id, &name), MODEL, MID, true, RID, t0())
                    .expect("tool start chunk");
                let got = chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0]
                    .index
                    .unwrap();
                let expected = first_seen.iter().position(|&x| x == block).unwrap() as i32;
                prop_assert_eq!(got, expected);
            }

            // Distinct indices count equals distinct blocks; next index follows.
            prop_assert_eq!(st.next_tool_index, first_seen.len() as i32);
        }
    }
}
