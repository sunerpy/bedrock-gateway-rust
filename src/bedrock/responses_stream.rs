//! Streaming Bedrock `ConverseStream` → OpenAI **Responses API** semantic-event
//! conversion.
//!
//! The Responses-API analogue of [`crate::bedrock::stream`] (the chat
//! ConverseStream→SSE spine). It turns a Bedrock `ConverseStream` output into a
//! [`crate::domain::ResponsesStream`] of typed [`ResponseStreamEvent`] lifecycle
//! events, via the same record/replay-friendly split the chat path uses: a pure,
//! synchronous state machine ([`ResponsesStreamState`]) plus a thin
//! `async_stream::stream!` wrapper ([`converse_stream_to_openai_responses`]) that
//! only drives `recv().await` and yields what the pure mapper returns.
//!
//! ## Event lifecycle (in order)
//!
//! 1. `response.created` → `response.in_progress` (emitted up-front, on the
//!    first Bedrock event seen — typically `messageStart`).
//! 2. On the first Bedrock `reasoningContent` delta: a `reasoning` output item
//!    (`response.output_item.added` → `response.reasoning_text.delta`* →
//!    `response.reasoning_text.done` → `response.output_item.done`) — emitted
//!    BEFORE the message item. Reasoning is a STRUCTURED item here, NOT the chat
//!    path's `<think>` inline rendering.
//! 3. On the first text delta: a `message` output item
//!    (`response.output_item.added` → `response.content_part.added` →
//!    `response.output_text.delta`* → `response.output_text.done` →
//!    `response.content_part.done` → `response.output_item.done`).
//! 4. Tool calls: a `function_call` output item (`response.output_item.added`
//!    with `item.type == "function_call"` → `response.output_item.done` with the
//!    complete `call_id` + JSON-parseable `arguments`). NO
//!    `function_call_arguments.delta/.done` events are emitted — codex
//!    reconstructs from the item add/done pair, and the variant does not exist in
//!    the T7 schema.
//! 5. `response.completed` carrying the FULL final [`ResponsesResponse`], built
//!    by REUSING [`crate::bedrock::responses_response::from_converse_output_to_responses`]
//!    over a Converse-output-equivalent JSON reconstructed from accumulated
//!    state + the final `usage`.
//!
//! A mid-stream Bedrock error maps to a single `response.failed` event whose
//! `error.code` comes from the shared [`crate::error::responses_error`] mapper
//! (the connection is NOT dropped). There is NO `[DONE]` sentinel — the
//! Responses protocol has none; codex terminates on `response.completed`.
//!
//! ## Purity & testability
//!
//! The intricate logic lives in [`ResponsesStreamState`], a synchronous state
//! machine with no I/O. Each Bedrock event maps to `Vec<ResponseStreamEvent>` via
//! [`ResponsesStreamState::map_event`] (returning a vector because one Bedrock
//! event can fan out into several lifecycle events — e.g. `messageStop` closes
//! the open message item AND completes the response). The async wrapper only
//! drives `recv().await`, so the entire event sequence is unit-tested offline
//! against constructed SDK events.
//!
//! ## De-hardcoding
//!
//! No model-id literals appear here. `model` flows through verbatim from the
//! caller and is echoed into the final `ResponsesResponse`.

use async_stream::stream;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput as ConverseStreamOutputOp;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConverseStreamOutput, ReasoningContentBlockDelta,
};
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::bedrock::responses_response::from_converse_output_to_responses;
use crate::domain::ResponsesStream;
use crate::error::{from_bedrock_sdk_error, AppError};
use crate::openai::responses_schema::{
    OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesRequest, ResponsesResponse,
};

/// The single summary index used for the reasoning summary family. The gateway
/// emits one summary part (index 0) carrying the full reasoning text.
const REASONING_SUMMARY_INDEX: u32 = 0;

/// Per-block tool-call bookkeeping, keyed by Bedrock `contentBlockIndex`.
///
/// Captures the `toolUseId`/`name` at `contentBlockStart`, accumulates the
/// streamed `input` JSON fragments, and records the output item index assigned
/// when the item was added — so the matching `response.output_item.done` carries
/// a complete `call_id` + JSON-parseable `arguments`.
#[derive(Debug, Clone, Default)]
struct ToolAccum {
    /// The Bedrock `toolUseId` (becomes the Responses `call_id`).
    call_id: String,
    /// The function name.
    name: String,
    /// Concatenated JSON input fragments.
    arguments: String,
    /// The `output_index` assigned to this item's `output_item.added`.
    output_index: u32,
}

/// The streaming Responses lifecycle state machine.
///
/// Pure and synchronous: feed every Bedrock `ConverseStreamOutput` event through
/// [`ResponsesStreamState::map_event`] in order; it returns the lifecycle events
/// to emit and mutates internal counters/accumulators. Construct one per stream.
///
/// Tracks `sequence_number` (monotonic from 0), `output_index` (per output
/// item), `content_index` (per content part within an item), and `item_id`s, and
/// accumulates the streamed text/reasoning/tool pieces so the terminal
/// `response.completed` can reuse the non-stream final-Response assembly.
#[derive(Debug)]
pub struct ResponsesStreamState {
    /// Echoed model id (verbatim from the caller; no literal inspection).
    model: String,
    /// The `resp_`-prefixed response id.
    response_id: String,
    /// The originating request (echoed params for the final Response).
    request: ResponsesRequest,

    /// Next monotonic sequence number (emit-only; never validated).
    seq: u64,
    /// Next output item index to assign.
    next_output_index: u32,

    /// Whether `response.created` + `response.in_progress` were already emitted.
    started: bool,

    // --- reasoning item state ---
    /// Whether a reasoning output item is currently open.
    reasoning_open: bool,
    /// The reasoning item's `output_index` (valid while `reasoning_open`).
    reasoning_output_index: u32,
    /// Accumulated reasoning text (for the `reasoning_text.done` + final item).
    reasoning_text: String,

    // --- message item state ---
    /// Whether a message output item is currently open.
    message_open: bool,
    /// The message item's `output_index` (valid while `message_open`).
    message_output_index: u32,
    /// Accumulated output text (for `output_text.done` + the final message).
    message_text: String,

    // --- tool-call accumulators (in arrival order) ---
    /// Bedrock block index → tool-call accumulation.
    tools: Vec<(i32, ToolAccum)>,

    /// Captured `stopReason` from `messageStop` (for the final Response).
    stop_reason: Option<String>,
    /// Captured final usage (inputTokens / outputTokens / cache*), if any.
    usage: Option<Value>,
    /// Whether the response was already finalized (completed/failed emitted).
    finalized: bool,

    /// Gateway request id, stamped onto the terminal completion log.
    request_id: Arc<str>,
    /// Handler-entry instant, used to compute end-to-end `duration_ms`.
    start: Instant,

    /// Content-free per-event-type counts, accumulated as lifecycle events are
    /// emitted. Logged once at debug level on `finish()` so operators can see
    /// the emitted event-type shape (e.g. whether any `output_text.delta` were
    /// produced) without ever logging payload content. Ordered by first emit.
    event_type_counts: Vec<(String, u64)>,
}

impl ResponsesStreamState {
    /// The stable `item_id` for the reasoning item (mirrors the non-stream
    /// mapper's `rs_` derivation).
    fn reasoning_item_id(&self) -> String {
        format!("rs_{}", self.response_id.trim_start_matches("resp_"))
    }

    /// The stable `item_id` for the message item (mirrors the non-stream
    /// mapper's `msg_` derivation).
    fn message_item_id(&self) -> String {
        format!("msg_{}", self.response_id.trim_start_matches("resp_"))
    }

    /// Create a fresh state machine for one stream.
    #[must_use]
    pub fn new(
        model: String,
        response_id: String,
        request: ResponsesRequest,
        request_id: Arc<str>,
        start: Instant,
    ) -> Self {
        Self {
            model,
            response_id,
            request,
            seq: 0,
            next_output_index: 0,
            started: false,
            reasoning_open: false,
            reasoning_output_index: 0,
            reasoning_text: String::new(),
            message_open: false,
            message_output_index: 0,
            message_text: String::new(),
            tools: Vec::new(),
            stop_reason: None,
            usage: None,
            finalized: false,
            request_id,
            start,
            event_type_counts: Vec::new(),
        }
    }

    /// Tally each emitted event by its content-free wire type tag, preserving
    /// first-emit order. Counts only; no payload is inspected or stored.
    fn tally(&mut self, events: &[ResponseStreamEvent]) {
        for ev in events {
            let ty = ev.event_type();
            if let Some(entry) = self.event_type_counts.iter_mut().find(|(k, _)| k == ty) {
                entry.1 += 1;
            } else {
                self.event_type_counts.push((ty.to_string(), 1));
            }
        }
    }

    /// Return the next sequence number and advance the counter (emit-only).
    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    /// A minimal in-progress [`ResponsesResponse`] skeleton used by the
    /// `response.created` / `response.in_progress` envelope events.
    fn in_progress_response(&self) -> ResponsesResponse {
        ResponsesResponse {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: now_unix_secs(),
            status: "in_progress".to_string(),
            output: Vec::new(),
            usage: crate::openai::responses_schema::ResponsesUsage {
                input_tokens: 0,
                input_tokens_details: None,
                output_tokens: 0,
                output_tokens_details: None,
                total_tokens: 0,
            },
            model: self.model.clone(),
            instructions: self.request.instructions.clone(),
            temperature: self.request.temperature,
            top_p: self.request.top_p,
            tool_choice: self.request.tool_choice.clone(),
            tools: self.request.tools.clone(),
            max_output_tokens: self.request.max_output_tokens,
            parallel_tool_calls: self.request.parallel_tool_calls,
            error: None,
            incomplete_details: None,
        }
    }

    /// Emit the leading `response.created` + `response.in_progress` pair exactly
    /// once, at the first event of the stream.
    fn ensure_started(&mut self, out: &mut Vec<ResponseStreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::Created {
            response: self.in_progress_response(),
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::InProgress {
            response: self.in_progress_response(),
            sequence_number: seq,
        });
    }

    /// Map one Bedrock `ConverseStreamOutput` event to zero or more Responses
    /// lifecycle events, mutating the accumulated state.
    ///
    /// The `#[non_exhaustive]` Bedrock enums are matched with a catch-all `_`
    /// arm, so newly-introduced variants are inert.
    pub fn map_event(&mut self, event: &ConverseStreamOutput) -> Vec<ResponseStreamEvent> {
        let mut out = Vec::new();
        self.ensure_started(&mut out);

        match event {
            // messageStart → role known; envelope already started. No item yet
            // (items are opened lazily on first reasoning/text/tool content).
            ConverseStreamOutput::MessageStart(_) => {}

            // contentBlockStart → tool-call start: open a function_call item.
            ConverseStreamOutput::ContentBlockStart(ev) => {
                let block_index = ev.content_block_index();
                if let Some(ContentBlockStart::ToolUse(tu)) = ev.start() {
                    // Close any open text/reasoning items before the tool item.
                    self.close_message_item(&mut out);
                    self.close_reasoning_item(&mut out);

                    let output_index = self.next_output_index;
                    self.next_output_index += 1;
                    let accum = ToolAccum {
                        call_id: tu.tool_use_id().to_string(),
                        name: tu.name().to_string(),
                        arguments: String::new(),
                        output_index,
                    };
                    let seq = self.next_seq();
                    out.push(ResponseStreamEvent::OutputItemAdded {
                        item: function_call_item(&accum, None),
                        output_index,
                        sequence_number: seq,
                    });
                    self.tools.push((block_index, accum));
                }
            }

            // contentBlockDelta → text / reasoning / tool input.
            ConverseStreamOutput::ContentBlockDelta(ev) => {
                let block_index = ev.content_block_index();
                if let Some(delta) = ev.delta() {
                    self.map_content_block_delta(delta, block_index, &mut out);
                }
            }

            // contentBlockStop → finalize the block. For tool blocks this emits
            // the function_call item.done (complete call_id + arguments).
            ConverseStreamOutput::ContentBlockStop(ev) => {
                let block_index = ev.content_block_index();
                self.close_tool_block(block_index, &mut out);
            }

            // messageStop → close open items, capture stopReason.
            ConverseStreamOutput::MessageStop(ev) => {
                self.stop_reason = Some(ev.stop_reason().as_str().to_string());
                self.close_message_item(&mut out);
                self.close_reasoning_item(&mut out);
            }

            // metadata → capture final usage; completed is emitted at recv end.
            ConverseStreamOutput::Metadata(ev) => {
                if let Some(usage) = ev.usage() {
                    self.usage = Some(usage_to_json(usage));
                }
            }

            // #[non_exhaustive] catch-all — newly introduced variants are inert.
            _ => {}
        }

        self.tally(&out);
        out
    }

    /// Handle a `contentBlockDelta` payload (text / reasoning / tool input).
    fn map_content_block_delta(
        &mut self,
        delta: &ContentBlockDelta,
        block_index: i32,
        out: &mut Vec<ResponseStreamEvent>,
    ) {
        match delta {
            // Regular text → open the message item lazily, then emit a delta.
            ContentBlockDelta::Text(text) => {
                // A text block closes any open reasoning item first (reasoning
                // always precedes the message item).
                self.close_reasoning_item(out);
                self.ensure_message_item(out);
                self.message_text.push_str(text);
                let seq = self.next_seq();
                out.push(ResponseStreamEvent::OutputTextDelta {
                    item_id: self.message_item_id(),
                    output_index: self.message_output_index,
                    content_index: 0,
                    delta: text.clone(),
                    sequence_number: seq,
                });
            }

            // Reasoning content → reasoning item, BEFORE the message item.
            // (signature / redactedContent / unknown reasoning deltas: inert.)
            ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(text)) => {
                self.ensure_reasoning_item(out);
                self.reasoning_text.push_str(text);
                let seq = self.next_seq();
                out.push(ResponseStreamEvent::ReasoningTextDelta {
                    item_id: self.reasoning_item_id(),
                    output_index: self.reasoning_output_index,
                    content_index: 0,
                    delta: text.clone(),
                    sequence_number: seq,
                });
                let seq = self.next_seq();
                out.push(ResponseStreamEvent::ReasoningSummaryTextDelta {
                    item_id: self.reasoning_item_id(),
                    output_index: self.reasoning_output_index,
                    summary_index: REASONING_SUMMARY_INDEX,
                    delta: text.clone(),
                    sequence_number: seq,
                });
            }

            // Tool input fragment → accumulate; the item.done is emitted on
            // contentBlockStop / messageStop with the full arguments.
            ContentBlockDelta::ToolUse(tool_delta) => {
                if let Some((_, accum)) = self.tools.iter_mut().find(|(idx, _)| *idx == block_index)
                {
                    accum.arguments.push_str(tool_delta.input());
                }
            }

            // citation / image / toolResult / unknown deltas: inert.
            _ => {}
        }
    }

    /// Open the reasoning output item (added) if not already open.
    ///
    /// Emits BOTH reasoning families: the `output_item.added` (reasoning) frame
    /// that opencode's native parser keys off, AND the
    /// `reasoning_summary_part.added` frame the vercel ai-sdk Responses parser
    /// requires (ai-sdk does not recognize `reasoning_text.*` and would
    /// otherwise drop reasoning to `unknown_chunk`). Both share `summary_index`
    /// 0 and the same `item_id`.
    fn ensure_reasoning_item(&mut self, out: &mut Vec<ResponseStreamEvent>) {
        if self.reasoning_open {
            return;
        }
        self.reasoning_open = true;
        self.reasoning_output_index = self.next_output_index;
        self.next_output_index += 1;
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemAdded {
            item: ResponseOutputItem::Reasoning {
                id: self.reasoning_item_id(),
                content: None,
                summary: Some(json!([])),
            },
            output_index: self.reasoning_output_index,
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ReasoningSummaryPartAdded {
            item_id: self.reasoning_item_id(),
            output_index: self.reasoning_output_index,
            summary_index: REASONING_SUMMARY_INDEX,
            sequence_number: seq,
        });
    }

    /// Close the reasoning item if open: `reasoning_text.done` →
    /// `output_item.done` carrying the accumulated summary text.
    fn close_reasoning_item(&mut self, out: &mut Vec<ResponseStreamEvent>) {
        if !self.reasoning_open {
            return;
        }
        self.reasoning_open = false;
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ReasoningTextDone {
            item_id: self.reasoning_item_id(),
            output_index: self.reasoning_output_index,
            content_index: 0,
            text: self.reasoning_text.clone(),
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ReasoningSummaryTextDone {
            item_id: self.reasoning_item_id(),
            output_index: self.reasoning_output_index,
            summary_index: REASONING_SUMMARY_INDEX,
            text: self.reasoning_text.clone(),
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ReasoningSummaryPartDone {
            item_id: self.reasoning_item_id(),
            output_index: self.reasoning_output_index,
            summary_index: REASONING_SUMMARY_INDEX,
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemDone {
            item: ResponseOutputItem::Reasoning {
                id: self.reasoning_item_id(),
                content: None,
                summary: Some(
                    json!([{ "type": "summary_text", "text": self.reasoning_text.clone() }]),
                ),
            },
            output_index: self.reasoning_output_index,
            sequence_number: seq,
        });
    }

    /// Open the message output item (added + content_part.added) if not open.
    fn ensure_message_item(&mut self, out: &mut Vec<ResponseStreamEvent>) {
        if self.message_open {
            return;
        }
        self.message_open = true;
        self.message_output_index = self.next_output_index;
        self.next_output_index += 1;
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemAdded {
            item: ResponseOutputItem::Message {
                id: self.message_item_id(),
                status: "in_progress".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
            },
            output_index: self.message_output_index,
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ContentPartAdded {
            item_id: self.message_item_id(),
            output_index: self.message_output_index,
            content_index: 0,
            part: OutputContentPart::OutputText {
                text: String::new(),
                annotations: Vec::new(),
                logprobs: None,
            },
            sequence_number: seq,
        });
    }

    /// Close the message item if open: `output_text.done` →
    /// `content_part.done` → `output_item.done` carrying the accumulated text.
    fn close_message_item(&mut self, out: &mut Vec<ResponseStreamEvent>) {
        if !self.message_open {
            return;
        }
        self.message_open = false;
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputTextDone {
            item_id: self.message_item_id(),
            output_index: self.message_output_index,
            content_index: 0,
            text: self.message_text.clone(),
            sequence_number: seq,
        });
        let part = OutputContentPart::OutputText {
            text: self.message_text.clone(),
            annotations: Vec::new(),
            logprobs: None,
        };
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::ContentPartDone {
            item_id: self.message_item_id(),
            output_index: self.message_output_index,
            content_index: 0,
            part: part.clone(),
            sequence_number: seq,
        });
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemDone {
            item: ResponseOutputItem::Message {
                id: self.message_item_id(),
                status: "completed".to_string(),
                role: "assistant".to_string(),
                content: vec![part],
            },
            output_index: self.message_output_index,
            sequence_number: seq,
        });
    }

    /// Close a tool block (on contentBlockStop): emit the function_call
    /// `output_item.done` with the complete `call_id` + accumulated arguments.
    fn close_tool_block(&mut self, block_index: i32, out: &mut Vec<ResponseStreamEvent>) {
        let Some(pos) = self.tools.iter().position(|(idx, _)| *idx == block_index) else {
            return;
        };
        let accum = self.tools[pos].1.clone();
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemDone {
            item: function_call_item(&accum, Some("completed")),
            output_index: accum.output_index,
            sequence_number: seq,
        });
    }

    /// Finalize the stream after the receiver is exhausted: close any items
    /// still open, then emit `response.completed` carrying the FULL final
    /// [`ResponsesResponse`] (built by reusing the non-stream assembly).
    ///
    /// Idempotent: a second call (or a call after [`Self::fail`]) is a no-op.
    pub fn finish(&mut self) -> Vec<ResponseStreamEvent> {
        let mut out = Vec::new();
        if self.finalized {
            return out;
        }
        // Defensive: close anything still open (a well-formed stream closes
        // these on messageStop, but a truncated stream may not).
        self.close_message_item(&mut out);
        self.close_reasoning_item(&mut out);
        self.finalized = true;

        let response = self.build_final_response();
        // End-of-stream observability (logs only — Option-B, no wire change).
        // cached_tokens is the cache-READ count already computed by
        // compute_token_usage inside build_final_response; only READ here.
        let cached_tokens = response
            .usage
            .input_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        let cache_hit = cached_tokens > 0;
        tracing::info!(
            request_id = %self.request_id,
            model = %self.model,
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            cached_tokens,
            cache_hit,
            duration_ms = self.start.elapsed().as_millis(),
            "responses streaming completed"
        );
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::Completed {
            response,
            sequence_number: seq,
        });
        self.tally(&out);
        let event_types = self
            .event_type_counts
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(",");
        let total_events: u64 = self.event_type_counts.iter().map(|(_, n)| n).sum();
        tracing::debug!(
            request_id = %self.request_id,
            total_events,
            event_types = %event_types,
            "responses streaming event-type summary"
        );
        out
    }

    /// Emit a single `response.failed` event for a mid-stream Bedrock error.
    ///
    /// The error code is produced by the SHARED [`crate::error::responses_error`]
    /// mapper so the streaming `response.failed` path and the pre-stream HTTP
    /// path agree on the codex-actionable code. Idempotent after finalization.
    pub fn fail(&mut self, err: &AppError) -> Vec<ResponseStreamEvent> {
        let mut out = Vec::new();
        if self.finalized {
            return out;
        }
        self.finalized = true;
        let mut response = self.in_progress_response();
        response.status = "failed".to_string();
        let (code, message) = crate::error::responses_error(err);
        response.error = Some(json!({
            "code": code,
            "message": message,
        }));
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::Failed {
            response,
            sequence_number: seq,
        });
        out
    }

    /// Reconstruct a Bedrock Converse-output-equivalent JSON from the accumulated
    /// streamed pieces, then REUSE
    /// [`crate::bedrock::responses_response::from_converse_output_to_responses`]
    /// to build the final [`ResponsesResponse`]. This avoids duplicating the
    /// final-Response assembly or the usage formula.
    fn build_final_response(&self) -> ResponsesResponse {
        let mut content: Vec<Value> = Vec::new();
        if !self.reasoning_text.is_empty() {
            content.push(json!({
                "reasoningContent": { "reasoningText": { "text": self.reasoning_text } }
            }));
        }
        if !self.message_text.is_empty() {
            content.push(json!({ "text": self.message_text }));
        }
        for (_, accum) in &self.tools {
            let input: Value = serde_json::from_str(&accum.arguments).unwrap_or(json!({}));
            content.push(json!({
                "toolUse": {
                    "toolUseId": accum.call_id,
                    "name": accum.name,
                    "input": input,
                }
            }));
        }

        let stop_reason = self
            .stop_reason
            .clone()
            .unwrap_or_else(|| "end_turn".to_string());
        let usage = self
            .usage
            .clone()
            .unwrap_or_else(|| json!({ "inputTokens": 0, "outputTokens": 0, "totalTokens": 0 }));

        let output = json!({
            "output": { "message": { "role": "assistant", "content": content } },
            "stopReason": stop_reason,
            "usage": usage,
        });

        // Reuse the non-stream assembly (T10). On the unexpected error path
        // (malformed reconstruction), fall back to a minimal completed envelope
        // so the stream still terminates cleanly.
        from_converse_output_to_responses(&output, &self.request, &self.model, &self.response_id)
            .unwrap_or_else(|_| {
                let mut resp = self.in_progress_response();
                resp.status = "completed".to_string();
                resp
            })
    }
}

/// Build a [`ResponseOutputItem::FunctionCall`] from a tool accumulator,
/// ensuring the `arguments` are a JSON-parseable string (defaulting to `{}` when
/// nothing was streamed, mirroring the non-stream mapper). `status` is `None` on
/// `output_item.added` (ai-sdk's added-chunk schema has no status) and
/// `Some("completed")` on `output_item.done` (ai-sdk's done-chunk schema
/// REQUIRES `status: z.literal("completed")`).
fn function_call_item(accum: &ToolAccum, status: Option<&str>) -> ResponseOutputItem {
    let arguments = if accum.arguments.trim().is_empty() {
        "{}".to_string()
    } else {
        accum.arguments.clone()
    };
    ResponseOutputItem::FunctionCall {
        id: Some(format!("fc_{}", accum.call_id)),
        call_id: accum.call_id.clone(),
        name: accum.name.clone(),
        arguments,
        status: status.map(str::to_string),
    }
}

/// Convert a Bedrock `TokenUsage` into the Converse-output `usage` JSON shape the
/// non-stream mapper reads (so the shared `compute_token_usage` runs over it).
fn usage_to_json(usage: &aws_sdk_bedrockruntime::types::TokenUsage) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("inputTokens".to_string(), json!(usage.input_tokens()));
    map.insert("outputTokens".to_string(), json!(usage.output_tokens()));
    map.insert("totalTokens".to_string(), json!(usage.total_tokens()));
    if let Some(r) = usage.cache_read_input_tokens() {
        map.insert("cacheReadInputTokens".to_string(), json!(r));
    }
    if let Some(w) = usage.cache_write_input_tokens() {
        map.insert("cacheWriteInputTokens".to_string(), json!(w));
    }
    Value::Object(map)
}

/// Current Unix time in whole seconds.
fn now_unix_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Convert a Bedrock `ConverseStream` output into an OpenAI Responses
/// [`ResponsesStream`].
///
/// Drives the Bedrock event receiver with `recv().await` inside an
/// `async_stream::stream!` block, feeding every event through the pure
/// [`ResponsesStreamState`] machine and yielding the lifecycle events it returns.
/// On receiver exhaustion the state machine's [`ResponsesStreamState::finish`]
/// emits `response.completed`. A receiver error maps via
/// [`from_bedrock_sdk_error`] to a single `response.failed` event, after which
/// the stream ends.
///
/// ## No `[DONE]` sentinel
///
/// The Responses protocol has no terminal sentinel; codex terminates on
/// `response.completed`. This stream yields typed events and then ENDS.
///
/// ## No timeout
///
/// No timeout wraps the loop — applying one would sever the connection. (The
/// route-level KeepAlive / timeout policy is T13's concern, not this module's.)
pub fn converse_stream_to_openai_responses(
    output: ConverseStreamOutputOp,
    model: String,
    response_id: String,
    request: ResponsesRequest,
    request_id: Arc<str>,
    start: Instant,
) -> ResponsesStream {
    let s = stream! {
        let mut receiver = output.stream;
        let mut state = ResponsesStreamState::new(model, response_id, request, request_id, start);
        loop {
            match receiver.recv().await {
                Ok(Some(event)) => {
                    for ev in state.map_event(&event) {
                        yield Ok(ev);
                    }
                }
                Ok(None) => {
                    for ev in state.finish() {
                        yield Ok(ev);
                    }
                    break;
                }
                Err(err) => {
                    let app_err = from_bedrock_sdk_error(&err.into_service_error());
                    for ev in state.fail(&app_err) {
                        yield Ok(ev);
                    }
                    break;
                }
            }
        }
    };
    s.boxed()
}

#[cfg(test)]
mod tests {
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

    /// Assert no `[DONE]` sentinel and no function_call_arguments.* events
    /// appear anywhere in the serialized sequence.
    fn assert_no_done_no_argdelta(events: &[ResponseStreamEvent]) {
        for ev in events {
            let s = serde_json::to_string(ev).unwrap();
            assert!(!s.contains("[DONE]"), "[DONE] sentinel leaked: {s}");
            assert!(
                !s.contains("function_call_arguments"),
                "function_call_arguments event leaked: {s}"
            );
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

    // -- Test 2: tool-use stream → function_call add+done, no arg-delta ------

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
        match &events[3] {
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
        let done_json = serde_json::to_value(&events[3]).unwrap();
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

    /// A `content_filtered` stop reason maps the final completed Response to
    /// `status == "incomplete"` with `incomplete_details.reason ==
    /// "content_filter"` (reusing the non-stream mapper). The stream still
    /// terminates on `response.completed` — there is NO separate refusal stream
    /// event from the Bedrock state machine.
    #[test]
    fn content_filter_stop_reason_yields_incomplete_completed() {
        let events = drive(&[
            ev_message_start(),
            ev_text("partial", 0),
            ev_message_stop(StopReason::ContentFiltered),
            ev_metadata(5, 1, 6),
        ]);

        // Terminates on response.completed (the lifecycle envelope), not a
        // refusal event.
        match events.last().expect("completed") {
            ResponseStreamEvent::Completed { response, .. } => {
                assert_eq!(response.status, "incomplete");
                let details = response
                    .incomplete_details
                    .as_ref()
                    .expect("incomplete_details present");
                assert_eq!(details["reason"], "content_filter");
            }
            other => panic!("expected response.completed last, got {other:?}"),
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
                // function_call item: add + done (no arg-delta events).
                "response.output_item.added",
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

    // -- finish() is idempotent ----------------------------------------------

    #[test]
    fn finish_is_idempotent() {
        let mut st = state();
        st.map_event(&ev_message_start());
        let first = st.finish();
        assert!(!first.is_empty());
        assert!(st.finish().is_empty(), "second finish must be empty");
    }
}
