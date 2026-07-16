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
//!    plus `response.function_call_arguments.delta` fragments), followed only
//!    after a successful `messageStop` and clean upstream EOF by
//!    `function_call_arguments.done` and `response.output_item.done`.
//!    Truncated/filtered/errored calls never become done.
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
use std::time::{Duration, Instant};

use crate::bedrock::responses_response::{
    from_converse_output_to_responses_with_tools, tool_output_item,
};
use crate::bedrock::responses_translate::{
    encode_reasoning_envelope, ResponsesToolKind, ResponsesToolRegistry,
};
use crate::domain::ResponsesStream;
use crate::error::{from_bedrock_sdk_error, AppError};
use crate::openai::responses_schema::{
    OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesRequest, ResponsesResponse,
};

/// The single summary index used for the reasoning summary family. The gateway
/// emits one summary part (index 0) carrying the full reasoning text.
const REASONING_SUMMARY_INDEX: u32 = 0;

/// Per-request runtime controls and observability context for the async stream
/// driver. Grouping these keeps the wire-state constructor independent.
pub(crate) struct ResponsesStreamRuntime {
    request_id: Arc<str>,
    start: Instant,
    idle_timeout: Duration,
}

impl ResponsesStreamRuntime {
    pub(crate) fn new(request_id: Arc<str>, start: Instant, idle_timeout: Duration) -> Self {
        Self {
            request_id,
            start,
            idle_timeout,
        }
    }
}

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
    /// Whether Bedrock emitted the matching `contentBlockStop`. This alone does
    /// NOT make the call executable: `messageStop` still has to confirm that
    /// the response was not truncated or content-filtered.
    closed: bool,
    /// Whether the matching Responses `output_item.done` was emitted after a
    /// successful `messageStop`, JSON validation, and clean upstream EOF.
    done_emitted: bool,
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
    tool_registry: ResponsesToolRegistry,

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
    reasoning_signature: Option<String>,
    reasoning_redacted: Vec<String>,

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

    /// Last Bedrock event kind observed. Metadata only; never contains payload
    /// content. Used to distinguish upstream stalls from downstream cancels.
    last_bedrock_event: &'static str,

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
        Self::new_with_tools(
            model,
            response_id,
            request,
            request_id,
            start,
            ResponsesToolRegistry::default(),
        )
    }

    fn new_with_tools(
        model: String,
        response_id: String,
        request: ResponsesRequest,
        request_id: Arc<str>,
        start: Instant,
        tool_registry: ResponsesToolRegistry,
    ) -> Self {
        Self {
            model,
            response_id,
            request,
            tool_registry,
            seq: 0,
            next_output_index: 0,
            started: false,
            reasoning_open: false,
            reasoning_output_index: 0,
            reasoning_text: String::new(),
            reasoning_signature: None,
            reasoning_redacted: Vec::new(),
            message_open: false,
            message_output_index: 0,
            message_text: String::new(),
            tools: Vec::new(),
            stop_reason: None,
            usage: None,
            finalized: false,
            last_bedrock_event: "none",
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
        if self.finalized {
            return out;
        }
        self.last_bedrock_event = match event {
            ConverseStreamOutput::MessageStart(_) => "message_start",
            ConverseStreamOutput::ContentBlockStart(_) => "content_block_start",
            ConverseStreamOutput::ContentBlockDelta(_) => "content_block_delta",
            ConverseStreamOutput::ContentBlockStop(_) => "content_block_stop",
            ConverseStreamOutput::MessageStop(_) => "message_stop",
            ConverseStreamOutput::Metadata(_) => "metadata",
            _ => "unknown",
        };
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
                        closed: false,
                        done_emitted: false,
                    };
                    let delay_added =
                        self.tool_registry
                            .resolve(&accum.name)
                            .is_some_and(|binding| {
                                matches!(
                                    binding.kind,
                                    ResponsesToolKind::LocalShell
                                        | ResponsesToolKind::Shell
                                        | ResponsesToolKind::ApplyPatch
                                )
                            });
                    if !delay_added {
                        let seq = self.next_seq();
                        out.push(ResponseStreamEvent::OutputItemAdded {
                            item: tool_item(&self.tool_registry, &accum, None),
                            output_index,
                            sequence_number: seq,
                        });
                    }
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

            // contentBlockStop only confirms that the content block ended. The
            // tool stays non-executable until messageStop supplies the overall
            // stop reason and the upstream stream reaches a clean EOF.
            ConverseStreamOutput::ContentBlockStop(ev) => {
                let block_index = ev.content_block_index();
                self.close_tool_block(block_index, &mut out);
            }

            // messageStop → close open items, capture stopReason.
            ConverseStreamOutput::MessageStop(ev) => {
                self.stop_reason = Some(ev.stop_reason().as_str().to_string());
                self.close_message_item(&mut out);
                self.close_reasoning_item(&mut out);
                if self.is_incomplete_stop() {
                    let (tool_blocks, tool_argument_bytes) = self.tool_progress();
                    if tool_blocks > 0 {
                        tracing::warn!(
                            request_id = %self.request_id,
                            model = %self.model,
                            stop_reason = %ev.stop_reason().as_str(),
                            tool_blocks,
                            tool_argument_bytes,
                            "responses tool calls left incomplete by upstream stop reason"
                        );
                    }
                }
            }

            // AWS defines metadata as the final ConverseStream protocol event.
            // Capture usage here, then finalize below without waiting for the
            // HTTP event-stream transport to yield EOF (it may remain open for
            // connection reuse).
            ConverseStreamOutput::Metadata(ev) => {
                if let Some(usage) = ev.usage() {
                    self.usage = Some(usage_to_json(usage));
                }
            }

            // #[non_exhaustive] catch-all — newly introduced variants are inert.
            _ => {}
        }

        self.tally(&out);
        if matches!(event, ConverseStreamOutput::Metadata(_)) {
            out.extend(self.finish());
        }
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
            ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                signature,
            )) => {
                self.reasoning_signature = Some(signature.clone());
            }
            ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::RedactedContent(
                redacted,
            )) => {
                use base64::Engine as _;
                self.ensure_reasoning_item(out);
                self.reasoning_redacted
                    .push(base64::engine::general_purpose::STANDARD.encode(redacted.as_ref()));
            }

            // Tool input fragment → accumulate; the item.done is emitted on
            // contentBlockStop / messageStop with the full arguments.
            ContentBlockDelta::ToolUse(tool_delta) => {
                if let Some((_, accum)) = self.tools.iter_mut().find(|(idx, _)| *idx == block_index)
                {
                    accum.arguments.push_str(tool_delta.input());
                    let accum = accum.clone();
                    if self
                        .tool_registry
                        .resolve(&accum.name)
                        .is_none_or(|binding| binding.kind == ResponsesToolKind::Function)
                    {
                        let seq = self.next_seq();
                        out.push(ResponseStreamEvent::FunctionCallArgumentsDelta {
                            item_id: format!("fc_{}", accum.call_id),
                            output_index: accum.output_index,
                            delta: tool_delta.input().to_string(),
                            sequence_number: seq,
                        });
                    }
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
                encrypted_content: None,
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

    fn reasoning_blocks(&self) -> Vec<Value> {
        let mut blocks = Vec::new();
        if !self.reasoning_text.is_empty() {
            blocks.push(json!({
                "reasoningContent": { "reasoningText": {
                    "text": self.reasoning_text,
                    "signature": self.reasoning_signature,
                } }
            }));
        }
        blocks.extend(
            self.reasoning_redacted
                .iter()
                .map(|data| json!({ "reasoningContent": { "redactedContent": data } })),
        );
        blocks
    }

    fn reasoning_encrypted_content(&self) -> Option<String> {
        encode_reasoning_envelope(&self.reasoning_blocks())
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
                encrypted_content: self.reasoning_encrypted_content(),
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

    /// Record a tool block's `contentBlockStop`. Final Responses completion is
    /// deliberately deferred until `messageStop` reveals the overall stop
    /// reason and the upstream stream ends cleanly, preventing truncation or a
    /// late receiver error from becoming executable.
    fn close_tool_block(&mut self, block_index: i32, _out: &mut Vec<ResponseStreamEvent>) {
        let Some(pos) = self.tools.iter().position(|(idx, _)| *idx == block_index) else {
            return;
        };
        if self.tools[pos].1.closed {
            return;
        }
        self.tools[pos].1.closed = true;
    }

    fn is_incomplete_stop(&self) -> bool {
        matches!(
            self.stop_reason.as_deref(),
            Some("max_tokens" | "content_filtered")
        )
    }

    fn tool_progress(&self) -> (usize, usize) {
        (
            self.tools.len(),
            self.tools
                .iter()
                .map(|(_, accum)| accum.arguments.len())
                .sum(),
        )
    }

    /// Validate every stopped tool before emitting any completion event, then
    /// finalize them in arrival order. This all-or-nothing check avoids a
    /// partially executable batch if one parallel call is malformed.
    fn finalize_closed_tools(&mut self, out: &mut Vec<ResponseStreamEvent>) -> Result<(), ()> {
        if self.tools.iter().any(|(_, accum)| !accum.closed) {
            return Err(());
        }
        let positions: Vec<usize> = self
            .tools
            .iter()
            .enumerate()
            .filter_map(|(pos, (_, accum))| (accum.closed && !accum.done_emitted).then_some(pos))
            .collect();
        if positions.iter().any(|pos| {
            serde_json::from_str::<Value>(&normalized_arguments(&self.tools[*pos].1)).is_err()
        }) {
            return Err(());
        }
        for pos in positions {
            self.finalize_tool_block(pos, out);
        }
        Ok(())
    }

    fn finalize_tool_block(&mut self, pos: usize, out: &mut Vec<ResponseStreamEvent>) {
        self.tools[pos].1.done_emitted = true;
        let accum = self.tools[pos].1.clone();
        if self
            .tool_registry
            .resolve(&accum.name)
            .is_some_and(|binding| {
                matches!(
                    binding.kind,
                    ResponsesToolKind::LocalShell
                        | ResponsesToolKind::Shell
                        | ResponsesToolKind::ApplyPatch
                )
            })
        {
            let seq = self.next_seq();
            out.push(ResponseStreamEvent::OutputItemAdded {
                item: tool_item(&self.tool_registry, &accum, Some("in_progress")),
                output_index: accum.output_index,
                sequence_number: seq,
            });
        }
        match self.tool_registry.resolve(&accum.name).map(|b| &b.kind) {
            None | Some(ResponsesToolKind::Function) => {
                let seq = self.next_seq();
                out.push(ResponseStreamEvent::FunctionCallArgumentsDone {
                    item_id: format!("fc_{}", accum.call_id),
                    output_index: accum.output_index,
                    arguments: normalized_arguments(&accum),
                    name: self
                        .tool_registry
                        .resolve(&accum.name)
                        .map_or_else(|| accum.name.clone(), |b| b.client_name.clone()),
                    sequence_number: seq,
                });
            }
            Some(ResponsesToolKind::Custom) => {
                let seq = self.next_seq();
                let input = parsed_tool_input(&accum)
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| json!(normalized_arguments(&accum)));
                let mut fields = std::collections::HashMap::new();
                fields.insert("input".to_string(), input);
                fields.insert(
                    "item_id".to_string(),
                    json!(format!("ct_{}", accum.call_id)),
                );
                fields.insert("output_index".to_string(), json!(accum.output_index));
                fields.insert("sequence_number".to_string(), json!(seq));
                out.push(ResponseStreamEvent::Other {
                    event_type: "response.custom_tool_call_input.done".to_string(),
                    fields,
                });
            }
            _ => {}
        }
        let seq = self.next_seq();
        out.push(ResponseStreamEvent::OutputItemDone {
            item: tool_item(&self.tool_registry, &accum, Some("completed")),
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
        // these on messageStop / contentBlockStop, but a truncated stream may
        // not). Order mirrors the non-stream mapper: reasoning, message, then
        // function_call items (#8 flushes tool blocks left open at truncation).
        self.close_message_item(&mut out);
        self.close_reasoning_item(&mut out);
        // Never synthesize output_item.done for a tool block Bedrock did not
        // close. Doing so turns truncated JSON into an executable tool call.
        if !self.is_incomplete_stop() && self.finalize_closed_tools(&mut out).is_err() {
            let (tool_blocks, tool_argument_bytes) = self.tool_progress();
            tracing::error!(
                request_id = %self.request_id,
                model = %self.model,
                stop_reason = self.stop_reason.as_deref().unwrap_or("missing"),
                tool_blocks,
                tool_argument_bytes,
                "responses upstream ended with invalid tool-call JSON"
            );
            let err =
                AppError::Internal("Upstream returned malformed tool-call arguments.".to_string());
            out.extend(self.fail(&err));
            self.tally(&out);
            return out;
        }
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
        let reasoning_tokens = response.usage.reasoning_tokens();
        let reasoning_used = reasoning_tokens > 0;
        tracing::info!(
            request_id = %self.request_id,
            model = %self.model,
            reasoning_effort = %self.request.reasoning_effort_label(),
            reasoning_used,
            reasoning_tokens,
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            cached_tokens,
            cache_hit,
            duration_ms = self.start.elapsed().as_millis(),
            "responses streaming completed"
        );
        let seq = self.next_seq();
        if response.status == "incomplete" {
            out.push(ResponseStreamEvent::Incomplete {
                response,
                sequence_number: seq,
            });
        } else {
            out.push(ResponseStreamEvent::Completed {
                response,
                sequence_number: seq,
            });
        }
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
        content.extend(self.reasoning_blocks());
        if !self.message_text.is_empty() {
            content.push(json!({ "text": self.message_text }));
        }
        for (_, accum) in &self.tools {
            let Ok(input) = serde_json::from_str::<Value>(&normalized_arguments(accum)) else {
                continue;
            };
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
        from_converse_output_to_responses_with_tools(
            &output,
            &self.request,
            &self.model,
            &self.response_id,
            &self.tool_registry,
        )
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
fn normalized_arguments(accum: &ToolAccum) -> String {
    if accum.arguments.trim().is_empty() {
        "{}".to_string()
    } else {
        accum.arguments.clone()
    }
}

fn parsed_tool_input(accum: &ToolAccum) -> Value {
    serde_json::from_str(&normalized_arguments(accum)).unwrap_or_else(|_| json!({}))
}

fn tool_item(
    registry: &ResponsesToolRegistry,
    accum: &ToolAccum,
    status: Option<&str>,
) -> ResponseOutputItem {
    let arguments = normalized_arguments(accum);
    let status = status.or_else(|| {
        registry
            .resolve(&accum.name)
            .filter(|binding| {
                matches!(
                    binding.kind,
                    ResponsesToolKind::Shell | ResponsesToolKind::ApplyPatch
                )
            })
            .map(|_| "in_progress")
    });
    tool_output_item(
        registry,
        &accum.call_id,
        &accum.name,
        &parsed_tool_input(accum),
        arguments,
        status,
    )
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
pub(crate) fn converse_stream_to_openai_responses(
    output: ConverseStreamOutputOp,
    model: String,
    response_id: String,
    request: ResponsesRequest,
    tool_registry: ResponsesToolRegistry,
    runtime: ResponsesStreamRuntime,
) -> ResponsesStream {
    let s = stream! {
        let mut receiver = output.stream;
        let ResponsesStreamRuntime {
            request_id,
            start,
            idle_timeout,
        } = runtime;
        let mut state = ResponsesStreamState::new_with_tools(
            model,
            response_id,
            request,
            request_id,
            start,
            tool_registry,
        );
        loop {
            match tokio::time::timeout(idle_timeout, receiver.recv()).await {
                Ok(Ok(Some(event))) => {
                    for ev in state.map_event(&event) {
                        yield Ok(ev);
                    }
                    // `metadata` is the last modeled ConverseStream event. The
                    // state finalizes while mapping it; stop polling the AWS
                    // receiver so the downstream SSE body closes immediately.
                    if state.finalized {
                        break;
                    }
                }
                Ok(Ok(None)) => {
                    for ev in state.finish() {
                        yield Ok(ev);
                    }
                    break;
                }
                Ok(Err(err)) => {
                    let app_err = from_bedrock_sdk_error(&err.into_service_error());
                    let (tool_blocks, tool_argument_bytes) = state.tool_progress();
                    tracing::error!(
                        request_id = %state.request_id,
                        model = %state.model,
                        last_bedrock_event = state.last_bedrock_event,
                        tool_blocks,
                        tool_argument_bytes,
                        error = %app_err,
                        "responses upstream stream failed"
                    );
                    for ev in state.fail(&app_err) {
                        yield Ok(ev);
                    }
                    break;
                }
                Err(_) => {
                    let (tool_blocks, tool_argument_bytes) = state.tool_progress();
                    tracing::error!(
                        request_id = %state.request_id,
                        model = %state.model,
                        last_bedrock_event = state.last_bedrock_event,
                        tool_blocks,
                        tool_argument_bytes,
                        idle_timeout_secs = idle_timeout.as_secs(),
                        "responses upstream stream idle timeout"
                    );
                    let app_err = AppError::UpstreamBedrock(
                        "Upstream response stream timed out while idle.".to_string(),
                    );
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

impl Drop for ResponsesStreamState {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let (tool_blocks, tool_argument_bytes) = self.tool_progress();
        tracing::warn!(
            request_id = %self.request_id,
            model = %self.model,
            last_bedrock_event = self.last_bedrock_event,
            tool_blocks,
            tool_argument_bytes,
            duration_ms = self.start.elapsed().as_millis(),
            "responses stream dropped before terminal event"
        );
    }
}

#[cfg(test)]
#[path = "responses_stream_tests.rs"]
mod tests;
