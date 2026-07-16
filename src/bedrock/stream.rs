//! Streaming Bedrock `ConverseStream` → OpenAI SSE chunk conversion.
//!
//! This module ports the Python streaming pipeline in
//! `.legacy-python/src/api/models/bedrock.py` into a pure, testable state
//! machine plus a thin async wrapper that drives the Bedrock event receiver.
//!
//! Ported functions (with provenance line ranges):
//! - `chat_stream`              (bedrock.py:659-707) → reasoning-token
//!   accumulation, `include_usage` gate, and the trailing `[DONE]` sentinel are
//!   reproduced by [`converse_stream_to_openai`].
//! - `_create_response_stream`  (bedrock.py:1407-1579) → the per-event mapping +
//!   the `<think>` state machine + tool-call index synthesis + metadata usage
//!   chunk, all in [`StreamState::map_event`].
//! - `_get_or_create_stream_tool_call_index` (bedrock.py:1581-1592) →
//!   [`StreamState::tool_index_for`].
//! - `_convert_finish_reason`   (bedrock.py:1858-1878) → reused from
//!   [`crate::bedrock::response::convert_finish_reason`].
//! - base.py:50-63 SSE framing → the chunk objects carry
//!   `object = "chat.completion.chunk"`; SSE framing (`data: …\n\n` and the
//!   `data: [DONE]` line) is the router's responsibility (task 22). This stream
//!   yields [`ChatStreamResponse`] items and ENDS; the router appends
//!   `data: [DONE]` after the stream is exhausted.
//!
//! ## Purity & testability (task §4)
//!
//! The Bedrock `EventReceiver` is hard to fake, so the intricate logic lives in
//! [`StreamState`], a synchronous state machine with no I/O. Each Bedrock event
//! maps to `Option<ChatStreamResponse>` via [`StreamState::map_event`]. The
//! async wrapper [`converse_stream_to_openai`] only drives `recv().await` and
//! forwards each event into the pure mapper — so the entire parity surface is
//! unit-tested offline against constructed SDK events.
//!
//! ## Option B (reasoning never on the wire)
//!
//! Reasoning is rendered inline as `<think>…</think>` inside `content`. The wire
//! `reasoning_content` field is never populated (it carries `skip_serializing`
//! in [`crate::openai::schema::ChatResponseMessage`] regardless). The `<think>`
//! state machine mirrors Python exactly:
//! - reasoning text → prepend `<think>` on first reasoning delta.
//! - regular text while a `<think>` is open → prepend `</think>` (reasoning→text
//!   transition).
//! - reasoning `signature` while open → emit `</think>`; otherwise skip.
//! - `messageStop` while open → emit a standalone `</think>` chunk FIRST, then
//!   the finish-reason chunk on the next event would be lost — so Python returns
//!   the `</think>` chunk and defers the stop. We reproduce that exactly.
//!
//! ## No timeouts (task §5)
//!
//! No timeout is ever applied to this stream — a timeout would sever the SSE
//! connection. The async wrapper loops on `recv().await` until the receiver is
//! exhausted or errors.
//!
//! ## De-hardcoding
//!
//! No model-id literals appear here. `model` flows through verbatim from the
//! caller and is echoed into every chunk's `model` field.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput as ConverseStreamOutputOp;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConverseStreamOutput, ReasoningContentBlockDelta,
};
use futures::StreamExt;

use crate::bedrock::response::convert_finish_reason;
use crate::bedrock::tokens::{compute_token_usage, estimate_reasoning_tokens};
use crate::domain::ChatStream;
use crate::error::from_bedrock_sdk_error;
use crate::openai::schema::{
    ChatResponseMessage, ChatStreamResponse, ChoiceDelta, CompletionTokensDetails,
    PromptTokensDetails, ResponseFunction, ToolCall, Usage,
};

/// Per-block tool-call bookkeeping (ports `stream_tool_call_meta` entries).
///
/// Maps a Bedrock `contentBlockIndex` to the synthesized contiguous OpenAI tool
/// `index` plus the `id`/`name` captured at `contentBlockStart`, so later input
/// fragments (which only carry `input`) can be attributed to the right call.
#[derive(Debug, Clone, Default)]
struct ToolMeta {
    index: i32,
    id: String,
    name: String,
}

/// The streaming `<think>`/tool/usage state machine.
///
/// This is a pure, synchronous port of the Python instance state used across
/// `chat_stream` / `_create_response_stream`:
/// - `think_emitted`            (bedrock.py:665)
/// - `stream_tool_call_meta`    (bedrock.py:666)
/// - `next_stream_tool_call_index` (bedrock.py:667)
/// - `reasoning_tokens`         (bedrock.py:668)
///
/// Construct one per stream, then feed every Bedrock event through
/// [`StreamState::map_event`] in order.
#[derive(Debug, Default)]
pub struct StreamState {
    /// Whether an unclosed `<think>` tag is currently open.
    think_emitted: bool,
    /// Bedrock block index → synthesized OpenAI tool-call metadata.
    tool_meta: HashMap<i32, ToolMeta>,
    /// Next contiguous OpenAI tool-call index to hand out.
    next_tool_index: i32,
    /// Accumulated reasoning tokens (estimated, cl100k_base) for the usage chunk.
    reasoning_tokens: u32,
}

impl StreamState {
    /// Create a fresh state machine for one stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Map Bedrock block index → contiguous OpenAI tool-call index.
    ///
    /// Ports `_get_or_create_stream_tool_call_index` (bedrock.py:1581-1592):
    /// reuse the existing index if the block is known, otherwise allocate the
    /// next contiguous index and remember it.
    fn tool_index_for(&mut self, block_index: i32) -> i32 {
        if let Some(meta) = self.tool_meta.get(&block_index) {
            return meta.index;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        // Seed a meta entry so a subsequent ContentBlockStart can fill id/name.
        self.tool_meta.entry(block_index).or_insert(ToolMeta {
            index,
            id: String::new(),
            name: String::new(),
        });
        index
    }

    /// Build a single-choice chunk carrying `delta` and an optional
    /// `finish_reason` (ports the `if message:` tail at bedrock.py:1564-1577).
    fn message_chunk(
        model: &str,
        message_id: &str,
        delta: ChatResponseMessage,
        finish_reason: Option<String>,
    ) -> ChatStreamResponse {
        ChatStreamResponse {
            id: message_id.to_string(),
            created: now_unix_secs(),
            model: model.to_string(),
            system_fingerprint: "fp".to_string(),
            choices: vec![ChoiceDelta {
                index: 0,
                finish_reason,
                logprobs: None,
                delta,
            }],
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    /// Map one Bedrock `ConverseStreamOutput` event to an optional OpenAI chunk.
    ///
    /// This is the core port of `_create_response_stream` (bedrock.py:1407-1579)
    /// together with the reasoning-token accumulation from `chat_stream`
    /// (bedrock.py:670-686). It mutates `self` (the `<think>` flag, tool
    /// bookkeeping, and the reasoning-token counter) and returns the chunk to
    /// emit, or `None` when the event produces no output (e.g. a lone
    /// `contentBlockStop`, or a reasoning `signature` with no open `<think>`).
    ///
    /// `include_usage` gates the metadata usage chunk (Python 692-699): when the
    /// caller did not request usage, the `Metadata` event yields `None` so no
    /// empty-`choices` chunk is emitted.
    ///
    /// The `#[non_exhaustive]` Bedrock enums are matched with a catch-all `_`
    /// arm that yields `None`, so newly-introduced variants are inert.
    pub fn map_event(
        &mut self,
        event: &ConverseStreamOutput,
        model: &str,
        message_id: &str,
        include_usage: bool,
        request_id: &str,
        start: Instant,
    ) -> Option<ChatStreamResponse> {
        match event {
            // messageStart → role delta (bedrock.py:1419-1423).
            ConverseStreamOutput::MessageStart(ev) => {
                let role = ev.role().as_str().to_string();
                let delta = ChatResponseMessage {
                    role: Some(role),
                    content: Some(String::new()),
                    ..Default::default()
                };
                Some(Self::message_chunk(model, message_id, delta, None))
            }

            // contentBlockStart → tool-call start (bedrock.py:1425-1449).
            ConverseStreamOutput::ContentBlockStart(ev) => {
                let block_index = ev.content_block_index();
                let tool_use = ev.start().and_then(|s| match s {
                    ContentBlockStart::ToolUse(tu) => Some(tu),
                    _ => None,
                })?;
                let index = self.tool_index_for(block_index);
                let id = tool_use.tool_use_id().to_string();
                let name = tool_use.name().to_string();
                // Record id/name so later input fragments reuse them.
                self.tool_meta.insert(
                    block_index,
                    ToolMeta {
                        index,
                        id: id.clone(),
                        name: name.clone(),
                    },
                );
                let delta = ChatResponseMessage {
                    tool_calls: Some(vec![ToolCall {
                        index: Some(index),
                        id: Some(id),
                        r#type: "function".to_string(),
                        function: ResponseFunction {
                            name: Some(name),
                            arguments: String::new(),
                        },
                    }]),
                    ..Default::default()
                };
                Some(Self::message_chunk(model, message_id, delta, None))
            }

            // contentBlockDelta → text / reasoning / tool input (bedrock.py:1451-1505).
            ConverseStreamOutput::ContentBlockDelta(ev) => {
                let block_index = ev.content_block_index();
                let delta = ev.delta()?;
                self.map_content_block_delta(delta, block_index, model, message_id)
            }

            // contentBlockStop → finalize block; no chunk needed (bedrock.py: implicit).
            ConverseStreamOutput::ContentBlockStop(_) => None,

            // messageStop always publishes the terminal finish reason. If an
            // inline reasoning block is still open, close it in the same chunk.
            ConverseStreamOutput::MessageStop(ev) => {
                let delta = if self.think_emitted {
                    self.think_emitted = false;
                    ChatResponseMessage {
                        content: Some("</think>".to_string()),
                        ..Default::default()
                    }
                } else {
                    ChatResponseMessage::default()
                };
                let finish_reason = convert_finish_reason(Some(ev.stop_reason().as_str()));
                Some(Self::message_chunk(model, message_id, delta, finish_reason))
            }

            // metadata → usage chunk with empty choices (bedrock.py:1526-1562 + 692-699).
            ConverseStreamOutput::Metadata(ev) => {
                let usage = ev.usage()?;
                let counts = compute_token_usage(
                    usage.input_tokens(),
                    usage.output_tokens(),
                    usage.cache_read_input_tokens().unwrap_or(0),
                    usage.cache_write_input_tokens().unwrap_or(0),
                );

                // Logs only (Option-B, no wire change). MUST stay before the
                // include_usage gate so completion is always logged even when no
                // usage chunk is emitted.
                let cache_hit = counts.cached_tokens > 0;
                tracing::info!(
                    request_id = %request_id,
                    model = %model,
                    prompt_tokens = counts.prompt_tokens,
                    completion_tokens = counts.completion_tokens,
                    total_tokens = counts.total_tokens,
                    cached_tokens = counts.cached_tokens,
                    cache_hit,
                    duration_ms = start.elapsed().as_millis(),
                    "chat streaming completed"
                );

                if !include_usage {
                    return None;
                }
                let cache_write = usage.cache_write_input_tokens().unwrap_or(0);
                let prompt_tokens_details = if counts.cached_tokens > 0 || cache_write > 0 {
                    Some(PromptTokensDetails {
                        cached_tokens: counts.cached_tokens,
                        audio_tokens: 0,
                    })
                } else {
                    None
                };

                // Patch accumulated reasoning tokens into completion details
                // (bedrock.py:682-686): only when reasoning was seen.
                let completion_tokens_details = if self.reasoning_tokens > 0 {
                    Some(CompletionTokensDetails {
                        reasoning_tokens: self.reasoning_tokens as i32,
                        audio_tokens: 0,
                    })
                } else {
                    None
                };

                Some(ChatStreamResponse {
                    id: message_id.to_string(),
                    created: now_unix_secs(),
                    model: model.to_string(),
                    system_fingerprint: "fp".to_string(),
                    choices: Vec::new(),
                    object: "chat.completion.chunk".to_string(),
                    usage: Some(Usage {
                        prompt_tokens: counts.prompt_tokens,
                        completion_tokens: counts.completion_tokens,
                        total_tokens: counts.total_tokens,
                        prompt_tokens_details,
                        completion_tokens_details,
                    }),
                })
            }

            // #[non_exhaustive] catch-all — newly introduced variants are inert.
            _ => None,
        }
    }

    /// Handle a `contentBlockDelta` payload (bedrock.py:1451-1505).
    fn map_content_block_delta(
        &mut self,
        delta: &ContentBlockDelta,
        block_index: i32,
        model: &str,
        message_id: &str,
    ) -> Option<ChatStreamResponse> {
        match delta {
            // Regular text — close <think> if open (bedrock.py:1453-1460).
            ContentBlockDelta::Text(text) => {
                let mut content = text.clone();
                if self.think_emitted {
                    content = format!("</think>{content}");
                    self.think_emitted = false;
                }
                let msg = ChatResponseMessage {
                    content: Some(content),
                    ..Default::default()
                };
                Some(Self::message_chunk(model, message_id, msg, None))
            }

            // Reasoning content (bedrock.py:1461-1475).
            ContentBlockDelta::ReasoningContent(rc) => match rc {
                ReasoningContentBlockDelta::Text(text) => {
                    // Accumulate reasoning tokens BEFORE rendering (parity with
                    // chat_stream pre-processing at bedrock.py:670-674).
                    self.reasoning_tokens += estimate_reasoning_tokens(text);
                    let mut content = text.clone();
                    if !self.think_emitted {
                        content = format!("<think>{content}");
                        self.think_emitted = true;
                    }
                    let msg = ChatResponseMessage {
                        content: Some(content),
                        ..Default::default()
                    };
                    Some(Self::message_chunk(model, message_id, msg, None))
                }
                ReasoningContentBlockDelta::Signature(_) => {
                    // signature_delta: close <think> if open, else skip
                    // (bedrock.py:1469-1475).
                    if self.think_emitted {
                        self.think_emitted = false;
                        let msg = ChatResponseMessage {
                            content: Some("</think>".to_string()),
                            ..Default::default()
                        };
                        Some(Self::message_chunk(model, message_id, msg, None))
                    } else {
                        None
                    }
                }
                // redactedContent / unknown reasoning deltas: no output.
                _ => None,
            },

            // Tool input fragment (bedrock.py:1476-1505).
            ContentBlockDelta::ToolUse(tool_delta) => {
                let index = self.tool_index_for(block_index);
                // input is already a JSON string fragment in the SDK type;
                // accumulate verbatim. Empty fragments stay "" (the caller's
                // accumulation across fragments yields the full JSON; default
                // {} is applied downstream when nothing is sent).
                let arguments = tool_delta.input().to_string();

                // Reuse stored id/name for this block (bedrock.py:1485-1491).
                let meta = self
                    .tool_meta
                    .get(&block_index)
                    .cloned()
                    .unwrap_or_default();
                let name = if meta.name.is_empty() {
                    None
                } else {
                    Some(meta.name.clone())
                };
                let id = if meta.id.is_empty() {
                    None
                } else {
                    Some(meta.id.clone())
                };

                let msg = ChatResponseMessage {
                    tool_calls: Some(vec![ToolCall {
                        index: Some(index),
                        id,
                        r#type: "function".to_string(),
                        function: ResponseFunction { name, arguments },
                    }]),
                    ..Default::default()
                };
                Some(Self::message_chunk(model, message_id, msg, None))
            }

            // citation / image / toolResult / unknown deltas: no output.
            _ => None,
        }
    }
}

/// Current Unix time in whole seconds (mirrors Python `int(time.time())`).
fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Convert a Bedrock `ConverseStream` output into an OpenAI [`ChatStream`].
///
/// Drives the Bedrock event receiver with `recv().await` inside an
/// `async_stream::stream!` block, feeding every event through the pure
/// [`StreamState`] machine. Ports the overall flow of `chat_stream`
/// (bedrock.py:659-707): per-event mapping, reasoning-token accumulation (inside
/// the state machine), and the `include_usage` gate.
///
/// ## `[DONE]` representation
///
/// This stream yields [`ChatStreamResponse`] items and then ENDS. The trailing
/// `data: [DONE]` SSE line (base.py:60-63) is appended by the router (task 22)
/// once the stream is exhausted — it is NOT represented as an item here.
///
/// ## Errors
///
/// A receiver error is mapped via [`from_bedrock_sdk_error`] and
/// yielded as the stream's final `Err` item, after which the stream ends.
///
/// ## No timeout
///
/// No timeout wraps the loop — applying one would sever the SSE connection.
pub fn converse_stream_to_openai(
    output: ConverseStreamOutputOp,
    model: String,
    message_id: String,
    include_usage: bool,
    request_id: Arc<str>,
    start: Instant,
) -> ChatStream {
    let s = stream! {
        let mut receiver = output.stream;
        let mut state = StreamState::new();
        loop {
            match receiver.recv().await {
                Ok(Some(event)) => {
                    if let Some(chunk) =
                        state.map_event(&event, &model, &message_id, include_usage, &request_id, start)
                    {
                        yield Ok(chunk);
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    yield Err(from_bedrock_sdk_error(&err.into_service_error()));
                    break;
                }
            }
        }
    };
    s.boxed()
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
