//! Non-streaming Bedrock Converse → OpenAI response mapping (pure functions).
//!
//! This module ports the Python non-streaming response builder in
//! `.legacy-python/src/api/models/bedrock.py` into a pure, testable mapping
//! that turns a Bedrock Converse output into an OpenAI [`ChatResponse`].
//!
//! Ported functions (with provenance line ranges):
//! - `_create_response`        (bedrock.py:1315-1405) → [`from_converse_output`]
//! - `_convert_finish_reason`  (bedrock.py:1858-1878) → [`convert_finish_reason`]
//! - `_estimate_reasoning_tokens` usage (bedrock.py:1298-1313) →
//!   [`crate::bedrock::tokens::estimate_reasoning_tokens`]
//! - usage math + call-site extraction (bedrock.py:621-648)
//!
//! ## Purity & input shape
//!
//! [`from_converse_output`] never calls Bedrock. It accepts the Converse output
//! as a [`serde_json::Value`] (the full response object containing
//! `output.message.content`, `usage`, and `stopReason`) so the core mapping is
//! synchronous and fully testable offline.
//!
//! ## Option B (reasoning never on the wire)
//!
//! When a `reasoningContent` block is present, reasoning is rendered inline as
//! `content = format!("<think>{reasoning}</think>{text}")` (bedrock.py:1360-1362)
//! and the wire `reasoning_content` field stays `None` —
//! [`crate::openai::schema::ChatResponseMessage`] carries `#[serde(skip_serializing)]`
//! so it never reaches the wire regardless.
//!
//! ## De-hardcoding
//!
//! No model-id literals appear here. The `model` string flows through verbatim
//! from the caller (echoed into the response `model` field).
//!
//! ## Metis FIX — `cacheWriteInputTokens` divergence
//!
//! Bedrock reports both `cacheReadInputTokens` (cache hits) and
//! `cacheWriteInputTokens` (tokens written into the cache). OpenAI's usage
//! schema only has `prompt_tokens_details.cached_tokens`, whose documented
//! semantics are READ tokens (cache hits). We therefore map `cached_tokens =
//! cacheReadInputTokens` and intentionally DO NOT surface
//! `cacheWriteInputTokens` anywhere on the wire — there is no standard OpenAI
//! field for it, and inventing one would break parity. The write count is read
//! from the input (so callers see it is acknowledged) but deliberately dropped.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::bedrock::capsule::{encode_capsule, reasoning_block_is_valid, CapsuleRuntime};
use crate::bedrock::tokens::{compute_token_usage, estimate_reasoning_tokens};
use crate::error::AppError;
use crate::openai::schema::{
    ChatResponse, ChatResponseMessage, Choice, CompletionTokensDetails, PromptTokensDetails,
    ResponseFunction, ToolCall, Usage,
};

/// Map a Bedrock Converse output to an OpenAI [`ChatResponse`].
///
/// `output` is the full Bedrock Converse response object, i.e. it carries
/// `output.message.content` (an array of content blocks), `usage`, and
/// `stopReason`. `model` is echoed verbatim into the response; `message_id`
/// becomes the response `id`.
///
/// Ports `_create_response` (bedrock.py:1315-1405) together with the call-site
/// token extraction (bedrock.py:621-648).
///
/// # Errors
///
/// Returns [`AppError::Internal`] if the output is missing the
/// `output.message.content` array (a malformed upstream response).
pub fn from_converse_output(
    output: &Value,
    model: &str,
    message_id: &str,
    capsule: Option<&CapsuleRuntime>,
) -> Result<ChatResponse, AppError> {
    // --- Extract content blocks (output.message.content) ---------------------
    let content = output
        .get("output")
        .and_then(|o| o.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            AppError::Internal(
                "bedrock converse output missing output.message.content array".to_string(),
            )
        })?;

    // --- Extract stopReason --------------------------------------------------
    let stop_reason = output.get("stopReason").and_then(Value::as_str);

    // --- Extract usage / token counts (bedrock.py:621-648) -------------------
    // Rebuild from parts via the shared helper (single source of truth, reused
    // by the streaming usage chunk). cacheWrite folds into prompt/total but is
    // never surfaced as its own wire field; cached_tokens reflects cacheRead.
    let usage = output.get("usage");
    let counts = compute_token_usage(
        usage_field(usage, "inputTokens"),
        usage_field(usage, "outputTokens"),
        usage_field(usage, "cacheReadInputTokens"),
        usage_field(usage, "cacheWriteInputTokens"),
    );
    let cache_read_tokens = counts.cached_tokens;
    let cache_write_tokens = usage_field(usage, "cacheWriteInputTokens");

    // --- Build the assistant message (bedrock.py:1327-1362) ------------------
    let mut message = ChatResponseMessage {
        role: Some("assistant".to_string()),
        ..Default::default()
    };

    // Iterate the content ONCE, collecting toolUse blocks into tool_calls and
    // text/reasoning separately. Tool extraction is DECOUPLED from stopReason
    // (#8): any toolUse block present yields a tool call, whatever the stop
    // reason (tool_use, max_tokens truncation, malformed_tool_use, ...).
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut text = String::new();
    let mut reasoning = String::new();
    let capsule = capsule.filter(|runtime| runtime.encoder_enabled);
    let mut reasoning_blocks = Vec::new();
    let mut invalid_reasoning_seen = false;
    let mut tool_use_seen = false;
    for c in content {
        if let Some(part) = c.get("toolUse") {
            tool_use_seen = true;
            let id = part.get("toolUseId").and_then(Value::as_str);
            let name = part.get("name").and_then(Value::as_str);
            // arguments = json.dumps(tool["input"]) (bedrock.py:1344).
            let input = part.get("input").cloned().unwrap_or(Value::Null);
            let arguments = serde_json::to_string(&input).map_err(|e| {
                AppError::Internal(format!("failed to serialize toolUse input: {e}"))
            })?;
            tool_calls.push(ToolCall {
                index: Some(tool_calls.len() as i32),
                id: id.map(str::to_string),
                r#type: "function".to_string(),
                function: ResponseFunction {
                    name: name.map(str::to_string),
                    arguments,
                },
            });
        } else if let Some(reasoning_block) = c.get("reasoningContent") {
            if capsule.is_some() && tool_use_seen {
                return Err(AppError::Internal(
                    "interleaved reasoning after tool use is not representable in a brtc_v1 capsule"
                        .to_string(),
                ));
            }
            if capsule.is_some() {
                if reasoning_block_is_valid(reasoning_block) {
                    reasoning_blocks.push(reasoning_block.clone());
                } else {
                    invalid_reasoning_seen = true;
                }
            }
            if let Some(t) = reasoning_block
                .get("reasoningText")
                .and_then(|rt| rt.get("text"))
                .and_then(Value::as_str)
            {
                reasoning.push_str(t);
            }
        } else if let Some(t) = c.get("text").and_then(Value::as_str) {
            // Mirrors Python: last text block wins (assignment, not append).
            text = t.to_string();
        }
        // Unknown blocks are ignored (Python logs a warning; parity-neutral).
    }

    if capsule.is_some() && !tool_calls.is_empty() && invalid_reasoning_seen {
        return Err(AppError::Internal(
            "reasoning block is missing replayable signed content for a brtc_v1 capsule"
                .to_string(),
        ));
    }
    if let Some(runtime) = capsule.filter(|_| !reasoning_blocks.is_empty()) {
        for call in &mut tool_calls {
            if let Some(tool_use_id) = call.id.as_deref() {
                call.id = Some(encode_capsule(
                    tool_use_id,
                    &reasoning_blocks,
                    &runtime.keyring,
                )?);
            }
        }
    }

    if !tool_calls.is_empty() {
        message.tool_calls = Some(tool_calls);
    }

    // content is set from reasoning/text as before. It is left None ONLY when
    // there is neither text nor reasoning AND tool_calls exist (the pure
    // tool_use shape). Otherwise the <think> inline rendering / plain text path
    // is unchanged.
    if !reasoning.is_empty() {
        // <think> inline rendering (bedrock.py:1360-1362). The wire
        // reasoning_content field stays None (Option B).
        message.content = Some(format!("<think>{reasoning}</think>{text}"));
    } else if !text.is_empty() || message.tool_calls.is_none() {
        message.content = Some(text);
    }

    // --- prompt_tokens_details: cached_tokens = cacheRead (bedrock.py:1364-1372) ---
    // Python only attaches details when cacheRead OR cacheWrite > 0. We mirror
    // that gate (so a no-cache response omits the details entirely), but only
    // ever surface the READ count.
    let prompt_tokens_details = if cache_read_tokens > 0 || cache_write_tokens > 0 {
        Some(PromptTokensDetails {
            cached_tokens: cache_read_tokens,
            audio_tokens: 0,
        })
    } else {
        None
    };

    // --- completion_tokens_details: reasoning_tokens (bedrock.py:1374-1381) ---
    let reasoning_tokens = estimate_reasoning_tokens_for(content);
    let completion_tokens_details = if reasoning_tokens > 0 {
        Some(CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as i32,
            audio_tokens: 0,
        })
    } else {
        None
    };

    // --- Assemble the response (bedrock.py:1383-1404) ------------------------
    let response = ChatResponse {
        id: message_id.to_string(),
        created: now_unix_secs(),
        model: model.to_string(),
        system_fingerprint: "fp".to_string(),
        choices: vec![Choice {
            index: 0,
            finish_reason: convert_finish_reason(stop_reason),
            logprobs: None,
            message,
        }],
        object: "chat.completion".to_string(),
        usage: Usage {
            prompt_tokens: counts.prompt_tokens,
            completion_tokens: counts.completion_tokens,
            total_tokens: counts.total_tokens,
            prompt_tokens_details,
            completion_tokens_details,
        },
    };

    Ok(response)
}

/// Sum reasoning text across `reasoningContent` blocks and estimate tokens.
///
/// Ports `_estimate_reasoning_tokens` (bedrock.py:1298-1313): concatenate
/// every `reasoningContent.reasoningText.text` then run the tiktoken estimate.
fn estimate_reasoning_tokens_for(content: &[Value]) -> u32 {
    let mut reasoning_text = String::new();
    for block in content {
        if let Some(t) = block
            .get("reasoningContent")
            .and_then(|rc| rc.get("reasoningText"))
            .and_then(|rt| rt.get("text"))
            .and_then(Value::as_str)
        {
            reasoning_text.push_str(t);
        }
    }
    estimate_reasoning_tokens(&reasoning_text)
}

/// Map a Bedrock Converse `stopReason` to an OpenAI `finish_reason`.
///
/// Ports `_convert_finish_reason` (bedrock.py:1858-1878):
/// - `tool_use` → `tool_calls`
/// - `finished` / `end_turn` / `stop_sequence` / `complete` → `stop`
/// - `max_tokens` → `length`
/// - `content_filtered` → `content_filter`
/// - anything else → lowercased passthrough
/// - `None` → `None`
pub fn convert_finish_reason(finish_reason: Option<&str>) -> Option<String> {
    let raw = finish_reason?;
    let lowered = raw.to_lowercase();
    let mapped = match lowered.as_str() {
        "tool_use" => "tool_calls",
        "finished" | "end_turn" | "stop_sequence" | "complete" => "stop",
        "max_tokens" => "length",
        "content_filtered" => "content_filter",
        // else: lowercased passthrough.
        _ => return Some(lowered),
    };
    Some(mapped.to_string())
}

/// Read an integer usage field, defaulting to 0 (matches Python `.get(k, 0)`).
fn usage_field(usage: Option<&Value>, key: &str) -> i32 {
    usage
        .and_then(|u| u.get(key))
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32
}

/// Current Unix time in whole seconds (mirrors Python `int(time.time())`).
fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "response_tests.rs"]
mod tests;
