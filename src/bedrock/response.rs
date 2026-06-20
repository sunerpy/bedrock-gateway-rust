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

    let finish_is_tool_use = stop_reason == Some("tool_use");

    if finish_is_tool_use {
        // tool_use: build tool_calls[] from toolUse blocks (bedrock.py:1331-1350).
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for (tool_index, part) in content.iter().filter_map(|p| p.get("toolUse")).enumerate() {
            let id = part.get("toolUseId").and_then(Value::as_str);
            let name = part.get("name").and_then(Value::as_str);
            // arguments = json.dumps(tool["input"]) (bedrock.py:1344).
            let input = part.get("input").cloned().unwrap_or(Value::Null);
            let arguments = serde_json::to_string(&input).map_err(|e| {
                AppError::Internal(format!("failed to serialize toolUse input: {e}"))
            })?;
            tool_calls.push(ToolCall {
                index: Some(tool_index as i32),
                id: id.map(str::to_string),
                r#type: "function".to_string(),
                function: ResponseFunction {
                    name: name.map(str::to_string),
                    arguments,
                },
            });
        }
        message.tool_calls = Some(tool_calls);
        // content stays None for tool_use (bedrock.py:1350).
        message.content = None;
    } else {
        // Text / reasoning path (bedrock.py:1351-1362).
        let mut text = String::new();
        let mut reasoning = String::new();
        for c in content {
            if let Some(reasoning_block) = c.get("reasoningContent") {
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

        if !reasoning.is_empty() {
            // <think> inline rendering (bedrock.py:1360-1362). The wire
            // reasoning_content field stays None (Option B).
            message.content = Some(format!("<think>{reasoning}</think>{text}"));
        } else {
            message.content = Some(text);
        }
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
mod tests {
    use super::*;
    use serde_json::json;

    // -- convert_finish_reason: full table (bedrock.py:1858-1878) -------------

    #[test]
    fn finish_reason_table_exact() {
        assert_eq!(
            convert_finish_reason(Some("tool_use")).as_deref(),
            Some("tool_calls")
        );
        assert_eq!(
            convert_finish_reason(Some("finished")).as_deref(),
            Some("stop")
        );
        assert_eq!(
            convert_finish_reason(Some("end_turn")).as_deref(),
            Some("stop")
        );
        assert_eq!(
            convert_finish_reason(Some("stop_sequence")).as_deref(),
            Some("stop")
        );
        assert_eq!(
            convert_finish_reason(Some("complete")).as_deref(),
            Some("stop")
        );
        assert_eq!(
            convert_finish_reason(Some("max_tokens")).as_deref(),
            Some("length")
        );
        assert_eq!(
            convert_finish_reason(Some("content_filtered")).as_deref(),
            Some("content_filter")
        );
        // else → lowercased passthrough.
        assert_eq!(
            convert_finish_reason(Some("guardrail_intervened")).as_deref(),
            Some("guardrail_intervened")
        );
        assert_eq!(
            convert_finish_reason(Some("END_TURN")).as_deref(),
            Some("stop")
        );
        assert_eq!(
            convert_finish_reason(Some("SomethingNew")).as_deref(),
            Some("somethingnew")
        );
        // None → None.
        assert_eq!(convert_finish_reason(None), None);
    }

    // -- Text response --------------------------------------------------------

    #[test]
    fn text_response_maps_content_and_usage() {
        let output = json!({
            "output": { "message": { "role": "assistant", "content": [{ "text": "Hi" }] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 8, "outputTokens": 2, "totalTokens": 10 }
        });

        let resp = from_converse_output(
            &output,
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "chatcmpl-1",
        )
        .expect("map text response");

        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.model, "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.system_fingerprint, "fp");
        assert_eq!(resp.choices.len(), 1);
        let choice = &resp.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        assert_eq!(choice.message.role.as_deref(), Some("assistant"));
        assert_eq!(choice.message.content.as_deref(), Some("Hi"));
        assert!(choice.message.tool_calls.is_none());
        assert!(choice.message.reasoning_content.is_none());

        // usage: rebuild-from-parts. No cache → prompt == input, total ==
        // input + output.
        assert_eq!(resp.usage.prompt_tokens, 8);
        assert_eq!(resp.usage.completion_tokens, 2);
        assert_eq!(resp.usage.total_tokens, 10);
        assert!(resp.usage.prompt_tokens_details.is_none());
        assert!(resp.usage.completion_tokens_details.is_none());
    }

    #[test]
    fn usage_rebuilds_when_total_tokens_absent() {
        // No totalTokens field: rebuild-from-parts ignores it entirely.
        // prompt = input = 5; total = 5 + 3 = 8.
        let output = json!({
            "output": { "message": { "content": [{ "text": "x" }] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 5, "outputTokens": 3 }
        });
        let resp = from_converse_output(&output, "m", "id").expect("map");
        assert_eq!(resp.usage.prompt_tokens, 5);
        assert_eq!(resp.usage.completion_tokens, 3);
        assert_eq!(resp.usage.total_tokens, 8);
        assert!(resp.usage.prompt_tokens_details.is_none());
    }

    // -- tool_use → tool_calls + content None ---------------------------------

    #[test]
    fn tool_use_builds_tool_calls_and_content_none() {
        let output = json!({
            "output": { "message": { "role": "assistant", "content": [
                { "toolUse": {
                    "toolUseId": "tool-abc",
                    "name": "get_weather",
                    "input": { "city": "Paris", "unit": "c" }
                }}
            ] } },
            "stopReason": "tool_use",
            "usage": { "inputTokens": 12, "outputTokens": 6, "totalTokens": 18 }
        });

        let resp = from_converse_output(&output, "m", "id").expect("map tool_use");
        let choice = &resp.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        // content is None for tool_use.
        assert!(choice.message.content.is_none());
        let calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[0].id.as_deref(), Some("tool-abc"));
        assert_eq!(calls[0].r#type, "function");
        assert_eq!(calls[0].function.name.as_deref(), Some("get_weather"));
        // arguments = serde_json::to_string(input).
        let args: Value = serde_json::from_str(&calls[0].function.arguments).expect("args json");
        assert_eq!(args, json!({ "city": "Paris", "unit": "c" }));
    }

    #[test]
    fn tool_use_multiple_blocks_indexed() {
        let output = json!({
            "output": { "message": { "content": [
                { "toolUse": { "toolUseId": "a", "name": "f1", "input": {} } },
                { "text": "ignored between" },
                { "toolUse": { "toolUseId": "b", "name": "f2", "input": { "k": 1 } } }
            ] } },
            "stopReason": "tool_use",
            "usage": { "outputTokens": 1, "totalTokens": 2 }
        });
        let resp = from_converse_output(&output, "m", "id").expect("map");
        let calls = resp.choices[0].message.tool_calls.as_ref().expect("calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, Some(0));
        assert_eq!(calls[0].id.as_deref(), Some("a"));
        assert_eq!(calls[1].index, Some(1));
        assert_eq!(calls[1].id.as_deref(), Some("b"));
        assert_eq!(calls[1].function.arguments, "{\"k\":1}");
    }

    // -- reasoning → <think>...</think> in content, no reasoning_content key ---

    #[test]
    fn reasoning_renders_inline_think_and_never_serializes_wire_key() {
        let output = json!({
            "output": { "message": { "role": "assistant", "content": [
                { "reasoningContent": { "reasoningText": { "text": "step by step" } } },
                { "text": "The answer is 4." }
            ] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 10, "outputTokens": 5, "totalTokens": 15 }
        });

        let resp = from_converse_output(&output, "m", "id").expect("map reasoning");
        let msg = &resp.choices[0].message;
        // <think> inline rendering.
        assert_eq!(
            msg.content.as_deref(),
            Some("<think>step by step</think>The answer is 4.")
        );
        // Wire reasoning_content field stays None (Option B).
        assert!(msg.reasoning_content.is_none());

        // reasoning_tokens estimated from the reasoning text only.
        let details = resp
            .usage
            .completion_tokens_details
            .as_ref()
            .expect("completion_tokens_details present for reasoning");
        assert!(details.reasoning_tokens > 0);

        // Serialized JSON MUST NOT contain a reasoning_content key.
        let json_str = serde_json::to_string(&resp).expect("serialize response");
        assert!(
            !json_str.contains("reasoning_content"),
            "reasoning_content leaked to wire: {json_str}"
        );
        // And <think> is part of the serialized content.
        assert!(json_str.contains("<think>step by step</think>The answer is 4."));
    }

    #[test]
    fn no_reasoning_means_no_completion_details() {
        let output = json!({
            "output": { "message": { "content": [{ "text": "plain" }] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
        });
        let resp = from_converse_output(&output, "m", "id").expect("map");
        assert!(resp.usage.completion_tokens_details.is_none());
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("plain"));
    }

    // -- usage cache: cached_tokens == cacheRead (READ only) ------------------

    #[test]
    fn cached_tokens_reflect_cache_read_only() {
        // Rebuild-from-parts: input=9, cacheRead=4, cacheWrite=7, output=5.
        // prompt = 9+4+7 = 20; total = 20+5 = 25. cached_tokens = cacheRead = 4.
        let output = json!({
            "output": { "message": { "content": [{ "text": "ok" }] } },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 9,
                "outputTokens": 5,
                "totalTokens": 20,
                "cacheReadInputTokens": 4,
                "cacheWriteInputTokens": 7
            }
        });

        let resp = from_converse_output(&output, "m", "id").expect("map cache");
        assert_eq!(resp.usage.prompt_tokens, 20); // 9 + 4 + 7
        assert_eq!(resp.usage.total_tokens, 25); // 20 + 5
        let details = resp
            .usage
            .prompt_tokens_details
            .as_ref()
            .expect("prompt_tokens_details present when cache metrics exist");
        // cached_tokens reflects READ only; cacheWrite (7) folds into prompt but
        // is never surfaced as its own field.
        assert_eq!(details.cached_tokens, 4);
        assert_eq!(details.audio_tokens, 0);

        // cacheWrite must NOT appear anywhere on the wire.
        let json_str = serde_json::to_string(&resp).expect("serialize");
        assert!(!json_str.contains("cacheWrite"));
        assert!(!json_str.to_lowercase().contains("cache_write"));
    }

    #[test]
    fn cache_details_present_even_when_only_write() {
        // Details attached if cacheRead OR cacheWrite > 0. cached_tokens still
        // reflects READ (=0 here). prompt = 0+0+3 = 3; total = 3+2 = 5.
        let output = json!({
            "output": { "message": { "content": [{ "text": "ok" }] } },
            "stopReason": "end_turn",
            "usage": {
                "outputTokens": 2,
                "totalTokens": 12,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 3
            }
        });
        let resp = from_converse_output(&output, "m", "id").expect("map");
        assert_eq!(resp.usage.prompt_tokens, 3); // 0 + 0 + 3
        assert_eq!(resp.usage.total_tokens, 5); // 3 + 2
        let details = resp.usage.prompt_tokens_details.as_ref().expect("details");
        assert_eq!(details.cached_tokens, 0);
    }

    // -- error path -----------------------------------------------------------

    #[test]
    fn missing_content_array_errors() {
        let output = json!({
            "output": { "message": {} },
            "stopReason": "end_turn",
            "usage": { "totalTokens": 1, "outputTokens": 0 }
        });
        let err = from_converse_output(&output, "m", "id").expect_err("should error");
        assert!(matches!(err, AppError::Internal(_)));
    }

    // -- serialized wire shape: only OpenAI keys ------------------------------

    #[test]
    fn serialized_top_level_keys_are_openai_only() {
        let output = json!({
            "output": { "message": { "content": [{ "text": "Hi" }] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
        });
        let resp = from_converse_output(&output, "m", "id").expect("map");
        let value: Value = serde_json::to_value(&resp).expect("to_value");
        let obj = value.as_object().expect("object");
        let allowed = [
            "id",
            "created",
            "model",
            "system_fingerprint",
            "choices",
            "object",
            "usage",
        ];
        for key in obj.keys() {
            assert!(allowed.contains(&key.as_str()), "unexpected key: {key}");
        }
    }
}
