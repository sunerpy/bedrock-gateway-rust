//! Unit tests for [`crate::bedrock::response`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec).
//!
//! The source file declares this via a `#[path]` mod tests, so the top-level
//! `use super::*;` resolves to the implementation module. Behavior is
//! unchanged; the original inline tests are preserved verbatim as flat test
//! functions, with a few edge-case examples added for branch coverage.
//!
//! Golden record/replay coverage for `from_converse_output` (semantic parity
//! against the pinned Python reference via `assert_semantic_eq`) lives in the
//! offline Tier-1 corpus at `tests/golden/corpus.rs` (`run_response`), so the
//! `assert_semantic_eq` helper — which is part of the integration-test crate —
//! is exercised there rather than duplicated here.

use std::collections::HashMap;

use super::*;
use crate::bedrock::capsule::{
    decode_capsule, is_capsule, CapsuleKeyring, CapsuleRuntime, DecodedCapsule,
};
use serde_json::json;

fn capsule_runtime(encoder_enabled: bool) -> CapsuleRuntime {
    CapsuleRuntime {
        keyring: CapsuleKeyring::new(
            HashMap::from([("current".to_string(), b"response-test-key".to_vec())]),
            Some("current".to_string()),
        ),
        encoder_enabled,
    }
}

fn decode_tool_call_id(id: &str, runtime: &CapsuleRuntime) -> DecodedCapsule {
    decode_capsule(id, &runtime.keyring).expect("tool call capsule decodes")
}

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
        None,
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
    let resp = from_converse_output(&output, "m", "id", None).expect("map");
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

    let resp = from_converse_output(&output, "m", "id", None).expect("map tool_use");
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
    let resp = from_converse_output(&output, "m", "id", None).expect("map");
    let calls = resp.choices[0].message.tool_calls.as_ref().expect("calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].index, Some(0));
    assert_eq!(calls[0].id.as_deref(), Some("a"));
    assert_eq!(calls[1].index, Some(1));
    assert_eq!(calls[1].id.as_deref(), Some("b"));
    assert_eq!(calls[1].function.arguments, "{\"k\":1}");
}

// -- #8: tool_calls extraction is DECOUPLED from stopReason --------------

#[test]
fn max_tokens_with_tooluse_returns_tool_calls_and_length() {
    // stopReason "max_tokens" (a truncation) + one toolUse block. The tool
    // call MUST be extracted (decoupled from stop reason) AND finish_reason
    // stays wire-faithful "length" — convert_finish_reason is unchanged.
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

    let resp = from_converse_output(&output, "m", "id", None).expect("map max_tokens+tool");
    let choice = &resp.choices[0];
    // finish_reason stays "length" (max_tokens truncation, wire-faithful).
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    // tool_calls populated regardless of the truncation stop reason.
    let calls = choice
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls present on max_tokens truncation");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].index, Some(0));
    assert_eq!(calls[0].id.as_deref(), Some("call-mt"));
    assert_eq!(calls[0].function.name.as_deref(), Some("get_weather"));
    let args: Value = serde_json::from_str(&calls[0].function.arguments).expect("args json");
    assert_eq!(args, json!({ "city": "Paris" }));
    // No text/reasoning → content is None.
    assert!(choice.message.content.is_none());
}

#[test]
fn malformed_tool_use_still_extracts_tool_calls() {
    // stopReason "malformed_tool_use": the toolUse block(s) that landed must
    // still be extracted; convert_finish_reason maps it via the lowercased
    // passthrough arm (unchanged).
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "toolUse": {
                "toolUseId": "call-mal",
                "name": "do_thing",
                "input": { "k": 1 }
            }}
        ] } },
        "stopReason": "malformed_tool_use",
        "usage": { "inputTokens": 3, "outputTokens": 2, "totalTokens": 5 }
    });

    let resp = from_converse_output(&output, "m", "id", None).expect("map malformed_tool_use");
    let choice = &resp.choices[0];
    // Lowercased passthrough — convert_finish_reason table unchanged.
    assert_eq!(choice.finish_reason.as_deref(), Some("malformed_tool_use"));
    let calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call-mal"));
    assert_eq!(calls[0].function.arguments, "{\"k\":1}");
}

#[test]
fn tool_use_with_text_keeps_both_content_and_tool_calls() {
    // Rare shape: a text block AND a toolUse block with an end_turn stop.
    // Decoupled extraction keeps BOTH — content from the text, tool_calls
    // from the toolUse. content is only None when there is neither text nor
    // reasoning.
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "text": "Let me check the weather." },
            { "toolUse": { "toolUseId": "call-both", "name": "get_weather", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 5, "outputTokens": 4, "totalTokens": 9 }
    });
    let resp = from_converse_output(&output, "m", "id", None).expect("map both");
    let choice = &resp.choices[0];
    assert_eq!(
        choice.message.content.as_deref(),
        Some("Let me check the weather.")
    );
    let calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call-both"));
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

    let resp = from_converse_output(&output, "m", "id", None).expect("map reasoning");
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
fn reasoning_only_renders_think_with_empty_tail() {
    // A reasoningContent block with NO following text block: the <think>
    // rendering still fires, leaving an empty tail after the closing tag.
    let output = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "reasoningContent": { "reasoningText": { "text": "just thinking" } } }
        ] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 4, "outputTokens": 2, "totalTokens": 6 }
    });
    let resp = from_converse_output(&output, "m", "id", None).expect("map reasoning-only");
    let msg = &resp.choices[0].message;
    assert_eq!(msg.content.as_deref(), Some("<think>just thinking</think>"));
    assert!(msg.reasoning_content.is_none());
    assert!(msg.tool_calls.is_none());
}

#[test]
fn no_reasoning_means_no_completion_details() {
    let output = json!({
        "output": { "message": { "content": [{ "text": "plain" }] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 1, "outputTokens": 1, "totalTokens": 2 }
    });
    let resp = from_converse_output(&output, "m", "id", None).expect("map");
    assert!(resp.usage.completion_tokens_details.is_none());
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("plain"));
}

#[test]
fn unknown_block_is_ignored_and_yields_empty_content() {
    // An unrecognized content block (neither toolUse, reasoningContent, nor
    // text) is ignored. With no text/reasoning and no tool_calls, content
    // falls through to the empty-string branch (tool_calls is None).
    let output = json!({
        "output": { "message": { "content": [{ "somethingElse": { "foo": 1 } }] } },
        "stopReason": "end_turn",
        "usage": { "inputTokens": 2, "outputTokens": 0, "totalTokens": 2 }
    });
    let resp = from_converse_output(&output, "m", "id", None).expect("map unknown block");
    let msg = &resp.choices[0].message;
    assert_eq!(msg.content.as_deref(), Some(""));
    assert!(msg.tool_calls.is_none());
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

    let resp = from_converse_output(&output, "m", "id", None).expect("map cache");
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
    let resp = from_converse_output(&output, "m", "id", None).expect("map");
    assert_eq!(resp.usage.prompt_tokens, 3); // 0 + 0 + 3
    assert_eq!(resp.usage.total_tokens, 5); // 3 + 2
    let details = resp.usage.prompt_tokens_details.as_ref().expect("details");
    assert_eq!(details.cached_tokens, 0);
}

#[test]
fn signed_reasoning_and_tool_use_mint_round_trip_capsule() {
    let reasoning_block = json!({
        "reasoningText": {
            "text": "private reasoning",
            "signature": "provider-signature"
        }
    });
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": reasoning_block.clone() },
            { "toolUse": { "toolUseId": "tool-123", "name": "lookup", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 1, "outputTokens": 1 }
    });
    let runtime = capsule_runtime(true);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let message = &response.choices[0].message;
    let calls = message.tool_calls.as_ref().expect("tool calls");
    let capsule = calls[0].id.as_deref().expect("tool call id");
    assert!(is_capsule(capsule));
    let decoded = decode_tool_call_id(capsule, &runtime);
    assert_eq!(decoded.tool_use_id, "tool-123");
    assert_eq!(decoded.reasoning_blocks, vec![reasoning_block]);
    assert_eq!(
        message.content.as_deref(),
        Some("<think>private reasoning</think>")
    );
}

#[test]
fn parallel_tool_uses_mint_capsules_with_shared_reasoning() {
    let reasoning_blocks = vec![
        json!({
            "reasoningText": {
                "text": "first",
                "signature": "signature-1"
            }
        }),
        json!({
            "reasoningText": {
                "text": "second",
                "signature": "signature-2"
            }
        }),
    ];
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": reasoning_blocks[0].clone() },
            { "reasoningContent": reasoning_blocks[1].clone() },
            { "toolUse": { "toolUseId": "tool-a", "name": "first_tool", "input": {} } },
            { "toolUse": { "toolUseId": "tool-b", "name": "second_tool", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let calls = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool calls");
    let decoded: Vec<DecodedCapsule> = calls
        .iter()
        .map(|call| {
            let id = call.id.as_deref().expect("tool call id");
            assert!(is_capsule(id));
            decode_tool_call_id(id, &runtime)
        })
        .collect();
    assert_eq!(decoded[0].tool_use_id, "tool-a");
    assert_eq!(decoded[1].tool_use_id, "tool-b");
    assert_eq!(decoded[0].reasoning_blocks, reasoning_blocks);
    assert_eq!(decoded[1].reasoning_blocks, decoded[0].reasoning_blocks);
}

#[test]
fn redacted_reasoning_and_tool_use_mint_capsule() {
    let reasoning_block = json!({ "redactedContent": "opaque-content" });
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": reasoning_block.clone() },
            { "toolUse": { "toolUseId": "tool-redacted", "name": "lookup", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let capsule = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .and_then(|calls| calls[0].id.as_deref())
        .expect("tool call capsule");
    let decoded = decode_tool_call_id(capsule, &runtime);
    assert_eq!(decoded.tool_use_id, "tool-redacted");
    assert_eq!(decoded.reasoning_blocks, vec![reasoning_block]);
}

#[test]
fn reasoning_after_tool_use_fails_closed() {
    let output = json!({
        "output": { "message": { "content": [
            { "toolUse": { "toolUseId": "tool-123", "name": "lookup", "input": {} } },
            { "reasoningContent": {
                "reasoningText": { "text": "late", "signature": "signature" }
            } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let error = from_converse_output(&output, "m", "id", Some(&runtime))
        .expect_err("interleaved reasoning must fail");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn unsigned_reasoning_before_tool_use_fails_closed() {
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": {
                "reasoningText": { "text": "unsigned" }
            } },
            { "toolUse": { "toolUseId": "tool-123", "name": "lookup", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let error = from_converse_output(&output, "m", "id", Some(&runtime))
        .expect_err("unsigned reasoning cannot be replayed safely");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn signed_reasoning_without_tool_use_only_renders_think() {
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": {
                "reasoningText": { "text": "standalone", "signature": "signature" }
            } },
            { "text": "answer" }
        ] } },
        "stopReason": "end_turn",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let message = &response.choices[0].message;
    assert!(message.tool_calls.is_none());
    assert_eq!(
        message.content.as_deref(),
        Some("<think>standalone</think>answer")
    );
}

#[test]
fn tool_use_without_reasoning_keeps_raw_id() {
    let output = json!({
        "output": { "message": { "content": [
            { "toolUse": { "toolUseId": "tool-raw", "name": "lookup", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(true);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let id = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .and_then(|calls| calls[0].id.as_deref())
        .expect("tool call id");
    assert_eq!(id, "tool-raw");
    assert!(!is_capsule(id));
}

#[test]
fn disabled_encoder_keeps_raw_tool_use_id() {
    let output = json!({
        "output": { "message": { "content": [
            { "reasoningContent": {
                "reasoningText": { "text": "private", "signature": "signature" }
            } },
            { "toolUse": { "toolUseId": "tool-disabled", "name": "lookup", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": {}
    });
    let runtime = capsule_runtime(false);

    let response = from_converse_output(&output, "m", "id", Some(&runtime)).expect("map response");

    let message = &response.choices[0].message;
    let id = message
        .tool_calls
        .as_ref()
        .and_then(|calls| calls[0].id.as_deref())
        .expect("tool call id");
    assert_eq!(id, "tool-disabled");
    assert!(!is_capsule(id));
    assert_eq!(message.content.as_deref(), Some("<think>private</think>"));
}

// -- error path -----------------------------------------------------------

#[test]
fn missing_content_array_errors() {
    let output = json!({
        "output": { "message": {} },
        "stopReason": "end_turn",
        "usage": { "totalTokens": 1, "outputTokens": 0 }
    });
    let err = from_converse_output(&output, "m", "id", None).expect_err("should error");
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
    let resp = from_converse_output(&output, "m", "id", None).expect("map");
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
