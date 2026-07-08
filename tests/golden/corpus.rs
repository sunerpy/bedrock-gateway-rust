//! Golden corpus tests (task 32).
//!
//! This module wires the static fixtures under `tests/golden/fixtures/` into the
//! Tier-1 offline parity suite. Each test loads a fixture pair and drives the
//! REAL Rust pipeline (`bedrock_gateway_rust::bedrock::*`) over it, asserting
//! semantic parity against the expected output via the harness comparators.
//!
//! Everything here runs OFFLINE — no AWS credentials, no network. Translation,
//! reasoning, tool, cache, response, and embedding paths are pure functions over
//! JSON; the streaming path drives the pure `StreamState::map_event` state
//! machine using SDK events reconstructed from a compact JSONL fixture shape.
//!
//! The fixtures encode the AGREED behaviour ported line-by-line from the pinned
//! Python reference (`.legacy-python/src/api/models/bedrock.py`, SHA `9a3e752`).
//! See `tests/golden/README.md` for the directory layout and the documented
//! intentional divergences (error envelope, cacheWrite-dropped).

#![allow(clippy::expect_used)]

use serde_json::{json, Value};

use base64::Engine as _;

use bedrock_gateway_rust::bedrock::capabilities::ConfigModelCapabilities;
use bedrock_gateway_rust::bedrock::embeddings::{CohereCodec, NovaCodec, TitanCodec};
use bedrock_gateway_rust::bedrock::responses_response::from_converse_output_to_responses;
use bedrock_gateway_rust::bedrock::responses_stream::ResponsesStreamState;
use bedrock_gateway_rust::bedrock::translate::{to_converse_args, ConverseExtras, ImageResolver};
use bedrock_gateway_rust::bedrock::{cache, reasoning, response, tools};
use bedrock_gateway_rust::config::ModelCapabilityConfig;
use bedrock_gateway_rust::domain::{EmbeddingBodyCodec, ModelCapabilities};
use bedrock_gateway_rust::openai::responses_schema::{ResponseStreamEvent, ResponsesRequest};
use bedrock_gateway_rust::openai::schema::ChatRequest;
use bedrock_gateway_rust::openai::schema::{EmbeddingData, EmbeddingsRequest, EncodingFormat};

use super::{
    assert_semantic_eq, assert_semantic_eq_with, assert_stream_eq, load_json, load_stream_fixture,
    load_text, load_translation_fixture, parse_sse, semantic_eq, StreamEvent,
    DEFAULT_VOLATILE_FIELDS,
};

// ===========================================================================
// Shared helpers (mirror the in-crate unit-test scaffolding offline)
// ===========================================================================

const MODELS_TOML: &str = "config/models.toml";

/// Build the config-driven capability resolver from the shipped `models.toml`.
/// Loaded relative to `CARGO_MANIFEST_DIR` so it works offline in CI.
fn caps() -> ConfigModelCapabilities {
    let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load config/models.toml");
    ConfigModelCapabilities::new(config)
}

/// An offline [`ImageResolver`]: `data:` URIs are decoded by the translation
/// itself, so `fetch` is never reached for the corpus. `supports_image` is
/// always true so multimodal fixtures exercise the image-block path without a
/// live model catalog.
struct CorpusResolver;

#[async_trait::async_trait]
impl ImageResolver for CorpusResolver {
    fn supports_image(&self, _model_id: &str) -> bool {
        true
    }
    async fn fetch(
        &self,
        _url: &str,
    ) -> Result<(Vec<u8>, String), bedrock_gateway_rust::error::AppError> {
        Err(bedrock_gateway_rust::error::AppError::Internal(
            "corpus resolver does not fetch remote images".to_string(),
        ))
    }
}

/// Compose the full request-translation pipeline exactly as
/// `BedrockChatProvider::assemble` does (provider.rs), but without any AWS
/// client. Returns the assembled Bedrock Converse args as a JSON object.
///
/// Order (parity with `assemble`):
/// 1. reasoning -> reasoning_fields seam,
/// 2. tool config (skip_tool_choice is false for every corpus model),
/// 3. prompt-caching control extract + strip,
/// 4. `to_converse_args` with the reasoning/tool seam,
/// 5. apply reasoning maxTokens/topP side-signals,
/// 6. cachePoint decoration (system + messages),
/// 7. tool-result normalization + placeholder injection.
async fn assemble_args(req: &ChatRequest, caps: &ConfigModelCapabilities) -> Value {
    let resolved = caps.resolve_foundation(&req.model);
    let mut chat = req.clone();

    // 1. Reasoning.
    let reasoning_outcome = match chat.reasoning_effort {
        Some(effort) => reasoning::build_reasoning_config(
            &resolved,
            effort,
            chat.max_tokens,
            chat.max_completion_tokens,
            caps,
        ),
        None => reasoning::ReasoningOutcome::default(),
    };

    // 2. Tool config (no corpus model triggers the llama tool_choice skip).
    let tool_config = match &chat.tools {
        Some(tool_list) if !tool_list.is_empty() => Some(
            tools::build_tool_config(tool_list, Some(&chat.tool_choice), false)
                .expect("build tool config"),
        ),
        _ => None,
    };

    // 3. Prompt-caching control: parse + strip from extra_body.
    let caching = cache::PromptCachingControl::extract_and_strip(&mut chat.extra_body);

    // 4. Translate with the reasoning + tool seam.
    let extras = ConverseExtras {
        reasoning_fields: if reasoning_outcome.additional_model_request_fields.is_empty() {
            None
        } else {
            Some(Value::Object(
                reasoning_outcome.additional_model_request_fields.clone(),
            ))
        },
        tool_config,
    };
    let resolver = CorpusResolver;
    let mut args = to_converse_args(&chat, caps, &resolver, &extras)
        .await
        .expect("to_converse_args");

    // 5. Apply reasoning side-signals to inferenceConfig.
    if let Value::Object(cfg) = &mut args.inference_config {
        if let Some(max_tokens) = reasoning_outcome.max_tokens {
            cfg.insert("maxTokens".to_string(), Value::from(max_tokens));
        }
        if reasoning_outcome.drop_top_p {
            cfg.remove("topP");
        }
    }

    // 6. Prompt caching decoration. Global default off (matches a clean env);
    // per-request prompt_caching controls drive enablement.
    let global_default = false;
    let system_enabled = caching.system_enabled(global_default);
    let messages_enabled = caching.messages_enabled(global_default);
    let max_checkpoints = caps
        .max_cache_tokens(&resolved)
        .map(|_| 4u32)
        .or(Some(4u32));

    let decorated_system = cache::decorate_system_blocks(
        std::mem::replace(&mut args.system, Value::Null),
        &resolved,
        caps,
        system_enabled,
    );
    let system_checkpoints = count_cache_points(&decorated_system);
    args.system = decorated_system;

    let decorated_messages = cache::decorate_messages(
        std::mem::replace(&mut args.messages, Value::Null),
        &resolved,
        caps,
        messages_enabled,
        system_checkpoints,
        max_checkpoints,
    );
    args.messages = decorated_messages;

    // 7. Tool-result normalization + placeholder safety net.
    if let Value::Array(turns) = &args.messages {
        let normalized = tools::normalize_tool_result_turns(turns);
        args.messages = Value::Array(normalized);
    }
    if let Value::Array(turns) = &args.messages {
        args.tool_config = tools::inject_placeholder_tool_config(turns, args.tool_config.take());
    }

    args.to_value()
}

/// Count trailing `cachePoint` blocks in a decorated `system` array (mirrors
/// `provider::count_cache_points`).
fn count_cache_points(system: &Value) -> u32 {
    let Value::Array(blocks) = system else {
        return 0;
    };
    blocks
        .iter()
        .filter(|b| b.as_object().is_some_and(|o| o.contains_key("cachePoint")))
        .count() as u32
}

/// Run a translation fixture: load `(openai_request, expected_bedrock_args)`,
/// drive the assembly pipeline, and assert semantic parity.
async fn run_translation(case: &str) {
    let fx = load_translation_fixture(case);
    let req: ChatRequest =
        serde_json::from_value(fx.openai_request.clone()).expect("parse openai_request");
    let actual = assemble_args(&req, &caps()).await;
    assert_semantic_eq(&fx.expected_bedrock_args, &actual);
}

// ===========================================================================
// Translation fixtures — OpenAI request → Bedrock Converse args
// ===========================================================================

#[tokio::test]
async fn translation_text_basic() {
    run_translation("text_basic").await;
}

#[tokio::test]
async fn translation_system_developer_blocks() {
    run_translation("system_developer_blocks").await;
}

#[tokio::test]
async fn translation_system_content_array() {
    run_translation("system_content_array").await;
}

#[tokio::test]
async fn translation_multimodal_data_uri_image() {
    run_translation("multimodal_data_uri_image").await;
}

#[tokio::test]
async fn translation_stop_string_singleton() {
    run_translation("stop_string_singleton").await;
}

#[tokio::test]
async fn translation_topp_conflict_drops_topp() {
    run_translation("topp_conflict_drops_topp").await;
}

#[tokio::test]
async fn translation_reasoning_adaptive_thinking() {
    run_translation("reasoning_adaptive_thinking").await;
}

#[tokio::test]
async fn translation_reasoning_budget_tokens() {
    run_translation("reasoning_budget_tokens").await;
}

#[tokio::test]
async fn translation_reasoning_deepseek_string() {
    run_translation("reasoning_deepseek_string").await;
}

#[tokio::test]
async fn translation_reasoning_none_ignored() {
    run_translation("reasoning_none_ignored").await;
}

#[tokio::test]
async fn translation_tools_single_turn_auto() {
    run_translation("tools_single_turn_auto").await;
}

#[tokio::test]
async fn translation_tools_multi_turn_placeholder() {
    run_translation("tools_multi_turn_placeholder").await;
}

#[tokio::test]
async fn translation_prompt_cache_system_point() {
    run_translation("prompt_cache_system_point").await;
}

// ===========================================================================
// Response fixtures — Bedrock Converse output → OpenAI chat.completion
// ===========================================================================

/// Run a response fixture: load `(bedrock_output, expected_openai_response)`,
/// drive `from_converse_output`, serialize, and assert semantic parity. Volatile
/// `id`/`created` are ignored by the comparator.
fn run_response(case: &str, model: &str, extra_ignore: &[&str]) {
    let bedrock_output = load_json(format!("response/{case}/bedrock_output.json"));
    let expected = load_json(format!("response/{case}/expected_openai_response.json"));
    let resp = response::from_converse_output(&bedrock_output, model, "chatcmpl-x")
        .expect("from_converse_output");
    let actual = serde_json::to_value(&resp).expect("serialize ChatResponse");

    let mut ignore: Vec<&str> = DEFAULT_VOLATILE_FIELDS.to_vec();
    ignore.extend_from_slice(extra_ignore);
    assert_semantic_eq_with(&expected, &actual, &ignore);
}

#[test]
fn response_text_basic() {
    run_response("text_basic", "anthropic.claude-3-sonnet-v1:0", &[]);
}

#[test]
fn response_tool_use_single() {
    run_response("tool_use_single", "anthropic.claude-3-sonnet-v1:0", &[]);
}

#[test]
fn response_reasoning_inline_think() {
    // reasoning_tokens is a tiktoken estimate (a detail, not a parity-critical
    // wire value). The fixture pins the documented value (3) but the comparator
    // ignores it so a future tiktoken table tweak does not break parity; the
    // <think> content + finish_reason + usage math are still asserted exactly.
    run_response(
        "reasoning_inline_think",
        "anthropic.claude-3-sonnet-v1:0",
        &["reasoning_tokens"],
    );
}

#[test]
fn response_cache_read_tokens() {
    // Rebuild-from-parts (cacheWrite > 0 case): prompt = input(5) +
    // cacheRead(1024) + cacheWrite(2048) = 3077; total = 3077 + output(4) =
    // 3081. cacheWrite folds into prompt/total but never surfaces as its own
    // wire field; cached_tokens = cacheRead (1024) only.
    run_response("cache_read_tokens", "us.amazon.nova-pro-v1:0", &[]);
}

#[test]
fn response_usage_no_cache_regression() {
    // No-cache regression: prompt_tokens == inputTokens (never negative),
    // total == input + output, no prompt_tokens_details.
    run_response(
        "usage_no_cache_regression",
        "anthropic.claude-3-sonnet-v1:0",
        &[],
    );
}

#[test]
fn response_finish_reason_length() {
    run_response(
        "finish_reason_length",
        "anthropic.claude-3-sonnet-v1:0",
        &[],
    );
}

#[test]
fn response_tool_use_max_tokens() {
    // #8: tool_calls extracted even on a max_tokens truncation; finish_reason
    // stays "length" (convert_finish_reason unchanged).
    run_response("tool_use_max_tokens", "anthropic.claude-3-sonnet-v1:0", &[]);
}

#[test]
fn response_finish_reason_content_filter() {
    run_response(
        "finish_reason_content_filter",
        "anthropic.claude-3-sonnet-v1:0",
        &[],
    );
}

#[test]
fn response_usage_fallback_no_total() {
    run_response(
        "usage_fallback_no_total",
        "anthropic.claude-3-sonnet-v1:0",
        &[],
    );
}

// ===========================================================================
// Streaming fixtures — Bedrock event stream → OpenAI SSE chunks
// ===========================================================================

mod stream_events {
    //! Reconstruct typed AWS SDK `ConverseStreamOutput` events from the compact
    //! JSONL fixture shape, so the corpus drives the real `StreamState` machine.
    use super::*;
    use aws_sdk_bedrockruntime::types::{
        ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
        ContentBlockStopEvent, ConversationRole, ConverseStreamMetadataEvent, ConverseStreamOutput,
        MessageStartEvent, MessageStopEvent, ReasoningContentBlockDelta, StopReason, TokenUsage,
        ToolUseBlockDelta, ToolUseBlockStart,
    };

    fn block_index(v: &Value) -> i32 {
        v.get("contentBlockIndex")
            .and_then(Value::as_i64)
            .unwrap_or(0) as i32
    }

    /// Build one SDK event from a single JSONL line value.
    pub fn build(line: &Value) -> ConverseStreamOutput {
        if let Some(ms) = line.get("messageStart") {
            let role = ms
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant");
            return ConverseStreamOutput::MessageStart(
                MessageStartEvent::builder()
                    .role(ConversationRole::from(role))
                    .build()
                    .unwrap(),
            );
        }
        if let Some(cbs) = line.get("contentBlockStart") {
            let tu = cbs.get("toolUse").expect("contentBlockStart.toolUse");
            let id = tu.get("toolUseId").and_then(Value::as_str).unwrap_or("");
            let name = tu.get("name").and_then(Value::as_str).unwrap_or("");
            return ConverseStreamOutput::ContentBlockStart(
                ContentBlockStartEvent::builder()
                    .start(ContentBlockStart::ToolUse(
                        ToolUseBlockStart::builder()
                            .tool_use_id(id)
                            .name(name)
                            .build()
                            .unwrap(),
                    ))
                    .content_block_index(block_index(cbs))
                    .build()
                    .unwrap(),
            );
        }
        if let Some(cbd) = line.get("contentBlockDelta") {
            let idx = block_index(cbd);
            let delta = if let Some(text) = cbd.get("text").and_then(Value::as_str) {
                ContentBlockDelta::Text(text.to_string())
            } else if let Some(rt) = cbd.get("reasoningText").and_then(Value::as_str) {
                ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(
                    rt.to_string(),
                ))
            } else if let Some(sig) = cbd.get("reasoningSignature").and_then(Value::as_str) {
                ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                    sig.to_string(),
                ))
            } else if let Some(input) = cbd.get("toolUseInput").and_then(Value::as_str) {
                ContentBlockDelta::ToolUse(
                    ToolUseBlockDelta::builder().input(input).build().unwrap(),
                )
            } else {
                panic!("unknown contentBlockDelta shape: {cbd}");
            };
            return ConverseStreamOutput::ContentBlockDelta(
                ContentBlockDeltaEvent::builder()
                    .delta(delta)
                    .content_block_index(idx)
                    .build()
                    .unwrap(),
            );
        }
        if let Some(cbstop) = line.get("contentBlockStop") {
            return ConverseStreamOutput::ContentBlockStop(
                ContentBlockStopEvent::builder()
                    .content_block_index(block_index(cbstop))
                    .build()
                    .unwrap(),
            );
        }
        if let Some(ms) = line.get("messageStop") {
            let reason = ms
                .get("stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn");
            return ConverseStreamOutput::MessageStop(
                MessageStopEvent::builder()
                    .stop_reason(StopReason::from(reason))
                    .build()
                    .unwrap(),
            );
        }
        if let Some(md) = line.get("metadata") {
            let usage = md.get("usage").expect("metadata.usage");
            let mut builder = TokenUsage::builder()
                .input_tokens(
                    usage
                        .get("inputTokens")
                        .and_then(Value::as_i64)
                        .unwrap_or(0) as i32,
                )
                .output_tokens(
                    usage
                        .get("outputTokens")
                        .and_then(Value::as_i64)
                        .unwrap_or(0) as i32,
                )
                .total_tokens(
                    usage
                        .get("totalTokens")
                        .and_then(Value::as_i64)
                        .unwrap_or(0) as i32,
                );
            if let Some(cr) = usage.get("cacheReadInputTokens").and_then(Value::as_i64) {
                builder = builder.cache_read_input_tokens(cr as i32);
            }
            if let Some(cw) = usage.get("cacheWriteInputTokens").and_then(Value::as_i64) {
                builder = builder.cache_write_input_tokens(cw as i32);
            }
            return ConverseStreamOutput::Metadata(
                ConverseStreamMetadataEvent::builder()
                    .usage(builder.build().unwrap())
                    .build(),
            );
        }
        panic!("unrecognized bedrock event line: {line}");
    }
}

/// Run a streaming fixture: load the Bedrock JSONL event stream, drive
/// `StreamState::map_event` per event, append the router's terminal `[DONE]`,
/// and assert SSE parity (count + event-type ordering + per-chunk values).
fn run_stream(case: &str, model: &str, include_usage: bool, extra_ignore: &[&str]) {
    use bedrock_gateway_rust::bedrock::stream::StreamState;

    let fx = load_stream_fixture(case);
    let mut state = StreamState::new();
    let mut actual: Vec<StreamEvent> = Vec::new();

    for (lineno, raw) in fx.bedrock_events.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("case {case} line {}: bad JSON: {e}", lineno + 1));
        let event = stream_events::build(&value);
        if let Some(chunk) = state.map_event(
            &event,
            model,
            "chatcmpl-x",
            include_usage,
            "req-test",
            std::time::Instant::now(),
        ) {
            let payload = serde_json::to_value(&chunk).expect("serialize chunk");
            actual.push(payload_to_event(payload));
        }
    }
    // The router appends the terminal `data: [DONE]` once the stream ends.
    actual.push(StreamEvent {
        kind: "done".to_string(),
        payload: Value::String("[DONE]".to_string()),
    });

    let mut ignore: Vec<&str> = DEFAULT_VOLATILE_FIELDS.to_vec();
    ignore.extend_from_slice(extra_ignore);
    if extra_ignore.is_empty() {
        assert_stream_eq(&fx.expected_sse, &actual);
    } else {
        // Re-parse expected with the extra ignore list applied via the same
        // comparator the harness uses for stream payloads.
        let res = bedrock_gateway_rust_stream_eq(&fx.expected_sse, &actual, &ignore);
        if let Err(diff) = res {
            panic!("stream parity mismatch ({case}):\n{diff}");
        }
    }
}

/// Re-derive a [`StreamEvent`] (kind + payload) from a serialized chunk, reusing
/// the harness's SSE parser so the ordering signature matches exactly.
fn payload_to_event(payload: Value) -> StreamEvent {
    // Render the chunk as a single SSE line and parse it back through the
    // harness so the `kind` derivation is identical to the expected side.
    let line = format!("data: {}", serde_json::to_string(&payload).unwrap());
    let mut parsed = parse_sse(&line).expect("parse chunk as sse");
    parsed.pop().expect("one event")
}

/// Stream comparator wrapper using the harness's `semantic_eq_stream` with a
/// custom ignore list (the harness exposes `semantic_eq_stream`).
fn bedrock_gateway_rust_stream_eq(
    expected: &[StreamEvent],
    actual: &[StreamEvent],
    ignore: &[&str],
) -> Result<(), String> {
    super::semantic_eq_stream(expected, actual, ignore)
}

#[test]
fn stream_text_sequence() {
    run_stream("text_sequence", "anthropic.claude-3-sonnet-v1:0", true, &[]);
}

#[test]
fn stream_reasoning_think_sequence() {
    // reasoning_tokens in the usage chunk is a tiktoken estimate; ignore it for
    // the same reason as the non-stream reasoning fixture.
    run_stream(
        "reasoning_think_sequence",
        "anthropic.claude-3-sonnet-v1:0",
        true,
        &["reasoning_tokens"],
    );
}

#[test]
fn stream_think_open_at_stop_closes() {
    // No metadata event ⇒ no usage chunk regardless of include_usage; the open
    // <think> is closed at messageStop and the finish_reason is deferred (parity).
    run_stream(
        "think_open_at_stop_closes",
        "anthropic.claude-3-sonnet-v1:0",
        true,
        &[],
    );
}

#[test]
fn stream_tool_use_sequence() {
    run_stream(
        "tool_use_sequence",
        "anthropic.claude-3-sonnet-v1:0",
        true,
        &[],
    );
}

#[test]
fn stream_tool_use_max_tokens_no_blockstop() {
    // #8: a tool block with input but NO contentBlockStop before a max_tokens
    // messageStop — the tool-call deltas are emitted at arrival, the finish
    // chunk is "length", nothing is dropped.
    run_stream(
        "tool_use_max_tokens_no_blockstop",
        "anthropic.claude-3-sonnet-v1:0",
        true,
        &[],
    );
}

// ===========================================================================
// Embeddings fixtures — codec encode/decode + base64 (Cohere / Titan)
// ===========================================================================

/// Select an embedding codec by model id (mirrors the registry family match).
fn codec_for_model(model: &str) -> Box<dyn EmbeddingBodyCodec> {
    if model.contains("cohere.embed") {
        Box::new(CohereCodec)
    } else if model.contains("titan-embed") {
        Box::new(TitanCodec)
    } else if model.contains("nova") {
        Box::new(NovaCodec)
    } else {
        panic!("no embedding codec for model {model}");
    }
}

/// Replicate `embeddings::build_data` (private in-crate): format decoded vectors
/// into OpenAI `data` entries honoring the requested encoding (float → raw
/// array; base64 → little-endian f32 bytes, base64-encoded).
fn build_data(vectors: Vec<Vec<f32>>, fmt: EncodingFormat) -> Value {
    let entries: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| {
            let data = match fmt {
                EncodingFormat::Float => EmbeddingData::Float(embedding),
                EncodingFormat::Base64 => {
                    let mut bytes = Vec::with_capacity(embedding.len() * 4);
                    for value in &embedding {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    EmbeddingData::Base64(base64::engine::general_purpose::STANDARD.encode(&bytes))
                }
            };
            json!({ "object": "embedding", "index": i, "embedding": data })
        })
        .collect();
    Value::Array(entries)
}

/// Run an embeddings fixture: encode the request body, decode the canned
/// Bedrock response, and rebuild the OpenAI response data, asserting parity on
/// both the request body and the formatted embedding data.
fn run_embeddings(case: &str) {
    let request: EmbeddingsRequest =
        serde_json::from_value(load_json(format!("embeddings/{case}/request.json")))
            .expect("parse embeddings request");
    let expected_body = load_json(format!("embeddings/{case}/expected_body.json"));
    let bedrock_response = load_text(format!("embeddings/{case}/bedrock_response.json"));
    let expected_embeddings = load_json(format!("embeddings/{case}/expected_embeddings.json"));

    let codec = codec_for_model(&request.model);

    // 1. encode → Bedrock request body.
    let body = codec.encode(&request).expect("codec encode");
    assert_semantic_eq(&expected_body, &body);

    // 2. decode canned Bedrock response → vectors, then format per encoding.
    let vectors = codec
        .decode(bedrock_response.as_bytes())
        .expect("codec decode");
    let data = build_data(vectors, request.encoding_format);
    let actual = json!({
        "object": "list",
        "model": request.model,
        "data": data,
    });
    assert_semantic_eq(&expected_embeddings, &actual);
}

#[test]
fn embeddings_cohere_float() {
    run_embeddings("cohere_float");
}

#[test]
fn embeddings_cohere_base64() {
    run_embeddings("cohere_base64");
}

#[test]
fn embeddings_titan_float() {
    run_embeddings("titan_float");
}

// ===========================================================================
// Responses (non-stream) fixtures — Bedrock Converse output → ResponsesResponse
// ===========================================================================
//
// Drives the REAL `bedrock::responses_response::from_converse_output_to_responses`
// over `(bedrock_output.json, openai_request.json) → expected_responses_response.json`.
// Asserts:
//   * `object == "response"`,
//   * `output[]` item ORDER (reasoning FIRST then message; function_call on
//     tool_use; NO `<think>` inline — that is the chat surface),
//   * Responses usage field NAMES (`input_tokens` / `output_tokens` /
//     `total_tokens`, plus `input_tokens_details.cached_tokens` and
//     `output_tokens_details.reasoning_tokens`); NEVER chat names
//     (`prompt_tokens` / `completion_tokens`),
//   * no top-level `output_text` wire field.
//
// Volatile fields ignored: the `resp_*` top-level `id` and `created_at` (both in
// the default ignore list). The derived `msg_*` / `rs_*` / `fc_*` ids ARE
// deterministic (derived from the fixed `response_id`) and are asserted exactly.

/// Fixed response id used to drive the Responses mappers offline. `msg_`/`rs_`
/// ids derive from this (trimming the `resp_` prefix), so fixtures pin
/// `msg_x` / `rs_x` deterministically while the volatile top-level `id` is
/// ignored by the comparator.
const RESPONSES_ID: &str = "resp_x";

/// The resolved model echoed verbatim into the Responses response `model` field
/// (distinct from the incoming `req.model`, mirroring the live resolution seam).
const RESPONSES_MODEL: &str = "resolved.model-id";

/// Load a Responses request fixture into a typed [`ResponsesRequest`].
fn load_responses_request(rel: &str) -> ResponsesRequest {
    serde_json::from_value(load_json(rel)).expect("parse Responses openai_request")
}

/// Run a Responses non-stream fixture: drive `from_converse_output_to_responses`
/// and assert semantic parity (ignoring volatile id/created_at plus any
/// `extra_ignore` such as the tiktoken `reasoning_tokens` estimate).
fn run_responses_response(case: &str, extra_ignore: &[&str]) {
    let bedrock_output = load_json(format!("responses_response/{case}/bedrock_output.json"));
    let req = load_responses_request(&format!("responses_response/{case}/openai_request.json"));
    let expected = load_json(format!(
        "responses_response/{case}/expected_responses_response.json"
    ));

    let resp =
        from_converse_output_to_responses(&bedrock_output, &req, RESPONSES_MODEL, RESPONSES_ID)
            .expect("from_converse_output_to_responses");
    let actual = serde_json::to_value(&resp).expect("serialize ResponsesResponse");

    // No `<think>` inline, no top-level `output_text` wire field — Responses
    // surface invariants (the chat surface owns `<think>`).
    let actual_str = serde_json::to_string(&actual).expect("string");
    assert!(
        !actual_str.contains("<think>"),
        "{case}: <think> leaked onto the Responses wire: {actual_str}"
    );
    assert!(
        actual.get("output_text").is_none(),
        "{case}: output_text is an SDK convenience, not a wire field"
    );

    let mut ignore: Vec<&str> = DEFAULT_VOLATILE_FIELDS.to_vec();
    ignore.extend_from_slice(extra_ignore);
    assert_semantic_eq_with(&expected, &actual, &ignore);
}

#[test]
fn responses_response_text() {
    // Text → single message item with one output_text part; usage carries
    // Responses field names; cached/reasoning details are zero.
    run_responses_response("responses_text", &[]);
}

#[test]
fn responses_response_tool_call() {
    // tool_use stop reason → a single function_call item with call_id/name and
    // JSON-string arguments; no message item.
    run_responses_response("responses_tool_call", &[]);
}

#[test]
fn responses_response_tool_call_max_tokens() {
    // #8: a toolUse block on a max_tokens truncation still emits a function_call
    // item; status is "incomplete" with reason "max_output_tokens".
    run_responses_response("tool_use_max_tokens", &[]);
}

#[test]
fn responses_response_reasoning() {
    // reasoning present → reasoning item FIRST, then the message item (NOT
    // <think> inline). Also exercises the Responses usage formula with
    // cacheRead(4)+cacheWrite(7): input_tokens = 10+4+7 = 21; total = 26;
    // cached_tokens = 4 (read side only). reasoning_tokens is a tiktoken
    // estimate → ignored (same policy as the chat reasoning fixture).
    run_responses_response("responses_reasoning", &["reasoning_tokens"]);
}

// ===========================================================================
// Responses (streaming) fixtures — Bedrock event stream → ResponseStreamEvents
// ===========================================================================
//
// Drives the REAL `bedrock::responses_stream::ResponsesStreamState` pure state
// machine (`map_event` per Bedrock event + `finish()`), reconstructing typed SDK
// events from the SAME compact JSONL shape the chat streaming corpus uses
// (`stream_events::build`). Asserts:
//   * EXACT event-`type` ordering across the full lifecycle,
//   * `sequence_number` monotonic from 0 with no gaps,
//   * NO `[DONE]` sentinel anywhere (the Responses protocol has none),
//   * NO state-machine-emitted `function_call_arguments.delta`/`.done` events
//     (schema accepts them for compatibility; codex reconstructs from the item
//     add/done pair),
//   * `response.completed` carries the FULL final Response (output[] + usage).
//
// Volatile `created_at` is ignored per-event; `reasoning_tokens` is added to the
// ignore list for the reasoning case.

/// Read the expected Responses event stream fixture (one JSON event per line).
fn load_responses_events(case: &str) -> Vec<Value> {
    let raw = load_text(format!(
        "responses_stream/{case}/expected_response_events.jsonl"
    ));
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{case}: bad event line: {e}")))
        .collect()
}

/// Drive the Responses streaming state machine over the Bedrock JSONL fixture,
/// returning the serialized lifecycle events (including the terminal
/// `response.completed` from `finish()`). No `[DONE]` is ever appended — the
/// Responses protocol has no sentinel.
fn drive_responses_stream(case: &str) -> Vec<Value> {
    let bedrock_events = load_text(format!("responses_stream/{case}/bedrock_events.jsonl"));
    let req = load_responses_request(&format!("responses_stream/{case}/openai_request.json"));

    let mut state = ResponsesStreamState::new(
        RESPONSES_MODEL.to_string(),
        RESPONSES_ID.to_string(),
        req,
        std::sync::Arc::from("req-test"),
        std::time::Instant::now(),
    );
    let mut actual: Vec<Value> = Vec::new();

    for (lineno, raw) in bedrock_events.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("case {case} line {}: bad JSON: {e}", lineno + 1));
        let event = stream_events::build(&value);
        for ev in state.map_event(&event) {
            actual.push(serde_json::to_value(&ev).expect("serialize ResponseStreamEvent"));
        }
    }
    for ev in state.finish() {
        actual.push(serde_json::to_value(&ev).expect("serialize ResponseStreamEvent"));
    }
    actual
}

/// The `type` tag of a serialized Responses stream event.
fn event_type(ev: &Value) -> &str {
    ev.get("type").and_then(Value::as_str).unwrap_or("<none>")
}

/// Compare a driven Responses event stream against the expected fixture stream.
///
/// Enforces (in order): exact event-`type` ordering, `sequence_number`
/// monotonic from 0, no `[DONE]` / no `function_call_arguments` anywhere, then
/// per-event semantic parity (volatile `created_at` + `extra_ignore` removed).
fn assert_responses_stream(
    case: &str,
    expected: &[Value],
    actual: &[Value],
    extra_ignore: &[&str],
) {
    // Length.
    assert_eq!(
        expected.len(),
        actual.len(),
        "{case}: event count differs (expected {}, got {})\nexpected types: {:?}\nactual types:   {:?}",
        expected.len(),
        actual.len(),
        expected.iter().map(event_type).collect::<Vec<_>>(),
        actual.iter().map(event_type).collect::<Vec<_>>(),
    );

    // Exact event-type ordering.
    let expected_types: Vec<&str> = expected.iter().map(event_type).collect();
    let actual_types: Vec<&str> = actual.iter().map(event_type).collect();
    assert_eq!(
        expected_types, actual_types,
        "{case}: event-type ordering differs"
    );

    // sequence_number monotonic from 0 (no gaps), on BOTH sides.
    for (label, stream) in [("expected", expected), ("actual", actual)] {
        for (i, ev) in stream.iter().enumerate() {
            let seq = ev
                .get("sequence_number")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("{case}: {label} event[{i}] missing sequence_number"));
            assert_eq!(
                seq, i as u64,
                "{case}: {label} sequence_number not monotonic-from-0 at index {i}"
            );
        }
    }

    // No [DONE] sentinel, no function_call_arguments.* events anywhere.
    for ev in actual {
        let s = serde_json::to_string(ev).expect("string");
        assert!(!s.contains("[DONE]"), "{case}: [DONE] sentinel leaked: {s}");
        assert!(
            !s.contains("function_call_arguments"),
            "{case}: function_call_arguments event leaked: {s}"
        );
    }

    // response.completed carries the FULL final Response (output + usage).
    let completed = actual
        .last()
        .filter(|ev| event_type(ev) == "response.completed")
        .unwrap_or_else(|| panic!("{case}: last event is not response.completed"));
    let response = completed
        .get("response")
        .unwrap_or_else(|| panic!("{case}: completed event missing response"));
    assert_eq!(
        response.get("status").and_then(Value::as_str),
        Some("completed"),
        "{case}: completed response status mismatch"
    );
    assert!(
        response.get("output").and_then(Value::as_array).is_some(),
        "{case}: completed response missing output[]"
    );
    assert!(
        response.get("usage").is_some(),
        "{case}: completed response missing usage"
    );

    // Per-event semantic parity.
    let mut ignore: Vec<&str> = DEFAULT_VOLATILE_FIELDS.to_vec();
    ignore.extend_from_slice(extra_ignore);
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if let Err(diff) = semantic_eq(e, a, &ignore) {
            panic!(
                "{case}: event[{i}] (type `{}`) payload mismatch:\n{diff}",
                event_type(e)
            );
        }
    }
}

/// Run a Responses streaming fixture end to end.
fn run_responses_stream(case: &str, extra_ignore: &[&str]) {
    let expected = load_responses_events(case);
    let actual = drive_responses_stream(case);
    assert_responses_stream(case, &expected, &actual, extra_ignore);
}

#[test]
fn responses_stream_text() {
    // Full text lifecycle: created → in_progress → output_item.added(message)
    // → content_part.added → output_text.delta* → output_text.done →
    // content_part.done → output_item.done → completed. No [DONE].
    run_responses_stream("responses_text", &[]);
}

#[test]
fn responses_stream_tool() {
    // Tool lifecycle: created → in_progress → output_item.added(function_call)
    // → output_item.done(function_call, full JSON args) → completed. NO
    // function_call_arguments.delta — codex reconstructs from add/done.
    run_responses_stream("responses_tool", &[]);
}

#[test]
fn responses_stream_reasoning() {
    // Reasoning lifecycle: reasoning item (added → reasoning_text.delta* →
    // reasoning_text.done → done) BEFORE the message item, then the message
    // lifecycle, then completed. reasoning_tokens (tiktoken estimate) ignored.
    run_responses_stream("responses_reasoning", &["reasoning_tokens"]);
}

// ===========================================================================
// Negative controls — the comparator MUST reject deliberately wrong fixtures
// ===========================================================================
//
// These prove the Responses comparators are not vacuous: a changed value or a
// reordered event sequence is rejected. They mutate a KNOWN-GOOD driven output
// and assert the comparator fails.

/// Negative control (non-stream): mutating an expected output value (here the
/// message text) must be REJECTED by the semantic comparator.
#[test]
fn negative_responses_response_changed_value_is_rejected() {
    let bedrock_output = load_json("responses_response/responses_text/bedrock_output.json");
    let req = load_responses_request("responses_response/responses_text/openai_request.json");
    let resp =
        from_converse_output_to_responses(&bedrock_output, &req, RESPONSES_MODEL, RESPONSES_ID)
            .expect("map");
    let actual = serde_json::to_value(&resp).expect("serialize");

    // Take the genuine expected, then corrupt the message text.
    let mut corrupted =
        load_json("responses_response/responses_text/expected_responses_response.json");
    corrupted["output"][0]["content"][0]["text"] = Value::String("WRONG TEXT".to_string());

    let result = semantic_eq(&corrupted, &actual, DEFAULT_VOLATILE_FIELDS);
    assert!(
        result.is_err(),
        "comparator MUST reject a changed Responses output text"
    );
    assert!(
        result.unwrap_err().contains("text"),
        "diff should name the offending field"
    );
}

/// Negative control (non-stream): swapping the Responses usage field name back
/// to a chat name (`prompt_tokens`) must be REJECTED — the surfaces must not be
/// conflated.
#[test]
fn negative_responses_usage_chat_field_name_is_rejected() {
    let bedrock_output = load_json("responses_response/responses_text/bedrock_output.json");
    let req = load_responses_request("responses_response/responses_text/openai_request.json");
    let resp =
        from_converse_output_to_responses(&bedrock_output, &req, RESPONSES_MODEL, RESPONSES_ID)
            .expect("map");
    let actual = serde_json::to_value(&resp).expect("serialize");

    let mut corrupted =
        load_json("responses_response/responses_text/expected_responses_response.json");
    let usage = corrupted["usage"].as_object_mut().expect("usage object");
    let input_tokens = usage.remove("input_tokens").expect("input_tokens present");
    usage.insert("prompt_tokens".to_string(), input_tokens);

    let result = semantic_eq(&corrupted, &actual, DEFAULT_VOLATILE_FIELDS);
    assert!(
        result.is_err(),
        "comparator MUST reject a chat-style usage field name on the Responses surface"
    );
}

/// Negative control (streaming): reordering the event sequence (here moving the
/// final `response.completed` to the front) must be REJECTED by the ordering
/// check.
#[test]
fn negative_responses_stream_reordered_is_rejected() {
    let expected = load_responses_events("responses_text");
    let actual = drive_responses_stream("responses_text");

    // Genuine driven output matches the fixture (sanity — not the negative part).
    assert_responses_stream("responses_text", &expected, &actual, &[]);

    // Now reorder the EXPECTED: rotate the last event (completed) to the front.
    let mut reordered = expected.clone();
    let last = reordered.pop().expect("non-empty");
    reordered.insert(0, last);

    // The ordering check must fail (panic) — capture it via catch_unwind.
    let case = "responses_text";
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_responses_stream(case, &reordered, &actual, &[]);
    }));
    assert!(
        result.is_err(),
        "comparator MUST reject a reordered Responses event sequence"
    );
}

/// Negative control (streaming): if the expected stream is corrupted to contain
/// a `function_call_arguments.delta` event, driving the REAL state machine (which
/// never emits one) must produce a mismatch — proving the corpus pins the
/// no-arg-delta invariant.
#[test]
fn negative_responses_stream_arg_delta_absent_from_machine() {
    let actual = drive_responses_stream("responses_tool");
    // The real machine emits no function_call_arguments.* event.
    for ev in &actual {
        let s = serde_json::to_string(ev).expect("string");
        assert!(
            !s.contains("function_call_arguments"),
            "state machine unexpectedly emitted function_call_arguments: {s}"
        );
    }
    // A doctored expected stream that DOES contain such an event differs in
    // both count and type ordering → comparator rejects it.
    let mut doctored = load_responses_events("responses_tool");
    doctored.insert(
        3,
        json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_call-1",
            "output_index": 0,
            "delta": "{",
            "sequence_number": 3
        }),
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_responses_stream("responses_tool", &doctored, &actual, &[]);
    }));
    assert!(
        result.is_err(),
        "comparator MUST reject an injected function_call_arguments.delta event"
    );
}

// ===========================================================================
// Captured real-stream replay — fixtures seeded from live opencode/codex SSE
// ===========================================================================
//
// These fixtures are REAL `POST /responses` streams captured from a running
// gateway (under `responses_sse_capture/*.sse`), in OpenAI Responses
// `event: <type>\ndata: <json>` SSE framing. They are the empirical ground
// truth for the wire shape opencode / ai-sdk / codex actually consume.
//
// For each captured stream this asserts:
//   * the SSE `event:` name EQUALS the payload's `type` (framing parity — codex
//     keys off the event name, ai-sdk off the payload type; they MUST agree),
//   * every `data:` payload deserializes through the gateway's
//     `ResponseStreamEvent` enum (no silent wire drift),
//   * `sequence_number` is monotonic from 0 with no gaps,
//   * NO `[DONE]` sentinel anywhere (the Responses protocol has none),
//   * the stream terminates on `response.completed`,
//   * the exact event-`type` ordering matches the documented lifecycle.

/// One parsed frame of a captured Responses SSE stream: the `event:` name and
/// the decoded `data:` JSON payload.
struct CapturedFrame {
    event_name: String,
    payload: Value,
}

/// Parse a captured `event: <type>\ndata: <json>` SSE body into ordered frames.
/// A `data: [DONE]` line is preserved as a frame with a `done` sentinel payload
/// so the no-`[DONE]` invariant can be asserted (it must never appear).
fn parse_captured_sse(body: &str) -> Vec<CapturedFrame> {
    let mut frames = Vec::new();
    let mut pending_event: Option<String> = None;
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix("event:") {
            pending_event = Some(name.trim().to_string());
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                frames.push(CapturedFrame {
                    event_name: pending_event.take().unwrap_or_default(),
                    payload: Value::String("[DONE]".to_string()),
                });
                continue;
            }
            let payload: Value = serde_json::from_str(data)
                .unwrap_or_else(|e| panic!("captured SSE data line is not JSON: {e}\n{data}"));
            frames.push(CapturedFrame {
                event_name: pending_event.take().unwrap_or_default(),
                payload,
            });
        }
    }
    frames
}

/// Validate a captured Responses SSE stream against the wire contract and the
/// expected event-type ordering.
fn assert_captured_stream(file: &str, expected_order: &[&str]) {
    let body = load_text(format!("responses_sse_capture/{file}"));
    let frames = parse_captured_sse(&body);
    assert!(!frames.is_empty(), "{file}: no SSE frames parsed");

    let mut order: Vec<String> = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        // No [DONE] sentinel — ever.
        assert_ne!(
            frame.payload,
            Value::String("[DONE]".to_string()),
            "{file}: [DONE] sentinel must never appear in a Responses stream"
        );

        let payload_type = frame
            .payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("{file}: frame[{i}] payload missing `type`"));

        // SSE event name must equal the payload type (framing parity).
        assert_eq!(
            frame.event_name, payload_type,
            "{file}: frame[{i}] SSE event name `{}` != payload type `{payload_type}`",
            frame.event_name
        );

        // Every payload must deserialize through the gateway's event enum.
        let event: ResponseStreamEvent = serde_json::from_value(frame.payload.clone())
            .unwrap_or_else(|e| panic!("{file}: frame[{i}] (`{payload_type}`) failed to deserialize through ResponseStreamEvent: {e}"));
        // And re-serialize to the same type tag (no drift).
        let back = serde_json::to_value(&event).expect("re-serialize");
        assert_eq!(
            back["type"], payload_type,
            "{file}: frame[{i}] round-trip changed the type tag"
        );

        // sequence_number monotonic from 0, no gaps.
        let actual_seq = frame
            .payload
            .get("sequence_number")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("{file}: frame[{i}] missing sequence_number"));
        assert_eq!(
            actual_seq, i as u64,
            "{file}: frame[{i}] sequence_number not monotonic-from-0"
        );

        order.push(payload_type.to_string());
    }

    // Exact event-type ordering.
    let actual_order: Vec<&str> = order.iter().map(String::as_str).collect();
    assert_eq!(
        actual_order, expected_order,
        "{file}: event-type ordering mismatch"
    );

    // Terminates on response.completed.
    assert_eq!(
        order.last().map(String::as_str),
        Some("response.completed"),
        "{file}: stream must terminate on response.completed"
    );
}

#[test]
fn captured_text_stream_matches_contract() {
    assert_captured_stream(
        "text_stream.sse",
        &[
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
    );
}

#[test]
fn captured_tool_stream_matches_contract() {
    // Tool lifecycle: the function_call item is framed by add+done with the full
    // JSON arguments on `.done`; NO function_call_arguments.delta is emitted
    // (codex reconstructs the call from the add/done pair).
    assert_captured_stream(
        "tool_stream.sse",
        &[
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
            "response.completed",
        ],
    );
    let body = load_text("responses_sse_capture/tool_stream.sse");
    assert!(
        !body.contains("function_call_arguments"),
        "captured tool stream must NOT contain function_call_arguments events"
    );
}

#[test]
fn captured_reasoning_stream_matches_contract() {
    // Reasoning item (added → reasoning_text.delta* → reasoning_text.done →
    // done) BEFORE the message item, then the message lifecycle, then completed.
    // Reasoning is a STRUCTURED item — never `<think>` on this surface.
    let body = load_text("responses_sse_capture/reasoning_stream.sse");
    assert!(
        !body.contains("<think>"),
        "reasoning must be a structured item on the Responses surface, never <think>"
    );

    let frames = parse_captured_sse(&body);
    let order: Vec<&str> = frames
        .iter()
        .map(|f| f.payload["type"].as_str().unwrap())
        .collect();
    // Reasoning item frames come BEFORE the message item frames.
    let first_reasoning = order
        .iter()
        .position(|t| *t == "response.reasoning_text.delta")
        .expect("reasoning delta present");
    let first_message_text = order
        .iter()
        .position(|t| *t == "response.output_text.delta")
        .expect("message text delta present");
    assert!(
        first_reasoning < first_message_text,
        "reasoning_text.delta must precede output_text.delta"
    );

    // Full per-frame contract validation (framing parity, monotonic seq, no
    // [DONE], ends on completed) without pinning the long delta-by-delta order.
    for (i, frame) in frames.iter().enumerate() {
        let payload_type = frame.payload["type"].as_str().unwrap();
        assert_eq!(frame.event_name, payload_type, "frame[{i}] framing parity");
        let event: ResponseStreamEvent = serde_json::from_value(frame.payload.clone())
            .unwrap_or_else(|e| panic!("frame[{i}] (`{payload_type}`) deserialize: {e}"));
        let _ = serde_json::to_value(&event).unwrap();
        let actual_seq = frame.payload["sequence_number"].as_u64().unwrap();
        assert_eq!(actual_seq, i as u64, "frame[{i}] sequence_number monotonic");
    }
    assert_eq!(
        order.last(),
        Some(&"response.completed"),
        "reasoning stream terminates on response.completed"
    );
}
