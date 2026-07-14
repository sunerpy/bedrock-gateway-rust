//! Unit and property-based tests for [`crate::bedrock::mantle_provider`],
//! relocated out of the source module for code organization (see the
//! `test-coverage-codecov` spec). The source file declares this via a
//! `#[path]` mod tests, so the top-level `use super::*;` resolves to the
//! implementation module. Property 4 (mantle model-name rewrite minimality)
//! lives in the nested `prop_tests` submodule.

use super::*;
use crate::domain::{ModelCapabilities, ResponsesBackend};
use crate::openai::responses_schema::{ResponsesInput, ResponsesRequest};
use std::collections::HashMap;
use std::time::Instant;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A capability stub that gates a single model to a fixed region set and
/// resolves `gpt-5.5` → `openai.gpt-5.5` (mirrors the T1/T2 alias contract).
/// Every other capability query returns a benign default — this provider only
/// consults `model_regions` + `resolve_foundation`.
struct StubCaps {
    regions: Option<Vec<String>>,
}

impl ModelCapabilities for StubCaps {
    fn has(&self, _model: &str, _cap: crate::domain::Capability) -> bool {
        false
    }
    fn resolve_foundation(&self, model_or_profile: &str) -> String {
        if model_or_profile == "gpt-5.5" {
            "openai.gpt-5.5".to_string()
        } else {
            model_or_profile.to_string()
        }
    }
    fn budget_ratios(&self, _model: &str) -> Option<crate::domain::BudgetRatios> {
        None
    }
    fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
        None
    }
    fn beta_headers(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }
    fn reasoning_path(&self, _model: &str) -> crate::domain::ReasoningPath {
        crate::domain::ReasoningPath::None
    }
    fn responses_backend(&self, _model: &str) -> ResponsesBackend {
        ResponsesBackend::Mantle
    }
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
    }
    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        self.regions.clone()
    }
}

fn settings_in_region(region: &str) -> Arc<AppSettings> {
    Arc::new(AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: region.to_string(),
        default_model: "m".to_string(),
        default_embedding_model: "e".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching: false,
        prompt_cache_ttl: "5m".to_string(),
        api_key: Some("k".to_string()),
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        bind_addr: "127.0.0.1".to_string(),
        port: 0,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        aws_max_retry_attempts: 8,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    })
}

fn provider_for(
    base_uri: &str,
    region: &str,
    regions: Option<Vec<String>>,
) -> MantleResponsesProvider {
    let client = MantleClient::new(
        reqwest::Client::new(),
        base_uri.to_string(),
        base_uri.to_string(),
        "test-bearer".to_string(),
    );
    let caps: Arc<dyn ModelCapabilities> = Arc::new(StubCaps { regions });
    MantleResponsesProvider::new(client, caps, settings_in_region(region))
}

fn normalized(raw: &str) -> NormalizedResponsesRequest {
    NormalizedResponsesRequest {
        request: ResponsesRequest {
            model: "gpt-5.5".to_string(),
            input: ResponsesInput::Text("hi".to_string()),
            instructions: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: None,
            reasoning: None,
            text: None,
            include: None,
            metadata: None,
            parallel_tool_calls: None,
            store: None,
            previous_response_id: None,
            extra: HashMap::new(),
        },
        resolved_model: "openai.gpt-5.5".to_string(),
        request_id: Arc::from("req-test"),
        received_at: Instant::now(),
        raw_body: Bytes::copy_from_slice(raw.as_bytes()),
    }
}

/// A mantle-shaped non-stream Responses JSON body with native usage.
const MANTLE_RESPONSE: &str = r#"{
        "id": "resp_abc",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "output": [
            {"type":"message","id":"msg_1","status":"completed","role":"assistant",
             "content":[{"type":"output_text","text":"hello","annotations":[]}]}
        ],
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        "model": "openai.gpt-5.5"
    }"#;

/// Given a 200 mantle Responses JSON,
/// When `respond` is called,
/// Then the body parses to `ResponsesResponse`, usage matches
/// `compute_token_usage(11,7,0,0)`, AND the request forwarded upstream had its
/// `model` rewritten to the canonical id with other fields intact.
#[tokio::test]
async fn respond_happy_path_rewrites_model_and_normalizes_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        // model rewritten to canonical AND the original `input` preserved
        // byte-stable (proves only `model` was patched).
        .and(body_partial_json(serde_json::json!({
            "model": "openai.gpt-5.5",
            "input": "hi",
            "keep_me": "intact"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(MANTLE_RESPONSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    // raw body carries the client's `model` (alias) + an extra field that
    // MUST survive byte-stable.
    let req = normalized(r#"{"model":"gpt-5.5","input":"hi","keep_me":"intact"}"#);

    let resp = provider
        .respond(&req)
        .await
        .expect("respond should succeed on 200");

    let expected = compute_token_usage(11, 7, 0, 0);
    assert_eq!(resp.usage.input_tokens, expected.prompt_tokens);
    assert_eq!(resp.usage.output_tokens, expected.completion_tokens);
    assert_eq!(resp.usage.total_tokens, expected.total_tokens);
    assert_eq!(resp.id, "resp_abc");
    assert_eq!(resp.model, "openai.gpt-5.5");
}

/// Given a model gated to a region the gateway is NOT running in,
/// When `respond` is called,
/// Then it fails with `AppError::BadRequest` BEFORE any HTTP call (wiremock
/// receives ZERO requests — enforced by `expect(0)`).
#[tokio::test]
async fn region_gate_rejects_before_any_http_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MANTLE_RESPONSE))
        .expect(0)
        .mount(&server)
        .await;

    // Running in us-west-2 but the model only allows us-east-2.
    let provider = provider_for(
        &server.uri(),
        "us-west-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-5.5","input":"hi"}"#);

    let err = provider
        .respond(&req)
        .await
        .expect_err("region gate must reject");
    assert!(matches!(err, AppError::BadRequest(_)));
    // `server` drop verifies `expect(0)`.
}

/// Given a 200 SSE stream from mantle,
/// When `respond_raw_stream` is called for an allowed region,
/// Then it returns `Some` and the forwarded bytes equal the upstream body.
#[tokio::test]
async fn raw_stream_happy_path_forwards_bytes() {
    const SSE: &str = "event: response.created\ndata: {\"type\":\"response.created\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(
            serde_json::json!({ "model": "openai.gpt-5.5" }),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-5.5","input":"hi","stream":true}"#);

    let stream = provider
        .respond_raw_stream(&req)
        .await
        .expect("raw stream should open on 200");
    let chunks: Vec<Bytes> = stream.map(|r| r.expect("ok chunk")).collect().await;
    let mut joined = Vec::new();
    for c in chunks {
        joined.extend_from_slice(&c);
    }
    assert_eq!(joined, SSE.as_bytes());
}

/// `respond_raw_stream` returns `None` on a region-gate failure (pre-stream),
/// so the typed path produces the error envelope. No HTTP call is made.
#[tokio::test]
async fn raw_stream_returns_none_when_region_gated() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .expect(0)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-west-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-5.5","input":"hi","stream":true}"#);
    assert!(provider.respond_raw_stream(&req).await.is_none());
}

/// A model with NO region allow-list (`None`) is served everywhere — the gate
/// passes regardless of the running region.
#[tokio::test]
async fn ungated_model_passes_region_gate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(
            serde_json::json!({ "model": "openai.gpt-5.5" }),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(MANTLE_RESPONSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server.uri(), "ap-southeast-1", None);
    let req = normalized(r#"{"model":"gpt-5.5","input":"hi"}"#);
    assert!(provider.respond(&req).await.is_ok());
}

/// A malformed raw body fails at the rewrite step with `BadRequest` and never
/// reaches the upstream.
#[tokio::test]
async fn malformed_body_rejected_before_http() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(MANTLE_RESPONSE))
        .expect(0)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized("not json");
    let err = provider
        .respond(&req)
        .await
        .expect_err("malformed body rejected");
    assert!(matches!(err, AppError::BadRequest(_)));
}

/// Property-based tests for the mantle model-name rewrite minimality.
///
/// Feature: test-coverage-codecov, Property 4: Mantle 模型名改写的最小性
/// (see `.kiro/specs/test-coverage-codecov/design.md`).
///
/// Validates: Requirements 1.2
mod prop_tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::{Map, Value};

    /// Strategy: an arbitrary JSON value (null / bool / integer / string /
    /// nested arrays and objects). Floats are deliberately excluded so the
    /// `serialize → rewrite → parse` round-trip is exact (no f64 precision
    /// drift), keeping the minimality assertion byte-faithful.
    fn arb_json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<i64>().prop_map(|i| Value::Number(i.into())),
            "[a-zA-Z0-9 _.-]{0,24}".prop_map(Value::String),
        ];
        leaf.prop_recursive(3, 24, 5, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
                prop::collection::hash_map("[a-zA-Z_][a-zA-Z0-9_]{0,7}", inner, 0..5)
                    .prop_map(|m| Value::Object(m.into_iter().collect())),
            ]
        })
    }

    /// Strategy: the non-`model` fields of a Responses request body. The
    /// reserved `model` key is filtered out so the test controls it explicitly
    /// (the rewrite always overwrites `model`).
    fn arb_body_fields() -> impl Strategy<Value = Map<String, Value>> {
        prop::collection::hash_map(
            "[a-zA-Z_][a-zA-Z0-9_]{0,10}"
                .prop_filter("reserve model key", |k| k.as_str() != "model"),
            arb_json_value(),
            0..8,
        )
        .prop_map(|m| m.into_iter().collect())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Feature: test-coverage-codecov, Property 4: Mantle 模型名改写的最小性.
        ///
        /// For any Responses request body (a JSON object) forwarded to the
        /// mantle backend, `rewrite_model` patches ONLY the top-level `"model"`
        /// key (to the canonical / alias-resolved foundation id) and preserves
        /// every other field verbatim — no field is added, removed, or mutated.
        #[test]
        fn prop_rewrite_only_touches_model_field(
            fields in arb_body_fields(),
            client_model in "[a-zA-Z0-9 _.-]{0,24}",
            request_model in "[a-zA-Z0-9._-]{1,32}",
        ) {
            // Incoming body: arbitrary fields + a client-supplied `model` value
            // that MUST be overwritten by the rewrite.
            let mut obj = fields.clone();
            obj.insert("model".to_string(), Value::String(client_model));
            let raw =
                Bytes::from(serde_json::to_vec(&Value::Object(obj)).expect("serialize body"));

            // The base URI is never contacted — `rewrite_model` is pure.
            let provider = provider_for("http://mantle.invalid", "us-east-2", None);
            let out = provider
                .rewrite_model(&request_model, &raw)
                .expect("rewrite must succeed on a JSON object body");
            let out_val: Value = serde_json::from_slice(&out).expect("output parses");
            let out_obj = out_val.as_object().expect("output is a JSON object");

            // (1) `model` rewritten to exactly the resolved foundation id.
            let canonical = provider.caps.resolve_foundation(&request_model);
            prop_assert_eq!(out_obj.get("model"), Some(&Value::String(canonical)));

            // (2) every non-model field preserved verbatim.
            for (k, v) in fields.iter() {
                prop_assert_eq!(out_obj.get(k), Some(v));
            }

            // (3) no key introduced beyond the original fields + `model`.
            for k in out_obj.keys() {
                prop_assert!(k == "model" || fields.contains_key(k));
            }

            // (4) exactly the original field count plus the always-present model key.
            prop_assert_eq!(out_obj.len(), fields.len() + 1);
        }
    }
}
