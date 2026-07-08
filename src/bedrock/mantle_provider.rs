//! [`ResponsesProvider`] for the Bedrock **mantle** (OpenAI-compatible) upstream.
//!
//! [`MantleResponsesProvider`] serves the Responses API for models whose config
//! entry declares `params.responses_backend = "mantle"` (e.g. the GPT-5.x
//! family). Unlike [`crate::bedrock::responses_provider::BedrockResponsesProvider`]
//! — which translates to/from Bedrock Converse — this provider is a **thin
//! routing shim** over the byte-oriented [`MantleClient`] (T4): mantle's upstream
//! already speaks the OpenAI Responses wire format, so this provider forwards the
//! client's verbatim request body and returns the upstream's verbatim response.
//!
//! ## What it does per request
//!
//! 1. **Region gate FIRST** (before any HTTP call): if the model declares a
//!    region allow-list ([`ModelCapabilities::model_regions`], T2) that does not
//!    contain the gateway's running region, fail with [`AppError::BadRequest`].
//! 2. **Model rewrite**: patch ONLY the top-level `"model"` key of the captured
//!    raw request body to the canonical foundation id
//!    ([`ModelCapabilities::resolve_foundation`], T2), keeping every other field
//!    byte-stable (this is why T5 captured `raw_body` — to preserve
//!    unknown/future fields). Forward that body upstream.
//! 3. **Non-stream** ([`Self::respond`]): call
//!    [`MantleClient::responses_nonstream`], deserialize the mantle JSON into a
//!    [`ResponsesResponse`], and re-normalize `usage` through the shared
//!    [`compute_token_usage`] helper (single source of truth — no hand-rolled
//!    token math).
//! 4. **Stream** ([`Self::respond_raw_stream`], the raw lane T5 added): open the
//!    upstream SSE via [`MantleClient::responses_stream`] and forward the bytes
//!    verbatim. The handler (T5) consults this BEFORE the typed
//!    [`Self::respond_stream`], so the raw lane is the happy path; the typed
//!    method is only a pre-stream-failure fallback that surfaces the error
//!    envelope.
//!
//! ## Logging discipline (AGENTS.md §4)
//!
//! The raw request body, the response body, and the bearer are NEVER logged.
//! Only structured metadata (region, streaming flag) appears in traces.

use std::sync::Arc;

use bytes::Bytes;
use futures::stream::StreamExt;
use serde_json::Value;

use crate::bedrock::mantle_client::MantleClient;
use crate::bedrock::tokens::compute_token_usage;
use crate::config::AppSettings;
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, RawResponsesStream, ResponsesProvider,
    ResponsesStream,
};
use crate::error::AppError;
use crate::openai::responses_schema::ResponsesResponse;

/// A [`ResponsesProvider`] that routes mantle-backed models (GPT-5.x) to the
/// OpenAI-compatible bedrock-mantle upstream via [`MantleClient`].
///
/// Cheap to clone — [`MantleClient`] is reference-counted internally and the
/// other fields are `Arc`-wrapped.
#[derive(Clone)]
pub struct MantleResponsesProvider {
    /// The byte-oriented mantle HTTP client (T4).
    client: MantleClient,
    /// Capability resolver (T2): backend selection, region allow-list, and
    /// alias→canonical foundation resolution. The ONLY source of model routing
    /// knowledge — no model-name string branching lives here.
    caps: Arc<dyn ModelCapabilities>,
    /// Application settings; `aws_region` is the gateway's running region used
    /// for the region gate and as the upstream `{region}` substitution.
    settings: Arc<AppSettings>,
}

impl MantleResponsesProvider {
    /// Construct a provider from its collaborators.
    ///
    /// `client` is a fully-built [`MantleClient`] (the caller — `build_app_state`
    /// in T7 — wires the shared `reqwest::Client`, the
    /// `mantle_base_url_template`, and the `bedrock_api_key` bearer into it).
    /// `caps` is the shared capability resolver; `settings` carries the running
    /// `aws_region`.
    #[must_use]
    pub fn new(
        client: MantleClient,
        caps: Arc<dyn ModelCapabilities>,
        settings: Arc<AppSettings>,
    ) -> Self {
        Self {
            client,
            caps,
            settings,
        }
    }

    /// Enforce the per-model region allow-list against the gateway's running
    /// region. Returns the region to call (`settings.aws_region`) on success, or
    /// [`AppError::BadRequest`] when the model is gated out of this region.
    ///
    /// `None` from [`ModelCapabilities::model_regions`] means no gate (served
    /// everywhere). Runs BEFORE any HTTP call.
    fn region_gate(&self, model: &str) -> Result<String, AppError> {
        let region = &self.settings.aws_region;
        if let Some(regions) = self.caps.model_regions(model) {
            if !regions.contains(region) {
                return Err(AppError::BadRequest(format!(
                    "model {model} is not available in region {region}"
                )));
            }
        }
        Ok(region.clone())
    }

    /// Patch ONLY the top-level `"model"` key of the captured raw request body to
    /// the canonical foundation id, keeping every other field byte-stable.
    ///
    /// The body is parsed to a [`serde_json::Value`], its `model` key is set to
    /// `resolve_foundation(model)`, and it is reserialized. A malformed body
    /// surfaces as [`AppError::BadRequest`] (it never reached the upstream).
    fn rewrite_model(&self, model: &str, raw_body: &Bytes) -> Result<Bytes, AppError> {
        let canonical = self.caps.resolve_foundation(model);
        let mut value: Value = serde_json::from_slice(raw_body)
            .map_err(|e| AppError::BadRequest(format!("invalid responses request body: {e}")))?;
        match value.as_object_mut() {
            Some(obj) => {
                obj.insert("model".to_string(), Value::String(canonical));
            }
            None => {
                return Err(AppError::BadRequest(
                    "responses request body must be a JSON object".to_string(),
                ));
            }
        }
        let bytes = serde_json::to_vec(&value)
            .map_err(|e| AppError::Internal(format!("failed to reserialize request body: {e}")))?;
        Ok(Bytes::from(bytes))
    }

    /// Run the shared pre-flight (region gate + model rewrite) for a request,
    /// returning the `(region, rewritten_body)` pair forwarded upstream.
    fn preflight(&self, req: &NormalizedResponsesRequest) -> Result<(String, Bytes), AppError> {
        let model = &req.request.model;
        let region = self.region_gate(model)?;
        let body = self.rewrite_model(model, &req.raw_body)?;
        Ok((region, body))
    }
}

/// Re-normalize a mantle [`ResponsesResponse`]'s `usage` through the shared
/// [`compute_token_usage`] helper so token accounting matches the rest of the
/// gateway (single source of truth). Mantle reports OpenAI-native usage, so
/// `cacheRead`/`cacheWrite` are `0`.
fn normalize_usage(mut resp: ResponsesResponse) -> ResponsesResponse {
    let counts = compute_token_usage(resp.usage.input_tokens, resp.usage.output_tokens, 0, 0);
    resp.usage.input_tokens = counts.prompt_tokens;
    resp.usage.output_tokens = counts.completion_tokens;
    resp.usage.total_tokens = counts.total_tokens;
    resp
}

#[async_trait::async_trait]
impl ResponsesProvider for MantleResponsesProvider {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        let (region, body) = self.preflight(req)?;

        tracing::debug!(region = %region, "invoking mantle responses (non-stream)");

        let bytes = self.client.responses_nonstream(&region, body).await?;
        let resp: ResponsesResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AppError::UpstreamBedrock(format!("failed to parse mantle response: {e}"))
        })?;
        Ok(normalize_usage(resp))
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        // The raw lane ([`Self::respond_raw_stream`]) is the happy path for
        // streaming — the handler consults it FIRST (T5). This typed method is
        // only reached as a fallback when the raw lane returned `None`, i.e. a
        // pre-stream failure. Re-run the pre-flight + connect to surface that
        // failure as a proper error envelope. A successful connect here is
        // unexpected (the raw lane would have claimed it); treat it as an
        // internal routing error rather than silently dropping the SSE.
        let (region, body) = self.preflight(req)?;
        match self.client.responses_stream(&region, body).await {
            Ok(_stream) => Err(AppError::Internal(
                "mantle streaming must use the raw passthrough lane".to_string(),
            )),
            Err(e) => Err(e),
        }
    }

    async fn respond_raw_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Option<RawResponsesStream> {
        // Pre-flight (region gate + model rewrite). A pre-stream failure returns
        // `None` so the typed `respond_stream` path produces the error envelope
        // (T5's recommended contract: `Some` only once the upstream SSE is
        // established).
        let (region, body) = self.preflight(req).ok()?;

        tracing::debug!(region = %region, "invoking mantle responses (raw stream)");

        let stream = self.client.responses_stream(&region, body).await.ok()?;
        Some(stream.boxed())
    }
}

#[cfg(test)]
mod tests {
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
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
            allowed_models: None,
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
}
