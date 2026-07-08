//! Composite [`ResponsesProvider`] that routes by backend + startup validation.
//!
//! The gateway serves the Responses API from TWO backends — the default Bedrock
//! Converse path ([`crate::bedrock::responses_provider::BedrockResponsesProvider`])
//! and the OpenAI-compatible bedrock-mantle upstream
//! ([`crate::bedrock::mantle_provider::MantleResponsesProvider`]) for the GPT-5.x
//! family. [`CompositeResponsesProvider`] hides both behind the single
//! `Arc<dyn ResponsesProvider>` that [`crate::server::state::AppState`] holds, so
//! `AppState`'s public shape is unchanged.
//!
//! ## Routing
//!
//! Routing is purely config-driven (zero model-name branching): every method
//! consults [`ModelCapabilities::responses_backend`] for the client's requested
//! model (which resolves the alias internally, T2) and dispatches to the matching
//! inner provider. [`ResponsesBackend::Mantle`] → the mantle provider;
//! [`ResponsesBackend::Converse`] (the default for every other model) → the
//! Converse provider.
//!
//! The raw-bytes streaming lane ([`ResponsesProvider::respond_raw_stream`]) is
//! overridden here and delegated to the selected inner provider — otherwise the
//! trait default (`None`) would win and the mantle raw passthrough would never
//! fire.
//!
//! ## Startup validation
//!
//! [`validate_mantle_startup`] scans the model config for any entry declaring the
//! mantle backend and fails boot fast if no `bedrock_api_key` is configured (the
//! mantle upstream needs that bearer). It also emits a WARNING — not a hard fail —
//! when a mantle model's region allow-list omits the gateway's running region, so
//! a misconfiguration surfaces at boot while the per-request gate (T6) still 400s.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::capabilities::ModelCapabilityConfig;
use crate::config::AppSettings;
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, RawResponsesStream, ResponsesBackend,
    ResponsesProvider, ResponsesStream,
};
use crate::error::AppError;
use crate::openai::responses_schema::ResponsesResponse;

/// The config value of `params.responses_backend` that selects the mantle
/// upstream. Matches the `ResponsesBackend::Mantle` mapping in
/// `crate::bedrock::capabilities`.
const MANTLE_BACKEND: &str = "mantle";

/// A [`ResponsesProvider`] that dispatches each request to one of two inner
/// providers based on the model's configured backend.
///
/// Cheap to clone — every field is `Arc`-backed.
#[derive(Clone)]
pub struct CompositeResponsesProvider {
    /// The default Bedrock Converse-backed Responses provider.
    converse: Arc<dyn ResponsesProvider>,
    /// The bedrock-mantle (OpenAI-compatible) Responses provider.
    mantle: Arc<dyn ResponsesProvider>,
    /// The capability resolver used to pick the backend per request. Sharing the
    /// SAME `Arc` the inner providers hold keeps routing consistent with their
    /// own alias/region resolution.
    caps: Arc<dyn ModelCapabilities>,
}

impl CompositeResponsesProvider {
    /// Construct the composite from its two inner providers and the shared
    /// capability resolver.
    #[must_use]
    pub fn new(
        converse: Arc<dyn ResponsesProvider>,
        mantle: Arc<dyn ResponsesProvider>,
        caps: Arc<dyn ModelCapabilities>,
    ) -> Self {
        Self {
            converse,
            mantle,
            caps,
        }
    }

    /// Select the inner provider for a request by its configured backend.
    ///
    /// `responses_backend` resolves the alias internally, so the client's
    /// requested model id is passed through unchanged.
    fn select(&self, req: &NormalizedResponsesRequest) -> &Arc<dyn ResponsesProvider> {
        match self.caps.responses_backend(&req.request.model) {
            ResponsesBackend::Mantle => &self.mantle,
            ResponsesBackend::Converse => &self.converse,
        }
    }
}

#[async_trait]
impl ResponsesProvider for CompositeResponsesProvider {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        self.select(req).respond(req).await
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        self.select(req).respond_stream(req).await
    }

    async fn respond_raw_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Option<RawResponsesStream> {
        // MUST delegate: the trait default returns `None`, so without this
        // override the mantle raw passthrough lane would never fire.
        self.select(req).respond_raw_stream(req).await
    }
}

/// Validate the mantle-backend configuration at boot.
///
/// Fail-fast rule (unless `disable_mantle` is true): if ANY model entry declares
/// `params.responses_backend = "mantle"` but no `bedrock_api_key` is configured,
/// return an error (the mantle upstream requires that bearer; booting without it
/// would 401 every GPT request at runtime with a confusing error).
///
/// Soft rule: for each mantle model whose `available_regions` allow-list is set
/// and does NOT contain the gateway's running region, log a WARNING. This surfaces
/// a likely misconfiguration at boot without hard-failing — the per-request region
/// gate (T6) still returns a 400, and another region's deployment of the same
/// config is legitimate.
pub fn validate_mantle_startup(
    config: &ModelCapabilityConfig,
    settings: &AppSettings,
) -> Result<(), AppError> {
    let mantle_models: Vec<&crate::config::capabilities::ModelEntry> = config
        .models
        .iter()
        .filter(|e| e.params.responses_backend.as_deref() == Some(MANTLE_BACKEND))
        .collect();

    if settings.disable_mantle {
        tracing::warn!(
            "mantle backend disabled (DISABLE_MANTLE=true); GPT-5.x models will be unavailable"
        );
        return Ok(());
    }

    if mantle_models.is_empty() {
        return Ok(());
    }

    if settings.bedrock_api_key.is_none() {
        return Err(AppError::Internal(
            "a model is configured for the mantle backend but no bedrock_api_key is set"
                .to_string(),
        ));
    }

    for entry in mantle_models {
        if let Some(regions) = entry.params.available_regions.as_deref() {
            if !regions.contains(&settings.aws_region) {
                tracing::warn!(
                    model = %entry.match_pattern,
                    region = %settings.aws_region,
                    "mantle model is not in its configured region allow-list for this region; \
                     requests for it will be rejected with 400 here"
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::capabilities::{ModelEntry, ModelParams};
    use crate::domain::{BudgetRatios, Capability, ReasoningPath, ResponsesProvider};
    use crate::openai::responses_schema::{
        OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesInput,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    };
    use bytes::Bytes;
    use futures::stream::StreamExt;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    /// A `ResponsesProvider` that records whether each method was invoked, so a
    /// routing test can assert which inner provider a request landed on. Each
    /// method returns a marker carrying `self.tag` so the value is checkable too.
    struct RecordingProvider {
        tag: &'static str,
        respond_hit: AtomicBool,
        respond_stream_hit: AtomicBool,
        respond_raw_hit: AtomicBool,
    }

    impl RecordingProvider {
        fn new(tag: &'static str) -> Arc<Self> {
            Arc::new(Self {
                tag,
                respond_hit: AtomicBool::new(false),
                respond_stream_hit: AtomicBool::new(false),
                respond_raw_hit: AtomicBool::new(false),
            })
        }
    }

    #[async_trait]
    impl ResponsesProvider for RecordingProvider {
        async fn respond(
            &self,
            _req: &NormalizedResponsesRequest,
        ) -> Result<ResponsesResponse, AppError> {
            self.respond_hit.store(true, Ordering::SeqCst);
            Ok(canned_response(self.tag))
        }

        async fn respond_stream(
            &self,
            _req: &NormalizedResponsesRequest,
        ) -> Result<ResponsesStream, AppError> {
            self.respond_stream_hit.store(true, Ordering::SeqCst);
            let event = ResponseStreamEvent::OutputTextDelta {
                item_id: self.tag.to_string(),
                output_index: 0,
                content_index: 0,
                delta: self.tag.to_string(),
                sequence_number: 0,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(event)])))
        }

        async fn respond_raw_stream(
            &self,
            _req: &NormalizedResponsesRequest,
        ) -> Option<RawResponsesStream> {
            self.respond_raw_hit.store(true, Ordering::SeqCst);
            let chunk: Result<Bytes, AppError> = Ok(Bytes::from(self.tag.as_bytes()));
            Some(Box::pin(futures::stream::iter(vec![chunk])))
        }
    }

    fn canned_response(model: &str) -> ResponsesResponse {
        ResponsesResponse {
            id: "resp-test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            status: "completed".to_string(),
            output: vec![ResponseOutputItem::Message {
                id: "msg".to_string(),
                status: "completed".to_string(),
                role: "assistant".to_string(),
                content: vec![OutputContentPart::OutputText {
                    text: model.to_string(),
                    annotations: Vec::new(),
                    logprobs: None,
                }],
            }],
            usage: ResponsesUsage {
                input_tokens: 1,
                input_tokens_details: None,
                output_tokens: 2,
                output_tokens_details: None,
                total_tokens: 3,
            },
            model: model.to_string(),
            instructions: None,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            error: None,
            incomplete_details: None,
        }
    }

    /// A caps stub that maps `gpt-*` (substring `gpt`) to Mantle and everything
    /// else to Converse — the routing input the composite keys on. This is a TEST
    /// double; production routing is pure config data (no name branching).
    struct RoutingCaps;

    impl ModelCapabilities for RoutingCaps {
        fn has(&self, _model: &str, _cap: Capability) -> bool {
            false
        }
        fn resolve_foundation(&self, model_or_profile: &str) -> String {
            model_or_profile.to_string()
        }
        fn budget_ratios(&self, _model: &str) -> Option<BudgetRatios> {
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
        fn reasoning_path(&self, _model: &str) -> ReasoningPath {
            ReasoningPath::None
        }
        fn responses_backend(&self, model: &str) -> ResponsesBackend {
            if model.contains("gpt") {
                ResponsesBackend::Mantle
            } else {
                ResponsesBackend::Converse
            }
        }
        fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
            None
        }
    }

    fn normalized(model: &str) -> NormalizedResponsesRequest {
        NormalizedResponsesRequest {
            request: ResponsesRequest {
                model: model.to_string(),
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
            resolved_model: model.to_string(),
            request_id: Arc::from("req-test"),
            received_at: Instant::now(),
            raw_body: Bytes::new(),
        }
    }

    fn settings_with(bedrock_api_key: Option<String>, region: &str) -> AppSettings {
        AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: region.to_string(),
            default_model: "m".to_string(),
            default_embedding_model: "e".to_string(),
            enable_cross_region_inference: false,
            enable_application_inference_profiles: false,
            enable_prompt_caching: false,
            disable_mantle: false,
            api_key: Some("k".to_string()),
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key,
            bind_addr: "127.0.0.1".to_string(),
            port: 0,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
        }
    }

    fn mantle_model_config(available_regions: Option<Vec<String>>) -> ModelCapabilityConfig {
        ModelCapabilityConfig {
            models: vec![ModelEntry {
                match_pattern: "openai.gpt-5.5".to_string(),
                capabilities: Vec::new(),
                params: ModelParams {
                    responses_backend: Some("mantle".to_string()),
                    available_regions,
                    ..ModelParams::default()
                },
            }],
            ..ModelCapabilityConfig::default()
        }
    }

    fn build() -> (
        CompositeResponsesProvider,
        Arc<RecordingProvider>,
        Arc<RecordingProvider>,
    ) {
        let converse = RecordingProvider::new("converse");
        let mantle = RecordingProvider::new("mantle");
        let caps: Arc<dyn ModelCapabilities> = Arc::new(RoutingCaps);
        let composite = CompositeResponsesProvider::new(
            converse.clone() as Arc<dyn ResponsesProvider>,
            mantle.clone() as Arc<dyn ResponsesProvider>,
            caps,
        );
        (composite, converse, mantle)
    }

    /// Given a gpt model (Mantle backend),
    /// When each composite method is called,
    /// Then the MANTLE inner provider is hit and the converse one is not.
    #[tokio::test]
    async fn routes_gpt_model_to_mantle_inner() {
        let (composite, converse, mantle) = build();
        let req = normalized("gpt-5.5");

        let resp = composite.respond(&req).await.expect("respond ok");
        assert_eq!(resp.model, "mantle");

        let mut s = composite.respond_stream(&req).await.expect("stream ok");
        let first = s.next().await.expect("event").expect("ok");
        assert!(matches!(first, ResponseStreamEvent::OutputTextDelta { .. }));

        let raw = composite
            .respond_raw_stream(&req)
            .await
            .expect("raw stream Some");
        let bytes: Vec<Bytes> = raw.map(|r| r.expect("ok")).collect().await;
        assert_eq!(bytes[0], Bytes::from(&b"mantle"[..]));

        assert!(mantle.respond_hit.load(Ordering::SeqCst));
        assert!(mantle.respond_stream_hit.load(Ordering::SeqCst));
        assert!(mantle.respond_raw_hit.load(Ordering::SeqCst));
        assert!(!converse.respond_hit.load(Ordering::SeqCst));
        assert!(!converse.respond_stream_hit.load(Ordering::SeqCst));
        assert!(!converse.respond_raw_hit.load(Ordering::SeqCst));
    }

    /// Given a claude model (Converse backend),
    /// When each composite method is called,
    /// Then the CONVERSE inner provider is hit and the mantle one is not. The
    /// raw-stream lane delegates to converse's default (`None`).
    #[tokio::test]
    async fn routes_claude_model_to_converse_inner() {
        let (composite, converse, mantle) = build();
        let req = normalized("anthropic.claude-sonnet-4-5");

        let resp = composite.respond(&req).await.expect("respond ok");
        assert_eq!(resp.model, "converse");

        let mut s = composite.respond_stream(&req).await.expect("stream ok");
        let _ = s.next().await.expect("event").expect("ok");

        // converse RecordingProvider DOES return Some for raw — the point of this
        // assertion is that it routed to converse, not mantle.
        let _ = composite.respond_raw_stream(&req).await;

        assert!(converse.respond_hit.load(Ordering::SeqCst));
        assert!(converse.respond_stream_hit.load(Ordering::SeqCst));
        assert!(converse.respond_raw_hit.load(Ordering::SeqCst));
        assert!(!mantle.respond_hit.load(Ordering::SeqCst));
        assert!(!mantle.respond_stream_hit.load(Ordering::SeqCst));
        assert!(!mantle.respond_raw_hit.load(Ordering::SeqCst));
    }

    /// Given a config with a mantle model but `bedrock_api_key: None`,
    /// When `validate_mantle_startup` runs,
    /// Then it returns `Err` (fail-fast at boot).
    #[test]
    fn startup_fails_when_mantle_model_but_no_bearer() {
        let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
        let settings = settings_with(None, "us-east-2");
        let err =
            validate_mantle_startup(&config, &settings).expect_err("missing bearer must fail boot");
        assert!(matches!(err, AppError::Internal(_)));
    }

    #[test]
    fn startup_passes_when_mantle_disabled_without_bearer() {
        let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
        let mut settings = settings_with(None, "us-east-2");
        settings.disable_mantle = true;

        assert!(validate_mantle_startup(&config, &settings).is_ok());
    }

    /// A mantle model WITH a bearer configured passes validation.
    #[test]
    fn startup_passes_when_mantle_model_has_bearer() {
        let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
        let settings = settings_with(Some("bedrock-bearer".to_string()), "us-east-2");
        assert!(validate_mantle_startup(&config, &settings).is_ok());
    }

    /// A config with NO mantle model passes regardless of the bearer (no gate).
    #[test]
    fn startup_passes_when_no_mantle_model() {
        let config = ModelCapabilityConfig {
            models: vec![ModelEntry {
                match_pattern: "anthropic.claude-sonnet-4-5".to_string(),
                capabilities: Vec::new(),
                params: ModelParams::default(),
            }],
            ..ModelCapabilityConfig::default()
        };
        let settings = settings_with(None, "us-east-2");
        assert!(validate_mantle_startup(&config, &settings).is_ok());
    }

    /// A mantle model whose region allow-list omits the running region passes
    /// (WARNING, not a hard fail) as long as the bearer is present.
    #[test]
    fn startup_warns_but_passes_on_region_mismatch() {
        let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
        let settings = settings_with(Some("bedrock-bearer".to_string()), "us-west-2");
        assert!(validate_mantle_startup(&config, &settings).is_ok());
    }
}
