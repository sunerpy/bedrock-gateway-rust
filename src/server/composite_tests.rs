//! Unit and property-based tests for [`crate::server::composite`], relocated
//! out of the source module for code organization (see the
//! `test-coverage-codecov` spec). The source file declares this via a
//! `#[path]` mod tests, so the top-level `use super::*;` resolves to the
//! implementation module. The composite-dispatch property lives in the nested
//! `prop_tests` submodule.

use super::*;
use crate::config::capabilities::{ModelEntry, ModelParams};
use crate::domain::{BudgetRatios, Capability, ReasoningPath, ResponsesProvider};
use crate::openai::responses_schema::{
    OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesInput, ResponsesRequest,
    ResponsesResponse, ResponsesUsage,
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
    ) -> Result<Option<RawResponsesStream>, AppError> {
        self.respond_raw_hit.store(true, Ordering::SeqCst);
        let chunk: Result<Bytes, AppError> = Ok(Bytes::from(self.tag.as_bytes()));
        Ok(Some(Box::pin(futures::stream::iter(vec![chunk]))))
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
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
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
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
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
        responses_stream_idle_timeout_secs: 180,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
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
    build_with_mantle_enabled(true)
}

fn build_with_mantle_enabled(
    mantle_enabled: bool,
) -> (
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
        mantle_enabled,
    );
    (composite, converse, mantle)
}

fn assert_mantle_disabled_err(err: AppError) {
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("requires a Bedrock API key"));
            assert!(message.contains("AWS_BEARER_TOKEN_BEDROCK"));
            assert!(message.contains("disabled on this instance"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
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
        .expect("raw lane succeeds")
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

/// Given a mantle-routed model but mantle is disabled on this instance,
/// When composite methods are called,
/// Then typed paths return a clean 400 and no inner provider is hit.
#[tokio::test]
async fn mantle_model_returns_bad_request_when_mantle_disabled() {
    let (composite, converse, mantle) = build_with_mantle_enabled(false);
    let req = normalized("gpt-5.5");

    match composite.respond(&req).await {
        Err(err) => assert_mantle_disabled_err(err),
        Ok(_) => panic!("respond should fail when mantle is disabled"),
    }

    match composite.respond_stream(&req).await {
        Err(err) => assert_mantle_disabled_err(err),
        Ok(_) => panic!("respond_stream should fail when mantle is disabled"),
    }

    match composite.respond_raw_stream(&req).await {
        Err(err) => assert_mantle_disabled_err(err),
        Ok(_) => panic!("raw stream should fail when mantle is disabled"),
    }
    assert!(!mantle.respond_hit.load(Ordering::SeqCst));
    assert!(!mantle.respond_stream_hit.load(Ordering::SeqCst));
    assert!(!mantle.respond_raw_hit.load(Ordering::SeqCst));
    assert!(!converse.respond_hit.load(Ordering::SeqCst));
    assert!(!converse.respond_stream_hit.load(Ordering::SeqCst));
    assert!(!converse.respond_raw_hit.load(Ordering::SeqCst));
}

/// Given a claude model (Converse backend),
/// When each composite method is called,
/// Then the CONVERSE inner provider is hit and the mantle one is not. The
/// raw-stream lane delegates to converse's default (`Ok(None)`).
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
    let _ = composite
        .respond_raw_stream(&req)
        .await
        .expect("raw lane succeeds");

    assert!(converse.respond_hit.load(Ordering::SeqCst));
    assert!(converse.respond_stream_hit.load(Ordering::SeqCst));
    assert!(converse.respond_raw_hit.load(Ordering::SeqCst));
    assert!(!mantle.respond_hit.load(Ordering::SeqCst));
    assert!(!mantle.respond_stream_hit.load(Ordering::SeqCst));
    assert!(!mantle.respond_raw_hit.load(Ordering::SeqCst));
}

/// Given a config with a mantle model but `bedrock_api_key: None`,
/// When `resolve_mantle_enabled` runs,
/// Then it disables mantle without failing boot.
#[test]
fn startup_disables_mantle_when_model_but_no_bearer() {
    let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
    let settings = settings_with(None, "us-east-2");
    assert!(!resolve_mantle_enabled(&config, &settings));
}

#[test]
fn startup_passes_when_mantle_disabled_without_bearer() {
    let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
    let mut settings = settings_with(None, "us-east-2");
    settings.disable_mantle = true;

    assert!(!resolve_mantle_enabled(&config, &settings));
}

/// A mantle model WITH a bearer configured passes validation.
#[test]
fn startup_passes_when_mantle_model_has_bearer() {
    let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
    let settings = settings_with(Some("bedrock-bearer".to_string()), "us-east-2");
    assert!(resolve_mantle_enabled(&config, &settings));
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
    assert!(!resolve_mantle_enabled(&config, &settings));
}

/// A mantle model whose region allow-list omits the running region passes
/// (WARNING, not a hard fail) as long as the bearer is present.
#[test]
fn startup_warns_but_passes_on_region_mismatch() {
    let config = mantle_model_config(Some(vec!["us-east-2".to_string()]));
    let settings = settings_with(Some("bedrock-bearer".to_string()), "us-west-2");
    assert!(resolve_mantle_enabled(&config, &settings));
}

mod prop_tests {
    use super::{build, normalized, RoutingCaps};
    use crate::domain::{ModelCapabilities, ResponsesBackend, ResponsesProvider};
    use proptest::prelude::*;
    use std::sync::atomic::Ordering;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov, Property: composite-dispatch.
        /// 对任意 model 字符串，`CompositeResponsesProvider::respond` 分发到的
        /// inner provider 与 `caps.responses_backend(model)` 选定的后端一致
        /// （分发一致性）。使用 stub `ResponsesProvider` + stub `ModelCapabilities`，
        /// 无 AWS/网络依赖。
        #[test]
        fn respond_dispatches_to_backend_chosen_by_caps(model in ".*") {
            let (composite, converse, mantle) = build();
            let req = normalized(&model);

            // The backend the caps resolver picks for this exact model string.
            let expected = RoutingCaps.responses_backend(&model);

            // mantle_enabled == true here, so no request is short-circuited: the
            // dispatch lands on whichever inner provider `expected` names.
            futures::executor::block_on(composite.respond(&req)).expect("respond ok");

            match expected {
                ResponsesBackend::Mantle => {
                    prop_assert!(mantle.respond_hit.load(Ordering::SeqCst));
                    prop_assert!(!converse.respond_hit.load(Ordering::SeqCst));
                }
                ResponsesBackend::Converse => {
                    prop_assert!(converse.respond_hit.load(Ordering::SeqCst));
                    prop_assert!(!mantle.respond_hit.load(Ordering::SeqCst));
                }
            }
        }
    }
}
