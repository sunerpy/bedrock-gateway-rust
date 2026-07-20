//! Unit tests for [`crate::server::composite_chat`], mirroring
//! `composite_tests.rs`. A `RecordingChatProvider` with an `AtomicBool` per
//! method lets a routing test assert which inner provider a request landed on.

use super::*;
use crate::config::capabilities::{ModelEntry, ModelParams};
use crate::domain::{BudgetRatios, Capability, ReasoningPath, ResponsesBackend};
use crate::openai::schema::{
    ChatRequest, ChatResponse, ChatResponseMessage, Choice, ContentInput, Message, Usage,
};
use bytes::Bytes;
use futures::stream::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

struct RecordingChatProvider {
    tag: &'static str,
    chat_hit: AtomicBool,
    chat_stream_hit: AtomicBool,
    chat_raw_stream_hit: AtomicBool,
    chat_raw_nonstream_hit: AtomicBool,
}

impl RecordingChatProvider {
    fn new(tag: &'static str) -> Arc<Self> {
        Arc::new(Self {
            tag,
            chat_hit: AtomicBool::new(false),
            chat_stream_hit: AtomicBool::new(false),
            chat_raw_stream_hit: AtomicBool::new(false),
            chat_raw_nonstream_hit: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl ChatProvider for RecordingChatProvider {
    async fn chat(&self, _req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        self.chat_hit.store(true, Ordering::SeqCst);
        Ok(canned_response(self.tag))
    }

    async fn chat_stream(&self, _req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        self.chat_stream_hit.store(true, Ordering::SeqCst);
        let chunk = crate::openai::schema::ChatStreamResponse {
            id: self.tag.to_string(),
            created: 0,
            model: self.tag.to_string(),
            system_fingerprint: "fp".to_string(),
            choices: Vec::new(),
            object: "chat.completion.chunk".to_string(),
            usage: None,
        };
        Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
    }

    async fn chat_raw_stream(
        &self,
        _req: &NormalizedChatRequest,
    ) -> Result<Option<RawChatStream>, AppError> {
        self.chat_raw_stream_hit.store(true, Ordering::SeqCst);
        let chunk: Result<Bytes, AppError> = Ok(Bytes::from(self.tag.as_bytes()));
        Ok(Some(Box::pin(futures::stream::iter(vec![chunk]))))
    }

    async fn chat_raw_nonstream(
        &self,
        _req: &NormalizedChatRequest,
    ) -> Option<Result<Bytes, AppError>> {
        self.chat_raw_nonstream_hit.store(true, Ordering::SeqCst);
        Some(Ok(Bytes::from(self.tag.as_bytes())))
    }
}

fn canned_response(model: &str) -> ChatResponse {
    ChatResponse {
        id: "chatcmpl-test".to_string(),
        created: 0,
        model: model.to_string(),
        system_fingerprint: "fp".to_string(),
        choices: vec![Choice {
            index: 0,
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            message: ChatResponseMessage {
                role: Some("assistant".to_string()),
                content: Some(model.to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
        }],
        object: "chat.completion".to_string(),
        usage: Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
    }
}

/// A caps stub that maps substring `gpt-oss` to Mantle and everything else to
/// Converse. TEST double; production routing is pure config data.
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
    fn responses_backend(&self, _model: &str) -> ResponsesBackend {
        ResponsesBackend::Converse
    }
    fn chat_backend(&self, model: &str) -> ChatBackend {
        if model.contains("gpt-oss") {
            ChatBackend::Mantle
        } else if model.contains("gpt-responses") {
            ChatBackend::Responses
        } else {
            ChatBackend::Converse
        }
    }
    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        None
    }
}

fn normalized(model: &str) -> NormalizedChatRequest {
    NormalizedChatRequest {
        request: ChatRequest {
            messages: vec![Message::User {
                name: None,
                content: ContentInput::Text("hi".to_string()),
            }],
            model: model.to_string(),
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            user: None,
            max_tokens: None,
            max_completion_tokens: None,
            reasoning_effort: None,
            n: None,
            tools: None,
            tool_choice: Default::default(),
            stop: None,
            response_format: None,
            extra_body: None,
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

fn mantle_chat_model_config(available_regions: Option<Vec<String>>) -> ModelCapabilityConfig {
    ModelCapabilityConfig {
        models: vec![ModelEntry {
            match_pattern: "openai.gpt-oss-120b".to_string(),
            capabilities: Vec::new(),
            params: ModelParams {
                chat_backend: Some("mantle".to_string()),
                available_regions,
                ..ModelParams::default()
            },
        }],
        ..ModelCapabilityConfig::default()
    }
}

fn build_with_mantle_enabled(
    mantle_enabled: bool,
) -> (
    CompositeChatProvider,
    Arc<RecordingChatProvider>,
    Arc<RecordingChatProvider>,
    Arc<RecordingChatProvider>,
) {
    let converse = RecordingChatProvider::new("converse");
    let mantle = RecordingChatProvider::new("mantle");
    let responses = RecordingChatProvider::new("responses");
    let caps: Arc<dyn ModelCapabilities> = Arc::new(RoutingCaps);
    let composite = CompositeChatProvider::new(
        converse.clone() as Arc<dyn ChatProvider>,
        mantle.clone() as Arc<dyn ChatProvider>,
        responses.clone() as Arc<dyn ChatProvider>,
        caps,
        mantle_enabled,
    );
    (composite, converse, mantle, responses)
}

#[tokio::test]
async fn routes_mantle_backend_to_mantle_provider() {
    let (composite, converse, mantle, responses) = build_with_mantle_enabled(true);
    let req = normalized("gpt-oss-120b");

    let resp = composite.chat(&req).await.expect("chat ok");
    assert_eq!(resp.model, "mantle");

    let mut s = composite.chat_stream(&req).await.expect("stream ok");
    let _ = s.next().await.expect("chunk").expect("ok");

    let raw = composite
        .chat_raw_stream(&req)
        .await
        .expect("raw lane succeeds")
        .expect("raw stream Some");
    let bytes: Vec<Bytes> = raw.map(|r| r.expect("ok")).collect().await;
    assert_eq!(bytes[0], Bytes::from(&b"mantle"[..]));

    let raw_ns = composite
        .chat_raw_nonstream(&req)
        .await
        .expect("raw nonstream Some")
        .expect("ok bytes");
    assert_eq!(raw_ns, Bytes::from(&b"mantle"[..]));

    assert!(mantle.chat_hit.load(Ordering::SeqCst));
    assert!(mantle.chat_stream_hit.load(Ordering::SeqCst));
    assert!(mantle.chat_raw_stream_hit.load(Ordering::SeqCst));
    assert!(mantle.chat_raw_nonstream_hit.load(Ordering::SeqCst));
    assert!(!converse.chat_hit.load(Ordering::SeqCst));
    assert!(!converse.chat_raw_stream_hit.load(Ordering::SeqCst));
    assert!(!responses.chat_hit.load(Ordering::SeqCst));
}

#[tokio::test]
async fn routes_converse_backend_to_converse_provider() {
    let (composite, converse, mantle, responses) = build_with_mantle_enabled(true);
    let req = normalized("anthropic.claude-sonnet-4-5");

    let resp = composite.chat(&req).await.expect("chat ok");
    assert_eq!(resp.model, "converse");
    let _ = composite
        .chat_raw_stream(&req)
        .await
        .expect("raw lane succeeds");

    assert!(converse.chat_hit.load(Ordering::SeqCst));
    assert!(converse.chat_raw_stream_hit.load(Ordering::SeqCst));
    assert!(!mantle.chat_hit.load(Ordering::SeqCst));
    assert!(!mantle.chat_raw_stream_hit.load(Ordering::SeqCst));
    assert!(!responses.chat_hit.load(Ordering::SeqCst));
}

#[tokio::test]
async fn routes_responses_backend_to_adapter_provider() {
    let (composite, converse, mantle, responses) = build_with_mantle_enabled(true);
    let req = normalized("gpt-responses");

    let response = composite.chat(&req).await.expect("chat ok");
    assert_eq!(response.model, "responses");
    let _ = composite.chat_stream(&req).await.expect("stream ok");

    assert!(responses.chat_hit.load(Ordering::SeqCst));
    assert!(responses.chat_stream_hit.load(Ordering::SeqCst));
    assert!(!converse.chat_hit.load(Ordering::SeqCst));
    assert!(!mantle.chat_hit.load(Ordering::SeqCst));
}

#[tokio::test]
async fn mantle_disabled_returns_bad_request() {
    let (composite, converse, mantle, responses) = build_with_mantle_enabled(false);
    let req = normalized("gpt-oss-120b");

    match composite.chat(&req).await {
        Err(AppError::BadRequest(msg)) => {
            assert!(msg.contains("requires a Bedrock API key"));
            assert!(msg.contains("disabled on this instance"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
    match composite.chat_stream(&req).await {
        Err(AppError::BadRequest(_)) => {}
        Err(other) => panic!("expected BadRequest, got {other:?}"),
        Ok(_) => panic!("expected BadRequest, got Ok(stream)"),
    }
    match composite.chat_raw_stream(&req).await {
        Err(AppError::BadRequest(_)) => {}
        Err(other) => panic!("expected BadRequest, got {other:?}"),
        Ok(_) => panic!("raw stream should fail when mantle is disabled"),
    }
    assert!(composite.chat_raw_nonstream(&req).await.is_none());

    assert!(!mantle.chat_hit.load(Ordering::SeqCst));
    assert!(!converse.chat_hit.load(Ordering::SeqCst));
    assert!(!responses.chat_hit.load(Ordering::SeqCst));
}

#[test]
fn resolve_mantle_chat_enabled_false_without_models() {
    let config = ModelCapabilityConfig {
        models: vec![ModelEntry {
            match_pattern: "anthropic.claude-sonnet-4-5".to_string(),
            capabilities: Vec::new(),
            params: ModelParams::default(),
        }],
        ..ModelCapabilityConfig::default()
    };
    let settings = settings_with(Some("bearer".to_string()), "us-east-2");
    assert!(!resolve_mantle_chat_enabled(&config, &settings));
}

#[test]
fn resolve_mantle_chat_enabled_false_without_bearer() {
    let config = mantle_chat_model_config(Some(vec!["us-east-2".to_string()]));
    let settings = settings_with(None, "us-east-2");
    assert!(!resolve_mantle_chat_enabled(&config, &settings));
}

#[test]
fn resolve_mantle_chat_enabled_false_when_disabled() {
    let config = mantle_chat_model_config(Some(vec!["us-east-2".to_string()]));
    let mut settings = settings_with(Some("bearer".to_string()), "us-east-2");
    settings.disable_mantle = true;
    assert!(!resolve_mantle_chat_enabled(&config, &settings));
}

#[test]
fn resolve_mantle_chat_enabled_true_with_bearer() {
    let config = mantle_chat_model_config(Some(vec!["us-east-2".to_string()]));
    let settings = settings_with(Some("bearer".to_string()), "us-east-2");
    assert!(resolve_mantle_chat_enabled(&config, &settings));
}

#[test]
fn resolve_mantle_chat_enabled_warns_but_passes_on_region_mismatch() {
    let config = mantle_chat_model_config(Some(vec!["us-east-1".to_string()]));
    let settings = settings_with(Some("bearer".to_string()), "us-west-2");
    assert!(resolve_mantle_chat_enabled(&config, &settings));
}
