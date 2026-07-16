//! Unit tests for [`crate::server::state`], kept in a sibling file (see the
//! `test-coverage-codecov` spec). The source module declares this via a
//! `#[path]` mod tests, so the top-level `use super::*;` resolves to the
//! implementation module.
//!
//! `AppState` is a pure dependency-injection container: `AppState::new` stores
//! each collaborator verbatim and the whole struct is `Arc`-backed so cloning is
//! cheap and shares state. These tests assemble the state from stub trait objects
//! (no AWS, no network) and assert the wiring: every injected collaborator is
//! reachable and dispatch lands on the exact provider that was injected.

use super::*;
use crate::domain::{
    BudgetRatios, Capability, ChatStream, NormalizedChatRequest, NormalizedResponsesRequest,
    ReasoningPath, ResponsesBackend, ResponsesStream,
};
use crate::error::AppError;
use crate::openai::responses_schema::{
    OutputContentPart, ResponseOutputItem, ResponsesResponse, ResponsesUsage,
};
use crate::openai::schema::{
    ChatResponse, ChatResponseMessage, Choice, EmbeddingsRequest, EmbeddingsResponse,
    EmbeddingsUsage, Usage,
};

// ---- Stub trait objects (the domain traits are mockable) -------------------

/// A chat provider that echoes the resolved model back so a dispatch test can
/// confirm the request reached THIS provider.
struct StubChat;

#[async_trait::async_trait]
impl ChatProvider for StubChat {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        Ok(ChatResponse {
            id: "chatcmpl-stub".to_string(),
            created: 0,
            model: req.resolved_model.clone(),
            system_fingerprint: "fp".to_string(),
            choices: vec![Choice {
                index: 0,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                message: ChatResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some("chat-stub".to_string()),
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
        })
    }

    async fn chat_stream(&self, _req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        Ok(Box::pin(futures::stream::iter(Vec::new())))
    }
}

/// A responses provider that stamps `"responses-stub"` into the output model so
/// the dispatch test can identify it.
struct StubResponses;

#[async_trait::async_trait]
impl ResponsesProvider for StubResponses {
    async fn respond(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Ok(ResponsesResponse {
            id: "resp-stub".to_string(),
            object: "response".to_string(),
            created_at: 0,
            status: "completed".to_string(),
            output: vec![ResponseOutputItem::Message {
                id: "msg".to_string(),
                status: "completed".to_string(),
                role: "assistant".to_string(),
                content: vec![OutputContentPart::OutputText {
                    text: "responses-stub".to_string(),
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
            model: "responses-stub".to_string(),
            instructions: None,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            error: None,
            incomplete_details: None,
        })
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        Ok(Box::pin(futures::stream::iter(Vec::new())))
    }
}

/// An embeddings provider that echoes the requested model back.
struct StubEmbeddings;

#[async_trait::async_trait]
impl EmbeddingProvider for StubEmbeddings {
    async fn embed(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError> {
        Ok(EmbeddingsResponse {
            object: "list".to_string(),
            data: Vec::new(),
            model: req.model.clone(),
            usage: EmbeddingsUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        })
    }
}

/// A caps stub that maps any `gpt` model to Mantle and everything else to
/// Converse, so a wiring test can confirm the injected resolver is consulted.
struct StubCaps;

impl ModelCapabilities for StubCaps {
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

fn settings() -> AppSettings {
    AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-east-2".to_string(),
        default_model: "default-model".to_string(),
        default_embedding_model: "default-embed".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching: false,
        prompt_cache_ttl: "5m".to_string(),
        disable_mantle: false,
        api_key: Some("k".to_string()),
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        bind_addr: "127.0.0.1".to_string(),
        port: 0,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    }
}

fn build_state() -> AppState {
    AppState::new(
        Arc::new(StubChat),
        Arc::new(StubResponses),
        Arc::new(StubEmbeddings),
        Arc::new(RwLock::new(ModelCatalog::default())),
        Arc::new(StubCaps),
        Arc::new("resolved-api-key".to_string()),
        Arc::new(settings()),
        Arc::new(CacheSupportRegistry::new()),
    )
}

fn normalized_chat(model: &str) -> NormalizedChatRequest {
    use crate::openai::schema::ChatRequest;
    // ChatRequest does not derive Default; build a minimal valid one from JSON
    // (only `messages` + `model` are mandatory).
    let request: ChatRequest =
        serde_json::from_value(serde_json::json!({ "messages": [], "model": model }))
            .expect("valid ChatRequest");
    NormalizedChatRequest {
        request,
        resolved_model: model.to_string(),
        request_id: Arc::from("req-test"),
        received_at: std::time::Instant::now(),
        raw_body: bytes::Bytes::new(),
    }
}

/// `AppState::new` stores each collaborator verbatim: the resolved API key, the
/// settings, and the catalog are all reachable through the assembled state.
#[tokio::test]
async fn new_wires_scalar_fields() {
    let state = build_state();

    assert_eq!(state.api_key.as_str(), "resolved-api-key");
    assert_eq!(state.settings.api_route_prefix, "/api/v1");
    assert_eq!(state.settings.default_model, "default-model");

    // The catalog is behind an RwLock and starts empty (Default).
    let catalog = state.catalog.read().await;
    assert!(catalog.models().is_empty());
}

/// The injected capability resolver is the one consulted for backend routing.
#[test]
fn new_wires_capability_resolver() {
    let state = build_state();
    assert_eq!(
        state.caps.responses_backend("gpt-5.5"),
        ResponsesBackend::Mantle
    );
    assert_eq!(
        state.caps.responses_backend("anthropic.claude-sonnet-4-5"),
        ResponsesBackend::Converse
    );
}

/// Each provider field dispatches to the exact stub that was injected.
#[tokio::test]
async fn new_wires_providers_for_dispatch() {
    let state = build_state();

    let chat = state
        .chat
        .chat(&normalized_chat("m"))
        .await
        .expect("chat ok");
    assert_eq!(
        chat.choices[0].message.content.as_deref(),
        Some("chat-stub")
    );

    let responses = state
        .responses
        .respond(&normalized_responses("m"))
        .await
        .expect("respond ok");
    assert_eq!(responses.model, "responses-stub");

    let embed = state
        .embeddings
        .embed(&embeddings_request("embed-model"))
        .await
        .expect("embed ok");
    assert_eq!(embed.model, "embed-model");
}

/// Cloning `AppState` is cheap and shares the SAME underlying `Arc`s (axum
/// clones the state per request; a deep copy would defeat that).
#[test]
fn clone_shares_arc_backed_state() {
    let state = build_state();
    let cloned = state.clone();

    assert!(Arc::ptr_eq(&state.api_key, &cloned.api_key));
    assert!(Arc::ptr_eq(&state.settings, &cloned.settings));
    assert!(Arc::ptr_eq(&state.catalog, &cloned.catalog));
    assert!(Arc::ptr_eq(&state.cache_support, &cloned.cache_support));
}

/// The shared cache-support registry is the same instance both providers would
/// see (constructed once and `Arc`-shared into the state).
#[test]
fn cache_support_is_shared_instance() {
    let state = build_state();
    let cloned = state.clone();
    // Mutating through one handle is visible through the clone (same registry).
    state.cache_support.mark_unsupported("some.model");
    assert!(cloned.cache_support.is_unsupported("some.model"));
}

fn normalized_responses(model: &str) -> NormalizedResponsesRequest {
    use crate::openai::responses_schema::{ResponsesInput, ResponsesRequest};
    use std::collections::HashMap;
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
        received_at: std::time::Instant::now(),
        raw_body: bytes::Bytes::new(),
    }
}

fn embeddings_request(model: &str) -> EmbeddingsRequest {
    use crate::openai::schema::{EmbeddingInput, EncodingFormat};
    EmbeddingsRequest {
        model: model.to_string(),
        input: EmbeddingInput::String("hi".to_string()),
        encoding_format: EncodingFormat::default(),
        dimensions: None,
        user: None,
    }
}
