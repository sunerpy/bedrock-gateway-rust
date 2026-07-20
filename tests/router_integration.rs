//! Integration tests for `server/routers::build_router`.
//!
//! Unlike the inline `#[cfg(test)]` tests in `src/server/routers/mod.rs`, this
//! suite lives in `tests/` and therefore exercises ONLY the crate's public API
//! (`bedrock_gateway_rust::...`). It drives requests through the assembled axum
//! `Router` with `tower::ServiceExt::oneshot`, asserting on the wire-visible
//! behavior: endpoint assembly, the public `/health` probe, the bearer 401 vs
//! method 405 distinction, the OpenAI error envelope shape, and the handler
//! error-severity branches (5xx `UpstreamBedrock`/`Internal` vs 4xx
//! `Throttled`/`BadRequest`) which map onto distinct HTTP status codes.
//!
//! Everything here runs fully OFFLINE: no AWS credentials, no network, no
//! `sleep`/timers. The Bedrock-backed providers are replaced with in-crate mock
//! implementations of the public domain traits, so `AppState` is assembled
//! without any AWS call. `ModelCapabilities` is the real config-driven resolver
//! loaded from `config/models.toml` (relative to the package root, where cargo
//! runs integration tests) so the mantle-gate branch is exercised authentically.
//!
//! No `src/` visibility changes were required: `build_router`, `AppState::new`,
//! the domain traits, `ConfigModelCapabilities`, `assemble_catalog`,
//! `FoundationModelFacts`, `ModelCatalog`, and `CacheSupportRegistry` are all
//! already `pub`.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::{to_bytes, Body};
use axum::http::{header::AUTHORIZATION, Request, StatusCode};
use axum::Router;
use serde_json::Value;
use tokio::sync::RwLock;
use tower::ServiceExt; // oneshot

use bedrock_gateway_rust::bedrock::cache_support::CacheSupportRegistry;
use bedrock_gateway_rust::bedrock::capabilities::ConfigModelCapabilities;
use bedrock_gateway_rust::bedrock::models::{assemble_catalog, FoundationModelFacts, ModelCatalog};
use bedrock_gateway_rust::config::{AppSettings, ModelCapabilityConfig};
use bedrock_gateway_rust::domain::{
    ChatProvider, ChatStream, EmbeddingProvider, ModelCapabilities, NormalizedChatRequest,
    NormalizedResponsesRequest, RawResponsesStream, ResponsesProvider, ResponsesStream,
};
use bedrock_gateway_rust::error::AppError;
use bedrock_gateway_rust::openai::responses_schema::{
    InputTokensDetails, OutputContentPart, ResponseOutputItem, ResponseStreamEvent,
    ResponsesResponse, ResponsesUsage,
};
use bedrock_gateway_rust::openai::schema::{
    ChatResponse, ChatResponseMessage, ChatStreamResponse, Choice, ChoiceDelta, Embedding,
    EmbeddingData, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage, PromptTokensDetails,
    Usage,
};
use bedrock_gateway_rust::server::routers::build_router;
use bedrock_gateway_rust::server::state::AppState;

const PREFIX: &str = "/api/v1";
const KEY: &str = "integration-key";
const CONVERSE_MODEL: &str = "anthropic.claude-3-sonnet-v1:0";

// ---- Error severity selector -------------------------------------------------

/// Which HTTP severity class an erroring mock provider should produce, so the
/// handler's `e.is_server_error()` branch is exercised both ways.
#[derive(Clone, Copy)]
enum Severity {
    /// 5xx path: maps to `502 Bad Gateway`, logged at `error!`.
    Server,
    /// 4xx path: maps to `429 Too Many Requests`, logged at `warn!`.
    Client,
}

impl Severity {
    fn error(self) -> AppError {
        match self {
            Severity::Server => AppError::UpstreamBedrock("upstream fault".to_string()),
            Severity::Client => AppError::Throttled("slow down".to_string()),
        }
    }

    fn expected_status(self) -> StatusCode {
        match self {
            Severity::Server => StatusCode::BAD_GATEWAY,
            Severity::Client => StatusCode::TOO_MANY_REQUESTS,
        }
    }
}

// ---- Mock providers ----------------------------------------------------------

/// Chat provider that always succeeds with a fixed reply.
struct OkChat;

#[async_trait]
impl ChatProvider for OkChat {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        Ok(ChatResponse {
            id: "chatcmpl-mock".to_string(),
            created: 0,
            model: req.resolved_model.clone(),
            system_fingerprint: "fp".to_string(),
            choices: vec![Choice {
                index: 0,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                message: ChatResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some("mock reply".to_string()),
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
        // Two real chunks so the SSE happy-path framing (chunk serialization +
        // trailing `data: [DONE]`) is exercised, for both /chat/completions and
        // the /completions text-completion SSE surface (which reuses this path).
        let chunks = vec![
            Ok(ChatStreamResponse {
                id: "chatcmpl-mock".to_string(),
                created: 0,
                model: "mock".to_string(),
                system_fingerprint: "fp".to_string(),
                choices: vec![ChoiceDelta {
                    index: 0,
                    finish_reason: None,
                    logprobs: None,
                    delta: ChatResponseMessage {
                        content: Some("hello".to_string()),
                        ..Default::default()
                    },
                }],
                object: "chat.completion.chunk".to_string(),
                usage: None,
            }),
            Ok(ChatStreamResponse {
                id: "chatcmpl-mock".to_string(),
                created: 0,
                model: "mock".to_string(),
                system_fingerprint: "fp".to_string(),
                choices: vec![ChoiceDelta {
                    index: 0,
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                    delta: ChatResponseMessage::default(),
                }],
                object: "chat.completion.chunk".to_string(),
                usage: None,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

/// Chat provider whose non-stream reply carries cache-read details, and whose
/// stream yields a mid-stream error item (surfaced Ok at open time). Exercises
/// the `cache_hit` logging branch and the in-stream error → inline envelope arm.
struct CacheHitChat;

#[async_trait]
impl ChatProvider for CacheHitChat {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        Ok(ChatResponse {
            id: "chatcmpl-mock".to_string(),
            created: 0,
            model: req.resolved_model.clone(),
            system_fingerprint: "fp".to_string(),
            choices: vec![Choice {
                index: 0,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                message: ChatResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some("cached reply".to_string()),
                    tool_calls: None,
                    reasoning_content: None,
                },
            }],
            object: "chat.completion".to_string(),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: 7,
                    audio_tokens: 0,
                }),
                completion_tokens_details: None,
            },
        })
    }

    async fn chat_stream(&self, _req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        // Stream opens successfully (Ok), but the first item is an error — this
        // reaches the handler AFTER the 200/SSE headers, so it is rendered as an
        // inline OpenAI error-envelope `data:` event rather than an HTTP status.
        let items: Vec<Result<ChatStreamResponse, AppError>> = vec![Err(
            AppError::UpstreamBedrock("mid-stream fault".to_string()),
        )];
        Ok(Box::pin(futures::stream::iter(items)))
    }
}

/// Chat provider that always fails with the configured severity.
struct ErrChat(Severity);

#[async_trait]
impl ChatProvider for ErrChat {
    async fn chat(&self, _req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        Err(self.0.error())
    }

    async fn chat_stream(&self, _req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        Err(self.0.error())
    }
}

/// Embeddings provider that always succeeds with an empty vector list.
struct OkEmbeddings;

#[async_trait]
impl EmbeddingProvider for OkEmbeddings {
    async fn embed(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError> {
        Ok(EmbeddingsResponse {
            object: "list".to_string(),
            data: vec![Embedding {
                object: "embedding".to_string(),
                embedding: EmbeddingData::Float(vec![0.1, 0.2, 0.3]),
                index: 0,
            }],
            model: req.model.clone(),
            usage: EmbeddingsUsage {
                prompt_tokens: 4,
                total_tokens: 4,
            },
        })
    }
}

/// Embeddings provider that always fails with the configured severity.
struct ErrEmbeddings(Severity);

#[async_trait]
impl EmbeddingProvider for ErrEmbeddings {
    async fn embed(&self, _req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError> {
        Err(self.0.error())
    }
}

/// Responses provider that always succeeds with a fixed completed response.
struct OkResponses;

#[async_trait]
impl ResponsesProvider for OkResponses {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Ok(responses_fixture(&req.resolved_model))
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        // Real lifecycle events so the typed responses SSE framing is exercised:
        // each event → `event: <type>\ndata: <json>`, with NO `[DONE]` sentinel.
        let response = responses_fixture(&req.resolved_model);
        let events = vec![
            Ok(ResponseStreamEvent::Created {
                response: response.clone(),
                sequence_number: 0,
            }),
            Ok(ResponseStreamEvent::OutputTextDelta {
                item_id: "msg-mock".to_string(),
                output_index: 0,
                content_index: 0,
                delta: "mock".to_string(),
                sequence_number: 1,
            }),
            Ok(ResponseStreamEvent::Completed {
                response,
                sequence_number: 2,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// Responses provider whose non-stream reply carries cache-read details
/// (`input_tokens_details.cached_tokens > 0`) to exercise the `cache_hit`
/// logging branch on the /responses non-stream path.
struct CacheHitResponses;

#[async_trait]
impl ResponsesProvider for CacheHitResponses {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        let mut fixture = responses_fixture(&req.resolved_model);
        fixture.usage.input_tokens_details = Some(InputTokensDetails { cached_tokens: 5 });
        Ok(fixture)
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Raw sentinel bytes forwarded verbatim by the Mantle passthrough lane.
const RAW_SSE_BYTES: &[u8] =
    b"event: response.created\ndata: {\"x\":1}\n\nevent: response.completed\ndata: {\"y\":2}\n\n";

/// Responses provider offering a raw-bytes passthrough stream (Mantle lane): the
/// handler must forward the upstream bytes verbatim with the SSE anti-buffering
/// headers, injecting no `[DONE]` and synthesizing no `response.completed`.
struct RawResponses;

#[async_trait]
impl ResponsesProvider for RawResponses {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Ok(responses_fixture(&req.resolved_model))
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        Err(AppError::Internal(
            "typed path must not be used".to_string(),
        ))
    }

    async fn respond_raw_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<Option<RawResponsesStream>, AppError> {
        let chunk: Result<bytes::Bytes, AppError> = Ok(bytes::Bytes::from_static(RAW_SSE_BYTES));
        Ok(Some(Box::pin(futures::stream::iter(vec![chunk]))))
    }
}

/// Responses provider that always fails with the configured severity.
struct ErrResponses(Severity);

#[async_trait]
impl ResponsesProvider for ErrResponses {
    async fn respond(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        Err(self.0.error())
    }

    async fn respond_stream(
        &self,
        _req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        Err(self.0.error())
    }
}

fn responses_fixture(model: &str) -> ResponsesResponse {
    ResponsesResponse {
        id: "resp-mock".to_string(),
        object: "response".to_string(),
        created_at: 0,
        status: "completed".to_string(),
        output: vec![ResponseOutputItem::Message {
            id: "msg-mock".to_string(),
            status: "completed".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentPart::OutputText {
                text: "mock reply".to_string(),
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

// ---- AppState / Router assembly (public API only) ----------------------------

fn settings() -> AppSettings {
    AppSettings {
        api_route_prefix: PREFIX.to_string(),
        debug: false,
        aws_region: "us-west-2".to_string(),
        default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
        enable_cross_region_inference: true,
        enable_application_inference_profiles: true,
        enable_prompt_caching: false,
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
        api_key: Some(KEY.to_string()),
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        bind_addr: "0.0.0.0".to_string(),
        port: 8080,
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

fn caps() -> Arc<dyn ModelCapabilities> {
    // Real config-driven resolver: exercises the authentic mantle-gate branch
    // (gpt-5.x → responses_backend = "mantle") without model-name matching.
    let config = ModelCapabilityConfig::load("config/models.toml").expect("load models.toml");
    Arc::new(ConfigModelCapabilities::new(config))
}

fn catalog() -> ModelCatalog {
    let s = settings();
    let fms = [FoundationModelFacts {
        model_id: CONVERSE_MODEL.to_string(),
        input_modalities: vec!["TEXT".to_string()],
        inference_types: vec!["ON_DEMAND".to_string()],
        response_streaming_supported: true,
        status: "ACTIVE".to_string(),
    }];
    assemble_catalog(&fms, &[], &s)
}

/// Assemble a router from the supplied providers via the public `AppState::new`
/// + `build_router`. Only the three providers vary between tests.
fn app_with(
    chat: Arc<dyn ChatProvider>,
    responses: Arc<dyn ResponsesProvider>,
    embeddings: Arc<dyn EmbeddingProvider>,
) -> Router {
    let state = AppState::new(
        chat,
        responses,
        embeddings,
        Arc::new(RwLock::new(catalog())),
        caps(),
        Arc::new(KEY.to_string()),
        Arc::new(settings()),
        Arc::new(CacheSupportRegistry::new()),
    );
    build_router(state, PREFIX)
}

/// The all-success router used by the happy-path assembly tests.
fn ok_app() -> Router {
    app_with(
        Arc::new(OkChat),
        Arc::new(OkResponses),
        Arc::new(OkEmbeddings),
    )
}

fn auth() -> String {
    format!("Bearer {KEY}")
}

/// Drive one request through the router and return `(status, body_bytes,
/// content_type)`.
async fn send(
    router: Router,
    method: &str,
    uri: &str,
    bearer: Option<&str>,
    body: Option<&str>,
) -> (StatusCode, Vec<u8>, Option<String>) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(a) = bearer {
        builder = builder.header(AUTHORIZATION, a);
    }
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    let req = builder
        .body(body.map(|b| Body::from(b.to_string())).unwrap_or_default())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, bytes.to_vec(), content_type)
}

/// Assert a body is the OpenAI error envelope shape (`{"error":{...}}`) with the
/// given `code`, and that it is NOT a bare/`detail` rejection.
fn assert_error_envelope(bytes: &[u8], code: &str) {
    let value: Value = serde_json::from_slice(bytes).expect("error body must be JSON");
    assert!(
        value.get("error").is_some(),
        "must carry an `error` object, got: {value}"
    );
    assert!(
        value.get("detail").is_none(),
        "must NOT be axum's plain-text/`detail` rejection, got: {value}"
    );
    assert_eq!(
        value["error"]["code"], code,
        "unexpected error.code: {value}"
    );
}

// ---- Endpoint assembly (build_router wires every route) ----------------------

#[tokio::test]
async fn build_router_mounts_all_endpoints_under_prefix() {
    // A GET on each mounted path with valid auth resolves the route (not 404).
    // POST-only routes answer a GET with 405, which still proves the path is
    // mounted; GET routes answer 200. Either way, NEVER 404.
    let cases = [
        ("POST", "/api/v1/chat/completions"),
        ("POST", "/api/v1/completions"),
        ("POST", "/api/v1/responses"),
        ("POST", "/api/v1/embeddings"),
        ("GET", "/api/v1/models"),
        ("GET", "/api/v1/models/anything"),
        ("GET", "/api/v1/health"),
    ];
    for (method, uri) in cases {
        let (status, _b, _ct) = send(ok_app(), method, uri, Some(&auth()), None).await;
        assert_ne!(status, StatusCode::NOT_FOUND, "{method} {uri} not mounted");
    }
}

// ---- Health probe (public, no auth) ------------------------------------------

#[tokio::test]
async fn health_is_public_and_returns_200_ok() {
    let (status, bytes, _ct) = send(ok_app(), "GET", "/api/v1/health", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(String::from_utf8(bytes).unwrap(), "OK");
}

// ---- Happy paths through the assembled router --------------------------------

#[tokio::test]
async fn chat_completions_returns_json_envelope() {
    let body =
        r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}]}"#;
    let (status, bytes, ct) = send(
        ok_app(),
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.unwrap().contains("application/json"));
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "chat.completion");
    assert_eq!(value["choices"][0]["message"]["content"], "mock reply");
}

#[tokio::test]
async fn models_list_and_lookup_served_from_catalog() {
    let (status, bytes, _ct) = send(ok_app(), "GET", "/api/v1/models", Some(&auth()), None).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "list");

    let (status, bytes, _ct) = send(
        ok_app(),
        "GET",
        "/api/v1/models/anthropic.claude-3-sonnet-v1:0",
        Some(&auth()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["id"], "anthropic.claude-3-sonnet-v1:0");
    assert_eq!(value["object"], "model");
}

#[tokio::test]
async fn unknown_model_lookup_returns_400_envelope() {
    let (status, bytes, _ct) = send(
        ok_app(),
        "GET",
        "/api/v1/models/nope.absent-v1:0",
        Some(&auth()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error_envelope(&bytes, "bad_request");
}

// ---- Auth: 401 without a valid bearer; envelope shape ------------------------

#[tokio::test]
async fn protected_routes_without_bearer_return_401_envelope() {
    for (method, uri, body) in [
        ("POST", "/api/v1/chat/completions", Some("{}")),
        ("POST", "/api/v1/completions", Some("{}")),
        ("POST", "/api/v1/responses", Some("{}")),
        ("POST", "/api/v1/embeddings", Some("{}")),
        ("GET", "/api/v1/models", None),
    ] {
        let (status, bytes, _ct) = send(ok_app(), method, uri, None, body).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{method} {uri} must be 401"
        );
        assert_error_envelope(&bytes, "unauthorized");
    }
}

#[tokio::test]
async fn protected_route_with_wrong_bearer_returns_401() {
    let body = r#"{"model":"x","messages":[]}"#;
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/chat/completions",
        Some("Bearer wrong-token"),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_error_envelope(&bytes, "unauthorized");
}

// ---- 405 vs 401 distinction --------------------------------------------------

#[tokio::test]
async fn wrong_method_on_authenticated_route_is_405_not_401() {
    // With a VALID bearer, the auth middleware passes and the method router
    // rejects a wrong METHOD on a mounted path with 405 — proving the layer is
    // `route_layer` (per-route), not a blanket layer that would 401 first.
    let (status, _b, _ct) = send(
        ok_app(),
        "GET", // chat/completions is POST-only
        "/api/v1/chat/completions",
        Some(&auth()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn wrong_method_without_bearer_is_401_auth_precedes_method() {
    // Without a bearer, the `route_layer` auth middleware runs before method
    // routing, so a wrong-method request is rejected as 401 (auth) rather than
    // 405 — documenting the auth-before-method ordering.
    let (status, bytes, _ct) = send(ok_app(), "GET", "/api/v1/chat/completions", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_error_envelope(&bytes, "unauthorized");
}

// ---- Malformed JSON → 400 OpenAI envelope ------------------------------------

#[tokio::test]
async fn malformed_json_returns_400_envelope() {
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some("{not valid json"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error_envelope(&bytes, "bad_request");
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["error"]["type"], "invalid_request_error");
}

// ---- Mantle gate: capability-driven, not model-name matching -----------------

#[tokio::test]
async fn responses_backed_model_is_allowed_on_chat_completions() {
    // `gpt-5.5` declares `chat_backend = "responses"` and therefore reaches the
    // injected Chat provider instead of the old Responses-only route guard.
    let body = r#"{"model":"gpt-5.5","messages":[{"role":"user","content":"hi"}]}"#;
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "chat.completion");
}

// ---- Handler error-severity branches (5xx error! vs 4xx warn!) ---------------
//
// The handlers branch on `e.is_server_error()` to choose the log level. That
// branch is not directly observable, but each severity maps to a distinct HTTP
// status, so exercising both proves both arms run without panicking and the
// error envelope is emitted.

#[tokio::test]
async fn chat_non_stream_error_severity_branches_map_to_status() {
    let body =
        r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}]}"#;
    for sev in [Severity::Server, Severity::Client] {
        let app = app_with(
            Arc::new(ErrChat(sev)),
            Arc::new(OkResponses),
            Arc::new(OkEmbeddings),
        );
        let (status, bytes, _ct) = send(
            app,
            "POST",
            "/api/v1/chat/completions",
            Some(&auth()),
            Some(body),
        )
        .await;
        assert_eq!(status, sev.expected_status());
        // Body is a full OpenAI error envelope.
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("error").is_some());
    }
}

#[tokio::test]
async fn chat_stream_error_severity_branches_map_to_status() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    for sev in [Severity::Server, Severity::Client] {
        let app = app_with(
            Arc::new(ErrChat(sev)),
            Arc::new(OkResponses),
            Arc::new(OkEmbeddings),
        );
        let (status, _b, _ct) = send(
            app,
            "POST",
            "/api/v1/chat/completions",
            Some(&auth()),
            Some(body),
        )
        .await;
        // Stream-open failure surfaces before the 200/SSE headers, so it maps
        // to the plain error status.
        assert_eq!(status, sev.expected_status());
    }
}

#[tokio::test]
async fn responses_error_severity_branches_map_to_status() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi"}"#;
    for sev in [Severity::Server, Severity::Client] {
        let app = app_with(
            Arc::new(OkChat),
            Arc::new(ErrResponses(sev)),
            Arc::new(OkEmbeddings),
        );
        let (status, bytes, _ct) =
            send(app, "POST", "/api/v1/responses", Some(&auth()), Some(body)).await;
        assert_eq!(status, sev.expected_status());
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("error").is_some());
    }
}

#[tokio::test]
async fn embeddings_error_severity_branches_map_to_status() {
    let body = r#"{"model":"cohere.embed-english-v3","input":"hi"}"#;
    for sev in [Severity::Server, Severity::Client] {
        let app = app_with(
            Arc::new(OkChat),
            Arc::new(OkResponses),
            Arc::new(ErrEmbeddings(sev)),
        );
        let (status, bytes, _ct) =
            send(app, "POST", "/api/v1/embeddings", Some(&auth()), Some(body)).await;
        assert_eq!(status, sev.expected_status());
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("error").is_some());
    }
}

#[tokio::test]
async fn completions_error_severity_branches_map_to_status() {
    // Legacy text-completions reuses the chat provider under the hood.
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"hi"}"#;
    for sev in [Severity::Server, Severity::Client] {
        let app = app_with(
            Arc::new(ErrChat(sev)),
            Arc::new(OkResponses),
            Arc::new(OkEmbeddings),
        );
        let (status, _b, _ct) = send(
            app,
            "POST",
            "/api/v1/completions",
            Some(&auth()),
            Some(body),
        )
        .await;
        assert_eq!(status, sev.expected_status());
    }
}

// ---- Chat streaming happy path (SSE framing + [DONE]) ------------------------

#[tokio::test]
async fn chat_completions_stream_returns_sse_with_done() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let (status, bytes, ct) = send(
        ok_app(),
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ct.as_deref()
            .unwrap_or_default()
            .contains("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
    let text = String::from_utf8(bytes).unwrap();
    // Streamed chunk frames plus the terminal [DONE] sentinel (chat surface).
    assert!(
        text.contains("chat.completion.chunk"),
        "missing chunk frames:\n{text}"
    );
    assert!(
        text.contains("hello"),
        "missing streamed delta content:\n{text}"
    );
    assert!(
        text.contains("data: [DONE]"),
        "chat SSE must terminate with [DONE]:\n{text}"
    );
}

/// A mid-stream provider error (stream opened Ok, then yields an `Err`) is
/// surfaced as an inline OpenAI error-envelope `data:` event — the connection
/// still carries a `200` + SSE headers, and the terminal `[DONE]` is appended.
#[tokio::test]
async fn chat_stream_mid_stream_error_renders_inline_envelope() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let app = app_with(
        Arc::new(CacheHitChat),
        Arc::new(OkResponses),
        Arc::new(OkEmbeddings),
    );
    let (status, bytes, ct) = send(
        app,
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "SSE opens with 200 before the error item"
    );
    assert!(ct
        .as_deref()
        .unwrap_or_default()
        .contains("text/event-stream"));
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("\"error\""),
        "mid-stream error must render an envelope:\n{text}"
    );
    assert!(text.contains("data: [DONE]"));
}

// ---- Chat non-stream cache-hit logging branch --------------------------------

#[tokio::test]
async fn chat_non_stream_cache_hit_branch_returns_json() {
    let body =
        r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}]}"#;
    let app = app_with(
        Arc::new(CacheHitChat),
        Arc::new(OkResponses),
        Arc::new(OkEmbeddings),
    );
    let (status, bytes, _ct) = send(
        app,
        "POST",
        "/api/v1/chat/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    // cached_tokens surfaces under prompt_tokens_details (cache-read only).
    assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
    assert_eq!(value["choices"][0]["message"]["content"], "cached reply");
}

// ---- Completions: non-stream success (200 text_completion shape) -------------

#[tokio::test]
async fn completions_non_stream_returns_text_completion() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"once upon"}"#;
    let (status, bytes, ct) = send(
        ok_app(),
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.unwrap().contains("application/json"));
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "text_completion");
    assert!(
        value["id"].as_str().unwrap().starts_with("cmpl-"),
        "id must carry the cmpl- prefix, got {}",
        value["id"]
    );
    assert_eq!(value["choices"][0]["text"], "mock reply");
    assert_eq!(value["choices"][0]["index"], 0);
    assert_eq!(value["usage"]["total_tokens"], 3);
}

/// `echo: true` prepends the prompt to the completion text.
#[tokio::test]
async fn completions_non_stream_echo_prefixes_prompt() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"PROMPT ","echo":true}"#;
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["choices"][0]["text"], "PROMPT mock reply");
}

// ---- Completions: streaming success (text_completion SSE chunks + [DONE]) -----

#[tokio::test]
async fn completions_stream_returns_sse_with_done() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"hi","stream":true}"#;
    let (status, bytes, ct) = send(
        ok_app(),
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ct.as_deref()
            .unwrap_or_default()
            .contains("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("text_completion"),
        "missing text_completion chunk shape:\n{text}"
    );
    assert!(
        text.contains("data: [DONE]"),
        "completions SSE must terminate with [DONE]:\n{text}"
    );
}

/// A mid-stream provider error on the completions SSE surface renders the inline
/// error envelope, then still appends the terminal `[DONE]`.
#[tokio::test]
async fn completions_stream_mid_stream_error_renders_inline_envelope() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"hi","stream":true}"#;
    let app = app_with(
        Arc::new(CacheHitChat),
        Arc::new(OkResponses),
        Arc::new(OkEmbeddings),
    );
    let (status, bytes, _ct) = send(
        app,
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).unwrap();
    assert!(
        text.contains("\"error\""),
        "expected inline error envelope:\n{text}"
    );
    assert!(text.contains("data: [DONE]"));
}

// ---- Completions: 400 branches (suffix, token-array prompt) ------------------

#[tokio::test]
async fn completions_suffix_returns_400_unsupported() {
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":"hi","suffix":"tail"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error_envelope(&bytes, "unsupported");
}

#[tokio::test]
async fn completions_token_array_prompt_returns_400() {
    // A token-array prompt cannot be rendered to a single string → 400.
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/completions",
        Some(&auth()),
        Some(r#"{"model":"anthropic.claude-3-sonnet-v1:0","prompt":[1,2,3]}"#),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_error_envelope(&bytes, "bad_request");
}

// ---- Embeddings: happy path returns data vectors -----------------------------

#[tokio::test]
async fn embeddings_returns_data_vectors() {
    let body = r#"{"model":"cohere.embed-english-v3","input":"hi"}"#;
    let (status, bytes, ct) = send(
        ok_app(),
        "POST",
        "/api/v1/embeddings",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.unwrap().contains("application/json"));
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "list");
    assert_eq!(value["model"], "cohere.embed-english-v3");
    let data = value["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 3);
    assert_eq!(value["usage"]["prompt_tokens"], 4);
}

/// The `embedding_input_count` log-metric branch handles a string-array input
/// (count = array length) without changing the wire response.
#[tokio::test]
async fn embeddings_string_array_input_counts_items() {
    let body = r#"{"model":"cohere.embed-english-v3","input":["a","b","c"]}"#;
    let (status, bytes, _ct) = send(
        ok_app(),
        "POST",
        "/api/v1/embeddings",
        Some(&auth()),
        Some(body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "list");
}

// ---- Responses: streaming lifecycle (typed events, NO [DONE]) ----------------

#[tokio::test]
async fn responses_stream_returns_sse_lifecycle_no_done() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi","stream":true}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/responses")
        .header(AUTHORIZATION, auth())
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = ok_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(ct.contains("text/event-stream"), "got {ct}");
    let model_header = resp
        .headers()
        .get("openai-model")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    assert_eq!(model_header.as_deref(), Some(CONVERSE_MODEL));
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        text.contains("event: response.created"),
        "missing created:\n{text}"
    );
    assert!(
        text.contains("event: response.completed"),
        "missing completed:\n{text}"
    );
    assert!(
        !text.contains("[DONE]"),
        "Responses SSE must NOT emit [DONE]:\n{text}"
    );
}

// ---- Responses: non-stream success + model header + cache-hit branch ---------

#[tokio::test]
async fn responses_non_stream_returns_json_with_model_header() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi"}"#;
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/responses")
        .header(AUTHORIZATION, auth())
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = ok_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let model_header = resp
        .headers()
        .get("openai-model")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    assert_eq!(model_header.as_deref(), Some(CONVERSE_MODEL));
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["object"], "response");
    assert_eq!(value["usage"]["input_tokens"], 1);
}

#[tokio::test]
async fn responses_non_stream_cache_hit_branch() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi"}"#;
    let app = app_with(
        Arc::new(OkChat),
        Arc::new(CacheHitResponses),
        Arc::new(OkEmbeddings),
    );
    let (status, bytes, _ct) =
        send(app, "POST", "/api/v1/responses", Some(&auth()), Some(body)).await;
    assert_eq!(status, StatusCode::OK);
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 5);
}

// ---- Responses: raw-bytes passthrough lane (Mantle) --------------------------

#[tokio::test]
async fn responses_raw_passthrough_forwards_bytes_verbatim() {
    let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi","stream":true}"#;
    let app = app_with(
        Arc::new(OkChat),
        Arc::new(RawResponses),
        Arc::new(OkEmbeddings),
    );
    let req = Request::builder()
        .method("POST")
        .uri("/api/v1/responses")
        .header(AUTHORIZATION, auth())
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(ct.contains("text/event-stream"), "got {ct}");
    let accel = resp
        .headers()
        .get("x-accel-buffering")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert_eq!(
        accel, "no",
        "raw passthrough must carry X-Accel-Buffering: no"
    );
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        bytes.as_ref(),
        RAW_SSE_BYTES,
        "bytes must pass through verbatim"
    );
    assert!(
        !bytes.windows(6).any(|w| w == b"[DONE]"),
        "raw passthrough must not inject [DONE]"
    );
}
