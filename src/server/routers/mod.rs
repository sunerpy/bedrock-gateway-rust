//! HTTP route handlers and the router builder.
//!
//! This module wires the five OpenAI-compatible endpoints onto axum, dispatching
//! through the domain trait objects ([`ChatProvider`]/[`EmbeddingProvider`]) and
//! the [`ModelCatalog`] held in [`AppState`]. It contains NO provider logic and
//! NO AWS SDK types — every handler is a thin adapter over the abstractions.
//!
//! ## Endpoints (mounted under the configurable `api_route_prefix`)
//!
//! - `POST {prefix}/chat/completions` — [`chat_completions`]. Deserializes a
//!   [`ChatRequest`] (with a JSON-rejection → OpenAI 400 envelope), resolves the
//!   foundation model via [`ModelCapabilities::resolve_foundation`] into a
//!   [`NormalizedChatRequest`], then dispatches: `stream: true` returns an SSE
//!   [`Sse`] response built from the provider's [`ChatStream`] (each chunk →
//!   `data: <json>`, terminated by `data: [DONE]`); otherwise `Json(ChatResponse)`.
//! - `POST {prefix}/embeddings` — [`embeddings`]. Dispatches via
//!   [`EmbeddingProvider`] → `Json(EmbeddingsResponse)`.
//! - `GET {prefix}/models` — [`list_models`]. Serves the cached catalog.
//! - `GET {prefix}/models/{id}` — [`get_model`]. Single model or 400.
//! - `GET {prefix}/health` — [`health`]. Always `200 OK`, NO auth.
//!
//! ## Auth & layering
//!
//! Bearer auth ([`crate::server::auth::require_bearer`]) is applied with
//! `route_layer` over the protected subtree (chat, embeddings, models) so a
//! wrong HTTP method still yields `405`. `/health` lives outside that subtree
//! and therefore needs no auth — matching the legacy Python (everything but
//! health is protected).
//!
//! ## No TimeoutLayer on streaming
//!
//! The chat route (which may stream) carries NO timeout layer — a timeout would
//! sever an in-flight SSE connection. The router here adds no timeout at all;
//! the bootstrap layer (task 24) may add a timeout to the non-streaming routes
//! only if desired, but never to chat.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::middleware;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;

use crate::domain::{gen_request_id, NormalizedChatRequest, NormalizedResponsesRequest};
use crate::error::AppError;
use crate::openai::responses_schema::{ResponseStreamEvent, ResponsesRequest};
use crate::openai::schema::{ChatRequest, EmbeddingsRequest};
use crate::server::auth::require_bearer;
use crate::server::state::AppState;

/// Response header carrying the resolved foundation model id back to the client.
const OPENAI_MODEL_HEADER: &str = "openai-model";

/// Request header a client may use to supply its own correlation id.
const REQUEST_ID_HEADER: &str = "x-request-id";

/// Resolve the per-request correlation id: reuse the client's `x-request-id`
/// header when present and non-empty, otherwise self-generate one.
fn resolve_request_id(headers: &HeaderMap) -> std::sync::Arc<str> {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(std::sync::Arc::from)
        .unwrap_or_else(|| std::sync::Arc::from(gen_request_id()))
}

/// `POST {prefix}/chat/completions`.
///
/// Deserializes the body as a [`ChatRequest`] using axum's JSON extractor with a
/// rejection arm that renders the OpenAI 400 envelope for malformed input.
/// Resolves the model into a [`NormalizedChatRequest`] and dispatches stream vs
/// non-stream through the [`ChatProvider`].
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, AppError> {
    let received_at = std::time::Instant::now();
    let request_id = resolve_request_id(&headers);
    // Malformed JSON / wrong content-type → OpenAI 400 envelope (not axum's
    // default plain-text rejection).
    let Json(request) = payload.map_err(|rej| AppError::BadRequest(rej.body_text()))?;

    let resolved_model = state.caps.resolve_foundation(&request.model);
    let is_stream = request.stream.unwrap_or(false);
    let client_model = request.model.clone();
    tracing::info!(
        request_id = %request_id,
        model = %client_model,
        stream = is_stream,
        "chat request received"
    );
    let normalized = NormalizedChatRequest {
        request,
        resolved_model,
        request_id: request_id.clone(),
        received_at,
    };

    if is_stream {
        match state.chat.chat_stream(&normalized).await {
            Ok(chat_stream) => {
                tracing::info!(
                    request_id = %request_id,
                    model = %client_model,
                    ttfb_ms = received_at.elapsed().as_millis(),
                    "chat streaming started"
                );
                Ok(sse_response(chat_stream))
            }
            Err(e) => {
                tracing::warn!(request_id = %request_id, model = %client_model, error = %e, "chat request failed");
                Err(e)
            }
        }
    } else {
        match state.chat.chat(&normalized).await {
            Ok(response) => {
                let finish_reason = response
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.as_deref());
                // Cache-hit observability (logs only — Option-B, no wire change).
                // cached_tokens is the cache-READ count already computed by
                // compute_token_usage and surfaced under prompt_tokens_details.
                let cached_tokens = response
                    .usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0);
                let cache_hit = cached_tokens > 0;
                tracing::info!(
                    request_id = %request_id,
                    model = %client_model,
                    finish_reason = ?finish_reason,
                    prompt_tokens = response.usage.prompt_tokens,
                    completion_tokens = response.usage.completion_tokens,
                    total_tokens = response.usage.total_tokens,
                    cached_tokens,
                    cache_hit,
                    latency_ms = received_at.elapsed().as_millis(),
                    "chat completed"
                );
                Ok(Json(response).into_response())
            }
            Err(e) => {
                tracing::warn!(request_id = %request_id, model = %client_model, error = %e, "chat request failed");
                Err(e)
            }
        }
    }
}

/// Build an SSE response from a provider [`ChatStream`].
///
/// Each [`crate::openai::schema::ChatStreamResponse`] chunk is serialized as the
/// `data:` payload of an SSE event; a terminal `data: [DONE]` event is appended
/// after the provider stream is exhausted (base.py:60-63). A provider error is
/// rendered as a final `data:` event carrying the OpenAI error envelope so the
/// client sees a structured error rather than a silently truncated stream.
fn sse_response(chat_stream: crate::domain::ChatStream) -> Response {
    let event_stream = chat_stream
        .map(|item| -> Result<Event, Infallible> {
            match item {
                Ok(chunk) => {
                    let data = serde_json::to_string(&chunk).unwrap_or_else(|e| {
                        // Serialization of our own owned types should never fail;
                        // if it somehow does, surface a structured error payload.
                        error_envelope_json(&AppError::Internal(format!(
                            "failed to serialize stream chunk: {e}"
                        )))
                    });
                    Ok(Event::default().data(data))
                }
                Err(err) => {
                    // Render the OpenAI error envelope inline as an SSE data event.
                    let envelope = error_envelope_json(&err);
                    Ok(Event::default().data(envelope))
                }
            }
        })
        // Append the terminal [DONE] sentinel once the stream ends.
        .chain(futures::stream::once(async {
            Ok(Event::default().data("[DONE]"))
        }));

    Sse::new(event_stream).into_response()
}

/// Serialize an [`AppError`] into the OpenAI error-envelope JSON string used as
/// an inline SSE `data:` payload for **chat** streaming errors.
///
/// Reuses [`AppError::envelope`] so chat streaming errors carry the identical
/// full envelope (`{error:{message,type,param?,code}}`) that the non-streaming
/// [`IntoResponse`] path renders — the two chat error wire shapes are therefore
/// consistent.
///
/// This is the CHAT-completions streaming error shape only. The Responses API
/// streaming path does NOT use it: mid-stream Responses failures emit a typed
/// `response.failed` lifecycle event (see
/// [`crate::bedrock::responses_stream::ResponsesStreamState::fail`]), whose
/// `error.code` comes from the shared [`crate::error::responses_error`] mapper.
fn error_envelope_json(err: &AppError) -> String {
    serde_json::to_string(&err.envelope()).unwrap_or_else(|_| "{\"error\":{}}".to_string())
}

/// `POST {prefix}/responses`.
///
/// Deserializes the body as a [`ResponsesRequest`] (JSON rejection → OpenAI 400
/// envelope), resolves the foundation model into a [`NormalizedResponsesRequest`],
/// then dispatches through the [`ResponsesProvider`]: `stream: true` returns an
/// SSE response built from the provider's [`ResponsesStream`] (each event →
/// `event: <type>\ndata: <json>`, with NO `[DONE]` terminator); otherwise
/// `Json(ResponsesResponse)`. Both branches set the `openai-model` header.
pub async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, AppError> {
    let received_at = std::time::Instant::now();
    let request_id = resolve_request_id(&headers);
    let Json(request) = payload.map_err(|rej| AppError::BadRequest(rej.body_text()))?;

    let resolved_model = state.caps.resolve_foundation(&request.model);
    let is_stream = request.stream.unwrap_or(false);
    let client_model = request.model.clone();
    tracing::info!(
        request_id = %request_id,
        model = %client_model,
        stream = is_stream,
        "responses request received"
    );
    let normalized = NormalizedResponsesRequest {
        request,
        resolved_model: resolved_model.clone(),
        request_id: request_id.clone(),
        received_at,
    };

    if is_stream {
        match state.responses.respond_stream(&normalized).await {
            Ok(stream) => {
                tracing::info!(
                    request_id = %request_id,
                    model = %client_model,
                    ttfb_ms = received_at.elapsed().as_millis(),
                    "responses streaming started"
                );
                Ok(with_model_header(
                    responses_sse_response(stream),
                    &resolved_model,
                ))
            }
            Err(e) => {
                tracing::warn!(request_id = %request_id, model = %client_model, error = %e, "responses request failed");
                Err(e)
            }
        }
    } else {
        match state.responses.respond(&normalized).await {
            Ok(response) => {
                let cached_tokens = response
                    .usage
                    .input_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0);
                let cache_hit = cached_tokens > 0;
                tracing::info!(
                    request_id = %request_id,
                    model = %client_model,
                    status = %response.status,
                    input_tokens = response.usage.input_tokens,
                    output_tokens = response.usage.output_tokens,
                    total_tokens = response.usage.total_tokens,
                    cached_tokens,
                    cache_hit,
                    latency_ms = received_at.elapsed().as_millis(),
                    "responses completed"
                );
                Ok(with_model_header(
                    Json(response).into_response(),
                    &resolved_model,
                ))
            }
            Err(e) => {
                tracing::warn!(request_id = %request_id, model = %client_model, error = %e, "responses request failed");
                Err(e)
            }
        }
    }
}

/// Build an SSE response from a provider [`ResponsesStream`].
///
/// Each [`ResponseStreamEvent`] is emitted as an SSE frame whose event name is
/// the event's serde `type` tag (e.g. `response.created`) and whose `data:` is
/// the serialized event JSON. Unlike the chat path there is NO `[DONE]`
/// terminator — the Responses lifecycle is closed by a typed
/// `response.completed`/`response.failed` event.
fn responses_sse_response(stream: crate::domain::ResponsesStream) -> Response {
    let event_stream = stream.map(|item| -> Result<Event, Infallible> {
        match item {
            Ok(event) => Ok(responses_event_frame(&event)),
            Err(err) => {
                // A pre-stream provider error reaching here has no typed
                // lifecycle event; surface the OpenAI error envelope inline.
                Ok(Event::default().data(error_envelope_json(&err)))
            }
        }
    });
    Sse::new(event_stream).into_response()
}

/// Convert one [`ResponseStreamEvent`] into an SSE [`Event`] with the correct
/// `event:` name (the serde `type` tag) and JSON `data:` payload.
fn responses_event_frame(event: &ResponseStreamEvent) -> Event {
    match serde_json::to_value(event) {
        Ok(value) => {
            let event_type = value
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let data = serde_json::to_string(&value).unwrap_or_else(|e| {
                error_envelope_json(&AppError::Internal(format!(
                    "failed to serialize responses event: {e}"
                )))
            });
            Event::default().event(event_type).data(data)
        }
        Err(e) => Event::default().data(error_envelope_json(&AppError::Internal(format!(
            "failed to serialize responses event: {e}"
        )))),
    }
}

/// Attach the `openai-model` response header carrying the resolved model id.
fn with_model_header(mut response: Response, resolved_model: &str) -> Response {
    if let Ok(value) = axum::http::HeaderValue::from_str(resolved_model) {
        response.headers_mut().insert(OPENAI_MODEL_HEADER, value);
    }
    response
}

/// `POST {prefix}/embeddings`.
///
/// Dispatches through the [`EmbeddingProvider`] → `Json(EmbeddingsResponse)`.
pub async fn embeddings(
    State(state): State<AppState>,
    payload: Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> Result<Response, AppError> {
    let Json(request) = payload.map_err(|rej| AppError::BadRequest(rej.body_text()))?;
    let input_count = embedding_input_count(&request.input);
    tracing::info!(
        model = %request.model,
        input_count,
        "embeddings request received"
    );
    let started = std::time::Instant::now();
    match state.embeddings.embed(&request).await {
        Ok(response) => {
            tracing::info!(
                model = %request.model,
                prompt_tokens = response.usage.prompt_tokens,
                latency_ms = started.elapsed().as_millis(),
                "embeddings completed"
            );
            Ok(Json(response).into_response())
        }
        Err(e) => {
            tracing::warn!(model = %request.model, error = %e, "embeddings request failed");
            Err(e)
        }
    }
}

/// Count the number of input items in an embeddings request **without reading
/// any content** — used only as a log metric (count is metadata, not text).
fn embedding_input_count(input: &crate::openai::schema::EmbeddingInput) -> usize {
    use crate::openai::schema::EmbeddingInput;
    match input {
        EmbeddingInput::String(_) => 1,
        EmbeddingInput::StringArray(v) => v.len(),
        EmbeddingInput::IntArray(_) => 1,
        EmbeddingInput::IntMatrix(v) => v.len(),
    }
}

/// `GET {prefix}/models`.
///
/// Serves the currently-cached catalog as the OpenAI `Models` list. Per the
/// task contract we serve the cached catalog (the bootstrap layer performs the
/// initial refresh and may schedule periodic refreshes); a live refresh on every
/// `/models` call would add a control-plane round-trip to a hot path.
pub async fn list_models(State(state): State<AppState>) -> Response {
    let catalog = state.catalog.read().await;
    let list = catalog.list();
    tracing::info!(count = list.data.len(), "models list served");
    Json(list).into_response()
}

/// `GET {prefix}/models/{id}`.
///
/// Returns the single model in OpenAI `Model` shape, or a 400 when the id is not
/// in the catalog.
pub async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, AppError> {
    let catalog = state.catalog.read().await;
    match catalog.get(&id) {
        Some(model) => Ok(Json(model).into_response()),
        None => Err(AppError::BadRequest(format!("model `{id}` not found"))),
    }
}

/// `GET {prefix}/health` — liveness probe, no auth, always `200 OK`.
pub async fn health() -> Response {
    (axum::http::StatusCode::OK, "OK").into_response()
}

/// Build the application [`Router`] mounting all five endpoints under `prefix`.
///
/// The protected subtree (chat, embeddings, models) carries the bearer
/// middleware via `route_layer`; `/health` is mounted separately and is public.
/// The whole router is parameterized by [`AppState`].
///
/// `prefix` is the configured `api_route_prefix` (e.g. `/api/v1`). Routes are
/// mounted at `{prefix}/chat/completions`, `{prefix}/embeddings`,
/// `{prefix}/models`, `{prefix}/models/{id}`, `{prefix}/health`.
pub fn build_router(state: AppState, prefix: &str) -> Router {
    let api_key = state.api_key.clone();

    // Protected routes: bearer auth applied with route_layer so a wrong method
    // still yields 405 (not 401).
    let protected = Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/responses", post(responses))
        .route("/embeddings", post(embeddings))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model))
        .route_layer(middleware::from_fn_with_state(
            api_key,
            require_bearer_with_key,
        ));

    // Health is public (no auth).
    let public = Router::new().route("/health", get(health));

    let prefix = normalize_prefix(prefix);

    Router::new()
        .nest(&prefix, protected.merge(public))
        .with_state(state)
}

/// Adapter so the bearer middleware's `State<Arc<String>>` is satisfied from the
/// `Arc<String>` we pass into `from_fn_with_state`.
async fn require_bearer_with_key(
    state: State<Arc<String>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<Response, AppError> {
    require_bearer(state, request, next).await
}

/// Normalize a configured prefix into an axum nest path: ensure a single leading
/// `/`, strip any trailing `/`. An empty/`"/"` prefix nests at the root.
fn normalize_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim();
    if trimmed.is_empty() || trimmed == "/" {
        return "/".to_string();
    }
    let with_lead = if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    };
    with_lead.trim_end_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bedrock::models::ModelCatalog;
    use crate::config::{AppSettings, ModelCapabilityConfig};
    use crate::domain::{
        ChatProvider, ChatStream, EmbeddingProvider, ModelCapabilities, NormalizedChatRequest,
        NormalizedResponsesRequest, ResponsesProvider, ResponsesStream,
    };
    use crate::openai::responses_schema::ResponsesResponse;
    use crate::openai::schema::{
        ChatResponse, ChatResponseMessage, ChatStreamResponse, Choice, ChoiceDelta,
        EmbeddingsResponse, EmbeddingsUsage, Usage,
    };
    use axum::body::{to_bytes, Body};
    use axum::http::{header::AUTHORIZATION, Request, StatusCode};
    use serde_json::Value;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // oneshot

    const PREFIX: &str = "/api/v1";
    const KEY: &str = "test-key";

    // ---- Mock providers (the domain traits are mockable) -------------------

    struct MockChat;

    #[async_trait::async_trait]
    impl ChatProvider for MockChat {
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

    struct MockEmbeddings;

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddings {
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

    struct MockResponses;

    #[async_trait::async_trait]
    impl ResponsesProvider for MockResponses {
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

    fn responses_fixture(model: &str) -> ResponsesResponse {
        use crate::openai::responses_schema::{
            OutputContentPart, ResponseOutputItem, ResponsesUsage,
        };
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
                    annotations: None,
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
            api_key: Some(KEY.to_string()),
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: None,
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
        }
    }

    fn caps() -> Arc<dyn ModelCapabilities> {
        let config = ModelCapabilityConfig::load("config/models.toml").expect("load models.toml");
        Arc::new(crate::bedrock::capabilities::ConfigModelCapabilities::new(
            config,
        ))
    }

    fn catalog() -> ModelCatalog {
        use crate::bedrock::models::{assemble_catalog, FoundationModelFacts};
        let s = settings();
        let fms = [FoundationModelFacts {
            model_id: "anthropic.claude-3-sonnet-v1:0".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            inference_types: vec!["ON_DEMAND".to_string()],
            response_streaming_supported: true,
            status: "ACTIVE".to_string(),
        }];
        assemble_catalog(&fms, &[], &s)
    }

    fn app() -> Router {
        let state = AppState::new(
            Arc::new(MockChat),
            Arc::new(MockResponses),
            Arc::new(MockEmbeddings),
            Arc::new(RwLock::new(catalog())),
            caps(),
            Arc::new(KEY.to_string()),
            Arc::new(settings()),
            Arc::new(crate::bedrock::cache_support::CacheSupportRegistry::new()),
        );
        build_router(state, PREFIX)
    }

    async fn send(
        router: Router,
        method: &str,
        uri: &str,
        auth: Option<&str>,
        body: Option<&str>,
    ) -> (StatusCode, Vec<u8>, Option<String>) {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(a) = auth {
            builder = builder.header(AUTHORIZATION, a);
        }
        if body.is_some() {
            builder = builder.header("content-type", "application/json");
        }
        let req = builder
            .body(
                body.map(|b| Body::from(b.to_string()))
                    .unwrap_or(Body::empty()),
            )
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

    fn auth() -> String {
        format!("Bearer {KEY}")
    }

    // ---- All five routes are registered ------------------------------------

    #[tokio::test]
    async fn chat_completions_non_stream_returns_json() {
        let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}]}"#;
        let (status, bytes, ct) = send(
            app(),
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
    async fn chat_completions_stream_returns_sse_with_done() {
        let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let (status, bytes, ct) = send(
            app(),
            "POST",
            "/api/v1/chat/completions",
            Some(&auth()),
            Some(body),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // SSE content-type.
        assert!(
            ct.as_deref().unwrap().contains("text/event-stream"),
            "expected text/event-stream, got {ct:?}"
        );
        let text = String::from_utf8(bytes).unwrap();
        // Contains streamed data chunks and the terminal [DONE] sentinel.
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("data: [DONE]"));
        assert!(text.contains("hello"));
    }

    #[tokio::test]
    async fn embeddings_route_returns_json() {
        let body = r#"{"model":"cohere.embed-english-v3","input":"hi"}"#;
        let (status, bytes, ct) = send(
            app(),
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
    }

    #[tokio::test]
    async fn models_list_route() {
        let (status, bytes, _ct) = send(app(), "GET", "/api/v1/models", Some(&auth()), None).await;
        assert_eq!(status, StatusCode::OK);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["object"], "list");
        let ids: Vec<&str> = value["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&"anthropic.claude-3-sonnet-v1:0"));
    }

    #[tokio::test]
    async fn model_get_route_found_and_not_found() {
        let (status, bytes, _ct) = send(
            app(),
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

        // Unknown model → 400 envelope.
        let (status, bytes, _ct) = send(
            app(),
            "GET",
            "/api/v1/models/nope.absent-v1:0",
            Some(&auth()),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }

    // ---- /health is public (no auth) ---------------------------------------

    #[tokio::test]
    async fn health_requires_no_auth() {
        let (status, bytes, _ct) = send(app(), "GET", "/api/v1/health", None, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(String::from_utf8(bytes).unwrap(), "OK");
    }

    // ---- 401 on bad / missing auth -----------------------------------------

    #[tokio::test]
    async fn protected_routes_reject_missing_auth() {
        for (method, uri, body) in [
            ("POST", "/api/v1/chat/completions", Some("{}")),
            ("POST", "/api/v1/embeddings", Some("{}")),
            ("GET", "/api/v1/models", None),
        ] {
            let (status, bytes, _ct) = send(app(), method, uri, None, body).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri} must be 401");
            let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            assert_eq!(value["error"]["code"], "unauthorized");
        }
    }

    #[tokio::test]
    async fn protected_route_rejects_wrong_token() {
        let body = r#"{"model":"x","messages":[]}"#;
        let (status, _bytes, _ct) = send(
            app(),
            "POST",
            "/api/v1/chat/completions",
            Some("Bearer wrong"),
            Some(body),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    // ---- malformed JSON → 400 OpenAI envelope ------------------------------

    #[tokio::test]
    async fn malformed_json_returns_400_envelope() {
        let (status, bytes, _ct) = send(
            app(),
            "POST",
            "/api/v1/chat/completions",
            Some(&auth()),
            Some("{not valid json"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "bad_request");
        // Must be the OpenAI envelope, not axum's plain-text rejection.
        assert!(value.get("error").is_some());
        assert!(value.get("detail").is_none());
    }

    #[tokio::test]
    async fn missing_required_field_returns_400_envelope() {
        // Valid JSON but missing `messages` → deserialization rejection → 400.
        let (status, bytes, _ct) = send(
            app(),
            "POST",
            "/api/v1/chat/completions",
            Some(&auth()),
            Some(r#"{"model":"x"}"#),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }

    // ---- /responses route --------------------------------------------------

    #[tokio::test]
    async fn responses_non_stream_returns_json() {
        let body = r#"{"model":"anthropic.claude-3-sonnet-v1:0","input":"hi"}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/responses")
            .header(AUTHORIZATION, auth())
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap()
            .to_string();
        assert!(ct.contains("application/json"), "got {ct}");
        let model_header = resp
            .headers()
            .get(OPENAI_MODEL_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        assert_eq!(
            model_header.as_deref(),
            Some("anthropic.claude-3-sonnet-v1:0")
        );
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["object"], "response");
        assert_eq!(value["usage"]["input_tokens"], 1);
    }

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
        let resp = app().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap()
            .to_string();
        assert!(ct.contains("text/event-stream"), "got {ct}");
        let model_header = resp
            .headers()
            .get(OPENAI_MODEL_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        assert_eq!(
            model_header.as_deref(),
            Some("anthropic.claude-3-sonnet-v1:0")
        );
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("event: response.created"),
            "missing created event:\n{text}"
        );
        assert!(
            text.contains("event: response.completed"),
            "missing completed event:\n{text}"
        );
        assert!(
            !text.contains("[DONE]"),
            "Responses SSE must not emit [DONE]"
        );
    }

    #[tokio::test]
    async fn responses_requires_auth() {
        let body = r#"{"model":"x","input":"hi"}"#;
        let (status, bytes, _ct) = send(app(), "POST", "/api/v1/responses", None, Some(body)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        assert_eq!(value["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn responses_malformed_json_returns_400_envelope() {
        let (status, bytes, _ct) = send(
            app(),
            "POST",
            "/api/v1/responses",
            Some(&auth()),
            Some("{not valid json"),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "bad_request");
        assert!(value.get("detail").is_none());
    }

    // ---- prefix normalization ----------------------------------------------

    #[test]
    fn normalize_prefix_variants() {
        assert_eq!(normalize_prefix("/api/v1"), "/api/v1");
        assert_eq!(normalize_prefix("api/v1"), "/api/v1");
        assert_eq!(normalize_prefix("/api/v1/"), "/api/v1");
        assert_eq!(normalize_prefix(""), "/");
        assert_eq!(normalize_prefix("/"), "/");
    }

    #[tokio::test]
    async fn wrong_method_yields_405_not_401() {
        // route_layer means a wrong method on a protected path is 405, not 401.
        let (status, _bytes, _ct) = send(
            app(),
            "GET",
            "/api/v1/chat/completions",
            Some(&auth()),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    }
}
