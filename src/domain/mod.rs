//! Core domain trait abstractions.
//!
//! This module is the abstraction boundary between the OpenAI protocol surface
//! (`crate::openai`) and the Bedrock provider implementation (`crate::bedrock`).
//! It is the mechanism that keeps the gateway hardcode-free and extensible: the
//! HTTP layer depends only on these traits and the shared domain types, never on
//! any concrete AWS SDK type. Concrete Bedrock implementations live behind these
//! traits and are wired in later tasks.
//!
//! These traits mirror the legacy Python ABCs in
//! `.legacy-python/src/api/models/base.py` (`BaseChatModel`,
//! `BaseEmbeddingsModel`), generalized into the smaller, composable
//! responsibilities the Rust gateway needs: chat, embeddings, model
//! capabilities, region routing, and embedding body codec.
//!
//! DE-HARDCODING CONTRACT: no model IDs, AWS SDK types, or model-specific logic
//! appear here. `Capability`, `BudgetRatios`, `RouteOverride`, and
//! `ReasoningPath` are re-exported from `crate::config` rather than redefined.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::AppError;
use crate::openai::responses_schema::{ResponseStreamEvent, ResponsesRequest, ResponsesResponse};
use crate::openai::schema::{
    ChatRequest, ChatResponse, ChatStreamResponse, EmbeddingsRequest, EmbeddingsResponse,
};

pub use crate::config::{BudgetRatios, Capability, ReasoningPath, RouteOverride};

/// Which upstream backend serves a model's Responses-API requests.
///
/// Config-driven (never inferred from the model name): a model entry in
/// `config/models.toml` selects `Mantle` by declaring
/// `params.responses_backend = "mantle"`; every other value — and the absence
/// of the field — maps to `Converse`. `Converse` is the conceptual default so
/// every existing model keeps routing through the Bedrock Converse path
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesBackend {
    /// The default Bedrock Converse / ConverseStream path.
    Converse,
    /// The Bedrock Mantle (OpenAI-compatible) upstream path.
    Mantle,
}

/// Generate a gateway request id of the form `req-{nanos:x}-{counter:x}`.
///
/// Dependency-free (no `uuid`): a Unix-nanos timestamp combined with a
/// process-wide monotonic counter so two requests landing in the same
/// nanosecond still get distinct ids.
#[must_use]
pub fn gen_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("req-{nanos:x}-{seq:x}")
}

/// A provider-agnostic, normalized chat request.
///
/// This is the minimal intermediate the protocol layer produces before handing
/// work to a [`ChatProvider`]: the original OpenAI-shaped [`ChatRequest`] plus
/// the resolved foundation model id (after profile/alias resolution). It is
/// deliberately provider-agnostic — task 15's Bedrock translation consumes a
/// `NormalizedChatRequest` and turns it into a Bedrock Converse payload, but
/// nothing here references Bedrock or the AWS SDK.
#[derive(Debug, Clone)]
pub struct NormalizedChatRequest {
    /// The original OpenAI-shaped request.
    pub request: ChatRequest,
    /// The resolved foundation model id (post profile/alias resolution).
    pub resolved_model: String,
    /// Gateway-assigned request id (client `x-request-id` or self-generated).
    /// Streaming providers stamp it onto the terminal completion log as an
    /// explicit event field (the JSON formatter drops span fields).
    pub request_id: Arc<str>,
    /// Handler-entry instant, used by streaming providers to compute the
    /// end-to-end `duration_ms` at stream completion.
    pub received_at: Instant,
}

/// A boxed, `'static` stream of streaming chat chunks.
///
/// `BoxStream` keeps [`ChatProvider`] object-safe while allowing each
/// implementation to return its own concrete stream type.
pub type ChatStream = BoxStream<'static, Result<ChatStreamResponse, AppError>>;

/// A chat-completion provider.
///
/// Mirrors `BaseChatModel.chat` / `chat_stream`
/// (`.legacy-python/src/api/models/base.py:35-43`). Object-safe so the server
/// can hold a `Box<dyn ChatProvider>` chosen at runtime.
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Handle a non-streaming chat completion request.
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError>;

    /// Handle a streaming chat completion request, yielding SSE-ready chunks.
    async fn chat_stream(&self, req: &NormalizedChatRequest) -> Result<ChatStream, AppError>;
}

/// A provider-agnostic, normalized Responses-API request.
///
/// The Responses-API analogue of [`NormalizedChatRequest`]: the original
/// OpenAI-shaped [`ResponsesRequest`] plus the resolved foundation model id
/// (after profile/alias resolution). Provider-agnostic by design — it references
/// neither Bedrock nor the AWS SDK; concrete providers consume it and translate
/// to their backend payload.
#[derive(Debug, Clone)]
pub struct NormalizedResponsesRequest {
    /// The original OpenAI-shaped Responses request.
    pub request: ResponsesRequest,
    /// The resolved foundation model id (post profile/alias resolution).
    pub resolved_model: String,
    /// Gateway-assigned request id (client `x-request-id` or self-generated).
    /// Streaming providers stamp it onto the terminal completion log as an
    /// explicit event field (the JSON formatter drops span fields).
    pub request_id: Arc<str>,
    /// Handler-entry instant, used by streaming providers to compute the
    /// end-to-end `duration_ms` at stream completion.
    pub received_at: Instant,
    /// The verbatim request body bytes as received on the wire.
    ///
    /// Captured before deserialization so a raw-passthrough provider (the Mantle
    /// OpenAI-compatible upstream) can forward the client's exact JSON body
    /// unchanged. The Converse path ignores this field. NEVER logged.
    pub raw_body: bytes::Bytes,
}

/// A boxed, `'static` stream of Responses-API streaming events.
///
/// The Responses-API analogue of [`ChatStream`]: `BoxStream` keeps
/// [`ResponsesProvider`] object-safe while letting each implementation return
/// its own concrete stream type.
pub type ResponsesStream = BoxStream<'static, Result<ResponseStreamEvent, AppError>>;

/// A boxed, `'static` stream of raw SSE body bytes for the Responses surface.
///
/// The passthrough analogue of [`ResponsesStream`]: instead of typed lifecycle
/// events, each item is a verbatim chunk of the upstream `text/event-stream`
/// body, forwarded to the client unparsed. Used by a raw-passthrough provider
/// (Mantle) whose upstream already speaks the OpenAI Responses SSE wire format,
/// where round-tripping through the typed [`ResponseStreamEvent`] would be lossy
/// (its `Other` catch-all flattens unknown fields into a `HashMap`).
pub type RawResponsesStream = BoxStream<'static, Result<bytes::Bytes, AppError>>;

/// A Responses-API provider.
///
/// Deliberately a SEPARATE surface from [`ChatProvider`] (not an overload):
/// the Responses API has its own request/response envelope and a typed
/// streaming-event lifecycle. Object-safe so the server can hold an
/// `Arc<dyn ResponsesProvider>` chosen at runtime.
#[async_trait]
pub trait ResponsesProvider: Send + Sync {
    /// Handle a non-streaming Responses-API request.
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError>;

    /// Handle a streaming Responses-API request, yielding lifecycle events.
    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError>;

    /// Offer a raw-bytes SSE passthrough lane for this request.
    ///
    /// Returns `Some` only for a provider whose upstream already emits the
    /// OpenAI Responses SSE wire format verbatim (Mantle): the handler then
    /// forwards those bytes unparsed. The default returns `None`, so every
    /// Converse-based provider keeps using the typed [`Self::respond_stream`]
    /// path with no change.
    async fn respond_raw_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Option<RawResponsesStream> {
        let _ = req;
        None
    }
}

/// An embeddings provider.
///
/// Mirrors `BaseEmbeddingsModel.embed`
/// (`.legacy-python/src/api/models/base.py:72-75`).
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Handle an embeddings request.
    async fn embed(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError>;
}

/// Read-only access to externalized model capability knowledge.
///
/// Implementations resolve queries against the loaded
/// [`crate::config::ModelCapabilityConfig`]; no model-specific logic lives in
/// this trait definition.
pub trait ModelCapabilities: Send + Sync {
    /// Does the resolved model declare the given capability?
    fn has(&self, model: &str, cap: Capability) -> bool;

    /// Resolve an incoming model id or inference profile to its underlying
    /// foundation model id.
    fn resolve_foundation(&self, model_or_profile: &str) -> String;

    /// Budget-token ratios for reasoning effort levels, if configured.
    fn budget_ratios(&self, model: &str) -> Option<BudgetRatios>;

    /// Minimum thinking `budget_tokens` floor for the `budget_tokens` reasoning
    /// path, if configured. `None` means the model declares no explicit floor
    /// and the caller applies its own protocol default.
    fn min_budget_tokens(&self, model: &str) -> Option<u32>;

    /// Maximum cacheable tokens for prompt caching, if configured.
    fn max_cache_tokens(&self, model: &str) -> Option<u32>;

    /// Minimum tokens required to enable prompt caching, if configured.
    fn cache_min_tokens(&self, model: &str) -> Option<u32>;

    /// Maximum number of prompt-cache checkpoints this model supports, if
    /// configured. `None` means the model declares no explicit ceiling.
    fn max_cache_checkpoints(&self, model: &str) -> Option<u32>;

    /// Beta headers to attach for this model.
    fn beta_headers(&self, model: &str) -> Vec<String>;

    /// The reasoning/extended-thinking strategy for this model.
    fn reasoning_path(&self, model: &str) -> ReasoningPath;

    /// Which upstream backend serves this model's Responses-API requests.
    /// Defaults to [`ResponsesBackend::Converse`] for every model that does not
    /// declare `params.responses_backend = "mantle"` in config.
    fn responses_backend(&self, model: &str) -> ResponsesBackend;

    /// The region allow-list for this model, if configured. `None` means no
    /// region gate (the model is served everywhere); a non-empty list is the
    /// set of regions in which the model is available.
    fn model_regions(&self, model: &str) -> Option<Vec<String>>;
}

/// Per-model region routing.
///
/// Implementations resolve against the loaded
/// [`crate::config::RegionRoutingConfig`]; `None` means "use the home region
/// and original model id unchanged".
pub trait RegionRouter: Send + Sync {
    /// Resolve a region/model override for the given incoming model id.
    fn route(&self, model: &str) -> Option<RouteOverride>;
}

/// Encode/decode of provider-specific embedding request/response bodies.
///
/// Keeps the wire encoding of an embeddings call (per embedding family) behind a
/// trait so the protocol layer stays free of provider serialization details.
pub trait EmbeddingBodyCodec: Send + Sync {
    /// Encode an OpenAI-shaped embeddings request into a provider request body.
    fn encode(&self, req: &EmbeddingsRequest) -> Result<serde_json::Value, AppError>;

    /// Decode a provider response body into raw embedding vectors.
    fn decode(&self, body: &[u8]) -> Result<Vec<Vec<f32>>, AppError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::schema::{
        ChatResponse, ChatResponseMessage, Choice, ContentInput, EmbeddingInput, EncodingFormat,
        Message, Usage,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Canned [`ChatResponse`] for the mock — proves the abstraction works
    /// without any Bedrock/AWS dependency.
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
        }
    }

    struct MockChatProvider;

    #[async_trait]
    impl ChatProvider for MockChatProvider {
        async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
            Ok(canned_response(&req.resolved_model))
        }

        async fn chat_stream(&self, _req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
            let chunk = ChatStreamResponse {
                id: "chatcmpl-test".to_string(),
                created: 0,
                model: "mock".to_string(),
                system_fingerprint: "fp".to_string(),
                choices: Vec::new(),
                object: "chat.completion.chunk".to_string(),
                usage: None,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
        }
    }

    fn normalized() -> NormalizedChatRequest {
        NormalizedChatRequest {
            request: ChatRequest {
                messages: vec![Message::User {
                    name: None,
                    content: ContentInput::Text("hi".to_string()),
                }],
                model: "incoming-model".to_string(),
                frequency_penalty: None,
                presence_penalty: None,
                stream: None,
                stream_options: None,
                temperature: None,
                top_p: None,
                user: None,
                max_tokens: Some(16),
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
            resolved_model: "resolved-foundation-model".to_string(),
            request_id: Arc::from("req-test"),
            received_at: Instant::now(),
        }
    }

    /// MUST DO: drive a mock through `Box<dyn ChatProvider>` to prove the trait
    /// is object-safe and usable without Bedrock.
    #[tokio::test]
    async fn mock_chat_provider_via_dyn() {
        let provider: Box<dyn ChatProvider> = Box::new(MockChatProvider);
        let req = normalized();

        let resp = provider.chat(&req).await.expect("mock chat must succeed");
        assert_eq!(resp.model, "resolved-foundation-model");
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("mock reply")
        );

        let mut stream = provider
            .chat_stream(&req)
            .await
            .expect("mock chat_stream must succeed");
        use futures::StreamExt;
        let first = stream.next().await.expect("one chunk").expect("ok chunk");
        assert_eq!(first.object, "chat.completion.chunk");
    }

    struct MockEmbeddingProvider;

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError> {
            Ok(EmbeddingsResponse {
                object: "list".to_string(),
                data: Vec::new(),
                model: req.model.clone(),
                usage: crate::openai::schema::EmbeddingsUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            })
        }
    }

    #[tokio::test]
    async fn mock_embedding_provider_via_dyn() {
        let provider: Box<dyn EmbeddingProvider> = Box::new(MockEmbeddingProvider);
        let req = EmbeddingsRequest {
            input: EmbeddingInput::String("hi".to_string()),
            model: "embed-model".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
            user: None,
        };
        let resp = provider.embed(&req).await.expect("mock embed must succeed");
        assert_eq!(resp.model, "embed-model");
    }

    use crate::openai::responses_schema::{
        OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesInput,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    };

    fn canned_responses_response(model: &str) -> ResponsesResponse {
        ResponsesResponse {
            id: "resp-test".to_string(),
            object: "response".to_string(),
            created_at: 0,
            status: "completed".to_string(),
            output: vec![ResponseOutputItem::Message {
                id: "msg-test".to_string(),
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

    struct MockResponsesProvider;

    #[async_trait]
    impl ResponsesProvider for MockResponsesProvider {
        async fn respond(
            &self,
            req: &NormalizedResponsesRequest,
        ) -> Result<ResponsesResponse, AppError> {
            Ok(canned_responses_response(&req.resolved_model))
        }

        async fn respond_stream(
            &self,
            _req: &NormalizedResponsesRequest,
        ) -> Result<ResponsesStream, AppError> {
            let event = ResponseStreamEvent::OutputTextDelta {
                item_id: "msg-test".to_string(),
                output_index: 0,
                content_index: 0,
                delta: "mock".to_string(),
                sequence_number: 0,
            };
            Ok(Box::pin(futures::stream::iter(vec![Ok(event)])))
        }
    }

    fn normalized_responses() -> NormalizedResponsesRequest {
        NormalizedResponsesRequest {
            request: ResponsesRequest {
                model: "incoming-model".to_string(),
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
            resolved_model: "resolved-foundation-model".to_string(),
            request_id: Arc::from("req-test"),
            received_at: Instant::now(),
            raw_body: bytes::Bytes::new(),
        }
    }

    /// MUST DO: drive a mock through `Arc<dyn ResponsesProvider>` to prove the
    /// new Responses surface is object-safe and usable without Bedrock.
    #[tokio::test]
    async fn mock_responses_provider_via_dyn() {
        let provider: Arc<dyn ResponsesProvider> = Arc::new(MockResponsesProvider);
        let req = normalized_responses();

        let resp = provider
            .respond(&req)
            .await
            .expect("mock respond must succeed");
        assert_eq!(resp.model, "resolved-foundation-model");
        match &resp.output[0] {
            ResponseOutputItem::Message { content, .. } => match &content[0] {
                OutputContentPart::OutputText { text, .. } => assert_eq!(text, "mock reply"),
                other => panic!("expected output_text part, got {other:?}"),
            },
            other => panic!("expected message output item, got {other:?}"),
        }

        let mut stream = provider
            .respond_stream(&req)
            .await
            .expect("mock respond_stream must succeed");
        use futures::StreamExt;
        let first = stream.next().await.expect("one event").expect("ok event");
        assert!(matches!(first, ResponseStreamEvent::OutputTextDelta { .. }));
    }

    #[tokio::test]
    async fn default_respond_raw_stream_returns_none() {
        let provider: Arc<dyn ResponsesProvider> = Arc::new(MockResponsesProvider);
        let req = normalized_responses();
        assert!(provider.respond_raw_stream(&req).await.is_none());
    }
}
