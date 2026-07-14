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
    OutputContentPart, ResponseOutputItem, ResponseStreamEvent, ResponsesInput, ResponsesRequest,
    ResponsesResponse, ResponsesUsage,
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

/// `gen_request_id` produces the documented `req-…` prefix and the
/// process-wide monotonic counter keeps successive ids distinct.
#[test]
fn gen_request_id_has_prefix_and_is_unique() {
    let a = gen_request_id();
    let b = gen_request_id();
    assert!(a.starts_with("req-"), "got {a}");
    assert!(b.starts_with("req-"), "got {b}");
    assert_ne!(a, b, "monotonic counter must yield distinct ids");
}

/// `ResponsesBackend` is a plain `Copy`/`Eq` enum with two distinct
/// variants; `Converse` is the conceptual default routing.
#[test]
fn responses_backend_equality_and_copy() {
    let a = ResponsesBackend::Mantle;
    let b = a; // exercises `Copy`
    assert_eq!(a, b);
    assert_ne!(ResponsesBackend::Converse, ResponsesBackend::Mantle);
    // `Debug` is derived and must render the variant name.
    assert_eq!(format!("{:?}", ResponsesBackend::Converse), "Converse");
}

/// Direct field assertions on a freshly-constructed
/// [`NormalizedChatRequest`] (construction + field wiring).
#[test]
fn normalized_chat_request_fields() {
    let req = normalized();
    assert_eq!(req.resolved_model, "resolved-foundation-model");
    assert_eq!(&*req.request_id, "req-test");
    assert_eq!(req.request.model, "incoming-model");
    // `received_at` is a valid monotonic instant.
    let _elapsed = req.received_at.elapsed();
    // `Clone` is derived and preserves the resolved model.
    let cloned = req.clone();
    assert_eq!(cloned.resolved_model, req.resolved_model);
}

/// Direct field assertions on a freshly-constructed
/// [`NormalizedResponsesRequest`], including the raw-body passthrough seam.
#[test]
fn normalized_responses_request_fields() {
    let req = normalized_responses();
    assert_eq!(req.resolved_model, "resolved-foundation-model");
    assert_eq!(&*req.request_id, "req-test");
    assert_eq!(req.request.model, "incoming-model");
    assert!(req.raw_body.is_empty(), "default raw_body is empty");
    let _elapsed = req.received_at.elapsed();
}
