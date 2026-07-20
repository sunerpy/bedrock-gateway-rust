//! wiremock-backed tests for [`crate::bedrock::mantle_chat_provider`].
//!
//! Mirrors `mantle_provider_tests.rs`: a `StubCaps` gates regions and resolves
//! `gpt-oss-120b → openai.gpt-oss-120b`, and every test points the client at a
//! local wiremock upstream. Fully offline; no AWS credentials.

use super::*;
use crate::domain::{ChatBackend, ModelCapabilities, ResponsesBackend};
use crate::openai::schema::{ChatRequest, ContentInput, Message};
use std::collections::HashMap;
use std::time::Instant;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct StubCaps {
    regions: Option<Vec<String>>,
}

impl ModelCapabilities for StubCaps {
    fn has(&self, _model: &str, _cap: crate::domain::Capability) -> bool {
        false
    }
    fn resolve_foundation(&self, model_or_profile: &str) -> String {
        if model_or_profile == "gpt-oss-120b" {
            "openai.gpt-oss-120b".to_string()
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
        ResponsesBackend::Converse
    }
    fn chat_backend(&self, _model: &str) -> ChatBackend {
        ChatBackend::Mantle
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
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
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
        responses_stream_idle_timeout_secs: 180,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    })
}

fn provider_for(base_uri: &str, region: &str, regions: Option<Vec<String>>) -> MantleChatProvider {
    let client = MantleClient::new(
        reqwest::Client::new(),
        base_uri.to_string(),
        base_uri.to_string(),
        "test-bearer".to_string(),
    );
    let caps: Arc<dyn ModelCapabilities> = Arc::new(StubCaps { regions });
    MantleChatProvider::new(client, caps, settings_in_region(region))
}

fn normalized(raw: &str) -> NormalizedChatRequest {
    NormalizedChatRequest {
        request: ChatRequest {
            messages: vec![Message::User {
                name: None,
                content: ContentInput::Text("hi".to_string()),
            }],
            model: "gpt-oss-120b".to_string(),
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
        resolved_model: "openai.gpt-oss-120b".to_string(),
        request_id: Arc::from("req-test"),
        received_at: Instant::now(),
        raw_body: Bytes::copy_from_slice(raw.as_bytes()),
    }
}

const CHAT_COMPLETION: &str = r#"{"object":"chat.completion","model":"openai.gpt-oss-120b","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}"#;

const CHAT_SSE: &str = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"reasoning\":\"thinking\"},\"index\":0}],\"obfuscation\":\"XyZ\"}\n\ndata: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n";

#[tokio::test]
async fn region_gate_rejects_out_of_region() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(CHAT_COMPLETION))
        .expect(0)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-west-2",
        Some(vec!["us-east-1".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[]}"#);

    // The raw lane preserves the pre-stream region-gate error directly.
    match provider.chat_raw_stream(&req).await {
        Err(AppError::BadRequest(_)) => {}
        Err(other) => panic!("expected BadRequest, got {other:?}"),
        Ok(_) => panic!("expected BadRequest, got Ok(raw lane)"),
    }
    // The non-stream raw lane returns Some(Err(BadRequest)).
    match provider.chat_raw_nonstream(&req).await {
        Some(Err(AppError::BadRequest(_))) => {}
        other => panic!("expected Some(Err(BadRequest)), got {other:?}"),
    }
}

#[tokio::test]
async fn rewrite_model_patches_only_model_key() {
    let provider = provider_for("http://mantle.invalid", "us-east-2", None);
    let raw = Bytes::from_static(br#"{"model":"gpt-oss-120b","messages":[],"keep":"me","n":2}"#);
    let out = provider
        .rewrite_model("gpt-oss-120b", &raw)
        .expect("rewrite ok");
    let value: Value = serde_json::from_slice(&out).expect("parse");
    let obj = value.as_object().expect("object");
    assert_eq!(
        obj.get("model"),
        Some(&Value::String("openai.gpt-oss-120b".to_string()))
    );
    assert_eq!(obj.get("keep"), Some(&Value::String("me".to_string())));
    assert_eq!(obj.get("n"), Some(&Value::Number(2.into())));
    assert_eq!(obj.len(), 4);
}

#[tokio::test]
async fn chat_raw_stream_forwards_bytes_verbatim() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(
            serde_json::json!({ "model": "openai.gpt-oss-120b" }),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(CHAT_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[],"stream":true}"#);
    let stream = provider
        .chat_raw_stream(&req)
        .await
        .expect("raw lane succeeds")
        .expect("raw stream Some");
    let chunks: Vec<Bytes> = stream.map(|r| r.expect("ok chunk")).collect().await;
    let mut joined = Vec::new();
    for c in chunks {
        joined.extend_from_slice(&c);
    }
    assert_eq!(joined, CHAT_SSE.as_bytes());
    let joined_str = String::from_utf8(joined).expect("utf8");
    assert!(!joined_str.contains("[DONE]"));
    assert!(joined_str.contains("reasoning"));
    assert!(joined_str.contains("obfuscation"));
}

#[tokio::test]
async fn chat_raw_stream_preserves_upstream_open_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[],"stream":true}"#);

    assert!(matches!(
        provider.chat_raw_stream(&req).await,
        Err(AppError::UpstreamBedrock(_))
    ));
}

#[tokio::test]
async fn chat_raw_nonstream_returns_upstream_bytes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_partial_json(serde_json::json!({
            "model": "openai.gpt-oss-120b",
            "keep": "me"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(CHAT_COMPLETION),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[],"keep":"me"}"#);
    let result = provider
        .chat_raw_nonstream(&req)
        .await
        .expect("Some for a mantle chat request");
    let bytes = result.expect("Ok bytes");
    // Verbatim passthrough — no re-serialization, no usage recomputation.
    assert_eq!(bytes, Bytes::from_static(CHAT_COMPLETION.as_bytes()));
}

#[tokio::test]
async fn chat_stream_typed_fallback_does_not_reconnect() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(CHAT_SSE),
        )
        .expect(0)
        .mount(&server)
        .await;

    let provider = provider_for(
        &server.uri(),
        "us-east-2",
        Some(vec!["us-east-2".to_string()]),
    );
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[],"stream":true}"#);
    match provider.chat_stream(&req).await {
        Err(AppError::Internal(msg)) => {
            assert!(msg.contains("raw passthrough lane"));
        }
        Err(other) => panic!("expected Internal, got {other:?}"),
        Ok(_) => panic!("expected Internal, got Ok(stream)"),
    }
}

#[tokio::test]
async fn chat_typed_fallback_errors() {
    let provider = provider_for("http://mantle.invalid", "us-east-2", None);
    let req = normalized(r#"{"model":"gpt-oss-120b","messages":[]}"#);
    // Never deserializes bytes; never contacts the upstream (pre-flight passes,
    // then the typed method signals the routing error).
    match provider.chat(&req).await {
        Err(AppError::Internal(msg)) => {
            assert!(msg.contains("raw passthrough lane"));
        }
        Err(other) => panic!("expected Internal, got {other:?}"),
        Ok(_) => panic!("expected Internal, got Ok(response)"),
    }
}
