//! wiremock-backed tests for the `bedrock-mantle` HTTP client.
//!
//! Relocated out of `mantle_client.rs` into this sibling file for code
//! organization (see the `test-coverage-codecov` spec, Option A). Behavior is
//! unchanged; functions are flat and share the implementation module via
//! `use super::*;`.
//!
//! Every test points the client at a local `wiremock` mock upstream and relies
//! on the mock's **immediate** responses — there is deliberately no `sleep`,
//! polling, or artificial delay. The suite is fully offline and needs no AWS
//! credentials.

use super::*;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mantle's base-URL template uses a literal `{region}` placeholder; a
/// wiremock server has no `{region}` segment, so for tests we point the
/// template directly at the mock's base URI (no placeholder to substitute).
fn client_for(base_uri: &str, bearer: &str) -> MantleClient {
    MantleClient::new(
        reqwest::Client::new(),
        base_uri.to_string(),
        base_uri.to_string(),
        bearer.to_string(),
    )
}

/// A well-formed SSE body with a terminating `response.completed` event. Note
/// there is intentionally NO `[DONE]` sentinel — the mantle lane forwards the
/// upstream bytes verbatim and never appends one.
const SSE_BODY: &str = "event: response.created\ndata: {\"type\":\"response.created\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";

/// A truncated SSE body: the stream cut off after `response.created` and before
/// any `response.completed` event (simulating a mid-stream break once 200 +
/// headers have already been committed).
const TRUNCATED_SSE_BODY: &str =
    "event: response.created\ndata: {\"type\":\"response.created\"}\n\nevent: response.output_text.delta\ndata: {\"delta\":\"par";

/// Given a 200 from the mantle `/responses` endpoint,
/// When `responses_nonstream` is called,
/// Then the returned bytes equal the mock body AND the request carried the
/// expected path and `Authorization: Bearer` header.
#[tokio::test]
async fn nonstream_returns_body_and_sends_bearer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-bearer"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let out = client
        .responses_nonstream("us-east-2", Bytes::from_static(b"{\"model\":\"gpt-5.5\"}"))
        .await
        .expect("nonstream should succeed on 200");

    assert_eq!(out, Bytes::from_static(SSE_BODY.as_bytes()));
}

/// Given a 200 SSE body,
/// When `responses_stream` is collected,
/// Then the concatenated chunk bytes equal the mock body verbatim, and NO
/// `[DONE]` sentinel is appended (raw passthrough contract).
#[tokio::test]
async fn stream_concatenates_to_full_body_without_done_sentinel() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(SSE_BODY),
        )
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let stream = client
        .responses_stream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect("stream should open on 200");

    let chunks: Vec<Bytes> = stream
        .map(|r| r.expect("each chunk should be Ok"))
        .collect()
        .await;
    let mut joined = Vec::new();
    for c in chunks {
        joined.extend_from_slice(&c);
    }
    // Byte-exact passthrough of the upstream body.
    assert_eq!(joined, SSE_BODY.as_bytes());
    // The gateway never synthesizes a `[DONE]` sentinel on the mantle lane.
    let joined_str = String::from_utf8(joined).expect("body is UTF-8");
    assert!(
        !joined_str.contains("[DONE]"),
        "mantle passthrough must not append a [DONE] sentinel"
    );
}

/// Given a 200 whose SSE body is truncated mid-stream (headers already sent, no
/// terminating `response.completed`),
/// When `responses_stream` is collected,
/// Then whatever bytes arrived are forwarded verbatim and NO error envelope is
/// synthesized — once 200 + headers commit, the client cannot wrap an error, so
/// the stream simply ends with the partial bytes.
#[tokio::test]
async fn stream_truncated_body_passes_through_without_envelope() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(TRUNCATED_SSE_BODY),
        )
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let stream = client
        .responses_stream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect("stream should open on 200 even if body is later truncated");

    let mut joined = Vec::new();
    let results: Vec<Result<Bytes, AppError>> = stream.collect().await;
    for r in results {
        // No chunk should be an error: the body simply ends early.
        let chunk = r.expect("truncated-but-clean body yields Ok chunks then ends");
        joined.extend_from_slice(&chunk);
    }

    // Exactly the partial bytes, nothing appended.
    assert_eq!(joined, TRUNCATED_SSE_BODY.as_bytes());
    let joined_str = String::from_utf8(joined).expect("body is UTF-8");
    // No OpenAI error envelope was injected after the 200 committed.
    assert!(
        !joined_str.contains("\"error\""),
        "no error envelope may be appended once the stream has started"
    );
    assert!(
        !joined_str.contains("[DONE]"),
        "no [DONE] sentinel is appended on truncation"
    );
}

/// Given a 429 from the upstream,
/// When `responses_nonstream` is called,
/// Then it maps to `AppError::Throttled` and the Display does NOT contain
/// the mocked response body text.
#[tokio::test]
async fn nonstream_429_maps_to_throttled_without_body_leak() {
    let server = MockServer::start().await;
    let secret_body = "SECRET-UPSTREAM-BODY-marker";
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(429).set_body_string(secret_body))
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let err = client
        .responses_nonstream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect_err("429 should be an error");

    assert!(matches!(err, AppError::Throttled(_)));
    assert!(
        !err.to_string().contains(secret_body),
        "error Display must not leak the upstream body"
    );
}

/// Given a 401 from the upstream,
/// When `responses_stream` is opened,
/// Then it maps to `AppError::Unauthorized` (unit variant) before any bytes
/// stream, and the Display does NOT contain the mocked body.
#[tokio::test]
async fn stream_401_maps_to_unauthorized_without_body_leak() {
    let server = MockServer::start().await;
    let secret_body = "SECRET-401-BODY-marker";
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(401).set_body_string(secret_body))
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let err = client
        .responses_stream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .err()
        .expect("401 should be a pre-stream error");

    assert!(matches!(err, AppError::Unauthorized));
    assert!(
        !err.to_string().contains(secret_body),
        "error Display must not leak the upstream body"
    );
}

/// The `{region}` placeholder is substituted and `/responses` appended.
#[test]
fn responses_url_substitutes_region_and_appends_path() {
    let client = MantleClient::new(
        reqwest::Client::new(),
        "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        "b".to_string(),
    );
    assert_eq!(
        client.responses_url("us-west-2"),
        "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"
    );
}

#[test]
fn responses_url_unchanged() {
    let client = MantleClient::new(
        reqwest::Client::new(),
        "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        "b".to_string(),
    );
    assert!(client
        .responses_url("us-west-2")
        .ends_with("/openai/v1/responses"));
}

#[test]
fn chat_url_substitutes_region_and_appends_path() {
    let client = MantleClient::new(
        reqwest::Client::new(),
        "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        "b".to_string(),
    );
    assert_eq!(
        client.chat_url("us-west-2"),
        "https://bedrock-mantle.us-west-2.api.aws/v1/chat/completions"
    );
    assert!(!client.chat_url("us-west-2").contains("/openai"));
}

#[tokio::test]
async fn chat_nonstream_returns_body_and_sends_bearer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-bearer"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string(r#"{"object":"chat.completion"}"#),
        )
        .expect(1)
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let out = client
        .chat_nonstream(
            "us-east-2",
            Bytes::from_static(b"{\"model\":\"gpt-oss-120b\"}"),
        )
        .await
        .expect("chat_nonstream should succeed on 200");
    assert_eq!(out, Bytes::from_static(br#"{"object":"chat.completion"}"#));
}

/// The mantle chat SSE body has NO `[DONE]` sentinel; the client concatenates
/// the upstream bytes verbatim and never appends one (the handler does).
#[tokio::test]
async fn chat_stream_concatenates_without_done_sentinel() {
    const CHAT_SSE: &str = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(CHAT_SSE),
        )
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let stream = client
        .chat_stream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect("stream should open on 200");
    let chunks: Vec<Bytes> = stream.map(|r| r.expect("ok chunk")).collect().await;
    let mut joined = Vec::new();
    for c in chunks {
        joined.extend_from_slice(&c);
    }
    assert_eq!(joined, CHAT_SSE.as_bytes());
    let joined_str = String::from_utf8(joined).expect("utf8");
    assert!(
        !joined_str.contains("[DONE]"),
        "client must not append a [DONE] sentinel"
    );
}

/// Non-standard `delta.reasoning` and per-chunk `obfuscation` fields survive
/// verbatim in the concatenated byte passthrough (locks the divergence).
#[tokio::test]
async fn chat_stream_preserves_reasoning_and_obfuscation() {
    const CHAT_SSE: &str = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"thinking...\"}}],\"obfuscation\":\"XyZ123\"}\n\n";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(CHAT_SSE),
        )
        .mount(&server)
        .await;

    let client = client_for(&server.uri(), "test-bearer");
    let stream = client
        .chat_stream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect("stream should open on 200");
    let chunks: Vec<Bytes> = stream.map(|r| r.expect("ok chunk")).collect().await;
    let mut joined = Vec::new();
    for c in chunks {
        joined.extend_from_slice(&c);
    }
    let joined_str = String::from_utf8(joined).expect("utf8");
    assert!(joined_str.contains("reasoning"));
    assert!(joined_str.contains("obfuscation"));
    assert!(joined_str.contains("XyZ123"));
}

#[tokio::test]
async fn chat_401_and_429_map_without_body_leak() {
    let secret = "SECRET-CHAT-BODY";
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string(secret))
        .mount(&server)
        .await;
    let client = client_for(&server.uri(), "b");
    let err = client
        .chat_nonstream("r", Bytes::from_static(b"{}"))
        .await
        .expect_err("401 is an error");
    assert!(matches!(err, AppError::Unauthorized));
    assert!(!err.to_string().contains(secret));

    let server2 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string(secret))
        .mount(&server2)
        .await;
    let client2 = client_for(&server2.uri(), "b");
    let err2 = client2
        .chat_stream("r", Bytes::from_static(b"{}"))
        .await
        .err()
        .expect("429 is a pre-stream error");
    assert!(matches!(err2, AppError::Throttled(_)));
    assert!(!err2.to_string().contains(secret));
}

/// A `{region}` substitution also drives the actual request path against a
/// wiremock upstream whose URI embeds a region-shaped segment, proving the
/// substituted URL is what gets called (end-to-end for `MANTLE_BASE_URL_TEMPLATE`).
#[tokio::test]
async fn region_substitution_routes_request_to_expected_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .expect(1)
        .mount(&server)
        .await;

    // Template embeds `{region}` in a query-free path suffix; the mock only
    // matches the appended `/openai/v1/responses`, so a successful call proves
    // the substituted URL was used.
    let template = format!("{}/openai/v1", server.uri());
    let chat_template = format!("{}/v1", server.uri());
    let client = MantleClient::new(
        reqwest::Client::new(),
        template,
        chat_template,
        "b".to_string(),
    );
    let out = client
        .responses_nonstream("us-east-2", Bytes::from_static(b"{}"))
        .await
        .expect("substituted URL should route to the mock");
    assert_eq!(out, Bytes::from_static(b"ok"));
}

/// Other 4xx (e.g. 400) map to `BadRequest`; 5xx map to `UpstreamBedrock`.
#[tokio::test]
async fn other_4xx_and_5xx_map_to_bad_request_and_upstream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    let client = client_for(&server.uri(), "b");
    let err = client
        .responses_nonstream("r", Bytes::from_static(b"{}"))
        .await
        .expect_err("400 is an error");
    assert!(matches!(err, AppError::BadRequest(_)));

    let server5 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server5)
        .await;
    let client5 = client_for(&server5.uri(), "b");
    let err5 = client5
        .responses_nonstream("r", Bytes::from_static(b"{}"))
        .await
        .expect_err("503 is an error");
    assert!(matches!(err5, AppError::UpstreamBedrock(_)));
}
