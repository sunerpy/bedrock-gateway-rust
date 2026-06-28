//! HTTP client for the AWS Bedrock `bedrock-mantle` OpenAI-compatible surface.
//!
//! Unlike the Converse-based [`crate::bedrock::client`] (which speaks the AWS
//! SDK), `bedrock-mantle` exposes an OpenAI-shaped `/openai/v1/responses`
//! endpoint that we call over plain HTTP with a Bearer token. This module is a
//! **thin, byte-oriented** client: it forwards an already-serialized request
//! body and returns the upstream bytes verbatim — either the full response
//! ([`MantleClient::responses_nonstream`]) or a raw SSE byte stream
//! ([`MantleClient::responses_stream`]).
//!
//! It deliberately does **not** parse the SSE into typed events — the Responses
//! stream events round-trip lossily through our typed model (see the
//! `bedrock-mantle-gpt` notepad), so the mantle lane is raw passthrough.
//! Translation into [`crate::domain::ResponsesProvider`] happens in a later
//! layer (T6); this is a plain HTTP client only.
//!
//! ## Error mapping (pre-stream HTTP status)
//!
//! Before any bytes are streamed, a non-2xx response is classified into an
//! [`AppError`]:
//!
//! - `401` → [`AppError::Unauthorized`]
//! - `429` → [`AppError::Throttled`]
//! - other `4xx` → [`AppError::BadRequest`]
//! - `5xx` → [`AppError::UpstreamBedrock`]
//!
//! The raw upstream response body is **never** placed into the error message or
//! logged — only the status code and structured metadata.

use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use reqwest::StatusCode;

use crate::error::AppError;

/// A thin HTTP client for the `bedrock-mantle` OpenAI-compatible Responses
/// surface.
///
/// Cheap to clone — the inner [`reqwest::Client`] is reference-counted and the
/// other fields are owned strings. Construct one per gateway instance and share
/// it across requests.
#[derive(Clone)]
pub struct MantleClient {
    /// Shared reqwest client (connection pooling + rustls).
    http: reqwest::Client,
    /// Base URL with a literal `{region}` placeholder, e.g.
    /// `https://bedrock-mantle.{region}.api.aws/openai/v1`. The `{region}`
    /// token is substituted per request and `/responses` is appended.
    base_url_template: String,
    /// Gateway→Bedrock bearer token, sent as `Authorization: Bearer {bearer}`.
    bearer: String,
}

impl MantleClient {
    /// Construct a new [`MantleClient`].
    ///
    /// `base_url_template` is the `AppSettings::mantle_base_url_template` value
    /// (a string containing the literal `{region}` placeholder). `bearer` is
    /// the gateway→Bedrock bearer (`AppSettings::bedrock_api_key`).
    #[must_use]
    pub fn new(http: reqwest::Client, base_url_template: String, bearer: String) -> Self {
        Self {
            http,
            base_url_template,
            bearer,
        }
    }

    /// Build the full `/responses` URL for `region` by substituting the literal
    /// `{region}` placeholder in the template and appending `/responses`.
    fn responses_url(&self, region: &str) -> String {
        format!(
            "{}/responses",
            self.base_url_template.replace("{region}", region)
        )
    }

    /// POST `body` to the mantle `/responses` endpoint for `region` and return
    /// the full response bytes (non-streaming).
    ///
    /// The request `body` is forwarded verbatim with `content-type:
    /// application/json` and a Bearer auth header. A pre-read non-2xx status is
    /// mapped to an [`AppError`] (see the module docs); the raw body is never
    /// surfaced in the error.
    pub async fn responses_nonstream(&self, region: &str, body: Bytes) -> Result<Bytes, AppError> {
        let resp = self.send(region, body).await?;
        let resp = error_for_status(resp)?;
        resp.bytes()
            .await
            .map_err(|e| AppError::UpstreamBedrock(format!("failed to read mantle response: {e}")))
    }

    /// POST `body` to the mantle `/responses` endpoint for `region` and return
    /// a raw byte stream of the SSE response (streaming passthrough).
    ///
    /// The pre-stream HTTP status is classified to an [`AppError`] *before* any
    /// bytes are yielded. Each subsequent chunk is yielded as-is; a transport
    /// error mid-stream becomes an [`AppError::UpstreamBedrock`] item.
    pub async fn responses_stream(
        &self,
        region: &str,
        body: Bytes,
    ) -> Result<BoxStream<'static, Result<Bytes, AppError>>, AppError> {
        let resp = self.send(region, body).await?;
        let resp = error_for_status(resp)?;
        let stream = resp.bytes_stream().map(|chunk| {
            chunk.map_err(|e| {
                AppError::UpstreamBedrock(format!("mantle stream transport error: {e}"))
            })
        });
        Ok(stream.boxed())
    }

    /// Issue the POST with the shared auth + content-type headers, mapping a
    /// transport-level failure (connect/timeout) to an [`AppError`].
    async fn send(&self, region: &str, body: Bytes) -> Result<reqwest::Response, AppError> {
        let url = self.responses_url(region);
        self.http
            .post(url)
            .bearer_auth(&self.bearer)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| AppError::UpstreamBedrock(format!("mantle request failed: {e}")))
    }
}

/// Classify a pre-stream non-2xx HTTP status into an [`AppError`].
///
/// The raw response body is intentionally discarded — it is never read into the
/// error message (no upstream body leakage into client errors or logs). Only
/// the status code informs the variant and message.
fn error_for_status(resp: reqwest::Response) -> Result<reqwest::Response, AppError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    match status {
        StatusCode::UNAUTHORIZED => Err(AppError::Unauthorized),
        StatusCode::TOO_MANY_REQUESTS => {
            Err(AppError::Throttled("mantle upstream throttled".to_string()))
        }
        s if s.is_client_error() => Err(AppError::BadRequest(format!(
            "mantle upstream rejected request (status {})",
            s.as_u16()
        ))),
        s => Err(AppError::UpstreamBedrock(format!(
            "mantle upstream error (status {})",
            s.as_u16()
        ))),
    }
}

#[cfg(test)]
mod tests {
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
            bearer.to_string(),
        )
    }

    const SSE_BODY: &str = "event: response.created\ndata: {\"type\":\"response.created\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\"}\n\n";

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
    /// Then the concatenated chunk bytes equal the mock body.
    #[tokio::test]
    async fn stream_concatenates_to_full_body() {
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
        assert_eq!(joined, SSE_BODY.as_bytes());
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
            "b".to_string(),
        );
        assert_eq!(
            client.responses_url("us-west-2"),
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1/responses"
        );
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
}
