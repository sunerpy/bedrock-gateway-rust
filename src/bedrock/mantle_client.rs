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
#[path = "mantle_client_tests.rs"]
mod tests;
