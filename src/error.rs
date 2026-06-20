//! Error types and handling.
//!
//! This module defines the application-wide [`AppError`] type and renders it
//! as the **full OpenAI error envelope** on the wire:
//!
//! ```json
//! {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}
//! ```
//!
//! ## Documented FIX vs Python
//!
//! The legacy Python service raised `fastapi.HTTPException(status_code, detail=str(e))`
//! (see `.legacy-python/src/api/models/bedrock.py` 1898-1906 and
//! `.legacy-python/src/api/schema.py` 209-214), which produces a *minimal*
//! `{"detail": "..."}` (or, for the `Error`/`ErrorMessage` models, only
//! `{"error": {"message": "..."}}`). Real OpenAI clients expect the full
//! envelope with `type`/`param`/`code`. We standardize on the full shape here.
//!
//! ## Status mapping
//!
//! Mirrors the Python Bedrock invoke error handling (Validation→400,
//! Throttling→429, else 500) but adds an explicit upstream/gateway distinction
//! (502) for Bedrock service faults so clients can tell apart "your request was
//! bad" (4xx) from "the upstream model provider failed" (502).

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use thiserror::Error;

use crate::openai::schema::{ErrorBody, OpenAiError};

/// Application-wide error type.
///
/// Every variant carries the human-readable message that will be surfaced in
/// the OpenAI error envelope. Construction is the responsibility of the call
/// site (handlers, the Bedrock mapper, auth middleware, etc.); rendering to a
/// response is centralized in the [`IntoResponse`] impl below.
#[derive(Debug, Error)]
pub enum AppError {
    /// Missing or invalid API key → 401.
    #[error("unauthorized")]
    Unauthorized,

    /// Malformed / invalid request → 400.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// A documented-but-unsupported feature or model → 400.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Upstream Bedrock service fault (gateway-level) → 502.
    #[error("upstream bedrock error: {0}")]
    UpstreamBedrock(String),

    /// Rate limited / quota exceeded → 429.
    #[error("throttled: {0}")]
    Throttled(String),

    /// Catch-all internal failure → 500.
    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    /// Returns the HTTP status code for this error.
    fn status(&self) -> StatusCode {
        match self {
            AppError::Unauthorized => StatusCode::UNAUTHORIZED, // 401
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST, // 400
            AppError::Unsupported(_) => StatusCode::BAD_REQUEST, // 400
            AppError::Throttled(_) => StatusCode::TOO_MANY_REQUESTS, // 429
            AppError::UpstreamBedrock(_) => StatusCode::BAD_GATEWAY, // 502
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR, // 500
        }
    }

    /// Returns the OpenAI `error.type` discriminator for this error.
    fn error_type(&self) -> &'static str {
        match self {
            AppError::Unauthorized => "invalid_request_error",
            AppError::BadRequest(_) => "invalid_request_error",
            AppError::Unsupported(_) => "invalid_request_error",
            AppError::Throttled(_) => "rate_limit_error",
            AppError::UpstreamBedrock(_) => "api_error",
            AppError::Internal(_) => "api_error",
        }
    }

    /// Returns the OpenAI `error.code` (stable machine-readable identifier).
    fn error_code(&self) -> &'static str {
        match self {
            AppError::Unauthorized => "unauthorized",
            AppError::BadRequest(_) => "bad_request",
            AppError::Unsupported(_) => "unsupported",
            AppError::Throttled(_) => "rate_limit_exceeded",
            AppError::UpstreamBedrock(_) => "upstream_error",
            AppError::Internal(_) => "internal_error",
        }
    }

    /// Renders the human-readable message carried by this error.
    fn message(&self) -> String {
        match self {
            AppError::Unauthorized => {
                "Incorrect API key provided or missing Authorization header.".to_string()
            }
            AppError::BadRequest(m)
            | AppError::Unsupported(m)
            | AppError::UpstreamBedrock(m)
            | AppError::Throttled(m)
            | AppError::Internal(m) => m.clone(),
        }
    }

    /// Builds the OpenAI error envelope body for this error.
    ///
    /// Public so the streaming SSE path can render the SAME full envelope inline
    /// as a `data:` event (rather than a reduced shape), keeping the streaming
    /// and non-streaming error wire shapes consistent.
    pub fn envelope(&self) -> OpenAiError {
        OpenAiError {
            error: ErrorBody {
                message: self.message(),
                r#type: Some(self.error_type().to_string()),
                // OpenAI's envelope always includes `param`; we have no
                // request parameter context to attribute, so it stays null.
                // `ErrorBody` skips `None`, which is wire-compatible — clients
                // treat an absent `param` as `null`.
                param: None,
                code: Some(self.error_code().to_string()),
            },
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = self.envelope();
        (status, Json(body)).into_response()
    }
}

/// Codex-actionable error code for the OpenAI **Responses API** `response.error`
/// payload (the `response.failed` stream event and any non-stream Responses
/// error path).
///
/// This is the SINGLE shared mapping point so the streaming path
/// ([`crate::bedrock::responses_stream::ResponsesStreamState::fail`]) and the
/// pre-stream HTTP path agree on the code surfaced to codex. It maps the
/// Bedrock-classified [`AppError`] variant (already produced by
/// [`from_bedrock_sdk_error`], whose variant IS the Bedrock exception class
/// after classification) to a codex-recognized Responses error code:
///
/// - [`AppError::Throttled`] (Bedrock `ThrottlingException`) →
///   `rate_limit_exceeded`
/// - [`AppError::UpstreamBedrock`] (Bedrock 5xx /
///   `ServiceUnavailableException` / `ModelNotReadyException` /
///   `InternalServerException` / `ModelTimeoutException` /
///   `ServiceQuotaExceededException` / `ModelErrorException`) →
///   `server_is_overloaded`
/// - [`AppError::BadRequest`] (Bedrock `ValidationException`, which Bedrock
///   raises for context/length violations) → `context_length_exceeded`
/// - everything else ([`AppError::Internal`], [`AppError::Unsupported`],
///   [`AppError::Unauthorized`]) → `server_error`
///
/// Returned as `(code, message)` so callers can build the
/// `{"code": ..., "message": ...}` Responses error object without duplicating
/// the classification.
#[must_use]
pub fn responses_error(err: &AppError) -> (&'static str, String) {
    let code = match err {
        AppError::Throttled(_) => "rate_limit_exceeded",
        AppError::UpstreamBedrock(_) => "server_is_overloaded",
        AppError::BadRequest(_) => "context_length_exceeded",
        AppError::Unsupported(_) | AppError::Unauthorized | AppError::Internal(_) => "server_error",
    };
    (code, err.message())
}

/// Maps an AWS Bedrock SDK error into an [`AppError`].
///
/// This is intentionally generic over any Smithy/SDK error that exposes error
/// metadata (which every `aws_sdk_bedrockruntime` / `aws_sdk_bedrock`
/// `SdkError<E, R>` and its `into_service_error()` output do via
/// [`ProvideErrorMetadata`]). We match on the modeled exception **code**
/// (the shape name, e.g. `"ThrottlingException"`), which is the stable,
/// service-defined discriminator — this avoids brittle coupling to a single
/// operation's error enum while still being a real, typed mapping rather than a
/// stub.
///
/// Mapping (mirrors Python Validation→400 / Throttling→429 / else 500, plus an
/// explicit upstream/gateway 502 for transient Bedrock faults):
/// - `ThrottlingException` → [`AppError::Throttled`] (429)
/// - `ValidationException` → [`AppError::BadRequest`] (400)
/// - `ModelNotReadyException` / `ServiceUnavailableException` /
///   `ModelTimeoutException` / `ServiceQuotaExceededException` /
///   `InternalServerException` → [`AppError::UpstreamBedrock`] (502)
/// - `AccessDeniedException` → [`AppError::Unauthorized`] (401)
/// - `ResourceNotFoundException` → [`AppError::BadRequest`] (400)
/// - anything else → [`AppError::Internal`] (500)
pub fn from_bedrock_sdk_error<E>(err: &E) -> AppError
where
    E: aws_smithy_types::error::metadata::ProvideErrorMetadata + std::fmt::Display,
{
    let code = err.code().unwrap_or_default();
    // Prefer the modeled message; fall back to Display when absent.
    let msg = err
        .message()
        .map(str::to_string)
        .unwrap_or_else(|| err.to_string());

    match code {
        "ThrottlingException" => AppError::Throttled(msg),
        "ValidationException" => AppError::BadRequest(msg),
        "ModelNotReadyException"
        | "ServiceUnavailableException"
        | "ModelTimeoutException"
        | "ServiceQuotaExceededException"
        | "InternalServerException"
        | "ModelErrorException" => AppError::UpstreamBedrock(msg),
        "AccessDeniedException" => AppError::Unauthorized,
        "ResourceNotFoundException" => AppError::BadRequest(msg),
        _ => AppError::Internal(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_smithy_types::error::metadata::ErrorMetadata;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use serde_json::Value;

    /// Drains an axum response body into parsed JSON for assertions.
    async fn body_json(resp: Response) -> (StatusCode, Value) {
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body bytes");
        let value: Value = serde_json::from_slice(&bytes).expect("parse json body");
        (status, value)
    }

    /// MUST DO: BadRequest renders 400 with a full envelope carrying
    /// `error.message` and `error.type`.
    #[tokio::test]
    async fn bad_request_renders_400_envelope() {
        let resp = AppError::BadRequest("x".to_string()).into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["message"], "x");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "bad_request");
        // Envelope must be nested under `error`, not plain text / `detail`.
        assert!(value.get("error").is_some());
        assert!(value.get("detail").is_none());
    }

    /// MUST DO: Throttled maps to 429 with the rate-limit type.
    #[tokio::test]
    async fn throttled_renders_429() {
        let resp = AppError::Throttled("slow down".to_string()).into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(value["error"]["message"], "slow down");
        assert_eq!(value["error"]["type"], "rate_limit_error");
    }

    #[tokio::test]
    async fn unauthorized_renders_401() {
        let resp = AppError::Unauthorized.into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert!(value["error"]["message"].as_str().is_some());
    }

    #[tokio::test]
    async fn upstream_bedrock_renders_502_api_error() {
        let resp = AppError::UpstreamBedrock("bedrock down".to_string()).into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(value["error"]["type"], "api_error");
        assert_eq!(value["error"]["message"], "bedrock down");
    }

    #[tokio::test]
    async fn internal_renders_500_api_error() {
        let resp = AppError::Internal("boom".to_string()).into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(value["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn unsupported_renders_400() {
        let resp = AppError::Unsupported("no streaming for this model".to_string()).into_response();
        let (status, value) = body_json(resp).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "unsupported");
    }

    fn meta_with_code(code: &str, message: &str) -> ErrorMetadata {
        ErrorMetadata::builder().code(code).message(message).build()
    }

    /// The Bedrock SDK mapper honors the modeled exception code.
    #[test]
    fn bedrock_sdk_error_maps_by_code() {
        let cases = [
            ("ThrottlingException", StatusCode::TOO_MANY_REQUESTS),
            ("ValidationException", StatusCode::BAD_REQUEST),
            ("ModelNotReadyException", StatusCode::BAD_GATEWAY),
            ("ServiceUnavailableException", StatusCode::BAD_GATEWAY),
            ("AccessDeniedException", StatusCode::UNAUTHORIZED),
            ("ResourceNotFoundException", StatusCode::BAD_REQUEST),
            ("SomethingElseException", StatusCode::INTERNAL_SERVER_ERROR),
        ];

        for (code, expected_status) in cases {
            let meta = meta_with_code(code, "detail msg");
            let app_err = from_bedrock_sdk_error(&meta);
            assert_eq!(
                app_err.status(),
                expected_status,
                "code {code} should map to {expected_status}"
            );
        }
    }

    /// Throttling specifically maps to the 429 rate-limit variant with the
    /// modeled message preserved.
    #[test]
    fn bedrock_throttling_maps_to_throttled() {
        let meta = meta_with_code("ThrottlingException", "Too many requests");
        let app_err = from_bedrock_sdk_error(&meta);

        assert!(matches!(app_err, AppError::Throttled(_)));
        assert_eq!(app_err.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(app_err.message(), "Too many requests");
    }

    /// Validation maps to a 400 BadRequest, mirroring Python's Validation→400.
    #[test]
    fn bedrock_validation_maps_to_bad_request() {
        let meta = meta_with_code("ValidationException", "bad input");
        let app_err = from_bedrock_sdk_error(&meta);

        assert!(matches!(app_err, AppError::BadRequest(_)));
        assert_eq!(app_err.status(), StatusCode::BAD_REQUEST);
    }

    /// When no modeled message exists, the mapper falls back to Display
    /// (never panics, no unwrap on the response path).
    #[test]
    fn bedrock_unknown_falls_back_to_internal() {
        let meta = ErrorMetadata::builder().code("WeirdException").build();
        let app_err = from_bedrock_sdk_error(&meta);
        assert!(matches!(app_err, AppError::Internal(_)));
        assert_eq!(app_err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// The shared Responses mapper turns a Bedrock `ThrottlingException`
    /// (classified to [`AppError::Throttled`]) into `rate_limit_exceeded`.
    #[test]
    fn responses_throttling_maps_to_rate_limit_exceeded() {
        let app_err = from_bedrock_sdk_error(&meta_with_code("ThrottlingException", "slow down"));
        let (code, message) = responses_error(&app_err);
        assert_eq!(code, "rate_limit_exceeded");
        assert_eq!(message, "slow down");
    }

    /// A Bedrock `ValidationException` (context/length) maps to
    /// `context_length_exceeded`.
    #[test]
    fn responses_validation_maps_to_context_length_exceeded() {
        let app_err = from_bedrock_sdk_error(&meta_with_code(
            "ValidationException",
            "input too long for context window",
        ));
        let (code, _) = responses_error(&app_err);
        assert_eq!(code, "context_length_exceeded");
    }

    /// Bedrock 5xx faults (`ServiceUnavailableException`,
    /// `ModelNotReadyException`, `InternalServerException`) all map to
    /// `server_is_overloaded`.
    #[test]
    fn responses_5xx_maps_to_server_is_overloaded() {
        for class in [
            "ServiceUnavailableException",
            "ModelNotReadyException",
            "InternalServerException",
        ] {
            let app_err = from_bedrock_sdk_error(&meta_with_code(class, "upstream fault"));
            let (code, _) = responses_error(&app_err);
            assert_eq!(code, "server_is_overloaded", "{class} should be overloaded");
        }
    }

    /// Anything unclassified falls back to the generic `server_error` code.
    #[test]
    fn responses_unknown_maps_to_server_error() {
        let (code, _) = responses_error(&AppError::Internal("boom".to_string()));
        assert_eq!(code, "server_error");
    }
}
