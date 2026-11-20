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
//!
//! 401 means only that the client's key failed the gateway's own check. When
//! Bedrock denies the gateway's AWS identity, the client gets 403
//! `permission_denied` with Bedrock's reason, ARNs, account IDs and key-like
//! tokens replaced.

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
    /// Missing or invalid client API key (the gateway's own check) → 401.
    #[error("unauthorized")]
    Unauthorized,

    /// Bedrock denied the gateway's own AWS identity → 403.
    ///
    /// The client's key passed; Bedrock refused the gateway's credentials or
    /// its account's access to the model (`AccessDeniedException` on the AWS
    /// SDK path, 401/403 from the mantle endpoint). Carries the upstream
    /// reason with ARNs, account IDs and key-like tokens already replaced;
    /// build it with [`AppError::upstream_access_denied`]. `Display` leaves
    /// the reason out, as [`AppError::UpstreamOpenAi`] leaves out its
    /// envelope, so request-failure logs hold no upstream response text.
    #[error("upstream access denied")]
    UpstreamAccessDenied(String),

    /// Malformed / invalid request → 400.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// A documented-but-unsupported feature or model → 400.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Upstream Bedrock service fault (gateway-level) → 502.
    #[error("upstream bedrock error: {0}")]
    UpstreamBedrock(String),

    /// A structured OpenAI-compatible upstream rejection.
    ///
    /// `Display` intentionally excludes the envelope so request-failure logs do
    /// not contain upstream response bodies. [`AppError::envelope`] returns the
    /// parsed envelope only to the requesting client.
    #[error("upstream OpenAI-compatible request rejected (status {status})")]
    UpstreamOpenAi {
        status: StatusCode,
        envelope: OpenAiError,
    },

    /// Rate limited / quota exceeded → 429.
    #[error("throttled: {0}")]
    Throttled(String),

    /// Catch-all internal failure → 500.
    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    /// Returns `true` if this error is a 5xx server fault (502 or 500).
    /// Used for level-aware logging: server errors → `error`, client errors → `warn`.
    pub fn is_server_error(&self) -> bool {
        self.status().is_server_error()
    }

    /// Build [`AppError::UpstreamAccessDenied`] from the reason the upstream
    /// gave, replacing ARNs, account IDs and key-like tokens in it.
    #[must_use]
    pub fn upstream_access_denied(reason: &str) -> Self {
        AppError::UpstreamAccessDenied(sanitize_upstream_reason(reason))
    }

    /// Build a structured error from a terminal OpenAI-compatible stream event.
    ///
    /// Streaming errors arrive after an HTTP 200, so there is no upstream HTTP
    /// status left to preserve. Known rate-limit codes map to 429; every other
    /// terminal provider failure maps to 502. The original envelope is retained
    /// for the requesting client while [`Self`]'s `Display` remains body-free.
    pub fn from_openai_stream_error(
        code: Option<&str>,
        message: Option<&str>,
        param: Option<&str>,
        error_type: Option<&str>,
    ) -> Self {
        let code = non_empty(code).unwrap_or("upstream_error");
        let rate_limited =
            is_rate_limit_marker(code) || non_empty(error_type).is_some_and(is_rate_limit_marker);
        let status = if rate_limited {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::BAD_GATEWAY
        };
        let error_type = non_empty(error_type).unwrap_or(if rate_limited {
            "rate_limit_error"
        } else {
            "api_error"
        });
        let message = non_empty(message).unwrap_or("Responses upstream reported a failed stream");

        AppError::UpstreamOpenAi {
            status,
            envelope: OpenAiError {
                error: ErrorBody {
                    message: message.to_string(),
                    r#type: Some(error_type.to_string()),
                    param: non_empty(param).map(str::to_string),
                    code: Some(code.to_string()),
                },
            },
        }
    }

    /// Machine-readable upstream code safe to include in structured logs.
    #[must_use]
    pub fn upstream_error_code(&self) -> Option<&str> {
        match self {
            AppError::UpstreamOpenAi { envelope, .. } => envelope.error.code.as_deref(),
            _ => None,
        }
    }

    /// Machine-readable upstream type safe to include in structured logs.
    #[must_use]
    pub fn upstream_error_type(&self) -> Option<&str> {
        match self {
            AppError::UpstreamOpenAi { envelope, .. } => envelope.error.r#type.as_deref(),
            _ => None,
        }
    }

    /// Returns the HTTP status code for this error.
    fn status(&self) -> StatusCode {
        match self {
            AppError::Unauthorized => StatusCode::UNAUTHORIZED, // 401
            AppError::UpstreamAccessDenied(_) => StatusCode::FORBIDDEN, // 403
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST, // 400
            AppError::Unsupported(_) => StatusCode::BAD_REQUEST, // 400
            AppError::Throttled(_) => StatusCode::TOO_MANY_REQUESTS, // 429
            AppError::UpstreamBedrock(_) => StatusCode::BAD_GATEWAY, // 502
            AppError::UpstreamOpenAi { status, .. } => *status,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR, // 500
        }
    }

    /// Returns the OpenAI `error.type` discriminator for this error.
    fn error_type(&self) -> &'static str {
        match self {
            AppError::Unauthorized => "invalid_request_error",
            AppError::UpstreamAccessDenied(_) => "invalid_request_error",
            AppError::BadRequest(_) => "invalid_request_error",
            AppError::Unsupported(_) => "invalid_request_error",
            AppError::Throttled(_) => "rate_limit_error",
            AppError::UpstreamBedrock(_) => "api_error",
            AppError::UpstreamOpenAi { .. } => "invalid_request_error",
            AppError::Internal(_) => "api_error",
        }
    }

    /// Returns the OpenAI `error.code` (stable machine-readable identifier).
    fn error_code(&self) -> &'static str {
        match self {
            AppError::Unauthorized => "unauthorized",
            AppError::UpstreamAccessDenied(_) => "permission_denied",
            AppError::BadRequest(_) => "bad_request",
            AppError::Unsupported(_) => "unsupported",
            AppError::Throttled(_) => "rate_limit_exceeded",
            AppError::UpstreamBedrock(_) => "upstream_error",
            AppError::UpstreamOpenAi { .. } => "upstream_error",
            AppError::Internal(_) => "internal_error",
        }
    }

    /// Renders the human-readable message carried by this error.
    fn message(&self) -> String {
        match self {
            AppError::Unauthorized => {
                "Incorrect API key provided or missing Authorization header.".to_string()
            }
            AppError::UpstreamAccessDenied(reason) if reason.is_empty() => {
                UPSTREAM_ACCESS_DENIED.to_string()
            }
            AppError::UpstreamAccessDenied(reason) => format!("{UPSTREAM_ACCESS_DENIED}: {reason}"),
            AppError::UpstreamOpenAi { .. } => {
                "upstream OpenAI-compatible request rejected".to_string()
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
        if let AppError::UpstreamOpenAi { envelope, .. } = self {
            return envelope.clone();
        }
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

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Start of the client-facing message of [`AppError::UpstreamAccessDenied`].
const UPSTREAM_ACCESS_DENIED: &str = "Bedrock denied the gateway access";

/// What [`sanitize_upstream_reason`] puts in place of an ARN.
const ARN_PLACEHOLDER: &str = "<arn>";
/// What [`sanitize_upstream_reason`] puts in place of an AWS account ID.
const ACCOUNT_ID_PLACEHOLDER: &str = "<account-id>";
/// What [`sanitize_upstream_reason`] puts in place of a key-like token.
const SECRET_PLACEHOLDER: &str = "<redacted>";

/// An AWS account ID is a run of exactly this many digits.
const ACCOUNT_ID_DIGITS: usize = 12;

/// Prefixes of AWS access key IDs (20 characters) and IAM unique IDs (21),
/// both upper-case letters and digits.
const AWS_KEY_ID_PREFIXES: [&str; 12] = [
    "AKIA", "ASIA", "ABIA", "ACCA", "AGPA", "AIDA", "AIPA", "ANPA", "ANVA", "APKA", "AROA", "ASCA",
];
const AWS_KEY_ID_LENS: std::ops::RangeInclusive<usize> = 20..=21;

/// Prefixes of bearer secrets: Bedrock API keys (long-term `ABSK`, short-term
/// `bedrock-api-key-`), OpenAI-style `sk-` keys and JWTs (`eyJ`).
const SECRET_PREFIXES: [&str; 4] = ["ABSK", "bedrock-api-key-", "sk-", "eyJ"];
/// A token with a secret prefix counts from this length on.
const MIN_PREFIXED_SECRET_LEN: usize = 20;

/// An opaque token (base64 with mixed case and digits, or hex) counts as
/// key-like from this length on. Model IDs stay shorter or carry `.`/`:`.
const MIN_OPAQUE_TOKEN_LEN: usize = 32;

/// Replace ARNs, AWS account IDs and key-like tokens in an upstream error
/// message, keeping the rest (model IDs, IAM actions, URLs, wording) as is.
///
/// The message is processed one whitespace-separated word at a time. Quotes,
/// brackets and punctuation that wrap or end a word stay around the
/// replacement, so `(arn:aws:...).` becomes `(<arn>).`. Within a word:
///
/// - from an `arn:aws` (any partition) to the end of the word → `<arn>`;
/// - a word after `Bearer`, an AWS access key ID or IAM unique ID, a token
///   with a secret prefix (`ABSK`, `bedrock-api-key-`, `sk-`, `eyJ`) or a
///   long opaque token → `<redacted>`, also as the value of a `name=value`
///   or `name:value` pair;
/// - a run of exactly 12 digits, or `1234-5678-9012` → `<account-id>`.
pub(crate) fn sanitize_upstream_reason(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut after_bearer = false;
    for piece in raw.trim().split_inclusive(char::is_whitespace) {
        let word = piece.trim_end_matches(char::is_whitespace);
        let (lead, core, trail) = split_word(word);
        out.push_str(lead);
        out.push_str(&sanitize_word_core(core, after_bearer));
        out.push_str(trail);
        out.push_str(&piece[word.len()..]);
        if !word.is_empty() {
            after_bearer = trail.is_empty() && core.eq_ignore_ascii_case("bearer");
        }
    }
    out
}

/// Punctuation that can open a word around an identifier.
const LEADING_PUNCTUATION: [char; 7] = ['"', '\'', '`', '(', '[', '{', '<'];
/// Punctuation that can close a word, or the sentence, after an identifier.
const TRAILING_PUNCTUATION: [char; 13] = [
    '"', '\'', '`', ')', ']', '}', '>', ',', ';', '.', ':', '!', '?',
];

/// Split a word into its leading punctuation, its core and its trailing
/// punctuation.
fn split_word(word: &str) -> (&str, &str, &str) {
    let rest = word.trim_start_matches(LEADING_PUNCTUATION);
    let lead = &word[..word.len() - rest.len()];
    let core = rest.trim_end_matches(TRAILING_PUNCTUATION);
    (lead, core, &rest[core.len()..])
}

fn sanitize_word_core(core: &str, after_bearer: bool) -> String {
    if core.is_empty() {
        return String::new();
    }
    if after_bearer || is_key_like(core) {
        return SECRET_PLACEHOLDER.to_string();
    }
    if let Some(start) = arn_start(core) {
        return format!("{}{ARN_PLACEHOLDER}", &core[..start]);
    }
    // `name=value` / `name:value`: look at each part on its own.
    let mut out = String::with_capacity(core.len());
    let mut part_start = 0;
    for (index, c) in core.char_indices() {
        if matches!(c, ':' | '=') {
            out.push_str(&sanitize_part(&core[part_start..index]));
            out.push(c);
            part_start = index + c.len_utf8();
        }
    }
    out.push_str(&sanitize_part(&core[part_start..]));
    out
}

fn sanitize_part(part: &str) -> String {
    if is_key_like(part) {
        SECRET_PLACEHOLDER.to_string()
    } else {
        replace_account_ids(part)
    }
}

/// Byte offset of an `arn:aws` that starts the word or follows a
/// non-alphanumeric character.
fn arn_start(core: &str) -> Option<usize> {
    core.match_indices("arn:aws")
        .map(|(index, _)| index)
        .find(|&index| {
            !core[..index]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric())
        })
}

fn is_key_like(token: &str) -> bool {
    let bytes = token.as_bytes();
    let has = |test: fn(&u8) -> bool| bytes.iter().any(test);

    let aws_key_id = AWS_KEY_ID_LENS.contains(&token.len())
        && AWS_KEY_ID_PREFIXES
            .iter()
            .any(|prefix| token.starts_with(prefix))
        && bytes
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit());
    let prefixed_secret = token.len() >= MIN_PREFIXED_SECRET_LEN
        && SECRET_PREFIXES
            .iter()
            .any(|prefix| token.starts_with(prefix));
    let long = token.len() >= MIN_OPAQUE_TOKEN_LEN;
    let base64 = long
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'_' | b'-'))
        && has(u8::is_ascii_uppercase)
        && has(u8::is_ascii_lowercase)
        && has(u8::is_ascii_digit);
    let hex = long
        && bytes.iter().all(u8::is_ascii_hexdigit)
        && has(u8::is_ascii_digit)
        && has(u8::is_ascii_alphabetic);
    aws_key_id || prefixed_secret || base64 || hex
}

/// Replace every run of exactly [`ACCOUNT_ID_DIGITS`] digits, and a whole
/// `1234-5678-9012`, with [`ACCOUNT_ID_PLACEHOLDER`].
fn replace_account_ids(part: &str) -> String {
    let bytes = part.as_bytes();
    let dashed = bytes.len() == 14
        && bytes.iter().enumerate().all(|(index, b)| match index {
            4 | 9 => *b == b'-',
            _ => b.is_ascii_digit(),
        });
    if dashed {
        return ACCOUNT_ID_PLACEHOLDER.to_string();
    }
    let mut out = String::with_capacity(part.len());
    let mut digits_start = None;
    for (index, c) in part.char_indices() {
        if c.is_ascii_digit() {
            if digits_start.is_none() {
                digits_start = Some(index);
            }
            continue;
        }
        if let Some(start) = digits_start.take() {
            push_digits(&mut out, &part[start..index]);
        }
        out.push(c);
    }
    if let Some(start) = digits_start {
        push_digits(&mut out, &part[start..]);
    }
    out
}

fn push_digits(out: &mut String, digits: &str) {
    if digits.len() == ACCOUNT_ID_DIGITS {
        out.push_str(ACCOUNT_ID_PLACEHOLDER);
    } else {
        out.push_str(digits);
    }
}

fn is_rate_limit_marker(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "rate_limit_exceeded" | "rate_limit_error" | "insufficient_quota" | "too_many_requests"
    )
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
/// - [`AppError::UpstreamOpenAi`] → classification by preserved HTTP status
/// - [`AppError::UpstreamAccessDenied`] (Bedrock `AccessDeniedException`, a
///   mantle 401/403) → `permission_denied`, the code of the HTTP envelope
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
        AppError::UpstreamOpenAi { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS => {
            "rate_limit_exceeded"
        }
        AppError::UpstreamOpenAi { status, .. } if status.is_client_error() => {
            "context_length_exceeded"
        }
        AppError::UpstreamOpenAi { .. } => "server_is_overloaded",
        AppError::UpstreamAccessDenied(_) => "permission_denied",
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
/// - `AccessDeniedException` → [`AppError::UpstreamAccessDenied`] (403): Bedrock
///   denied the gateway's own identity (IAM, or a model not available to the
///   account), which the client cannot fix with another key, so it is never
///   the client-key 401
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
        "AccessDeniedException" => AppError::upstream_access_denied(&msg),
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
    async fn upstream_openai_preserves_envelope_without_display_leak() {
        let marker = "prompt tokens exceed model maximum";
        let error = AppError::UpstreamOpenAi {
            status: StatusCode::BAD_REQUEST,
            envelope: OpenAiError {
                error: ErrorBody {
                    message: marker.to_string(),
                    r#type: Some("invalid_request_error".to_string()),
                    param: None,
                    code: Some("validation_error".to_string()),
                },
            },
        };

        assert!(!error.to_string().contains(marker));
        let (status, value) = body_json(error.into_response()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["message"], marker);
        assert_eq!(value["error"]["code"], "validation_error");
    }

    #[tokio::test]
    async fn openai_stream_error_preserves_envelope_and_maps_server_failure_to_502() {
        let marker = "provider detail visible only to the requesting client";
        let error =
            AppError::from_openai_stream_error(Some("server_error"), Some(marker), None, None);

        assert!(error.is_server_error());
        assert_eq!(error.upstream_error_code(), Some("server_error"));
        assert_eq!(error.upstream_error_type(), Some("api_error"));
        assert!(!error.to_string().contains(marker));

        let (status, value) = body_json(error.into_response()).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(value["error"]["message"], marker);
        assert_eq!(value["error"]["type"], "api_error");
        assert_eq!(value["error"]["code"], "server_error");
    }

    #[tokio::test]
    async fn openai_stream_rate_limit_maps_to_429() {
        let error = AppError::from_openai_stream_error(
            Some("rate_limit_exceeded"),
            Some("retry later"),
            Some("requests"),
            None,
        );

        assert!(!error.is_server_error());
        assert_eq!(error.upstream_error_type(), Some("rate_limit_error"));

        let (status, value) = body_json(error.into_response()).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(value["error"]["param"], "requests");
        assert_eq!(value["error"]["code"], "rate_limit_exceeded");
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
            ("AccessDeniedException", StatusCode::FORBIDDEN),
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

    /// Server errors (5xx) are correctly classified as true; client errors (4xx) as false.
    #[test]
    fn app_error_is_server_error_classification() {
        // 5xx faults → true
        assert!(AppError::UpstreamBedrock("x".into()).is_server_error());
        assert!(AppError::Internal("x".into()).is_server_error());

        // 4xx client errors → false
        assert!(!AppError::BadRequest("x".into()).is_server_error());
        assert!(!AppError::Unsupported("x".into()).is_server_error());
        assert!(!AppError::Throttled("x".into()).is_server_error());
        assert!(!AppError::Unauthorized.is_server_error());
        assert!(!AppError::upstream_access_denied("x").is_server_error());
    }

    /// The reason Bedrock gave for `global.zai.glm-5.3` on 2026-11-20
    /// (`Converse`, HTTP 403 `AccessDeniedException`).
    const GLM_5_3_DENIAL: &str = "zai.glm-5.3 is not available for this account. You can explore \
        other available models on Amazon Bedrock. For additional access options, contact AWS \
        Sales at https://aws.amazon.com/contact-us/sales-support/";

    /// An IAM denial of the gateway's role, with the ARNs Bedrock puts in it.
    const IAM_DENIAL: &str = "User: arn:aws:sts::123456789012:assumed-role/GatewayTaskRole/a1b2c3 \
        is not authorized to perform: bedrock:InvokeModelWithResponseStream on resource: \
        arn:aws:bedrock:us-east-2:123456789012:inference-profile/global.zai.glm-5.3 because no \
        identity-based policy allows the bedrock:InvokeModelWithResponseStream action";

    /// The gateway's own client-key check keeps its 401 envelope.
    #[tokio::test]
    async fn client_key_failure_still_renders_401_unauthorized() {
        let (status, value) = body_json(AppError::Unauthorized.into_response()).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "unauthorized");
        assert_eq!(
            value["error"]["message"],
            "Incorrect API key provided or missing Authorization header."
        );
    }

    /// Bedrock denying the gateway's identity is a 403 `permission_denied`
    /// carrying Bedrock's reason, not the 401 that blames the client's key.
    #[tokio::test]
    async fn bedrock_access_denied_renders_403_with_reason() {
        let app_err =
            from_bedrock_sdk_error(&meta_with_code("AccessDeniedException", GLM_5_3_DENIAL));

        assert!(matches!(app_err, AppError::UpstreamAccessDenied(_)));
        let (status, value) = body_json(app_err.into_response()).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "permission_denied");
        assert_eq!(
            value["error"]["message"],
            format!("Bedrock denied the gateway access: {GLM_5_3_DENIAL}")
        );
        assert!(value["error"].get("param").is_none());
    }

    /// The rendered reason has its ARNs and account IDs replaced, and the
    /// error's `Display` (what request-failure logs print) holds no reason.
    #[tokio::test]
    async fn bedrock_access_denied_body_is_sanitized_and_display_is_reason_free() {
        let app_err = from_bedrock_sdk_error(&meta_with_code("AccessDeniedException", IAM_DENIAL));

        assert_eq!(app_err.to_string(), "upstream access denied");
        let (status, value) = body_json(app_err.into_response()).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(
            value["error"]["message"],
            "Bedrock denied the gateway access: User: <arn> is not authorized to perform: \
             bedrock:InvokeModelWithResponseStream on resource: <arn> because no \
             identity-based policy allows the bedrock:InvokeModelWithResponseStream action"
        );
        let body = value.to_string();
        for leaked in [
            "123456789012",
            "arn:aws",
            "GatewayTaskRole",
            "Incorrect API key",
        ] {
            assert!(!body.contains(leaked), "{leaked} leaked: {body}");
        }
    }

    /// A denial without a message still names the denial.
    #[test]
    fn upstream_access_denied_without_reason_keeps_the_prefix() {
        let app_err = AppError::upstream_access_denied("  ");
        assert_eq!(
            app_err.envelope().error.message,
            "Bedrock denied the gateway access"
        );
    }

    /// The Responses `response.failed` path gets the same code and message.
    #[test]
    fn responses_access_denied_maps_to_permission_denied() {
        let app_err = from_bedrock_sdk_error(&meta_with_code("AccessDeniedException", IAM_DENIAL));
        let (code, message) = responses_error(&app_err);
        assert_eq!(code, "permission_denied");
        assert_eq!(message, app_err.envelope().error.message);
        assert!(!message.contains("123456789012"), "{message}");
    }

    #[test]
    fn sanitize_replaces_arns_of_every_partition_keeping_punctuation() {
        let cases = [
            ("on arn:aws:iam::123456789012:role/x.", "on <arn>."),
            ("(arn:aws-cn:iam::123456789012:role/x),", "(<arn>),"),
            (
                "'arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:foundation-model/m'",
                "'<arn>'",
            ),
            (
                "principal=arn:aws:iam::123456789012:user/BedrockAPIKey-ab12",
                "principal=<arn>",
            ),
            ("arn:aws:bedrock:::foundation-model/zai.glm-5.3", "<arn>"),
        ];
        for (raw, expected) in cases {
            assert_eq!(sanitize_upstream_reason(raw), expected, "{raw}");
        }
    }

    #[test]
    fn sanitize_replaces_account_ids() {
        let cases = [
            ("account 123456789012.", "account <account-id>."),
            ("account 1234-5678-9012", "account <account-id>"),
            (
                "https://123456789012.signin.aws.amazon.com/console",
                "https://<account-id>.signin.aws.amazon.com/console",
            ),
            // Only a run of exactly 12 digits is an account ID.
            (
                "12345678901 1234567890123 20250929",
                "12345678901 1234567890123 20250929",
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(sanitize_upstream_reason(raw), expected, "{raw}");
        }
    }

    #[test]
    fn sanitize_redacts_key_like_tokens() {
        let cases = [
            ("key AKIAIOSFODNN7EXAMPLE", "key <redacted>"),
            ("ASIAY34FZKBOKMUTVV7A.", "<redacted>."),
            (
                "AROA3XFRBF535PLBIFPI4:gateway-session",
                "<redacted>:gateway-session",
            ),
            (
                "Invalid key: ABSKQmVkcm9ja0FQSUtleS1leGFtcGxl",
                "Invalid key: <redacted>",
            ),
            ("bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29t", "<redacted>"),
            ("sk-proj-abcdefghijklmnop1234", "<redacted>"),
            (
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
                "<redacted>",
            ),
            (
                "Authorization: Bearer  abc123",
                "Authorization: Bearer  <redacted>",
            ),
            (
                "secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "secret <redacted>",
            ),
            (
                "token:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "token:<redacted>",
            ),
            (
                "sig 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                "sig <redacted>",
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(sanitize_upstream_reason(raw), expected, "{raw}");
        }
    }

    /// Model IDs, IAM actions, URLs, request IDs and wording survive, and
    /// so does the whitespace between words.
    #[test]
    fn sanitize_keeps_ordinary_text() {
        for raw in [
            GLM_5_3_DENIAL,
            "anthropic.claude-sonnet-4-5-20250929-v1:0 us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "claude-sonnet-4-5-20250929-v1 model-not-available-in-region-us-east-2",
            "(aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)",
            "request 3fa85f64-5717-4562-b3fc-2c963f66afa6 failed;\n  retry after 15 minutes",
            "Bearer, then nothing secret",
            "",
        ] {
            assert_eq!(sanitize_upstream_reason(raw), raw);
        }
    }

    #[test]
    fn sanitize_is_idempotent() {
        let once = sanitize_upstream_reason(IAM_DENIAL);
        assert_eq!(sanitize_upstream_reason(&once), once);
    }
}
