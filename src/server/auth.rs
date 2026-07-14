//! API key resolution and bearer authentication.
//!
//! This module reproduces the legacy Python boot-time auth behavior
//! (`.legacy-python/src/api/auth.py`, 43 lines) in Rust:
//!
//! 1. **3-tier API key resolution** with a fixed priority order:
//!    SSM Parameter Store → Secrets Manager → `API_KEY` env/setting → error.
//!    The key is resolved **once at boot** and cached in `AppState` by the
//!    caller (task 24). It is **never** fetched per request.
//! 2. **Bearer authentication** as an axum middleware that compares the
//!    `Authorization: Bearer <token>` header against the resolved key.
//!
//! ## Why a custom middleware instead of `ValidateRequestHeaderLayer::bearer`
//!
//! `tower_http::validate_request::ValidateRequestHeaderLayer::bearer` is the
//! idiomatic choice and would reject mismatches with a bare `401` and an empty
//! body. Real OpenAI clients, however, expect the **full OpenAI error
//! envelope** on `401`:
//!
//! ```json
//! {"error": {"message": "...", "type": "invalid_request_error",
//!            "param": null, "code": "unauthorized"}}
//! ```
//!
//! To stay wire-compatible we use [`axum::middleware::from_fn_with_state`] and
//! return [`AppError::Unauthorized`], whose [`IntoResponse`] renders that
//! envelope (see [`crate::error`]). Apply it with `route_layer` on the
//! protected routes so that a wrong HTTP method still yields `405` (not `401`).
//! `/health` is deliberately left outside the protected subtree (wired in
//! task 24) and therefore requires no auth.
//!
//! ## Security
//!
//! The resolved API key is treated as a secret: it is never logged (telemetry
//! must never emit it), never included in error messages, and comparison is a
//! straightforward equality check on the presented token.

use std::sync::Arc;

use aws_config::SdkConfig;
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

use crate::error::AppError;

/// The configured API key inputs, mirroring the three legacy environment
/// variables (`API_KEY_PARAM_NAME`, `API_KEY_SECRET_ARN`, `API_KEY`).
///
/// Splitting these three optional inputs out of `AppSettings` lets us unit-test
/// the **pure priority decision** ([`decide_key_source`]) without constructing
/// a full settings object or touching AWS.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApiKeyInputs {
    /// SSM Parameter Store parameter name (highest priority).
    pub param_name: Option<String>,
    /// Secrets Manager secret ARN (second priority).
    pub secret_arn: Option<String>,
    /// Literal API key from env/setting (lowest priority).
    pub api_key: Option<String>,
}

impl ApiKeyInputs {
    /// Builds the inputs from the relevant [`AppSettings`] fields.
    ///
    /// [`AppSettings`]: crate::config::AppSettings
    pub fn from_settings(settings: &crate::config::AppSettings) -> Self {
        Self {
            param_name: settings.api_key_param_name.clone(),
            secret_arn: settings.api_key_secret_arn.clone(),
            api_key: settings.api_key.clone(),
        }
    }
}

/// The resolved source for the API key, chosen purely from the configured
/// inputs and independent of any AWS call.
///
/// Resolving the *source* separately from performing the AWS fetch keeps the
/// priority logic unit-testable: [`decide_key_source`] is a pure function, and
/// [`resolve_api_key`] layers the (un-testable-without-AWS) network calls on
/// top of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeySource {
    /// Fetch from SSM Parameter Store using this parameter name.
    SsmParameter(String),
    /// Fetch from Secrets Manager using this secret ARN.
    SecretsManager(String),
    /// Use this literal key value directly.
    Literal(String),
}

/// Pure priority decision: SSM param > Secrets Manager > literal key.
///
/// Returns the [`KeySource`] to use, or [`AppError::Internal`] when none of the
/// three inputs is configured (matching the legacy Python `RuntimeError`
/// "API Key is not configured."). Empty strings are treated as "set" here to
/// mirror the Python truthiness only when non-empty; we additionally guard
/// against empty values so a blank env var does not silently select an empty
/// key.
pub fn decide_key_source(inputs: &ApiKeyInputs) -> Result<KeySource, AppError> {
    if let Some(param) = non_empty(&inputs.param_name) {
        // Backward compatibility tier (Python preferred Secrets Manager but
        // checked the SSM parameter first); preserved verbatim.
        return Ok(KeySource::SsmParameter(param));
    }
    if let Some(arn) = non_empty(&inputs.secret_arn) {
        return Ok(KeySource::SecretsManager(arn));
    }
    if let Some(key) = non_empty(&inputs.api_key) {
        return Ok(KeySource::Literal(key));
    }
    Err(AppError::Internal(
        "API Key is not configured. Please set up your API Key.".to_string(),
    ))
}

/// Returns the inner string only when present and non-empty.
fn non_empty(opt: &Option<String>) -> Option<String> {
    opt.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Resolves the API key **once at boot**, following the 3-tier priority order.
///
/// Priority (exactly as legacy Python):
/// 1. `settings.api_key_param_name` → SSM `get_parameter(WithDecryption=true)`.
/// 2. `settings.api_key_secret_arn` → Secrets Manager `get_secret_value`,
///    parse the `SecretString` JSON, extract the `"api_key"` field.
/// 3. `settings.api_key` → use the literal value.
/// 4. otherwise → [`AppError::Internal`] ("API Key is not configured...").
///
/// The caller stores the returned `String` in `AppState`; this function must
/// **not** be called per request.
pub async fn resolve_api_key(
    settings: &crate::config::AppSettings,
    aws_config: &SdkConfig,
) -> Result<String, AppError> {
    let inputs = ApiKeyInputs::from_settings(settings);
    match decide_key_source(&inputs)? {
        KeySource::SsmParameter(name) => fetch_ssm_parameter(aws_config, &name).await,
        KeySource::SecretsManager(arn) => fetch_secrets_manager(aws_config, &arn).await,
        KeySource::Literal(key) => Ok(key),
    }
}

/// Fetches a decrypted SSM Parameter Store value.
///
/// Mirrors Python `ssm.get_parameter(Name=..., WithDecryption=True)
/// ["Parameter"]["Value"]`.
async fn fetch_ssm_parameter(aws_config: &SdkConfig, name: &str) -> Result<String, AppError> {
    let client = aws_sdk_ssm::Client::new(aws_config);
    let resp = client
        .get_parameter()
        .name(name)
        .with_decryption(true)
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("failed to read SSM parameter: {e}")))?;

    resp.parameter()
        .and_then(|p| p.value())
        .map(str::to_string)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            AppError::Internal("SSM parameter did not contain a value for the API Key.".to_string())
        })
}

/// Fetches and parses an API key from a Secrets Manager secret.
///
/// Mirrors Python `json.loads(sm.get_secret_value(SecretId=...)
/// ["SecretString"])["api_key"]`, including the two distinct error messages
/// (unable to retrieve vs. missing `api_key` field).
async fn fetch_secrets_manager(aws_config: &SdkConfig, arn: &str) -> Result<String, AppError> {
    let client = aws_sdk_secretsmanager::Client::new(aws_config);
    let resp = client
        .get_secret_value()
        .secret_id(arn)
        .send()
        .await
        .map_err(|_| {
            AppError::Internal(
                "Unable to retrieve API KEY, please ensure the secret ARN is correct".to_string(),
            )
        })?;

    let secret_string = resp.secret_string().ok_or_else(|| {
        AppError::Internal(
            "Unable to retrieve API KEY, please ensure the secret ARN is correct".to_string(),
        )
    })?;

    extract_api_key_field(secret_string)
}

/// Parses a Secrets Manager `SecretString` JSON and extracts the `"api_key"`
/// field.
///
/// Pulled out as a pure helper so the JSON-parsing branch is unit-testable
/// without a live Secrets Manager call.
fn extract_api_key_field(secret_string: &str) -> Result<String, AppError> {
    let value: serde_json::Value = serde_json::from_str(secret_string).map_err(|_| {
        AppError::Internal(
            "Unable to retrieve API KEY, please ensure the secret ARN is correct".to_string(),
        )
    })?;

    value
        .get("api_key")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            AppError::Internal("Please ensure the secret contains a \"api_key\" field".to_string())
        })
}

/// Bearer-authentication middleware.
///
/// Compares the request's `Authorization: Bearer <token>` header against the
/// resolved API key (provided as shared state). On any mismatch — missing
/// header, wrong scheme, or wrong token — it returns [`AppError::Unauthorized`]
/// which renders the full OpenAI `401` error envelope.
///
/// Wire this with `route_layer` over the protected route subtree, e.g.:
///
/// ```ignore
/// use axum::{middleware, Router, routing::get};
/// use std::sync::Arc;
/// use bedrock_gateway_rust::server::auth::require_bearer;
///
/// let key = Arc::new("secret".to_string());
/// let protected = Router::new()
///     .route("/chat/completions", get(handler))
///     .route_layer(middleware::from_fn_with_state(key, require_bearer));
/// // `/health` lives outside `protected`, so it needs no auth.
/// ```
pub async fn require_bearer(
    State(api_key): State<Arc<String>>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let presented = bearer_token(&request);
    match presented {
        Some(token) if token == api_key.as_str() => Ok(next.run(request).await),
        _ => Err(AppError::Unauthorized),
    }
}

/// Extracts the bearer token from the `Authorization` header, if present and
/// well-formed (`Bearer <token>`, scheme case-insensitive).
fn bearer_token(request: &Request) -> Option<String> {
    let header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    let (scheme, token) = header.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        let token = token.trim();
        if token.is_empty() {
            None
        } else {
            Some(token.to_string())
        }
    } else {
        None
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
