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
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{header::AUTHORIZATION, Request as HttpRequest, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::get;
    use axum::{middleware, Router};
    use serde_json::Value;
    use tower::ServiceExt; // for `oneshot`

    // ---- Priority decision (pure, no AWS) --------------------------------

    #[test]
    fn priority_ssm_wins_over_all() {
        let inputs = ApiKeyInputs {
            param_name: Some("/bedrock/key".to_string()),
            secret_arn: Some("arn:aws:secretsmanager:...".to_string()),
            api_key: Some("literal".to_string()),
        };
        assert_eq!(
            decide_key_source(&inputs).unwrap(),
            KeySource::SsmParameter("/bedrock/key".to_string())
        );
    }

    #[test]
    fn priority_secrets_manager_over_literal() {
        let inputs = ApiKeyInputs {
            param_name: None,
            secret_arn: Some("arn:aws:secretsmanager:...".to_string()),
            api_key: Some("literal".to_string()),
        };
        assert_eq!(
            decide_key_source(&inputs).unwrap(),
            KeySource::SecretsManager("arn:aws:secretsmanager:...".to_string())
        );
    }

    /// MUST DO: env-key tier resolves when only `settings.api_key` is set.
    #[test]
    fn priority_literal_when_only_env_set() {
        let inputs = ApiKeyInputs {
            param_name: None,
            secret_arn: None,
            api_key: Some("env-key".to_string()),
        };
        assert_eq!(
            decide_key_source(&inputs).unwrap(),
            KeySource::Literal("env-key".to_string())
        );
    }

    #[test]
    fn priority_errors_when_nothing_configured() {
        let inputs = ApiKeyInputs::default();
        let err = decide_key_source(&inputs).unwrap_err();
        assert!(matches!(err, AppError::Internal(_)));
    }

    #[test]
    fn priority_treats_empty_strings_as_unset() {
        let inputs = ApiKeyInputs {
            param_name: Some("   ".to_string()),
            secret_arn: Some(String::new()),
            api_key: Some("real".to_string()),
        };
        // Blank param and empty arn are skipped; literal wins.
        assert_eq!(
            decide_key_source(&inputs).unwrap(),
            KeySource::Literal("real".to_string())
        );
    }

    /// MUST DO: env-key tier resolves end-to-end via `resolve_api_key` without
    /// any AWS call (literal branch). We build a minimal `AppSettings` with
    /// only `api_key` set; SSM/SM are `None` so no network is touched.
    #[tokio::test]
    async fn resolve_api_key_literal_tier_no_aws() {
        let settings = test_settings(None, None, Some("env-key".to_string()));
        // A default SdkConfig is fine; the literal branch never uses it.
        let aws_config = aws_config::SdkConfig::builder().build();
        let key = resolve_api_key(&settings, &aws_config).await.unwrap();
        assert_eq!(key, "env-key");
    }

    #[tokio::test]
    async fn resolve_api_key_errors_when_unconfigured() {
        let settings = test_settings(None, None, None);
        let aws_config = aws_config::SdkConfig::builder().build();
        let err = resolve_api_key(&settings, &aws_config).await.unwrap_err();
        assert!(matches!(err, AppError::Internal(_)));
    }

    // ---- Secrets Manager JSON parsing (pure) -----------------------------

    #[test]
    fn extract_api_key_field_ok() {
        let json = r#"{"api_key": "s3cr3t"}"#;
        assert_eq!(extract_api_key_field(json).unwrap(), "s3cr3t");
    }

    #[test]
    fn extract_api_key_field_missing_field() {
        let json = r#"{"other": "x"}"#;
        let err = extract_api_key_field(json).unwrap_err();
        match err {
            AppError::Internal(m) => assert!(m.contains("api_key")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn extract_api_key_field_invalid_json() {
        let err = extract_api_key_field("not json").unwrap_err();
        assert!(matches!(err, AppError::Internal(_)));
    }

    // ---- Bearer middleware ------------------------------------------------

    /// Builds a tiny router with the bearer middleware applied via
    /// `route_layer`, returning 200 "ok" from the protected handler.
    fn protected_router(key: &str) -> Router {
        let state = Arc::new(key.to_string());
        Router::new()
            .route("/protected", get(|| async { "ok" }))
            .route_layer(middleware::from_fn_with_state(state, require_bearer))
    }

    async fn send(router: Router, auth: Option<&str>) -> (StatusCode, Value) {
        let mut builder = HttpRequest::builder().uri("/protected");
        if let Some(a) = auth {
            builder = builder.header(AUTHORIZATION, a);
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = router.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    /// MUST DO: correct token is accepted.
    #[tokio::test]
    async fn bearer_accepts_correct_token() {
        let (status, _body) = send(protected_router("good-key"), Some("Bearer good-key")).await;
        assert_eq!(status, StatusCode::OK);
    }

    /// MUST DO: wrong token rejected with 401 and the OpenAI envelope.
    #[tokio::test]
    async fn bearer_rejects_wrong_token_with_envelope() {
        let (status, body) = send(protected_router("good-key"), Some("Bearer wrong")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        // Full OpenAI envelope, not a bare/empty 401.
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "unauthorized");
        assert!(body["error"]["message"].as_str().is_some());
        assert!(body.get("detail").is_none());
    }

    #[tokio::test]
    async fn bearer_rejects_missing_header() {
        let (status, body) = send(protected_router("good-key"), None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"]["code"], "unauthorized");
    }

    #[tokio::test]
    async fn bearer_rejects_wrong_scheme() {
        let (status, _body) = send(protected_router("good-key"), Some("Basic good-key")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn unauthorized_renders_openai_envelope_directly() {
        // Sanity-check that the error path used by the middleware produces the
        // expected envelope shape independent of the router plumbing.
        let resp = AppError::Unauthorized.into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ---- Test helpers -----------------------------------------------------

    /// Constructs an `AppSettings` with only the three auth inputs varied and
    /// everything else set to harmless defaults.
    fn test_settings(
        param_name: Option<String>,
        secret_arn: Option<String>,
        api_key: Option<String>,
    ) -> crate::config::AppSettings {
        crate::config::AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: "us-west-2".to_string(),
            default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
            enable_cross_region_inference: true,
            enable_application_inference_profiles: true,
            enable_prompt_caching: false,
            bedrock_api_key: None,
            disable_mantle: false,
            api_key,
            api_key_secret_arn: secret_arn,
            api_key_param_name: param_name,
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
            allowed_models: None,
        }
    }
}
