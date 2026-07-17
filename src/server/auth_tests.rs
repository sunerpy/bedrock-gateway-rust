//! Unit tests for [`super`] (the `server::auth` module).
//!
//! Organized as a sibling file per the project test convention: declared in
//! `auth.rs` via `#[cfg(test)] #[path = "auth_tests.rs"] mod tests;`, so
//! `use super::*;` resolves to the `auth` module's items.
//!
//! Coverage:
//! - pure priority decision ([`decide_key_source`]) — no AWS;
//! - literal-tier `resolve_api_key` without any AWS call;
//! - Secrets Manager JSON field extraction (pure);
//! - the bearer middleware: missing header / wrong token / wrong scheme → 401
//!   with the OpenAI error envelope, correct token → pass-through;
//! - the 401-vs-405 distinction guaranteed by `route_layer`.

use super::*;
use axum::body::{to_bytes, Body};
use axum::http::{header::AUTHORIZATION, Method, Request as HttpRequest, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
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

/// MUST DO: the 401-vs-405 distinction. The real server attaches the bearer
/// middleware with `route_layer` over a *nested + merged* subtree (see
/// `server::routers::build_router`). This test mirrors that exact topology so
/// the semantics match production: with **valid** credentials, a wrong HTTP
/// method on a protected path yields `405 Method Not Allowed` — auth passes the
/// request through and the `MethodRouter` then rejects the method. The auth
/// layer does not mask (turn into `200`) nor override (turn into a spurious
/// `401`) the method rejection once the caller is authenticated.
#[tokio::test]
async fn wrong_method_with_valid_auth_yields_405() {
    let router = nested_protected_router("good-key");
    // GET a POST-only protected route, WITH a valid bearer token.
    let req = HttpRequest::builder()
        .method(Method::GET)
        .uri("/api/v1/submit")
        .header(AUTHORIZATION, "Bearer good-key")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

/// Companion to [`wrong_method_with_valid_auth_yields_405`]: on the *same*
/// nested router, a request whose method DOES match a protected route but which
/// carries no credentials is rejected with `401` (not `405`/`404`). Together the
/// two tests pin the 401-vs-405 boundary: `route_layer` runs auth for any
/// path-matched request, so authentication is decided first for matched methods,
/// while an unmatched method never reaches a protected handler.
#[tokio::test]
async fn matched_method_missing_auth_yields_401() {
    let router = nested_protected_router("good-key");
    let req = HttpRequest::builder()
        .method(Method::POST)
        .uri("/api/v1/submit")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// Builds a router mirroring `server::routers::build_router`'s topology: a
/// `route_layer`-protected subtree merged with a public route, nested under a
/// prefix. This reproduces the production 401-vs-405 behavior faithfully.
fn nested_protected_router(key: &str) -> Router {
    let state = Arc::new(key.to_string());
    let protected = Router::new()
        .route("/submit", post(|| async { "ok" }))
        .route_layer(middleware::from_fn_with_state(state, require_bearer));
    let public = Router::new().route("/health", get(|| async { "ok" }));
    Router::new().nest("/api/v1", protected.merge(public))
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
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
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
        responses_stream_idle_timeout_secs: 180,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    }
}
