//! AWS configuration and Amazon Bedrock runtime/control-plane client setup.
//!
//! This module mirrors the botocore configuration used by the legacy Python
//! gateway (`.legacy-python/src/api/models/bedrock.py`):
//!
//! - **Connect timeout / read timeout** — driven by [`AppSettings`] (defaults
//!   60s connect, 900s read for long streaming responses).
//! - **Retries** — adaptive mode with a configurable maximum attempt count
//!   (default 8), matching botocore's `{"max_attempts": 8, "mode": "adaptive"}`.
//!
//! Unlike the Python implementation — which kept a per-region cache of
//! `bedrock-runtime` clients — the AWS SDK for Rust lets us keep **one** shared
//! client and apply a per-request region override via
//! `.customize().config_override(...)`. See [`region_config_override`] for the
//! encapsulated helper and the documented call-site pattern.
//!
//! Credential resolution is the SDK default chain (environment → profile →
//! IMDS/ECS) and is **lazy**: constructing the config/clients here does not
//! require live credentials, so the gateway never panics at startup when
//! credentials are absent. Credentials are resolved at request time.

use std::time::Duration;

use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::config::Token;
use aws_smithy_runtime_api::client::auth::http::HTTP_BEARER_AUTH_SCHEME_ID;
use aws_smithy_types::retry::RetryConfig;
use aws_smithy_types::timeout::TimeoutConfig;

use crate::config::AppSettings;

/// Build the shared [`aws_config::SdkConfig`] used to construct every Bedrock
/// client, applying botocore-parity timeouts and adaptive retry pulled from
/// [`AppSettings`].
///
/// All values (region, connect/read timeouts, max retry attempts) come from
/// settings — nothing is hardcoded. Credential resolution is lazy, so this is
/// safe to call at startup even when no credentials are configured.
pub async fn build_aws_config(settings: &AppSettings) -> aws_config::SdkConfig {
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(settings.aws_connect_timeout_secs))
        .read_timeout(Duration::from_secs(settings.aws_read_timeout_secs))
        .build();

    let retry_config = RetryConfig::adaptive().with_max_attempts(settings.aws_max_retry_attempts);

    let mut loader = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(settings.aws_region.clone()))
        .retry_config(retry_config)
        .timeout_config(timeout_config);

    // Upstream-Bedrock auth. An explicit `bedrock_api_key` injects an HTTP
    // bearer token and pins the bearer auth scheme, replacing SigV4 for this
    // config. With no explicit value we touch nothing: aws-config still detects
    // `AWS_BEARER_TOKEN_BEDROCK` from the process env on its own, and otherwise
    // falls back to the standard SigV4 chain — so a single non-branching path
    // covers env-bearer, explicit-bearer, and SigV4.
    if let Some(token) = settings.bedrock_api_key.as_deref() {
        loader = loader
            .token_provider(Token::new(token.to_string(), None))
            .auth_scheme_preference([HTTP_BEARER_AUTH_SCHEME_ID]);
    }

    loader.load().await
}

/// Shared Amazon Bedrock clients built from a single [`aws_config::SdkConfig`].
///
/// Holds **one** `bedrock-runtime` client (model inference: converse,
/// converse_stream, invoke_model, embeddings) and **one** `bedrock`
/// control-plane client (e.g. listing foundation models). Both are cheap to
/// clone (internally reference-counted) and safe to share across requests.
///
/// Per-request region routing is handled with [`region_config_override`]
/// applied at the call site — do **not** build additional per-region clients.
#[derive(Clone, Debug)]
pub struct BedrockClients {
    /// `bedrock-runtime` client for model inference operations.
    pub runtime: aws_sdk_bedrockruntime::Client,
    /// `bedrock` control-plane client for management operations.
    pub control: aws_sdk_bedrock::Client,
}

impl BedrockClients {
    /// Construct both Bedrock clients from a shared [`aws_config::SdkConfig`].
    ///
    /// This does not perform any network or credential resolution; it only
    /// wires up the clients. Credentials resolve lazily on first request.
    pub fn new(sdk_config: &aws_config::SdkConfig) -> Self {
        Self {
            runtime: aws_sdk_bedrockruntime::Client::new(sdk_config),
            control: aws_sdk_bedrock::Client::new(sdk_config),
        }
    }

    /// Convenience constructor that builds the shared [`aws_config::SdkConfig`]
    /// from [`AppSettings`] and then the clients.
    pub async fn from_settings(settings: &AppSettings) -> Self {
        let sdk_config = build_aws_config(settings).await;
        Self::new(&sdk_config)
    }
}

/// Build a `bedrock-runtime` config override that targets a different AWS
/// region for a single request, without constructing a new client.
///
/// This encapsulates the botocore-parity per-region routing the Python gateway
/// previously achieved with a per-region client cache (`_get_runtime_client`).
/// In the Rust SDK we keep one shared [`BedrockClients::runtime`] and override
/// only the region per call.
///
/// # Call-site pattern
///
/// Apply the override to any operation builder via `.customize()`:
///
/// ```no_run
/// # use bedrock_gateway_rust::bedrock::client::{region_config_override, BedrockClients};
/// # async fn example(clients: &BedrockClients) -> Result<(), Box<dyn std::error::Error>> {
/// let response = clients
///     .runtime
///     .converse()
///     .model_id("anthropic.claude-3-5-sonnet-20241022-v2:0")
///     .customize()
///     .config_override(region_config_override("us-east-1"))
///     .send()
///     .await?;
/// # let _ = response;
/// # Ok(())
/// # }
/// ```
///
/// The override is consumed by `.config_override(...)` which accepts anything
/// `Into<aws_sdk_bedrockruntime::config::Builder>`.
pub fn region_config_override(
    region: impl Into<String>,
) -> aws_sdk_bedrockruntime::config::Builder {
    aws_sdk_bedrockruntime::config::Builder::new().region(Region::new(region.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_smithy_types::retry::RetryMode;

    /// Build an `AppSettings` with explicit AWS timeout/retry/region values for
    /// construction-level tests. Avoids depending on env/file config loading.
    fn test_settings(region: &str, connect: u64, read: u64, attempts: u32) -> AppSettings {
        AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: region.to_string(),
            default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
            enable_cross_region_inference: true,
            enable_application_inference_profiles: true,
            enable_prompt_caching: false,
            api_key: None,
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: None,
            disable_mantle: false,
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: connect,
            aws_read_timeout_secs: read,
            aws_max_retry_attempts: attempts,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
            allowed_models: None,
        }
    }

    #[tokio::test]
    async fn build_aws_config_applies_settings_without_panic() {
        // read_timeout=60, max_attempts=3 per task spec.
        let settings = test_settings("us-east-2", 30, 60, 3);

        let sdk_config = build_aws_config(&settings).await;

        // Region is taken from settings.
        assert_eq!(
            sdk_config.region().map(Region::as_ref),
            Some("us-east-2"),
            "region should come from AppSettings"
        );

        // Timeouts are derived from settings (no hardcoded values).
        let timeout = sdk_config
            .timeout_config()
            .expect("timeout_config should be set");
        assert_eq!(timeout.connect_timeout(), Some(Duration::from_secs(30)));
        assert_eq!(timeout.read_timeout(), Some(Duration::from_secs(60)));

        // Adaptive retry with the configured max attempts.
        let retry = sdk_config
            .retry_config()
            .expect("retry_config should be set");
        assert_eq!(retry.max_attempts(), 3);
        assert_eq!(
            retry.mode(),
            RetryMode::Adaptive,
            "retry mode should be adaptive"
        );
    }

    #[tokio::test]
    async fn build_clients_from_config_does_not_require_live_creds() {
        // Construction must succeed even without credentials available.
        let settings = test_settings("us-west-2", 60, 900, 8);
        let sdk_config = build_aws_config(&settings).await;

        // Constructing clients must not panic / not resolve credentials.
        let clients = BedrockClients::new(&sdk_config);
        // Clients are clonable & shareable.
        let _runtime = clients.runtime.clone();
        let _control = clients.control.clone();
    }

    #[tokio::test]
    async fn explicit_bedrock_token_builds_and_sets_bearer_preference() {
        let mut settings = test_settings("us-east-2", 60, 900, 8);
        settings.bedrock_api_key = Some("test-bedrock-bearer-token".to_string());

        let sdk_config = build_aws_config(&settings).await;

        // Building clients from a bearer-token config must not panic; credential
        // resolution stays lazy (deferred to request time).
        let clients = BedrockClients::new(&sdk_config);
        let _runtime = clients.runtime.clone();

        // The bearer auth scheme must be preferred when a token is injected.
        let preference = sdk_config
            .auth_scheme_preference()
            .expect("auth_scheme_preference should be set when bedrock_api_key is present");
        assert!(
            preference
                .clone()
                .into_iter()
                .any(|id| id == HTTP_BEARER_AUTH_SCHEME_ID),
            "bearer auth scheme should be present in the preference list"
        );
    }

    #[tokio::test]
    async fn no_bedrock_token_leaves_sigv4_path_untouched() {
        // Without an explicit token, no bearer preference is pinned; the SDK is
        // free to use AWS_BEARER_TOKEN_BEDROCK from env or fall back to SigV4.
        // Construction must not panic and must not eagerly resolve credentials.
        let settings = test_settings("us-west-2", 60, 900, 8);
        assert!(settings.bedrock_api_key.is_none());

        let sdk_config = build_aws_config(&settings).await;
        let clients = BedrockClients::new(&sdk_config);
        let _runtime = clients.runtime.clone();
        let _control = clients.control.clone();

        assert!(
            sdk_config.auth_scheme_preference().is_none(),
            "no auth scheme preference should be pinned without an explicit token"
        );
    }

    #[test]
    fn region_override_produces_builder_for_target_region() {
        // The helper must encode the requested override region, distinct from
        // the home region, without constructing a new client.
        let override_builder = region_config_override("eu-west-1");
        let built = override_builder.build();

        assert_eq!(
            built.region().map(Region::as_ref),
            Some("eu-west-1"),
            "config override must carry the requested region"
        );
    }

    #[test]
    fn region_override_differs_from_home_region() {
        let home = region_config_override("us-west-2").build();
        let routed = region_config_override("ap-northeast-1").build();

        assert_ne!(
            home.region().map(Region::as_ref),
            routed.region().map(Region::as_ref),
            "routed override region must differ from home region"
        );
        assert_eq!(routed.region().map(Region::as_ref), Some("ap-northeast-1"));
    }
}
