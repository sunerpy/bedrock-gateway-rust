use anyhow::Result;
use config::builder::DefaultState;
use config::{Config, ConfigBuilder as RawConfigBuilder, Environment, File};
use serde::{Deserialize, Serialize};

type ConfigBuilder = RawConfigBuilder<DefaultState>;

/// Application settings with layered configuration loading:
/// defaults → optional file (config/app.toml) → environment variable overrides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// API route prefix (default: "/api/v1")
    pub api_route_prefix: String,

    /// Debug mode flag (default: false)
    pub debug: bool,

    /// AWS region (default: "us-west-2")
    pub aws_region: String,

    /// Default model ID (default: current Claude model)
    /// Note: Python default "anthropic.claude-3-sonnet-20240229-v1:0" is outdated.
    /// Using "anthropic.claude-3-5-sonnet-20241022-v2:0" (latest stable).
    pub default_model: String,

    /// Default embedding model (default: "cohere.embed-multilingual-v3")
    pub default_embedding_model: String,

    /// Enable cross-region inference (default: true)
    pub enable_cross_region_inference: bool,

    /// Enable application inference profiles (default: true)
    pub enable_application_inference_profiles: bool,

    /// Enable prompt caching (default: false)
    pub enable_prompt_caching: bool,

    /// Optional API key (env: API_KEY)
    pub api_key: Option<String>,

    /// Optional API key secret ARN (env: API_KEY_SECRET_ARN)
    pub api_key_secret_arn: Option<String>,

    /// Optional API key parameter name (env: API_KEY_PARAM_NAME)
    pub api_key_param_name: Option<String>,

    /// Upstream-Bedrock bearer token the gateway presents *to* AWS Bedrock
    /// (gateway → Bedrock). Env: `AWS_BEARER_TOKEN_BEDROCK` (AWS standard name),
    /// or the `BEDROCK_API_KEY` alias. When set it is injected as an HTTP bearer
    /// token in `bedrock::client::build_aws_config`, replacing the SigV4 chain;
    /// when unset the SDK falls back to the standard SigV4 chain (access
    /// key/secret, profile, IMDS, ECS task role) and reads
    /// `AWS_BEARER_TOKEN_BEDROCK` from the process env on its own.
    ///
    /// Distinct from [`AppSettings::api_key`], which is the token *clients*
    /// present *to the gateway* (client → gateway). The two are unrelated.
    pub bedrock_api_key: Option<String>,

    /// Disable the bedrock-mantle backend and startup validation (default: false).
    pub disable_mantle: bool,

    /// Server bind address (default: "0.0.0.0")
    pub bind_addr: String,

    /// Server port (default: 8080)
    pub port: u16,

    /// Log level (default: "info")
    pub log_level: String,

    /// AWS connection timeout in seconds (default: 60, from botocore parity)
    pub aws_connect_timeout_secs: u64,

    /// AWS read timeout in seconds (default: 900, from botocore parity)
    pub aws_read_timeout_secs: u64,

    /// AWS maximum retry attempts (default: 8, from botocore parity)
    pub aws_max_retry_attempts: u32,

    /// URL template for the bedrock-mantle OpenAI-compatible upstream, with a
    /// `{region}` placeholder substituted per request. Default:
    /// `https://bedrock-mantle.{region}.api.aws/openai/v1`. Env:
    /// `MANTLE_BASE_URL_TEMPLATE` (or `APP_MANTLE_BASE_URL_TEMPLATE`).
    pub mantle_base_url_template: String,
}

impl AppSettings {
    /// Load configuration from layered sources:
    /// 1. Built-in defaults (hardcoded in code)
    /// 2. Optional file: `config/app.toml` (if present)
    /// 3. Environment variables (override all previous layers)
    pub fn load() -> Result<Self> {
        let mut builder = Config::builder()
            .set_default("api_route_prefix", "/api/v1")?
            .set_default("debug", false)?
            .set_default("aws_region", "us-west-2")?
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")?
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")?
            .set_default("enable_cross_region_inference", true)?
            .set_default("enable_application_inference_profiles", true)?
            // Master switch: TRUE so clients that never send
            // `extra_body.prompt_caching` still get transparent cache hits on
            // supported models. ENABLE_PROMPT_CACHING=false disables all;
            // per-request `extra_body.prompt_caching` overrides either way.
            .set_default("enable_prompt_caching", true)?
            .set_default("disable_mantle", false)?
            .set_default("bind_addr", "0.0.0.0")?
            .set_default("port", 8080)?
            .set_default("log_level", "info")?
            .set_default("aws_connect_timeout_secs", 60u64)?
            .set_default("aws_read_timeout_secs", 900u64)?
            .set_default("aws_max_retry_attempts", 8u32)?
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )?
            .add_source(File::with_name("config/app").required(false))
            .add_source(
                Environment::with_prefix("APP")
                    .separator("_")
                    .try_parsing(true),
            );

        // Overlay Python-parity BARE env names on top of the APP_ layer. The
        // reference gateway and deployment templates set un-prefixed names, so
        // standard names must win. An explicit allow-list is used (not a blanket
        // un-prefixed Environment source) to avoid greedily capturing unrelated
        // process env such as HOME/PATH. AWS_REGION feeds the explicit
        // .region(...) call in bedrock::client; the SDK credential chain reads
        // it independently too.
        builder = apply_bare_env_overrides(builder)?;

        let settings: AppSettings = builder.build()?.try_deserialize()?;
        Ok(settings)
    }
}

/// Overlay the documented Python-parity bare environment variables onto the
/// config builder, each as a typed override applied only when the variable is
/// present and non-empty.
///
/// Bare names honored (parity with the Python gateway + deployment templates):
/// `API_ROUTE_PREFIX`, `AWS_REGION`, `DEBUG`, `DEFAULT_MODEL`,
/// `DEFAULT_EMBEDDING_MODEL`, `ENABLE_CROSS_REGION_INFERENCE`,
/// `ENABLE_APPLICATION_INFERENCE_PROFILES`, `ENABLE_PROMPT_CACHING`,
/// `DISABLE_MANTLE`, `API_KEY`, `API_KEY_SECRET_ARN`, `API_KEY_PARAM_NAME`,
/// `AWS_BEARER_TOKEN_BEDROCK` (alias `BEDROCK_API_KEY`), plus the operational
/// knobs `PORT`, `BIND_ADDR`, `LOG_LEVEL`, `MANTLE_BASE_URL_TEMPLATE`.
fn apply_bare_env_overrides(mut builder: ConfigBuilder) -> Result<ConfigBuilder> {
    // String-valued overrides.
    for (env_name, field) in [
        ("API_ROUTE_PREFIX", "api_route_prefix"),
        ("AWS_REGION", "aws_region"),
        ("DEFAULT_MODEL", "default_model"),
        ("DEFAULT_EMBEDDING_MODEL", "default_embedding_model"),
        ("API_KEY", "api_key"),
        ("API_KEY_SECRET_ARN", "api_key_secret_arn"),
        ("API_KEY_PARAM_NAME", "api_key_param_name"),
        ("BIND_ADDR", "bind_addr"),
        ("LOG_LEVEL", "log_level"),
        ("MANTLE_BASE_URL_TEMPLATE", "mantle_base_url_template"),
    ] {
        if let Some(value) = non_empty_env(env_name) {
            builder = builder.set_override(field, value)?;
        }
    }

    // Upstream-Bedrock bearer token (gateway → Bedrock). The AWS standard name
    // `AWS_BEARER_TOKEN_BEDROCK` is authoritative; `BEDROCK_API_KEY` is only a
    // fallback alias so the standard name wins when both are present.
    if let Some(value) =
        non_empty_env("AWS_BEARER_TOKEN_BEDROCK").or_else(|| non_empty_env("BEDROCK_API_KEY"))
    {
        builder = builder.set_override("bedrock_api_key", value)?;
    }

    // Boolean-valued overrides (lenient truthiness parity with Python).
    for (env_name, field) in [
        ("DEBUG", "debug"),
        (
            "ENABLE_CROSS_REGION_INFERENCE",
            "enable_cross_region_inference",
        ),
        (
            "ENABLE_APPLICATION_INFERENCE_PROFILES",
            "enable_application_inference_profiles",
        ),
        ("ENABLE_PROMPT_CACHING", "enable_prompt_caching"),
        ("DISABLE_MANTLE", "disable_mantle"),
    ] {
        if let Some(value) = non_empty_env(env_name) {
            builder = builder.set_override(field, parse_bool(&value))?;
        }
    }

    // Integer-valued overrides. Each parses leniently: a non-numeric value is
    // silently ignored (the lower-priority layer wins), never a panic — matching
    // the historical `PORT` behavior.
    for (env_name, field) in [
        ("PORT", "port"),
        ("AWS_CONNECT_TIMEOUT_SECS", "aws_connect_timeout_secs"),
        ("AWS_READ_TIMEOUT_SECS", "aws_read_timeout_secs"),
        ("AWS_MAX_RETRY_ATTEMPTS", "aws_max_retry_attempts"),
    ] {
        if let Some(value) = non_empty_env(env_name) {
            if let Ok(parsed) = value.parse::<i64>() {
                builder = builder.set_override(field, parsed)?;
            }
        }
    }

    Ok(builder)
}

/// Read an environment variable, returning `Some` only when present and
/// non-empty (after trimming).
fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Lenient boolean parse matching common truthy spellings (`true`/`1`/`yes`/
/// `on`, case-insensitive). Anything else is `false`.
fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_GUARD: Mutex<()> = Mutex::new(());

    const TIMEOUT_RETRY_VARS: [&str; 4] = [
        "AWS_CONNECT_TIMEOUT_SECS",
        "AWS_READ_TIMEOUT_SECS",
        "AWS_MAX_RETRY_ATTEMPTS",
        "PORT",
    ];

    fn clear_timeout_retry_vars() {
        for name in TIMEOUT_RETRY_VARS {
            std::env::remove_var(name);
        }
    }

    fn minimal_builder() -> ConfigBuilder {
        Config::builder()
            .set_default("api_route_prefix", "/api/v1")
            .unwrap()
            .set_default("debug", false)
            .unwrap()
            .set_default("aws_region", "us-west-2")
            .unwrap()
            .set_default("default_model", "fallback-model")
            .unwrap()
            .set_default("default_embedding_model", "fallback-embed")
            .unwrap()
            .set_default("enable_cross_region_inference", true)
            .unwrap()
            .set_default("enable_application_inference_profiles", true)
            .unwrap()
            .set_default("enable_prompt_caching", false)
            .unwrap()
            .set_default("disable_mantle", false)
            .unwrap()
            .set_default("bind_addr", "0.0.0.0")
            .unwrap()
            .set_default("port", 8080)
            .unwrap()
            .set_default("log_level", "info")
            .unwrap()
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap()
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap()
            .set_default("aws_max_retry_attempts", 8u32)
            .unwrap()
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap()
    }

    #[test]
    fn test_defaults_load_without_env_or_file() {
        let config = Config::builder()
            .set_default("api_route_prefix", "/api/v1")
            .unwrap()
            .set_default("debug", false)
            .unwrap()
            .set_default("aws_region", "us-west-2")
            .unwrap()
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            .unwrap()
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")
            .unwrap()
            .set_default("enable_cross_region_inference", true)
            .unwrap()
            .set_default("enable_application_inference_profiles", true)
            .unwrap()
            .set_default("enable_prompt_caching", true)
            .unwrap()
            .set_default("disable_mantle", false)
            .unwrap()
            .set_default("bind_addr", "0.0.0.0")
            .unwrap()
            .set_default("port", 8080)
            .unwrap()
            .set_default("log_level", "info")
            .unwrap()
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap()
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap()
            .set_default("aws_max_retry_attempts", 8u32)
            .unwrap()
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap()
            .build()
            .unwrap();

        let settings: AppSettings = config.try_deserialize().unwrap();

        assert_eq!(settings.api_route_prefix, "/api/v1");
        assert!(!settings.debug);
        assert_eq!(settings.aws_region, "us-west-2");
        assert_eq!(
            settings.default_model,
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        );
        assert_eq!(
            settings.default_embedding_model,
            "cohere.embed-multilingual-v3"
        );
        assert!(settings.enable_cross_region_inference);
        assert!(settings.enable_application_inference_profiles);
        assert!(settings.enable_prompt_caching);
        assert!(!settings.disable_mantle);
        assert_eq!(settings.bind_addr, "0.0.0.0");
        assert_eq!(settings.port, 8080);
        assert_eq!(settings.log_level, "info");
        assert_eq!(settings.aws_connect_timeout_secs, 60);
        assert_eq!(settings.aws_read_timeout_secs, 900);
        assert_eq!(settings.aws_max_retry_attempts, 8);
    }

    #[test]
    fn test_config_file_override() {
        let mut builder = Config::builder();

        builder = builder.set_default("api_route_prefix", "/api/v1").unwrap();
        builder = builder.set_default("debug", false).unwrap();
        builder = builder.set_default("aws_region", "us-west-2").unwrap();
        builder = builder.set_default("port", 8080).unwrap();
        builder = builder.set_default("log_level", "info").unwrap();
        builder = builder.set_default("enable_prompt_caching", false).unwrap();
        builder = builder.set_default("disable_mantle", false).unwrap();
        builder = builder.set_default("bind_addr", "0.0.0.0").unwrap();
        builder = builder
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            .unwrap();
        builder = builder
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")
            .unwrap();
        builder = builder
            .set_default("enable_cross_region_inference", true)
            .unwrap();
        builder = builder
            .set_default("enable_application_inference_profiles", true)
            .unwrap();
        builder = builder
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap();
        builder = builder
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap();
        builder = builder.set_default("aws_max_retry_attempts", 8u32).unwrap();
        builder = builder
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap();

        builder = builder.set_override("api_route_prefix", "/v2").unwrap();
        builder = builder.set_override("debug", true).unwrap();
        builder = builder.set_override("aws_region", "eu-west-1").unwrap();
        builder = builder.set_override("port", 3000).unwrap();
        builder = builder.set_override("log_level", "debug").unwrap();
        builder = builder.set_override("enable_prompt_caching", true).unwrap();

        let config = builder.build().unwrap();
        let settings: AppSettings = config.try_deserialize().unwrap();

        assert_eq!(settings.api_route_prefix, "/v2");
        assert!(settings.debug);
        assert_eq!(settings.aws_region, "eu-west-1");
        assert_eq!(settings.port, 3000);
        assert_eq!(settings.log_level, "debug");
        assert!(settings.enable_prompt_caching);
    }

    #[test]
    fn test_optional_fields_none() {
        let config = Config::builder()
            .set_default("api_route_prefix", "/api/v1")
            .unwrap()
            .set_default("debug", false)
            .unwrap()
            .set_default("aws_region", "us-west-2")
            .unwrap()
            .set_default("bind_addr", "0.0.0.0")
            .unwrap()
            .set_default("port", 8080)
            .unwrap()
            .set_default("log_level", "info")
            .unwrap()
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            .unwrap()
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")
            .unwrap()
            .set_default("enable_cross_region_inference", true)
            .unwrap()
            .set_default("enable_application_inference_profiles", true)
            .unwrap()
            .set_default("enable_prompt_caching", false)
            .unwrap()
            .set_default("disable_mantle", false)
            .unwrap()
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap()
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap()
            .set_default("aws_max_retry_attempts", 8u32)
            .unwrap()
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap()
            .build()
            .unwrap();

        let settings: AppSettings = config.try_deserialize().unwrap();

        assert_eq!(settings.api_key, None);
        assert_eq!(settings.api_key_secret_arn, None);
        assert_eq!(settings.api_key_param_name, None);
        assert_eq!(settings.bedrock_api_key, None);
    }

    #[test]
    fn test_optional_fields_with_values() {
        let mut builder = Config::builder();

        builder = builder.set_default("api_route_prefix", "/api/v1").unwrap();
        builder = builder.set_default("debug", false).unwrap();
        builder = builder.set_default("aws_region", "us-west-2").unwrap();
        builder = builder.set_default("bind_addr", "0.0.0.0").unwrap();
        builder = builder.set_default("port", 8080).unwrap();
        builder = builder.set_default("log_level", "info").unwrap();
        builder = builder
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            .unwrap();
        builder = builder
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")
            .unwrap();
        builder = builder
            .set_default("enable_cross_region_inference", true)
            .unwrap();
        builder = builder
            .set_default("enable_application_inference_profiles", true)
            .unwrap();
        builder = builder.set_default("enable_prompt_caching", false).unwrap();
        builder = builder.set_default("disable_mantle", false).unwrap();
        builder = builder
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap();
        builder = builder
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap();
        builder = builder.set_default("aws_max_retry_attempts", 8u32).unwrap();
        builder = builder
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap();

        builder = builder.set_override("api_key", "test-key").unwrap();
        builder = builder
            .set_override("api_key_secret_arn", "arn:aws:secretsmanager:...")
            .unwrap();
        builder = builder
            .set_override("api_key_param_name", "/bedrock/api-key")
            .unwrap();

        let config = builder.build().unwrap();
        let settings: AppSettings = config.try_deserialize().unwrap();

        assert_eq!(settings.api_key, Some("test-key".to_string()));
        assert_eq!(
            settings.api_key_secret_arn,
            Some("arn:aws:secretsmanager:...".to_string())
        );
        assert_eq!(
            settings.api_key_param_name,
            Some("/bedrock/api-key".to_string())
        );
    }

    #[test]
    fn test_aws_timeouts_and_retries_override() {
        let mut builder = Config::builder();

        builder = builder.set_default("api_route_prefix", "/api/v1").unwrap();
        builder = builder.set_default("debug", false).unwrap();
        builder = builder.set_default("aws_region", "us-west-2").unwrap();
        builder = builder.set_default("bind_addr", "0.0.0.0").unwrap();
        builder = builder.set_default("port", 8080).unwrap();
        builder = builder.set_default("log_level", "info").unwrap();
        builder = builder
            .set_default("default_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
            .unwrap();
        builder = builder
            .set_default("default_embedding_model", "cohere.embed-multilingual-v3")
            .unwrap();
        builder = builder
            .set_default("enable_cross_region_inference", true)
            .unwrap();
        builder = builder
            .set_default("enable_application_inference_profiles", true)
            .unwrap();
        builder = builder.set_default("enable_prompt_caching", false).unwrap();
        builder = builder.set_default("disable_mantle", false).unwrap();
        builder = builder
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap();
        builder = builder
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap();
        builder = builder.set_default("aws_max_retry_attempts", 8u32).unwrap();
        builder = builder
            .set_default(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap();

        builder = builder
            .set_override("aws_connect_timeout_secs", 120u64)
            .unwrap();
        builder = builder
            .set_override("aws_read_timeout_secs", 1800u64)
            .unwrap();
        builder = builder
            .set_override("aws_max_retry_attempts", 10u32)
            .unwrap();

        let config = builder.build().unwrap();
        let settings: AppSettings = config.try_deserialize().unwrap();

        assert_eq!(settings.aws_connect_timeout_secs, 120);
        assert_eq!(settings.aws_read_timeout_secs, 1800);
        assert_eq!(settings.aws_max_retry_attempts, 10);
    }

    #[test]
    fn parse_bool_truthy_and_falsy() {
        for truthy in ["true", "TRUE", "1", "yes", "On", " on "] {
            assert!(parse_bool(truthy), "{truthy:?} should be true");
        }
        for falsy in ["false", "0", "no", "off", "", "anything"] {
            assert!(!parse_bool(falsy), "{falsy:?} should be false");
        }
    }

    #[test]
    fn disable_mantle_defaults_to_false() {
        let settings: AppSettings = minimal_builder()
            .build()
            .unwrap()
            .try_deserialize()
            .unwrap();

        assert!(!settings.disable_mantle);
    }

    #[test]
    fn bare_disable_mantle_env_var_overrides_default() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("DISABLE_MANTLE");

        std::env::set_var("DISABLE_MANTLE", "true");
        let settings = AppSettings::load().unwrap();

        assert!(settings.disable_mantle);

        std::env::remove_var("DISABLE_MANTLE");
    }

    #[test]
    fn bare_env_overrides_map_to_typed_fields() {
        let base = Config::builder()
            .set_default("api_route_prefix", "/api/v1")
            .unwrap()
            .set_default("debug", false)
            .unwrap()
            .set_default("aws_region", "us-west-2")
            .unwrap()
            .set_default("default_model", "fallback-model")
            .unwrap()
            .set_default("default_embedding_model", "fallback-embed")
            .unwrap()
            .set_default("enable_cross_region_inference", true)
            .unwrap()
            .set_default("enable_application_inference_profiles", true)
            .unwrap()
            .set_default("enable_prompt_caching", false)
            .unwrap()
            .set_default("disable_mantle", false)
            .unwrap()
            .set_default("bind_addr", "0.0.0.0")
            .unwrap()
            .set_default("port", 8080)
            .unwrap()
            .set_default("log_level", "info")
            .unwrap()
            .set_default("aws_connect_timeout_secs", 60u64)
            .unwrap()
            .set_default("aws_read_timeout_secs", 900u64)
            .unwrap()
            .set_default("aws_max_retry_attempts", 8u32)
            .unwrap();

        let overridden = base
            .set_override("aws_region", "eu-central-1")
            .unwrap()
            .set_override("api_key", "bare-key")
            .unwrap()
            .set_override("enable_prompt_caching", parse_bool("yes"))
            .unwrap()
            .set_override("port", 9000i64)
            .unwrap()
            .set_override(
                "mantle_base_url_template",
                "https://bedrock-mantle.{region}.api.aws/openai/v1",
            )
            .unwrap();

        let settings: AppSettings = overridden.build().unwrap().try_deserialize().unwrap();

        assert_eq!(settings.aws_region, "eu-central-1");
        assert_eq!(settings.api_key, Some("bare-key".to_string()));
        assert!(settings.enable_prompt_caching);
        assert_eq!(settings.port, 9000);
    }

    #[test]
    fn bedrock_api_key_distinct_from_gateway_api_key() {
        let settings: AppSettings = minimal_builder()
            .set_override("api_key", "gateway-client-token")
            .unwrap()
            .set_override("bedrock_api_key", "upstream-bedrock-token")
            .unwrap()
            .build()
            .unwrap()
            .try_deserialize()
            .unwrap();

        assert_eq!(settings.api_key, Some("gateway-client-token".to_string()));
        assert_eq!(
            settings.bedrock_api_key,
            Some("upstream-bedrock-token".to_string())
        );
        assert_ne!(settings.api_key, settings.bedrock_api_key);
    }

    #[test]
    fn bare_timeout_retry_env_vars_override_defaults() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        clear_timeout_retry_vars();

        std::env::set_var("AWS_CONNECT_TIMEOUT_SECS", "30");
        std::env::set_var("AWS_READ_TIMEOUT_SECS", "120");
        std::env::set_var("AWS_MAX_RETRY_ATTEMPTS", "3");

        let settings = AppSettings::load().unwrap();

        assert_eq!(settings.aws_connect_timeout_secs, 30);
        assert_eq!(settings.aws_read_timeout_secs, 120);
        assert_eq!(settings.aws_max_retry_attempts, 3);

        clear_timeout_retry_vars();
    }

    #[test]
    fn bare_timeout_retry_env_vars_non_numeric_falls_back_to_defaults() {
        let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        clear_timeout_retry_vars();

        std::env::set_var("AWS_CONNECT_TIMEOUT_SECS", "not-a-number");
        std::env::set_var("AWS_READ_TIMEOUT_SECS", "");
        std::env::set_var("AWS_MAX_RETRY_ATTEMPTS", "abc");

        let settings = AppSettings::load().unwrap();

        assert_eq!(settings.aws_connect_timeout_secs, 60);
        assert_eq!(settings.aws_read_timeout_secs, 900);
        assert_eq!(settings.aws_max_retry_attempts, 8);

        clear_timeout_retry_vars();
    }
}
