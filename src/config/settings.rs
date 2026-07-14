use anyhow::Result;
use config::builder::DefaultState;
use config::{Config, ConfigBuilder as RawConfigBuilder, Environment, File};
use serde::{Deserialize, Serialize};

type ConfigBuilder = RawConfigBuilder<DefaultState>;

/// Schema-level fallback for `prompt_cache_ttl` when a deserialization source
/// omits it. The authoritative default (`"5m"`) is also set via
/// `.set_default("prompt_cache_ttl", "5m")` in [`AppSettings::load`]; this keeps
/// `try_deserialize` from failing on sources that predate the field.
fn default_prompt_cache_ttl() -> String {
    "5m".to_string()
}

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

    /// Default prompt-cache TTL applied to injected `cachePoint`s when a request
    /// does not override it via `extra_body.prompt_caching.ttl`. Env:
    /// `PROMPT_CACHE_TTL` (or `APP_PROMPT_CACHE_TTL`). Default `"5m"`. A `"1h"`
    /// value is only honored on models declaring `cache_ttl_1h`; otherwise it is
    /// silently downgraded to `"5m"`.
    #[serde(default = "default_prompt_cache_ttl")]
    pub prompt_cache_ttl: String,

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

    /// Optional comma-separated allow-list of model-id substrings (env:
    /// `ALLOWED_MODELS`). When present it OVERRIDES the optional `models.toml`
    /// `allowed_models` list. Applied at catalog-build time to filter both
    /// `GET /models` and `GET /models/{id}` with zero per-request cost. `None`
    /// or empty ⇒ allow all (backward compatible).
    #[serde(default)]
    pub allowed_models: Option<String>,

    /// OTLP exporter endpoint (env: standard `OTEL_EXPORTER_OTLP_ENDPOINT`).
    /// When `Some` AND the crate is built with the `otel` feature, an OTLP
    /// tracer + meter provider is initialized and an OpenTelemetry tracing layer
    /// is added. `None` (or a build without the `otel` feature) ⇒ behavior is
    /// identical to a build with no OpenTelemetry export at all.
    #[serde(default)]
    pub otel_exporter_otlp_endpoint: Option<String>,

    /// Explicit opt-in to attach prompt/completion text as span attributes (env:
    /// `OTEL_CAPTURE_CONTENT`, default `false`). This is the SINGLE place the
    /// logging-privacy contract is intentionally relaxed: when `true` (and the
    /// `otel` feature is compiled in), the OTLP span-attribute builder may carry
    /// message content. Default `false` keeps spans REDACTED (metadata/counts
    /// only). Enabling it emits a loud startup WARN.
    #[serde(default)]
    pub otel_capture_content: bool,
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
            .set_default("prompt_cache_ttl", "5m")?
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
            .set_default("otel_capture_content", false)?
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
/// knobs `PORT`, `BIND_ADDR`, `LOG_LEVEL`, `MANTLE_BASE_URL_TEMPLATE`,
/// `ALLOWED_MODELS`, `PROMPT_CACHE_TTL`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
/// `OTEL_CAPTURE_CONTENT`.
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
        ("ALLOWED_MODELS", "allowed_models"),
        ("PROMPT_CACHE_TTL", "prompt_cache_ttl"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "otel_exporter_otlp_endpoint"),
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
        ("OTEL_CAPTURE_CONTENT", "otel_capture_content"),
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
#[path = "settings_tests.rs"]
mod tests;
