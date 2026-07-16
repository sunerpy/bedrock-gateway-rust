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
        .set_default("prompt_cache_ttl", "5m")
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
        .set_default("max_body_size_mb", 20u32)
        .unwrap()
        .set_default("max_body_size_mb", 20u32)
        .unwrap()
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
        .set_default("prompt_cache_ttl", "5m")
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
        .set_default("max_body_size_mb", 20u32)
        .unwrap()
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
    assert_eq!(settings.prompt_cache_ttl, "5m");
}

#[test]
fn default_ttl_is_5m() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("PROMPT_CACHE_TTL");
    std::env::remove_var("APP_PROMPT_CACHE_TTL");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(
        settings.prompt_cache_ttl, "5m",
        "unset PROMPT_CACHE_TTL must default to 5m"
    );
}

#[test]
fn bare_prompt_cache_ttl_env_overrides_default() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("APP_PROMPT_CACHE_TTL");
    std::env::set_var("PROMPT_CACHE_TTL", "1h");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.prompt_cache_ttl, "1h");
    std::env::remove_var("PROMPT_CACHE_TTL");
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
    builder = builder.set_default("max_body_size_mb", 20u32).unwrap();
    builder = builder
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
        .set_default("max_body_size_mb", 20u32)
        .unwrap()
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
    builder = builder.set_default("max_body_size_mb", 20u32).unwrap();
    builder = builder
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
    builder = builder.set_default("max_body_size_mb", 20u32).unwrap();
    builder = builder
        .set_default(
            "mantle_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/openai/v1",
        )
        .unwrap()
        .set_default(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
        .unwrap()
        .set_default("max_body_size_mb", 20u32)
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
        .unwrap()
        .set_override(
            "mantle_chat_base_url_template",
            "https://bedrock-mantle.{region}.api.aws/v1",
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
fn allowed_models_env_parses() {
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("ALLOWED_MODELS");

    // Unset ⇒ None (allow-all, backward compatible).
    let settings = AppSettings::load().unwrap();
    assert_eq!(settings.allowed_models, None);

    // Set to a comma-separated list ⇒ raw string is captured verbatim.
    std::env::set_var("ALLOWED_MODELS", "claude,nova");
    let settings = AppSettings::load().unwrap();
    assert_eq!(settings.allowed_models, Some("claude,nova".to_string()));

    // Empty ⇒ treated as unset (None), preserving allow-all.
    std::env::set_var("ALLOWED_MODELS", "");
    let settings = AppSettings::load().unwrap();
    assert_eq!(settings.allowed_models, None);

    std::env::remove_var("ALLOWED_MODELS");
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

#[test]
fn otel_settings_parse_from_env() {
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::env::remove_var("OTEL_CAPTURE_CONTENT");

    // Unset ⇒ endpoint None, capture false (default REDACTED).
    let settings = AppSettings::load().unwrap();
    assert_eq!(settings.otel_exporter_otlp_endpoint, None);
    assert!(!settings.otel_capture_content);

    // Set both ⇒ endpoint captured verbatim, capture flag true.
    std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317");
    std::env::set_var("OTEL_CAPTURE_CONTENT", "true");
    let settings = AppSettings::load().unwrap();
    assert_eq!(
        settings.otel_exporter_otlp_endpoint,
        Some("http://collector:4317".to_string())
    );
    assert!(settings.otel_capture_content);

    std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::env::remove_var("OTEL_CAPTURE_CONTENT");
}

#[test]
fn bare_env_name_wins_over_app_prefixed() {
    // Python-parity contract: when both the bare name and the `APP_`-prefixed
    // name are set, the bare name (applied via set_override) must win.
    // `PORT` is a single-word field, so the `APP_` env source (which treats
    // `_` as a nesting separator) maps `APP_PORT` cleanly onto `port`; both
    // values differ from the 8080 default so the assertion is unambiguous.
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("PORT");
    std::env::remove_var("APP_PORT");

    std::env::set_var("APP_PORT", "7000");
    std::env::set_var("PORT", "9000");

    let settings = AppSettings::load().expect("settings load");
    assert_eq!(
        settings.port, 9000,
        "bare env name must override the APP_-prefixed layer"
    );

    std::env::remove_var("PORT");
    std::env::remove_var("APP_PORT");
}

#[test]
fn app_prefixed_env_applies_when_no_bare_name() {
    // With only the `APP_`-prefixed variable set (no bare name), the prefixed
    // layer supplies the value (distinct from the 8080 default).
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("PORT");
    std::env::remove_var("APP_PORT");

    std::env::set_var("APP_PORT", "7001");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.port, 7001);

    std::env::remove_var("APP_PORT");
}

#[test]
fn aws_bearer_token_wins_over_bedrock_api_key_alias() {
    // The AWS standard name `AWS_BEARER_TOKEN_BEDROCK` is authoritative;
    // `BEDROCK_API_KEY` is only a fallback alias.
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
    std::env::remove_var("BEDROCK_API_KEY");

    // Alias alone populates bedrock_api_key.
    std::env::set_var("BEDROCK_API_KEY", "alias-token");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.bedrock_api_key, Some("alias-token".to_string()));

    // Standard name present ⇒ it wins over the alias.
    std::env::set_var("AWS_BEARER_TOKEN_BEDROCK", "standard-token");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.bedrock_api_key, Some("standard-token".to_string()));

    std::env::remove_var("AWS_BEARER_TOKEN_BEDROCK");
    std::env::remove_var("BEDROCK_API_KEY");
}

#[test]
fn max_body_size_mb_defaults_to_20() {
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("MAX_BODY_SIZE_MB");
    std::env::remove_var("APP_MAX_BODY_SIZE_MB");

    // Unset ⇒ the built-in default of 20 MB.
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(
        settings.max_body_size_mb, 20,
        "unset MAX_BODY_SIZE_MB must default to 20"
    );

    // The minimal builder (no env, no file) must also default to 20.
    let settings: AppSettings = minimal_builder()
        .build()
        .unwrap()
        .try_deserialize()
        .unwrap();
    assert_eq!(settings.max_body_size_mb, 20);
}

#[test]
fn max_body_size_mb_env_parses() {
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("MAX_BODY_SIZE_MB");
    std::env::remove_var("APP_MAX_BODY_SIZE_MB");

    // Bare name ⇒ parsed onto the typed field.
    std::env::set_var("MAX_BODY_SIZE_MB", "50");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.max_body_size_mb, 50);

    // Non-numeric ⇒ leniently ignored, default retained (parity with PORT).
    std::env::set_var("MAX_BODY_SIZE_MB", "not-a-number");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(settings.max_body_size_mb, 20);

    // Bare name wins over the APP_-prefixed layer.
    std::env::set_var("APP_MAX_BODY_SIZE_MB", "10");
    std::env::set_var("MAX_BODY_SIZE_MB", "64");
    let settings = AppSettings::load().expect("settings load");
    assert_eq!(
        settings.max_body_size_mb, 64,
        "bare MAX_BODY_SIZE_MB must override the APP_-prefixed layer"
    );

    std::env::remove_var("MAX_BODY_SIZE_MB");
    std::env::remove_var("APP_MAX_BODY_SIZE_MB");
}

#[test]
fn capture_content_defaults_false() {
    let _guard = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    std::env::remove_var("OTEL_CAPTURE_CONTENT");

    // Default must be false — content capture is opt-in only.
    let settings = AppSettings::load().unwrap();
    assert!(!settings.otel_capture_content);

    // The minimal builder (no env) must also default to false.
    let settings: AppSettings = minimal_builder()
        .build()
        .unwrap()
        .try_deserialize()
        .unwrap();
    assert!(!settings.otel_capture_content);
    assert_eq!(settings.otel_exporter_otlp_endpoint, None);
}
