//! Unit tests for the telemetry module.
//!
//! Relocated out of `mod.rs` for code organization (see the
//! `test-coverage-codecov` spec). Behavior is unchanged; `use super::*`
//! continues to resolve to the parent `telemetry` module since this file is
//! declared as its child `mod tests;`.

use super::*;

// NOTE: We deliberately avoid calling `init_telemetry` in unit tests, since
// it installs a *global* subscriber that can only be set once per process.
// Instead we test the constituent pieces (filter construction, reloading,
// layer building) in isolation — this keeps tests independent and panic-free.

#[test]
fn build_env_filter_falls_back_to_provided_level() {
    // Guard: only meaningful when RUST_LOG is not influencing the result.
    if std::env::var("RUST_LOG").is_ok() {
        return;
    }
    let filter = build_env_filter("debug");
    // The filter renders its directives back out; "debug" should appear.
    assert!(filter.to_string().contains("debug"));
}

#[test]
fn build_env_filter_empty_level_defaults_to_info() {
    if std::env::var("RUST_LOG").is_ok() {
        return;
    }
    let filter = build_env_filter("");
    assert!(filter.to_string().contains("info"));
}

#[test]
fn build_env_filter_accepts_complex_directives() {
    if std::env::var("RUST_LOG").is_ok() {
        return;
    }
    let filter = build_env_filter("bedrock_gateway=trace,info");
    let rendered = filter.to_string();
    assert!(rendered.contains("bedrock_gateway"));
    assert!(rendered.contains("trace"));
}

#[test]
fn json_layer_constructs_without_panic() {
    // Build a JSON fmt layer + reloadable filter over a fresh registry.
    // This exercises the exact construction path used by init_telemetry
    // in production mode, without installing it globally.
    let (filter_layer, _handle) = build_reloadable_filter("info");
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_current_span(false)
        .with_span_list(false)
        .with_target(true)
        .with_level(true);
    let _subscriber = Registry::default().with(filter_layer).with(fmt_layer);
    // Reaching here means the layer stack composed successfully.
}

#[test]
fn pretty_layer_constructs_without_panic() {
    // Same as above for development (pretty) mode.
    let (filter_layer, _handle) = build_reloadable_filter("debug");
    let fmt_layer = tracing_subscriber::fmt::layer()
        .pretty()
        .with_target(true)
        .with_level(true);
    let _subscriber = Registry::default().with(filter_layer).with(fmt_layer);
}

#[test]
fn set_level_via_reload_handle_accepts_valid_level() {
    // Install a throwaway subscriber on the *current thread only* so that
    // the reload handle is backed by a live subscriber. Using a local
    // default guard avoids touching the process-global subscriber.
    let (filter_layer, handle) = build_reloadable_filter("info");
    let fmt_layer = tracing_subscriber::fmt::layer().json();
    let subscriber = Registry::default().with(filter_layer).with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    // A valid directive must succeed.
    assert!(set_level(&handle, "debug").is_ok());
    assert!(set_level(&handle, "bedrock_gateway=trace,warn").is_ok());
}

#[test]
fn set_level_rejects_invalid_level() {
    let (filter_layer, handle) = build_reloadable_filter("info");
    let fmt_layer = tracing_subscriber::fmt::layer().json();
    let subscriber = Registry::default().with(filter_layer).with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    // An invalid level keyword after '=' must be rejected, not silently
    // accepted. (Bare unknown words are treated by EnvFilter as targets, so
    // we use a malformed level directive to force a parse error.)
    let result = set_level(&handle, "foo=not_a_real_level");
    assert!(result.is_err());
}

#[test]
fn set_global_level_errors_when_uninitialized_or_ok_when_set() {
    // Either telemetry was never initialized in this test binary (Err), or a
    // prior test/integration initialized it (Ok). Both are valid outcomes;
    // we only assert it does not panic and returns a Result.
    let _ = set_global_level("info");
}

#[test]
fn parse_level_handles_known_and_unknown() {
    assert_eq!(parse_level("trace").unwrap(), Level::TRACE);
    assert_eq!(parse_level("DEBUG").unwrap(), Level::DEBUG);
    assert_eq!(parse_level("Info").unwrap(), Level::INFO);
    assert_eq!(parse_level("warn").unwrap(), Level::WARN);
    assert_eq!(parse_level("warning").unwrap(), Level::WARN);
    assert_eq!(parse_level("error").unwrap(), Level::ERROR);
    assert!(parse_level("bogus").is_err());
}

#[test]
fn service_fields_are_populated() {
    assert_eq!(SERVICE_NAME, "bedrock-gateway");
    assert_eq!(SERVICE_VERSION, env!("CARGO_PKG_VERSION"));
}

/// Exercises the real `init_telemetry` entry point end to end: it installs
/// the process-global subscriber (production JSON branch, no OTLP endpoint
/// so the otel layer is inert), stores the reload handle, and is idempotent
/// on every subsequent call. Kept in a SINGLE test so the one-shot global
/// install is never raced by parallel tests — no other test calls
/// `init_telemetry`.
#[test]
fn init_telemetry_installs_and_is_idempotent() {
    // First call performs the real global install and must hand back a
    // reload handle. `debug = false` drives the structured-JSON branch;
    // the default `OtelConfig` has no endpoint, so `build_otel_layer`
    // resolves to the inert `None` layer.
    let handle = init_telemetry(false, "info", &OtelConfig::default())
        .expect("first init_telemetry must succeed");

    // The globally stored handle can drive a dynamic level change on the
    // now-live global subscriber (ReloadHandle dynamic level).
    set_level(&handle, "debug").expect("dynamic level change via handle");

    // The second and later calls are no-ops that return the stored handle
    // rather than triggering the double-install panic `tracing` would
    // otherwise raise — even with different arguments (debug = true, a
    // different level directive).
    init_telemetry(true, "trace", &OtelConfig::default())
        .expect("second init_telemetry is idempotent");

    // With telemetry initialized, the global-handle convenience wrapper
    // resolves and applies a valid directive against the live subscriber.
    set_global_level("warn").expect("set_global_level after init");
}

/// The public metrics entry point is callable without a configured
/// endpoint: under `--all-features` it routes through the global (no-op)
/// meter; without the feature it is a total no-op. Either way it must not
/// panic, and both the `Some`/`None` `finish_reason` shapes are exercised.
#[test]
fn record_request_metrics_is_callable_without_endpoint() {
    record_request_metrics("some-model", Some("stop"), 12, 30, 7);
    record_request_metrics("another-model", None, 0, 0, 0);
}

/// `OtelConfig` derives (Default / Clone / Debug) are exercised so the
/// config seam is covered on both the default and `otel` builds.
#[test]
fn otel_config_default_clone_debug() {
    let cfg = OtelConfig::default();
    assert!(cfg.endpoint.is_none());
    assert!(!cfg.capture_content);

    let cloned = cfg.clone();
    assert_eq!(cloned.endpoint, cfg.endpoint);
    assert_eq!(cloned.capture_content, cfg.capture_content);

    // Debug formatting must not panic and should name the type.
    assert!(format!("{cfg:?}").contains("OtelConfig"));
}

#[test]
fn build_env_filter_invalid_directive_falls_back_to_info() {
    // Only meaningful when RUST_LOG is not short-circuiting the fn.
    if std::env::var("RUST_LOG").is_ok() {
        return;
    }
    // A malformed level after '=' makes `EnvFilter::try_new` fail, exercising
    // the `unwrap_or_else(|_| EnvFilter::new("info"))` fallback arm.
    let filter = build_env_filter("foo=not_a_real_level");
    assert!(filter.to_string().contains("info"));
}

/// The OTLP startup events are metadata-only tracing emissions (no network, no
/// provider init). Calling the emitter directly with an endpoint + content
/// capture exercises the `Some(endpoint)` branch and the `capture_content`
/// WARN arm without standing up any exporter. Feature-gated because the emitter
/// is a no-op stub when `otel` is disabled.
#[cfg(feature = "otel")]
#[test]
fn emit_otel_startup_events_covers_endpoint_and_capture_arms() {
    // Endpoint set + capture ON: hits the info line AND the capture WARN.
    let with_capture = OtelConfig {
        endpoint: Some("http://localhost:4318".to_string()),
        capture_content: true,
    };
    emit_otel_startup_events(&with_capture);

    // Endpoint set + capture OFF: hits the info line, skips the WARN.
    let without_capture = OtelConfig {
        endpoint: Some("http://localhost:4318".to_string()),
        capture_content: false,
    };
    emit_otel_startup_events(&without_capture);

    // Empty endpoint is filtered out (treated as no endpoint): neither arm.
    let empty_endpoint = OtelConfig {
        endpoint: Some(String::new()),
        capture_content: true,
    };
    emit_otel_startup_events(&empty_endpoint);

    // No endpoint at all: the guard is false, nothing emitted.
    emit_otel_startup_events(&OtelConfig::default());
}

/// `build_otel_layer` with no configured endpoint resolves to the inert `None`
/// layer on BOTH builds (feature-on: the `None` match arm; feature-off: the
/// stub). This never touches `init_otel`, so no exporter is created.
#[test]
fn build_otel_layer_none_without_endpoint() {
    let layer = build_otel_layer::<OtelBaseSubscriber>(&OtelConfig::default())
        .expect("no-endpoint layer build must succeed");
    assert!(
        layer.is_none(),
        "absent endpoint must yield an inert None layer"
    );
}
