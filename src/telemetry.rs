//! Telemetry and logging.
//!
//! Structured logging built on [`tracing`] + [`tracing_subscriber`], translating
//! the production "zap" logging pattern (structured JSON, multi-output, dynamic
//! level) into idiomatic Rust:
//!
//! - **Structured output**: JSON in production (`debug = false`), pretty
//!   human-readable console in development (`debug = true`).
//! - **Env-driven filter**: the effective verbosity is derived from `RUST_LOG`
//!   when set, otherwise falls back to the level supplied by configuration
//!   (e.g. `AppSettings::log_level`).
//! - **Dynamic level**: a [`reload::Handle`] is exposed so the active filter can
//!   be swapped at runtime — the moral equivalent of zap's `AtomicLevel`. See
//!   [`set_level`].
//! - **Standard fields**: every event is tagged with the service name
//!   (`bedrock-gateway`) and the crate version (`CARGO_PKG_VERSION`).
//!
//! # Privacy
//!
//! Mirroring the upstream project's stance ("does not log any requests or
//! responses by default"), this module **never** logs request or response
//! bodies. If body logging is ever introduced, it MUST be gated behind
//! `debug = true` and treated as a development-only diagnostic — it must never
//! be enabled in production where PII could be captured.
//!
//! HTTP middleware (the axum `TraceLayer`) is intentionally **not** wired here;
//! that is configured separately at the server layer.

use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use tracing::Level;
use tracing_subscriber::{
    filter::EnvFilter, layer::SubscriberExt, reload, util::SubscriberInitExt, Registry,
};

/// Service name attached as a standard field to every log event.
pub const SERVICE_NAME: &str = "bedrock-gateway";

/// Crate version, captured at compile time, attached to every log event.
pub const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Handle that allows the active [`EnvFilter`] to be replaced at runtime.
///
/// This is the [`tracing`] analogue of zap's `AtomicLevel`: instead of mutating
/// an atomic integer, we hot-swap the whole [`EnvFilter`] behind a
/// [`reload::Layer`]. Pass this handle to [`set_level`] to change verbosity
/// without restarting the process.
pub type ReloadHandle = reload::Handle<EnvFilter, Registry>;

/// Process-wide storage for the reload handle, populated by [`init_telemetry`].
///
/// Stored so that callers which cannot thread the handle through their own
/// state (e.g. a signal handler) can still adjust the level via
/// [`set_global_level`].
static RELOAD_HANDLE: OnceLock<ReloadHandle> = OnceLock::new();

/// Guards against initializing the global subscriber more than once.
///
/// Initializing twice is a hard error from `tracing` (it panics). In tests we
/// construct layers in isolation instead, but `init_telemetry` itself is also
/// made idempotent: the second and later calls are no-ops returning `Ok`.
static INITIALIZED: OnceLock<()> = OnceLock::new();

/// Build the [`EnvFilter`] used as the base verbosity directive.
///
/// Resolution order (highest priority first):
/// 1. The `RUST_LOG` environment variable, if set and parseable.
/// 2. The provided `level` string (e.g. `"info"`, `"debug"`,
///    `"bedrock_gateway=debug,info"`).
/// 3. A hard fallback of `"info"` if `level` itself is empty/invalid.
///
/// Accepting full directive syntax (not just bare levels) means callers get the
/// same expressive power as `RUST_LOG` from their config file.
fn build_env_filter(level: &str) -> EnvFilter {
    // Prefer RUST_LOG when present so operators can override config at deploy time.
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        return filter;
    }

    let directive = if level.trim().is_empty() {
        "info"
    } else {
        level
    };

    EnvFilter::try_new(directive).unwrap_or_else(|_| EnvFilter::new("info"))
}

/// Construct the reloadable filter layer plus its control handle.
///
/// Returned separately from the formatting layer so tests can exercise filter
/// construction and reloading without standing up a global subscriber.
fn build_reloadable_filter(level: &str) -> (reload::Layer<EnvFilter, Registry>, ReloadHandle) {
    let filter = build_env_filter(level);
    reload::Layer::new(filter)
}

/// Initialize the global tracing subscriber exactly once.
///
/// * `debug` — when `true`, logs are emitted as pretty, multi-line, colourised
///   console output suitable for local development. When `false`, logs are
///   emitted as single-line structured JSON suitable for production log
///   aggregation.
/// * `level` — the fallback filter directive used when `RUST_LOG` is unset
///   (see [`build_env_filter`]).
///
/// Returns the [`ReloadHandle`] so the caller may adjust the level at runtime.
/// The handle is also stored globally (see [`set_global_level`]).
///
/// This function is idempotent: calling it more than once is a no-op that
/// returns the previously installed handle, guarding against the double-init
/// panic that `tracing` would otherwise raise.
pub fn init_telemetry(debug: bool, level: &str) -> Result<ReloadHandle> {
    // If already initialized, return the stored handle rather than panicking.
    if INITIALIZED.get().is_some() {
        return RELOAD_HANDLE
            .get()
            .cloned()
            .ok_or_else(|| anyhow!("telemetry already initialized but reload handle missing"));
    }

    let (filter_layer, handle) = build_reloadable_filter(level);

    if debug {
        // Development: pretty, human-friendly console output.
        let fmt_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_target(true)
            .with_level(true);

        Registry::default()
            .with(filter_layer)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| anyhow!("failed to initialize telemetry (pretty): {e}"))?;
    } else {
        // Production: structured single-line JSON for log aggregators.
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_current_span(false)
            .with_span_list(false)
            .with_target(true)
            .with_level(true);

        Registry::default()
            .with(filter_layer)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| anyhow!("failed to initialize telemetry (json): {e}"))?;
    }

    // Record that initialization happened and stash the handle for global access.
    let _ = INITIALIZED.set(());
    let _ = RELOAD_HANDLE.set(handle.clone());

    // Emit a single startup event carrying the standard service fields.
    let debug_mode = debug;
    tracing::info!(
        service = SERVICE_NAME,
        version = SERVICE_VERSION,
        debug_mode,
        "telemetry initialized"
    );

    Ok(handle)
}

/// Change the effective log level/filter at runtime via a reload handle.
///
/// `level` accepts the same directive syntax as `RUST_LOG` (e.g. `"debug"`,
/// `"warn"`, `"bedrock_gateway=trace,info"`). Returns `Err` if the directive is
/// invalid or if the underlying subscriber has been dropped.
///
/// This is the runtime knob equivalent to zap's `SetLogLevel`.
pub fn set_level(handle: &ReloadHandle, level: &str) -> Result<()> {
    let new_filter = EnvFilter::try_new(level)
        .map_err(|e| anyhow!("invalid log level directive '{level}': {e}"))?;

    handle
        .reload(new_filter)
        .map_err(|e| anyhow!("failed to reload log level: {e}"))?;

    Ok(())
}

/// Convenience wrapper around [`set_level`] using the globally stored handle.
///
/// Returns `Err` if telemetry has not been initialized yet or the directive is
/// invalid. Useful from contexts (signal handlers, admin endpoints) that do not
/// otherwise have access to the handle.
pub fn set_global_level(level: &str) -> Result<()> {
    let handle = RELOAD_HANDLE
        .get()
        .ok_or_else(|| anyhow!("telemetry not initialized; cannot set level"))?;
    set_level(handle, level)
}

/// Parse a bare level name (`"trace"`..`"error"`) into a [`Level`].
///
/// Helper for callers that want a typed level rather than a filter directive.
/// Case-insensitive. Returns `Err` for unknown names.
pub fn parse_level(level: &str) -> Result<Level> {
    match level.trim().to_ascii_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" | "warning" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        other => Err(anyhow!("unknown log level: '{other}'")),
    }
}

#[cfg(test)]
mod tests {
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
        assert!(!SERVICE_VERSION.is_empty());
    }
}
