//! Bedrock Gateway binary entrypoint.
//!
//! Thin composition glue only: initialize telemetry, load settings, then hand
//! off to [`bedrock_gateway_rust::server::serve`]. No business logic lives here.
//!
//! Special mode: `--health-check` flag performs a self-probe without starting
//! the server. This is used by container health checks (distroless has no shell/curl).

use anyhow::{Context, Result};
use std::process;

use bedrock_gateway_rust::config::AppSettings;
use bedrock_gateway_rust::{server, telemetry};

#[tokio::main]
async fn main() -> Result<()> {
    // Check for --health-check flag (must come before any logging/config setup)
    if std::env::args().any(|arg| arg == "--health-check") {
        return health_check().await;
    }

    let settings = AppSettings::load().context("failed to load application settings")?;

    let otel = telemetry::OtelConfig {
        endpoint: settings.otel_exporter_otlp_endpoint.clone(),
        capture_content: settings.otel_capture_content,
    };
    telemetry::init_telemetry(settings.debug, &settings.log_level, &otel)
        .context("failed to initialize telemetry")?;

    server::serve(settings).await
}

/// Health check probe: query http://127.0.0.1:PORT/API_ROUTE_PREFIX/health
/// Exit with 0 if status 200, else exit with 1.
/// Does NOT require API key or start the server.
/// Short timeout (~2s) to fail fast if no server is running.
async fn health_check() -> Result<()> {
    let settings = AppSettings::load().context("failed to load application settings")?;

    let url = format!(
        "http://127.0.0.1:{}{}/health",
        settings.port, settings.api_route_prefix
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .context("failed to build HTTP client for health check")?;

    match client.get(&url).send().await {
        Ok(response) if response.status() == 200 => {
            process::exit(0);
        }
        Ok(response) => {
            eprintln!("health check failed: status {}", response.status());
            process::exit(1);
        }
        Err(e) => {
            eprintln!("health check failed: {}", e);
            process::exit(1);
        }
    }
}
