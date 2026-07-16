//! HTTP server bootstrap, layer wiring, and graceful shutdown.
//!
//! This module is the composition root for the running service. It assembles
//! the concrete Bedrock-backed providers, the model catalog, and the resolved
//! API key into the shared [`state::AppState`], builds the router (task 22's
//! [`routers::build_router`]), wraps it with the cross-cutting HTTP layers
//! ([`TraceLayer`] for request tracing and a permissive [`CorsLayer`] for parity
//! with the legacy FastAPI CORS middleware), binds the listener, and serves with
//! a SIGTERM/Ctrl-C-aware graceful shutdown.
//!
//! ## No timeout on the streaming route
//!
//! The chat route serves Server-Sent Events. A request-timeout layer would sever
//! an in-flight SSE connection, so **no** `TimeoutLayer` is applied here. Tracing
//! and CORS are connection-safe and applied globally.
//!
//! ## Best-effort catalog refresh
//!
//! The initial model-catalog refresh hits the Bedrock control plane and may fail
//! (no credentials, transient 5xx, region without Bedrock). A failure here must
//! NOT crash boot — the service still serves `/health` and per-request inference
//! works without the catalog. The refresh is therefore best-effort: a failure is
//! logged and an empty catalog is used until `/models` is hit or a later refresh
//! succeeds.
//!
//! ## Secrets
//!
//! The resolved API key is never logged. Telemetry emits only non-sensitive
//! fields (bind address, prefix, model defaults).

pub mod auth;

pub mod composite;

pub mod composite_chat;

pub mod routers;

pub mod state;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::DefaultBodyLimit;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tower_http::LatencyUnit;
use tracing::Level;

use crate::bedrock::cache_support::CacheSupportRegistry;
use crate::bedrock::client::{build_aws_config, BedrockClients};
use crate::bedrock::embeddings::BedrockEmbeddingProvider;
use crate::bedrock::mantle_chat_provider::MantleChatProvider;
use crate::bedrock::mantle_client::MantleClient;
use crate::bedrock::mantle_provider::MantleResponsesProvider;
use crate::bedrock::models::ModelCatalog;
use crate::bedrock::provider::BedrockChatProvider;
use crate::bedrock::responses_provider::BedrockResponsesProvider;
use crate::bedrock::translate::ReqwestImageResolver;
use crate::config::{AppSettings, EmbeddingRegistry, ModelCapabilityConfig, RegionRoutingConfig};
use crate::domain::{ChatProvider, EmbeddingProvider, ModelCapabilities, ResponsesProvider};
use crate::server::composite::{resolve_mantle_enabled, CompositeResponsesProvider};
use crate::server::composite_chat::{resolve_mantle_chat_enabled, CompositeChatProvider};
use crate::server::state::AppState;

/// Default config directory used when `CONFIG_DIR` is unset (backward-compatible
/// with existing deployments and the Docker WORKDIR fix).
const DEFAULT_CONFIG_DIR: &str = "config";
/// File names of the three externalizable TOML configs, joined onto the resolved
/// config directory to form each external override path.
const MODELS_CONFIG_FILE: &str = "models.toml";
const EMBEDDINGS_CONFIG_FILE: &str = "embeddings.toml";
const REGIONS_CONFIG_FILE: &str = "regions.toml";

/// Assemble the [`AppState`] from loaded settings, building the AWS config,
/// Bedrock clients, capability resolver, providers, and (best-effort) catalog.
///
/// Pulled out of [`serve`] so an integration test can build the state from a
/// real (credential-free) settings object and exercise the router without
/// binding a socket. A failing catalog refresh degrades to an empty catalog
/// rather than erroring, matching the production best-effort contract.
async fn build_app_state(settings: Arc<AppSettings>) -> Result<AppState> {
    let aws_config = build_aws_config(&settings).await;
    let clients = BedrockClients::new(&aws_config);

    let api_key = auth::resolve_api_key(&settings, &aws_config)
        .await
        .map_err(|e| anyhow::anyhow!("failed to resolve API key: {e}"))?;
    let api_key = Arc::new(api_key);

    let catalog = match ModelCatalog::refresh(&clients.control, &settings).await {
        Ok(catalog) => catalog,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "model catalog refresh failed at boot; continuing with empty catalog (best-effort)"
            );
            ModelCatalog::default()
        }
    };

    let config_dir = std::env::var("CONFIG_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_CONFIG_DIR.to_string());
    let config_dir = std::path::Path::new(&config_dir);

    let caps_config =
        ModelCapabilityConfig::load_with_fallback(Some(&config_dir.join(MODELS_CONFIG_FILE)));

    let mantle_enabled = resolve_mantle_enabled(&caps_config, &settings);
    let mantle_chat_enabled = resolve_mantle_chat_enabled(&caps_config, &settings);

    // Mantle-backed models are absent from the control-plane catalog; surface
    // their bare alias names so `/models` lists them. Computed before
    // `caps_config` is moved.
    let mantle_alias_names = caps_config.mantle_alias_names();

    // Effective allow-list: env `ALLOWED_MODELS` (comma-separated) OVERRIDES the
    // optional `models.toml` `allowed_models` list. Resolved before `caps_config`
    // is moved into the resolver. Empty ⇒ allow all (identity filter).
    let effective_allow_list = resolve_allow_list(&settings, &caps_config);

    let caps: Arc<dyn ModelCapabilities> = Arc::new(
        crate::bedrock::capabilities::ConfigModelCapabilities::with_profiles(
            caps_config,
            catalog.profile_metadata().clone(),
        ),
    );

    let catalog = Arc::new(RwLock::new(
        catalog
            .with_extra_models(mantle_alias_names)
            .apply_allow_list(&effective_allow_list),
    ));

    let regions = Arc::new(RegionRoutingConfig::load_with_fallback(Some(
        &config_dir.join(REGIONS_CONFIG_FILE),
    )));

    let embedding_registry =
        EmbeddingRegistry::load_with_fallback(Some(&config_dir.join(EMBEDDINGS_CONFIG_FILE)));

    let image_resolver = Arc::new(ReqwestImageResolver::new(supports_image_predicate(
        catalog.clone(),
    )));

    // One shared negative cache for prompt-caching support, injected into both
    // providers so a learned rejection is visible on the chat and Responses
    // surfaces alike.
    let cache_support = Arc::new(CacheSupportRegistry::new());

    let converse_chat: Arc<dyn ChatProvider> = Arc::new(BedrockChatProvider::new(
        clients.clone(),
        caps.clone(),
        regions.clone(),
        image_resolver.clone(),
        settings.clone(),
        cache_support.clone(),
    ));
    let converse_responses: Arc<dyn ResponsesProvider> = Arc::new(BedrockResponsesProvider::new(
        clients.clone(),
        caps.clone(),
        regions,
        image_resolver,
        settings.clone(),
        cache_support.clone(),
    ));

    let mantle_client = MantleClient::new(
        reqwest::Client::new(),
        settings.mantle_base_url_template.clone(),
        settings.mantle_chat_base_url_template.clone(),
        settings.bedrock_api_key.clone().unwrap_or_default(),
    );
    let mantle_responses: Arc<dyn ResponsesProvider> = Arc::new(MantleResponsesProvider::new(
        mantle_client.clone(),
        caps.clone(),
        settings.clone(),
    ));

    let responses: Arc<dyn ResponsesProvider> = Arc::new(CompositeResponsesProvider::new(
        converse_responses,
        mantle_responses,
        caps.clone(),
        mantle_enabled,
    ));

    let mantle_chat: Arc<dyn ChatProvider> = Arc::new(MantleChatProvider::new(
        mantle_client,
        caps.clone(),
        settings.clone(),
    ));
    let chat: Arc<dyn ChatProvider> = Arc::new(CompositeChatProvider::new(
        converse_chat,
        mantle_chat,
        caps.clone(),
        mantle_chat_enabled,
    ));

    let embeddings: Arc<dyn EmbeddingProvider> =
        Arc::new(BedrockEmbeddingProvider::new(clients, embedding_registry));

    Ok(AppState::new(
        chat,
        responses,
        embeddings,
        catalog,
        caps,
        api_key,
        settings,
        cache_support,
    ))
}

/// Resolve the effective model allow-list: the env `ALLOWED_MODELS` value
/// (comma-separated, whitespace trimmed, empty entries dropped) takes precedence
/// over the `models.toml` `allowed_models` list; when the env var is unset the
/// config list is used. An empty result means "allow all" (identity filter).
fn resolve_allow_list(settings: &AppSettings, caps_config: &ModelCapabilityConfig) -> Vec<String> {
    match settings.allowed_models.as_deref() {
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        None => caps_config.allowed_models.clone(),
    }
}

/// Build the `supports_image` predicate for the multimodal URL resolver by
/// consulting the live catalog for an `IMAGE` input modality on the resolved
/// model id. The closure clones the `Arc<RwLock<…>>` so it stays valid for the
/// lifetime of the provider; it uses a blocking read inside an async context via
/// `try_read`, falling back to `false` when the lock is momentarily contended
/// (a refresh in progress) — a conservative default that simply skips remote
/// image fetching rather than risking a deadlock.
fn supports_image_predicate(
    catalog: Arc<RwLock<ModelCatalog>>,
) -> impl Fn(&str) -> bool + Send + Sync + 'static {
    move |model_id: &str| {
        let Ok(guard) = catalog.try_read() else {
            return false;
        };
        guard
            .models()
            .get(model_id)
            .map(|info| {
                info.modalities
                    .iter()
                    .any(|m| m.eq_ignore_ascii_case("IMAGE"))
            })
            .unwrap_or(false)
    }
}

/// Wrap the application router with the cross-cutting HTTP layers.
///
/// [`TraceLayer`] provides structured request/response access logging at the
/// **INFO** level so each request shows up under the default `info` filter: the
/// request span carries `method` + `uri` (path), and `on_response` emits the
/// `status` and `latency` in milliseconds. It NEVER logs headers (no
/// `Authorization`) or bodies — [`DefaultMakeSpan`] records only method/uri/
/// version, which is privacy-safe. [`CorsLayer::permissive`] mirrors the legacy
/// FastAPI configuration (`allow_origins=["*"]`, all methods, all headers). No
/// timeout layer is added because the chat route streams SSE.
///
/// `body_limit_bytes` caps the accepted request body via [`DefaultBodyLimit`],
/// replacing axum's 2 MB default so base64-encoded image payloads fit; an
/// over-limit body is rejected with `413 Payload Too Large`.
fn apply_layers(router: axum::Router, body_limit_bytes: usize) -> axum::Router {
    router
        .layer(DefaultBodyLimit::max(body_limit_bytes))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::DEBUG))
                .on_response(
                    DefaultOnResponse::new()
                        .level(Level::DEBUG)
                        .latency_unit(LatencyUnit::Millis),
                ),
        )
        .layer(CorsLayer::permissive())
}

/// Boot the gateway: assemble state, build + layer the router, bind the listener
/// from `settings.bind_addr:settings.port`, and serve until a shutdown signal.
///
/// Returns once the server has shut down gracefully (drained in-flight requests
/// after a SIGTERM / Ctrl-C), or with an error if state assembly or socket
/// binding fails. Telemetry initialization is the caller's responsibility
/// (`main`), performed before this is called.
pub async fn serve(settings: AppSettings) -> Result<()> {
    let settings = Arc::new(settings);

    let bind = format!("{}:{}", settings.bind_addr, settings.port);
    let prefix = settings.api_route_prefix.clone();

    let body_limit = (settings.max_body_size_mb as usize).saturating_mul(1024 * 1024);

    let state = build_app_state(settings.clone()).await?;
    let app = apply_layers(routers::build_router(state, &prefix), body_limit);

    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind listener on {bind}"))?;

    tracing::info!(
        bind = %bind,
        prefix = %prefix,
        default_model = %settings.default_model,
        default_embedding_model = %settings.default_embedding_model,
        "bedrock-gateway listening"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;

    tracing::info!("bedrock-gateway shut down cleanly");
    Ok(())
}

/// Resolve when the process receives a termination signal.
///
/// Listens for BOTH Ctrl-C (`SIGINT`) and `SIGTERM`. SIGTERM handling is
/// mandatory for container orchestrators (ECS, Kubernetes) and Lambda, which
/// send SIGTERM to request a graceful stop before a hard kill. On non-unix
/// platforms only Ctrl-C is awaited.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "failed to install Ctrl-C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => tracing::error!(error = %e, "failed to install SIGTERM handler"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received Ctrl-C, shutting down"),
        () = terminate => tracing::info!("received SIGTERM, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use std::collections::HashMap;
    use tower::ServiceExt;

    /// Settings with a literal API key so `resolve_api_key` takes the no-AWS
    /// literal branch — the whole bootstrap then runs without credentials.
    fn boot_settings() -> AppSettings {
        AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: "us-west-2".to_string(),
            default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
            enable_cross_region_inference: false,
            enable_application_inference_profiles: false,
            enable_prompt_caching: false,
            prompt_cache_ttl: "5m".to_string(),
            api_key: Some("testkey".to_string()),
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: Some("test-bedrock-bearer".to_string()),
            disable_mantle: false,
            bind_addr: "127.0.0.1".to_string(),
            port: 0,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
            max_body_size_mb: 20,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
            mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
            allowed_models: None,
            otel_exporter_otlp_endpoint: None,
            otel_capture_content: false,
        }
    }

    #[test]
    fn resolve_allow_list_env_overrides_config() {
        let mut settings = boot_settings();
        let caps_config = ModelCapabilityConfig {
            allowed_models: vec!["nova".to_string()],
            ..ModelCapabilityConfig::default()
        };

        settings.allowed_models = None;
        assert_eq!(
            resolve_allow_list(&settings, &caps_config),
            vec!["nova".to_string()]
        );

        settings.allowed_models = Some(" claude , gpt ".to_string());
        assert_eq!(
            resolve_allow_list(&settings, &caps_config),
            vec!["claude".to_string(), "gpt".to_string()]
        );
    }

    #[test]
    fn supports_image_predicate_reads_catalog_modalities() {
        use crate::bedrock::models::{assemble_catalog, FoundationModelFacts};
        let settings = boot_settings();
        let fms = [
            FoundationModelFacts {
                model_id: "vision.model-v1:0".to_string(),
                input_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
                inference_types: vec!["ON_DEMAND".to_string()],
                response_streaming_supported: true,
                status: "ACTIVE".to_string(),
            },
            FoundationModelFacts {
                model_id: "text.model-v1:0".to_string(),
                input_modalities: vec!["TEXT".to_string()],
                inference_types: vec!["ON_DEMAND".to_string()],
                response_streaming_supported: true,
                status: "ACTIVE".to_string(),
            },
        ];
        let catalog = Arc::new(RwLock::new(assemble_catalog(&fms, &[], &settings)));
        let predicate = supports_image_predicate(catalog);

        assert!(predicate("vision.model-v1:0"), "IMAGE modality => true");
        assert!(!predicate("text.model-v1:0"), "TEXT-only => false");
        assert!(!predicate("unknown.model"), "absent => false");
    }

    /// The bootstrap assembles a working `AppState` without AWS credentials
    /// (literal API key, cross-region/profiles disabled so the catalog refresh
    /// is best-effort and any failure degrades to empty) and the resulting
    /// router serves `GET {prefix}/health` with `200 OK`.
    #[tokio::test]
    async fn boot_state_and_health_responds_200() {
        let settings = Arc::new(boot_settings());
        let prefix = settings.api_route_prefix.clone();
        let state = build_app_state(settings)
            .await
            .expect("state assembly must succeed without AWS creds");
        let app = apply_layers(routers::build_router(state, &prefix), 20 * 1024 * 1024);

        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&bytes[..], b"OK");
    }

    /// CORS parity: a permissive layer must echo the wildcard
    /// `access-control-allow-origin` on responses.
    #[tokio::test]
    async fn cors_headers_present_on_health() {
        let settings = Arc::new(boot_settings());
        let prefix = settings.api_route_prefix.clone();
        let state = build_app_state(settings).await.unwrap();
        let app = apply_layers(routers::build_router(state, &prefix), 20 * 1024 * 1024);

        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .header("origin", "https://example.com")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("*"),
            "permissive CORS must echo wildcard origin"
        );
    }

    /// A full end-to-end boot on an ephemeral port: bind a real listener, serve
    /// with a controllable shutdown trigger, curl `/health` over TCP, then
    /// signal shutdown and assert the server task exits cleanly (no panic).
    #[tokio::test]
    async fn server_boots_on_ephemeral_port_serves_health_then_shuts_down() {
        let settings = Arc::new(boot_settings());
        let prefix = settings.api_route_prefix.clone();
        let state = build_app_state(settings).await.unwrap();
        let app = apply_layers(routers::build_router(state, &prefix), 20 * 1024 * 1024);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });

        let url = format!("http://{addr}/api/v1/health");
        let body = reqwest::get(&url).await.unwrap();
        assert_eq!(body.status(), 200);
        assert_eq!(body.text().await.unwrap(), "OK");

        tx.send(()).unwrap();
        server.await.expect("server task must exit cleanly");
    }

    /// A request body over the configured limit is rejected with `413 Payload
    /// Too Large`, while a body under the limit is NOT rejected with 413. Uses a
    /// tiny 1-byte limit so a small JSON payload trips the guard deterministically.
    #[tokio::test]
    async fn body_limit_rejects_over_limit_with_413() {
        let settings = Arc::new(boot_settings());
        let prefix = settings.api_route_prefix.clone();
        let state = build_app_state(settings).await.unwrap();
        let app = apply_layers(routers::build_router(state, &prefix), 1);

        let over = Request::builder()
            .method("POST")
            .uri("/api/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer testkey")
            .body(Body::from("{\"model\":\"x\"}"))
            .unwrap();
        let resp = app.clone().oneshot(over).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "body over the configured limit must return 413"
        );

        let under = Request::builder()
            .method("POST")
            .uri("/api/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer testkey")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(under).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "an empty body must not be rejected by the size limit"
        );
    }

    /// Sanity: the explicit field map in `build_app_state` keeps the catalog and
    /// caps consistent — the profile metadata seeded into the resolver matches
    /// the catalog's. Guards against a future edit decoupling the two.
    #[tokio::test]
    async fn caps_seeded_from_catalog_profile_metadata() {
        let settings = Arc::new(boot_settings());
        let state = build_app_state(settings).await.unwrap();
        let profiles: HashMap<String, String> =
            state.catalog.read().await.profile_metadata().clone();
        for (profile_id, foundation) in profiles {
            assert_eq!(
                state.caps.resolve_foundation(&profile_id),
                foundation,
                "resolver must resolve seeded profile id to its foundation model"
            );
        }
    }
}
