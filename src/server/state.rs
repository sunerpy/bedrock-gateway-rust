//! Shared HTTP server state.
//!
//! [`AppState`] is the dependency container threaded through every axum handler
//! via [`axum::extract::State`]. It is intentionally defined here (not in the
//! router module) so that task 24's server bootstrap can construct and own it
//! while the routers (task 22) consume it — both depend on this one shared
//! shape rather than redefining it.
//!
//! Every field is an `Arc` (or holds `Arc`-backed clients) so the whole state
//! is cheaply clonable: axum clones it per request, and the model catalog can be
//! swapped behind an `RwLock` for live refresh without rebuilding the router.
//!
//! ## De-hardcoding
//!
//! The HTTP layer holds only the domain trait objects
//! ([`ChatProvider`]/[`EmbeddingProvider`]) plus the model catalog and the
//! resolved API key — it never references a concrete provider or any AWS SDK
//! type. The concrete [`crate::bedrock::provider::BedrockChatProvider`] /
//! [`crate::bedrock::embeddings::BedrockEmbeddingProvider`] are injected as
//! `Arc<dyn …>` by the bootstrap layer.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::bedrock::cache_support::CacheSupportRegistry;
use crate::bedrock::models::ModelCatalog;
use crate::config::AppSettings;
use crate::domain::{ChatProvider, EmbeddingProvider, ModelCapabilities, ResponsesProvider};

/// Application state shared across all HTTP handlers.
///
/// Cloning is cheap (every field is `Arc`-backed). The router holds a single
/// `AppState` and axum clones it per request.
#[derive(Clone)]
pub struct AppState {
    /// The chat-completions provider (Bedrock at runtime; mockable in tests).
    pub chat: Arc<dyn ChatProvider>,
    /// The Responses-API provider (Bedrock at runtime; mockable in tests). A
    /// separate surface from [`ChatProvider`] — not an overload.
    pub responses: Arc<dyn ResponsesProvider>,
    /// The embeddings provider.
    pub embeddings: Arc<dyn EmbeddingProvider>,
    /// The model catalog, behind an `RwLock` so `/models` can trigger a live
    /// refresh (Python parity) without rebuilding the router.
    pub catalog: Arc<RwLock<ModelCatalog>>,
    /// The config-driven capability resolver, used to normalize an incoming
    /// model id to its resolved foundation model before dispatch.
    pub caps: Arc<dyn ModelCapabilities>,
    /// The API key resolved once at boot (SSM → Secrets Manager → literal).
    /// Stored as `Arc<String>` so the bearer middleware can hold it as state.
    pub api_key: Arc<String>,
    /// The loaded application settings (route prefix, default model, etc.).
    pub settings: Arc<AppSettings>,
    /// Shared negative cache of foundation ids that reject prompt caching. The
    /// SAME `Arc` is injected into both providers so the cache safety net's
    /// learned rejections are visible across the chat and Responses surfaces.
    pub cache_support: Arc<CacheSupportRegistry>,
}

impl AppState {
    /// Construct the application state from its collaborators.
    ///
    /// The caller (task 24's bootstrap) builds the concrete providers, refreshes
    /// the catalog, resolves the API key, and passes everything in here.
    // This is a plain dependency-injection sink: each argument is a distinct
    // collaborator stored verbatim, so grouping them into a struct would only
    // add indirection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chat: Arc<dyn ChatProvider>,
        responses: Arc<dyn ResponsesProvider>,
        embeddings: Arc<dyn EmbeddingProvider>,
        catalog: Arc<RwLock<ModelCatalog>>,
        caps: Arc<dyn ModelCapabilities>,
        api_key: Arc<String>,
        settings: Arc<AppSettings>,
        cache_support: Arc<CacheSupportRegistry>,
    ) -> Self {
        Self {
            chat,
            responses,
            embeddings,
            catalog,
            caps,
            api_key,
            settings,
            cache_support,
        }
    }
}
