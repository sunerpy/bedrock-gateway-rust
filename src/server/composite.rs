//! Composite [`ResponsesProvider`] that routes by backend + startup validation.
//!
//! The gateway serves the Responses API from TWO backends — the default Bedrock
//! Converse path ([`crate::bedrock::responses_provider::BedrockResponsesProvider`])
//! and the OpenAI-compatible bedrock-mantle upstream
//! ([`crate::bedrock::mantle_provider::MantleResponsesProvider`]) for the GPT-5.x
//! family. [`CompositeResponsesProvider`] hides both behind the single
//! `Arc<dyn ResponsesProvider>` that [`crate::server::state::AppState`] holds, so
//! `AppState`'s public shape is unchanged.
//!
//! ## Routing
//!
//! Routing is purely config-driven (zero model-name branching): every method
//! consults [`ModelCapabilities::responses_backend`] for the client's requested
//! model (which resolves the alias internally, T2) and dispatches to the matching
//! inner provider. [`ResponsesBackend::Mantle`] → the mantle provider;
//! [`ResponsesBackend::Converse`] (the default for every other model) → the
//! Converse provider.
//!
//! The raw-bytes streaming lane ([`ResponsesProvider::respond_raw_stream`]) is
//! overridden here and delegated to the selected inner provider — otherwise the
//! trait default (`Ok(None)`) would win and the mantle raw passthrough would
//! never fire.
//!
//! ## Startup resolution
//!
//! [`resolve_mantle_enabled`] scans the model config for any entry declaring the
//! mantle backend and decides whether mantle can serve requests on this instance.
//! It emits WARNINGs for disabled mantle and for region allow-list mismatches, but
//! never fails gateway boot.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::capabilities::ModelCapabilityConfig;
use crate::config::AppSettings;
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, RawResponsesStream, ResponsesBackend,
    ResponsesProvider, ResponsesStream,
};
use crate::error::AppError;
use crate::openai::responses_schema::ResponsesResponse;

/// The config value of `params.responses_backend` that selects the mantle
/// upstream. Matches the `ResponsesBackend::Mantle` mapping in
/// `crate::bedrock::capabilities`.
const MANTLE_BACKEND: &str = "mantle";

/// A [`ResponsesProvider`] that dispatches each request to one of two inner
/// providers based on the model's configured backend.
///
/// Cheap to clone — every field is `Arc`-backed.
#[derive(Clone)]
pub struct CompositeResponsesProvider {
    /// The default Bedrock Converse-backed Responses provider.
    converse: Arc<dyn ResponsesProvider>,
    /// The bedrock-mantle (OpenAI-compatible) Responses provider.
    mantle: Arc<dyn ResponsesProvider>,
    /// The capability resolver used to pick the backend per request. Sharing the
    /// SAME `Arc` the inner providers hold keeps routing consistent with their
    /// own alias/region resolution.
    caps: Arc<dyn ModelCapabilities>,
    /// Whether mantle requests may be dispatched to the mantle provider.
    mantle_enabled: bool,
}

impl CompositeResponsesProvider {
    /// Construct the composite from its two inner providers and the shared
    /// capability resolver.
    #[must_use]
    pub fn new(
        converse: Arc<dyn ResponsesProvider>,
        mantle: Arc<dyn ResponsesProvider>,
        caps: Arc<dyn ModelCapabilities>,
        mantle_enabled: bool,
    ) -> Self {
        Self {
            converse,
            mantle,
            caps,
            mantle_enabled,
        }
    }

    fn backend(&self, req: &NormalizedResponsesRequest) -> ResponsesBackend {
        self.caps.responses_backend(&req.request.model)
    }

    /// Select the inner provider for a request by its configured backend.
    fn select(&self, backend: ResponsesBackend) -> &Arc<dyn ResponsesProvider> {
        match backend {
            ResponsesBackend::Mantle => &self.mantle,
            ResponsesBackend::Converse => &self.converse,
        }
    }

    fn mantle_unavailable_err(req: &NormalizedResponsesRequest) -> AppError {
        AppError::BadRequest(format!(
            "model {} requires a Bedrock API key (set AWS_BEARER_TOKEN_BEDROCK) to enable the GPT-5.x mantle backend; it is disabled on this instance",
            req.request.model
        ))
    }
}

#[async_trait]
impl ResponsesProvider for CompositeResponsesProvider {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        let backend = self.backend(req);
        if backend == ResponsesBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).respond(req).await
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        let backend = self.backend(req);
        if backend == ResponsesBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).respond_stream(req).await
    }

    async fn respond_raw_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<Option<RawResponsesStream>, AppError> {
        // MUST delegate: the trait default returns `Ok(None)`, so without this
        // override the mantle raw passthrough lane would never fire.
        let backend = self.backend(req);
        if backend == ResponsesBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).respond_raw_stream(req).await
    }
}

/// Resolve whether the mantle backend is effectively enabled at boot.
///
/// If mantle models are configured but `DISABLE_MANTLE=true` or no
/// `bedrock_api_key` is set, only mantle is disabled; the gateway still starts.
///
/// Soft rule: for each mantle model whose `available_regions` allow-list is set
/// and does NOT contain the gateway's running region, log a WARNING. This surfaces
/// a likely misconfiguration at boot without hard-failing — the per-request region
/// gate (T6) still returns a 400, and another region's deployment of the same
/// config is legitimate.
#[must_use]
pub fn resolve_mantle_enabled(config: &ModelCapabilityConfig, settings: &AppSettings) -> bool {
    let mantle_models: Vec<&crate::config::capabilities::ModelEntry> = config
        .models
        .iter()
        .filter(|e| e.params.responses_backend.as_deref() == Some(MANTLE_BACKEND))
        .collect();

    if settings.disable_mantle {
        tracing::warn!(
            "mantle backend disabled (DISABLE_MANTLE=true); GPT-5.x models will be unavailable"
        );
        return false;
    }

    if mantle_models.is_empty() {
        return false;
    }

    if settings.bedrock_api_key.is_none() {
        tracing::warn!(
            "mantle models are configured but no bedrock_api_key is set (AWS_BEARER_TOKEN_BEDROCK); GPT-5.x models are disabled — set the key to enable them. Other models are unaffected."
        );
        return false;
    }

    for entry in mantle_models {
        if let Some(regions) = entry.params.available_regions.as_deref() {
            if !regions.contains(&settings.aws_region) {
                tracing::warn!(
                    model = %entry.match_pattern,
                    region = %settings.aws_region,
                    "mantle model is not in its configured region allow-list for this region; \
                     requests for it will be rejected with 400 here"
                );
            }
        }
    }

    true
}

#[cfg(test)]
#[path = "composite_tests.rs"]
mod tests;
