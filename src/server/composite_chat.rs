//! Composite [`ChatProvider`] that routes by backend + startup validation.
//!
//! The chat-surface analogue of [`crate::server::composite::CompositeResponsesProvider`].
//! The gateway serves `/chat/completions` from three config-selected backends:
//! the default Bedrock Converse path, the OpenAI-compatible bedrock-mantle
//! upstream
//! ([`crate::bedrock::mantle_chat_provider::MantleChatProvider`]) for the gpt-oss
//! family, and a Responses-to-Chat adapter for Responses-only models.
//! [`CompositeChatProvider`] hides them behind the single
//! `Arc<dyn ChatProvider>` that [`crate::server::state::AppState`] holds.
//!
//! ## Routing
//!
//! Routing is purely config-driven (zero model-name branching): every method
//! consults [`ModelCapabilities::chat_backend`] for the requested model (which
//! resolves the alias internally) and dispatches to the matching inner provider.
//!
//! Both raw lanes ([`ChatProvider::chat_raw_stream`] /
//! [`ChatProvider::chat_raw_nonstream`]) are overridden and delegated to the
//! selected inner provider — otherwise the trait defaults (`None`) would win and
//! the mantle raw passthrough would never fire.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use crate::config::capabilities::ModelCapabilityConfig;
use crate::config::AppSettings;
use crate::domain::{
    ChatBackend, ChatProvider, ChatStream, ModelCapabilities, NormalizedChatRequest, RawChatStream,
};
use crate::error::AppError;
use crate::openai::schema::ChatResponse;

/// The config value of `params.chat_backend` that selects the mantle upstream.
const MANTLE_BACKEND: &str = "mantle";

/// A [`ChatProvider`] that dispatches each request to one of three inner providers
/// based on the model's configured chat backend.
///
/// Cheap to clone — every field is `Arc`-backed.
#[derive(Clone)]
pub struct CompositeChatProvider {
    /// The default Bedrock Converse-backed chat provider.
    converse: Arc<dyn ChatProvider>,
    /// The bedrock-mantle (OpenAI-compatible) chat provider.
    mantle: Arc<dyn ChatProvider>,
    /// Protocol adapter backed by the model's configured Responses provider.
    responses: Arc<dyn ChatProvider>,
    /// The capability resolver used to pick the backend per request.
    caps: Arc<dyn ModelCapabilities>,
    /// Whether mantle requests may be dispatched to the mantle provider.
    mantle_enabled: bool,
}

impl CompositeChatProvider {
    /// Construct the composite from its two inner providers and the shared
    /// capability resolver.
    #[must_use]
    pub fn new(
        converse: Arc<dyn ChatProvider>,
        mantle: Arc<dyn ChatProvider>,
        responses: Arc<dyn ChatProvider>,
        caps: Arc<dyn ModelCapabilities>,
        mantle_enabled: bool,
    ) -> Self {
        Self {
            converse,
            mantle,
            responses,
            caps,
            mantle_enabled,
        }
    }

    fn backend(&self, req: &NormalizedChatRequest) -> ChatBackend {
        self.caps.chat_backend(&req.request.model)
    }

    fn select(&self, backend: ChatBackend) -> &Arc<dyn ChatProvider> {
        match backend {
            ChatBackend::Mantle => &self.mantle,
            ChatBackend::Responses => &self.responses,
            ChatBackend::Converse => &self.converse,
        }
    }

    fn mantle_unavailable_err(req: &NormalizedChatRequest) -> AppError {
        AppError::BadRequest(format!(
            "model {} requires a Bedrock API key (set AWS_BEARER_TOKEN_BEDROCK) to enable the mantle-chat backend; it is disabled on this instance",
            req.request.model
        ))
    }
}

#[async_trait]
impl ChatProvider for CompositeChatProvider {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        let backend = self.backend(req);
        if backend == ChatBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).chat(req).await
    }

    async fn chat_stream(&self, req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        let backend = self.backend(req);
        if backend == ChatBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).chat_stream(req).await
    }

    async fn chat_raw_stream(
        &self,
        req: &NormalizedChatRequest,
    ) -> Result<Option<RawChatStream>, AppError> {
        // MUST delegate: the trait default returns `Ok(None)`, so without this
        // override the mantle raw passthrough lane would never fire.
        let backend = self.backend(req);
        if backend == ChatBackend::Mantle && !self.mantle_enabled {
            return Err(Self::mantle_unavailable_err(req));
        }
        self.select(backend).chat_raw_stream(req).await
    }

    async fn chat_raw_nonstream(
        &self,
        req: &NormalizedChatRequest,
    ) -> Option<Result<Bytes, AppError>> {
        let backend = self.backend(req);
        if backend == ChatBackend::Mantle && !self.mantle_enabled {
            return None;
        }
        self.select(backend).chat_raw_nonstream(req).await
    }
}

/// Resolve whether the mantle-chat backend is effectively enabled at boot.
///
/// The chat analogue of [`crate::server::composite::resolve_mantle_enabled`],
/// filtering on `chat_backend == "mantle"`. If mantle-chat models are configured
/// but `DISABLE_MANTLE=true` or no `bedrock_api_key` is set, only mantle-chat is
/// disabled; the gateway still starts. A region allow-list mismatch emits a
/// WARNING but never hard-fails (the per-request gate returns 400 instead).
#[must_use]
pub fn resolve_mantle_chat_enabled(config: &ModelCapabilityConfig, settings: &AppSettings) -> bool {
    let mantle_models: Vec<&crate::config::capabilities::ModelEntry> = config
        .models
        .iter()
        .filter(|e| e.params.chat_backend.as_deref() == Some(MANTLE_BACKEND))
        .collect();

    if settings.disable_mantle {
        tracing::warn!(
            "mantle-chat backend disabled (DISABLE_MANTLE=true); mantle chat models will be unavailable"
        );
        return false;
    }

    if mantle_models.is_empty() {
        return false;
    }

    if settings.bedrock_api_key.is_none() {
        tracing::warn!(
            "mantle-chat models are configured but no bedrock_api_key is set (AWS_BEARER_TOKEN_BEDROCK); those chat models are disabled — set the key to enable them. Other models are unaffected."
        );
        return false;
    }

    for entry in mantle_models {
        if let Some(regions) = entry.params.available_regions.as_deref() {
            if !regions.contains(&settings.aws_region) {
                tracing::warn!(
                    model = %entry.match_pattern,
                    region = %settings.aws_region,
                    "mantle-chat model is not in its configured region allow-list for this region; \
                     requests for it will be rejected with 400 here"
                );
            }
        }
    }

    true
}

#[cfg(test)]
#[path = "composite_chat_tests.rs"]
mod tests;
