//! [`ChatProvider`] for the Bedrock **mantle** (OpenAI-compatible) chat surface.
//!
//! [`MantleChatProvider`] serves `/chat/completions` for models whose config
//! entry declares `params.chat_backend = "mantle"` (e.g. the gpt-oss family).
//! Like [`crate::bedrock::mantle_provider::MantleResponsesProvider`] it is a thin
//! routing shim over the byte-oriented [`MantleClient`]: mantle's upstream already
//! speaks the OpenAI chat wire format, so this provider forwards the client's
//! verbatim request body and returns the upstream's verbatim response.
//!
//! ## Raw bytes end to end
//!
//! Per the LOCKED design, BOTH the streaming and non-streaming happy paths are
//! raw `Bytes` passthrough — NO deserialization, NO usage re-normalization. The
//! non-standard `delta.reasoning` and per-chunk `obfuscation` fields survive
//! because the bytes are never parsed. The typed [`Self::chat`] / [`Self::chat_stream`]
//! methods are UNREACHABLE fallbacks that return [`AppError::Internal`]; the
//! composite dispatcher consults the raw methods first.
//!
//! ## Logging discipline (AGENTS.md §4)
//!
//! The raw request body, the response body, and the bearer are NEVER logged.
//! Only structured metadata (region, streaming flag) appears in traces.

use std::sync::Arc;

use bytes::Bytes;
use futures::stream::StreamExt;
use serde_json::Value;

use crate::bedrock::mantle_client::MantleClient;
use crate::config::AppSettings;
use crate::domain::{
    ChatProvider, ChatStream, ModelCapabilities, NormalizedChatRequest, RawChatStream,
};
use crate::error::AppError;
use crate::openai::schema::ChatResponse;

/// A [`ChatProvider`] that routes mantle-backed chat models (gpt-oss) to the
/// OpenAI-compatible bedrock-mantle upstream via [`MantleClient`].
///
/// Cheap to clone — [`MantleClient`] is reference-counted internally and the
/// other fields are `Arc`-wrapped.
#[derive(Clone)]
pub struct MantleChatProvider {
    /// The byte-oriented mantle HTTP client.
    client: MantleClient,
    /// Capability resolver: region allow-list + alias→canonical resolution. The
    /// ONLY source of model routing knowledge — no model-name string branching.
    caps: Arc<dyn ModelCapabilities>,
    /// Application settings; `aws_region` is the gateway's running region used
    /// for the region gate and as the upstream `{region}` substitution.
    settings: Arc<AppSettings>,
}

impl MantleChatProvider {
    /// Construct a provider from its collaborators.
    #[must_use]
    pub fn new(
        client: MantleClient,
        caps: Arc<dyn ModelCapabilities>,
        settings: Arc<AppSettings>,
    ) -> Self {
        Self {
            client,
            caps,
            settings,
        }
    }

    /// Enforce the per-model region allow-list against the gateway's running
    /// region. Returns the region to call on success, or [`AppError::BadRequest`]
    /// when the model is gated out of this region. Runs BEFORE any HTTP call.
    fn region_gate(&self, model: &str) -> Result<String, AppError> {
        let region = &self.settings.aws_region;
        if let Some(regions) = self.caps.model_regions(model) {
            if !regions.contains(region) {
                return Err(AppError::BadRequest(format!(
                    "model {model} is not available in region {region}"
                )));
            }
        }
        Ok(region.clone())
    }

    /// Patch ONLY the top-level `"model"` key of the captured raw request body to
    /// the canonical foundation id, keeping every other field byte-stable.
    fn rewrite_model(&self, model: &str, raw_body: &Bytes) -> Result<Bytes, AppError> {
        let canonical = self.caps.resolve_foundation(model);
        let mut value: Value = serde_json::from_slice(raw_body)
            .map_err(|e| AppError::BadRequest(format!("invalid chat request body: {e}")))?;
        match value.as_object_mut() {
            Some(obj) => {
                obj.insert("model".to_string(), Value::String(canonical));
            }
            None => {
                return Err(AppError::BadRequest(
                    "chat request body must be a JSON object".to_string(),
                ));
            }
        }
        let bytes = serde_json::to_vec(&value)
            .map_err(|e| AppError::Internal(format!("failed to reserialize request body: {e}")))?;
        Ok(Bytes::from(bytes))
    }

    /// Run the shared pre-flight (region gate + model rewrite), returning the
    /// `(region, rewritten_body)` pair forwarded upstream.
    fn preflight(&self, req: &NormalizedChatRequest) -> Result<(String, Bytes), AppError> {
        let model = &req.request.model;
        let region = self.region_gate(model)?;
        let body = self.rewrite_model(model, &req.raw_body)?;
        Ok((region, body))
    }

    /// The shared "raw passthrough lane only" error for the typed fallbacks.
    fn typed_fallback_err() -> AppError {
        AppError::Internal("mantle chat must use the raw passthrough lane".to_string())
    }
}

#[async_trait::async_trait]
impl ChatProvider for MantleChatProvider {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        // Typed fallback — UNREACHABLE happy path. Surface a pre-flight failure,
        // otherwise signal the routing error. NEVER deserializes mantle bytes.
        let (_region, _body) = self.preflight(req)?;
        Err(Self::typed_fallback_err())
    }

    async fn chat_stream(&self, req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        let _ = req;
        Err(Self::typed_fallback_err())
    }

    async fn chat_raw_nonstream(
        &self,
        req: &NormalizedChatRequest,
    ) -> Option<Result<Bytes, AppError>> {
        // The non-stream lane has not yet committed a 200, so a pre-flight error
        // CAN be enveloped — return `Some(Err(..))` so the handler surfaces it.
        let (region, body) = match self.preflight(req) {
            Ok(pair) => pair,
            Err(e) => return Some(Err(e)),
        };
        tracing::debug!(region = %region, "invoking mantle chat (non-stream)");
        Some(self.client.chat_nonstream(&region, body).await)
    }

    async fn chat_raw_stream(
        &self,
        req: &NormalizedChatRequest,
    ) -> Result<Option<RawChatStream>, AppError> {
        // Preserve pre-flight/connect errors so the handler can return the
        // original envelope without issuing a second upstream request.
        let (region, body) = self.preflight(req)?;
        tracing::debug!(region = %region, "invoking mantle chat (raw stream)");
        let stream = self.client.chat_stream(&region, body).await?;
        Ok(Some(stream.boxed()))
    }
}

#[cfg(test)]
#[path = "mantle_chat_provider_tests.rs"]
mod tests;
