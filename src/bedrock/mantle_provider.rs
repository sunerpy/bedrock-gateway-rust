//! [`ResponsesProvider`] for the Bedrock **mantle** (OpenAI-compatible) upstream.
//!
//! [`MantleResponsesProvider`] serves the Responses API for models whose config
//! entry declares `params.responses_backend = "mantle"` (e.g. the GPT-5.x
//! family). Unlike [`crate::bedrock::responses_provider::BedrockResponsesProvider`]
//! — which translates to/from Bedrock Converse — this provider is a **thin
//! routing shim** over the byte-oriented [`MantleClient`] (T4): mantle's upstream
//! already speaks the OpenAI Responses wire format, so this provider forwards the
//! client's verbatim request body and returns the upstream's verbatim response.
//!
//! ## What it does per request
//!
//! 1. **Region gate FIRST** (before any HTTP call): if the model declares a
//!    region allow-list ([`ModelCapabilities::model_regions`], T2) that does not
//!    contain the gateway's running region, fail with [`AppError::BadRequest`].
//! 2. **Model rewrite**: patch ONLY the top-level `"model"` key of the captured
//!    raw request body to the canonical foundation id
//!    ([`ModelCapabilities::resolve_foundation`], T2), keeping every other field
//!    byte-stable (this is why T5 captured `raw_body` — to preserve
//!    unknown/future fields). Forward that body upstream.
//! 3. **Non-stream** ([`Self::respond`]): call
//!    [`MantleClient::responses_nonstream`], deserialize the mantle JSON into a
//!    [`ResponsesResponse`], and re-normalize `usage` through the shared
//!    [`compute_token_usage`] helper (single source of truth — no hand-rolled
//!    token math).
//! 4. **Stream** ([`Self::respond_raw_stream`], the raw lane T5 added): open the
//!    upstream SSE via [`MantleClient::responses_stream`] and forward the bytes
//!    verbatim. The handler (T5) consults this BEFORE the typed
//!    [`Self::respond_stream`], so the raw lane is the happy path. Pre-stream
//!    failures are returned directly from the raw lane before headers are sent;
//!    the typed method remains only an invariant guard.
//!
//! ## Logging discipline (AGENTS.md §4)
//!
//! The raw request body, the response body, and the bearer are NEVER logged.
//! Only structured metadata (region, streaming flag) appears in traces.

use std::sync::Arc;

use bytes::Bytes;
use futures::stream::StreamExt;
use serde_json::Value;

use crate::bedrock::mantle_client::{MantleClient, ResponsesStreamTerminal};
use crate::bedrock::tokens::compute_token_usage;
use crate::config::AppSettings;
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, RawResponsesStream, ResponsesProvider,
    ResponsesStream,
};
use crate::error::AppError;
use crate::openai::responses_schema::ResponsesResponse;

/// A [`ResponsesProvider`] that routes mantle-backed models (GPT-5.x) to the
/// OpenAI-compatible bedrock-mantle upstream via [`MantleClient`].
///
/// Cheap to clone — [`MantleClient`] is reference-counted internally and the
/// other fields are `Arc`-wrapped.
#[derive(Clone)]
pub struct MantleResponsesProvider {
    /// The byte-oriented mantle HTTP client (T4).
    client: MantleClient,
    /// Capability resolver (T2): backend selection, region allow-list, and
    /// alias→canonical foundation resolution. The ONLY source of model routing
    /// knowledge — no model-name string branching lives here.
    caps: Arc<dyn ModelCapabilities>,
    /// Application settings; `aws_region` is the gateway's running region used
    /// for the region gate and as the upstream `{region}` substitution.
    settings: Arc<AppSettings>,
}

impl MantleResponsesProvider {
    /// Construct a provider from its collaborators.
    ///
    /// `client` is a fully-built [`MantleClient`] (the caller — `build_app_state`
    /// in T7 — wires the shared `reqwest::Client`, the
    /// `mantle_base_url_template`, and the `bedrock_api_key` bearer into it).
    /// `caps` is the shared capability resolver; `settings` carries the running
    /// `aws_region`.
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
    /// region. Returns the region to call (`settings.aws_region`) on success, or
    /// [`AppError::BadRequest`] when the model is gated out of this region.
    ///
    /// `None` from [`ModelCapabilities::model_regions`] means no gate (served
    /// everywhere). Runs BEFORE any HTTP call.
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
    ///
    /// The body is parsed to a [`serde_json::Value`], its `model` key is set to
    /// `resolve_foundation(model)`, and it is reserialized. A malformed body
    /// surfaces as [`AppError::BadRequest`] (it never reached the upstream).
    fn rewrite_model(&self, model: &str, raw_body: &Bytes) -> Result<Bytes, AppError> {
        let canonical = self.caps.resolve_foundation(model);
        let mut value: Value = serde_json::from_slice(raw_body)
            .map_err(|e| AppError::BadRequest(format!("invalid responses request body: {e}")))?;
        match value.as_object_mut() {
            Some(obj) => {
                obj.insert("model".to_string(), Value::String(canonical));
            }
            None => {
                return Err(AppError::BadRequest(
                    "responses request body must be a JSON object".to_string(),
                ));
            }
        }
        let bytes = serde_json::to_vec(&value)
            .map_err(|e| AppError::Internal(format!("failed to reserialize request body: {e}")))?;
        Ok(Bytes::from(bytes))
    }

    /// Run the shared pre-flight (region gate + model rewrite) for a request,
    /// returning the `(region, rewritten_body)` pair forwarded upstream.
    fn preflight(&self, req: &NormalizedResponsesRequest) -> Result<(String, Bytes), AppError> {
        let model = &req.request.model;
        let region = self.region_gate(model)?;
        let body = self.rewrite_model(model, &req.raw_body)?;
        Ok((region, body))
    }
}

/// Re-normalize a mantle [`ResponsesResponse`]'s `usage` through the shared
/// [`compute_token_usage`] helper so token accounting matches the rest of the
/// gateway (single source of truth). Mantle reports OpenAI-native usage, so
/// `cacheRead`/`cacheWrite` are `0`.
fn normalize_usage(mut resp: ResponsesResponse) -> ResponsesResponse {
    let counts = compute_token_usage(resp.usage.input_tokens, resp.usage.output_tokens, 0, 0);
    resp.usage.input_tokens = counts.prompt_tokens;
    resp.usage.output_tokens = counts.completion_tokens;
    resp.usage.total_tokens = counts.total_tokens;
    resp
}

#[async_trait::async_trait]
impl ResponsesProvider for MantleResponsesProvider {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        let (region, body) = self.preflight(req)?;

        tracing::debug!(region = %region, "invoking mantle responses (non-stream)");

        let bytes = self.client.responses_nonstream(&region, body).await?;
        let resp: ResponsesResponse = serde_json::from_slice(&bytes).map_err(|e| {
            AppError::UpstreamBedrock(format!("failed to parse mantle response: {e}"))
        })?;
        Ok(normalize_usage(resp))
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        let _ = req;
        Err(AppError::Internal(
            "mantle streaming must use the raw passthrough lane".to_string(),
        ))
    }

    async fn respond_raw_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<Option<RawResponsesStream>, AppError> {
        // Preserve pre-flight/connect errors so callers can return the original
        // envelope without issuing a second upstream request.
        let (region, body) = self.preflight(req)?;

        tracing::debug!(region = %region, "invoking mantle responses (raw stream)");

        let request_id = Arc::clone(&req.request_id);
        let model = req.request.model.clone();
        let reasoning_effort = req.request.reasoning_effort_label().to_string();
        let started_at = req.received_at;
        let observer = Box::new(move |terminal: ResponsesStreamTerminal| {
            let reasoning_tokens = terminal.reasoning_tokens.unwrap_or(0);
            let reasoning_used = reasoning_tokens > 0;
            let failed = matches!(terminal.event_type.as_str(), "response.failed" | "error");
            if failed {
                tracing::warn!(
                    request_id = %request_id,
                    model = %model,
                    status = terminal.status.as_deref().unwrap_or("unknown"),
                    terminal_event = %terminal.event_type,
                    upstream_error_code = terminal.error_code.as_deref().unwrap_or("unknown"),
                    upstream_error_type = terminal.error_type.as_deref().unwrap_or("unknown"),
                    reasoning_effort = %reasoning_effort,
                    reasoning_used,
                    reasoning_tokens,
                    reasoning_usage_available = terminal.reasoning_tokens.is_some(),
                    input_tokens = ?terminal.input_tokens,
                    output_tokens = ?terminal.output_tokens,
                    total_tokens = ?terminal.total_tokens,
                    duration_ms = started_at.elapsed().as_millis(),
                    "responses raw streaming failed"
                );
            } else {
                tracing::info!(
                    request_id = %request_id,
                    model = %model,
                    status = terminal.status.as_deref().unwrap_or("unknown"),
                    terminal_event = %terminal.event_type,
                    reasoning_effort = %reasoning_effort,
                    reasoning_used,
                    reasoning_tokens,
                    reasoning_usage_available = terminal.reasoning_tokens.is_some(),
                    input_tokens = ?terminal.input_tokens,
                    output_tokens = ?terminal.output_tokens,
                    total_tokens = ?terminal.total_tokens,
                    duration_ms = started_at.elapsed().as_millis(),
                    "responses raw streaming completed"
                );
            }
        });
        let stream = self
            .client
            .responses_stream_with_observer(&region, body, observer)
            .await?;
        Ok(Some(stream.boxed()))
    }
}

#[cfg(test)]
#[path = "mantle_provider_tests.rs"]
mod tests;
