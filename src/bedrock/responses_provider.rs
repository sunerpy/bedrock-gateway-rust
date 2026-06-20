//! Concrete [`ResponsesProvider`] for Amazon Bedrock.
//!
//! [`BedrockResponsesProvider`] holds the same collaborators as
//! [`crate::bedrock::provider::BedrockChatProvider`] (shared clients, capability
//! resolver, region-routing table, image resolver, and settings) because the
//! Responses-API translation reuses the same Bedrock Converse machinery as chat.
//!
//! The non-stream [`BedrockResponsesProvider::respond`] is fully wired (T10): it
//! composes the Responses input translation + request-level reasoning +
//! `cachePoint` decoration into a Bedrock Converse call (reusing the chat
//! provider's JSON→SDK bridge), then runs the pure non-stream mapper
//! [`crate::bedrock::responses_response::from_converse_output_to_responses`].
//!
//! [`BedrockResponsesProvider::respond_stream`] remains a seam until T11/T13.
//!
//! ## Shared converse-call reuse
//!
//! Rather than re-derive the JSON→SDK bridge, this module reuses the
//! `pub(crate)` builders in [`crate::bedrock::provider`]. The cachePoint budget
//! assembly mirrors the chat provider's tools→system→messages ordering with a
//! single shared checkpoint budget.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aws_sdk_bedrockruntime::operation::converse::{ConverseError, ConverseOutput};
use aws_sdk_bedrockruntime::operation::converse_stream::{
    ConverseStreamError, ConverseStreamOutput,
};
use serde_json::{json, Map, Value};

use crate::bedrock::cache_support::{send_with_cache_strip_retry, CacheSupportRegistry, SendError};
use crate::bedrock::capabilities::normalize_for_match;
use crate::bedrock::client::{region_config_override, BedrockClients};
use crate::bedrock::provider::{
    build_sdk_inference_config, build_sdk_messages, build_sdk_system, build_sdk_tool_config,
    converse_output_to_json,
};
use crate::bedrock::responses_response::from_converse_output_to_responses;
use crate::bedrock::responses_stream::converse_stream_to_openai_responses;
use crate::bedrock::responses_translate::{reasoning_outcome, to_responses_converse_input};
use crate::bedrock::translate::ImageResolver;
use crate::bedrock::{cache, provider};
use crate::config::{AppSettings, RegionRoutingConfig};
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, ResponsesProvider, ResponsesStream,
    RouteOverride,
};
use crate::error::AppError;
use crate::openai::responses_schema::{ResponsesRequest, ResponsesResponse, ResponsesTool};

/// Concrete [`ResponsesProvider`] backed by Amazon Bedrock Converse.
///
/// Mirrors [`crate::bedrock::provider::BedrockChatProvider`]'s dependency set so
/// the Responses surface can reuse the same Bedrock clients, capability
/// resolver, region routing, multimodal image resolver, and settings. Cheap to
/// clone (every field is `Arc`-wrapped or itself cheaply clonable).
#[derive(Clone)]
pub struct BedrockResponsesProvider {
    clients: BedrockClients,
    caps: Arc<dyn ModelCapabilities>,
    regions: Arc<RegionRoutingConfig>,
    image_resolver: Arc<dyn ImageResolver>,
    settings: Arc<AppSettings>,
    /// Shared negative cache of foundation ids that reject prompt caching — the
    /// SAME `Arc` the chat provider holds (wired in `build_app_state`), so a
    /// rejection seen on either surface suppresses cache-point injection on both.
    /// Consulted by the read-gate in [`Self::assemble`] and updated by the
    /// strip-and-retry safety net at the send points.
    cache_support: Arc<CacheSupportRegistry>,
}

impl BedrockResponsesProvider {
    /// Construct a provider from its collaborators (mirrors
    /// [`crate::bedrock::provider::BedrockChatProvider::new`]).
    pub fn new(
        clients: BedrockClients,
        caps: Arc<dyn ModelCapabilities>,
        regions: Arc<RegionRoutingConfig>,
        image_resolver: Arc<dyn ImageResolver>,
        settings: Arc<AppSettings>,
        cache_support: Arc<CacheSupportRegistry>,
    ) -> Self {
        Self {
            clients,
            caps,
            regions,
            image_resolver,
            settings,
            cache_support,
        }
    }

    /// Assemble the Bedrock Converse JSON slots (messages / system /
    /// inferenceConfig / toolConfig) from a Responses request, with reasoning
    /// fields and `cachePoint` decoration applied in the chat-canonical
    /// tools→system→messages order under one shared checkpoint budget.
    ///
    /// Returns `(messages, system, inference_config, additional_fields,
    /// tool_config)` as JSON values ready for the JSON→SDK bridge.
    ///
    /// `force_caching_off` is the cache safety net's strip path: when `true` (or
    /// when the read-gate finds the model already memoized as unsupported) the
    /// per-zone enablement is forced off so no `cachePoint` is injected and
    /// `cache_points_injected` is returned `false` — the same no-cachePoint
    /// assembly the master-switch-off path produces (a re-assemble, not a
    /// surgical edit of a built SDK struct).
    async fn assemble(
        &self,
        req: &ResponsesRequest,
        resolved: &str,
        force_caching_off: bool,
    ) -> Result<AssembledConverse, AppError> {
        let caps = self.caps.as_ref();

        let parsed =
            to_responses_converse_input(req, resolved, self.image_resolver.as_ref(), caps).await?;
        let mut messages = parsed.messages;
        let mut system = parsed.system;

        let reasoning = reasoning_outcome(req, resolved, caps);

        // inferenceConfig: maxTokens (reasoning side-signal wins) + temperature /
        // topP (topP dropped when reasoning requests it).
        let mut inference = Map::new();
        let effective_max = reasoning.max_tokens.or(req.max_output_tokens);
        if let Some(max_tokens) = effective_max {
            inference.insert("maxTokens".to_string(), Value::from(max_tokens));
        }
        if let Some(temp) = req.temperature {
            inference.insert("temperature".to_string(), Value::from(temp));
        }
        if let Some(top_p) = req.top_p {
            if !reasoning.drop_top_p {
                inference.insert("topP".to_string(), Value::from(top_p));
            }
        }
        let inference_config = Value::Object(inference);

        let additional_fields = if reasoning.additional_model_request_fields.is_empty() {
            None
        } else {
            Some(Value::Object(reasoning.additional_model_request_fields))
        };

        // toolConfig from the Responses flattened-function tools (the rejection
        // matrix in to_responses_converse_input already vetoed built-in tools).
        let mut tool_config = build_responses_tool_config(req);

        // cachePoint decoration: tools → system → messages, one shared budget.
        let global_default = self.settings.enable_prompt_caching;
        // Read-gate: a model already memoized as caching-unsupported (or an
        // explicit strip-retry) forces all zones off so this re-assembly never
        // re-injects the cachePoints that were just rejected.
        let caching_off = force_caching_off
            || self
                .cache_support
                .is_unsupported(&normalize_for_match(resolved));
        let enabled = !caching_off && global_default;
        const DEFAULT_MAX_CACHE_CHECKPOINTS: u32 = 4;
        let max_checkpoints = Some(
            caps.max_cache_checkpoints(resolved)
                .unwrap_or(DEFAULT_MAX_CACHE_CHECKPOINTS),
        );
        let mut used: u32 = 0;

        if let Some(Value::Object(tc)) = tool_config.as_mut() {
            if let Some(tools_val) = tc.remove("tools") {
                let decorated = cache::decorate_tools(
                    tools_val,
                    resolved,
                    caps,
                    enabled,
                    used,
                    max_checkpoints,
                );
                used += provider::count_cache_points(&decorated);
                tc.insert("tools".to_string(), decorated);
            }
        }

        let decorated_system = cache::decorate_system_blocks(system, resolved, caps, enabled);
        used += provider::count_cache_points(&decorated_system);
        system = decorated_system;

        messages =
            cache::decorate_messages(messages, resolved, caps, enabled, used, max_checkpoints);
        used += provider::count_cache_points(&messages);

        Ok(AssembledConverse {
            messages,
            system,
            inference_config,
            additional_fields,
            tool_config,
            cache_points_injected: used > 0,
        })
    }

    /// Build the typed SDK `converse` call from an [`AssembledConverse`] and send
    /// it (applying the per-request region override at the call site).
    ///
    /// `request_model` is the ORIGINAL incoming model id (cross-region prefix
    /// intact). It is what reaches Bedrock and keys the region table — mirroring
    /// the chat provider. The resolved (prefix-stripped) foundation id is for
    /// capability matching only; sending it to Bedrock triggers an on-demand 400.
    /// See [`Self::outbound_model_id`].
    ///
    /// Returns the raw service error in a [`SendError`] so the shared cache
    /// safety net can inspect `.code()`/`.message()` before mapping; JSON→SDK
    /// build failures surface as [`SendError::App`] (never a cache rejection).
    async fn send_converse(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<ConverseOutput, SendError<ConverseError>> {
        let route = self.regions.route_for(request_model);
        let model_id = Self::outbound_model_id(request_model, route.as_ref());

        let messages = build_sdk_messages(&assembled.messages).map_err(SendError::App)?;
        let system = build_sdk_system(&assembled.system).map_err(SendError::App)?;
        let inference_config = build_sdk_inference_config(&assembled.inference_config);

        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock converse (responses)"
        );

        let mut call = self
            .clients
            .runtime
            .converse()
            .model_id(&model_id)
            .set_messages(Some(messages))
            .set_system(Some(system))
            .inference_config(inference_config);

        if let Some(fields) = &assembled.additional_fields {
            call = call.additional_model_request_fields(provider::json_to_document(fields));
        }
        if let Some(tc) = &assembled.tool_config {
            call = call.tool_config(build_sdk_tool_config(tc).map_err(SendError::App)?);
        }

        if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        } else {
            call.send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        }
    }

    /// Build the typed SDK `converse_stream` call from an [`AssembledConverse`]
    /// and send it. Mirrors [`Self::send_converse`]; `request_model` (the
    /// original prefixed id) reaches Bedrock and keys the region table for the
    /// same on-demand-400 reason. The rejection surfaces at `.send()` BEFORE any
    /// stream event (confirmed live), so the strip-and-retry safety net is
    /// identical to the non-stream path.
    async fn send_converse_stream(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<ConverseStreamOutput, SendError<ConverseStreamError>> {
        let route = self.regions.route_for(request_model);
        let model_id = Self::outbound_model_id(request_model, route.as_ref());

        let messages = build_sdk_messages(&assembled.messages).map_err(SendError::App)?;
        let system = build_sdk_system(&assembled.system).map_err(SendError::App)?;
        let inference_config = build_sdk_inference_config(&assembled.inference_config);

        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock converse_stream (responses)"
        );

        let mut call = self
            .clients
            .runtime
            .converse_stream()
            .model_id(&model_id)
            .set_messages(Some(messages))
            .set_system(Some(system))
            .inference_config(inference_config);

        if let Some(fields) = &assembled.additional_fields {
            call = call.additional_model_request_fields(provider::json_to_document(fields));
        }
        if let Some(tc) = &assembled.tool_config {
            call = call.tool_config(build_sdk_tool_config(tc).map_err(SendError::App)?);
        }

        if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        } else {
            call.send()
                .await
                .map_err(|e| SendError::Service(e.into_service_error()))
        }
    }

    /// Pick the model id sent to Bedrock: a region override's rewritten id when
    /// one matched, else the original `request_model` UNCHANGED (prefix intact).
    /// Never the resolved foundation id — that strips the cross-region prefix and
    /// makes Bedrock reject on-demand invocation. Mirrors the chat provider.
    fn outbound_model_id(request_model: &str, route: Option<&RouteOverride>) -> String {
        route
            .map(|r| r.rewritten_model_id.clone())
            .unwrap_or_else(|| request_model.to_string())
    }
}

/// The assembled Bedrock Converse JSON slots produced by
/// [`BedrockResponsesProvider::assemble`].
struct AssembledConverse {
    messages: Value,
    system: Value,
    inference_config: Value,
    additional_fields: Option<Value>,
    tool_config: Option<Value>,
    /// Whether any `cachePoint` landed across the tools/system/messages zones —
    /// consumed by the cache safety net at the send points (read-gate strip).
    cache_points_injected: bool,
}

/// Build a Bedrock `toolConfig` JSON object from the Responses request's
/// flattened-function tools, mirroring the `{"tools": [{"toolSpec": ...}]}`
/// shape the chat path produces. Returns `None` when no tools are present.
fn build_responses_tool_config(req: &ResponsesRequest) -> Option<Value> {
    let tools = req.tools.as_ref()?;
    if tools.is_empty() {
        return None;
    }
    let specs: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let ResponsesTool::Function {
                name,
                description,
                parameters,
                ..
            } = tool;
            let description = match description {
                Some(d) => Value::String(d.clone()),
                None => Value::Null,
            };
            let parameters = parameters
                .clone()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            json!({
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": { "json": parameters },
                }
            })
        })
        .collect();
    Some(json!({ "tools": specs }))
}

/// Generate a `resp_`-prefixed response id (mirrors the chat `chatcmpl-` id
/// generation pattern: a Unix-nanos hex suffix, dependency-free).
fn resp_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("resp_{nanos:x}")
}

#[async_trait::async_trait]
impl ResponsesProvider for BedrockResponsesProvider {
    async fn respond(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesResponse, AppError> {
        let resolved = &req.resolved_model;
        let request_model = &req.request.model;
        let assembled = self.assemble(&req.request, resolved, false).await?;
        let normalized = normalize_for_match(resolved);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            assembled.cache_points_injected,
            || self.send_converse(request_model, &assembled),
            || async {
                let retry = self
                    .assemble(&req.request, resolved, true)
                    .await
                    .map_err(SendError::App)?;
                self.send_converse(request_model, &retry).await
            },
        )
        .await?;

        let output_json = converse_output_to_json(&output);
        let response_id = resp_id();
        from_converse_output_to_responses(
            &output_json,
            &req.request,
            &req.request.model,
            &response_id,
        )
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        let resolved = &req.resolved_model;
        let request_model = &req.request.model;
        let assembled = self.assemble(&req.request, resolved, false).await?;
        let normalized = normalize_for_match(resolved);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            assembled.cache_points_injected,
            || self.send_converse_stream(request_model, &assembled),
            || async {
                let retry = self
                    .assemble(&req.request, resolved, true)
                    .await
                    .map_err(SendError::App)?;
                self.send_converse_stream(request_model, &retry).await
            },
        )
        .await?;

        Ok(converse_stream_to_openai_responses(
            output,
            req.request.model.clone(),
            resp_id(),
            req.request.clone(),
            req.request_id.clone(),
            req.received_at,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::responses_schema::ResponsesInput;
    use std::collections::HashMap;

    fn base_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "incoming".to_string(),
            input: ResponsesInput::Text("hi".to_string()),
            instructions: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            stream: None,
            reasoning: None,
            text: None,
            include: None,
            metadata: None,
            parallel_tool_calls: None,
            store: None,
            previous_response_id: None,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn resp_id_has_responses_prefix() {
        assert!(resp_id().starts_with("resp_"));
    }

    #[test]
    fn tool_config_built_from_flattened_function_tools() {
        let mut req = base_request();
        req.tools = Some(vec![ResponsesTool::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: Some(json!({ "type": "object", "properties": {} })),
            strict: None,
        }]);
        let tc = build_responses_tool_config(&req).expect("tool config");
        let specs = tc["tools"].as_array().expect("tools array");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0]["toolSpec"]["name"], "get_weather");
        assert_eq!(specs[0]["toolSpec"]["description"], "Get weather");
        assert!(specs[0]["toolSpec"]["inputSchema"]["json"].is_object());
    }

    #[test]
    fn tool_config_none_when_no_tools() {
        assert!(build_responses_tool_config(&base_request()).is_none());
    }

    /// Regression for the cross-region-prefix 400: when the incoming model carries
    /// a geo prefix (`us.anthropic.claude-...`) and the resolved foundation id has
    /// it stripped (`anthropic.claude-...`), the id sent to Bedrock MUST be the
    /// prefixed request model — sending the bare resolved id triggers Bedrock's
    /// on-demand-throughput 400. With no region override the result is the
    /// request model verbatim.
    #[test]
    fn outbound_model_id_uses_prefixed_request_model_not_resolved() {
        let request_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";
        let resolved = "anthropic.claude-sonnet-4-5-20250929-v1:0";

        let outbound = BedrockResponsesProvider::outbound_model_id(request_model, None);

        assert_eq!(outbound, request_model);
        assert_ne!(outbound, resolved);
    }

    /// A matching region override wins and supplies its rewritten id (the same
    /// precedence the chat provider applies).
    #[test]
    fn outbound_model_id_prefers_region_override() {
        let request_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";
        let route = RouteOverride {
            region: "eu-central-1".to_string(),
            rewritten_model_id: "eu.anthropic.claude-sonnet-4-5-20250929-v1:0".to_string(),
        };

        let outbound = BedrockResponsesProvider::outbound_model_id(request_model, Some(&route));

        assert_eq!(outbound, "eu.anthropic.claude-sonnet-4-5-20250929-v1:0");
    }

    /// The streaming seam is wired (T11): `respond_stream` now assembles and
    /// invokes `converse_stream`. Without AWS credentials the upstream call
    /// fails, so this still returns `Err` — it guards the constructor +
    /// dependency set + the wired call path without AWS creds. The happy-path
    /// event sequence is unit-tested in `responses_stream::tests`; the live
    /// path is exercised in T15.
    #[tokio::test]
    async fn stream_path_invokes_converse_stream() {
        use crate::bedrock::client::{build_aws_config, BedrockClients};
        use crate::bedrock::translate::ReqwestImageResolver;
        use crate::config::ModelCapabilityConfig;

        let settings = Arc::new(AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: "us-west-2".to_string(),
            default_model: "m".to_string(),
            default_embedding_model: "e".to_string(),
            enable_cross_region_inference: false,
            enable_application_inference_profiles: false,
            enable_prompt_caching: false,
            api_key: Some("k".to_string()),
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: None,
            bind_addr: "127.0.0.1".to_string(),
            port: 0,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
        });
        let aws_config = build_aws_config(&settings).await;
        let clients = BedrockClients::new(&aws_config);
        let caps: Arc<dyn ModelCapabilities> =
            Arc::new(crate::bedrock::capabilities::ConfigModelCapabilities::new(
                ModelCapabilityConfig::default(),
            ));
        let regions = Arc::new(RegionRoutingConfig::default());
        let image_resolver = Arc::new(ReqwestImageResolver::new(|_: &str| false));

        let provider: Arc<dyn ResponsesProvider> = Arc::new(BedrockResponsesProvider::new(
            clients,
            caps,
            regions,
            image_resolver,
            settings,
            Arc::new(crate::bedrock::cache_support::CacheSupportRegistry::new()),
        ));

        let req = NormalizedResponsesRequest {
            request: base_request(),
            resolved_model: "resolved".to_string(),
            request_id: Arc::from("req-test"),
            received_at: std::time::Instant::now(),
        };
        assert!(
            provider.respond_stream(&req).await.is_err(),
            "stream path errors without AWS creds"
        );
    }
}
