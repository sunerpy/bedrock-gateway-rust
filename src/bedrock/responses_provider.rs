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
use aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput;
use aws_sdk_bedrockruntime::primitives::Blob;
use serde_json::{json, Map, Value};

use crate::bedrock::anthropic_messages::{
    build_messages_body, messages_response_to_converse_output, MessagesBodyInput,
};
use crate::bedrock::cache_support::{send_with_cache_strip_retry, CacheSupportRegistry, SendError};
use crate::bedrock::capabilities::normalize_for_match;
use crate::bedrock::client::{region_config_override, BedrockClients};
use crate::bedrock::provider::{
    build_sdk_inference_config, build_sdk_messages, build_sdk_output_config, build_sdk_system,
    build_sdk_tool_config, converse_output_to_json,
};
use crate::bedrock::responses_response::from_converse_output_to_responses_with_tools;
use crate::bedrock::responses_stream::{
    converse_stream_to_openai_responses, invoke_stream_to_openai_responses, ResponsesStreamRuntime,
};
use crate::bedrock::responses_translate::{
    build_responses_tools_with, collect_searched_tools, reasoning_outcome,
    request_uses_tool_search, responses_output_format, to_responses_converse_input_with,
    ResponsesToolRegistry, SearchedTool,
};
use crate::bedrock::translate::{merge_strict_text_format, ImageResolver};
use crate::bedrock::{cache, provider, tools};
use crate::config::{AppSettings, Capability, RegionRoutingConfig};
use crate::domain::{
    ModelCapabilities, NormalizedResponsesRequest, ResponsesProvider, ResponsesStream,
    RouteOverride,
};
use crate::error::{from_bedrock_sdk_error, AppError};
use crate::openai::responses_schema::{ResponsesRequest, ResponsesResponse, ResponsesToolChoice};

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
    ///
    /// `tool_search` selects the tool-search lane: the client `tool_search`
    /// tool and the search input items are translated (see
    /// [`to_responses_converse_input_with`]) and the tools loaded by
    /// `tool_search_output` items are collected for the Messages body.
    async fn assemble(
        &self,
        req: &ResponsesRequest,
        resolved: &str,
        force_caching_off: bool,
        tool_search: bool,
    ) -> Result<AssembledConverse, AppError> {
        let caps = self.caps.as_ref();

        let parsed = to_responses_converse_input_with(
            req,
            resolved,
            self.image_resolver.as_ref(),
            caps,
            tool_search,
        )
        .await?;
        let mut messages = parsed.messages;
        let mut system = parsed.system;

        let reasoning = reasoning_outcome(req, resolved, caps);

        // inferenceConfig: maxTokens (reasoning side-signal wins) + temperature /
        // topP. Models declaring DropSamplingParams reject both fields; reasoning
        // independently drops topP when its selected path requires that.
        let mut inference = Map::new();
        let effective_max = reasoning.max_tokens.or(req.max_output_tokens);
        if let Some(max_tokens) = effective_max {
            inference.insert("maxTokens".to_string(), Value::from(max_tokens));
        }
        let drop_sampling_params = caps.has(resolved, Capability::DropSamplingParams);
        if !drop_sampling_params {
            if let Some(temp) = req.temperature {
                inference.insert("temperature".to_string(), Value::from(temp));
            }
            if let Some(top_p) = req.top_p.filter(|_| !reasoning.drop_top_p) {
                inference.insert("topP".to_string(), Value::from(top_p));
            }
        }
        let inference_config = Value::Object(inference);

        // Structured output from `text.format`, translated exactly like chat
        // `response_format` (sent for every model, OpenAI strict field).
        let output_format = responses_output_format(req, resolved, caps)?;
        let mut additional = reasoning.additional_model_request_fields;
        if output_format.as_ref().is_some_and(|f| f.strict_field) {
            merge_strict_text_format(&mut additional);
        }
        let output_config = output_format.map(|f| f.output_config);

        let additional_fields = if additional.is_empty() {
            None
        } else {
            Some(Value::Object(additional))
        };

        // toolConfig from the Responses flattened-function tools (the rejection
        // matrix in to_responses_converse_input already vetoed built-in tools).
        let (mut tool_config, mut tool_registry) =
            build_responses_tool_config_with_registry(req, tool_search)?;
        let searched_tools = if tool_search {
            collect_searched_tools(req)?
        } else {
            Vec::new()
        };
        tool_registry.add_searched(&searched_tools);
        if tool_config.is_none() {
            tool_config = tools::synthesize_tool_config_from_messages(&messages);
            // Searched tools are declared as deferred tools; a placeholder of
            // the same name would collide with that declaration.
            drop_synthesized_searched(&mut tool_config, &searched_tools);
        }

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

        // Resolve the UNIFORM per-request cache TTL. On the Responses surface the
        // Option-B control rides `extra_body.prompt_caching.ttl`, which flattens
        // into `req.extra["extra_body"]`. A `1h` request on a model lacking
        // `Capability::CacheTtl1h` is silently downgraded to `5m` with a
        // metadata-only WARN (no content).
        let ctrl = cache::PromptCachingControl::parse(req.extra.get("extra_body"));
        let resolved_ttl = cache::resolve_cache_ttl(
            ctrl.ttl.as_deref(),
            &self.settings.prompt_cache_ttl,
            resolved,
            caps,
        );
        if resolved_ttl.downgraded {
            tracing::warn!(
                model = %resolved,
                requested_ttl = %resolved_ttl.requested,
                effective_ttl = %resolved_ttl.effective,
                "prompt-cache 1h TTL not supported by model; downgraded to 5m"
            );
        }
        let ttl = Some(resolved_ttl.effective.as_str());
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
                    ttl,
                );
                used += provider::count_cache_points(&decorated);
                tc.insert("tools".to_string(), decorated);
            }
        }

        let decorated_system = cache::decorate_system_blocks(system, resolved, caps, enabled, ttl);
        used += provider::count_cache_points(&decorated_system);
        system = decorated_system;

        messages = cache::decorate_messages(
            messages,
            resolved,
            caps,
            enabled,
            used,
            max_checkpoints,
            ttl,
        );
        used += provider::count_cache_points(&messages);

        Ok(AssembledConverse {
            messages,
            system,
            inference_config,
            additional_fields,
            tool_config,
            output_config,
            tool_registry,
            searched_tools,
            cache_points_injected: used > 0,
        })
    }

    /// Build the typed SDK `converse` call from an [`AssembledConverse`] and send
    /// it (applying the per-request region override at the call site).
    ///
    /// `request_model` is the ORIGINAL incoming model id (cross-region prefix
    /// intact). It — or its `[[alias]]` target, see [`Self::callable_model_id`]
    /// — is what reaches Bedrock and keys the region table, mirroring the chat
    /// provider. The resolved (prefix-stripped) foundation id is for capability
    /// matching only; sending it to Bedrock triggers an on-demand 400.
    /// See [`Self::outbound_model_id`].
    ///
    /// Returns the raw service error in a [`SendError`] so the shared cache
    /// safety net can inspect `.code()`/`.message()` before mapping; JSON→SDK
    /// build failures surface as [`SendError::App`] (never a cache rejection).
    ///
    /// `result_large_err` is allowed deliberately: the lint fires because the
    /// SDK's own `ConverseError` makes `SendError::Service` ~152 bytes, but this
    /// `Result`'s size is dominated by the `Ok` variant — `ConverseOutput` is the
    /// entire Bedrock response and is far larger. Boxing the error would add an
    /// allocation on every failure without shrinking the `Result`, and the error
    /// must stay un-boxed so `is_cache_unsupported_error` can read
    /// `.code()`/`.message()` off the raw service error.
    #[allow(clippy::result_large_err)]
    async fn send_converse(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<ConverseOutput, SendError<ConverseError>> {
        let callable = self.callable_model_id(request_model);
        let route = self.regions.route_for(&callable);
        let model_id = Self::outbound_model_id(&callable, route.as_ref());

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
        if let Some(oc) = &assembled.output_config {
            call = call.output_config(build_sdk_output_config(oc).map_err(SendError::App)?);
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
    ///
    /// `result_large_err` is allowed for the same reason as
    /// [`Self::send_converse`]: the `Ok` variant dominates the `Result` size and
    /// the raw service error must stay un-boxed for the cache predicate.
    #[allow(clippy::result_large_err)]
    async fn send_converse_stream(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<ConverseStreamOutput, SendError<ConverseStreamError>> {
        let callable = self.callable_model_id(request_model);
        let route = self.regions.route_for(&callable);
        let model_id = Self::outbound_model_id(&callable, route.as_ref());

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
        if let Some(oc) = &assembled.output_config {
            call = call.output_config(build_sdk_output_config(oc).map_err(SendError::App)?);
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

    /// Whether this request takes the tool-search lane: the lane is enabled,
    /// the model declares [`Capability::ToolSearch`], and the request carries a
    /// client `tool_search` tool or search input items. Every other request
    /// stays on Converse unchanged.
    fn uses_tool_search(&self, req: &ResponsesRequest, resolved: &str) -> bool {
        !self.settings.disable_tool_search
            && self.caps.has(resolved, Capability::ToolSearch)
            && request_uses_tool_search(req)
    }

    /// Serialize the Messages body for an assembled tool-search request.
    fn messages_body(
        &self,
        resolved: &str,
        assembled: &AssembledConverse,
    ) -> Result<Vec<u8>, AppError> {
        let body = build_messages_body(&MessagesBodyInput {
            messages: &assembled.messages,
            system: &assembled.system,
            inference_config: &assembled.inference_config,
            additional_fields: assembled.additional_fields.as_ref(),
            tool_config: assembled.tool_config.as_ref(),
            output_config: assembled.output_config.as_ref(),
            searched_tools: &assembled.searched_tools,
            default_max_tokens: self.caps.default_max_tokens(resolved),
        })?;
        serde_json::to_vec(&body)
            .map_err(|e| AppError::Internal(format!("failed to serialize InvokeModel body: {e}")))
    }

    /// Send the tool-search lane's non-stream `InvokeModel` call. The model id
    /// and region routing are the Converse path's.
    ///
    /// No prompt-cache strip-and-retry here: every cache point on this lane is
    /// gateway-generated, so a 400 that mentions caching means the mapping is
    /// wrong, and a silent retry without caching would hide it (and mark the
    /// model cache-unsupported for the whole process).
    async fn send_invoke(&self, request_model: &str, body: Vec<u8>) -> Result<Value, AppError> {
        let callable = self.callable_model_id(request_model);
        let route = self.regions.route_for(&callable);
        let model_id = Self::outbound_model_id(&callable, route.as_ref());
        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock invoke_model (responses tool search)"
        );
        let call = self
            .clients
            .runtime
            .invoke_model()
            .model_id(&model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body));
        let output = if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
        } else {
            call.send().await
        }
        .map_err(|e| from_bedrock_sdk_error(&e.into_service_error()))?;
        serde_json::from_slice(output.body.as_ref()).map_err(|e| {
            AppError::UpstreamBedrock(format!("InvokeModel returned a non-JSON body: {e}"))
        })
    }

    /// Send the tool-search lane's `InvokeModelWithResponseStream` call.
    /// Validation errors arrive before the stream opens and map to an HTTP
    /// error, never an SSE body.
    async fn send_invoke_stream(
        &self,
        request_model: &str,
        body: Vec<u8>,
    ) -> Result<InvokeModelWithResponseStreamOutput, AppError> {
        let callable = self.callable_model_id(request_model);
        let route = self.regions.route_for(&callable);
        let model_id = Self::outbound_model_id(&callable, route.as_ref());
        tracing::debug!(
            model = %model_id,
            region = ?route.as_ref().map(|r| &r.region),
            "invoking bedrock invoke_model_with_response_stream (responses tool search)"
        );
        let call = self
            .clients
            .runtime
            .invoke_model_with_response_stream()
            .model_id(&model_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body));
        if let Some(route) = &route {
            call.customize()
                .config_override(region_config_override(route.region.clone()))
                .send()
                .await
        } else {
            call.send().await
        }
        .map_err(|e| from_bedrock_sdk_error(&e.into_service_error()))
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

    /// The id to call for `request_model`: its `[[alias]]` target when it is an
    /// alias (the target may be a cross-region profile), else the request model
    /// unchanged. Mirrors the chat translation's `ConverseArgs::model_id`.
    fn callable_model_id(&self, request_model: &str) -> String {
        self.caps
            .alias_target(request_model)
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
    /// `outputConfig` from `text.format` (structured output), when requested.
    output_config: Option<Value>,
    tool_registry: ResponsesToolRegistry,
    /// Tools loaded by `tool_search_output` items (tool-search lane only),
    /// declared with `defer_loading` in the Messages body.
    searched_tools: Vec<SearchedTool>,
    /// Whether any `cachePoint` landed across the tools/system/messages zones —
    /// consumed by the cache safety net at the send points (read-gate strip).
    cache_points_injected: bool,
}

/// Build Bedrock `toolConfig` together with the reversible Responses mapping.
/// OpenAI-hosted tools without a Converse equivalent are omitted; supported
/// client-executed tools retain a reversible output-item mapping.
///
/// On the tool-search lane, `tool_choice: "none"` keeps the tools (with a
/// `none` choice the Messages builder understands): searched tools referenced
/// by `tool_reference` blocks must stay declared.
fn build_responses_tool_config_with_registry(
    req: &ResponsesRequest,
    tool_search: bool,
) -> Result<(Option<Value>, ResponsesToolRegistry), AppError> {
    let (specs, registry) = build_responses_tools_with(req, tool_search)?;
    if specs.is_empty() {
        return Ok((None, registry));
    }
    if matches!(
        &req.tool_choice,
        Some(ResponsesToolChoice::String(choice)) if choice == "none"
    ) {
        if tool_search {
            return Ok((
                Some(json!({ "tools": specs, "toolChoice": { "none": {} } })),
                registry,
            ));
        }
        return Ok((None, registry));
    }

    let mut config = json!({ "tools": specs });
    let choice = match &req.tool_choice {
        None => None,
        Some(ResponsesToolChoice::String(choice)) if choice == "auto" => {
            Some(json!({ "auto": {} }))
        }
        Some(ResponsesToolChoice::String(choice)) if choice == "required" => {
            Some(json!({ "any": {} }))
        }
        Some(ResponsesToolChoice::String(choice)) => {
            return Err(AppError::BadRequest(format!(
                "unsupported Responses tool_choice '{choice}'"
            )))
        }
        Some(ResponsesToolChoice::Object(choice)) => {
            let choice_type = choice
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function");
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    choice
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                })
                .ok_or_else(|| {
                    AppError::BadRequest(
                        "specific Responses tool_choice must include a tool name".to_string(),
                    )
                })?;
            let namespace = choice.get("namespace").and_then(Value::as_str);
            let internal = namespace
                .map(|ns| format!("{ns}__{name}"))
                .or_else(|| registry.bedrock_name_for(name).map(str::to_string))
                .ok_or_else(|| {
                    AppError::BadRequest(format!(
                        "Responses tool_choice refers to unknown tool '{name}'"
                    ))
                })?;
            match choice_type {
                "function" | "custom" | "local_shell" | "shell" | "apply_patch" => {
                    Some(json!({ "tool": { "name": internal } }))
                }
                other => {
                    return Err(AppError::Unsupported(format!(
                        "Responses tool_choice type '{other}' is not available through Bedrock Converse"
                    )))
                }
            }
        }
    };
    if let (Some(obj), Some(choice)) = (config.as_object_mut(), choice) {
        obj.insert("toolChoice".to_string(), choice);
    }
    Ok((Some(config), registry))
}

#[cfg(test)]
fn build_responses_tool_config(req: &ResponsesRequest) -> Option<Value> {
    build_responses_tool_config_with_registry(req, false)
        .ok()
        .and_then(|(config, _)| config)
}

/// Remove placeholder tool specs (synthesized for a continuation with no
/// `tools`) that name a searched tool.
fn drop_synthesized_searched(tool_config: &mut Option<Value>, searched: &[SearchedTool]) {
    if searched.is_empty() {
        return;
    }
    let Some(tools) = tool_config
        .as_mut()
        .and_then(|config| config.get_mut("tools"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    tools.retain(|spec| {
        let name = spec
            .pointer("/toolSpec/name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        !searched.iter().any(|tool| tool.bedrock_name == name)
    });
    if tools.is_empty() {
        *tool_config = None;
    }
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

        if self.uses_tool_search(&req.request, resolved) {
            let assembled = self.assemble(&req.request, resolved, false, true).await?;
            let body = self.messages_body(resolved, &assembled)?;
            let output = self.send_invoke(request_model, body).await?;
            let converse = messages_response_to_converse_output(&output)?;
            return from_converse_output_to_responses_with_tools(
                &converse,
                &req.request,
                &req.request.model,
                &resp_id(),
                &assembled.tool_registry,
            );
        }

        let assembled = self.assemble(&req.request, resolved, false, false).await?;
        let normalized = normalize_for_match(resolved);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            assembled.cache_points_injected,
            || self.send_converse(request_model, &assembled),
            || async {
                let retry = self
                    .assemble(&req.request, resolved, true, false)
                    .await
                    .map_err(SendError::App)?;
                self.send_converse(request_model, &retry).await
            },
        )
        .await?;

        let output_json = converse_output_to_json(&output);
        let response_id = resp_id();
        from_converse_output_to_responses_with_tools(
            &output_json,
            &req.request,
            &req.request.model,
            &response_id,
            &assembled.tool_registry,
        )
    }

    async fn respond_stream(
        &self,
        req: &NormalizedResponsesRequest,
    ) -> Result<ResponsesStream, AppError> {
        let resolved = &req.resolved_model;
        let request_model = &req.request.model;
        let runtime = ResponsesStreamRuntime::new(
            req.request_id.clone(),
            req.received_at,
            std::time::Duration::from_secs(self.settings.responses_stream_idle_timeout_secs),
        );

        if self.uses_tool_search(&req.request, resolved) {
            let assembled = self.assemble(&req.request, resolved, false, true).await?;
            let body = self.messages_body(resolved, &assembled)?;
            let output = self.send_invoke_stream(request_model, body).await?;
            return Ok(invoke_stream_to_openai_responses(
                output,
                req.request.model.clone(),
                resp_id(),
                req.request.clone(),
                assembled.tool_registry,
                runtime,
            ));
        }

        let assembled = self.assemble(&req.request, resolved, false, false).await?;
        let normalized = normalize_for_match(resolved);

        let output = send_with_cache_strip_retry(
            &self.cache_support,
            &normalized,
            assembled.cache_points_injected,
            || self.send_converse_stream(request_model, &assembled),
            || async {
                let retry = self
                    .assemble(&req.request, resolved, true, false)
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
            assembled.tool_registry,
            runtime,
        ))
    }
}

#[cfg(test)]
#[path = "responses_provider_tests.rs"]
mod tests;
