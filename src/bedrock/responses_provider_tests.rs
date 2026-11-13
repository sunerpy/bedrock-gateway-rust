//! Unit tests for [`crate::bedrock::responses_provider`], relocated out of the
//! source module for code organization (see the sidecar-test convention in `AGENTS.md`).
//! The source file declares this via a `#[path]` mod tests, so the top-level
//! `use super::*;` resolves to the implementation module.

use super::*;
use crate::config::ModelCapabilityConfig;
use crate::openai::responses_schema::{
    FunctionCallOutputValue, ResponseInputItem, ResponsesInput, ResponsesTool,
};
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

fn continuation_input(tool_name: &str) -> ResponsesInput {
    ResponsesInput::Items(vec![
        ResponseInputItem::FunctionCall {
            call_id: "call_1".to_string(),
            name: tool_name.to_string(),
            arguments: "{}".to_string(),
            namespace: None,
        },
        ResponseInputItem::FunctionCallOutput {
            call_id: "call_1".to_string(),
            output: FunctionCallOutputValue::Text("sunny".to_string()),
        },
    ])
}

fn test_settings(enable_prompt_caching: bool) -> Arc<AppSettings> {
    Arc::new(AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-west-2".to_string(),
        default_model: "m".to_string(),
        default_embedding_model: "e".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching,
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: false,
        chat_reasoning_capsule_active_kid: None,
        chat_reasoning_capsule_keys: None,
        api_key: Some("k".to_string()),
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        disable_tool_search: false,
        bind_addr: "127.0.0.1".to_string(),
        port: 0,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        responses_stream_idle_timeout_secs: 180,
        model_catalog_refresh_secs: 0,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    })
}

async fn test_provider(enable_prompt_caching: bool) -> BedrockResponsesProvider {
    test_provider_with_config(enable_prompt_caching, ModelCapabilityConfig::default()).await
}

async fn test_provider_with_config(
    enable_prompt_caching: bool,
    capability_config: ModelCapabilityConfig,
) -> BedrockResponsesProvider {
    use crate::bedrock::client::{build_aws_config, BedrockClients};
    use crate::bedrock::translate::ReqwestImageResolver;

    let settings = test_settings(enable_prompt_caching);
    let aws_config = build_aws_config(&settings).await;
    let clients = BedrockClients::new(&aws_config);
    let caps: Arc<dyn ModelCapabilities> = Arc::new(
        crate::bedrock::capabilities::ConfigModelCapabilities::new(capability_config),
    );
    let regions = Arc::new(RegionRoutingConfig::default());
    let image_resolver = Arc::new(ReqwestImageResolver::new(|_: &str| false));

    BedrockResponsesProvider::new(
        clients,
        caps,
        regions,
        image_resolver,
        settings,
        Arc::new(crate::bedrock::cache_support::CacheSupportRegistry::new()),
    )
}

fn sampling_capability_config() -> ModelCapabilityConfig {
    ModelCapabilityConfig::from_toml_str(
        r#"
[[model]]
match = "drop-sampling-model"
capabilities = ["drop_sampling_params"]

[[model]]
match = "reasoning-model"
capabilities = []
[model.params]
reasoning_path = "adaptive_thinking"
"#,
    )
    .expect("sampling capability config")
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

#[test]
fn responses_tool_choice_maps_all_bedrock_supported_modes() {
    let mut req = base_request();
    req.tools = Some(vec![ResponsesTool::Function {
        name: "lookup".to_string(),
        description: None,
        parameters: Some(json!({"type":"object"})),
        strict: None,
    }]);

    req.tool_choice = Some(ResponsesToolChoice::String("auto".to_string()));
    let (auto, _) = build_responses_tool_config_with_registry(&req, false).expect("auto");
    assert_eq!(auto.expect("config")["toolChoice"], json!({"auto": {}}));

    req.tool_choice = Some(ResponsesToolChoice::String("required".to_string()));
    let (required, _) = build_responses_tool_config_with_registry(&req, false).expect("required");
    assert_eq!(required.expect("config")["toolChoice"], json!({"any": {}}));

    req.tool_choice = Some(ResponsesToolChoice::Object(json!({
        "type": "function", "name": "lookup"
    })));
    let (specific, _) = build_responses_tool_config_with_registry(&req, false).expect("specific");
    assert_eq!(
        specific.expect("config")["toolChoice"],
        json!({"tool": {"name":"lookup"}})
    );

    req.tool_choice = Some(ResponsesToolChoice::String("none".to_string()));
    let (none, _) = build_responses_tool_config_with_registry(&req, false).expect("none");
    assert!(none.is_none());
}

#[test]
fn hosted_responses_tool_is_omitted_without_hiding_function_tools() {
    let mut req = base_request();
    req.tools = Some(vec![
        serde_json::from_value(json!({"type":"web_search"})).unwrap(),
        serde_json::from_value(json!({
            "type":"function",
            "name":"lookup",
            "description":"Look up a value",
            "parameters":{"type":"object","properties":{}}
        }))
        .unwrap(),
    ]);
    let (config, registry) =
        build_responses_tool_config_with_registry(&req, false).expect("hosted tool is ignored");
    let config = config.expect("supported tool config remains");
    let specs = config["tools"].as_array().expect("tools array");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "lookup");
    assert!(registry.resolve("web_search").is_none());
    assert!(registry.resolve("lookup").is_some());
}

#[tokio::test]
async fn responses_assemble_synthesizes_toolconfig_for_continuation() {
    let provider = test_provider(false).await;
    let mut req = base_request();
    req.input = continuation_input("get_weather");

    let assembled = provider
        .assemble(&req, "resolved", false, false)
        .await
        .expect("assemble continuation request");

    let tool_config = assembled.tool_config.expect("synthesized tool config");
    let specs = tool_config["tools"].as_array().expect("tools array");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "get_weather");
}

#[tokio::test]
async fn responses_assemble_real_tools_unchanged() {
    let provider = test_provider(false).await;
    let mut req = base_request();
    req.input = continuation_input("history_tool");
    req.tools = Some(vec![ResponsesTool::Function {
        name: "declared_tool".to_string(),
        description: Some("Declared tool".to_string()),
        parameters: Some(json!({ "type": "object", "properties": {} })),
        strict: None,
    }]);

    let assembled = provider
        .assemble(&req, "resolved", false, false)
        .await
        .expect("assemble request with declared tools");

    let tool_config = assembled.tool_config.expect("declared tool config");
    let specs = tool_config["tools"].as_array().expect("tools array");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "declared_tool");
    assert_eq!(specs[0]["toolSpec"]["description"], "Declared tool");
}

#[tokio::test]
async fn responses_assemble_drops_sampling_params_for_capability_model() {
    let provider = test_provider_with_config(false, sampling_capability_config()).await;
    let mut req = base_request();
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);

    let assembled = provider
        .assemble(&req, "us.vendor.drop-sampling-model-v1", false, false)
        .await
        .expect("assemble capability model request");

    assert!(assembled.inference_config.get("temperature").is_none());
    assert!(assembled.inference_config.get("topP").is_none());
}

#[tokio::test]
async fn responses_assemble_preserves_sampling_params_without_capability() {
    let provider = test_provider_with_config(false, sampling_capability_config()).await;
    let mut req = base_request();
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);

    let assembled = provider
        .assemble(&req, "us.vendor.regular-model-v1", false, false)
        .await
        .expect("assemble regular model request");

    let temperature = assembled.inference_config["temperature"]
        .as_f64()
        .expect("temperature");
    let top_p = assembled.inference_config["topP"].as_f64().expect("topP");
    assert!((temperature - 0.7).abs() < 1e-6);
    assert!((top_p - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn responses_assemble_reasoning_still_drops_top_p_independently() {
    let provider = test_provider_with_config(false, sampling_capability_config()).await;
    let mut req = base_request();
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);
    req.reasoning =
        Some(serde_json::from_value(json!({ "effort": "high" })).expect("reasoning request"));

    let assembled = provider
        .assemble(&req, "us.vendor.reasoning-model-v1", false, false)
        .await
        .expect("assemble reasoning request");

    let temperature = assembled.inference_config["temperature"]
        .as_f64()
        .expect("temperature");
    assert!((temperature - 0.7).abs() < 1e-6);
    assert!(assembled.inference_config.get("topP").is_none());
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

/// A bare `[[alias]]` name is called under its target (here a cross-region
/// profile); any other id is called exactly as the client sent it.
#[tokio::test]
async fn callable_model_id_follows_alias_target() {
    let config = ModelCapabilityConfig::from_toml_str(
        "[[alias]]\nfrom = \"bare-name\"\nto = \"global.vendor.model\"\n",
    )
    .expect("alias config");
    let provider = test_provider_with_config(false, config).await;

    assert_eq!(
        provider.callable_model_id("bare-name"),
        "global.vendor.model"
    );
    assert_eq!(
        provider.callable_model_id("us.vendor.model"),
        "us.vendor.model"
    );
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
    let provider: Arc<dyn ResponsesProvider> = Arc::new(test_provider(false).await);

    let req = NormalizedResponsesRequest {
        request: base_request(),
        resolved_model: "resolved".to_string(),
        request_id: Arc::from("req-test"),
        received_at: std::time::Instant::now(),
        raw_body: bytes::Bytes::new(),
    };
    assert!(
        provider.respond_stream(&req).await.is_err(),
        "stream path errors without AWS creds"
    );
}

fn project_capability_config() -> ModelCapabilityConfig {
    ModelCapabilityConfig::load("config/models.toml").expect("config/models.toml")
}

/// `text.format` reaches Converse as `outputConfig` on both send paths' input,
/// and a strict schema on an OpenAI-text-format model adds the strict field
/// next to the reasoning effort instead of replacing it.
#[tokio::test]
async fn responses_assemble_emits_output_config_for_text_format() {
    let provider = test_provider_with_config(false, project_capability_config()).await;
    let mut req = base_request();
    req.model = "gpt-6.1-sol".to_string();
    req.reasoning =
        Some(serde_json::from_value(json!({ "effort": "high" })).expect("reasoning request"));
    req.text = Some(
        serde_json::from_value(json!({ "format": {
            "type": "json_schema",
            "name": "capital",
            "strict": true,
            "schema": { "type": "object", "properties": { "capital": { "type": "string" } },
                        "required": ["capital"], "additionalProperties": false }
        }}))
        .expect("text config"),
    );

    let assembled = provider
        .assemble(&req, "global.openai.gpt-6.1-sol", false, false)
        .await
        .expect("assemble");

    let oc = assembled.output_config.expect("outputConfig emitted");
    assert_eq!(
        oc["textFormat"]["structure"]["jsonSchema"]["name"],
        "capital"
    );
    assert_eq!(
        assembled.additional_fields,
        Some(json!({
            "reasoning": { "effort": "high" },
            "text": { "format": { "strict": true } }
        }))
    );
    build_sdk_output_config(&oc).expect("outputConfig converts to the SDK type");
}

#[tokio::test]
async fn responses_assemble_sends_text_format_for_claude_5_5() {
    // Bedrock serves structured output on Opus 5.5 (live 2026-10-10); the
    // gateway sends the outputConfig instead of answering 400 itself.
    let provider = test_provider_with_config(false, project_capability_config()).await;
    let mut req = base_request();
    req.model = "global.anthropic.claude-opus-5-5".to_string();
    req.text = Some(
        serde_json::from_value(json!({ "format": { "type": "json_object" } }))
            .expect("text config"),
    );

    let assembled = provider
        .assemble(&req, "global.anthropic.claude-opus-5-5", false, false)
        .await
        .expect("assemble");
    let oc = assembled.output_config.expect("outputConfig sent");
    assert_eq!(oc["textFormat"]["type"], "json_schema");
    assert_eq!(
        oc["textFormat"]["structure"]["jsonSchema"]["name"],
        "response"
    );
}

#[tokio::test]
async fn responses_assemble_without_text_format_sends_no_output_config() {
    let provider = test_provider_with_config(false, project_capability_config()).await;
    let req = base_request();
    let assembled = provider
        .assemble(&req, "global.anthropic.claude-sonnet-4-6", false, false)
        .await
        .expect("assemble");
    assert!(assembled.output_config.is_none());
}

// ---------------------------------------------------------------------------
// Tool-search lane (zuno 0.160.1 golden, tests/golden/fixtures/responses_tool_search)
// ---------------------------------------------------------------------------

const TOOL_SEARCH_MODEL: &str = "global.anthropic.claude-sonnet-5-5";

/// A golden request with the two fields the model broker rewrites in
/// production: the Claude profile and `max_output_tokens`.
fn golden_request(raw: &str) -> ResponsesRequest {
    let mut value: Value = serde_json::from_str(raw).expect("golden json");
    value["model"] = json!(TOOL_SEARCH_MODEL);
    value["max_output_tokens"] = json!(32000);
    serde_json::from_value(value).expect("golden request parses")
}

fn golden(name: &str) -> ResponsesRequest {
    golden_request(match name {
        "request-1" => {
            include_str!("../../tests/golden/fixtures/responses_tool_search/request-1.json")
        }
        "request-2" => {
            include_str!("../../tests/golden/fixtures/responses_tool_search/request-2.json")
        }
        "request-2-no-match" => include_str!(
            "../../tests/golden/fixtures/responses_tool_search/request-2-no-match.json"
        ),
        "request-3" => {
            include_str!("../../tests/golden/fixtures/responses_tool_search/request-3.json")
        }
        other => panic!("unknown golden {other}"),
    })
}

async fn tool_search_provider() -> BedrockResponsesProvider {
    test_provider_with_config(true, project_capability_config()).await
}

/// The Messages body and registry the tool-search lane builds for `req`.
async fn tool_search_body(
    provider: &BedrockResponsesProvider,
    req: &ResponsesRequest,
) -> (Value, ResponsesToolRegistry) {
    assert!(provider.uses_tool_search(req, TOOL_SEARCH_MODEL));
    let assembled = provider
        .assemble(req, TOOL_SEARCH_MODEL, false, true)
        .await
        .expect("assemble");
    let body = provider
        .messages_body(TOOL_SEARCH_MODEL, &assembled)
        .expect("messages body");
    (
        serde_json::from_slice(&body).expect("body json"),
        assembled.tool_registry,
    )
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

fn last_messages(body: &Value, n: usize) -> Vec<Value> {
    let messages = body["messages"].as_array().expect("messages");
    messages[messages.len() - n..].to_vec()
}

/// Acceptance 1: request-1 → 10 tools in zuno's order (namespace flattened in
/// place, `tool_search` last), no `defer_loading`, the only tools
/// `cache_control` on `tool_search`.
#[tokio::test]
async fn tool_search_request_1_maps_tools_in_order() {
    let provider = tool_search_provider().await;
    let (body, _) = tool_search_body(&provider, &golden("request-1")).await;

    assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
    assert_eq!(body["max_tokens"], 32000);
    assert_eq!(
        tool_names(&body),
        vec![
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
            "request_user_input",
            "mcp__ze_tools__direct_alpha",
            "mcp__ze_tools__direct_beta",
            "get_goal",
            "create_goal",
            "update_goal",
            "tool_search",
        ]
    );
    assert!(!body.to_string().contains("defer_loading"));
    let tools = body["tools"].as_array().unwrap();
    let cached: Vec<&str> = tools
        .iter()
        .filter(|t| t.get("cache_control").is_some())
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(cached, vec!["tool_search"]);
    assert_eq!(
        tools[9]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "5m" })
    );

    let raw: Value = serde_json::from_str(include_str!(
        "../../tests/golden/fixtures/responses_tool_search/request-1.json"
    ))
    .unwrap();
    let search = raw["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["type"] == "tool_search")
        .unwrap();
    assert_eq!(tools[9]["description"], search["description"]);
    assert_eq!(tools[9]["input_schema"], search["parameters"]);
    // No Converse-shaped leftovers and no beta.
    let text = body.to_string();
    for key in [
        "cachePoint",
        "toolSpec",
        "toolUse",
        "anthropic_beta",
        "reasoningContent",
    ] {
        assert!(!text.contains(key), "{key} leaked into the Messages body");
    }
    // reasoning.effort reaches the body top level.
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "medium");
}

/// Acceptance 2: request-2 appends both searched tools as deferred tools and
/// ends with the search tool_use and a tool_result of two tool_reference
/// blocks; the first 10 tools are byte-identical to request-1.
#[tokio::test]
async fn tool_search_request_2_declares_searched_tools_deferred() {
    let provider = tool_search_provider().await;
    let (first, _) = tool_search_body(&provider, &golden("request-1")).await;
    let (body, registry) = tool_search_body(&provider, &golden("request-2")).await;

    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 12);
    assert_eq!(&tools[..10], first["tools"].as_array().unwrap().as_slice());
    assert_eq!(
        tools[10],
        json!({
            "name": "mcp__ze_tools_deferred__ticket_search",
            "description": "Search tickets by keyword.",
            "input_schema": { "type": "object", "properties": {} },
            "defer_loading": true,
        })
    );
    assert_eq!(tools[11]["name"], "mcp__ze_tools_deferred__ticket_get");
    assert_eq!(tools[11]["defer_loading"], true);
    assert!(tools[11].get("cache_control").is_none());

    assert_eq!(
        last_messages(&body, 2),
        vec![
            json!({ "role": "assistant", "content": [{
                "type": "tool_use", "id": "tscall_1", "name": "tool_search",
                "input": { "query": "ticket", "limit": 5 }
            }]}),
            json!({ "role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "tscall_1", "content": [
                    { "type": "tool_reference", "tool_name": "mcp__ze_tools_deferred__ticket_search" },
                    { "type": "tool_reference", "tool_name": "mcp__ze_tools_deferred__ticket_get" }
                ]
            }]}),
        ]
    );
    let binding = registry
        .resolve("mcp__ze_tools_deferred__ticket_search")
        .expect("searched tool is bound");
    assert_eq!(binding.client_name, "ticket_search");
    assert_eq!(binding.namespace.as_deref(), Some("mcp__ze_tools_deferred"));
}

/// Acceptance 3: a search that found nothing declares no deferred tool and
/// answers with a text block.
#[tokio::test]
async fn tool_search_no_match_answers_with_text() {
    let provider = tool_search_provider().await;
    let (body, _) = tool_search_body(&provider, &golden("request-2-no-match")).await;

    assert_eq!(body["tools"].as_array().unwrap().len(), 10);
    assert!(!body.to_string().contains("defer_loading"));
    let last = last_messages(&body, 1).remove(0);
    assert_eq!(last["content"][0]["type"], "tool_result");
    assert_eq!(last["content"][0]["tool_use_id"], "tscall_1");
    assert_eq!(
        last["content"][0]["content"],
        json!([{ "type": "text", "text": "No tools matched the query." }])
    );
}

/// Acceptance 4: request-3 keeps request-2's tools exactly and replays the
/// call of the searched tool under its flattened name.
#[tokio::test]
async fn tool_search_request_3_replays_searched_call() {
    let provider = tool_search_provider().await;
    let (second, _) = tool_search_body(&provider, &golden("request-2")).await;
    let (body, _) = tool_search_body(&provider, &golden("request-3")).await;

    assert_eq!(body["tools"], second["tools"]);
    assert_eq!(
        last_messages(&body, 2),
        vec![
            json!({ "role": "assistant", "content": [{
                "type": "tool_use", "id": "call_2",
                "name": "mcp__ze_tools_deferred__ticket_search", "input": {}
            }]}),
            json!({ "role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "call_2", "content": [
                    { "type": "text", "text": "Wall time: 0.0023 seconds\nOutput:" },
                    { "type": "text", "text": "called ticket_search" }
                ]
            }]}),
        ]
    );
}

/// Acceptance 5 (non-stream half): Claude's answers map back to zuno's items.
#[tokio::test]
async fn tool_search_answers_map_to_zuno_items() {
    use crate::bedrock::anthropic_messages::messages_response_to_converse_output;

    let provider = tool_search_provider().await;
    let req = golden("request-2");
    let (_, registry) = tool_search_body(&provider, &req).await;

    let search = messages_response_to_converse_output(&json!({
        "content": [{ "type": "tool_use", "id": "toolu_1", "name": "tool_search",
                      "input": { "query": "ticket", "limit": 5 } }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 10, "output_tokens": 5 }
    }))
    .unwrap();
    let response = from_converse_output_to_responses_with_tools(
        &search, &req, &req.model, "resp_x", &registry,
    )
    .unwrap();
    let item = serde_json::to_value(&response.output[0]).unwrap();
    assert_eq!(
        item,
        json!({
            "type": "tool_search_call", "id": "ts_toolu_1", "call_id": "toolu_1",
            "execution": "client", "status": "completed",
            "arguments": { "query": "ticket", "limit": 5 }
        })
    );

    let call = messages_response_to_converse_output(&json!({
        "content": [{ "type": "tool_use", "id": "toolu_2",
                      "name": "mcp__ze_tools_deferred__ticket_search", "input": {} }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 10, "output_tokens": 5 }
    }))
    .unwrap();
    let response =
        from_converse_output_to_responses_with_tools(&call, &req, &req.model, "resp_x", &registry)
            .unwrap();
    let item = serde_json::to_value(&response.output[0]).unwrap();
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["name"], "ticket_search");
    assert_eq!(item["namespace"], "mcp__ze_tools_deferred");
    assert_eq!(item["arguments"], "{}");
    assert_eq!(item["call_id"], "toolu_2");
}

/// Acceptance 6: without a client `tool_search` (non-zuno clients) the request
/// stays on Converse; the lane is also off for models without the capability
/// and when DISABLE_TOOL_SEARCH is set.
#[tokio::test]
async fn tool_search_lane_selection() {
    let provider = tool_search_provider().await;
    let mut req = golden("request-1");
    req.tools
        .as_mut()
        .unwrap()
        .retain(|t| !matches!(t, ResponsesTool::ToolSearch { .. }));
    assert!(!provider.uses_tool_search(&req, TOOL_SEARCH_MODEL));

    let req = golden("request-1");
    assert!(provider.uses_tool_search(&req, TOOL_SEARCH_MODEL));
    assert!(!provider.uses_tool_search(&req, "us.amazon.nova-pro-v1:0"));
    assert!(!provider.uses_tool_search(&req, "global.openai.gpt-6.1-sol"));
    // Claude generations the tool-search page does not list.
    assert!(!provider.uses_tool_search(&req, "us.anthropic.claude-3-7-sonnet-20250219-v1:0"));

    // Hosted (server) execution is not the client lane.
    let mut hosted = golden("request-1");
    for tool in hosted.tools.as_mut().unwrap() {
        if let ResponsesTool::ToolSearch { fields } = tool {
            fields.insert("execution".to_string(), json!("server"));
        }
    }
    assert!(!provider.uses_tool_search(&hosted, TOOL_SEARCH_MODEL));

    let mut settings = (*test_settings(true)).clone();
    settings.disable_tool_search = true;
    let mut disabled = tool_search_provider().await;
    disabled.settings = Arc::new(settings);
    assert!(!disabled.uses_tool_search(&req, TOOL_SEARCH_MODEL));
}

/// The Converse path keeps today's behavior for the same request: the
/// `tool_search` tool is dropped and search items are a 400.
#[tokio::test]
async fn converse_path_unchanged_for_tool_search_items() {
    let provider = tool_search_provider().await;
    let assembled = provider
        .assemble(&golden("request-1"), TOOL_SEARCH_MODEL, false, false)
        .await
        .unwrap();
    let names: Vec<String> = assembled.tool_config.as_ref().unwrap()["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t.pointer("/toolSpec/name").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    assert_eq!(names.len(), 9);
    assert!(!names.iter().any(|n| n == "tool_search"));

    let err = provider
        .assemble(&golden("request-2"), TOOL_SEARCH_MODEL, false, false)
        .await
        .err()
        .expect("search items are rejected on Converse");
    assert!(err.to_string().contains("tool_search_call"));
}

/// Search history without a `tool_search` tool (another client): the searched
/// tools are still declared, the placeholder for the search tool stays
/// non-deferred, and no placeholder collides with a deferred declaration.
#[tokio::test]
async fn tool_search_history_without_search_tool() {
    let provider = tool_search_provider().await;
    let mut req = golden("request-3");
    req.tools = None;
    let (body, _) = tool_search_body(&provider, &req).await;
    let names = tool_names(&body);
    assert_eq!(
        names,
        vec![
            "tool_search",
            "mcp__ze_tools_deferred__ticket_search",
            "mcp__ze_tools_deferred__ticket_get",
        ]
    );
    assert!(body["tools"][0].get("defer_loading").is_none());
}

/// Pre-flight 400s: an invalid upstream name, a searched tool that collides
/// with a declared tool, and a missing `max_tokens` with no configured default.
#[tokio::test]
async fn tool_search_preflight_rejections() {
    let provider = tool_search_provider().await;

    let mut req = golden("request-2");
    if let ResponsesInput::Items(items) = &mut req.input {
        for item in items {
            if let ResponseInputItem::Other { item_type, fields } = item {
                if item_type == "tool_search_output" {
                    fields.insert(
                        "tools".to_string(),
                        json!([{ "type": "function", "name": "get_goal",
                                 "parameters": { "type": "object", "properties": {} } }]),
                    );
                }
            }
        }
    }
    // A top-level function loaded by search keeps its bare name, so it collides
    // with the declared `get_goal`: 400 before the call.
    let assembled = provider
        .assemble(&req, TOOL_SEARCH_MODEL, false, true)
        .await
        .unwrap();
    let err = provider
        .messages_body(TOOL_SEARCH_MODEL, &assembled)
        .unwrap_err();
    assert!(matches!(err, AppError::BadRequest(_)));
    assert!(err.to_string().contains("'get_goal' is both declared"));

    let mut req = golden("request-2");
    if let ResponsesInput::Items(items) = &mut req.input {
        for item in items {
            if let ResponseInputItem::Other { item_type, fields } = item {
                if item_type == "tool_search_output" {
                    fields.insert(
                        "tools".to_string(),
                        json!([{ "type": "namespace", "name": "mcp__ze_tools",
                                 "description": "dup",
                                 "tools": [{ "type": "function", "name": "direct_alpha" }] }]),
                    );
                }
            }
        }
    }
    let assembled = provider
        .assemble(&req, TOOL_SEARCH_MODEL, false, true)
        .await
        .unwrap();
    let err = provider
        .messages_body(TOOL_SEARCH_MODEL, &assembled)
        .unwrap_err();
    assert!(matches!(err, AppError::BadRequest(_)));
    assert!(err.to_string().contains("mcp__ze_tools__direct_alpha"));

    let mut req = golden("request-2");
    if let ResponsesInput::Items(items) = &mut req.input {
        for item in items {
            if let ResponseInputItem::Other { item_type, fields } = item {
                if item_type == "tool_search_output" {
                    fields.insert(
                        "tools".to_string(),
                        json!([{ "type": "namespace", "name": "bad.ns", "description": "d",
                                 "tools": [{ "type": "function", "name": "x" }] }]),
                    );
                }
            }
        }
    }
    let assembled = provider
        .assemble(&req, TOOL_SEARCH_MODEL, false, true)
        .await
        .unwrap();
    let err = provider
        .messages_body(TOOL_SEARCH_MODEL, &assembled)
        .unwrap_err();
    assert!(err.to_string().contains("bad.ns__x"));

    // No max_output_tokens: the configured default applies.
    let mut req = golden("request-1");
    req.max_output_tokens = None;
    req.reasoning = None;
    let (body, _) = tool_search_body(&provider, &req).await;
    assert_eq!(body["max_tokens"], 32000);
}

/// tool_choice "none" keeps the declared tools on this lane so tool_reference
/// blocks still resolve.
#[tokio::test]
async fn tool_search_tool_choice_none_keeps_tools() {
    let provider = tool_search_provider().await;
    let mut req = golden("request-2");
    req.tool_choice = Some(ResponsesToolChoice::String("none".to_string()));
    let (body, _) = tool_search_body(&provider, &req).await;
    assert_eq!(body["tools"].as_array().unwrap().len(), 12);
    assert_eq!(body["tool_choice"], json!({ "type": "none" }));
}

/// A top-level function loaded by search (OpenAI's client example shape) is
/// declared under its bare name, answered with `namespace` equal to the name,
/// and its replayed call maps back to the bare declaration.
#[tokio::test]
async fn tool_search_top_level_function_round_trips_bare_name() {
    use crate::bedrock::anthropic_messages::messages_response_to_converse_output;

    let provider = tool_search_provider().await;
    let loaded = json!([{ "type": "function", "name": "lookup_order",
        "description": "Look up an order.", "defer_loading": true,
        "parameters": { "type": "object", "properties": { "id": { "type": "string" } } } }]);
    let with_loaded = |mut req: ResponsesRequest| {
        if let ResponsesInput::Items(items) = &mut req.input {
            for item in items {
                if let ResponseInputItem::Other { item_type, fields } = item {
                    if item_type == "tool_search_output" {
                        fields.insert("tools".to_string(), loaded.clone());
                    }
                }
            }
        }
        req
    };

    let req = with_loaded(golden("request-2"));
    let (body, registry) = tool_search_body(&provider, &req).await;
    assert_eq!(tool_names(&body).last().unwrap(), "lookup_order");
    assert_eq!(body["tools"][10]["defer_loading"], true);
    assert_eq!(
        last_messages(&body, 1)[0]["content"][0]["content"],
        json!([{ "type": "tool_reference", "tool_name": "lookup_order" }])
    );

    let answer = messages_response_to_converse_output(&json!({
        "content": [{ "type": "tool_use", "id": "toolu_7", "name": "lookup_order",
                      "input": { "id": "A1" } }],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    }))
    .unwrap();
    let response = from_converse_output_to_responses_with_tools(
        &answer, &req, &req.model, "resp_x", &registry,
    )
    .unwrap();
    let item = serde_json::to_value(&response.output[0]).unwrap();
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["name"], "lookup_order");
    assert_eq!(item["namespace"], "lookup_order");

    // The client replays that call with namespace == name.
    let mut next = with_loaded(golden("request-3"));
    if let ResponsesInput::Items(items) = &mut next.input {
        for item in items {
            if let ResponseInputItem::FunctionCall {
                name, namespace, ..
            } = item
            {
                *name = "lookup_order".to_string();
                *namespace = Some("lookup_order".to_string());
            }
        }
    }
    let (body, _) = tool_search_body(&provider, &next).await;
    assert_eq!(
        last_messages(&body, 2)[0]["content"][0]["name"],
        "lookup_order"
    );

    // Outside the lane a namespace == name call keeps the old flattening.
    let mut converse = golden("request-1");
    converse.input = ResponsesInput::Items(vec![ResponseInputItem::FunctionCall {
        call_id: "c".to_string(),
        name: "x".to_string(),
        arguments: "{}".to_string(),
        namespace: Some("x".to_string()),
    }]);
    let assembled = provider
        .assemble(&converse, TOOL_SEARCH_MODEL, false, false)
        .await
        .unwrap();
    assert_eq!(
        assembled.messages[0]["content"][0]["toolUse"]["name"],
        "x__x"
    );
}
