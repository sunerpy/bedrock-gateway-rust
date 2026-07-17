//! Unit tests for [`crate::bedrock::responses_provider`], relocated out of the
//! source module for code organization (see the `test-coverage-codecov` spec).
//! The source file declares this via a `#[path]` mod tests, so the top-level
//! `use super::*;` resolves to the implementation module.

use super::*;
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
        bind_addr: "127.0.0.1".to_string(),
        port: 0,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        responses_stream_idle_timeout_secs: 180,
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
    use crate::bedrock::client::{build_aws_config, BedrockClients};
    use crate::bedrock::translate::ReqwestImageResolver;
    use crate::config::ModelCapabilityConfig;

    let settings = test_settings(enable_prompt_caching);
    let aws_config = build_aws_config(&settings).await;
    let clients = BedrockClients::new(&aws_config);
    let caps: Arc<dyn ModelCapabilities> = Arc::new(
        crate::bedrock::capabilities::ConfigModelCapabilities::new(ModelCapabilityConfig::default()),
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
    let (auto, _) = build_responses_tool_config_with_registry(&req).expect("auto");
    assert_eq!(auto.expect("config")["toolChoice"], json!({"auto": {}}));

    req.tool_choice = Some(ResponsesToolChoice::String("required".to_string()));
    let (required, _) = build_responses_tool_config_with_registry(&req).expect("required");
    assert_eq!(required.expect("config")["toolChoice"], json!({"any": {}}));

    req.tool_choice = Some(ResponsesToolChoice::Object(json!({
        "type": "function", "name": "lookup"
    })));
    let (specific, _) = build_responses_tool_config_with_registry(&req).expect("specific");
    assert_eq!(
        specific.expect("config")["toolChoice"],
        json!({"tool": {"name":"lookup"}})
    );

    req.tool_choice = Some(ResponsesToolChoice::String("none".to_string()));
    let (none, _) = build_responses_tool_config_with_registry(&req).expect("none");
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
        build_responses_tool_config_with_registry(&req).expect("hosted tool is ignored");
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
        .assemble(&req, "resolved", false)
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
        .assemble(&req, "resolved", false)
        .await
        .expect("assemble request with declared tools");

    let tool_config = assembled.tool_config.expect("declared tool config");
    let specs = tool_config["tools"].as_array().expect("tools array");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0]["toolSpec"]["name"], "declared_tool");
    assert_eq!(specs[0]["toolSpec"]["description"], "Declared tool");
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
