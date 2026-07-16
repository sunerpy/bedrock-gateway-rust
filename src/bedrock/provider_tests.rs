//! Unit tests for [`crate::bedrock::provider`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec). The
//! source file declares this via a `#[path]` mod tests, so the top-level
//! `use super::*;` resolves to the implementation module.

use super::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::bedrock::client::build_aws_config;
use crate::bedrock::translate::ReqwestImageResolver;
use crate::domain::{BudgetRatios, Capability, ReasoningPath, ResponsesBackend};
use crate::openai::schema::{
    Function, Message as OpenAiMessage, ResponseFunction, Tool as OpenAiTool, ToolCall,
    ToolChoice as OpenAiToolChoice, ToolContentInput,
};
use serde_json::json;

#[derive(Debug)]
struct StubCaps;

impl ModelCapabilities for StubCaps {
    fn has(&self, _model: &str, _cap: Capability) -> bool {
        false
    }

    fn resolve_foundation(&self, model_or_profile: &str) -> String {
        model_or_profile.to_string()
    }

    fn budget_ratios(&self, _model: &str) -> Option<BudgetRatios> {
        None
    }

    fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
        None
    }

    fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
        None
    }

    fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
        None
    }

    fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
        None
    }

    fn beta_headers(&self, _model: &str) -> Vec<String> {
        Vec::new()
    }

    fn reasoning_path(&self, _model: &str) -> ReasoningPath {
        ReasoningPath::None
    }

    fn responses_backend(&self, _model: &str) -> ResponsesBackend {
        ResponsesBackend::Converse
    }

    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
    }

    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        None
    }
}

fn test_settings() -> AppSettings {
    AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-east-1".to_string(),
        default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
        enable_cross_region_inference: true,
        enable_application_inference_profiles: true,
        enable_prompt_caching: true,
        prompt_cache_ttl: "5m".to_string(),
        api_key: None,
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        bind_addr: "0.0.0.0".to_string(),
        port: 8080,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    }
}

async fn test_provider() -> BedrockChatProvider {
    let settings = Arc::new(test_settings());
    let sdk_config = build_aws_config(&settings).await;
    BedrockChatProvider::new(
        BedrockClients::new(&sdk_config),
        Arc::new(StubCaps),
        Arc::new(RegionRoutingConfig::default()),
        Arc::new(ReqwestImageResolver::new(|_| false)),
        settings,
        Arc::new(CacheSupportRegistry::new()),
    )
}

fn base_chat_request(messages: Vec<OpenAiMessage>, tools: Option<Vec<OpenAiTool>>) -> ChatRequest {
    ChatRequest {
        messages,
        model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
        stream_options: None,
        temperature: None,
        top_p: None,
        user: None,
        max_tokens: Some(2048),
        max_completion_tokens: None,
        reasoning_effort: None,
        n: None,
        tools,
        tool_choice: OpenAiToolChoice::default(),
        stop: None,
        response_format: None,
        extra_body: None,
        extra: HashMap::new(),
    }
}

fn normalized_chat_request(request: ChatRequest) -> NormalizedChatRequest {
    let resolved_model = request.model.clone();
    NormalizedChatRequest {
        request,
        resolved_model,
        request_id: Arc::<str>::from("req-test-toolconfig"),
        received_at: Instant::now(),
        raw_body: bytes::Bytes::new(),
    }
}

fn assistant_tool_call(name: &str) -> OpenAiMessage {
    OpenAiMessage::Assistant {
        name: None,
        content: None,
        tool_calls: Some(vec![ToolCall {
            index: Some(0),
            id: Some("call_weather".to_string()),
            r#type: "function".to_string(),
            function: ResponseFunction {
                name: Some(name.to_string()),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            },
        }]),
    }
}

fn tool_result_message() -> OpenAiMessage {
    OpenAiMessage::Tool {
        content: ToolContentInput::Text("sunny".to_string()),
        tool_call_id: "call_weather".to_string(),
    }
}

fn function_tool(name: &str, description: &str) -> OpenAiTool {
    OpenAiTool {
        r#type: "function".to_string(),
        function: Function {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        },
    }
}

fn first_tool_spec(tool_config: &Value) -> Option<&Value> {
    tool_config
        .get("tools")
        .and_then(Value::as_array)
        .and_then(|tools| tools.first())
        .and_then(|tool| tool.get("toolSpec"))
}

fn first_tool_name(tool_config: &Value) -> Option<&str> {
    first_tool_spec(tool_config)
        .and_then(|spec| spec.get("name"))
        .and_then(Value::as_str)
}

fn first_tool_description(tool_config: &Value) -> Option<&str> {
    first_tool_spec(tool_config)
        .and_then(|spec| spec.get("description"))
        .and_then(Value::as_str)
}

#[tokio::test]
async fn chat_assemble_synthesizes_toolconfig_for_continuation() {
    let provider = test_provider().await;
    let req = normalized_chat_request(base_chat_request(
        vec![assistant_tool_call("get_weather"), tool_result_message()],
        None,
    ));

    let (args, _) = provider.assemble(&req, false).await.expect("assemble");
    let tool_config = args.tool_config.as_ref().expect("toolConfig");

    assert_eq!(first_tool_name(tool_config), Some("get_weather"));
}

#[tokio::test]
async fn chat_assemble_real_tools_unchanged() {
    let provider = test_provider().await;
    let req = normalized_chat_request(base_chat_request(
        vec![assistant_tool_call("get_weather"), tool_result_message()],
        Some(vec![function_tool("get_weather", "Real weather tool")]),
    ));

    let (args, _) = provider.assemble(&req, false).await.expect("assemble");
    let tool_config = args.tool_config.as_ref().expect("toolConfig");

    assert_eq!(first_tool_name(tool_config), Some("get_weather"));
    assert_eq!(
        first_tool_description(tool_config),
        Some("Real weather tool")
    );
}

// ----- json_to_document / document_to_json round trips -------------------

#[test]
fn json_document_round_trip_preserves_shape() {
    let value = json!({
        "thinking": { "type": "enabled", "budget_tokens": 1024 },
        "anthropic_beta": ["ctx-1m", "other"],
        "flag": true,
        "ratio": 0.5,
        "neg": -7,
        "nothing": null
    });
    let doc = json_to_document(&value);
    let back = document_to_json(&doc);
    assert_eq!(back, value, "round trip must preserve the JSON shape");
}

#[test]
fn json_to_document_keeps_integers_as_integers() {
    let doc = json_to_document(&json!(42));
    assert!(matches!(doc, Document::Number(Number::PosInt(42))));
    let neg = json_to_document(&json!(-5));
    assert!(matches!(neg, Document::Number(Number::NegInt(-5))));
    let f = json_to_document(&json!(1.5));
    assert!(matches!(f, Document::Number(Number::Float(_))));
}

// ----- build_sdk_messages ------------------------------------------------

#[test]
fn build_sdk_messages_text_turn() {
    let messages = json!([
        { "role": "user", "content": [{ "text": "Hello!" }] }
    ]);
    let sdk = build_sdk_messages(&messages).expect("messages map");
    assert_eq!(sdk.len(), 1);
    assert_eq!(sdk[0].role(), &ConversationRole::User);
    match &sdk[0].content()[0] {
        ContentBlock::Text(t) => assert_eq!(t, "Hello!"),
        other => panic!("expected text block, got {other:?}"),
    }
}

#[test]
fn build_sdk_messages_tool_use_and_result() {
    let messages = json!([
        { "role": "assistant", "content": [
            { "toolUse": { "toolUseId": "t1", "name": "get_weather", "input": { "city": "Paris" } } }
        ]},
        { "role": "user", "content": [
            { "toolResult": { "toolUseId": "t1", "content": [{ "text": "sunny" }] } }
        ]}
    ]);
    let sdk = build_sdk_messages(&messages).expect("messages map");
    assert_eq!(sdk.len(), 2);
    // assistant toolUse
    match &sdk[0].content()[0] {
        ContentBlock::ToolUse(tu) => {
            assert_eq!(tu.tool_use_id(), "t1");
            assert_eq!(tu.name(), "get_weather");
            assert_eq!(document_to_json(tu.input()), json!({ "city": "Paris" }));
        }
        other => panic!("expected toolUse, got {other:?}"),
    }
    // user toolResult
    match &sdk[1].content()[0] {
        ContentBlock::ToolResult(tr) => {
            assert_eq!(tr.tool_use_id(), "t1");
            match &tr.content()[0] {
                ToolResultContentBlock::Text(t) => assert_eq!(t, "sunny"),
                other => panic!("expected text result, got {other:?}"),
            }
        }
        other => panic!("expected toolResult, got {other:?}"),
    }
}

#[test]
fn build_sdk_messages_image_and_cache_point() {
    // "hi" = [104, 105].
    let messages = json!([
        { "role": "user", "content": [
            { "text": "look" },
            { "image": { "format": "png", "source": { "bytes": [104, 105] } } },
            { "cachePoint": { "type": "default" } }
        ]}
    ]);
    let sdk = build_sdk_messages(&messages).expect("messages map");
    let content = sdk[0].content();
    assert_eq!(content.len(), 3);
    match &content[1] {
        ContentBlock::Image(img) => {
            assert_eq!(img.format(), &ImageFormat::Png);
        }
        other => panic!("expected image, got {other:?}"),
    }
    assert!(matches!(content[2], ContentBlock::CachePoint(_)));
}

#[test]
fn build_sdk_messages_filters_empty_text_blocks() {
    let messages = json!([
        { "role": "user", "content": [{ "text": "" }, { "text": "Hello!" }] }
    ]);
    let sdk = build_sdk_messages(&messages).expect("messages map");
    let content = sdk[0].content();

    assert_eq!(content.len(), 1);
    match &content[0] {
        ContentBlock::Text(t) => assert_eq!(t, "Hello!"),
        other => panic!("expected text block, got {other:?}"),
    }
}

#[test]
fn build_sdk_messages_rejects_all_empty_text_turn() {
    let messages = json!([
        { "role": "user", "content": [{ "text": "" }] }
    ]);
    let err = build_sdk_messages(&messages).expect_err("must reject all-empty turn");

    match err {
        AppError::Internal(message) => {
            assert!(message.contains("message turn content contained no SDK content blocks"));
        }
        other => panic!("expected Internal, got {other:?}"),
    }
}

#[test]
fn build_sdk_messages_rejects_missing_role() {
    let messages = json!([{ "content": [{ "text": "x" }] }]);
    let err = build_sdk_messages(&messages).expect_err("must error");
    assert!(matches!(err, AppError::Internal(_)));
}

// ----- build_sdk_system --------------------------------------------------

#[test]
fn build_sdk_system_text_and_cache_point() {
    let system = json!([
        { "text": "You are helpful." },
        { "cachePoint": { "type": "default" } }
    ]);
    let sdk = build_sdk_system(&system).expect("system map");
    assert_eq!(sdk.len(), 2);
    match &sdk[0] {
        SystemContentBlock::Text(t) => assert_eq!(t, "You are helpful."),
        other => panic!("expected text, got {other:?}"),
    }
    assert!(matches!(sdk[1], SystemContentBlock::CachePoint(_)));
}

#[test]
fn build_sdk_system_empty_array() {
    let sdk = build_sdk_system(&json!([])).expect("empty system");
    assert!(sdk.is_empty());
}

#[test]
fn build_sdk_system_skips_empty_text_blocks() {
    let system = json!([
        { "text": "" },
        { "text": "You are helpful." }
    ]);
    let sdk = build_sdk_system(&system).expect("system map");

    assert_eq!(sdk.len(), 1);
    match &sdk[0] {
        SystemContentBlock::Text(t) => assert_eq!(t, "You are helpful."),
        other => panic!("expected text, got {other:?}"),
    }
}

// ----- build_sdk_inference_config ----------------------------------------

#[test]
fn build_sdk_inference_config_full() {
    let cfg = json!({
        "maxTokens": 2048,
        "temperature": 0.7,
        "topP": 0.9,
        "stopSequences": ["STOP", "END"]
    });
    let ic = build_sdk_inference_config(&cfg);
    assert_eq!(ic.max_tokens(), Some(2048));
    assert!((ic.temperature().unwrap() - 0.7).abs() < 1e-6);
    assert!((ic.top_p().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(
        ic.stop_sequences(),
        &["STOP".to_string(), "END".to_string()]
    );
}

#[test]
fn build_sdk_inference_config_minimal() {
    let ic = build_sdk_inference_config(&json!({ "maxTokens": 16 }));
    assert_eq!(ic.max_tokens(), Some(16));
    assert!(ic.temperature().is_none());
    assert!(ic.top_p().is_none());
}

// ----- build_sdk_tool_config ---------------------------------------------

#[test]
fn build_sdk_tool_config_with_choice() {
    let tc = json!({
        "tools": [
            { "toolSpec": {
                "name": "get_weather",
                "description": "Get the weather",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }}
        ],
        "toolChoice": { "any": {} }
    });
    let sdk = build_sdk_tool_config(&tc).expect("tool config");
    assert_eq!(sdk.tools().len(), 1);
    match &sdk.tools()[0] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "get_weather");
            assert_eq!(spec.description(), Some("Get the weather"));
        }
        other => panic!("expected toolSpec, got {other:?}"),
    }
    assert!(matches!(sdk.tool_choice(), Some(ToolChoice::Any(_))));
}

#[test]
fn build_sdk_tool_config_auto_and_specific() {
    let auto = build_sdk_tool_choice(&json!({ "auto": {} })).expect("auto");
    assert!(matches!(auto, ToolChoice::Auto(_)));
    let any = build_sdk_tool_choice(&json!({ "any": {} })).expect("any");
    assert!(matches!(any, ToolChoice::Any(_)));
    let specific = build_sdk_tool_choice(&json!({ "tool": { "name": "fx" } })).expect("specific");
    match specific {
        ToolChoice::Tool(t) => assert_eq!(t.name(), "fx"),
        other => panic!("expected specific, got {other:?}"),
    }
}

#[test]
fn build_sdk_tool_config_no_choice_omits_choice() {
    let tc = json!({
        "tools": [
            { "toolSpec": { "name": "n", "inputSchema": { "json": { "type": "object" } } } }
        ]
    });
    let sdk = build_sdk_tool_config(&tc).expect("config");
    assert!(sdk.tool_choice().is_none());
}

#[test]
fn build_sdk_tool_config_with_cache_point() {
    let tc = json!({
        "tools": [
            { "toolSpec": {
                "name": "get_weather",
                "description": "Get the weather",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }},
            { "cachePoint": { "type": "default" } }
        ]
    });
    let sdk = build_sdk_tool_config(&tc).expect("tool config with cache");
    assert_eq!(sdk.tools().len(), 2);
    match &sdk.tools()[0] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "get_weather");
            assert_eq!(spec.description(), Some("Get the weather"));
        }
        other => panic!("expected toolSpec at [0], got {other:?}"),
    }
    match &sdk.tools()[1] {
        Tool::CachePoint(_) => {}
        other => panic!("expected cachePoint at [1], got {other:?}"),
    }
}

#[test]
fn build_sdk_tool_config_empty_description_omitted() {
    let tc = json!({
        "tools": [
            { "toolSpec": {
                "name": "tool_with_empty_desc",
                "description": "",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }},
            { "toolSpec": {
                "name": "tool_with_real_desc",
                "description": "A real description",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }},
            { "toolSpec": {
                "name": "tool_with_whitespace_desc",
                "description": "   ",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }},
            { "toolSpec": {
                "name": "tool_no_desc",
                "inputSchema": { "json": { "type": "object", "properties": {} } }
            }}
        ]
    });
    let sdk = build_sdk_tool_config(&tc).expect("tool config");
    assert_eq!(sdk.tools().len(), 4);

    // Tool 1: empty description should be omitted (None)
    match &sdk.tools()[0] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "tool_with_empty_desc");
            assert_eq!(
                spec.description(),
                None,
                "empty description must be None, not Some(\"\")"
            );
        }
        other => panic!("expected toolSpec at [0], got {other:?}"),
    }

    // Tool 2: real description must be preserved
    match &sdk.tools()[1] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "tool_with_real_desc");
            assert_eq!(spec.description(), Some("A real description"));
        }
        other => panic!("expected toolSpec at [1], got {other:?}"),
    }

    // Tool 3: whitespace-only description should be omitted (None)
    match &sdk.tools()[2] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "tool_with_whitespace_desc");
            assert_eq!(
                spec.description(),
                None,
                "whitespace-only description must be None"
            );
        }
        other => panic!("expected toolSpec at [2], got {other:?}"),
    }

    // Tool 4: no description key should remain None
    match &sdk.tools()[3] {
        Tool::ToolSpec(spec) => {
            assert_eq!(spec.name(), "tool_no_desc");
            assert_eq!(spec.description(), None);
        }
        other => panic!("expected toolSpec at [3], got {other:?}"),
    }
}

// ----- skip_tool_choice_for (the documented model-shape check) -----------

#[test]
fn skip_tool_choice_only_for_llama_3_1() {
    assert!(skip_tool_choice_for("meta.llama3-1-8b-instruct-v1:0"));
    assert!(skip_tool_choice_for("us.meta.llama3-1-70b-instruct-v1:0"));
    // Other models keep tool_choice.
    assert!(!skip_tool_choice_for("anthropic.claude-3-sonnet-v1:0"));
    assert!(!skip_tool_choice_for("meta.llama3-3-70b-instruct-v1:0"));
}

// ----- converse_output_to_json (response bridge) -------------------------
//
// The SDK ConverseOutput operation type is constructible via its builder, so
// the response bridge is exercised end-to-end offline (no AWS).

#[test]
fn converse_output_to_json_text_and_usage() {
    use aws_sdk_bedrockruntime::operation::converse::ConverseOutput as ConverseOp;
    use aws_sdk_bedrockruntime::types::{ConverseOutput, Message, StopReason, TokenUsage};

    let message = Message::builder()
        .role(ConversationRole::Assistant)
        .set_content(Some(vec![ContentBlock::Text("Hi there".to_string())]))
        .build()
        .unwrap();
    let op = ConverseOp::builder()
        .output(ConverseOutput::Message(message))
        .stop_reason(StopReason::EndTurn)
        .usage(
            TokenUsage::builder()
                .input_tokens(8)
                .output_tokens(2)
                .total_tokens(10)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let value = converse_output_to_json(&op);
    assert_eq!(value["output"]["message"]["content"][0]["text"], "Hi there");
    assert_eq!(value["stopReason"], "end_turn");
    assert_eq!(value["usage"]["inputTokens"], 8);
    assert_eq!(value["usage"]["outputTokens"], 2);
    assert_eq!(value["usage"]["totalTokens"], 10);

    // And feeding it through the pure mapper yields a clean ChatResponse.
    let resp = response::from_converse_output(&value, "m", "chatcmpl-x").expect("map");
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hi there"));
    assert_eq!(resp.usage.prompt_tokens, 8);
}

#[test]
fn converse_output_to_json_tool_use() {
    use aws_sdk_bedrockruntime::operation::converse::ConverseOutput as ConverseOp;
    use aws_sdk_bedrockruntime::types::{ConverseOutput, Message, StopReason, TokenUsage};

    let tu = ToolUseBlock::builder()
        .tool_use_id("call_1")
        .name("get_weather")
        .input(json_to_document(&json!({ "city": "Paris" })))
        .build()
        .unwrap();
    let message = Message::builder()
        .role(ConversationRole::Assistant)
        .set_content(Some(vec![ContentBlock::ToolUse(tu)]))
        .build()
        .unwrap();
    let op = ConverseOp::builder()
        .output(ConverseOutput::Message(message))
        .stop_reason(StopReason::ToolUse)
        .usage(
            TokenUsage::builder()
                .input_tokens(5)
                .output_tokens(3)
                .total_tokens(8)
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();

    let value = converse_output_to_json(&op);
    assert_eq!(value["stopReason"], "tool_use");
    let tool_use = &value["output"]["message"]["content"][0]["toolUse"];
    assert_eq!(tool_use["toolUseId"], "call_1");
    assert_eq!(tool_use["name"], "get_weather");
    assert_eq!(tool_use["input"], json!({ "city": "Paris" }));

    let resp = response::from_converse_output(&value, "m", "id").expect("map");
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    let calls = resp.choices[0].message.tool_calls.as_ref().expect("calls");
    assert_eq!(calls[0].id.as_deref(), Some("call_1"));
}

#[test]
fn build_sdk_tool_result_empty_text_becomes_json_empty_object() {
    let turn = tools::tool_message_to_tool_result_turn("call_1", "");
    let result =
        build_tool_result_block(&turn["content"][0]["toolResult"]).expect("tool result block");

    assert_eq!(result.content().len(), 1);
    match &result.content()[0] {
        ToolResultContentBlock::Json(doc) => assert_eq!(document_to_json(doc), json!({})),
        other => panic!("expected JSON empty object, got {other:?}"),
    }
}

#[test]
fn build_sdk_tool_result_empty_content_gets_json_empty_object() {
    let result = build_tool_result_block(&json!({
        "toolUseId": "call_1",
        "content": []
    }))
    .expect("tool result block");

    assert_eq!(result.content().len(), 1);
    match &result.content()[0] {
        ToolResultContentBlock::Json(doc) => assert_eq!(document_to_json(doc), json!({})),
        other => panic!("expected JSON empty object, got {other:?}"),
    }
}
