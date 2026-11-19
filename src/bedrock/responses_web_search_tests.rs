//! Tests for the web search loop ([`crate::bedrock::responses_web_search`],
//! plan C2 §11.4).
//!
//! The loop runs offline: a scripted [`StepTransport`] stands in for Bedrock
//! (it records every assembled request) and [`FakeSearch`] for AgentCore. The
//! assembly, the cache-strip retry, the translation and the stream state
//! machines are the production code. The `#[ignore]`d live tests at the end
//! call Bedrock and AgentCore in us-east-2 (L2–L8 of the design).

use super::*;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStart, ContentBlockStartEvent,
    ContentBlockStopEvent, ConversationRole, ConverseStreamMetadataEvent, ConverseStreamOutput,
    MessageStartEvent, MessageStopEvent, ReasoningContentBlockDelta, StopReason, TokenUsage,
    ToolUseBlockDelta, ToolUseBlockStart,
};
use aws_smithy_types::error::ErrorMetadata;
use base64::Engine as _;
use futures::StreamExt;

use crate::bedrock::anthropic_messages::MessagesStreamAdapter;
use crate::bedrock::cache_support::CacheSupportRegistry;
use crate::bedrock::capabilities::ConfigModelCapabilities;
use crate::bedrock::client::BedrockClients;
use crate::bedrock::translate::ImageResolver;
use crate::bedrock::web_search::{
    SearchError, SearchOutcome, SearchResult, WebSearchBackend, WebSearchLimits,
};
use crate::config::{AppSettings, ModelCapabilityConfig, RegionRoutingConfig};

const SONNET: &str = "global.anthropic.claude-sonnet-5-5";
const OPUS: &str = "global.anthropic.claude-opus-5-5";

// ---------------------------------------------------------------------------
// Fixtures (tests/golden/fixtures/responses_web_search, see its README)
// ---------------------------------------------------------------------------

macro_rules! fixture {
    ($name:literal) => {
        include_str!(concat!(
            "../../tests/golden/fixtures/responses_web_search/",
            $name
        ))
    };
}

fn fixture_json(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture json")
}

/// An AgentCore `InvokeWebSearch` response body as the backend would parse it.
fn agentcore_outcome(raw: &str) -> SearchOutcome {
    let body = fixture_json(raw);
    let results = body["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|r| SearchResult {
            title: r["title"].as_str().map(str::to_string),
            url: r["url"].as_str().map(str::to_string),
            text: r["text"].as_str().expect("text").to_string(),
            published_at: r["publishedAt"].as_str().map(str::to_string),
        })
        .collect();
    SearchOutcome {
        results,
        request_id: Some("agentcore-req-1".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Provider, settings, fakes
// ---------------------------------------------------------------------------

fn test_settings() -> AppSettings {
    AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-east-2".to_string(),
        default_model: "m".to_string(),
        default_embedding_model: "e".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching: true,
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
        web_search_backend: "agentcore".to_string(),
        web_search_region: None,
        web_search_max_calls: 5,
        web_search_timeout_secs: 12,
        web_search_max_results: 5,
        web_search_result_chars: 1500,
        web_search_max_tps: 5,
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
    }
}

fn limits() -> WebSearchLimits {
    WebSearchLimits::from_settings(&test_settings())
}

struct NoImages;

#[async_trait::async_trait]
impl ImageResolver for NoImages {
    fn supports_image(&self, _model_id: &str) -> bool {
        false
    }
    async fn fetch(&self, _url: &str) -> Result<(Vec<u8>, String), AppError> {
        Err(AppError::BadRequest("no images in these tests".to_string()))
    }
}

/// A Bedrock SDK config that never reaches AWS: static credentials, no
/// retries, and `endpoint` (a local mock server) when given.
fn offline_sdk_config(endpoint: Option<&str>) -> aws_config::SdkConfig {
    let credentials =
        aws_credential_types::Credentials::new("AKIDTEST", "secret", None, None, "test");
    let mut builder = aws_config::SdkConfig::builder()
        .region(aws_config::Region::new("us-east-2"))
        .behavior_version(aws_config::BehaviorVersion::latest())
        .credentials_provider(
            aws_credential_types::provider::SharedCredentialsProvider::new(credentials),
        )
        .retry_config(aws_smithy_types::retry::RetryConfig::disabled());
    if let Some(endpoint) = endpoint {
        builder = builder.endpoint_url(endpoint);
    }
    builder.build()
}

fn provider_with(
    backend: Option<Arc<dyn WebSearchBackend>>,
    limits: WebSearchLimits,
    endpoint: Option<&str>,
) -> BedrockResponsesProvider {
    let caps = Arc::new(ConfigModelCapabilities::new(
        ModelCapabilityConfig::load("config/models.toml").expect("config/models.toml"),
    ));
    BedrockResponsesProvider::new(
        BedrockClients::new(&offline_sdk_config(endpoint)),
        caps,
        Arc::new(RegionRoutingConfig::default()),
        Arc::new(NoImages),
        Arc::new(test_settings()),
        Arc::new(CacheSupportRegistry::new()),
        backend,
        limits,
    )
}

fn provider(search: &Arc<FakeSearch>) -> BedrockResponsesProvider {
    provider_with(
        Some(search.clone() as Arc<dyn WebSearchBackend>),
        limits(),
        None,
    )
}

/// One scripted search answer.
enum Reply {
    Results(SearchOutcome),
    Fail(SearchError),
    /// Never answers; sets the flag when the search future is dropped.
    Hang(Arc<AtomicBool>),
}

/// A scripted [`WebSearchBackend`] that records every query. With no reply
/// left it returns no results.
#[derive(Default)]
struct FakeSearch {
    replies: Mutex<VecDeque<Reply>>,
    queries: Mutex<Vec<(String, u32)>>,
    started: AtomicUsize,
}

impl FakeSearch {
    fn new(replies: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            ..Self::default()
        })
    }

    fn queries(&self) -> Vec<String> {
        self.queries
            .lock()
            .unwrap()
            .iter()
            .map(|(q, _)| q.clone())
            .collect()
    }
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl WebSearchBackend for FakeSearch {
    async fn search(&self, query: &str, max_results: u32) -> Result<SearchOutcome, SearchError> {
        self.queries
            .lock()
            .unwrap()
            .push((query.to_string(), max_results));
        self.started.fetch_add(1, Ordering::SeqCst);
        let reply = self.replies.lock().unwrap().pop_front();
        match reply {
            Some(Reply::Results(outcome)) => Ok(outcome),
            Some(Reply::Fail(error)) => Err(error),
            Some(Reply::Hang(flag)) => {
                let _guard = DropFlag(flag);
                futures::future::pending::<()>().await;
                unreachable!("a pending future never completes")
            }
            None => Ok(SearchOutcome::default()),
        }
    }
}

fn results(items: &[(&str, &str, &str)]) -> Reply {
    Reply::Results(SearchOutcome {
        results: items
            .iter()
            .map(|(title, url, text)| SearchResult {
                title: Some((*title).to_string()),
                url: Some((*url).to_string()),
                text: (*text).to_string(),
                published_at: None,
            })
            .collect(),
        request_id: Some("ac-1".to_string()),
    })
}

/// One scripted Bedrock call.
enum Step {
    Converse(Value),
    /// A Converse service error (`code`, `message`).
    ConverseService(&'static str, &'static str),
    ConverseStream(Vec<Result<ConverseStreamOutput, AppError>>),
    ConverseStreamService(&'static str, &'static str),
    /// A Messages response (InvokeModel lane).
    Invoke(Value),
    /// Messages stream events, mapped through the real adapter.
    InvokeStream(Vec<Value>),
    /// A send error before any event.
    Fail(AppError),
}

/// What the loop sent for one call.
#[derive(Debug, Clone, PartialEq)]
struct Sent {
    /// `converse` or `invoke`.
    api: &'static str,
    /// Converse: `{messages, system, tool_config, inference_config,
    /// additional_fields}`; InvokeModel: the Messages body.
    body: Value,
    /// The InvokeModel body bytes.
    bytes: Option<Vec<u8>>,
}

/// A scripted [`StepTransport`].
#[derive(Default)]
struct Scripted {
    steps: Mutex<VecDeque<Step>>,
    sent: Mutex<Vec<Sent>>,
}

impl Scripted {
    fn new(steps: Vec<Step>) -> Arc<Self> {
        Arc::new(Self {
            steps: Mutex::new(steps.into()),
            sent: Mutex::new(Vec::new()),
        })
    }

    fn sent(&self) -> Vec<Sent> {
        self.sent.lock().unwrap().clone()
    }

    fn next(&self) -> Step {
        self.steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("the loop made more Bedrock calls than scripted")
    }

    fn record_converse(&self, assembled: &AssembledConverse) {
        self.sent.lock().unwrap().push(Sent {
            api: "converse",
            body: json!({
                "messages": assembled.messages,
                "system": assembled.system,
                "tool_config": assembled.tool_config,
                "inference_config": assembled.inference_config,
                "additional_fields": assembled.additional_fields,
            }),
            bytes: None,
        });
    }

    fn record_invoke(&self, body: &[u8]) {
        self.sent.lock().unwrap().push(Sent {
            api: "invoke",
            body: serde_json::from_slice(body).expect("Messages body is JSON"),
            bytes: Some(body.to_vec()),
        });
    }
}

#[async_trait::async_trait]
impl StepTransport for Scripted {
    async fn converse(
        &self,
        _request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<Value, SendError<ConverseError>> {
        self.record_converse(assembled);
        match self.next() {
            Step::Converse(output) => Ok(output),
            Step::ConverseService(code, message) => {
                Err(SendError::Service(ConverseError::generic(
                    ErrorMetadata::builder().code(code).message(message).build(),
                )))
            }
            Step::Fail(error) => Err(SendError::App(error)),
            _ => panic!("scripted step is not a Converse call"),
        }
    }

    async fn converse_stream(
        &self,
        _request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<StepEvents, SendError<ConverseStreamError>> {
        self.record_converse(assembled);
        match self.next() {
            Step::ConverseStream(events) => Ok(futures::stream::iter(events).boxed()),
            Step::ConverseStreamService(code, message) => {
                Err(SendError::Service(ConverseStreamError::generic(
                    ErrorMetadata::builder().code(code).message(message).build(),
                )))
            }
            Step::Fail(error) => Err(SendError::App(error)),
            _ => panic!("scripted step is not a ConverseStream call"),
        }
    }

    async fn invoke(&self, _request_model: &str, body: Vec<u8>) -> Result<Value, AppError> {
        self.record_invoke(&body);
        match self.next() {
            Step::Invoke(output) => Ok(output),
            Step::Fail(error) => Err(error),
            _ => panic!("scripted step is not an InvokeModel call"),
        }
    }

    async fn invoke_stream(
        &self,
        _request_model: &str,
        body: Vec<u8>,
    ) -> Result<StepEvents, AppError> {
        self.record_invoke(&body);
        match self.next() {
            Step::InvokeStream(events) => {
                let mut adapter = MessagesStreamAdapter::new();
                let mut out = Vec::new();
                for event in events {
                    match adapter.map_event(&event) {
                        Ok(mapped) => out.extend(mapped.into_iter().map(Ok)),
                        Err(error) => {
                            out.push(Err(error));
                            break;
                        }
                    }
                }
                Ok(futures::stream::iter(out).boxed())
            }
            Step::Fail(error) => Err(error),
            _ => panic!("scripted step is not an InvokeModelWithResponseStream call"),
        }
    }
}

// ---------------------------------------------------------------------------
// Step outputs: written once as Converse output, run on every path
// ---------------------------------------------------------------------------

fn think(text: &str, signature: &str) -> Value {
    json!({ "reasoningContent": { "reasoningText": { "text": text, "signature": signature } } })
}

fn say(text: &str) -> Value {
    json!({ "text": text })
}

fn search_call(id: &str, query: &str) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": "web_search", "input": { "query": query } } })
}

fn tool_call(id: &str, name: &str, input: Value) -> Value {
    json!({ "toolUse": { "toolUseId": id, "name": name, "input": input } })
}

fn output(content: Vec<Value>, stop: &str, usage: (i32, i32)) -> Value {
    json!({
        "output": { "message": { "role": "assistant", "content": content } },
        "stopReason": stop,
        "usage": { "inputTokens": usage.0, "outputTokens": usage.1, "totalTokens": usage.0 + usage.1 },
    })
}

/// The four ways a step reaches the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    Converse,
    ConverseStream,
    Invoke,
    InvokeStream,
}

impl Path {
    const ALL: [Path; 4] = [
        Path::Converse,
        Path::ConverseStream,
        Path::Invoke,
        Path::InvokeStream,
    ];

    fn streams(self) -> bool {
        matches!(self, Path::ConverseStream | Path::InvokeStream)
    }

    fn tool_search(self) -> bool {
        matches!(self, Path::Invoke | Path::InvokeStream)
    }

    fn step(self, output: &Value) -> Step {
        match self {
            Path::Converse => Step::Converse(output.clone()),
            Path::ConverseStream => Step::ConverseStream(converse_stream_events(output)),
            Path::Invoke => Step::Invoke(messages_response(output)),
            Path::InvokeStream => Step::InvokeStream(messages_stream(output)),
        }
    }
}

fn content_of(output: &Value) -> Vec<Value> {
    output["output"]["message"]["content"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn usage_of(output: &Value) -> (i32, i32) {
    let count = |key: &str| output["usage"][key].as_i64().unwrap_or(0) as i32;
    (count("inputTokens"), count("outputTokens"))
}

fn delta(index: i32, delta: ContentBlockDelta) -> ConverseStreamOutput {
    ConverseStreamOutput::ContentBlockDelta(
        ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(delta)
            .build()
            .unwrap(),
    )
}

/// A Converse output as the ConverseStream events Bedrock would send.
fn converse_stream_events(output: &Value) -> Vec<Result<ConverseStreamOutput, AppError>> {
    let mut events = vec![ConverseStreamOutput::MessageStart(
        MessageStartEvent::builder()
            .role(ConversationRole::Assistant)
            .build()
            .unwrap(),
    )];
    for (i, block) in content_of(output).iter().enumerate() {
        let index = i as i32;
        if let Some(text) = block.pointer("/reasoningContent/reasoningText") {
            let body = text["text"].as_str().unwrap();
            if !body.is_empty() {
                events.push(delta(
                    index,
                    ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(
                        body.to_string(),
                    )),
                ));
            }
            events.push(delta(
                index,
                ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                    text["signature"].as_str().unwrap().to_string(),
                )),
            ));
        } else if let Some(text) = block.get("text").and_then(Value::as_str) {
            // Two deltas, so a link can span them.
            let chars: Vec<char> = text.chars().collect();
            let (a, b) = chars.split_at(chars.len() / 2);
            for part in [a, b] {
                if !part.is_empty() {
                    events.push(delta(index, ContentBlockDelta::Text(part.iter().collect())));
                }
            }
        } else if let Some(call) = block.get("toolUse") {
            events.push(ConverseStreamOutput::ContentBlockStart(
                ContentBlockStartEvent::builder()
                    .content_block_index(index)
                    .start(ContentBlockStart::ToolUse(
                        ToolUseBlockStart::builder()
                            .tool_use_id(call["toolUseId"].as_str().unwrap())
                            .name(call["name"].as_str().unwrap())
                            .build()
                            .unwrap(),
                    ))
                    .build()
                    .unwrap(),
            ));
            events.push(delta(
                index,
                ContentBlockDelta::ToolUse(
                    ToolUseBlockDelta::builder()
                        .input(call["input"].to_string())
                        .build()
                        .unwrap(),
                ),
            ));
        }
        events.push(ConverseStreamOutput::ContentBlockStop(
            ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .unwrap(),
        ));
    }
    events.push(ConverseStreamOutput::MessageStop(
        MessageStopEvent::builder()
            .stop_reason(StopReason::from(output["stopReason"].as_str().unwrap()))
            .build()
            .unwrap(),
    ));
    let (input, output_tokens) = usage_of(output);
    events.push(ConverseStreamOutput::Metadata(
        ConverseStreamMetadataEvent::builder()
            .usage(
                TokenUsage::builder()
                    .input_tokens(input)
                    .output_tokens(output_tokens)
                    .total_tokens(input + output_tokens)
                    .build()
                    .unwrap(),
            )
            .build(),
    ));
    events.into_iter().map(Ok).collect()
}

fn messages_blocks(output: &Value) -> Vec<Value> {
    content_of(output)
        .iter()
        .map(|block| {
            if let Some(text) = block.pointer("/reasoningContent/reasoningText") {
                json!({ "type": "thinking", "thinking": text["text"], "signature": text["signature"] })
            } else if let Some(text) = block.get("text") {
                json!({ "type": "text", "text": text })
            } else {
                let call = &block["toolUse"];
                json!({ "type": "tool_use", "id": call["toolUseId"], "name": call["name"], "input": call["input"] })
            }
        })
        .collect()
}

/// A Converse output as the InvokeModel Messages response.
fn messages_response(output: &Value) -> Value {
    let (input, output_tokens) = usage_of(output);
    json!({
        "id": "msg_bdrk", "type": "message", "role": "assistant",
        "content": messages_blocks(output),
        "stop_reason": output["stopReason"],
        "usage": { "input_tokens": input, "output_tokens": output_tokens },
    })
}

/// A Converse output as InvokeModelWithResponseStream Messages events.
fn messages_stream(output: &Value) -> Vec<Value> {
    let (input, output_tokens) = usage_of(output);
    let mut events = vec![json!({ "type": "message_start", "message": {
        "id": "msg_bdrk", "type": "message", "role": "assistant", "content": [],
        "usage": { "input_tokens": input, "output_tokens": 1 } } })];
    for (i, block) in messages_blocks(output).iter().enumerate() {
        match block["type"].as_str().unwrap() {
            "thinking" => {
                events.push(json!({ "type": "content_block_start", "index": i,
                    "content_block": { "type": "thinking", "thinking": "", "signature": "" } }));
                if !block["thinking"].as_str().unwrap().is_empty() {
                    events.push(json!({ "type": "content_block_delta", "index": i,
                        "delta": { "type": "thinking_delta", "thinking": block["thinking"] } }));
                }
                events.push(json!({ "type": "content_block_delta", "index": i,
                    "delta": { "type": "signature_delta", "signature": block["signature"] } }));
            }
            "text" => {
                events.push(json!({ "type": "content_block_start", "index": i,
                    "content_block": { "type": "text", "text": "" } }));
                events.push(json!({ "type": "content_block_delta", "index": i,
                    "delta": { "type": "text_delta", "text": block["text"] } }));
            }
            _ => {
                events.push(
                    json!({ "type": "content_block_start", "index": i, "content_block": {
                    "type": "tool_use", "id": block["id"], "name": block["name"], "input": {} } }),
                );
                events.push(json!({ "type": "content_block_delta", "index": i,
                    "delta": { "type": "input_json_delta", "partial_json": block["input"].to_string() } }));
            }
        }
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(json!({ "type": "message_delta",
        "delta": { "stop_reason": output["stopReason"] },
        "usage": { "output_tokens": output_tokens } }));
    events.push(json!({ "type": "message_stop" }));
    events
}

// ---------------------------------------------------------------------------
// Requests and running the loop
// ---------------------------------------------------------------------------

fn request_1(tool_search: bool) -> ResponsesRequest {
    let raw = if tool_search {
        fixture!("toolsearch-request-1.json")
    } else {
        fixture!("request-1.json")
    };
    serde_json::from_str(raw).expect("request fixture")
}

fn normalized(request: ResponsesRequest, header: WebSearchHeader) -> NormalizedResponsesRequest {
    let resolved = request.model.clone();
    NormalizedResponsesRequest {
        request,
        resolved_model: resolved,
        request_id: Arc::from("req-ws-test"),
        received_at: Instant::now(),
        raw_body: bytes::Bytes::new(),
        web_search: header,
    }
}

/// The request with `input` replaced.
fn with_input(mut request: ResponsesRequest, input: Vec<Value>) -> ResponsesRequest {
    request.input = serde_json::from_value(Value::Array(input)).expect("input items");
    request
}

fn input_of(request: &ResponsesRequest) -> Vec<Value> {
    match serde_json::to_value(&request.input).unwrap() {
        Value::Array(items) => items,
        other => vec![json!({ "type": "message", "role": "user", "content": other })],
    }
}

/// What one run produced: the response (stream: from `response.completed` or
/// `response.incomplete`), every stream event, and what the loop sent.
struct Run {
    response: Option<ResponsesResponse>,
    events: Vec<Value>,
    sent: Vec<Sent>,
}

impl Run {
    fn response(&self) -> &ResponsesResponse {
        self.response.as_ref().expect("the run produced a response")
    }

    fn output(&self) -> Vec<Value> {
        self.response()
            .output
            .iter()
            .map(|item| serde_json::to_value(item).unwrap())
            .collect()
    }

    fn types(&self) -> Vec<String> {
        self.output()
            .iter()
            .map(|item| item["type"].as_str().unwrap().to_string())
            .collect()
    }

    fn event_types(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|e| e["type"].as_str().unwrap().to_string())
            .collect()
    }
}

async fn run_on(
    path: Path,
    provider: &BedrockResponsesProvider,
    req: &NormalizedResponsesRequest,
    outputs: &[Value],
) -> Result<Run, AppError> {
    let steps = outputs.iter().map(|output| path.step(output)).collect();
    run_steps(path, provider, req, steps).await
}

async fn run_steps(
    path: Path,
    provider: &BedrockResponsesProvider,
    req: &NormalizedResponsesRequest,
    steps: Vec<Step>,
) -> Result<Run, AppError> {
    let scripted = Scripted::new(steps);
    let plan = provider.web_search_plan(req);
    assert!(plan.declared, "these tests run declared requests");
    let lane = provider.lane(&req.request, &req.resolved_model);
    assert_eq!(lane == Lane::ToolSearch, path.tool_search(), "{path:?}");
    if path.streams() {
        let runtime = ResponsesStreamRuntime::new(
            req.request_id.clone(),
            Instant::now(),
            Duration::from_secs(5),
        );
        let transport: Arc<dyn StepTransport> = scripted.clone();
        let stream = respond_stream(
            provider.clone(),
            transport,
            req.clone(),
            plan,
            lane,
            runtime,
        )
        .await?;
        let events: Vec<ResponseStreamEvent> = stream.map(|event| event.unwrap()).collect().await;
        let response = events.iter().find_map(|event| match event {
            ResponseStreamEvent::Completed { response, .. }
            | ResponseStreamEvent::Incomplete { response, .. } => Some(response.clone()),
            _ => None,
        });
        Ok(Run {
            response,
            events: events
                .iter()
                .map(|event| serde_json::to_value(event).unwrap())
                .collect(),
            sent: scripted.sent(),
        })
    } else {
        let response = respond(provider, scripted.as_ref(), req, plan, lane).await?;
        Ok(Run {
            response: Some(response),
            events: Vec::new(),
            sent: scripted.sent(),
        })
    }
}

/// Item ids derive from the response id; replace its suffix with `S`.
fn normalize_ids(value: &mut Value, response_id: &str) {
    let suffix = response_id.trim_start_matches("resp_");
    match value {
        Value::String(s) if s.contains(suffix) => *s = s.replace(suffix, "S"),
        Value::Array(items) => items.iter_mut().for_each(|v| normalize_ids(v, response_id)),
        Value::Object(map) => {
            if map.contains_key("created_at") {
                map.insert("created_at".to_string(), json!(0));
            }
            map.values_mut().for_each(|v| normalize_ids(v, response_id));
        }
        _ => {}
    }
}

fn normalized_output(run: &Run) -> Vec<Value> {
    let mut output = Value::Array(run.output());
    normalize_ids(&mut output, &run.response().id);
    match output {
        Value::Array(items) => items,
        _ => unreachable!(),
    }
}

/// The payload of a gateway envelope.
fn envelope_payload(envelope: &str) -> Value {
    let encoded = envelope
        .strip_prefix("bedrock-reasoning-v1:")
        .expect("gateway envelope");
    serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded)
            .expect("base64url"),
    )
    .expect("payload json")
}

/// The carrier items of an output, decoded.
fn carriers(output: &[Value]) -> Vec<Value> {
    output
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .filter_map(|item| item["encrypted_content"].as_str())
        .map(envelope_payload)
        .filter(|payload| payload["version"] == 2)
        .collect()
}

/// The `toolResult` payload texts of a carrier, parsed.
fn carrier_results(carrier: &Value) -> Vec<Value> {
    carrier["tool_results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            serde_json::from_str(r["toolResult"]["content"][0]["text"].as_str().unwrap()).unwrap()
        })
        .collect()
}

/// zuno 0.160.1's replay of response items (the P1 probe): reasoning keeps
/// `{type, id, summary, encrypted_content}`, a message keeps its text only, a
/// `web_search_call` keeps `{type, id, status, action{type, query, queries}}`,
/// tool calls are replayed as they are.
fn zuno_replay(output: &[Value]) -> Vec<Value> {
    output
        .iter()
        .map(|item| match item["type"].as_str().unwrap() {
            "reasoning" => {
                let mut replay =
                    json!({ "type": "reasoning", "id": item["id"], "summary": item["summary"] });
                if let Some(encrypted) = item.get("encrypted_content") {
                    replay["encrypted_content"] = encrypted.clone();
                }
                replay
            }
            "message" => {
                let text: String = item["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect();
                json!({ "type": "message", "id": item["id"], "role": "assistant",
                        "content": [{ "type": "output_text", "text": text }] })
            }
            "web_search_call" => json!({
                "type": "web_search_call", "id": item["id"], "status": item["status"],
                "action": { "type": "search", "query": item["action"]["query"],
                            "queries": item["action"]["queries"] }
            }),
            _ => item.clone(),
        })
        .collect()
}

fn user(text: &str) -> Value {
    json!({ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": text }] })
}

/// The next client request after `run`: the original input, the replayed
/// output, then `tail` (tool outputs, a new user message).
fn next_request(original: &ResponsesRequest, run: &Run, tail: Vec<Value>) -> ResponsesRequest {
    let mut input = input_of(original);
    input.extend(zuno_replay(&run.output()));
    input.extend(tail);
    with_input(original.clone(), input)
}

// ---------------------------------------------------------------------------
// Plan (§2.2)
// ---------------------------------------------------------------------------

/// `(header, backend, capable, scan, declared, gate)`.
type PlanRow<'a> = (
    WebSearchHeader,
    bool,
    bool,
    Option<&'a HistoryScan>,
    bool,
    ExecGate,
);

#[test]
fn plan_matrix() {
    use WebSearchHeader::{Absent, Off, On};
    let marked = HistoryScan {
        marked: true,
        carrier_payloads: vec![r#"{"query":"q","results":[{"rank":1,"text":"t","text_truncated":false,"title":"T","url":"https://a.example/x"}],"source":"s","trust":"t","type":"web_search_results"}"#.to_string()],
        used_in_answer: 2,
        unmarked_tool_loop: false,
    };
    let unmarked = HistoryScan::default();
    // A tool loop whose thinking has no mark.
    let unmarked_loop = HistoryScan {
        unmarked_tool_loop: true,
        ..HistoryScan::default()
    };
    let marked_loop = HistoryScan {
        unmarked_tool_loop: true,
        ..marked.clone()
    };
    // (header, backend, capable, scan) → (declared, gate)
    let rows: Vec<PlanRow<'_>> = vec![
        (
            Absent,
            true,
            true,
            Some(&unmarked),
            false,
            ExecGate::Disabled,
        ),
        (Off, true, true, Some(&unmarked), false, ExecGate::Disabled),
        (Absent, false, false, None, false, ExecGate::Disabled),
        (On, true, true, Some(&unmarked), true, ExecGate::Search),
        (On, true, true, Some(&marked), true, ExecGate::Search),
        (On, true, true, None, true, ExecGate::Search),
        (
            On,
            false,
            true,
            Some(&unmarked),
            false,
            ExecGate::Unavailable,
        ),
        (
            On,
            true,
            false,
            Some(&unmarked),
            false,
            ExecGate::Unavailable,
        ),
        (On, false, true, Some(&marked), true, ExecGate::Unavailable),
        (On, true, false, Some(&marked), true, ExecGate::Unavailable),
        (Absent, true, true, Some(&marked), true, ExecGate::Disabled),
        (Off, false, false, Some(&marked), true, ExecGate::Disabled),
        // A tool loop with unmarked thinking does not declare the tool; with
        // marks it stays declared.
        (
            On,
            true,
            true,
            Some(&unmarked_loop),
            false,
            ExecGate::Search,
        ),
        (On, true, true, Some(&marked_loop), true, ExecGate::Search),
        (
            Off,
            true,
            true,
            Some(&marked_loop),
            true,
            ExecGate::Disabled,
        ),
    ];
    for (header, backend, capable, scan, declared, gate) in rows {
        let plan = plan_web_search(header, backend, capable, scan);
        assert_eq!(
            (plan.declared, plan.gate),
            (declared, gate),
            "{header:?} backend={backend} capable={capable} marked={:?}",
            scan.map(|s| s.marked)
        );
    }
    let plan = plan_web_search(Off, false, false, Some(&marked));
    assert_eq!(plan.used_in_answer, 2);
    assert_eq!(plan.sources.len(), 1);
    assert!(plan_web_search(On, true, true, None).sources.is_empty());
}

/// The provider's plan reads the capability from `config/models.toml` and the
/// backend switch from its own field.
#[test]
fn provider_plan_reads_capability_and_backend() {
    let search = FakeSearch::new(Vec::new());
    let with_backend = provider(&search);
    let without_backend = provider_with(None, limits(), None);
    let on = |model: &str| {
        let mut request = request_1(false);
        request.model = model.to_string();
        normalized(request, WebSearchHeader::On)
    };
    assert!(with_backend.web_search_plan(&on(SONNET)).declared);
    assert!(with_backend.web_search_plan(&on(OPUS)).declared);
    assert!(
        !with_backend
            .web_search_plan(&on("global.anthropic.claude-sonnet-5"))
            .declared
    );
    assert!(
        !with_backend
            .web_search_plan(&on("us.amazon.nova-pro-v1:0"))
            .declared
    );
    assert!(!without_backend.web_search_plan(&on(SONNET)).declared);
    assert_eq!(
        with_backend.web_search_plan(&on(SONNET)).gate,
        ExecGate::Search
    );
}

#[test]
fn loop_failure_code_classifies_later_binding_errors() {
    let binding = AppError::BadRequest(
        "The thinking block at index 3 is bound to a different conversation prefix.".to_string(),
    );
    let signature = AppError::BadRequest("Invalid `signature` in `thinking` block".to_string());
    let other = AppError::BadRequest("Too many tokens".to_string());
    assert_eq!(loop_failure_code(0, &binding), "context_length_exceeded");
    assert_eq!(loop_failure_code(1, &binding), "invalid_prompt");
    assert_eq!(loop_failure_code(2, &signature), "invalid_prompt");
    assert_eq!(loop_failure_code(1, &other), "context_length_exceeded");
    assert_eq!(
        loop_failure_code(1, &AppError::Throttled("slow down".to_string())),
        "rate_limit_exceeded"
    );
    assert_eq!(
        loop_failure_code(
            3,
            &AppError::UpstreamBedrock("thinking signature".to_string())
        ),
        "server_is_overloaded"
    );
}

// ---------------------------------------------------------------------------
// One search, then the answer (the golden scenario)
// ---------------------------------------------------------------------------

fn golden_steps() -> Vec<Value> {
    vec![
        fixture_json(fixture!("converse-1.json")),
        fixture_json(fixture!("converse-2.json")),
    ]
}

fn golden_search() -> Arc<FakeSearch> {
    FakeSearch::new(vec![Reply::Results(agentcore_outcome(fixture!(
        "agentcore-1.json"
    )))])
}

/// ConverseStream events from the compact JSONL shape of
/// `tests/golden/fixtures/` (`messageStart`, `contentBlockStart.toolUse`,
/// `contentBlockDelta.{text,reasoningText,reasoningSignature,toolUseInput}`,
/// `contentBlockStop`, `messageStop`, `metadata.usage`).
fn converse_jsonl(raw: &str) -> Vec<Result<ConverseStreamOutput, AppError>> {
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let line: Value = serde_json::from_str(line).expect("jsonl line");
            let index = |v: &Value| v["contentBlockIndex"].as_i64().unwrap_or(0) as i32;
            let event = if line.get("messageStart").is_some() {
                ConverseStreamOutput::MessageStart(
                    MessageStartEvent::builder()
                        .role(ConversationRole::Assistant)
                        .build()
                        .unwrap(),
                )
            } else if let Some(start) = line.get("contentBlockStart") {
                let call = &start["toolUse"];
                ConverseStreamOutput::ContentBlockStart(
                    ContentBlockStartEvent::builder()
                        .content_block_index(index(start))
                        .start(ContentBlockStart::ToolUse(
                            ToolUseBlockStart::builder()
                                .tool_use_id(call["toolUseId"].as_str().unwrap())
                                .name(call["name"].as_str().unwrap())
                                .build()
                                .unwrap(),
                        ))
                        .build()
                        .unwrap(),
                )
            } else if let Some(d) = line.get("contentBlockDelta") {
                let body = if let Some(text) = d["text"].as_str() {
                    ContentBlockDelta::Text(text.to_string())
                } else if let Some(text) = d["reasoningText"].as_str() {
                    ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Text(
                        text.to_string(),
                    ))
                } else if let Some(signature) = d["reasoningSignature"].as_str() {
                    ContentBlockDelta::ReasoningContent(ReasoningContentBlockDelta::Signature(
                        signature.to_string(),
                    ))
                } else {
                    ContentBlockDelta::ToolUse(
                        ToolUseBlockDelta::builder()
                            .input(d["toolUseInput"].as_str().unwrap())
                            .build()
                            .unwrap(),
                    )
                };
                delta(index(d), body)
            } else if let Some(stop) = line.get("contentBlockStop") {
                ConverseStreamOutput::ContentBlockStop(
                    ContentBlockStopEvent::builder()
                        .content_block_index(index(stop))
                        .build()
                        .unwrap(),
                )
            } else if let Some(stop) = line.get("messageStop") {
                ConverseStreamOutput::MessageStop(
                    MessageStopEvent::builder()
                        .stop_reason(StopReason::from(stop["stopReason"].as_str().unwrap()))
                        .build()
                        .unwrap(),
                )
            } else {
                let usage = &line["metadata"]["usage"];
                let count = |key: &str| usage[key].as_i64().map(|v| v as i32);
                ConverseStreamOutput::Metadata(
                    ConverseStreamMetadataEvent::builder()
                        .usage(
                            TokenUsage::builder()
                                .input_tokens(count("inputTokens").unwrap_or(0))
                                .output_tokens(count("outputTokens").unwrap_or(0))
                                .total_tokens(count("totalTokens").unwrap_or(0))
                                .set_cache_read_input_tokens(count("cacheReadInputTokens"))
                                .set_cache_write_input_tokens(count("cacheWriteInputTokens"))
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
            };
            Ok(event)
        })
        .collect()
}

fn messages_jsonl(raw: &str) -> Vec<Value> {
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("jsonl line"))
        .collect()
}

/// The golden scenario's two steps, from the fixture files of each path.
fn golden_fixture_steps(path: Path) -> Vec<Step> {
    match path {
        Path::Converse => golden_steps().into_iter().map(Step::Converse).collect(),
        Path::ConverseStream => vec![
            Step::ConverseStream(converse_jsonl(fixture!("converse-stream-1.jsonl"))),
            Step::ConverseStream(converse_jsonl(fixture!("converse-stream-2.jsonl"))),
        ],
        Path::Invoke => vec![
            Step::Invoke(fixture_json(fixture!("messages-1.json"))),
            Step::Invoke(fixture_json(fixture!("messages-2.json"))),
        ],
        Path::InvokeStream => vec![
            Step::InvokeStream(messages_jsonl(fixture!("messages-stream-1.jsonl"))),
            Step::InvokeStream(messages_stream(&golden_steps()[1])),
        ],
    }
}

async fn golden_run(path: Path) -> (Run, Arc<FakeSearch>) {
    let search = golden_search();
    let provider = provider(&search);
    let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
    let run = run_steps(path, &provider, &req, golden_fixture_steps(path))
        .await
        .expect("the loop answers");
    (run, search)
}

/// Readable form of an output: ids normalized, every gateway envelope
/// replaced by its decoded payload.
fn readable(run: &Run) -> Value {
    let mut output = normalized_output(run);
    for item in &mut output {
        if let Some(envelope) = item.get("encrypted_content").and_then(Value::as_str) {
            let payload = envelope_payload(envelope);
            item["encrypted_content"] = payload;
        }
    }
    Value::Array(output)
}

#[tokio::test]
async fn single_search_non_stream_golden() {
    let (run, search) = golden_run(Path::Converse).await;
    let response = run.response();
    assert_eq!(response.status, "completed");
    assert_eq!(search.queries(), vec!["AWS announcements November 2026"]);
    assert_eq!(
        readable(&run),
        fixture_json(fixture!("expected-response.json")),
        "output (ids normalized, envelopes decoded)"
    );
    // Summed raw usage through the one token formula; the echo fields are
    // the original request's.
    assert_eq!(response.usage.input_tokens, 2100 + 1800 + 2600 + 1800);
    assert_eq!(response.usage.output_tokens, 95 + 140);
    assert_eq!(
        response
            .usage
            .input_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens,
        1800
    );
    assert_eq!(
        response.usage.total_tokens,
        response.usage.input_tokens + response.usage.output_tokens
    );
    assert!(response.usage.reasoning_tokens() > 0);
    assert_eq!(response.max_output_tokens, Some(32000));
    assert_eq!(response.tools.as_ref().unwrap().len(), 4);
    assert!(matches!(
        &response.tool_choice,
        Some(ResponsesToolChoice::String(choice)) if choice == "auto"
    ));
    assert_eq!(run.sent.len(), 2);
}

/// The four paths produce the same output for the same steps (stream items
/// are the `output_item.done` items, in arrival order): the golden one.
#[tokio::test]
async fn every_path_produces_the_same_output() {
    let reference = fixture_json(fixture!("expected-response.json"));
    let (plain, _) = golden_run(Path::Converse).await;
    let reference_usage = serde_json::to_value(&plain.response().usage).unwrap();
    for path in Path::ALL {
        let (run, search) = golden_run(path).await;
        assert_eq!(readable(&run), reference, "{path:?}");
        if path != Path::InvokeStream {
            // The InvokeModel stream of step 1 is generated without cache
            // counts; the other three carry the fixtures' usage.
            assert_eq!(
                serde_json::to_value(&run.response().usage).unwrap(),
                reference_usage,
                "{path:?}: summed usage"
            );
        }
        assert_eq!(search.queries().len(), 1, "{path:?}");
        if path.streams() {
            let done: Vec<Value> = run
                .events
                .iter()
                .filter(|e| e["type"] == "response.output_item.done")
                .map(|e| e["item"].clone())
                .collect();
            assert_eq!(
                done,
                run.output(),
                "{path:?}: completed.output is the done items"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The invariant (§7): the loop's continuation = the replay's rebuild
// ---------------------------------------------------------------------------

/// Messages without cache points: Converse `{"cachePoint"}` blocks removed,
/// Messages `cache_control` keys removed.
fn strip_cache(messages: &Value) -> Vec<Value> {
    messages
        .as_array()
        .unwrap()
        .iter()
        .map(|message| {
            let mut message = message.clone();
            if let Some(content) = message["content"].as_array_mut() {
                content.retain(|block| block.get("cachePoint").is_none());
                for block in content.iter_mut() {
                    if let Some(block) = block.as_object_mut() {
                        block.remove("cache_control");
                    }
                }
            }
            message
        })
        .collect()
}

/// `(tools, system, messages)` of a sent request, in its own API's shape.
fn prefix_parts(sent: &Sent) -> (Value, Value, Value) {
    match sent.api {
        "converse" => (
            sent.body["tool_config"]["tools"].clone(),
            sent.body["system"].clone(),
            sent.body["messages"].clone(),
        ),
        _ => (
            sent.body["tools"].clone(),
            sent.body.get("system").cloned().unwrap_or(Value::Null),
            sent.body["messages"].clone(),
        ),
    }
}

/// Assert the four invariant clauses between every continuation the loop sent
/// (`loop_sent[1..]`) and the request a replay builds (`replay_sent`, sent
/// twice to check determinism).
fn assert_invariant(loop_sent: &[Sent], replay_sent: &Sent, replay_again: &Sent, label: &str) {
    let (r_tools, r_system, r_messages) = prefix_parts(replay_sent);
    let names: Vec<&str> = r_tools
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|tool| {
            tool.pointer("/toolSpec/name")
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(
        names.last(),
        Some(&"web_search"),
        "{label}: the replay declares the tool"
    );
    let r_messages = strip_cache(&r_messages);
    for (k, continuation) in loop_sent.iter().enumerate().skip(1) {
        let (tools, system, messages) = prefix_parts(continuation);
        assert_eq!(tools, r_tools, "{label}: tools of call {k}");
        assert_eq!(system, r_system, "{label}: system of call {k}");
        let messages = strip_cache(&messages);
        // A continuation ends with the results of the step before it.
        let last = messages.last().expect("messages");
        assert_eq!(last["role"], "user", "{label}: call {k}");
        assert!(
            last["content"].as_array().unwrap().iter().all(|block| {
                block.get("toolResult").is_some() || block["type"] == "tool_result"
            }),
            "{label}: call {k} ends with tool results: {last}"
        );
        assert!(
            messages.len() < r_messages.len(),
            "{label}: call {k} has at least as many messages as the replay"
        );
        assert_eq!(
            messages.as_slice(),
            &r_messages[..messages.len()],
            "{label}: messages of call {k} are a prefix of the replay"
        );
    }
    match (&replay_sent.bytes, &replay_again.bytes) {
        (Some(a), Some(b)) => assert_eq!(a, b, "{label}: the replay's body bytes"),
        _ => assert_eq!(
            serde_json::to_vec(&replay_sent.body).unwrap(),
            serde_json::to_vec(&replay_again.body).unwrap(),
            "{label}: the replay's assembled request"
        ),
    }
}

/// Run `steps` for `req`, then the next client request (`tail` appended) with
/// `header`, twice, and check the invariant.
async fn check_invariant(
    path: Path,
    req: NormalizedResponsesRequest,
    steps: &[Value],
    replies: Vec<Reply>,
    tail: Vec<Value>,
    header: WebSearchHeader,
    label: &str,
) -> Run {
    let search = FakeSearch::new(replies);
    let provider = provider(&search);
    let run = run_on(path, &provider, &req, steps)
        .await
        .expect("loop runs");
    assert_eq!(run.sent.len(), steps.len(), "{label}");

    let next = normalized(next_request(&req.request, &run, tail), header);
    let answer = output(vec![say("Thanks!")], "end_turn", (10, 2));
    let replay = run_on(path, &provider, &next, std::slice::from_ref(&answer))
        .await
        .expect("replay runs");
    let again = run_on(path, &provider, &next, std::slice::from_ref(&answer))
        .await
        .expect("replay runs again");
    assert_invariant(&run.sent, &replay.sent[0], &again.sent[0], label);
    run
}

fn two_search_steps() -> Vec<Value> {
    vec![
        output(
            vec![
                think("Search first.", "sig-a"),
                say("Searching."),
                search_call("t1", "aws news"),
            ],
            "tool_use",
            (100, 20),
        ),
        output(
            vec![
                think("Need two more.", "sig-b"),
                search_call("t2", "bedrock"),
                search_call("t3", "agentcore"),
            ],
            "tool_use",
            (150, 30),
        ),
        output(
            vec![
                think("Enough.", "sig-c"),
                say("AWS shipped search ([AWS](https://aws.example/1))."),
            ],
            "end_turn",
            (200, 40),
        ),
    ]
}

fn three_results() -> Vec<Reply> {
    vec![
        results(&[("AWS", "https://aws.example/1", "one")]),
        results(&[("B", "https://b.example/2", "two")]),
        results(&[]),
    ]
}

#[tokio::test]
async fn invariant_converse_lane() {
    for header in [
        WebSearchHeader::Off,
        WebSearchHeader::Absent,
        WebSearchHeader::On,
    ] {
        let run = check_invariant(
            Path::Converse,
            normalized(request_1(false), WebSearchHeader::On),
            &two_search_steps(),
            three_results(),
            vec![user("Thanks, and S3?")],
            header,
            "converse",
        )
        .await;
        assert_eq!(carriers(&run.output()).len(), 2);
    }
}

#[tokio::test]
async fn invariant_tool_search_lane() {
    for header in [WebSearchHeader::Off, WebSearchHeader::On] {
        check_invariant(
            Path::Invoke,
            normalized(request_1(true), WebSearchHeader::On),
            &two_search_steps(),
            three_results(),
            vec![user("Thanks, and S3?")],
            header,
            "tool search",
        )
        .await;
    }
}

/// The streamed loop sends the same continuations as the non-stream loop.
#[tokio::test]
async fn invariant_holds_for_streamed_steps() {
    for (path, tool_search) in [(Path::ConverseStream, false), (Path::InvokeStream, true)] {
        let search = FakeSearch::new(three_results());
        let provider = provider(&search);
        let req = normalized(request_1(tool_search), WebSearchHeader::On);
        let streamed = run_on(path, &provider, &req, &two_search_steps())
            .await
            .unwrap();
        let search = FakeSearch::new(three_results());
        let provider = self::provider(&search);
        let plain_path = if tool_search {
            Path::Invoke
        } else {
            Path::Converse
        };
        let plain = run_on(plain_path, &provider, &req, &two_search_steps())
            .await
            .unwrap();
        assert_eq!(streamed.sent.len(), plain.sent.len());
        for (k, (a, b)) in streamed.sent.iter().zip(&plain.sent).enumerate() {
            let (ta, sa, ma) = prefix_parts(a);
            let (tb, sb, mb) = prefix_parts(b);
            assert_eq!((ta, sa, ma), (tb, sb, mb), "{path:?} call {k}");
        }
    }
}

mod invariant_props {
    use super::*;
    use proptest::prelude::*;

    /// One generated search step: per call, how many results it gets.
    #[derive(Debug, Clone)]
    struct GenStep {
        think: bool,
        text: bool,
        calls: Vec<usize>,
    }

    fn gen_step() -> impl Strategy<Value = GenStep> {
        (
            any::<bool>(),
            any::<bool>(),
            prop::collection::vec(0usize..=3, 1..=3),
        )
            .prop_map(|(think, text, calls)| GenStep { think, text, calls })
    }

    fn build(steps: &[GenStep], final_think: bool) -> (Vec<Value>, Vec<Reply>) {
        let mut outputs = Vec::new();
        let mut replies = Vec::new();
        let mut call = 0;
        for (s, step) in steps.iter().enumerate() {
            let mut content = Vec::new();
            if step.think {
                content.push(think(&format!("step {s}"), &format!("sig-{s}")));
            }
            if step.text {
                content.push(say(&format!("Looking ({s}).")));
            }
            for n in &step.calls {
                call += 1;
                content.push(search_call(
                    &format!("toolu_{call}"),
                    &format!("query {call}"),
                ));
                let items: Vec<(String, String, String)> = (0..*n)
                    .map(|r| {
                        (
                            format!("T{call}.{r}"),
                            format!("https://r{call}.example/{r}"),
                            format!("text {call}.{r}"),
                        )
                    })
                    .collect();
                let refs: Vec<(&str, &str, &str)> = items
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                replies.push(results(&refs));
            }
            outputs.push(output(content, "tool_use", (100, 10)));
        }
        let mut answer = Vec::new();
        if final_think {
            answer.push(think("final", "sig-final"));
        }
        answer.push(say("Answer [r](https://r1.example/0)."));
        outputs.push(output(answer, "end_turn", (100, 10)));
        (outputs, replies)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Random step sequences: 1–4 search steps of 1–3 parallel calls,
        /// thinking and text present or not, 0–3 results per call, then the
        /// answer. On both lanes every continuation is a prefix of the
        /// replay's rebuild.
        #[test]
        fn invariant_random_steps(
            steps in prop::collection::vec(gen_step(), 1..=4),
            final_think in any::<bool>(),
            tool_search in any::<bool>(),
        ) {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let (outputs, replies) = build(&steps, final_think);
            let path = if tool_search { Path::Invoke } else { Path::Converse };
            runtime.block_on(check_invariant(
                path,
                normalized(request_1(tool_search), WebSearchHeader::On),
                &outputs,
                replies,
                vec![user("next")],
                WebSearchHeader::Off,
                "random steps",
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Scenarios (§11.4)
// ---------------------------------------------------------------------------

fn answer(text: &str) -> Value {
    output(
        vec![think("Answer now.", "sig-answer"), say(text)],
        "end_turn",
        (300, 50),
    )
}

fn item_ids(run: &Run) -> Vec<String> {
    normalized_output(run)
        .iter()
        .map(|item| item["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn parallel_searches_run_in_block_order() {
    for path in Path::ALL {
        let search = FakeSearch::new(vec![
            results(&[("A", "https://a.example/", "a")]),
            results(&[("B", "https://b.example/", "b")]),
        ]);
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            fixture_json(fixture!("parallel-converse-1.json")),
            answer("Both: [A](https://a.example/) and [B](https://b.example/)."),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(
            search.queries(),
            vec!["AgentCore Web Search launch", "Bedrock news this week"],
            "{path:?}"
        );
        assert_eq!(
            item_ids(&run),
            vec![
                "rs_S",
                "ws_bgw_tooluse_ws_a",
                "ws_bgw_tooluse_ws_b",
                "rs_S_ws0",
                "rs_S_1",
                "msg_S_1"
            ],
            "{path:?}"
        );
        let carrier = &carriers(&run.output())[0];
        assert_eq!(
            carrier["tool_uses"][0]["toolUse"]["toolUseId"],
            "tooluse_ws_a"
        );
        assert_eq!(
            carrier["tool_uses"][1]["toolUse"]["toolUseId"],
            "tooluse_ws_b"
        );
        assert_eq!(
            carrier["tool_results"][0]["toolResult"]["toolUseId"],
            "tooluse_ws_a"
        );
        assert_eq!(carrier["used"], 2);
        let message = run.output().last().unwrap().clone();
        assert_eq!(
            message["content"][0]["annotations"]
                .as_array()
                .unwrap()
                .len(),
            2,
            "{path:?}: both links point at results"
        );
    }
}

#[tokio::test]
async fn mixed_step_hands_back_client_calls_carrier_last() {
    for path in Path::ALL {
        let search = FakeSearch::new(vec![results(&[("Q", "https://q.example/", "quota")])]);
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [fixture_json(fixture!("mixed-converse-1.json"))];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(
            run.sent.len(),
            1,
            "{path:?}: a mixed step ends the response"
        );
        assert_eq!(run.response().status, "completed");
        assert_eq!(
            run.types(),
            vec!["reasoning", "web_search_call", "function_call", "reasoning"],
            "{path:?}"
        );
        let output = run.output();
        assert_eq!(output[2]["call_id"], "tooluse_fn_1");
        assert_eq!(output[2]["name"], "lookup_ticket");
        assert_eq!(
            serde_json::from_str::<Value>(output[2]["arguments"].as_str().unwrap()).unwrap(),
            json!({ "id": "T-1042" })
        );
        let carrier = &carriers(&output)[0];
        assert_eq!(carrier["tool_uses"].as_array().unwrap().len(), 1);
        assert_eq!(
            carrier["tool_uses"][0]["toolUse"]["toolUseId"],
            "tooluse_ws_m"
        );
        assert_eq!(carrier["used"], 1);
        assert_eq!(search.queries(), vec!["AgentCore web search quota"]);
        if path.streams() {
            let types = run.event_types();
            let fc_added = types
                .iter()
                .position(|t| t == "response.function_call_arguments.done")
                .unwrap();
            let ws_done = run
                .events
                .iter()
                .position(|e| {
                    e["type"] == "response.output_item.done"
                        && e["item"]["type"] == "web_search_call"
                })
                .unwrap();
            assert!(
                ws_done < fc_added,
                "{path:?}: searches first, then client calls"
            );
            assert!(
                !types
                    .iter()
                    .any(|t| t == "response.function_call_arguments.delta"),
                "{path:?}: client calls are not streamed in a declared request"
            );
        }
    }
}

/// The client's continuation after a mixed step replays deterministically,
/// with the user message's results in the assistant message's call order.
#[tokio::test]
async fn mixed_replay_is_deterministic_and_aligned() {
    for path in [Path::Converse, Path::Invoke] {
        let search = FakeSearch::new(vec![results(&[("Q", "https://q.example/", "quota")])]);
        let provider = provider(&search);
        let original = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [fixture_json(fixture!("mixed-converse-1.json"))];
        let run = run_on(path, &provider, &original, &steps).await.unwrap();
        let next = normalized(
            next_request(
                &original.request,
                &run,
                vec![
                    json!({ "type": "function_call_output", "call_id": "tooluse_fn_1",
                             "output": "T-1042: open" }),
                ],
            ),
            WebSearchHeader::On,
        );
        let first = run_on(path, &provider, &next, &[answer("Done.")])
            .await
            .unwrap();
        let second = run_on(path, &provider, &next, &[answer("Done.")])
            .await
            .unwrap();
        assert_eq!(first.sent[0], second.sent[0], "{path:?}");
        let (_, _, messages) = prefix_parts(&first.sent[0]);
        let messages = strip_cache(&messages);
        let n = messages.len();
        let ids = |message: &Value, key: &str, id: &str| -> Vec<String> {
            message["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|block| {
                    block
                        .pointer(&format!("/{key}/{id}"))
                        .or_else(|| block.get(id))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect()
        };
        let (use_key, use_id, result_key, result_id) = if path.tool_search() {
            ("", "id", "", "tool_use_id")
        } else {
            ("toolUse", "toolUseId", "toolResult", "toolUseId")
        };
        let calls = ids(&messages[n - 2], use_key, use_id);
        let answered = ids(&messages[n - 1], result_key, result_id);
        assert_eq!(calls, vec!["tooluse_fn_1", "tooluse_ws_m"], "{path:?}");
        assert_eq!(answered, calls, "{path:?}: results follow the call order");
    }
}

/// A history with marks and the header `off`: the tool stays declared, a
/// call gets `disabled_for_this_message`, nothing is searched and no
/// `web_search_call` item is emitted.
#[tokio::test]
async fn off_with_marks_answers_disabled_without_search_or_item() {
    for path in Path::ALL {
        let search = FakeSearch::new(three_results());
        let provider = provider(&search);
        let first = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let earlier = run_on(path, &provider, &first, &golden_steps())
            .await
            .unwrap();
        let before = search.queries().len();

        let next = normalized(
            next_request(&first.request, &earlier, vec![user("Search again?")]),
            WebSearchHeader::Off,
        );
        let plan = provider.web_search_plan(&next);
        assert!(plan.declared);
        assert_eq!(plan.gate, ExecGate::Disabled);
        let steps = [
            output(vec![search_call("t9", "again")], "tool_use", (10, 5)),
            answer("Web search is off for this message."),
        ];
        let run = run_on(path, &provider, &next, &steps).await.unwrap();
        assert_eq!(search.queries().len(), before, "{path:?}: no search ran");
        assert!(
            !run.types().contains(&"web_search_call".to_string()),
            "{path:?}"
        );
        let carrier = &carriers(&run.output())[0];
        assert_eq!(
            carrier_results(carrier),
            vec![
                json!({ "error_code": "disabled_for_this_message", "query": "again",
                         "type": "web_search_error" })
            ]
        );
        assert_eq!(carrier["used"], 0, "{path:?}: a new answer");
        // The tool is still declared on the next call.
        let (tools, _, _) = prefix_parts(&run.sent[1]);
        assert!(tools
            .to_string()
            .contains(crate::bedrock::web_search::WEB_SEARCH_TOOL_DESCRIPTION));
    }
}

#[tokio::test]
async fn backend_off_with_marks_answers_unavailable() {
    let search = FakeSearch::new(three_results());
    let with_backend = provider(&search);
    let first = normalized(request_1(false), WebSearchHeader::On);
    let earlier = run_on(Path::Converse, &with_backend, &first, &golden_steps())
        .await
        .unwrap();

    let without_backend = provider_with(None, limits(), None);
    let next = normalized(
        next_request(&first.request, &earlier, vec![user("More?")]),
        WebSearchHeader::On,
    );
    let plan = without_backend.web_search_plan(&next);
    assert!(plan.declared, "marks declare the tool without a backend");
    assert_eq!(plan.gate, ExecGate::Unavailable);
    let steps = [
        output(vec![search_call("t9", "more")], "tool_use", (10, 5)),
        answer("Search is unavailable."),
    ];
    let run = run_on(Path::Converse, &without_backend, &next, &steps)
        .await
        .unwrap();
    let carrier = &carriers(&run.output())[0];
    assert_eq!(
        carrier_results(carrier),
        vec![json!({ "error_code": "unavailable", "query": "more", "type": "web_search_error" })]
    );
    assert!(!run.types().contains(&"web_search_call".to_string()));
}

/// The first request that declares the tool drops the thinking of earlier
/// turns (signed under a tool list without it) and keeps their text.
#[tokio::test]
async fn first_declaration_drops_unmarked_thinking() {
    let unmarked = crate::bedrock::responses_translate::encode_reasoning_envelope(&[think(
        "Earlier thinking.",
        "sig-old",
    )])
    .unwrap();
    let mut input = input_of(&request_1(false));
    input.push(json!({ "type": "reasoning", "id": "rs_old", "summary": [],
                       "encrypted_content": unmarked }));
    input.push(json!({ "type": "message", "role": "assistant",
                       "content": [{ "type": "output_text", "text": "Earlier answer." }] }));
    input.push(user("Now search the web."));
    for path in [Path::Converse, Path::Invoke] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(
            with_input(request_1(path.tool_search()), input.clone()),
            WebSearchHeader::On,
        );
        let run = run_on(path, &provider, &req, &[answer("Here.")])
            .await
            .unwrap();
        let (_, _, messages) = prefix_parts(&run.sent[0]);
        let text = messages.to_string();
        assert!(
            !text.contains("sig-old"),
            "{path:?}: unmarked thinking dropped"
        );
        assert!(text.contains("Earlier answer."), "{path:?}: its text stays");
    }
}

#[tokio::test]
async fn max_calls_then_step_cap() {
    for path in [Path::Converse, Path::ConverseStream] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(false), WebSearchHeader::On);
        // The model calls web_search on every step.
        let steps: Vec<Value> = (0..7)
            .map(|i| {
                output(
                    vec![search_call(&format!("t{i}"), &format!("q{i}"))],
                    "tool_use",
                    (10, 5),
                )
            })
            .collect();
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.sent.len(), 7, "{path:?}: max_calls + 2 Bedrock calls");
        assert_eq!(
            search.queries(),
            vec!["q0", "q1", "q2", "q3", "q4"],
            "{path:?}"
        );
        let output = run.output();
        let searches = output
            .iter()
            .filter(|i| i["type"] == "web_search_call")
            .count();
        assert_eq!(searches, 5, "{path:?}: refused calls have no item");
        let carriers = carriers(&output);
        assert_eq!(carriers.len(), 7);
        assert_eq!(
            carrier_results(&carriers[5])[0]["error_code"],
            "max_uses_exceeded",
            "{path:?}: the sixth call"
        );
        assert_eq!(
            carrier_results(&carriers[6])[0]["error_code"],
            "max_uses_exceeded"
        );
        assert_eq!(carriers[6]["used"], 5);
        assert_eq!(run.response().status, "completed");
        let last = output.last().unwrap();
        assert_eq!(last["type"], "message");
        assert!(last["id"].as_str().unwrap().ends_with("_stop"));
        assert_eq!(
            last["content"][0]["text"],
            "The answer stopped after 5 web searches: the search limit for one answer was reached."
        );
    }
}

#[tokio::test]
async fn search_errors_become_failed_items_and_error_payloads() {
    for path in [Path::Converse, Path::InvokeStream] {
        let search = FakeSearch::new(vec![
            Reply::Fail(SearchError::Throttled),
            Reply::Fail(SearchError::Unavailable),
            Reply::Fail(SearchError::InvalidInput),
        ]);
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(
                vec![
                    search_call("t1", "one"),
                    search_call("t2", "two"),
                    search_call("t3", "three"),
                ],
                "tool_use",
                (10, 5),
            ),
            answer("The searches failed."),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        let output = run.output();
        let calls: Vec<&Value> = output
            .iter()
            .filter(|i| i["type"] == "web_search_call")
            .collect();
        assert_eq!(
            calls.len(),
            3,
            "{path:?}: real searches, even failed ones, have items"
        );
        for call in &calls {
            assert_eq!(call["status"], "failed");
            assert!(call["action"].get("sources").is_none());
            assert_eq!(call["action"]["queries"][0], call["action"]["query"]);
        }
        let carrier = &carriers(&output)[0];
        assert_eq!(
            carrier_results(carrier),
            vec![
                json!({ "error_code": "too_many_requests", "query": "one", "type": "web_search_error" }),
                json!({ "error_code": "unavailable", "query": "two", "type": "web_search_error" }),
                json!({ "error_code": "invalid_tool_input", "query": "three", "type": "web_search_error" }),
            ]
        );
        assert_eq!(carrier["used"], 3, "{path:?}: failed searches count");
        if path.streams() {
            let completed = run
                .events
                .iter()
                .filter(|e| e["type"] == "response.web_search_call.completed")
                .count();
            assert_eq!(
                completed, 0,
                "{path:?}: no completed event for a failed search"
            );
        }
    }
}

#[tokio::test]
async fn invalid_query_is_invalid_tool_input_without_item() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let req = normalized(request_1(false), WebSearchHeader::On);
    let steps = [
        output(
            vec![
                tool_call("t1", "web_search", json!({ "query": "  \u{200b} " })),
                tool_call("t2", "web_search", json!({})),
                tool_call("t3", "web_search", json!({ "query": 5 })),
                tool_call("t4", "web_search", json!("not an object")),
            ],
            "tool_use",
            (10, 5),
        ),
        answer("I could not search."),
    ];
    let run = run_on(Path::Converse, &provider, &req, &steps)
        .await
        .unwrap();
    assert!(search.queries().is_empty());
    assert!(!run.types().contains(&"web_search_call".to_string()));
    let carrier = &carriers(&run.output())[0];
    for result in carrier_results(carrier) {
        assert_eq!(
            result,
            json!({ "error_code": "invalid_tool_input", "type": "web_search_error" })
        );
    }
    assert_eq!(carrier["used"], 0);
    // A non-object input is stored as `{}`, which the carrier can hold.
    assert_eq!(carrier["tool_uses"][3]["toolUse"]["input"], json!({}));
}

// ---------------------------------------------------------------------------
// Terminal states (§5.2, §5.3) and limits
// ---------------------------------------------------------------------------

fn stop_text(reason: Exhaustion, n: u32) -> String {
    reason.message(n)
}

/// The gateway's message is the response's last item, and in a stream its
/// events are the six of §5.3 just before `response.completed`.
fn assert_gateway_stop(run: &Run, path: Path, expected_text: &str) {
    let response = run.response();
    assert_eq!(response.status, "completed", "{path:?}");
    assert!(response.incomplete_details.is_none(), "{path:?}");
    let output = run.output();
    let last = output.last().unwrap();
    assert_eq!(last["type"], "message", "{path:?}");
    let s = response.id.trim_start_matches("resp_");
    assert_eq!(last["id"], format!("msg_{s}_stop"), "{path:?}");
    assert_eq!(last["content"][0]["text"], expected_text, "{path:?}");
    assert_eq!(last["content"][0]["annotations"], json!([]));
    if path.streams() {
        let types = run.event_types();
        assert!(
            !types.iter().any(|t| t == "response.incomplete"),
            "{path:?}"
        );
        let tail: Vec<&str> = types[types.len() - 7..]
            .iter()
            .map(String::as_str)
            .collect();
        assert_eq!(
            tail,
            vec![
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
            "{path:?}"
        );
        let delta = &run.events[run.events.len() - 5];
        assert_eq!(delta["delta"], expected_text);
        assert_eq!(delta["item_id"], format!("msg_{s}_stop"));
    }
}

#[tokio::test]
async fn max_tokens_stop_after_search_ends_completed_with_gateway_message() {
    for path in Path::ALL {
        let search = FakeSearch::new(vec![results(&[("A", "https://a.example/", "a")])]);
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(
                vec![think("s", "sig-s"), search_call("t1", "q")],
                "tool_use",
                (10, 5),
            ),
            output(
                vec![think("a", "sig-a2"), say("A partial ans")],
                "max_tokens",
                (10, 5),
            ),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        let output = run.output();
        // The step's reasoning and text are still output, before the message.
        assert_eq!(
            output[output.len() - 2]["content"][0]["text"],
            "A partial ans"
        );
        assert_gateway_stop(&run, path, &stop_text(Exhaustion::OutputBudget, 1));
    }
}

/// A continuation request whose answer already searched (`used_in_answer`
/// from the replayed carrier) ends the same way, though it searches nothing
/// itself.
#[tokio::test]
async fn max_tokens_stop_in_a_continuation_after_search_ends_completed() {
    for path in [Path::Converse, Path::ConverseStream] {
        let search = FakeSearch::new(vec![
            results(&[("A", "https://a.example/", "a")]),
            results(&[("B", "https://b.example/", "b")]),
        ]);
        let provider = provider(&search);
        let first = normalized(request_1(false), WebSearchHeader::On);
        let mixed = output(
            vec![
                think("both", "sig-m"),
                tool_call("fn1", "lookup_ticket", json!({ "id": "T-1" })),
                search_call("w1", "one"),
                search_call("w2", "two"),
            ],
            "tool_use",
            (10, 5),
        );
        let earlier = run_on(Path::Converse, &provider, &first, &[mixed])
            .await
            .unwrap();
        let next = normalized(
            next_request(
                &first.request,
                &earlier,
                vec![json!({ "type": "function_call_output", "call_id": "fn1", "output": "open" })],
            ),
            WebSearchHeader::On,
        );
        assert_eq!(provider.web_search_plan(&next).used_in_answer, 2);
        let steps = [output(vec![say("Cut off")], "max_tokens", (10, 5))];
        let run = run_on(path, &provider, &next, &steps).await.unwrap();
        assert_gateway_stop(&run, path, &stop_text(Exhaustion::OutputBudget, 2));
    }
}

#[tokio::test]
async fn max_tokens_stop_without_search_stays_incomplete() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [output(vec![say("Cut")], "max_tokens", (10, 5))];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        let response = run.response();
        assert_eq!(response.status, "incomplete", "{path:?}");
        assert_eq!(
            response.incomplete_details,
            Some(json!({ "reason": "max_output_tokens" }))
        );
        assert!(!run.types().iter().any(|t| t == "web_search_call"));
        assert!(!run
            .output()
            .iter()
            .any(|i| i["id"].as_str().unwrap().ends_with("_stop")));
        if path.streams() {
            assert_eq!(run.event_types().last().unwrap(), "response.incomplete");
        }
    }
}

#[tokio::test]
async fn remaining_budget_is_passed() {
    for path in [Path::Converse, Path::Invoke] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(vec![search_call("t1", "q")], "tool_use", (10, 95)),
            output(vec![search_call("t2", "q2")], "tool_use", (10, 905)),
            answer("Done."),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        let max_tokens: Vec<i64> = run
            .sent
            .iter()
            .map(|sent| {
                sent.body
                    .pointer("/inference_config/maxTokens")
                    .or_else(|| sent.body.get("max_tokens"))
                    .and_then(Value::as_i64)
                    .unwrap()
            })
            .collect();
        assert_eq!(max_tokens, vec![32000, 31905, 31000], "{path:?}");
        // The echo is the request's own value.
        assert_eq!(run.response().max_output_tokens, Some(32000));
    }
}

#[tokio::test]
async fn budget_exhausted_after_search_ends_completed_with_gateway_message() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let mut request = request_1(path.tool_search());
        request.max_output_tokens = Some(1500);
        let req = normalized(request, WebSearchHeader::On);
        let steps = [output(
            vec![
                think("s", "sig-s"),
                say("Searching"),
                search_call("t1", "q"),
            ],
            "tool_use",
            (10, 600),
        )];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.sent.len(), 1, "{path:?}: 900 tokens left is too little");
        assert_eq!(search.queries(), vec!["q"]);
        let types = run.types();
        assert_eq!(
            types,
            vec![
                "reasoning",
                "message",
                "web_search_call",
                "reasoning",
                "message"
            ],
            "{path:?}: the carrier precedes the gateway's message"
        );
        assert_gateway_stop(&run, path, &stop_text(Exhaustion::OutputBudget, 1));
    }
}

#[tokio::test]
async fn step_limit_ends_completed_with_gateway_message() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let limits = WebSearchLimits {
            max_calls: 1,
            ..limits()
        };
        let provider = provider_with(
            Some(search.clone() as Arc<dyn WebSearchBackend>),
            limits,
            None,
        );
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps: Vec<Value> = (0..3)
            .map(|i| {
                output(
                    vec![search_call(&format!("t{i}"), &format!("q{i}"))],
                    "tool_use",
                    (10, 5),
                )
            })
            .collect();
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.sent.len(), 3, "{path:?}: max_calls + 2");
        assert_eq!(search.queries(), vec!["q0"]);
        let carriers = carriers(&run.output());
        assert_eq!(carriers.len(), 3);
        assert_eq!(
            carrier_results(&carriers[2])[0]["error_code"],
            "max_uses_exceeded"
        );
        assert_gateway_stop(&run, path, &stop_text(Exhaustion::StepLimit, 1));
    }
}

#[test]
fn gateway_stop_texts() {
    assert_eq!(
        Exhaustion::StepLimit.message(1),
        "The answer stopped after 1 web search: the search limit for one answer was reached."
    );
    assert_eq!(
        Exhaustion::OutputBudget.message(3),
        "The answer stopped after 3 web searches: the output token budget ran out."
    );
    assert_eq!(
        Exhaustion::ContentFilter.message(2),
        "The answer stopped after 2 web searches: the content filter stopped the answer."
    );
    assert_eq!(
        Exhaustion::NoAnswer.message(1),
        "The answer stopped after 1 web search: the model gave no answer."
    );
    assert_eq!(
        no_search_message(ExecGate::Disabled),
        "The answer stopped: web search is off for this message."
    );
    assert_eq!(
        no_search_message(ExecGate::Unavailable),
        "The answer stopped: web search is unavailable."
    );
    assert_eq!(
        no_search_message(ExecGate::Search),
        "The answer stopped: the step limit was reached before any web search ran."
    );
}

/// Without a real search in the answer a spent budget keeps the single-call
/// meaning: `incomplete` with `max_output_tokens`, no gateway message.
#[tokio::test]
async fn budget_exhausted_without_search_stays_incomplete() {
    for path in [Path::Converse, Path::ConverseStream] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let mut request = request_1(false);
        request.max_output_tokens = Some(1500);
        // An invalid query: the call is answered, but nothing is searched.
        let req = normalized(request, WebSearchHeader::On);
        let steps = [output(
            vec![tool_call("t1", "web_search", json!({ "query": "" }))],
            "tool_use",
            (10, 600),
        )];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert!(search.queries().is_empty());
        let response = run.response();
        assert_eq!(response.status, "incomplete", "{path:?}");
        assert_eq!(
            response.incomplete_details,
            Some(json!({ "reason": "max_output_tokens" }))
        );
        assert_eq!(run.types(), vec!["reasoning"], "{path:?}: only the carrier");
        if path.streams() {
            assert_eq!(run.event_types().last().unwrap(), "response.incomplete");
        }
    }
}

#[tokio::test]
async fn required_becomes_auto_after_first_search() {
    for path in [Path::Converse, Path::Invoke] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let mut request = request_1(path.tool_search());
        request.tool_choice = Some(ResponsesToolChoice::String("required".to_string()));
        request.reasoning = None;
        let req = normalized(request, WebSearchHeader::On);
        let steps = [
            output(vec![search_call("t1", "q")], "tool_use", (10, 5)),
            output(vec![say("Done.")], "end_turn", (10, 5)),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        let choice = |sent: &Sent| {
            sent.body
                .pointer("/tool_config/toolChoice")
                .or_else(|| sent.body.get("tool_choice"))
                .cloned()
                .unwrap()
        };
        let (first, second) = (choice(&run.sent[0]), choice(&run.sent[1]));
        if path.tool_search() {
            assert_eq!(
                (first, second),
                (json!({ "type": "any" }), json!({ "type": "auto" }))
            );
        } else {
            assert_eq!(
                (first, second),
                (json!({ "any": {} }), json!({ "auto": {} }))
            );
        }
        assert!(matches!(
            &run.response().tool_choice,
            Some(ResponsesToolChoice::String(choice)) if choice == "required"
        ));
    }
}

/// A call cut off by `max_tokens` is neither run nor output.
#[tokio::test]
async fn truncated_search_call_is_not_executed() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [output(
            vec![
                think("s", "sig-s"),
                say("Let me"),
                search_call("t1", "half"),
            ],
            "max_tokens",
            (10, 5),
        )];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert!(search.queries().is_empty(), "{path:?}");
        assert_eq!(run.types(), vec!["reasoning", "message"], "{path:?}");
        assert_eq!(run.response().status, "incomplete");
    }
}

// ---------------------------------------------------------------------------
// The answer's search count across requests (§5.3)
// ---------------------------------------------------------------------------

/// Three searches, then a mixed step: the carrier says `used: 3`; the client's
/// continuation may run two more, and the sixth call is refused.
#[tokio::test]
async fn used_accumulates_across_a_mixed_step_continuation() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let first = normalized(request_1(false), WebSearchHeader::On);
    let steps = [
        output(
            vec![search_call("a1", "one"), search_call("a2", "two")],
            "tool_use",
            (10, 5),
        ),
        output(
            vec![
                tool_call("fn1", "lookup_ticket", json!({ "id": "T-9" })),
                search_call("a3", "three"),
            ],
            "tool_use",
            (10, 5),
        ),
    ];
    let earlier = run_on(Path::Converse, &provider, &first, &steps)
        .await
        .unwrap();
    let earlier_carriers = carriers(&earlier.output());
    assert_eq!(earlier_carriers.last().unwrap()["used"], 3);

    let next = normalized(
        next_request(
            &first.request,
            &earlier,
            vec![json!({ "type": "function_call_output", "call_id": "fn1", "output": "open" })],
        ),
        WebSearchHeader::On,
    );
    assert_eq!(provider.web_search_plan(&next).used_in_answer, 3);
    let steps = [
        output(
            vec![
                search_call("b1", "four"),
                search_call("b2", "five"),
                search_call("b3", "six"),
            ],
            "tool_use",
            (10, 5),
        ),
        answer("Done."),
    ];
    let run = run_on(Path::Converse, &provider, &next, &steps)
        .await
        .unwrap();
    assert_eq!(
        search.queries(),
        vec!["one", "two", "three", "four", "five"],
        "five real searches in the answer"
    );
    let carrier = &carriers(&run.output())[0];
    assert_eq!(carrier["used"], 5);
    assert_eq!(
        carrier_results(carrier)[2]["error_code"],
        "max_uses_exceeded"
    );
}

#[tokio::test]
async fn a_new_user_message_starts_a_new_answer() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let first = normalized(request_1(false), WebSearchHeader::On);
    let steps: Vec<Value> = vec![
        output(
            (0..5)
                .map(|i| search_call(&format!("a{i}"), &format!("q{i}")))
                .collect(),
            "tool_use",
            (10, 5),
        ),
        answer("Five searches."),
    ];
    let earlier = run_on(Path::Converse, &provider, &first, &steps)
        .await
        .unwrap();
    assert_eq!(carriers(&earlier.output())[0]["used"], 5);

    let next = normalized(
        next_request(&first.request, &earlier, vec![user("A new question")]),
        WebSearchHeader::On,
    );
    assert_eq!(provider.web_search_plan(&next).used_in_answer, 0);
    let run = run_on(
        Path::Converse,
        &provider,
        &next,
        &[
            output(vec![search_call("n1", "fresh")], "tool_use", (10, 5)),
            answer("ok"),
        ],
    )
    .await
    .unwrap();
    assert_eq!(search.queries().last().unwrap(), "fresh");
    assert_eq!(carriers(&run.output())[0]["used"], 1);
}

// ---------------------------------------------------------------------------
// Ids (§6.1)
// ---------------------------------------------------------------------------

fn assert_unique_ids(run: &Run, label: &str) {
    let ids: Vec<String> = run
        .output()
        .iter()
        .map(|item| item["id"].as_str().unwrap().to_string())
        .collect();
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "{label}: {ids:?}");
    if !run.events.is_empty() {
        // Every added item is done with the same id and output_index; the
        // done ids are unique.
        let mut open: Option<(String, u64)> = None;
        let mut done = Vec::new();
        for event in &run.events {
            match event["type"].as_str().unwrap() {
                "response.output_item.added" => {
                    assert!(open.is_none(), "{label}: two items open at once");
                    open = Some((
                        event["item"]["id"].as_str().unwrap().to_string(),
                        event["output_index"].as_u64().unwrap(),
                    ));
                }
                "response.output_item.done" => {
                    let id = event["item"]["id"].as_str().unwrap().to_string();
                    let index = event["output_index"].as_u64().unwrap();
                    if let Some((open_id, open_index)) = open.take() {
                        assert_eq!((open_id, open_index), (id.clone(), index), "{label}");
                    }
                    done.push(id);
                }
                _ => {}
            }
        }
        assert!(open.is_none(), "{label}: an item never finished");
        assert_eq!(done, ids, "{label}: done items are the output");
    }
}

#[tokio::test]
async fn every_output_item_id_is_unique() {
    for path in Path::ALL {
        let search = FakeSearch::new(three_results());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let run = run_on(path, &provider, &req, &two_search_steps())
            .await
            .unwrap();
        assert_unique_ids(&run, &format!("{path:?} multi-step"));
        assert_eq!(
            item_ids(&run),
            vec![
                "rs_S",
                "msg_S",
                "ws_bgw_t1",
                "rs_S_ws0",
                "rs_S_1",
                "ws_bgw_t2",
                "ws_bgw_t3",
                "rs_S_ws1",
                "rs_S_2",
                "msg_S_2"
            ],
            "{path:?}"
        );

        let search = FakeSearch::new(Vec::new());
        let provider = self::provider(&search);
        let run = run_on(
            path,
            &provider,
            &req,
            &[fixture_json(fixture!("mixed-converse-1.json"))],
        )
        .await
        .unwrap();
        assert_unique_ids(&run, &format!("{path:?} mixed"));

        let limits = WebSearchLimits {
            max_calls: 1,
            ..limits()
        };
        let provider = provider_with(
            Some(FakeSearch::new(Vec::new()) as Arc<dyn WebSearchBackend>),
            limits,
            None,
        );
        let steps: Vec<Value> = (0..3)
            .map(|i| {
                output(
                    vec![
                        think("t", &format!("s{i}")),
                        say("x"),
                        search_call(&format!("t{i}"), "q"),
                    ],
                    "tool_use",
                    (10, 5),
                )
            })
            .collect();
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_unique_ids(&run, &format!("{path:?} step limit"));

        let mut request = request_1(path.tool_search());
        request.max_output_tokens = Some(1100);
        let budget_req = normalized(request, WebSearchHeader::On);
        let provider = self::provider(&FakeSearch::new(Vec::new()));
        let run = run_on(
            path,
            &provider,
            &budget_req,
            &[output(
                vec![say("x"), search_call("t1", "q")],
                "tool_use",
                (10, 100),
            )],
        )
        .await
        .unwrap();
        assert_unique_ids(&run, &format!("{path:?} output budget"));
    }
}

// ---------------------------------------------------------------------------
// Stream events (§6.4) and errors (§5.6)
// ---------------------------------------------------------------------------

/// The §6.4 outline, event by event (`(type, item type or "")`).
fn outline(run: &Run) -> Vec<(String, String)> {
    run.events
        .iter()
        .map(|e| {
            let item = e["item"]["type"].as_str().unwrap_or_default().to_string();
            (e["type"].as_str().unwrap().to_string(), item)
        })
        .collect()
}

fn ev(kind: &str, item: &str) -> (String, String) {
    (kind.to_string(), item.to_string())
}

/// Shared stream checks: sequence numbers from 0 without gaps, output
/// indexes in emit order, one created / in_progress / terminal event, no text
/// delta while a web_search_call is open, and the terminal output equal to
/// the done items.
fn assert_stream_shape(run: &Run, label: &str) {
    for (i, event) in run.events.iter().enumerate() {
        assert_eq!(
            event["sequence_number"], i as u64,
            "{label}: sequence_number"
        );
    }
    let types = run.event_types();
    for once in ["response.created", "response.in_progress"] {
        assert_eq!(
            types.iter().filter(|t| *t == once).count(),
            1,
            "{label}: {once}"
        );
    }
    assert_eq!(types[0], "response.created");
    assert_eq!(types[1], "response.in_progress");
    let terminal = types
        .iter()
        .filter(|t| {
            matches!(
                t.as_str(),
                "response.completed" | "response.incomplete" | "response.failed"
            )
        })
        .count();
    assert_eq!(terminal, 1, "{label}: one terminal event");
    let mut next_index = 0;
    let mut open_search = false;
    for event in &run.events {
        match event["type"].as_str().unwrap() {
            "response.output_item.added" => {
                assert_eq!(event["output_index"], next_index, "{label}: output_index");
                next_index += 1;
                open_search = event["item"]["type"] == "web_search_call";
            }
            "response.output_item.done" => {
                if event["item"]["type"] == "web_search_call" {
                    open_search = false;
                }
            }
            "response.output_text.delta" | "response.reasoning_text.delta" => {
                assert!(!open_search, "{label}: a delta while a search is open");
            }
            _ => {}
        }
    }
    assert_unique_ids(run, label);
}

#[tokio::test]
async fn stream_event_order_single_search() {
    let (run, _) = golden_run(Path::ConverseStream).await;
    assert_stream_shape(&run, "single search");
    let reasoning = || {
        vec![
            ev("response.output_item.added", "reasoning"),
            ev("response.reasoning_summary_part.added", ""),
            ev("response.reasoning_text.delta", ""),
            ev("response.reasoning_summary_text.delta", ""),
            ev("response.reasoning_text.delta", ""),
            ev("response.reasoning_summary_text.delta", ""),
            ev("response.reasoning_text.done", ""),
            ev("response.reasoning_summary_text.done", ""),
            ev("response.reasoning_summary_part.done", ""),
            ev("response.output_item.done", "reasoning"),
        ]
    };
    let mut expected = vec![ev("response.created", ""), ev("response.in_progress", "")];
    expected.extend(reasoning());
    expected.extend([
        ev("response.output_item.added", "message"),
        ev("response.content_part.added", ""),
        ev("response.output_text.delta", ""),
        ev("response.output_text.delta", ""),
        ev("response.output_text.done", ""),
        ev("response.content_part.done", ""),
        ev("response.output_item.done", "message"),
        ev("response.output_item.added", "web_search_call"),
        ev("response.web_search_call.in_progress", ""),
        ev("response.web_search_call.searching", ""),
        ev("response.web_search_call.completed", ""),
        ev("response.output_item.done", "web_search_call"),
        ev("response.output_item.added", "reasoning"),
        ev("response.output_item.done", "reasoning"),
    ]);
    expected.extend(reasoning());
    expected.extend([
        ev("response.output_item.added", "message"),
        ev("response.content_part.added", ""),
        ev("response.output_text.delta", ""),
        ev("response.output_text.delta", ""),
        ev("response.output_text.annotation.added", ""),
        ev("response.output_text.done", ""),
        ev("response.content_part.done", ""),
        ev("response.output_item.done", "message"),
        ev("response.completed", ""),
    ]);
    assert_eq!(outline(&run), expected);

    let event = |kind: &str| {
        run.events
            .iter()
            .find(|e| e["type"] == kind)
            .unwrap()
            .clone()
    };
    // The search item announces its query when it starts.
    let added = run
        .events
        .iter()
        .find(|e| {
            e["type"] == "response.output_item.added" && e["item"]["type"] == "web_search_call"
        })
        .unwrap();
    assert_eq!(
        added["item"],
        json!({ "type": "web_search_call", "id": "ws_bgw_tooluse_ws_1", "status": "in_progress",
                "action": { "type": "search", "query": "AWS announcements November 2026" } })
    );
    let progress = event("response.web_search_call.searching");
    assert_eq!(progress["item_id"], "ws_bgw_tooluse_ws_1");
    assert_eq!(progress["output_index"], added["output_index"]);
    // The carrier is announced without its envelope and done with it.
    let s = run.response().id.trim_start_matches("resp_").to_string();
    let carrier_events: Vec<&Value> = run
        .events
        .iter()
        .filter(|e| e["item"]["id"] == format!("rs_{s}_ws0"))
        .collect();
    assert_eq!(carrier_events.len(), 2);
    assert_eq!(carrier_events[0]["type"], "response.output_item.added");
    assert_eq!(carrier_events[0]["item"]["summary"], json!([]));
    assert!(carrier_events[0]["item"].get("encrypted_content").is_none());
    assert_eq!(carrier_events[1]["type"], "response.output_item.done");
    assert!(carrier_events[1]["item"]["encrypted_content"]
        .as_str()
        .unwrap()
        .starts_with("bedrock-reasoning-v1:"));
    let annotation = event("response.output_text.annotation.added");
    assert_eq!(annotation["annotation_index"], 0);
    assert_eq!(
        annotation["annotation"]["url"],
        "https://aws.amazon.com/new/"
    );
    let completed = event("response.completed");
    assert_eq!(completed["response"]["status"], "completed");
    assert_eq!(completed["response"]["usage"]["output_tokens"], 95 + 140);
}

#[tokio::test]
async fn stream_event_order_mixed_step() {
    let search = FakeSearch::new(vec![results(&[("Q", "https://q.example/", "quota")])]);
    let provider = provider(&search);
    let req = normalized(request_1(false), WebSearchHeader::On);
    let run = run_on(
        Path::ConverseStream,
        &provider,
        &req,
        &[fixture_json(fixture!("mixed-converse-1.json"))],
    )
    .await
    .unwrap();
    assert_stream_shape(&run, "mixed");
    let tail: Vec<(String, String)> = outline(&run)
        .into_iter()
        .skip_while(|(kind, item)| {
            !(kind == "response.output_item.added" && item == "web_search_call")
        })
        .collect();
    assert_eq!(
        tail,
        vec![
            ev("response.output_item.added", "web_search_call"),
            ev("response.web_search_call.in_progress", ""),
            ev("response.web_search_call.searching", ""),
            ev("response.web_search_call.completed", ""),
            ev("response.output_item.done", "web_search_call"),
            ev("response.output_item.added", "function_call"),
            ev("response.function_call_arguments.done", ""),
            ev("response.output_item.done", "function_call"),
            ev("response.output_item.added", "reasoning"),
            ev("response.output_item.done", "reasoning"),
            ev("response.completed", ""),
        ]
    );
    let function_added = run
        .events
        .iter()
        .find(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call")
        .unwrap();
    assert!(function_added["item"].get("status").is_none());
    assert_eq!(function_added["item"]["name"], "lookup_ticket");
}

#[tokio::test]
async fn stream_first_step_error_is_http_error() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    for (path, step) in [
        (
            Path::ConverseStream,
            Step::ConverseStreamService("ValidationException", "messages: bad request"),
        ),
        (
            Path::InvokeStream,
            Step::Fail(AppError::BadRequest("bad body".to_string())),
        ),
    ] {
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let err = run_steps(path, &provider, &req, vec![step])
            .await
            .err()
            .expect("HTTP error");
        assert!(matches!(err, AppError::BadRequest(_)), "{path:?}: {err:?}");
    }
}

fn failed_code(run: &Run) -> String {
    let failed = run.events.last().unwrap();
    assert_eq!(failed["type"], "response.failed");
    failed["response"]["error"]["code"]
        .as_str()
        .unwrap()
        .to_string()
}

const BINDING_ERROR: &str =
    "messages.2.content.0: The thinking block is bound to a different conversation prefix.";

#[tokio::test]
async fn stream_later_binding_error_is_invalid_prompt() {
    let first = output(
        vec![think("s", "sig-s"), search_call("t1", "q")],
        "tool_use",
        (10, 5),
    );
    for (path, later) in [
        (
            Path::ConverseStream,
            Step::ConverseStreamService("ValidationException", BINDING_ERROR),
        ),
        (
            Path::InvokeStream,
            Step::InvokeStream(vec![json!({ "type": "error", "error": {
                "type": "invalid_request_error",
                "message": "Invalid `signature` in `thinking` block" } })]),
        ),
        (
            Path::InvokeStream,
            Step::Fail(AppError::BadRequest(BINDING_ERROR.to_string())),
        ),
    ] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let run = run_steps(path, &provider, &req, vec![path.step(&first), later])
            .await
            .unwrap();
        assert_eq!(failed_code(&run), "invalid_prompt", "{path:?}");
        assert_eq!(run.sent.len(), 2);
        // The first step's items reached the client before the failure.
        assert!(run
            .events
            .iter()
            .any(|e| e["type"] == "response.output_item.done"
                && e["item"]["type"] == "web_search_call"));
    }
}

#[tokio::test]
async fn stream_later_other_error_uses_responses_error() {
    let first = output(vec![search_call("t1", "q")], "tool_use", (10, 5));
    let mid_stream = {
        let mut events = converse_stream_events(&answer("never finished"));
        events.truncate(3);
        events.push(Err(AppError::BadRequest(
            "Too many input tokens".to_string(),
        )));
        Step::ConverseStream(events)
    };
    for (later, code) in [
        (
            Step::ConverseStreamService("ThrottlingException", "Too many requests"),
            "rate_limit_exceeded",
        ),
        (
            Step::ConverseStreamService("ValidationException", "Input is too long"),
            "context_length_exceeded",
        ),
        (mid_stream, "context_length_exceeded"),
        (
            Step::Fail(AppError::UpstreamBedrock("boom".to_string())),
            "server_is_overloaded",
        ),
    ] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(false), WebSearchHeader::On);
        let run = run_steps(
            Path::ConverseStream,
            &provider,
            &req,
            vec![Path::ConverseStream.step(&first), later],
        )
        .await
        .unwrap();
        assert_eq!(failed_code(&run), code);
    }
    // A step-0 error inside the stream is never invalid_prompt.
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let req = normalized(request_1(false), WebSearchHeader::On);
    let run = run_steps(
        Path::ConverseStream,
        &provider,
        &req,
        vec![Step::ConverseStream(vec![Err(AppError::BadRequest(
            BINDING_ERROR.to_string(),
        ))])],
    )
    .await
    .unwrap();
    assert_eq!(failed_code(&run), "context_length_exceeded");
}

/// Without streaming, a later step's error is the HTTP error.
#[tokio::test]
async fn non_stream_later_error_is_an_http_error() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let req = normalized(request_1(false), WebSearchHeader::On);
    let first = output(vec![search_call("t1", "q")], "tool_use", (10, 5));
    let err = run_steps(
        Path::Converse,
        &provider,
        &req,
        vec![
            Step::Converse(first),
            Step::ConverseService("ValidationException", BINDING_ERROR),
        ],
    )
    .await
    .err()
    .expect("error");
    assert!(
        matches!(err, AppError::BadRequest(message) if message.contains("bound to a different"))
    );
}

#[tokio::test]
async fn dropping_the_stream_cancels_the_search() {
    let dropped = Arc::new(AtomicBool::new(false));
    let search = FakeSearch::new(vec![Reply::Hang(dropped.clone())]);
    let provider = provider(&search);
    let req = normalized(request_1(false), WebSearchHeader::On);
    let scripted = Scripted::new(vec![Step::ConverseStream(converse_stream_events(&output(
        vec![search_call("t1", "slow")],
        "tool_use",
        (10, 5),
    )))]);
    let plan = provider.web_search_plan(&req);
    let runtime = ResponsesStreamRuntime::new(
        req.request_id.clone(),
        Instant::now(),
        Duration::from_secs(5),
    );
    let transport: Arc<dyn StepTransport> = scripted.clone();
    let mut stream = respond_stream(
        provider.clone(),
        transport,
        req.clone(),
        plan,
        Lane::Converse,
        runtime,
    )
    .await
    .unwrap();
    loop {
        let event = stream.next().await.expect("event").unwrap();
        if event.event_type() == "response.web_search_call.searching" {
            break;
        }
    }
    // The next poll starts the search, which never answers.
    assert!(
        tokio::time::timeout(Duration::from_millis(50), stream.next())
            .await
            .is_err()
    );
    assert_eq!(search.started.load(Ordering::SeqCst), 1);
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the search is still running"
    );
    drop(stream);
    assert!(
        dropped.load(Ordering::SeqCst),
        "dropping the stream cancelled the search"
    );
}

/// A continuation that Bedrock rejects for its cache points is retried once
/// without them, as on the single-call path, and the model is remembered as
/// cache-unsupported.
#[tokio::test]
async fn cache_strip_retry_on_continuation() {
    const NO_CACHE: &str =
        "You invoked an unsupported model or your request did not allow prompt caching.";
    for path in [Path::Converse, Path::ConverseStream] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(false), WebSearchHeader::On);
        let first = output(vec![search_call("t1", "q")], "tool_use", (10, 5));
        let rejected = if path.streams() {
            Step::ConverseStreamService("AccessDeniedException", NO_CACHE)
        } else {
            Step::ConverseService("AccessDeniedException", NO_CACHE)
        };
        let run = run_steps(
            path,
            &provider,
            &req,
            vec![path.step(&first), rejected, path.step(&answer("Done."))],
        )
        .await
        .unwrap();
        assert_eq!(run.response().status, "completed", "{path:?}");
        assert_eq!(run.sent.len(), 3, "{path:?}");
        let has_cache_point = |sent: &Sent| sent.body.to_string().contains("cachePoint");
        assert!(has_cache_point(&run.sent[0]));
        assert!(has_cache_point(&run.sent[1]));
        assert!(
            !has_cache_point(&run.sent[2]),
            "{path:?}: the retry has no cache point"
        );
        assert!(provider
            .cache_support()
            .is_unsupported(&normalize_for_match(SONNET)));
    }
}

// ---------------------------------------------------------------------------
// Requests that do not declare the tool (§0 item 1)
// ---------------------------------------------------------------------------

/// Undeclared requests run the single-call path end to end, against a local
/// mock of the Bedrock runtime API (never AWS): exactly one Bedrock call each,
/// none of them declares the synthetic tool, no search runs, and the response
/// is the single-call mapping of the mocked output. The bytes that path
/// assembles are pinned against the previous code in
/// `responses_provider::tests::undeclared_requests_keep_todays_bytes`.
#[tokio::test]
async fn undeclared_requests_never_reach_the_loop() {
    use crate::bedrock::anthropic_messages::messages_response_to_converse_output;
    use crate::bedrock::responses_response::from_converse_output_to_responses_with_tools;
    use crate::bedrock::responses_translate::build_responses_tools_with;
    use crate::domain::ResponsesProvider;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let converse_wire = json!({
        "output": { "message": { "role": "assistant", "content": [
            { "text": "Hello from Converse." },
            { "toolUse": { "toolUseId": "tooluse_1", "name": "mcp__ze_tools__direct_alpha", "input": {} } }
        ] } },
        "stopReason": "tool_use",
        "usage": { "inputTokens": 30, "outputTokens": 4, "totalTokens": 34 },
        "metrics": { "latencyMs": 12 }
    });
    let messages_wire = json!({
        "id": "msg_1", "type": "message", "role": "assistant",
        "content": [
            { "type": "text", "text": "Hello from InvokeModel." },
            { "type": "tool_use", "id": "toolu_1", "name": "mcp__ze_tools__direct_alpha", "input": {} }
        ],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 30, "output_tokens": 4 }
    });
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/[^/]+/converse$"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&converse_wire),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path_regex(r"^/model/[^/]+/invoke$"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(&messages_wire),
        )
        .mount(&server)
        .await;

    let search = FakeSearch::new(Vec::new());
    let with_backend = provider_with(
        Some(search.clone() as Arc<dyn WebSearchBackend>),
        limits(),
        Some(&server.uri()),
    );
    let without_backend = provider_with(None, limits(), Some(&server.uri()));
    let nova = {
        let mut request = request_1(false);
        request.model = "us.amazon.nova-pro-v1:0".to_string();
        request.reasoning = None;
        request
    };
    let cases = [
        (
            "absent",
            &with_backend,
            request_1(false),
            WebSearchHeader::Absent,
        ),
        ("off", &with_backend, request_1(false), WebSearchHeader::Off),
        (
            "on, no capability",
            &with_backend,
            nova,
            WebSearchHeader::On,
        ),
        (
            "on, backend off",
            &without_backend,
            request_1(false),
            WebSearchHeader::On,
        ),
        (
            "off, tool search lane",
            &with_backend,
            request_1(true),
            WebSearchHeader::Off,
        ),
        (
            "on, backend off, tool search lane",
            &without_backend,
            request_1(true),
            WebSearchHeader::On,
        ),
    ];
    for (label, provider, request, header) in &cases {
        let req = normalized(request.clone(), *header);
        assert!(!provider.web_search_plan(&req).declared, "{label}");
        let response = provider
            .respond(&req)
            .await
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let tool_search = provider.lane(&req.request, &req.resolved_model) == Lane::ToolSearch;
        let converse = if tool_search {
            messages_response_to_converse_output(&messages_wire).unwrap()
        } else {
            let mut out = converse_wire.clone();
            out.as_object_mut().unwrap().remove("metrics");
            out
        };
        let (_, registry) = build_responses_tools_with(&req.request, tool_search, false).unwrap();
        let expected = from_converse_output_to_responses_with_tools(
            &converse,
            &req.request,
            &req.request.model,
            &response.id,
            &registry,
        )
        .unwrap();
        let mut got = serde_json::to_value(&response).unwrap();
        let mut want = serde_json::to_value(&expected).unwrap();
        got["created_at"] = json!(0);
        want["created_at"] = json!(0);
        assert_eq!(got, want, "{label}");
        assert_eq!(got["output"][1]["namespace"], "mcp__ze_tools", "{label}");
    }

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), cases.len(), "one Bedrock call per request");
    for request in &received {
        let body = String::from_utf8_lossy(&request.body);
        assert!(
            !body.contains("Search the public web and get ranked results"),
            "an undeclared request declared the synthetic tool: {}",
            request.url
        );
    }
    assert_eq!(
        received
            .iter()
            .filter(|r| r.url.path().ends_with("/invoke"))
            .count(),
        2
    );
    assert!(search.queries().is_empty());
}

// ---------------------------------------------------------------------------
// Live tests (L2–L8 of the design; L1 is in agentcore_search_tests.rs)
//
// Bedrock and AgentCore are both pinned to us-east-2 in code, because a host
// may export another AWS_REGION. Run with:
//
//   env -u AWS_REGION -u AWS_DEFAULT_REGION -u AWS_ACCESS_KEY_ID \
//     -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN BEDROCK_INTEGRATION=1 \
//     AWS_PROFILE=us cargo test --lib web_search -- --ignored
//
// L7 also needs AWS_BEARER_TOKEN_BEDROCK for the mantle (GPT) upstream.
// ---------------------------------------------------------------------------

const LIVE_REGION: &str = "us-east-2";

fn live_settings() -> AppSettings {
    AppSettings {
        aws_region: LIVE_REGION.to_string(),
        web_search_region: Some(LIVE_REGION.to_string()),
        enable_prompt_caching: true,
        ..test_settings()
    }
}

/// The real provider: Bedrock clients and AgentCore for us-east-2 from the
/// default credential chain. `None` without `BEDROCK_INTEGRATION`.
async fn live_provider() -> Option<BedrockResponsesProvider> {
    if std::env::var("BEDROCK_INTEGRATION").is_err() {
        return None;
    }
    let settings = live_settings();
    let aws = crate::bedrock::client::build_aws_config(&settings).await;
    let limits = WebSearchLimits::from_settings(&settings);
    let backend =
        crate::bedrock::agentcore_search::resolve_web_search_backend(&settings, &aws, &limits)
            .expect("AgentCore backend for us-east-2");
    let caps = Arc::new(ConfigModelCapabilities::new(
        ModelCapabilityConfig::load("config/models.toml").expect("config/models.toml"),
    ));
    Some(BedrockResponsesProvider::new(
        BedrockClients::new(&aws),
        caps,
        Arc::new(RegionRoutingConfig::default()),
        Arc::new(NoImages),
        Arc::new(settings),
        Arc::new(CacheSupportRegistry::new()),
        Some(backend),
        limits,
    ))
}

fn live_request(
    model: &str,
    tool_search: bool,
    question: &str,
    effort: Option<&str>,
) -> ResponsesRequest {
    let mut request = request_1(tool_search);
    request.model = model.to_string();
    request.reasoning = effort
        .map(|effort| serde_json::from_value(json!({ "effort": effort })).expect("reasoning"));
    request.max_output_tokens = Some(16000);
    let mut input = input_of(&request);
    input.truncate(2);
    input.push(user(question));
    with_input(request, input)
}

/// Stream `request` through the real provider and collect the run.
async fn live_stream(
    provider: &BedrockResponsesProvider,
    request: ResponsesRequest,
    header: WebSearchHeader,
) -> Result<Run, AppError> {
    use crate::domain::ResponsesProvider;
    let req = normalized(request, header);
    let stream = provider.respond_stream(&req).await?;
    let events: Vec<ResponseStreamEvent> = stream.map(|event| event.unwrap()).collect().await;
    let response = events.iter().find_map(|event| match event {
        ResponseStreamEvent::Completed { response, .. }
        | ResponseStreamEvent::Incomplete { response, .. } => Some(response.clone()),
        _ => None,
    });
    let events: Vec<Value> = events
        .iter()
        .map(|event| serde_json::to_value(event).unwrap())
        .collect();
    if response.is_none() {
        return Err(AppError::UpstreamBedrock(format!(
            "stream ended without a response: {}",
            events.last().cloned().unwrap_or_default()
        )));
    }
    Ok(Run {
        response,
        events,
        sent: Vec::new(),
    })
}

const NEWS_QUESTION: &str =
    "Search the web: what did Amazon Bedrock AgentCore launch most recently? Answer in two sentences and cite the page as a markdown link.";

/// L2/L8: a search and a cited answer; the replay plus a follow-up with the
/// header off is accepted (sticky declaration, Bedrock's real prefix check).
async fn live_search_then_follow_up(model: &str, tool_search: bool) {
    let Some(provider) = live_provider().await else {
        return;
    };
    let first = live_request(model, tool_search, NEWS_QUESTION, Some("medium"));
    let run = live_stream(&provider, first.clone(), WebSearchHeader::On)
        .await
        .expect("search run");
    eprintln!(
        "{model}: {}",
        serde_json::to_string(&run.event_types()).unwrap()
    );
    assert!(
        run.types().iter().any(|t| t == "web_search_call"),
        "{model}: no search"
    );
    let answer = run.output().last().cloned().unwrap();
    assert!(answer["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("http"));

    let follow_up = next_request(
        &first,
        &run,
        vec![user("Thanks. In one sentence, why does it matter?")],
    );
    let next = live_stream(&provider, follow_up, WebSearchHeader::Off)
        .await
        .expect("follow-up with the header off");
    assert_eq!(next.response().status, "completed");
}

#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l2_sonnet_search_then_follow_up_off() {
    live_search_then_follow_up(SONNET, false).await;
}

/// L3: a conversation with unmarked thinking turns search on; the first
/// declaration drops that thinking and Bedrock accepts the request.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l3_first_declaration_after_unmarked_thinking() {
    let Some(provider) = live_provider().await else {
        return;
    };
    let first = live_request(
        SONNET,
        false,
        "Name one AWS Region in Europe.",
        Some("medium"),
    );
    let run = live_stream(&provider, first.clone(), WebSearchHeader::Off)
        .await
        .expect("plain run");
    let next = next_request(&first, &run, vec![user(NEWS_QUESTION)]);
    let searched = live_stream(&provider, next, WebSearchHeader::On)
        .await
        .expect("first declaration");
    assert_eq!(searched.response().status, "completed");
}

/// L4: L2 on the InvokeModel lane (client tool search).
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l4_tool_search_lane() {
    live_search_then_follow_up(SONNET, true).await;
}

/// L5: the same replay sent twice is accepted twice (Converse writes
/// `Document` keys in hash order; the prefix check must not care).
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l5_same_replay_twice() {
    let Some(provider) = live_provider().await else {
        return;
    };
    let first = live_request(SONNET, false, NEWS_QUESTION, Some("medium"));
    let run = live_stream(&provider, first.clone(), WebSearchHeader::On)
        .await
        .expect("search run");
    let follow_up = next_request(&first, &run, vec![user("Summarize that in five words.")]);
    for attempt in 0..2 {
        let next = live_stream(&provider, follow_up.clone(), WebSearchHeader::On)
            .await
            .unwrap_or_else(|e| panic!("attempt {attempt}: {e}"));
        assert_eq!(next.response().status, "completed");
    }
}

/// L6: without thinking, a mixed-step history in the aligned order is
/// accepted; the unaligned order is sent straight to Converse and its result
/// printed, as a documented fact.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l6_mixed_history_alignment() {
    use crate::bedrock::provider::{build_sdk_messages, build_sdk_tool_config};
    use crate::bedrock::responses_translate::encode_web_search_carrier;
    let Some(provider) = live_provider().await else {
        return;
    };
    let carrier = encode_web_search_carrier(
        &[
            crate::bedrock::responses_translate::web_search_tool_use_block(
                "tooluse_ws_live",
                &json!({ "query": "AgentCore web search" }),
            ),
        ],
        &[crate::bedrock::web_search::tool_result_block(
            "tooluse_ws_live",
            r#"{"error_code":"unavailable","query":"AgentCore web search","type":"web_search_error"}"#,
        )],
        1,
    );
    let mut request = live_request(
        SONNET,
        false,
        "Look up ticket T-1 and search the web.",
        None,
    );
    let mut input = input_of(&request);
    input.extend([
        json!({ "type": "function_call", "call_id": "tooluse_fn_live", "name": "lookup_ticket",
                "arguments": "{\"id\":\"T-1\"}" }),
        json!({ "type": "reasoning", "id": "rs_live_ws0", "summary": [], "encrypted_content": carrier }),
        json!({ "type": "function_call_output", "call_id": "tooluse_fn_live", "output": "T-1: open" }),
    ]);
    request = with_input(request, input);
    let aligned = live_stream(&provider, request.clone(), WebSearchHeader::Off)
        .await
        .expect("aligned mixed history");
    assert_eq!(aligned.response().status, "completed");

    // The same history with the results in input order (search result first).
    let parsed = parse_responses_input(
        &request,
        SONNET,
        &NoImages,
        TranslateOptions {
            tool_search: false,
            web_search: true,
        },
    )
    .await
    .unwrap();
    let assembled = provider
        .assemble_parsed(&request, SONNET, &parsed, true, false, true)
        .unwrap();
    let mut messages = assembled.messages.clone();
    let last = messages.as_array_mut().unwrap().last_mut().unwrap();
    last["content"].as_array_mut().unwrap().reverse();
    let aws = crate::bedrock::client::build_aws_config(&live_settings()).await;
    let client = aws_sdk_bedrockruntime::Client::new(&aws);
    let result = client
        .converse()
        .model_id(SONNET)
        .set_messages(Some(build_sdk_messages(&messages).unwrap()))
        .tool_config(build_sdk_tool_config(assembled.tool_config.as_ref().unwrap()).unwrap())
        .inference_config(
            aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                .max_tokens(512)
                .build(),
        )
        .send()
        .await;
    eprintln!(
        "unaligned toolResult order: {}",
        match &result {
            Ok(_) => "accepted".to_string(),
            Err(e) => format!(
                "rejected: {}",
                e.as_service_error()
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            ),
        }
    );
}

/// L7: a searched conversation continues on GPT through mantle, which gets
/// neither the carrier nor the gateway's `web_search_call` items.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l7_searched_conversation_moves_to_gpt() {
    use crate::bedrock::mantle_client::MantleClient;
    use crate::bedrock::mantle_provider::MantleResponsesProvider;
    use crate::domain::ResponsesProvider;
    let Some(provider) = live_provider().await else {
        return;
    };
    let Ok(key) = std::env::var("AWS_BEARER_TOKEN_BEDROCK") else {
        eprintln!("L7 skipped: AWS_BEARER_TOKEN_BEDROCK is not set");
        return;
    };
    let first = live_request(SONNET, false, NEWS_QUESTION, Some("medium"));
    let run = live_stream(&provider, first.clone(), WebSearchHeader::On)
        .await
        .expect("search run");
    let mut gpt = next_request(
        &first,
        &run,
        vec![user("Now answer briefly: what is Bedrock?")],
    );
    gpt.model = "openai.gpt-5.6-terra".to_string();
    gpt.stream = Some(false);
    gpt.max_output_tokens = Some(2000);
    let settings = Arc::new(live_settings());
    let mantle = MantleResponsesProvider::new(
        MantleClient::new(
            reqwest::Client::new(),
            settings.mantle_base_url_template.clone(),
            settings.mantle_chat_base_url_template.clone(),
            key,
        ),
        Arc::new(ConfigModelCapabilities::new(
            ModelCapabilityConfig::load("config/models.toml").unwrap(),
        )),
        settings,
    );
    let raw_body = bytes::Bytes::from(serde_json::to_vec(&gpt).unwrap());
    let req = NormalizedResponsesRequest {
        raw_body,
        ..normalized(gpt, WebSearchHeader::Off)
    };
    let response = mantle
        .respond(&req)
        .await
        .expect("mantle accepts the history");
    assert_eq!(response.status, "completed");
}

/// L8: L2 on Opus 5.5.
#[tokio::test]
#[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
async fn live_l8_opus_search_then_follow_up_off() {
    live_search_then_follow_up(OPUS, false).await;
}

/// After a search, a `max_tokens` stop ends `completed`; a client call cut
/// off with it is not handed back, on any path (the client would run it).
#[tokio::test]
async fn truncated_client_call_after_search_is_not_handed_back() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(vec![search_call("t1", "q")], "tool_use", (10, 5)),
            output(
                vec![
                    say("Calling"),
                    tool_call("fn1", "lookup_ticket", json!({ "id": "T" })),
                ],
                "max_tokens",
                (10, 5),
            ),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert!(
            !run.types().iter().any(|t| t == "function_call"),
            "{path:?}"
        );
        assert_gateway_stop(&run, path, &stop_text(Exhaustion::OutputBudget, 1));
    }
}

/// The stop reason a content-filtered step has on `path`: Messages calls it
/// `refusal`, which the InvokeModel lane reads as Converse's
/// `content_filtered`.
fn filtered(path: Path) -> &'static str {
    if path.tool_search() {
        "refusal"
    } else {
        "content_filtered"
    }
}

/// After a search, a content-filtered answer ends like a `max_tokens` stop:
/// `completed` with the gateway's message, and a client call cut off with it
/// is not handed back. zuno e97a2747 retries `incomplete` with
/// `content_filter` (`codex-rs/protocol/src/error.rs`, `retry_delay`) after
/// recording a developer message (`codex-rs/core/src/responses_retry.rs`);
/// the gateway moves that message into `system`, so the retry would fail on
/// every marked thinking block, and it would search again.
#[tokio::test]
async fn content_filter_after_search_ends_completed_with_gateway_message() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(vec![search_call("t1", "q")], "tool_use", (10, 5)),
            output(
                vec![
                    say("Filtered"),
                    tool_call("fn1", "lookup_ticket", json!({ "id": "T" })),
                ],
                filtered(path),
                (10, 5),
            ),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert!(
            !run.types().iter().any(|t| t == "function_call"),
            "{path:?}"
        );
        assert_eq!(
            run.output()[run.output().len() - 2]["content"][0]["text"],
            "Filtered",
            "{path:?}: the step's text is still output"
        );
        assert_gateway_stop(
            &run,
            path,
            "The answer stopped after 1 web search: the content filter stopped the answer.",
        );
    }
}

/// Without a real search in the answer, a content-filtered answer keeps the
/// single-call meaning: `incomplete` with `content_filter`.
#[tokio::test]
async fn content_filter_without_search_stays_incomplete() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [output(vec![say("Filtered")], filtered(path), (10, 5))];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.response().status, "incomplete", "{path:?}");
        assert_eq!(
            run.response().incomplete_details,
            Some(json!({ "reason": "content_filter" })),
            "{path:?}"
        );
        assert!(
            !run.output()
                .iter()
                .any(|i| i["id"].as_str().unwrap().ends_with("_stop")),
            "{path:?}"
        );
    }
}

/// After a search the model may end its turn without an answer: an empty
/// `end_turn`, or thinking only (more common right after tool results). The
/// response then ends with the gateway's message instead of `completed`
/// without an answer, which zuno would take as the end of the turn.
#[tokio::test]
async fn empty_answer_after_search_ends_completed_with_gateway_message() {
    for path in Path::ALL {
        for last in [Vec::new(), vec![think("Nothing to add.", "sig-x")]] {
            let search = FakeSearch::new(vec![results(&[("A", "https://a.example/", "a")])]);
            let provider = provider(&search);
            let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
            let steps = [
                output(
                    vec![
                        think("s", "sig-s"),
                        say("Let me look that up."),
                        search_call("t1", "q"),
                    ],
                    "tool_use",
                    (10, 5),
                ),
                output(last.clone(), "end_turn", (10, 5)),
            ];
            let run = run_on(path, &provider, &req, &steps).await.unwrap();
            assert_eq!(run.sent.len(), 2, "{path:?}");
            assert_gateway_stop(
                &run,
                path,
                "The answer stopped after 1 web search: the model gave no answer.",
            );
            assert_unique_ids(&run, &format!("{path:?} empty answer"));
        }
    }
}

/// A final step that answers only with client calls is not an empty answer:
/// the calls are handed back and the answer goes on after them.
#[tokio::test]
async fn client_calls_after_search_are_not_an_empty_answer() {
    for path in Path::ALL {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(path.tool_search()), WebSearchHeader::On);
        let steps = [
            output(vec![search_call("t1", "q")], "tool_use", (10, 5)),
            output(
                vec![tool_call("fn1", "lookup_ticket", json!({ "id": "T" }))],
                "tool_use",
                (10, 5),
            ),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.response().status, "completed", "{path:?}");
        assert_eq!(
            run.types(),
            vec!["web_search_call", "reasoning", "function_call"],
            "{path:?}"
        );
    }
}

/// Without a real search an empty answer keeps the single-call shape.
#[tokio::test]
async fn empty_answer_without_search_has_no_gateway_message() {
    for path in [Path::Converse, Path::ConverseStream] {
        let search = FakeSearch::new(Vec::new());
        let provider = provider(&search);
        let req = normalized(request_1(false), WebSearchHeader::On);
        let run = run_on(
            path,
            &provider,
            &req,
            &[output(Vec::new(), "end_turn", (10, 5))],
        )
        .await
        .unwrap();
        assert_eq!(run.response().status, "completed", "{path:?}");
        assert!(
            !run.output()
                .iter()
                .any(|i| i["id"].as_str().unwrap().ends_with("_stop")),
            "{path:?}"
        );
    }
}

/// An input that ends on an assistant item (here a marked reasoning item, as
/// a client replays it when it retries after a partial response) is sent to
/// a model that cannot prefill with the continuation turn after it. The
/// first step's thinking is bound to that turn, so every later step keeps it:
/// each continuation extends the first call.
#[tokio::test]
async fn the_continuation_turn_stays_in_every_step() {
    let marked = crate::bedrock::responses_translate::encode_reasoning_envelope_with(
        &[think("Partial thinking.", "sig-b")],
        Some(crate::bedrock::responses_translate::WEB_SEARCH_MARK),
    )
    .unwrap();
    for path in Path::ALL {
        let search = FakeSearch::new(vec![results(&[("A", "https://a.example/", "a")])]);
        let provider = provider(&search);
        let mut input = input_of(&request_1(path.tool_search()));
        input.push(
            json!({ "type": "reasoning", "id": "rs_earlier_1", "summary": [],
                           "encrypted_content": marked }),
        );
        let req = normalized(
            with_input(request_1(path.tool_search()), input),
            WebSearchHeader::On,
        );
        let steps = [
            output(
                vec![think("Search it.", "sig-r"), search_call("t1", "q")],
                "tool_use",
                (10, 5),
            ),
            answer("Done."),
        ];
        let run = run_on(path, &provider, &req, &steps).await.unwrap();
        assert_eq!(run.sent.len(), 2, "{path:?}");
        let (tools_0, system_0, messages_0) = prefix_parts(&run.sent[0]);
        let (tools_1, system_1, messages_1) = prefix_parts(&run.sent[1]);
        let (first, second) = (strip_cache(&messages_0), strip_cache(&messages_1));
        let last = first.last().unwrap();
        assert_eq!(last["role"], "user", "{path:?}");
        assert_eq!(
            last["content"][0]["text"], "Please continue your response from where you left off.",
            "{path:?}: the first call ends with the continuation turn"
        );
        assert_eq!(tools_1, tools_0, "{path:?}");
        assert_eq!(system_1, system_0, "{path:?}");
        assert!(second.len() > first.len(), "{path:?}");
        assert_eq!(
            &second[..first.len()],
            first.as_slice(),
            "{path:?}: the second call extends the first"
        );
    }
}

/// The step limit before any real search names what stopped the searches
/// (web search is off for this message, or unavailable), not the search
/// limit.
#[tokio::test]
async fn step_limit_without_a_search_names_the_reason() {
    let marked = crate::bedrock::responses_translate::encode_reasoning_envelope_with(
        &[think("Earlier.", "sig-e")],
        Some(crate::bedrock::responses_translate::WEB_SEARCH_MARK),
    )
    .unwrap();
    let mut input = input_of(&request_1(false));
    input.push(json!({ "type": "reasoning", "id": "rs_e", "summary": [],
                       "encrypted_content": marked }));
    input.push(json!({ "type": "message", "role": "assistant",
                       "content": [{ "type": "output_text", "text": "Earlier answer." }] }));
    input.push(user("And now?"));
    let limits = WebSearchLimits {
        max_calls: 1,
        ..limits()
    };
    let calls: Vec<Value> = (0..3)
        .map(|i| {
            output(
                vec![search_call(&format!("t{i}"), "q")],
                "tool_use",
                (10, 5),
            )
        })
        .collect();
    let invalid: Vec<Value> = (0..3)
        .map(|i| {
            output(
                vec![tool_call(
                    &format!("t{i}"),
                    "web_search",
                    json!({ "query": "" }),
                )],
                "tool_use",
                (10, 5),
            )
        })
        .collect();
    for (header, backend, steps, expected) in [
        (
            WebSearchHeader::Off,
            true,
            &calls,
            "The answer stopped: web search is off for this message.",
        ),
        (
            WebSearchHeader::On,
            false,
            &calls,
            "The answer stopped: web search is unavailable.",
        ),
        (
            WebSearchHeader::On,
            true,
            &invalid,
            "The answer stopped: the step limit was reached before any web search ran.",
        ),
    ] {
        for path in [Path::Converse, Path::ConverseStream] {
            let search = FakeSearch::new(Vec::new());
            let backend = backend.then(|| search.clone() as Arc<dyn WebSearchBackend>);
            let provider = provider_with(backend, limits, None);
            let req = normalized(with_input(request_1(false), input.clone()), header);
            let run = run_on(path, &provider, &req, steps).await.unwrap();
            assert!(search.queries().is_empty(), "{path:?} {header:?}");
            assert_eq!(run.sent.len(), 3, "{path:?} {header:?}");
            assert_gateway_stop(&run, path, expected);
        }
    }
}

/// A request in the middle of a tool loop whose thinking has no mark does not
/// declare the tool now: the first declaration would drop that thinking,
/// the thinking of the loop's last assistant message included. Search starts
/// with the next user message; a marked history still declares, and so does
/// a loop without gateway thinking, which loses nothing.
#[test]
fn first_declaration_waits_for_the_next_user_message() {
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let unmarked = crate::bedrock::responses_translate::encode_reasoning_envelope(&[think(
        "Look the ticket up.",
        "sig-p",
    )])
    .unwrap();
    let mut input = input_of(&request_1(false));
    input.push(json!({ "type": "reasoning", "id": "rs_p", "summary": [],
                       "encrypted_content": unmarked }));
    input.push(
        json!({ "type": "function_call", "call_id": "fn1", "name": "lookup_ticket",
                       "arguments": "{\"id\":\"T-1\"}" }),
    );
    input.push(json!({ "type": "function_call_output", "call_id": "fn1", "output": "open" }));
    let plan = |input: &[Value]| {
        provider.web_search_plan(&normalized(
            with_input(request_1(false), input.to_vec()),
            WebSearchHeader::On,
        ))
    };
    assert!(!plan(&input).declared, "a tool loop without marks");
    // A developer message after the outputs does not end the loop.
    let mut with_developer = input.clone();
    with_developer.push(json!({ "type": "message", "role": "developer",
                                "content": [{ "type": "input_text", "text": "note" }] }));
    assert!(!plan(&with_developer).declared);
    // The next user message starts a new answer: declared.
    let mut next = input.clone();
    next.push(user("And the quota?"));
    assert!(plan(&next).declared);
    // A marked history stays declared in a tool loop.
    let marked = crate::bedrock::responses_translate::encode_reasoning_envelope_with(
        &[think("Look the ticket up.", "sig-p")],
        Some(crate::bedrock::responses_translate::WEB_SEARCH_MARK),
    )
    .unwrap();
    let mut marked_loop = input.clone();
    marked_loop[input.len() - 3] = json!({ "type": "reasoning", "id": "rs_p", "summary": [],
                                           "encrypted_content": marked });
    assert!(plan(&marked_loop).declared);
    // A tool loop without gateway thinking (thinking off): declared.
    let mut no_thinking = input.clone();
    no_thinking.remove(input.len() - 3);
    assert!(plan(&no_thinking).declared);
}

/// A gateway `web_search_call` after the newest carrier belongs to a step
/// whose carrier never reached the client (the stream broke between that
/// step's searches). The search ran, so it counts toward the answer's limit.
#[tokio::test]
async fn searches_without_a_carrier_count_toward_the_answer() {
    let marked = crate::bedrock::responses_translate::encode_reasoning_envelope_with(
        &[think("Search twice.", "sig-w")],
        Some(crate::bedrock::responses_translate::WEB_SEARCH_MARK),
    )
    .unwrap();
    let call = |id: &str| {
        json!({ "type": "web_search_call", "id": format!("ws_bgw_{id}"), "status": "completed",
                "action": { "type": "search", "query": id, "queries": [id] } })
    };
    let mut input = input_of(&request_1(false));
    input.push(json!({ "type": "reasoning", "id": "rs_w", "summary": [],
                       "encrypted_content": marked }));
    for id in ["o1", "o2", "o3", "o4"] {
        input.push(call(id));
    }
    // A GPT search (mantle's own id) is not the gateway's.
    input.push(
        json!({ "type": "web_search_call", "id": "ws_67f0", "status": "completed",
                       "action": { "type": "search", "query": "gpt" } }),
    );
    let search = FakeSearch::new(Vec::new());
    let provider = provider(&search);
    let req = normalized(with_input(request_1(false), input), WebSearchHeader::On);
    assert_eq!(provider.web_search_plan(&req).used_in_answer, 4);
    let steps = [
        output(
            vec![search_call("t1", "five"), search_call("t2", "six")],
            "tool_use",
            (10, 5),
        ),
        answer("Done."),
    ];
    let run = run_on(Path::Converse, &provider, &req, &steps)
        .await
        .unwrap();
    assert_eq!(search.queries(), vec!["five"], "one search left");
    let carrier = &carriers(&run.output())[0];
    assert_eq!(carrier["used"], 5);
    assert_eq!(
        carrier_results(carrier)[1]["error_code"],
        "max_uses_exceeded"
    );
}
