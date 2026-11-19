//! The gateway-executed web search loop on the Responses surface (plan C2).
//!
//! A request reaches this module when it *declares* the gateway's web search
//! (`WebSearchPlan::declared`): its `X-Weniq-Web-Search` header is `on` on a
//! model with the `web_search` capability while a backend is configured, or
//! its history already carries the gateway's web search marks. Every other
//! request takes the single-call path of [`BedrockResponsesProvider`],
//! unchanged.
//!
//! One response can make several Bedrock calls (*steps*). A step that calls
//! only the synthetic `web_search` tool is executed here: each call runs
//! against the [`crate::bedrock::web_search::WebSearchBackend`] (or gets an
//! error result without a search), the step's hidden `toolUse` / `toolResult`
//! blocks are written into a *carrier* reasoning item, and the loop calls
//! Bedrock again with the step appended. A step that also calls client tools
//! (a *mixed* step) runs its searches, hands the client calls back and ends
//! the response; any other step is the final one.
//!
//! ## Why the continuation is built from the output items
//!
//! Claude 5.5 binds each thinking block to everything before it (`system`,
//! `tools`, earlier messages). The client replays this response's items on
//! its next request, and the gateway rebuilds the history from them; the
//! thinking of step `k + 1` was produced under the loop's continuation, so the
//! two must match. The loop therefore never sends Bedrock's raw content back:
//! it turns the items it emitted into input items the way zuno replays them
//! and parses them with the same translation a replay goes through.
//!
//! ## Terminal states
//!
//! After a real search in the answer, the response never ends `incomplete`
//! (zuno retries that and would search again) and never ends `completed`
//! without an answer: running out of steps or of output budget appends a
//! gateway message (`msg_S_stop`) that says why, with status `completed`.
//! Without a real search the terminal states are the single-call ones.

use std::sync::Arc;
use std::time::Instant;

use async_stream::stream;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::bedrock::anthropic_messages::messages_response_to_converse_output;
use crate::bedrock::cache_support::{send_with_cache_strip_retry, SendError};
use crate::bedrock::capabilities::normalize_for_match;
use crate::bedrock::responses_provider::{resp_id, AssembledConverse, BedrockResponsesProvider};
use crate::bedrock::responses_response::{
    carrier_item, created_at_now, map_step, message_item, step_status, tool_output_item,
    usage_from_sum, web_search_call_item, StepItemIds, StepToolUse, UsageSum,
};
use crate::bedrock::responses_stream::{
    tool_events, ResponsesStreamRuntime, ResponsesStreamState, StepEvents, StepOptions, ToolAccum,
};
use crate::bedrock::responses_translate::{
    encode_web_search_carrier, parse_items, parse_responses_input, web_search_tool_use_block,
    HistoryScan, ParsedInput, ResponsesToolKind, ResponsesToolRegistry, TranslateOptions,
};
use crate::bedrock::web_search::{
    build_results_payload, error_payload, sanitize_query, tool_result_block, url_citations,
    web_search_call_id, CitationSources, Source, WebSearchErrorCode,
};
use crate::domain::{NormalizedResponsesRequest, ResponsesStream, WebSearchHeader};
use crate::error::{responses_error, AppError};
use crate::openai::responses_schema::{
    OutputContentPart, ResponseContentPart, ResponseInputItem, ResponseOutputItem,
    ResponseStreamEvent, ResponsesContent, ResponsesInput, ResponsesRequest, ResponsesResponse,
    ResponsesRole, ResponsesToolChoice,
};

/// A continuation needs at least this much of `max_output_tokens` left;
/// otherwise the response ends after the search step.
pub(crate) const MIN_CONTINUATION_TOKENS: i32 = 1024;

/// Whether a declared request's `web_search` calls can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecGate {
    /// Header `on`, backend configured, model capable: calls run.
    Search,
    /// The header is absent or `off`: calls get `disabled_for_this_message`.
    Disabled,
    /// Header `on`, but no backend or the model lacks the capability: calls
    /// get `unavailable`.
    Unavailable,
}

/// The Bedrock API a request uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Converse / ConverseStream.
    Converse,
    /// InvokeModel with an Anthropic Messages body (client tool search).
    ToolSearch,
}

/// The web search decision for one request (plan C2 §2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WebSearchPlan {
    /// The request declares the synthetic tool and takes the loop.
    pub declared: bool,
    pub gate: ExecGate,
    /// Results of the replayed carriers, which a citation may point at.
    pub sources: CitationSources,
    /// Real searches the current answer already ran in earlier requests.
    pub used_in_answer: u32,
}

/// Decide declaration and execution.
///
/// Execution needs the header `on`, a backend and the capability. Declaration
/// also holds whenever the history carries a mark: the thinking of a
/// conversation that declared the tool was signed under a tool list that has
/// it, so the declaration must not depend on the switch or the model. `scan`
/// is `None` when the history could not be scanned; it then counts as
/// unmarked.
pub(crate) fn plan_web_search(
    header: WebSearchHeader,
    backend_on: bool,
    capable: bool,
    scan: Option<&HistoryScan>,
) -> WebSearchPlan {
    let executable = header == WebSearchHeader::On && backend_on && capable;
    let gate = if executable {
        ExecGate::Search
    } else if header == WebSearchHeader::On {
        ExecGate::Unavailable
    } else {
        ExecGate::Disabled
    };
    let mut sources = CitationSources::default();
    for payload in scan.iter().flat_map(|scan| &scan.carrier_payloads) {
        sources.extend_from_payload(payload);
    }
    WebSearchPlan {
        declared: executable || scan.is_some_and(|scan| scan.marked),
        gate,
        sources,
        used_in_answer: scan.map_or(0, |scan| scan.used_in_answer),
    }
}

/// The Bedrock calls of one step. [`BedrockResponsesProvider`] implements it
/// with its own sends; tests script it. Only I/O sits behind it: assembly, the
/// cache-strip retry and the state machines are the production code.
#[async_trait::async_trait]
pub(crate) trait StepTransport: Send + Sync {
    /// Converse; the output as Converse-shaped JSON.
    async fn converse(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<Value, SendError<ConverseError>>;

    /// ConverseStream.
    async fn converse_stream(
        &self,
        request_model: &str,
        assembled: &AssembledConverse,
    ) -> Result<StepEvents, SendError<ConverseStreamError>>;

    /// InvokeModel with a Messages body; the Messages response JSON.
    async fn invoke(&self, request_model: &str, body: Vec<u8>) -> Result<Value, AppError>;

    /// InvokeModelWithResponseStream with a Messages body.
    async fn invoke_stream(
        &self,
        request_model: &str,
        body: Vec<u8>,
    ) -> Result<StepEvents, AppError>;
}

/// The `response.failed` code for an error of step `step` (0-based).
///
/// A later step that Bedrock rejects for a thinking block bound to another
/// prefix gets `invalid_prompt`: retrying cannot succeed, and the default
/// mapping (`context_length_exceeded`) would make the client compact its
/// context and hide a gateway defect. Every other error keeps the shared
/// mapping of [`responses_error`].
pub(crate) fn loop_failure_code(step: usize, err: &AppError) -> &'static str {
    if step >= 1 {
        if let AppError::BadRequest(message) = err {
            let message = message.to_lowercase();
            if message.contains("bound to a different conversation")
                || (message.contains("thinking") && message.contains("signature"))
            {
                return "invalid_prompt";
            }
        }
    }
    responses_error(err).0
}

/// Why the gateway ended an answer after a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exhaustion {
    /// The step limit was reached while the model still called `web_search`.
    StepLimit,
    /// Too little of `max_output_tokens` is left, or Bedrock stopped at
    /// `max_tokens`.
    OutputBudget,
}

impl Exhaustion {
    fn as_str(self) -> &'static str {
        match self {
            Self::StepLimit => "step_limit",
            Self::OutputBudget => "output_budget",
        }
    }

    /// The text of the gateway's message; `searches` is the answer's count.
    pub(crate) fn message(self, searches: u32) -> String {
        let noun = if searches == 1 {
            "web search"
        } else {
            "web searches"
        };
        let why = match self {
            Self::StepLimit => "the search limit for one answer was reached",
            Self::OutputBudget => "the output token budget ran out",
        };
        format!("The answer stopped after {searches} {noun}: {why}.")
    }
}

/// What one step is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepKind {
    /// Only `web_search` calls (stop reason `tool_use`): run them, continue.
    SearchOnly,
    /// `web_search` and client calls: run the searches, hand back the calls.
    Mixed,
    /// Anything else: the answer.
    Final,
}

fn is_web_search(registry: &ResponsesToolRegistry, name: &str) -> bool {
    registry
        .resolve(name)
        .is_some_and(|binding| binding.kind == ResponsesToolKind::WebSearch)
}

fn classify<'a>(
    stop_reason: Option<&str>,
    names: impl Iterator<Item = &'a str>,
    registry: &ResponsesToolRegistry,
) -> StepKind {
    if stop_reason != Some("tool_use") {
        return StepKind::Final;
    }
    let (mut searches, mut others) = (0usize, 0usize);
    for name in names {
        if is_web_search(registry, name) {
            searches += 1;
        } else {
            others += 1;
        }
    }
    match (searches, others) {
        (0, _) => StepKind::Final,
        (_, 0) => StepKind::SearchOnly,
        _ => StepKind::Mixed,
    }
}

/// A `web_search` call of a step: its id and its input object.
#[derive(Debug, Clone)]
struct SearchCall {
    call_id: String,
    /// An object: a non-object input becomes `{}`, which the carrier can hold.
    input: Value,
}

impl SearchCall {
    fn new(call_id: &str, input: &Value) -> Self {
        Self {
            call_id: call_id.to_string(),
            input: if input.is_object() {
                input.clone()
            } else {
                json!({})
            },
        }
    }
}

/// What the gateway does with one `web_search` call (plan C2 §5.3).
#[derive(Debug, Clone, PartialEq, Eq)]
enum CallPlan {
    /// No search: answer with this error code (and the cleaned query, when
    /// it has one).
    Refuse(WebSearchErrorCode, Option<String>),
    /// Search for this cleaned query.
    Search(String),
}

/// Decide one call, in this order: the gate, the answer's search count, the
/// query, and the step limit (`allow_search` false: the response ends after
/// this step, so a search would not reach the model in it).
fn plan_call(
    gate: ExecGate,
    used: u32,
    max_calls: u32,
    allow_search: bool,
    input: &Value,
) -> CallPlan {
    let query = sanitize_query(input).ok();
    match gate {
        ExecGate::Disabled => {
            return CallPlan::Refuse(WebSearchErrorCode::DisabledForThisMessage, query)
        }
        ExecGate::Unavailable => return CallPlan::Refuse(WebSearchErrorCode::Unavailable, query),
        ExecGate::Search => {}
    }
    if used >= max_calls {
        return CallPlan::Refuse(WebSearchErrorCode::MaxUsesExceeded, query);
    }
    let Some(query) = query else {
        return CallPlan::Refuse(WebSearchErrorCode::InvalidToolInput, None);
    };
    if !allow_search {
        return CallPlan::Refuse(WebSearchErrorCode::MaxUsesExceeded, Some(query));
    }
    CallPlan::Search(query)
}

/// One executed search.
struct SearchRun {
    payload: String,
    /// `None` when the search failed.
    sources: Option<Vec<Source>>,
}

impl SearchRun {
    fn status(&self) -> &'static str {
        if self.sources.is_some() {
            "completed"
        } else {
            "failed"
        }
    }
}

/// The request-scoped collaborators of the loop.
struct Ctx<'a> {
    provider: &'a BedrockResponsesProvider,
    transport: &'a dyn StepTransport,
    /// The original request (its echo fields go into the response).
    request: &'a ResponsesRequest,
    resolved: &'a str,
    request_id: &'a Arc<str>,
    lane: Lane,
    gate: ExecGate,
    response_id: &'a str,
}

impl Ctx<'_> {
    fn opts(&self) -> TranslateOptions {
        TranslateOptions {
            tool_search: self.lane == Lane::ToolSearch,
            web_search: true,
        }
    }

    /// Run one search against the backend; failures become error payloads.
    async fn run_search(&self, step: usize, query: &str) -> SearchRun {
        let limits = self.provider.web_search_limits();
        let Some(backend) = self.provider.web_search_backend() else {
            return SearchRun {
                payload: error_payload(Some(query), WebSearchErrorCode::Unavailable),
                sources: None,
            };
        };
        let started = Instant::now();
        let result = backend.search(query, limits.max_results).await;
        let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        match result {
            Ok(outcome) => {
                let (payload, sources) = build_results_payload(query, &outcome, limits);
                tracing::info!(
                    request_id = %self.request_id,
                    model = %self.request.model,
                    step,
                    outcome = "ok",
                    results = sources.len(),
                    duration_ms,
                    agentcore_request_id = outcome.request_id.as_deref().unwrap_or("none"),
                    "web search finished"
                );
                SearchRun {
                    payload,
                    sources: Some(sources),
                }
            }
            Err(error) => {
                let code = error.code();
                tracing::info!(
                    request_id = %self.request_id,
                    model = %self.request.model,
                    step,
                    outcome = code.as_str(),
                    results = 0,
                    duration_ms,
                    "web search finished"
                );
                SearchRun {
                    payload: error_payload(Some(query), code),
                    sources: None,
                }
            }
        }
    }
}

/// The request copy the per-step state machines echo from: the original
/// request without its `input`, which they never read.
fn echo_request(request: &ResponsesRequest) -> ResponsesRequest {
    ResponsesRequest {
        model: request.model.clone(),
        input: ResponsesInput::Text(String::new()),
        instructions: request.instructions.clone(),
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_output_tokens: request.max_output_tokens,
        stream: request.stream,
        reasoning: request.reasoning.clone(),
        text: request.text.clone(),
        include: request.include.clone(),
        metadata: request.metadata.clone(),
        parallel_tool_calls: request.parallel_tool_calls,
        store: request.store,
        previous_response_id: request.previous_response_id.clone(),
        extra: request.extra.clone(),
    }
}

/// The loop's mutable state.
struct LoopState {
    /// The original input plus every search step so far, parsed.
    parsed: ParsedInput,
    /// The request each step is assembled from: the original with the
    /// remaining `max_output_tokens` and, after a search, `required` →
    /// `auto`.
    step_req: ResponsesRequest,
    registry: ResponsesToolRegistry,
    /// Bedrock calls made.
    step: usize,
    max_steps: usize,
    /// Real searches of this answer (from earlier requests too).
    used: u32,
    /// Real searches of this response.
    searches: u32,
    usage: UsageSum,
    /// The response's output items, in order.
    items: Vec<ResponseOutputItem>,
    /// What a citation may point at: the replayed carriers' results and this
    /// response's results.
    sources: CitationSources,
}

impl LoopState {
    async fn start(ctx: &Ctx<'_>, plan: &WebSearchPlan) -> Result<Self, AppError> {
        let parsed = parse_responses_input(
            ctx.request,
            ctx.resolved,
            ctx.provider.image_resolver(),
            ctx.opts(),
        )
        .await?;
        let max_calls = ctx.provider.web_search_limits().max_calls;
        let max_steps = usize::try_from(max_calls.saturating_add(2)).unwrap_or(usize::MAX);
        Ok(Self {
            parsed,
            step_req: ctx.request.clone(),
            registry: ResponsesToolRegistry::default(),
            step: 0,
            max_steps,
            used: plan.used_in_answer,
            searches: 0,
            usage: UsageSum::default(),
            items: Vec::new(),
            sources: plan.sources.clone(),
        })
    }

    fn assemble(
        &self,
        ctx: &Ctx<'_>,
        force_caching_off: bool,
    ) -> Result<AssembledConverse, AppError> {
        ctx.provider.assemble_parsed(
            &self.step_req,
            ctx.resolved,
            &self.parsed,
            force_caching_off,
            ctx.lane == Lane::ToolSearch,
            true,
        )
    }

    /// Assemble and send the next non-stream step; Converse-shaped output.
    async fn send_step(&mut self, ctx: &Ctx<'_>) -> Result<Value, AppError> {
        let assembled = self.assemble(ctx, false)?;
        self.registry = assembled.tool_registry.clone();
        let state: &Self = self;
        let model = ctx.request.model.as_str();
        match ctx.lane {
            Lane::Converse => {
                let normalized = normalize_for_match(ctx.resolved);
                send_with_cache_strip_retry(
                    ctx.provider.cache_support(),
                    &normalized,
                    assembled.cache_points_injected,
                    || ctx.transport.converse(model, &assembled),
                    || async {
                        let retry = state.assemble(ctx, true).map_err(SendError::App)?;
                        ctx.transport.converse(model, &retry).await
                    },
                )
                .await
            }
            Lane::ToolSearch => {
                let body = ctx.provider.messages_body(ctx.resolved, &assembled)?;
                let output = ctx.transport.invoke(model, body).await?;
                messages_response_to_converse_output(&output)
            }
        }
    }

    /// Assemble and send the next streamed step.
    async fn send_stream_step(&mut self, ctx: &Ctx<'_>) -> Result<StepEvents, AppError> {
        let assembled = self.assemble(ctx, false)?;
        self.registry = assembled.tool_registry.clone();
        let state: &Self = self;
        let model = ctx.request.model.as_str();
        match ctx.lane {
            Lane::Converse => {
                let normalized = normalize_for_match(ctx.resolved);
                send_with_cache_strip_retry(
                    ctx.provider.cache_support(),
                    &normalized,
                    assembled.cache_points_injected,
                    || ctx.transport.converse_stream(model, &assembled),
                    || async {
                        let retry = state.assemble(ctx, true).map_err(SendError::App)?;
                        ctx.transport.converse_stream(model, &retry).await
                    },
                )
                .await
            }
            Lane::ToolSearch => {
                let body = ctx.provider.messages_body(ctx.resolved, &assembled)?;
                ctx.transport.invoke_stream(model, body).await
            }
        }
    }

    /// Whether the step just made (`self.step` already counts it) was the
    /// last one allowed.
    fn at_step_limit(&self) -> bool {
        self.step >= self.max_steps
    }

    /// Record a real search: counts and citation sources.
    fn record_search(&mut self, run: &SearchRun) {
        if let Some(sources) = &run.sources {
            for source in sources {
                self.sources.insert(&source.url, source.title.as_deref());
            }
        }
    }

    /// The carrier of a search step and its item id.
    fn carrier(&self, ctx: &Ctx<'_>, uses: &[Value], results: &[Value]) -> (String, String) {
        let s = ctx.response_id.trim_start_matches("resp_");
        let id = format!("rs_{s}_ws{}", self.step - 1);
        (id, encode_web_search_carrier(uses, results, self.used))
    }

    /// Append a finished search step to the parsed input, as a client replays
    /// it: the step's reasoning and message items (`done_items`), then the
    /// carrier. Also set up the next step's request.
    ///
    /// Returns the reason the answer must stop instead, if any.
    async fn continue_after(
        &mut self,
        ctx: &Ctx<'_>,
        done_items: &[ResponseOutputItem],
        carrier_id: &str,
        carrier: &str,
    ) -> Result<Option<Exhaustion>, AppError> {
        let mut items = replay_items_for_step(done_items);
        items.push(ResponseInputItem::Reasoning {
            id: Some(carrier_id.to_string()),
            content: None,
            summary: Some(json!([])),
            encrypted_content: Some(carrier.to_string()),
        });
        let more = parse_items(
            &items,
            ctx.resolved,
            ctx.provider.image_resolver(),
            ctx.opts(),
        )
        .await?;
        self.parsed.append(more);

        if let Some(max) = ctx.request.max_output_tokens {
            let remaining = max.saturating_sub(self.usage.output);
            if remaining < MIN_CONTINUATION_TOKENS {
                return Ok(Some(Exhaustion::OutputBudget));
            }
            self.step_req.max_output_tokens = Some(remaining);
        }
        if matches!(
            &self.step_req.tool_choice,
            Some(ResponsesToolChoice::String(choice)) if choice == "required"
        ) {
            self.step_req.tool_choice = Some(ResponsesToolChoice::String("auto".to_string()));
        }
        Ok(None)
    }

    /// How the answer ends after `exhaustion`: `(message, status, details)`.
    /// With a real search in the answer, the gateway's message and
    /// `completed`; without one, a spent budget keeps the single-call meaning
    /// (`incomplete`, `max_output_tokens`) and there is no message.
    fn stop_item(
        &self,
        ctx: &Ctx<'_>,
        exhaustion: Exhaustion,
    ) -> (Option<ResponseOutputItem>, String, Option<Value>) {
        if self.used == 0 && exhaustion == Exhaustion::OutputBudget {
            return (
                None,
                "incomplete".to_string(),
                Some(json!({ "reason": "max_output_tokens" })),
            );
        }
        tracing::warn!(
            request_id = %ctx.request_id,
            model = %ctx.request.model,
            reason = exhaustion.as_str(),
            searches = self.used,
            "web search answer stopped by the gateway"
        );
        let s = ctx.response_id.trim_start_matches("resp_");
        let item = message_item(
            &format!("msg_{s}_stop"),
            &exhaustion.message(self.used),
            Vec::new(),
        );
        (Some(item), "completed".to_string(), None)
    }

    fn final_response(
        &self,
        ctx: &Ctx<'_>,
        status: String,
        incomplete_details: Option<Value>,
    ) -> ResponsesResponse {
        let mut output = self.items.clone();
        if output.is_empty() {
            // The output is never empty (the single-call rule).
            let ids = StepItemIds::for_step(ctx.response_id, 0);
            output.push(message_item(&ids.message, "", Vec::new()));
        }
        let request = ctx.request;
        ResponsesResponse {
            id: ctx.response_id.to_string(),
            object: "response".to_string(),
            created_at: created_at_now(),
            status,
            output,
            usage: usage_from_sum(&self.usage),
            model: request.model.clone(),
            instructions: request.instructions.clone(),
            temperature: request.temperature,
            top_p: request.top_p,
            tool_choice: request.tool_choice.clone(),
            tools: request.tools.clone(),
            max_output_tokens: request.max_output_tokens,
            parallel_tool_calls: request.parallel_tool_calls,
            error: None,
            incomplete_details,
        }
    }
}

/// The input items a client (zuno) replays for a step's reasoning and
/// message items: reasoning keeps its `encrypted_content` (an item without
/// one replays nothing), a message keeps its text (an empty one replays
/// nothing). `web_search_call` items are left out: the translation skips them.
pub(crate) fn replay_items_for_step(done_items: &[ResponseOutputItem]) -> Vec<ResponseInputItem> {
    let mut items = Vec::new();
    for item in done_items {
        match item {
            ResponseOutputItem::Reasoning {
                id,
                summary,
                encrypted_content: Some(encrypted),
                ..
            } => items.push(ResponseInputItem::Reasoning {
                id: Some(id.clone()),
                content: None,
                summary: summary.clone(),
                encrypted_content: Some(encrypted.clone()),
            }),
            ResponseOutputItem::Message { content, .. } => {
                let text: String = content
                    .iter()
                    .filter_map(|part| match part {
                        OutputContentPart::OutputText { text, .. } => Some(text.as_str()),
                        OutputContentPart::Refusal { .. } => None,
                    })
                    .collect();
                if !text.is_empty() {
                    items.push(ResponseInputItem::Message {
                        role: ResponsesRole::Assistant,
                        content: ResponsesContent::Parts(vec![ResponseContentPart::OutputText {
                            text,
                        }]),
                    });
                }
            }
            _ => {}
        }
    }
    items
}

/// Serve a declared request without streaming.
///
/// # Errors
/// Any step's error, mapped as on the single-call path.
pub(crate) async fn respond(
    provider: &BedrockResponsesProvider,
    transport: &dyn StepTransport,
    req: &NormalizedResponsesRequest,
    plan: WebSearchPlan,
    lane: Lane,
) -> Result<ResponsesResponse, AppError> {
    let response_id = resp_id();
    let ctx = Ctx {
        provider,
        transport,
        request: &req.request,
        resolved: &req.resolved_model,
        request_id: &req.request_id,
        lane,
        gate: plan.gate,
        response_id: &response_id,
    };
    let mut state = LoopState::start(&ctx, &plan).await?;
    let max_calls = provider.web_search_limits().max_calls;
    let (status, incomplete_details) = loop {
        let output = state.send_step(&ctx).await?;
        let ids = StepItemIds::for_step(&response_id, state.step);
        let step = state.step;
        let mapped = map_step(&output, &ids, true)?;
        state
            .usage
            .add_step(mapped.usage.as_ref(), mapped.reasoning_tokens);
        state.step += 1;

        let mut done_items = Vec::new();
        if let Some(reasoning) = &mapped.reasoning {
            done_items.push(reasoning.clone());
        }
        if !mapped.text.is_empty() {
            let annotations = url_citations(&mapped.text, &state.sources);
            done_items.push(message_item(&ids.message, &mapped.text, annotations));
        }
        state.items.extend(done_items.iter().cloned());

        let kind = classify(
            mapped.stop_reason.as_deref(),
            mapped.tool_uses.iter().map(|call| call.name.as_str()),
            &state.registry,
        );
        let client_calls: Vec<&StepToolUse> = mapped
            .tool_uses
            .iter()
            .filter(|call| !is_web_search(&state.registry, &call.name))
            .collect();
        if kind == StepKind::Final {
            if mapped.stop_reason.as_deref() == Some("max_tokens") && state.used > 0 {
                // The answer ends `completed`, so a call cut off by
                // `max_tokens` must not be handed back: the client would run
                // it. (Under `incomplete` the single-call path lists such
                // calls; the client does not run them.)
                let (item, status, details) = state.stop_item(&ctx, Exhaustion::OutputBudget);
                state.items.extend(item);
                break (status, details);
            }
            for call in &client_calls {
                state.items.push(client_call_item(&state.registry, call)?);
            }
            break step_status(mapped.stop_reason.as_deref());
        }

        let at_limit = kind == StepKind::SearchOnly && state.at_step_limit();
        let mut uses = Vec::new();
        let mut results = Vec::new();
        let searches: Vec<SearchCall> = mapped
            .tool_uses
            .iter()
            .filter(|call| is_web_search(&state.registry, &call.name))
            .map(|call| SearchCall::new(&call.call_id, &call.input))
            .collect();
        for call in searches {
            let payload = match plan_call(ctx.gate, state.used, max_calls, !at_limit, &call.input) {
                CallPlan::Refuse(code, query) => error_payload(query.as_deref(), code),
                CallPlan::Search(query) => {
                    state.used += 1;
                    state.searches += 1;
                    let run = ctx.run_search(step, &query).await;
                    state.record_search(&run);
                    state.items.push(web_search_call_item(
                        &web_search_call_id(&call.call_id),
                        run.status(),
                        &query,
                        run.sources.as_deref(),
                    ));
                    run.payload
                }
            };
            uses.push(web_search_tool_use_block(&call.call_id, &call.input));
            results.push(tool_result_block(&call.call_id, &payload));
        }
        if kind == StepKind::Mixed {
            for call in &client_calls {
                state.items.push(client_call_item(&state.registry, call)?);
            }
        }
        let (carrier_id, carrier) = state.carrier(&ctx, &uses, &results);
        state.items.push(carrier_item(&carrier_id, Some(&carrier)));
        if kind == StepKind::Mixed {
            break ("completed".to_string(), None);
        }
        let exhaustion = if at_limit {
            Some(Exhaustion::StepLimit)
        } else {
            state
                .continue_after(&ctx, &done_items, &carrier_id, &carrier)
                .await?
        };
        if let Some(exhaustion) = exhaustion {
            let (item, status, details) = state.stop_item(&ctx, exhaustion);
            state.items.extend(item);
            break (status, details);
        }
    };
    let response = state.final_response(&ctx, status, incomplete_details);
    tracing::info!(
        request_id = %req.request_id,
        model = %req.request.model,
        status = %response.status,
        web_search_calls = state.searches,
        web_search_steps = state.step,
        "responses web search loop finished"
    );
    Ok(response)
}

/// A client call of a non-stream step, mapped as the single-call path maps it.
fn client_call_item(
    registry: &ResponsesToolRegistry,
    call: &StepToolUse,
) -> Result<ResponseOutputItem, AppError> {
    let arguments = serde_json::to_string(&call.input)
        .map_err(|e| AppError::Internal(format!("failed to serialize toolUse input: {e}")))?;
    let status = registry
        .resolve(&call.name)
        .filter(|binding| binding.kind != ResponsesToolKind::Function)
        .map(|_| "completed");
    Ok(tool_output_item(
        registry,
        &call.call_id,
        &call.name,
        &call.input,
        arguments,
        status,
    ))
}

/// Sequence and output-index counters shared by every event of the response.
struct Counters {
    seq: u64,
    next_output_index: u32,
}

impl Counters {
    fn seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    fn output_index(&mut self) -> u32 {
        let i = self.next_output_index;
        self.next_output_index += 1;
        i
    }
}

/// A `response.web_search_call.*` progress event.
fn search_progress(kind: &str, item_id: &str, output_index: u32, seq: u64) -> ResponseStreamEvent {
    let mut fields = std::collections::HashMap::new();
    fields.insert("item_id".to_string(), json!(item_id));
    fields.insert("output_index".to_string(), json!(output_index));
    fields.insert("sequence_number".to_string(), json!(seq));
    ResponseStreamEvent::Other {
        event_type: format!("response.web_search_call.{kind}"),
        fields,
    }
}

/// The events of the gateway's own message item, streamed as one delta.
fn message_events(item: &ResponseOutputItem, counters: &mut Counters) -> Vec<ResponseStreamEvent> {
    let ResponseOutputItem::Message { id, content, .. } = item else {
        return Vec::new();
    };
    let text = content
        .iter()
        .find_map(|part| match part {
            OutputContentPart::OutputText { text, .. } => Some(text.clone()),
            OutputContentPart::Refusal { .. } => None,
        })
        .unwrap_or_default();
    let output_index = counters.output_index();
    let empty_part = OutputContentPart::OutputText {
        text: String::new(),
        annotations: Vec::new(),
        logprobs: None,
    };
    let part = OutputContentPart::OutputText {
        text: text.clone(),
        annotations: Vec::new(),
        logprobs: None,
    };
    vec![
        ResponseStreamEvent::OutputItemAdded {
            item: ResponseOutputItem::Message {
                id: id.clone(),
                status: "in_progress".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
            },
            output_index,
            sequence_number: counters.seq(),
        },
        ResponseStreamEvent::ContentPartAdded {
            item_id: id.clone(),
            output_index,
            content_index: 0,
            part: empty_part,
            sequence_number: counters.seq(),
        },
        ResponseStreamEvent::OutputTextDelta {
            item_id: id.clone(),
            output_index,
            content_index: 0,
            delta: text.clone(),
            sequence_number: counters.seq(),
        },
        ResponseStreamEvent::OutputTextDone {
            item_id: id.clone(),
            output_index,
            content_index: 0,
            text,
            sequence_number: counters.seq(),
        },
        ResponseStreamEvent::ContentPartDone {
            item_id: id.clone(),
            output_index,
            content_index: 0,
            part,
            sequence_number: counters.seq(),
        },
        ResponseStreamEvent::OutputItemDone {
            item: item.clone(),
            output_index,
            sequence_number: counters.seq(),
        },
    ]
}

/// The events of the client calls of a stream step, in block order, each
/// with the next output index: announced (`tool_search_call` excepted, which
/// zuno only accepts as `output_item.done`), completed, done.
fn client_call_events(
    registry: &ResponsesToolRegistry,
    calls: Vec<ToolAccum>,
    counters: &mut Counters,
) -> Vec<ResponseStreamEvent> {
    let mut out = Vec::new();
    for mut accum in calls {
        accum.output_index = counters.output_index();
        let announce = registry
            .resolve(&accum.name)
            .is_none_or(|binding| binding.kind != ResponsesToolKind::ToolSearch);
        out.extend(tool_events(registry, &accum, announce, &mut counters.seq));
    }
    out
}

/// `response.failed` for an error between two steps (no step's state machine
/// is running).
fn failure_events(
    req: &NormalizedResponsesRequest,
    response_id: &str,
    echo: &ResponsesRequest,
    start: Instant,
    seq: u64,
    err: &AppError,
    code: &str,
) -> Vec<ResponseStreamEvent> {
    let mut state = ResponsesStreamState::new_step(
        req.request.model.clone(),
        response_id.to_string(),
        echo.clone(),
        Arc::clone(&req.request_id),
        start,
        ResponsesToolRegistry::default(),
        StepOptions {
            seq,
            started: true,
            ..StepOptions::default()
        },
    );
    state.fail_with_code(err, code)
}

/// Push every `output_item.done` item of `events` onto `items`.
fn record_done(items: &mut Vec<ResponseOutputItem>, events: &[ResponseStreamEvent]) {
    for event in events {
        if let ResponseStreamEvent::OutputItemDone { item, .. } = event {
            items.push(item.clone());
        }
    }
}

/// Serve a declared request as a stream.
///
/// The first step is sent before the stream is returned, so its rejection is
/// an HTTP error, as on the single-call path. Later failures end the stream
/// with `response.failed` ([`loop_failure_code`]).
///
/// # Errors
/// The input translation's error, or the first step's send error.
pub(crate) async fn respond_stream(
    provider: BedrockResponsesProvider,
    transport: Arc<dyn StepTransport>,
    req: NormalizedResponsesRequest,
    plan: WebSearchPlan,
    lane: Lane,
    runtime: ResponsesStreamRuntime,
) -> Result<ResponsesStream, AppError> {
    let response_id = resp_id();
    let (mut state, first) = {
        let ctx = Ctx {
            provider: &provider,
            transport: transport.as_ref(),
            request: &req.request,
            resolved: &req.resolved_model,
            request_id: &req.request_id,
            lane,
            gate: plan.gate,
            response_id: &response_id,
        };
        let mut state = LoopState::start(&ctx, &plan).await?;
        let first = state.send_stream_step(&ctx).await?;
        (state, first)
    };
    let gate = plan.gate;
    let s = stream! {
        let ctx = Ctx {
            provider: &provider,
            transport: transport.as_ref(),
            request: &req.request,
            resolved: &req.resolved_model,
            request_id: &req.request_id,
            lane,
            gate,
            response_id: &response_id,
        };
        let echo = echo_request(&req.request);
        let max_calls = provider.web_search_limits().max_calls;
        let mut counters = Counters { seq: 0, next_output_index: 0 };
        let mut started = false;
        let mut next_events = Some(first);
        let (status, incomplete_details) = 'response: loop {
            let step = state.step;
            let mut events = match next_events.take() {
                Some(events) => events,
                None => match state.send_stream_step(&ctx).await {
                    Ok(events) => events,
                    Err(err) => {
                        tracing::error!(
                            request_id = %req.request_id,
                            model = %req.request.model,
                            step,
                            error = %err,
                            "responses web search step failed"
                        );
                        let code = loop_failure_code(step, &err);
                        let start = runtime.start;
                        for ev in failure_events(&req, &response_id, &echo, start, counters.seq, &err, code) {
                            yield Ok(ev);
                        }
                        return;
                    }
                },
            };
            let ids = StepItemIds::for_step(&response_id, step);
            let mut machine = ResponsesStreamState::new_step(
                req.request.model.clone(),
                response_id.clone(),
                echo.clone(),
                Arc::clone(&req.request_id),
                runtime.start,
                state.registry.clone(),
                StepOptions {
                    ids: Some(ids),
                    mark_envelopes: true,
                    defer_tool_events: true,
                    step_mode: true,
                    seq: counters.seq,
                    next_output_index: counters.next_output_index,
                    started,
                    citations: Some(Arc::new(state.sources.clone())),
                },
            );
            // The step's reasoning and message items, in done order.
            let mut done_items: Vec<ResponseOutputItem> = Vec::new();
            let mut failure: Option<AppError> = None;
            loop {
                match tokio::time::timeout(runtime.idle_timeout, events.next()).await {
                    Ok(Some(Ok(event))) => {
                        for ev in machine.map_event(&event) {
                            if let ResponseStreamEvent::OutputItemDone { item, .. } = &ev {
                                done_items.push(item.clone());
                            }
                            yield Ok(ev);
                        }
                        if machine.step_complete() {
                            break;
                        }
                    }
                    Ok(Some(Err(err))) => {
                        failure = Some(err);
                        break;
                    }
                    Ok(None) => break,
                    Err(_) => {
                        tracing::error!(
                            request_id = %req.request_id,
                            model = %req.request.model,
                            step,
                            last_bedrock_event = machine.last_bedrock_event(),
                            idle_timeout_secs = runtime.idle_timeout.as_secs(),
                            "responses upstream stream idle timeout"
                        );
                        failure = Some(AppError::UpstreamBedrock(
                            "Upstream response stream timed out while idle.".to_string(),
                        ));
                        break;
                    }
                }
            }
            drop(events);
            if let Some(err) = failure {
                let (tool_blocks, tool_argument_bytes) = machine.tool_progress_counts();
                tracing::error!(
                    request_id = %req.request_id,
                    model = %req.request.model,
                    step,
                    last_bedrock_event = machine.last_bedrock_event(),
                    tool_blocks,
                    tool_argument_bytes,
                    error = %err,
                    "responses upstream stream failed"
                );
                for ev in machine.fail_with_code(&err, loop_failure_code(step, &err)) {
                    yield Ok(ev);
                }
                return;
            }
            let end = machine.take_step();
            record_done(&mut done_items, &end.events);
            for ev in end.events {
                yield Ok(ev);
            }
            counters.seq = end.seq;
            counters.next_output_index = end.next_output_index;
            started = true;
            state.usage.add_step(end.usage.as_ref(), end.reasoning_tokens);
            state.step += 1;
            state.items.extend(done_items.iter().cloned());

            let incomplete_stop = matches!(
                end.stop_reason.as_deref(),
                Some("max_tokens" | "content_filtered")
            );
            if !incomplete_stop && !end.tools_valid {
                tracing::error!(
                    request_id = %req.request_id,
                    model = %req.request.model,
                    step,
                    stop_reason = end.stop_reason.as_deref().unwrap_or("missing"),
                    tool_blocks = end.tools.len(),
                    "responses upstream ended with invalid tool-call JSON"
                );
                let err = AppError::Internal(
                    "Upstream returned malformed tool-call arguments.".to_string(),
                );
                let code = loop_failure_code(step, &err);
                let start = runtime.start;
                for ev in failure_events(&req, &response_id, &echo, start, counters.seq, &err, code) {
                    yield Ok(ev);
                }
                return;
            }

            let kind = classify(
                end.stop_reason.as_deref(),
                end.tools.iter().map(|accum| accum.name.as_str()),
                &state.registry,
            );
            let client_calls: Vec<ToolAccum> = end
                .tools
                .iter()
                .filter(|accum| !is_web_search(&state.registry, &accum.name))
                .cloned()
                .collect();
            if kind == StepKind::Final {
                // A call cut off by max_tokens or filtered is never executable.
                if !incomplete_stop {
                    let events = client_call_events(&state.registry, client_calls, &mut counters);
                    record_done(&mut state.items, &events);
                    for ev in events {
                        yield Ok(ev);
                    }
                }
                if end.stop_reason.as_deref() == Some("max_tokens") && state.used > 0 {
                    let (item, status, details) = state.stop_item(&ctx, Exhaustion::OutputBudget);
                    if let Some(item) = item {
                        for ev in message_events(&item, &mut counters) {
                            yield Ok(ev);
                        }
                        state.items.push(item);
                    }
                    break 'response (status, details);
                }
                break 'response step_status(end.stop_reason.as_deref());
            }

            let at_limit = kind == StepKind::SearchOnly && state.at_step_limit();
            let mut uses = Vec::new();
            let mut results = Vec::new();
            let searches: Vec<SearchCall> = end
                .tools
                .iter()
                .filter(|accum| is_web_search(&state.registry, &accum.name))
                .map(|accum| SearchCall::new(&accum.call_id, &accum.input()))
                .collect();
            for call in searches {
                let plan = plan_call(ctx.gate, state.used, max_calls, !at_limit, &call.input);
                let payload = match plan {
                    CallPlan::Refuse(code, query) => error_payload(query.as_deref(), code),
                    CallPlan::Search(query) => {
                        state.used += 1;
                        state.searches += 1;
                        let item_id = web_search_call_id(&call.call_id);
                        let output_index = counters.output_index();
                        yield Ok(ResponseStreamEvent::OutputItemAdded {
                            item: web_search_call_item(&item_id, "in_progress", &query, None),
                            output_index,
                            sequence_number: counters.seq(),
                        });
                        for kind in ["in_progress", "searching"] {
                            yield Ok(search_progress(kind, &item_id, output_index, counters.seq()));
                        }
                        let run = ctx.run_search(step, &query).await;
                        state.record_search(&run);
                        // A failed search has no `completed` event.
                        if run.sources.is_some() {
                            let seq = counters.seq();
                            yield Ok(search_progress("completed", &item_id, output_index, seq));
                        }
                        let item = web_search_call_item(
                            &item_id,
                            run.status(),
                            &query,
                            run.sources.as_deref(),
                        );
                        state.items.push(item.clone());
                        yield Ok(ResponseStreamEvent::OutputItemDone {
                            item,
                            output_index,
                            sequence_number: counters.seq(),
                        });
                        run.payload
                    }
                };
                uses.push(web_search_tool_use_block(&call.call_id, &call.input));
                results.push(tool_result_block(&call.call_id, &payload));
            }
            if kind == StepKind::Mixed {
                let events = client_call_events(&state.registry, client_calls, &mut counters);
                record_done(&mut state.items, &events);
                for ev in events {
                    yield Ok(ev);
                }
            }
            let (carrier_id, carrier) = state.carrier(&ctx, &uses, &results);
            let output_index = counters.output_index();
            yield Ok(ResponseStreamEvent::OutputItemAdded {
                item: carrier_item(&carrier_id, None),
                output_index,
                sequence_number: counters.seq(),
            });
            let item = carrier_item(&carrier_id, Some(&carrier));
            state.items.push(item.clone());
            yield Ok(ResponseStreamEvent::OutputItemDone {
                item,
                output_index,
                sequence_number: counters.seq(),
            });
            if kind == StepKind::Mixed {
                break 'response ("completed".to_string(), None);
            }
            let exhaustion = if at_limit {
                Some(Exhaustion::StepLimit)
            } else {
                match state.continue_after(&ctx, &done_items, &carrier_id, &carrier).await {
                    Ok(exhaustion) => exhaustion,
                    Err(err) => {
                        let code = loop_failure_code(state.step, &err);
                        let start = runtime.start;
                        for ev in failure_events(&req, &response_id, &echo, start, counters.seq, &err, code) {
                            yield Ok(ev);
                        }
                        return;
                    }
                }
            };
            if let Some(exhaustion) = exhaustion {
                let (item, status, details) = state.stop_item(&ctx, exhaustion);
                if let Some(item) = item {
                    for ev in message_events(&item, &mut counters) {
                        yield Ok(ev);
                    }
                    state.items.push(item);
                }
                break 'response (status, details);
            }
        };
        let response = state.final_response(&ctx, status, incomplete_details);
        let cached_tokens = response
            .usage
            .input_tokens_details
            .as_ref()
            .map_or(0, |details| details.cached_tokens);
        let reasoning_tokens = response.usage.reasoning_tokens();
        tracing::info!(
            request_id = %req.request_id,
            model = %req.request.model,
            status = %response.status,
            reasoning_effort = %req.request.reasoning_effort_label(),
            reasoning_used = reasoning_tokens > 0,
            reasoning_tokens,
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            total_tokens = response.usage.total_tokens,
            cached_tokens,
            cache_hit = cached_tokens > 0,
            web_search_calls = state.searches,
            web_search_steps = state.step,
            duration_ms = runtime.start.elapsed().as_millis(),
            "responses streaming completed"
        );
        let sequence_number = counters.seq();
        if response.status == "incomplete" {
            yield Ok(ResponseStreamEvent::Incomplete { response, sequence_number });
        } else {
            yield Ok(ResponseStreamEvent::Completed { response, sequence_number });
        }
    };
    Ok(s.boxed())
}

#[cfg(test)]
#[path = "responses_web_search_tests.rs"]
mod tests;
