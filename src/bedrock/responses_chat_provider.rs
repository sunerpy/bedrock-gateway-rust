//! OpenAI Responses API to Chat Completions protocol adapter.
//!
//! Models selecting `chat_backend = "responses"` are invoked through the
//! existing [`ResponsesProvider`] and mapped back to the standard Chat wire
//! shape. The adapter is stateless: signed Responses reasoning items needed for
//! a tool continuation are carried in an HMAC-authenticated `rsc_v1` tool-call
//! id and replayed when the client returns that id.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{json, Map, Value};

use crate::bedrock::capsule::{
    decode_responses_capsule, encode_responses_capsule, is_responses_capsule,
    responses_reasoning_item_is_valid, CapsuleRuntime,
};
use crate::domain::{
    gen_request_id, ChatProvider, ChatStream, NormalizedChatRequest, NormalizedResponsesRequest,
    ResponsesProvider,
};
use crate::error::AppError;
use crate::openai::responses_schema::{
    FunctionCallOutputValue, ReasoningConfig, ResponseContentPart, ResponseInputItem,
    ResponseOutputItem, ResponsesContent, ResponsesInput, ResponsesRequest, ResponsesRole,
    ResponsesTool, ResponsesToolChoice, TextConfig,
};
use crate::openai::schema::{
    ChatRequest, ChatResponse, ChatResponseMessage, ChatStreamResponse, Choice, ChoiceDelta,
    CompletionTokensDetails, ContentInput, ContentPart, Message, PromptTokensDetails,
    ResponseFormat, ResponseFunction, SystemContentInput, ToolCall, ToolChoice, ToolContentInput,
    Usage,
};

const MAX_SSE_FRAME_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub struct ResponsesChatProvider {
    responses: Arc<dyn ResponsesProvider>,
    capsule: Arc<CapsuleRuntime>,
}

impl ResponsesChatProvider {
    #[must_use]
    pub fn new(responses: Arc<dyn ResponsesProvider>, capsule: Arc<CapsuleRuntime>) -> Self {
        Self { responses, capsule }
    }

    fn normalize(
        &self,
        req: &NormalizedChatRequest,
        stream: bool,
    ) -> Result<NormalizedResponsesRequest, AppError> {
        let request = chat_request_to_responses(&req.request, &self.capsule, stream)?;
        let raw_body = serde_json::to_vec(&request)
            .map(Bytes::from)
            .map_err(|error| {
                AppError::Internal(format!(
                    "failed to serialize adapted Responses request: {error}"
                ))
            })?;
        Ok(NormalizedResponsesRequest {
            request,
            resolved_model: req.resolved_model.clone(),
            request_id: Arc::clone(&req.request_id),
            received_at: req.received_at,
            raw_body,
        })
    }
}

#[async_trait::async_trait]
impl ChatProvider for ResponsesChatProvider {
    async fn chat(&self, req: &NormalizedChatRequest) -> Result<ChatResponse, AppError> {
        let normalized = self.normalize(req, false)?;
        let response = self.responses.respond(&normalized).await?;
        responses_to_chat(response, &self.capsule)
    }

    async fn chat_stream(&self, req: &NormalizedChatRequest) -> Result<ChatStream, AppError> {
        let normalized = self.normalize(req, true)?;
        let Some(raw) = self.responses.respond_raw_stream(&normalized).await? else {
            return match self.responses.respond_stream(&normalized).await {
                Err(error) => Err(error),
                Ok(_) => Err(AppError::Internal(
                    "Responses-backed chat requires the raw Responses SSE lane".to_string(),
                )),
            };
        };

        let model = req.request.model.clone();
        let message_id = new_chat_completion_id();
        let request_id = Arc::clone(&req.request_id);
        let include_usage = req
            .request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage);
        let capsule = Arc::clone(&self.capsule);
        let output = stream! {
            let mut raw = raw;
            let mut decoder = SseDecoder::default();
            let mut state = ResponsesChatStreamState::new(
                request_id,
                message_id,
                model,
                include_usage,
                capsule,
            );
            yield Ok(state.role_chunk());

            let mut failed = false;
            while let Some(item) = raw.next().await {
                let bytes = match item {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        state.adapter_failed = true;
                        yield Err(error);
                        failed = true;
                        break;
                    }
                };
                let events = match decoder.push(&bytes) {
                    Ok(events) => events,
                    Err(error) => {
                        state.adapter_failed = true;
                        yield Err(error);
                        failed = true;
                        break;
                    }
                };
                for event in events {
                    match state.map_event(&event) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                        Err(error) => {
                            state.adapter_failed = true;
                            yield Err(error);
                            failed = true;
                            break;
                        }
                    }
                }
                if failed || state.terminal_seen {
                    break;
                }
            }

            if !failed && !state.terminal_seen {
                match decoder.finish() {
                    Ok(events) => {
                        for event in events {
                            match state.map_event(&event) {
                                Ok(chunks) => {
                                    for chunk in chunks {
                                        yield Ok(chunk);
                                    }
                                }
                                Err(error) => {
                                    state.adapter_failed = true;
                                    yield Err(error);
                                    failed = true;
                                    break;
                                }
                            }
                        }
                    }
                    Err(error) => {
                        state.adapter_failed = true;
                        yield Err(error);
                        failed = true;
                    }
                }
            }
            if !failed && !state.terminal_seen {
                state.adapter_failed = true;
                yield Err(AppError::UpstreamBedrock(
                    "Responses stream ended without a terminal event".to_string(),
                ));
            }
            if !failed && state.terminal_seen {
                state.adapter_finished = true;
                tracing::info!(
                    request_id = %state.request_id,
                    completion_id = %state.message_id,
                    model = %state.model,
                    finish_reason = state.finish_reason.unwrap_or("unknown"),
                    tool_calls = state.tool_indices.len(),
                    terminal_event = state.terminal_event.unwrap_or("unknown"),
                    output_item_types = %type_set_label(&state.output_item_types),
                    unknown_output_item_types = %type_set_label(&state.unknown_output_item_types),
                    visible_text_bytes = state.visible_text_bytes,
                    "responses chat streaming completed"
                );
            }
        };
        Ok(output.boxed())
    }
}

fn chat_request_to_responses(
    req: &ChatRequest,
    capsule: &CapsuleRuntime,
    stream: bool,
) -> Result<ResponsesRequest, AppError> {
    if req.n.is_some_and(|n| n != 1) {
        return Err(AppError::BadRequest(
            "Responses-backed chat supports only n=1".to_string(),
        ));
    }
    if req.stop.is_some() {
        return Err(AppError::BadRequest(
            "stop is not supported by Responses-backed chat models".to_string(),
        ));
    }

    let mut input = Vec::new();
    for message in &req.messages {
        append_message_items(message, capsule, &mut input)?;
    }
    dedupe_replayed_reasoning_items(&mut input)?;

    let tools = req
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| {
                    if tool.r#type != "function" {
                        return Err(AppError::BadRequest(format!(
                            "unsupported chat tool type `{}`",
                            tool.r#type
                        )));
                    }
                    Ok(ResponsesTool::Function {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: Some(tool.function.parameters.clone()),
                        strict: None,
                    })
                })
                .collect::<Result<Vec<_>, AppError>>()
        })
        .transpose()?;

    let mut extra = req.extra.clone();
    if let Some(Value::Object(extra_body)) = &req.extra_body {
        for (key, value) in extra_body {
            extra.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    if let Some(value) = req.frequency_penalty {
        extra.insert("frequency_penalty".to_string(), json!(value));
    }
    if let Some(value) = req.presence_penalty {
        extra.insert("presence_penalty".to_string(), json!(value));
    }
    if let Some(value) = &req.user {
        extra.insert("user".to_string(), json!(value));
    }
    let parallel_tool_calls = extra
        .remove("parallel_tool_calls")
        .and_then(|value| value.as_bool());

    let reasoning = req.reasoning_effort.map(|effort| ReasoningConfig {
        effort: Some(
            serde_json::to_value(effort)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_else(|| "medium".to_string()),
        ),
        summary: Some("auto".to_string()),
    });

    Ok(ResponsesRequest {
        model: req.model.clone(),
        input: ResponsesInput::Items(input),
        instructions: None,
        tools,
        tool_choice: Some(chat_tool_choice_to_responses(&req.tool_choice)),
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_completion_tokens.or(req.max_tokens),
        stream: Some(stream),
        reasoning,
        text: response_format_to_text_config(req.response_format.as_ref())?,
        include: Some(vec!["reasoning.encrypted_content".to_string()]),
        metadata: None,
        parallel_tool_calls,
        store: Some(false),
        previous_response_id: None,
        extra,
    })
}

fn dedupe_replayed_reasoning_items(input: &mut Vec<ResponseInputItem>) -> Result<(), AppError> {
    let mut seen = HashMap::<String, usize>::new();
    let mut deduplicated: Vec<ResponseInputItem> = Vec::with_capacity(input.len());
    for item in input.drain(..) {
        let ResponseInputItem::Reasoning { id, .. } = &item else {
            deduplicated.push(item);
            continue;
        };
        if let Some(previous_index) = seen.get(id) {
            if deduplicated[*previous_index] != item {
                return Err(AppError::BadRequest(
                    "duplicate Responses reasoning item id has conflicting payloads".to_string(),
                ));
            }
            continue;
        }
        seen.insert(id.clone(), deduplicated.len());
        deduplicated.push(item);
    }
    *input = deduplicated;
    Ok(())
}

fn append_message_items(
    message: &Message,
    capsule: &CapsuleRuntime,
    output: &mut Vec<ResponseInputItem>,
) -> Result<(), AppError> {
    match message {
        Message::System { content, .. } => output.push(ResponseInputItem::Message {
            role: ResponsesRole::System,
            content: system_content_to_responses(content)?,
        }),
        Message::Developer { content, .. } => output.push(ResponseInputItem::Message {
            role: ResponsesRole::Developer,
            content: system_content_to_responses(content)?,
        }),
        Message::User { content, .. } => output.push(ResponseInputItem::Message {
            role: ResponsesRole::User,
            content: content_to_responses(content),
        }),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => append_assistant_items(content, tool_calls.as_deref(), capsule, output)?,
        Message::Tool {
            content,
            tool_call_id,
        } => {
            let call_id = if is_responses_capsule(tool_call_id) {
                decode_responses_capsule(tool_call_id, &capsule.keyring)?.call_id
            } else {
                tool_call_id.clone()
            };
            output.push(ResponseInputItem::FunctionCallOutput {
                call_id,
                output: FunctionCallOutputValue::Text(tool_content_to_string(content)?),
            });
        }
    }
    Ok(())
}

fn append_assistant_items(
    content: &Option<ContentInput>,
    tool_calls: Option<&[ToolCall]>,
    capsule: &CapsuleRuntime,
    output: &mut Vec<ResponseInputItem>,
) -> Result<(), AppError> {
    let mut calls = Vec::new();
    let mut shared_reasoning: Option<Vec<Value>> = None;
    let mut raw_id_seen = false;
    if let Some(tool_calls) = tool_calls {
        for call in tool_calls {
            let id = call.id.clone().ok_or_else(|| {
                AppError::BadRequest("assistant tool call is missing id".to_string())
            })?;
            let call_id = if is_responses_capsule(&id) {
                let decoded = decode_responses_capsule(&id, &capsule.keyring)?;
                if shared_reasoning
                    .as_ref()
                    .is_some_and(|items| items != &decoded.reasoning_items)
                {
                    return Err(AppError::BadRequest(
                        "parallel Responses reasoning capsules carry different items".to_string(),
                    ));
                }
                if shared_reasoning.is_none() {
                    shared_reasoning = Some(decoded.reasoning_items);
                }
                decoded.call_id
            } else {
                raw_id_seen = true;
                id
            };
            calls.push(ResponseInputItem::FunctionCall {
                call_id,
                name: call.function.name.clone().ok_or_else(|| {
                    AppError::BadRequest("assistant tool call is missing function name".to_string())
                })?,
                arguments: call.function.arguments.clone(),
                namespace: None,
            });
        }
    }
    if raw_id_seen && shared_reasoning.is_some() {
        return Err(AppError::BadRequest(
            "assistant tool calls mix raw ids and Responses reasoning capsules".to_string(),
        ));
    }

    let replayed_content = if let Some(items) = &shared_reasoning {
        for item in items {
            output.push(serde_json::from_value(item.clone()).map_err(|_| {
                AppError::BadRequest(
                    "Responses reasoning capsule contains an invalid reasoning item".to_string(),
                )
            })?);
        }
        strip_replayed_summary(content, items)?
    } else {
        content.clone()
    };
    if content_has_output(&replayed_content) {
        output.push(ResponseInputItem::Message {
            role: ResponsesRole::Assistant,
            content: assistant_content_to_responses(
                replayed_content
                    .as_ref()
                    .expect("content_has_output requires content"),
            ),
        });
    }
    output.extend(calls);
    Ok(())
}

fn strip_replayed_summary(
    content: &Option<ContentInput>,
    reasoning_items: &[Value],
) -> Result<Option<ContentInput>, AppError> {
    let summary = reasoning_summary(reasoning_items);
    if summary.is_empty() || content.is_none() {
        return Ok(content.clone());
    }
    let prefix = format!("<think>{summary}</think>");
    match content {
        Some(ContentInput::Text(text)) => text
            .strip_prefix(&prefix)
            .map(|rest| Some(ContentInput::Text(rest.to_string())))
            .ok_or_else(|| {
                AppError::BadRequest(
                    "assistant reasoning prefix does not match the Responses capsule".to_string(),
                )
            }),
        Some(ContentInput::Parts(parts)) => {
            let mut parts = parts.clone();
            let Some(ContentPart::Text(first)) = parts.first_mut() else {
                return Err(AppError::BadRequest(
                    "assistant content parts have no leading reasoning text".to_string(),
                ));
            };
            first.text = first
                .text
                .strip_prefix(&prefix)
                .map(str::to_string)
                .ok_or_else(|| {
                    AppError::BadRequest(
                        "assistant reasoning prefix does not match the Responses capsule"
                            .to_string(),
                    )
                })?;
            Ok(Some(ContentInput::Parts(parts)))
        }
        None => Ok(None),
    }
}

fn content_has_output(content: &Option<ContentInput>) -> bool {
    match content {
        Some(ContentInput::Text(text)) => !text.is_empty(),
        Some(ContentInput::Parts(parts)) => !parts.is_empty(),
        None => false,
    }
}

fn content_to_responses(content: &ContentInput) -> ResponsesContent {
    match content {
        ContentInput::Text(text) => ResponsesContent::Text(text.clone()),
        ContentInput::Parts(parts) => ResponsesContent::Parts(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => ResponseContentPart::InputText {
                        text: text.text.clone(),
                    },
                    ContentPart::Image(image) => ResponseContentPart::InputImage {
                        image_url: image.image_url.url.clone(),
                        detail: Some(image.image_url.detail.clone()),
                    },
                })
                .collect(),
        ),
    }
}

fn assistant_content_to_responses(content: &ContentInput) -> ResponsesContent {
    if let ContentInput::Parts(parts) = content {
        let text_only = parts
            .iter()
            .map(|part| match part {
                ContentPart::Text(text) => Some(text.text.as_str()),
                ContentPart::Image(_) => None,
            })
            .collect::<Option<String>>();
        if let Some(text) = text_only {
            // Mantle rejects the text-only input_text array accepted by OpenAI.
            return ResponsesContent::Text(text);
        }
    }
    content_to_responses(content)
}

fn system_content_to_responses(content: &SystemContentInput) -> Result<ResponsesContent, AppError> {
    match content {
        SystemContentInput::Text(text) => Ok(ResponsesContent::Text(text.clone())),
        SystemContentInput::Parts(parts) => {
            let mut output = Vec::with_capacity(parts.len());
            for part in parts {
                let ContentPart::Text(text) = part else {
                    return Err(AppError::BadRequest(
                        "system and developer messages cannot contain images".to_string(),
                    ));
                };
                output.push(ResponseContentPart::InputText {
                    text: text.text.clone(),
                });
            }
            Ok(ResponsesContent::Parts(output))
        }
    }
}

fn tool_content_to_string(content: &ToolContentInput) -> Result<String, AppError> {
    match content {
        ToolContentInput::Text(text) => Ok(text.clone()),
        ToolContentInput::Parts(parts) => serde_json::to_string(parts)
            .map_err(|error| AppError::BadRequest(format!("invalid tool result content: {error}"))),
    }
}

fn chat_tool_choice_to_responses(choice: &ToolChoice) -> ResponsesToolChoice {
    match choice {
        ToolChoice::String(value) => ResponsesToolChoice::String(value.clone()),
        ToolChoice::Object(value) => {
            let flattened = value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(|name| json!({"type": "function", "name": name}))
                .unwrap_or_else(|| value.clone());
            ResponsesToolChoice::Object(flattened)
        }
    }
}

fn response_format_to_text_config(
    format: Option<&ResponseFormat>,
) -> Result<Option<TextConfig>, AppError> {
    let Some(format) = format else {
        return Ok(None);
    };
    let value = match format {
        ResponseFormat::Text => json!({"type": "text"}),
        ResponseFormat::JsonObject => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema { json_schema } => {
            let mut output = Map::new();
            output.insert("type".to_string(), json!("json_schema"));
            if let Some(name) = &json_schema.name {
                output.insert("name".to_string(), json!(name));
            }
            if let Some(description) = &json_schema.description {
                output.insert("description".to_string(), json!(description));
            }
            if let Some(strict) = json_schema.strict {
                output.insert("strict".to_string(), json!(strict));
            }
            if let Some(schema) = &json_schema.schema {
                output.insert("schema".to_string(), schema.clone());
            }
            Value::Object(output)
        }
    };
    Ok(Some(TextConfig {
        format: Some(value),
    }))
}

fn responses_to_chat(
    response: crate::openai::responses_schema::ResponsesResponse,
    capsule: &CapsuleRuntime,
) -> Result<ChatResponse, AppError> {
    if response.status == "failed" || response.error.is_some() {
        return Err(AppError::UpstreamBedrock(
            "Responses upstream returned a failed response".to_string(),
        ));
    }
    let mut reasoning_items = Vec::new();
    let mut all_reasoning_items = Vec::new();
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut function_call_seen = false;
    let mut output_item_types = BTreeSet::new();
    let mut unknown_output_item_types = BTreeSet::new();
    for item in &response.output {
        match item {
            ResponseOutputItem::Reasoning { .. } => {
                output_item_types.insert("reasoning".to_string());
                if function_call_seen {
                    return Err(AppError::Internal(
                        "interleaved Responses reasoning after a function call is not replayable"
                            .to_string(),
                    ));
                }
                let value = serde_json::to_value(item).map_err(|error| {
                    AppError::Internal(format!("failed to serialize reasoning item: {error}"))
                })?;
                if responses_reasoning_item_is_valid(&value) {
                    reasoning_items.push(value.clone());
                }
                all_reasoning_items.push(value);
            }
            ResponseOutputItem::Message { content, .. } => {
                output_item_types.insert("message".to_string());
                for part in content {
                    match part {
                        crate::openai::responses_schema::OutputContentPart::OutputText {
                            text: part,
                            ..
                        } => text.push_str(part),
                        crate::openai::responses_schema::OutputContentPart::Refusal { refusal } => {
                            text.push_str(refusal);
                        }
                    }
                }
            }
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                output_item_types.insert("function_call".to_string());
                function_call_seen = true;
                calls.push((call_id.clone(), name.clone(), arguments.clone()));
            }
            ResponseOutputItem::Other { item_type, .. } => {
                output_item_types.insert(item_type.clone());
                unknown_output_item_types.insert(item_type.clone());
                let call_like = is_call_like_output_item_type(item_type);
                tracing::warn!(
                    response_id = %response.id,
                    model = %response.model,
                    output_item_type = %item_type,
                    call_like,
                    "responses chat adapter observed an unsupported output item type"
                );
                if call_like {
                    return Err(unsupported_call_item_error(item_type));
                }
            }
        }
    }

    if !calls.is_empty() && reasoning_items.len() != all_reasoning_items.len() {
        return Err(AppError::Internal(
            "Responses tool call is missing replayable encrypted reasoning".to_string(),
        ));
    }
    if !calls.is_empty() && !reasoning_items.is_empty() && !capsule.encoder_enabled {
        return Err(AppError::Internal(
            "Responses reasoning tool replay requires CHAT_REASONING_CAPSULE_ENABLED=true"
                .to_string(),
        ));
    }

    let summary = reasoning_summary(&all_reasoning_items);
    let rendered = if summary.is_empty() {
        text
    } else {
        format!("<think>{summary}</think>{text}")
    };
    let tool_calls = calls
        .into_iter()
        .enumerate()
        .map(|(index, (call_id, name, arguments))| {
            let id = if reasoning_items.is_empty() {
                call_id
            } else {
                encode_responses_capsule(&call_id, &reasoning_items, &capsule.keyring)?
            };
            Ok(ToolCall {
                index: Some(index as i32),
                id: Some(id),
                r#type: "function".to_string(),
                function: ResponseFunction {
                    name: Some(name),
                    arguments,
                },
            })
        })
        .collect::<Result<Vec<_>, AppError>>()?;
    let has_tools = !tool_calls.is_empty();
    let content = if rendered.is_empty() && has_tools {
        None
    } else {
        Some(rendered)
    };
    let finish_reason = if has_tools {
        "tool_calls"
    } else if response.status == "incomplete" {
        "length"
    } else {
        "stop"
    };
    tracing::info!(
        response_id = %response.id,
        model = %response.model,
        finish_reason,
        tool_calls = tool_calls.len(),
        output_item_types = %type_set_label(&output_item_types),
        unknown_output_item_types = %type_set_label(&unknown_output_item_types),
        visible_text_bytes = content.as_deref().map_or(0, str::len),
        "responses chat non-stream adaptation completed"
    );

    Ok(ChatResponse {
        id: response.id.replacen("resp_", "chatcmpl-", 1),
        created: response.created_at,
        model: response.model,
        system_fingerprint: "fp".to_string(),
        choices: vec![Choice {
            index: 0,
            finish_reason: Some(finish_reason.to_string()),
            logprobs: None,
            message: ChatResponseMessage {
                role: Some("assistant".to_string()),
                content,
                tool_calls: has_tools.then_some(tool_calls),
                reasoning_content: None,
            },
        }],
        object: "chat.completion".to_string(),
        usage: responses_usage_to_chat(&response.usage),
    })
}

fn is_call_like_output_item_type(item_type: &str) -> bool {
    item_type.ends_with("_call")
}

fn unsupported_call_item_error(item_type: &str) -> AppError {
    AppError::UpstreamBedrock(format!(
        "Responses-backed chat cannot represent upstream output item type `{item_type}`"
    ))
}

fn type_set_label(types: &BTreeSet<String>) -> String {
    if types.is_empty() {
        "none".to_string()
    } else {
        types.iter().cloned().collect::<Vec<_>>().join(",")
    }
}

fn responses_usage_to_chat(usage: &crate::openai::responses_schema::ResponsesUsage) -> Usage {
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        prompt_tokens_details: usage.input_tokens_details.as_ref().map(|details| {
            PromptTokensDetails {
                cached_tokens: details.cached_tokens,
                audio_tokens: 0,
            }
        }),
        completion_tokens_details: usage.output_tokens_details.as_ref().map(|details| {
            CompletionTokensDetails {
                reasoning_tokens: details.reasoning_tokens,
                audio_tokens: 0,
            }
        }),
    }
}

fn reasoning_summary(items: &[Value]) -> String {
    let mut output = String::new();
    for item in items {
        if let Some(summary) = item.get("summary").and_then(Value::as_array) {
            for part in summary {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    output.push_str(text);
                }
            }
        }
    }
    output
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<Value>, AppError> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() > MAX_SSE_FRAME_BYTES && frame_boundary(&self.buffer).is_none() {
            return Err(AppError::UpstreamBedrock(
                "Responses SSE frame exceeds the size limit".to_string(),
            ));
        }
        let mut events = Vec::new();
        while let Some((position, delimiter_len)) = frame_boundary(&self.buffer) {
            let frame = self.buffer[..position].to_vec();
            self.buffer.drain(..position + delimiter_len);
            if let Some(event) = parse_sse_frame(&frame)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<Value>, AppError> {
        if self.buffer.iter().all(u8::is_ascii_whitespace) {
            self.buffer.clear();
            return Ok(Vec::new());
        }
        let frame = std::mem::take(&mut self.buffer);
        Ok(parse_sse_frame(&frame)?.into_iter().collect())
    }
}

fn frame_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left <= right => Some((left, 2)),
        (Some(_), Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, Some(right)) => Some((right, 4)),
        (None, None) => None,
    }
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<Value>, AppError> {
    let mut data = Vec::new();
    for raw_line in frame.split(|byte| *byte == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let Some(value) = line.strip_prefix(b"data:") else {
            continue;
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        data.push(value);
    }
    if data.is_empty() {
        return Ok(None);
    }
    let mut joined = Vec::new();
    for (index, line) in data.into_iter().enumerate() {
        if index > 0 {
            joined.push(b'\n');
        }
        joined.extend_from_slice(line);
    }
    if joined == b"[DONE]" {
        return Ok(None);
    }
    serde_json::from_slice(&joined)
        .map(Some)
        .map_err(|error| AppError::UpstreamBedrock(format!("invalid Responses SSE event: {error}")))
}

struct ResponsesChatStreamState {
    request_id: Arc<str>,
    message_id: String,
    model: String,
    created: i64,
    include_usage: bool,
    capsule: Arc<CapsuleRuntime>,
    reasoning_items: BTreeMap<u32, Value>,
    summary: String,
    summary_streamed: bool,
    think_open: bool,
    tool_indices: HashMap<u32, i32>,
    tool_arguments: HashMap<u32, String>,
    tool_calls_seen: bool,
    terminal_seen: bool,
    finish_reason: Option<&'static str>,
    terminal_event: Option<&'static str>,
    output_item_types: BTreeSet<String>,
    unknown_output_item_types: BTreeSet<String>,
    visible_text_bytes: usize,
    adapter_failed: bool,
    adapter_finished: bool,
}

impl ResponsesChatStreamState {
    fn new(
        request_id: Arc<str>,
        message_id: String,
        model: String,
        include_usage: bool,
        capsule: Arc<CapsuleRuntime>,
    ) -> Self {
        Self {
            request_id,
            message_id,
            model,
            created: now_unix_secs(),
            include_usage,
            capsule,
            reasoning_items: BTreeMap::new(),
            summary: String::new(),
            summary_streamed: false,
            think_open: false,
            tool_indices: HashMap::new(),
            tool_arguments: HashMap::new(),
            tool_calls_seen: false,
            terminal_seen: false,
            finish_reason: None,
            terminal_event: None,
            output_item_types: BTreeSet::new(),
            unknown_output_item_types: BTreeSet::new(),
            visible_text_bytes: 0,
            adapter_failed: false,
            adapter_finished: false,
        }
    }

    fn role_chunk(&self) -> ChatStreamResponse {
        self.message_chunk(
            ChatResponseMessage {
                role: Some("assistant".to_string()),
                content: Some(String::new()),
                ..Default::default()
            },
            None,
        )
    }

    fn message_chunk(
        &self,
        delta: ChatResponseMessage,
        finish_reason: Option<String>,
    ) -> ChatStreamResponse {
        ChatStreamResponse {
            id: self.message_id.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: "fp".to_string(),
            choices: vec![ChoiceDelta {
                index: 0,
                finish_reason,
                logprobs: None,
                delta,
            }],
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    fn content_chunk(&mut self, content: String) -> ChatStreamResponse {
        self.visible_text_bytes += content.len();
        self.message_chunk(
            ChatResponseMessage {
                content: Some(content),
                ..Default::default()
            },
            None,
        )
    }

    fn close_think(&mut self, output: &mut Vec<ChatStreamResponse>) {
        if self.think_open {
            self.think_open = false;
            output.push(self.content_chunk("</think>".to_string()));
        }
    }

    fn observe_output_item_type<'a>(
        &mut self,
        item: &'a Value,
        event_type: &str,
    ) -> Result<&'a str, AppError> {
        let item_type = item.get("type").and_then(Value::as_str).ok_or_else(|| {
            AppError::UpstreamBedrock("Responses output item has no type".to_string())
        })?;
        self.output_item_types.insert(item_type.to_string());

        if !matches!(item_type, "reasoning" | "message" | "function_call") {
            let first_observation = self.unknown_output_item_types.insert(item_type.to_string());
            let call_like = is_call_like_output_item_type(item_type);
            if first_observation {
                tracing::warn!(
                    request_id = %self.request_id,
                    completion_id = %self.message_id,
                    model = %self.model,
                    sse_event_type = %event_type,
                    output_item_type = %item_type,
                    call_like,
                    "responses chat adapter observed an unsupported output item type"
                );
            }
            if call_like {
                return Err(unsupported_call_item_error(item_type));
            }
        }

        Ok(item_type)
    }

    fn observe_terminal_output(&mut self, event: &Value, event_type: &str) -> Result<(), AppError> {
        let Some(output) = event
            .get("response")
            .and_then(|response| response.get("output"))
        else {
            return Ok(());
        };
        let output = output.as_array().ok_or_else(|| {
            AppError::UpstreamBedrock("Responses terminal output is not an array".to_string())
        })?;
        for item in output {
            self.observe_output_item_type(item, event_type)?;
        }
        Ok(())
    }

    fn store_reasoning_item(&mut self, output_index: u32, mut item: Value) {
        if let Some(existing) = self.reasoning_items.get(&output_index) {
            if item.get("encrypted_content").is_none() {
                if let Some(encrypted) = existing.get("encrypted_content") {
                    item.as_object_mut()
                        .expect("reasoning output item is an object")
                        .insert("encrypted_content".to_string(), encrypted.clone());
                }
            }
            if item.get("summary").is_none() {
                if let Some(summary) = existing.get("summary") {
                    item.as_object_mut()
                        .expect("reasoning output item is an object")
                        .insert("summary".to_string(), summary.clone());
                }
            }
        }
        self.reasoning_items.insert(output_index, item);
    }

    fn wire_tool_id(&self, call_id: &str) -> Result<String, AppError> {
        let items = self.reasoning_items.values().cloned().collect::<Vec<_>>();
        if items.is_empty() {
            return Ok(call_id.to_string());
        }
        if !items.iter().all(responses_reasoning_item_is_valid) {
            return Err(AppError::Internal(
                "Responses tool call started before replayable encrypted reasoning was complete"
                    .to_string(),
            ));
        }
        if !self.capsule.encoder_enabled {
            return Err(AppError::Internal(
                "Responses reasoning tool replay requires CHAT_REASONING_CAPSULE_ENABLED=true"
                    .to_string(),
            ));
        }
        encode_responses_capsule(call_id, &items, &self.capsule.keyring)
    }

    fn tool_metadata_chunk(
        &mut self,
        output_index: u32,
        call_id: &str,
        name: &str,
    ) -> Result<ChatStreamResponse, AppError> {
        let index = if let Some(index) = self.tool_indices.get(&output_index) {
            *index
        } else {
            let index = self.tool_indices.len() as i32;
            self.tool_indices.insert(output_index, index);
            index
        };
        self.tool_arguments.entry(output_index).or_default();
        self.tool_calls_seen = true;
        Ok(self.message_chunk(
            ChatResponseMessage {
                tool_calls: Some(vec![ToolCall {
                    index: Some(index),
                    id: Some(self.wire_tool_id(call_id)?),
                    r#type: "function".to_string(),
                    function: ResponseFunction {
                        name: Some(name.to_string()),
                        arguments: String::new(),
                    },
                }]),
                ..Default::default()
            },
            None,
        ))
    }

    fn tool_arguments_chunk(
        &mut self,
        output_index: u32,
        delta: &str,
    ) -> Result<Option<ChatStreamResponse>, AppError> {
        if delta.is_empty() {
            return Ok(None);
        }
        let Some(index) = self.tool_indices.get(&output_index).copied() else {
            return Err(AppError::UpstreamBedrock(
                "Responses tool arguments arrived before tool metadata".to_string(),
            ));
        };
        self.tool_arguments
            .entry(output_index)
            .or_default()
            .push_str(delta);
        Ok(Some(self.message_chunk(
            ChatResponseMessage {
                tool_calls: Some(vec![ToolCall {
                    index: Some(index),
                    id: None,
                    r#type: "function".to_string(),
                    function: ResponseFunction {
                        name: None,
                        arguments: delta.to_string(),
                    },
                }]),
                ..Default::default()
            },
            None,
        )))
    }

    fn complete_tool_arguments(
        &mut self,
        output_index: u32,
        arguments: &str,
    ) -> Result<Option<ChatStreamResponse>, AppError> {
        let accumulated = self
            .tool_arguments
            .get(&output_index)
            .map_or("", String::as_str);
        if arguments == accumulated {
            return Ok(None);
        }
        let suffix = arguments.strip_prefix(accumulated).ok_or_else(|| {
            AppError::UpstreamBedrock(
                "Responses tool argument deltas do not match the completed arguments".to_string(),
            )
        })?;
        self.tool_arguments_chunk(output_index, suffix)
    }

    fn map_event(&mut self, event: &Value) -> Result<Vec<ChatStreamResponse>, AppError> {
        let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
            AppError::UpstreamBedrock("Responses SSE event has no type".to_string())
        })?;
        let mut output = Vec::new();
        match event_type {
            "response.reasoning_summary_text.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !delta.is_empty() {
                    self.summary.push_str(delta);
                    self.summary_streamed = true;
                    let rendered = if self.think_open {
                        delta.to_string()
                    } else {
                        self.think_open = true;
                        format!("<think>{delta}")
                    };
                    output.push(self.content_chunk(rendered));
                }
            }
            "response.reasoning_summary_text.done" => {
                let text = event
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !self.summary_streamed && !text.is_empty() {
                    self.summary = text.to_string();
                    output.push(self.content_chunk(format!("<think>{text}</think>")));
                } else {
                    self.summary = text.to_string();
                    self.close_think(&mut output);
                }
            }
            "response.output_text.delta" => {
                self.close_think(&mut output);
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !delta.is_empty() {
                    output.push(self.content_chunk(delta.to_string()));
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                let output_index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as u32;
                let item = event.get("item").ok_or_else(|| {
                    AppError::UpstreamBedrock("Responses output item event has no item".to_string())
                })?;
                match self.observe_output_item_type(item, event_type)? {
                    "reasoning" => {
                        if self.tool_calls_seen {
                            return Err(AppError::Internal(
                                "interleaved Responses reasoning after a function call is not replayable"
                                    .to_string(),
                            ));
                        }
                        self.store_reasoning_item(output_index, item.clone());
                        if event_type == "response.output_item.done" && !self.summary_streamed {
                            let summary = reasoning_summary(std::slice::from_ref(item));
                            if !summary.is_empty() {
                                self.summary = summary.clone();
                                output
                                    .push(self.content_chunk(format!("<think>{summary}</think>")));
                            }
                        }
                    }
                    "function_call" => {
                        self.close_think(&mut output);
                        let call_id =
                            item.get("call_id").and_then(Value::as_str).ok_or_else(|| {
                                AppError::UpstreamBedrock(
                                    "Responses function call has no call_id".to_string(),
                                )
                            })?;
                        let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
                            AppError::UpstreamBedrock(
                                "Responses function call has no name".to_string(),
                            )
                        })?;
                        if !self.tool_indices.contains_key(&output_index) {
                            output.push(self.tool_metadata_chunk(output_index, call_id, name)?);
                        }
                        if event_type == "response.output_item.done" {
                            let arguments = item
                                .get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            if let Some(chunk) =
                                self.complete_tool_arguments(output_index, arguments)?
                            {
                                output.push(chunk);
                            }
                        }
                    }
                    "message" => {}
                    _ => {}
                }
            }
            "response.function_call_arguments.delta" => {
                let output_index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as u32;
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(chunk) = self.tool_arguments_chunk(output_index, delta)? {
                    output.push(chunk);
                }
            }
            "response.function_call_arguments.done" => {
                let output_index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as u32;
                let arguments = event
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(chunk) = self.complete_tool_arguments(output_index, arguments)? {
                    output.push(chunk);
                }
            }
            "response.completed" | "response.incomplete" => {
                self.observe_terminal_output(event, event_type)?;
                self.close_think(&mut output);
                self.terminal_seen = true;
                self.terminal_event = Some(if event_type == "response.incomplete" {
                    "response.incomplete"
                } else {
                    "response.completed"
                });
                let finish_reason = if event_type == "response.incomplete" {
                    "length"
                } else if self.tool_calls_seen {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.finish_reason = Some(finish_reason);
                output.push(self.message_chunk(
                    ChatResponseMessage::default(),
                    Some(finish_reason.to_string()),
                ));
                if self.include_usage {
                    if let Some(usage) = event.get("response").and_then(|value| value.get("usage"))
                    {
                        output.push(self.usage_chunk(usage));
                    }
                }
            }
            "response.failed" | "error" => {
                self.terminal_seen = true;
                self.terminal_event = Some(if event_type == "error" {
                    "error"
                } else {
                    "response.failed"
                });
                return Err(AppError::UpstreamBedrock(
                    "Responses upstream reported a failed stream".to_string(),
                ));
            }
            _ => {}
        }
        Ok(output)
    }

    fn usage_chunk(&self, usage: &Value) -> ChatStreamResponse {
        let prompt_tokens = usage
            .get("input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32;
        let completion_tokens = usage
            .get("output_tokens")
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32;
        let total_tokens = usage
            .get("total_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(prompt_tokens as i64 + completion_tokens as i64)
            as i32;
        let cached_tokens = usage
            .get("input_tokens_details")
            .and_then(|value| value.get("cached_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32;
        let reasoning_tokens = usage
            .get("output_tokens_details")
            .and_then(|value| value.get("reasoning_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32;
        ChatStreamResponse {
            id: self.message_id.clone(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: "fp".to_string(),
            choices: Vec::new(),
            object: "chat.completion.chunk".to_string(),
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens,
                    audio_tokens: 0,
                }),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens,
                    audio_tokens: 0,
                }),
            }),
        }
    }
}

impl Drop for ResponsesChatStreamState {
    fn drop(&mut self) {
        if self.adapter_failed {
            tracing::warn!(
                request_id = %self.request_id,
                completion_id = %self.message_id,
                model = %self.model,
                finish_reason = self.finish_reason.unwrap_or("none"),
                terminal_event = self.terminal_event.unwrap_or("none"),
                tool_calls = self.tool_indices.len(),
                output_item_types = %type_set_label(&self.output_item_types),
                unknown_output_item_types = %type_set_label(&self.unknown_output_item_types),
                visible_text_bytes = self.visible_text_bytes,
                "responses chat stream adapter failed"
            );
        } else if !self.terminal_seen {
            tracing::warn!(
                request_id = %self.request_id,
                completion_id = %self.message_id,
                model = %self.model,
                tool_calls = self.tool_indices.len(),
                output_item_types = %type_set_label(&self.output_item_types),
                unknown_output_item_types = %type_set_label(&self.unknown_output_item_types),
                visible_text_bytes = self.visible_text_bytes,
                "responses chat stream dropped before terminal event"
            );
        } else if !self.adapter_finished && self.finish_reason.is_some() {
            tracing::warn!(
                request_id = %self.request_id,
                completion_id = %self.message_id,
                model = %self.model,
                finish_reason = self.finish_reason.unwrap_or("unknown"),
                terminal_event = self.terminal_event.unwrap_or("unknown"),
                tool_calls = self.tool_indices.len(),
                output_item_types = %type_set_label(&self.output_item_types),
                unknown_output_item_types = %type_set_label(&self.unknown_output_item_types),
                visible_text_bytes = self.visible_text_bytes,
                "responses chat stream dropped after terminal event before adapter EOF"
            );
        }
    }
}

fn new_chat_completion_id() -> String {
    let request_id = gen_request_id();
    let suffix = request_id.strip_prefix("req-").unwrap_or(&request_id);
    format!("chatcmpl-{suffix}")
}

fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "responses_chat_provider_tests.rs"]
mod tests;
