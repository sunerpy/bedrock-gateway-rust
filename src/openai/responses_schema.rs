//! OpenAI **Responses API** wire schema (serde).
//!
//! This module is intentionally SEPARATE from [`crate::openai::schema`] (the
//! Chat Completions schema). The Responses API uses a different request/response
//! envelope, different field names (e.g. `input_tokens` instead of
//! `prompt_tokens`), and a typed streaming-event lifecycle. None of the chat
//! structs are reused or modified.
//!
//! ## Option B guardrail
//!
//! Bedrock-only features are surfaced only through the OpenAI-sanctioned
//! controlled-passthrough mechanism: [`ResponsesRequest`] carries a flattened
//! `extra: HashMap<String, Value>` (mirroring the chat schema's pattern). No
//! invented top-level fields beyond the OpenAI Responses spec are added.
//!
//! ## Scope
//!
//! Streaming events cover the full standard lifecycle plus reasoning-text
//! deltas. The `response.function_call_arguments.delta/.done` events are
//! intentionally NOT modeled: the downstream consumer (codex) does not consume
//! them and they are out of scope for this gateway.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Top-level Responses API request (`POST /responses`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Controlled passthrough of unknown top-level fields (extra_body parity).
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// The `input` field accepts either a plain string or an array of input items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

/// Reasoning controls (`reasoning`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Text output controls (`text`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
}

// ---------------------------------------------------------------------------
// Tools (Responses flattened-function shape)
// ---------------------------------------------------------------------------

/// A tool the model may call.
///
/// Responses uses a FLATTENED function shape:
/// `{"type":"function","name":...,"description":...,"parameters":...,"strict":...}`
/// — NOT the chat schema's nested `{"function":{...}}` form.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// `tool_choice` accepts either a string (e.g. "auto") or an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    String(String),
    Object(Value),
}

// ---------------------------------------------------------------------------
// Input items
// ---------------------------------------------------------------------------

/// A single item in an array-form `input`.
///
/// Internally tagged by `type`, except for the easy-message shorthand
/// (`{"role":..., "content":"..."}`) which carries no `type`. Deserialization
/// is implemented manually to accept both forms; serialization always emits the
/// tagged form.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseInputItem {
    Message {
        role: ResponsesRole,
        content: ResponsesContent,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
    Reasoning {
        id: String,
        content: Option<Value>,
        summary: Option<Value>,
        encrypted_content: Option<String>,
    },
}

/// Role on a message input item.
///
/// `Assistant` is accepted on the INPUT side because codex replays the prior
/// assistant turn as an input message item (`role:"assistant"` carrying an
/// `output_text` content part) on every multi-turn request. The translation
/// layer maps it to a Bedrock `assistant` turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponsesRole {
    User,
    Assistant,
    System,
    Developer,
}

/// Message content: a plain string or an array of typed content parts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

/// Typed input content parts.
///
/// `OutputText` is accepted on the INPUT side because codex echoes the prior
/// assistant turn's text as an `output_text` content part when replaying the
/// conversation. The translation layer treats it identically to `input_text`
/// (a Bedrock text block).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(flatten)]
        fields: HashMap<String, Value>,
    },
}

// Manual (de)serialization for ResponseInputItem to support BOTH the tagged
// form and the bare `{role, content}` easy-message shorthand.

impl Serialize for ResponseInputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Always serialize the tagged form.
        InputItemRepr::from_item(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ResponseInputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        // Easy-message shorthand: object with `role` and no `type`.
        let has_type = value.get("type").is_some();
        let has_role = value.get("role").is_some();
        if !has_type && has_role {
            #[derive(Deserialize)]
            struct EasyMessage {
                role: ResponsesRole,
                content: ResponsesContent,
            }
            let easy: EasyMessage =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            return Ok(ResponseInputItem::Message {
                role: easy.role,
                content: easy.content,
            });
        }
        let repr: InputItemRepr =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(repr.into_item())
    }
}

/// Internal tagged representation used for round-tripping `ResponseInputItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum InputItemRepr {
    #[serde(rename = "message")]
    Message {
        role: ResponsesRole,
        content: ResponsesContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

impl InputItemRepr {
    fn from_item(item: &ResponseInputItem) -> Self {
        match item.clone() {
            ResponseInputItem::Message { role, content } => {
                InputItemRepr::Message { role, content }
            }
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => InputItemRepr::FunctionCall {
                call_id,
                name,
                arguments,
            },
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                InputItemRepr::FunctionCallOutput { call_id, output }
            }
            ResponseInputItem::Reasoning {
                id,
                content,
                summary,
                encrypted_content,
            } => InputItemRepr::Reasoning {
                id,
                content,
                summary,
                encrypted_content,
            },
        }
    }

    fn into_item(self) -> ResponseInputItem {
        match self {
            InputItemRepr::Message { role, content } => {
                ResponseInputItem::Message { role, content }
            }
            InputItemRepr::FunctionCall {
                call_id,
                name,
                arguments,
            } => ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            },
            InputItemRepr::FunctionCallOutput { call_id, output } => {
                ResponseInputItem::FunctionCallOutput { call_id, output }
            }
            InputItemRepr::Reasoning {
                id,
                content,
                summary,
                encrypted_content,
            } => ResponseInputItem::Reasoning {
                id,
                content,
                summary,
                encrypted_content,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Non-streaming Responses API response object.
///
/// `output_text` is a client-side SDK convenience and is NOT a wire field — it
/// is intentionally absent here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    /// `resp_`-prefixed identifier.
    pub id: String,
    /// Always `"response"`.
    pub object: String,
    pub created_at: i64,
    pub status: String,
    pub output: Vec<ResponseOutputItem>,
    pub usage: ResponsesUsage,
    // Echoed request params.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,
}

// ---------------------------------------------------------------------------
// Output items
// ---------------------------------------------------------------------------

/// An item in `output`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<Value>,
    },
    #[serde(rename = "message")]
    Message {
        id: String,
        status: String,
        /// Always `"assistant"`.
        role: String,
        content: Vec<OutputContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// A content part inside an output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        annotations: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<Value>,
    },
}

// ---------------------------------------------------------------------------
// Usage (Responses field names — input_tokens, NOT prompt_tokens)
// ---------------------------------------------------------------------------

/// Input token usage details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: i32,
}

/// Output token usage details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i32,
}

/// Token usage for the Responses API.
///
/// Note the distinct field names from the chat [`crate::openai::schema::Usage`]:
/// `input_tokens` (not `prompt_tokens`) and `output_tokens` (not
/// `completion_tokens`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    pub total_tokens: i32,
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// The streaming-event lifecycle, tagged by `type`.
///
/// Covers the full standard lifecycle plus reasoning-text deltas. The
/// `response.function_call_arguments.delta/.done` events are intentionally
/// omitted (out of scope; not consumed downstream). Every variant carries a
/// monotonically increasing `sequence_number`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    #[serde(rename = "response.created")]
    Created {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        item: ResponseOutputItem,
        output_index: u32,
        sequence_number: u64,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContentPart,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: OutputContentPart,
        sequence_number: u64,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        item: ResponseOutputItem,
        output_index: u32,
        sequence_number: u64,
    },
    #[serde(rename = "response.completed")]
    Completed {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.failed")]
    Failed {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        sequence_number: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_string_input_request() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "input": "Hello, world"
        }))
        .unwrap();
        assert_eq!(req.model, "anthropic.claude-3-5-sonnet-20241022-v2:0");
        match req.input {
            ResponsesInput::Text(ref s) => assert_eq!(s, "Hello, world"),
            ResponsesInput::Items(_) => panic!("expected Text input"),
        }
        assert!(req.extra.is_empty());
    }

    #[test]
    fn deserialize_item_array_request() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"role": "developer", "content": "you are helpful"},
                {"type": "message", "role": "system", "content": [
                    {"type": "input_text", "text": "ctx"},
                    {"type": "input_image", "image_url": "http://x/y.png", "detail": "high"}
                ]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "42"},
                {"type": "reasoning", "id": "r1", "summary": ["s"]}
            ]
        }))
        .unwrap();
        let items = match req.input {
            ResponsesInput::Items(items) => items,
            ResponsesInput::Text(_) => panic!("expected Items input"),
        };
        assert_eq!(items.len(), 6);
        // Tagged message.
        assert!(matches!(
            items[0],
            ResponseInputItem::Message {
                role: ResponsesRole::User,
                content: ResponsesContent::Text(_)
            }
        ));
        // Bare easy-message shorthand (no `type`).
        assert!(matches!(
            items[1],
            ResponseInputItem::Message {
                role: ResponsesRole::Developer,
                content: ResponsesContent::Text(_)
            }
        ));
        // Parts content.
        assert!(matches!(
            items[2],
            ResponseInputItem::Message {
                role: ResponsesRole::System,
                content: ResponsesContent::Parts(_)
            }
        ));
        assert!(matches!(items[3], ResponseInputItem::FunctionCall { .. }));
        assert!(matches!(
            items[4],
            ResponseInputItem::FunctionCallOutput { .. }
        ));
        assert!(matches!(items[5], ResponseInputItem::Reasoning { .. }));
    }

    #[test]
    fn flattened_function_tool_round_trips() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "x",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
                "strict": true
            }]
        }))
        .unwrap();
        let tools = req.tools.expect("tools");
        let v = serde_json::to_value(&tools[0]).unwrap();
        // FLATTENED — no nested "function" key.
        assert!(v.get("function").is_none());
        assert_eq!(v["type"], "function");
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["strict"], true);
    }

    #[test]
    fn input_item_round_trips_to_tagged_form() {
        let item = ResponseInputItem::Message {
            role: ResponsesRole::User,
            content: ResponsesContent::Text("hi".to_string()),
        };
        let v = serde_json::to_value(&item).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "user");
        let back: ResponseInputItem = serde_json::from_value(v).unwrap();
        assert_eq!(back, item);
    }

    /// codex multi-turn replay: a prior assistant turn arrives as an INPUT
    /// message item with `role:"assistant"` and an `output_text` content part.
    /// Both must deserialize (regression guard for the T15 HTTP 400).
    #[test]
    fn assistant_role_with_output_text_part_deserializes() {
        let item: ResponseInputItem = serde_json::from_value(json!({
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "output_text", "text": "prior reply" }
            ]
        }))
        .expect("assistant + output_text input must parse");
        match item {
            ResponseInputItem::Message {
                role: ResponsesRole::Assistant,
                content: ResponsesContent::Parts(ref parts),
            } => {
                assert_eq!(parts.len(), 1);
                assert!(matches!(
                    parts[0],
                    ResponseContentPart::OutputText { ref text } if text == "prior reply"
                ));
            }
            _ => panic!("expected assistant message with an output_text part"),
        }
    }

    /// `assistant` role round-trips through serialization (rename "assistant").
    #[test]
    fn assistant_role_round_trips() {
        let v = serde_json::to_value(ResponsesRole::Assistant).unwrap();
        assert_eq!(v, json!("assistant"));
        let back: ResponsesRole = serde_json::from_value(v).unwrap();
        assert_eq!(back, ResponsesRole::Assistant);
    }

    #[test]
    fn serialize_response_uses_responses_field_names() {
        let resp = ResponsesResponse {
            id: "resp_abc".to_string(),
            object: "response".to_string(),
            created_at: 1_700_000_000,
            status: "completed".to_string(),
            output: vec![ResponseOutputItem::Message {
                id: "msg_1".to_string(),
                status: "completed".to_string(),
                role: "assistant".to_string(),
                content: vec![OutputContentPart::OutputText {
                    text: "hello".to_string(),
                    annotations: None,
                    logprobs: None,
                }],
            }],
            usage: ResponsesUsage {
                input_tokens: 10,
                input_tokens_details: Some(InputTokensDetails { cached_tokens: 4 }),
                output_tokens: 5,
                output_tokens_details: Some(OutputTokensDetails {
                    reasoning_tokens: 2,
                }),
                total_tokens: 15,
            },
            model: "m".to_string(),
            instructions: None,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            error: None,
            incomplete_details: None,
        };
        let v = serde_json::to_value(&resp).unwrap();

        assert_eq!(v["object"], "response");
        // Responses field names — input_tokens, NOT prompt_tokens.
        assert_eq!(v["usage"]["input_tokens"], 10);
        assert_eq!(v["usage"]["output_tokens"], 5);
        assert_eq!(v["usage"]["total_tokens"], 15);
        assert!(v["usage"].get("prompt_tokens").is_none());
        assert!(v["usage"].get("completion_tokens").is_none());
        assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 4);
        assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 2);
        // output_text is NOT a top-level wire field.
        assert!(v.get("output_text").is_none());
        // Skipped Option fields must not appear (no unknown keys).
        let obj = v.as_object().unwrap();
        let allowed: std::collections::HashSet<&str> = [
            "id",
            "object",
            "created_at",
            "status",
            "output",
            "usage",
            "model",
        ]
        .into_iter()
        .collect();
        for key in obj.keys() {
            assert!(allowed.contains(key.as_str()), "unexpected key: {key}");
        }
        // Round-trips back.
        let _back: ResponsesResponse = serde_json::from_value(v).unwrap();
    }

    #[test]
    fn stream_events_emit_spec_type_tags() {
        let dummy_response = || ResponsesResponse {
            id: "resp_1".to_string(),
            object: "response".to_string(),
            created_at: 0,
            status: "in_progress".to_string(),
            output: vec![],
            usage: ResponsesUsage {
                input_tokens: 0,
                input_tokens_details: None,
                output_tokens: 0,
                output_tokens_details: None,
                total_tokens: 0,
            },
            model: "m".to_string(),
            instructions: None,
            temperature: None,
            top_p: None,
            tool_choice: None,
            tools: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            error: None,
            incomplete_details: None,
        };
        let part = || OutputContentPart::OutputText {
            text: String::new(),
            annotations: None,
            logprobs: None,
        };
        let item = || ResponseOutputItem::Message {
            id: "msg_1".to_string(),
            status: "in_progress".to_string(),
            role: "assistant".to_string(),
            content: vec![],
        };

        let cases: Vec<(ResponseStreamEvent, &str)> = vec![
            (
                ResponseStreamEvent::Created {
                    response: dummy_response(),
                    sequence_number: 0,
                },
                "response.created",
            ),
            (
                ResponseStreamEvent::InProgress {
                    response: dummy_response(),
                    sequence_number: 1,
                },
                "response.in_progress",
            ),
            (
                ResponseStreamEvent::OutputItemAdded {
                    item: item(),
                    output_index: 0,
                    sequence_number: 2,
                },
                "response.output_item.added",
            ),
            (
                ResponseStreamEvent::ContentPartAdded {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    part: part(),
                    sequence_number: 3,
                },
                "response.content_part.added",
            ),
            (
                ResponseStreamEvent::OutputTextDelta {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    delta: "h".to_string(),
                    sequence_number: 4,
                },
                "response.output_text.delta",
            ),
            (
                ResponseStreamEvent::OutputTextDone {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    text: "hello".to_string(),
                    sequence_number: 5,
                },
                "response.output_text.done",
            ),
            (
                ResponseStreamEvent::ContentPartDone {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    part: part(),
                    sequence_number: 6,
                },
                "response.content_part.done",
            ),
            (
                ResponseStreamEvent::OutputItemDone {
                    item: item(),
                    output_index: 0,
                    sequence_number: 7,
                },
                "response.output_item.done",
            ),
            (
                ResponseStreamEvent::Completed {
                    response: dummy_response(),
                    sequence_number: 8,
                },
                "response.completed",
            ),
            (
                ResponseStreamEvent::Failed {
                    response: dummy_response(),
                    sequence_number: 9,
                },
                "response.failed",
            ),
            (
                ResponseStreamEvent::ReasoningTextDelta {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    delta: "t".to_string(),
                    sequence_number: 10,
                },
                "response.reasoning_text.delta",
            ),
            (
                ResponseStreamEvent::ReasoningTextDone {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    text: "thought".to_string(),
                    sequence_number: 11,
                },
                "response.reasoning_text.done",
            ),
        ];

        for (event, expected_type) in cases {
            let v = serde_json::to_value(&event).unwrap();
            assert_eq!(v["type"], expected_type, "wrong type tag");
            assert!(v.get("sequence_number").is_some());
        }

        // Specifically assert the output_text.delta shape.
        let delta = ResponseStreamEvent::OutputTextDelta {
            item_id: "msg_1".to_string(),
            output_index: 0,
            content_index: 0,
            delta: "x".to_string(),
            sequence_number: 4,
        };
        let v = serde_json::to_value(&delta).unwrap();
        assert_eq!(v["type"], "response.output_text.delta");
        assert_eq!(v["delta"], "x");
    }

    /// Guard: there must be NO `function_call_arguments.delta` event variant.
    ///
    /// Constructing any such serialized event must fail to deserialize into
    /// `ResponseStreamEvent` because no matching variant exists.
    #[test]
    fn no_function_call_arguments_delta_variant() {
        let raw = json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": "{",
            "sequence_number": 0
        });
        let result: Result<ResponseStreamEvent, _> = serde_json::from_value(raw);
        assert!(
            result.is_err(),
            "function_call_arguments.delta must not be a known variant"
        );

        let raw_done = json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "arguments": "{}",
            "sequence_number": 0
        });
        let result_done: Result<ResponseStreamEvent, _> = serde_json::from_value(raw_done);
        assert!(
            result_done.is_err(),
            "function_call_arguments.done must not be a known variant"
        );
    }
}
