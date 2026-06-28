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
//! Streaming events cover the standard lifecycle plus documented compatibility
//! events that clients may send or parse, including function-call argument,
//! reasoning-summary, refusal, and hosted-tool progress events. The Bedrock
//! stream state machine still emits only the minimal lifecycle it can produce.

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
///
/// This enum models the official OpenAI Responses API tool superset (not just
/// what codex emits): the flattened `function` tool, the `namespace` tool
/// container (a named group of inner tools — openai-python
/// `namespace_tool_param.py`), the `custom` tool, and an [`Unknown`] catch-all
/// for ANY other / hosted / future tool type so the wire boundary NEVER 400s on
/// an unrecognized `type`. The translation layer (`responses_translate.rs`)
/// decides which variants are kept (flattened into Bedrock `toolConfig`) and
/// which are silently dropped — deserialization itself is total.
///
/// [`Unknown`]: ResponsesTool::Unknown
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    /// A user-defined function tool (FLATTENED Responses shape). Always kept.
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
    /// A namespace tool container: a named group of inner tools. The gateway
    /// FLATTENS each inner tool into the Bedrock `toolConfig`, prefixing the
    /// inner name as `{namespace_name}__{inner_name}` so different namespaces
    /// can't collide. `tools` is the array key (NOT `functions`); all fields
    /// are required per the codex serializer / openai-python superset.
    #[serde(rename = "namespace")]
    Namespace {
        name: String,
        description: String,
        tools: Vec<ResponsesNamespaceInner>,
    },
    /// A custom (free-form / grammar) tool. Accepted and modeled; treated as a
    /// user-defined tool the gateway maps to a Bedrock `toolSpec` using its
    /// `name` + `description` (the `format` is acknowledged but not mapped —
    /// Bedrock has no free-form-tool grammar slot).
    #[serde(rename = "custom")]
    Custom {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<Value>,
    },
    /// Any other tool type (hosted server tools — `web_search`,
    /// `image_generation`, `code_interpreter`, `tool_search`, `mcp`,
    /// `computer`, … — plus any future type). Deserializes WITHOUT error so the
    /// wire boundary never 400s; the translation layer silently drops these
    /// (they have no Bedrock equivalent).
    #[serde(other)]
    Unknown,
}

/// An inner tool inside a [`ResponsesTool::Namespace`] container.
///
/// codex 0.140.0 only emits `function`, but the OpenAI SDK superset also allows
/// `custom`, so both are modeled for forward-compatibility. The nested
/// `Function` variant reuses the SAME field shape as the top-level
/// [`ResponsesTool::Function`] (name, optional description / parameters /
/// strict).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesNamespaceInner {
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
    #[serde(rename = "custom")]
    Custom {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<Value>,
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
        output: FunctionCallOutputValue,
    },
    ItemReference {
        id: String,
    },
    Other {
        item_type: String,
        fields: HashMap<String, Value>,
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

/// `function_call_output.output` accepts either a plain string or an ordered
/// array of content items. opencode preserves screenshot/read tool results this
/// way so image data is not JSON-stringified into text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutputValue {
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
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutputValue,
    },
    #[serde(rename = "item_reference")]
    ItemReference { id: String },
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
    #[serde(untagged)]
    Other {
        #[serde(rename = "type")]
        item_type: String,
        #[serde(flatten)]
        fields: HashMap<String, Value>,
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
            ResponseInputItem::ItemReference { id } => InputItemRepr::ItemReference { id },
            ResponseInputItem::Other { item_type, fields } => {
                InputItemRepr::Other { item_type, fields }
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
            InputItemRepr::ItemReference { id } => ResponseInputItem::ItemReference { id },
            InputItemRepr::Other { item_type, fields } => {
                ResponseInputItem::Other { item_type, fields }
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
        /// `None` on the non-stream output item and on the streaming
        /// `output_item.added` event (@ai-sdk/openai's added-chunk function_call
        /// member has no `status`). `Some("completed")` ONLY on the streaming
        /// `output_item.done` event — ai-sdk's done-chunk schema pins
        /// `status: z.literal("completed")` as REQUIRED, and omitting it drops
        /// the chunk to `unknown_chunk` so the tool call never reconstructs.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(untagged)]
    Other {
        #[serde(rename = "type")]
        item_type: String,
        #[serde(flatten)]
        fields: HashMap<String, Value>,
    },
}

/// A content part inside an output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        /// REQUIRED on the wire: always an array (empty `[]` when none).
        /// @ai-sdk/openai parses this with `z.array(...)` (no `.nullish()`);
        /// omitting it or sending `null` fails validation. Never add
        /// `skip_serializing_if` here.
        #[serde(default)]
        annotations: Vec<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<Value>,
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
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
/// Covers the standard lifecycle plus documented compatibility events. Some
/// variants exist for serde compatibility with clients and official SDK event
/// unions even when this gateway's Bedrock stream state machine does not emit
/// them itself. Every emitted gateway event carries a monotonically increasing
/// `sequence_number`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseStreamEvent {
    #[serde(rename = "response.queued")]
    Queued {
        response: ResponsesResponse,
        sequence_number: u64,
    },
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
    #[serde(rename = "response.incomplete")]
    Incomplete {
        response: ResponsesResponse,
        sequence_number: u64,
    },
    #[serde(rename = "error")]
    Error {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        param: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sequence_number: Option<u64>,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: String,
        output_index: u32,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: String,
        output_index: u32,
        arguments: String,
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
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        item_id: String,
        output_index: u32,
        summary_index: u32,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        item_id: String,
        output_index: u32,
        summary_index: u32,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        item_id: String,
        output_index: u32,
        summary_index: u32,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        item_id: String,
        output_index: u32,
        summary_index: u32,
        text: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        item_id: String,
        output_index: u32,
        content_index: u32,
        refusal: String,
        sequence_number: u64,
    },
    #[serde(untagged)]
    Other {
        #[serde(rename = "type")]
        event_type: String,
        #[serde(flatten)]
        fields: HashMap<String, Value>,
    },
}

impl ResponseStreamEvent {
    /// The wire `type` string for this event (matches the `serde(rename = ...)`
    /// discriminant). Content-free: returns only the protocol event-type tag,
    /// never any message text, arguments, or token values. Used by the
    /// stream-diagnostic logging in `bedrock::responses_stream` to record the
    /// emitted event-type sequence/counts without touching payload content.
    #[must_use]
    pub fn event_type(&self) -> &str {
        match self {
            ResponseStreamEvent::Queued { .. } => "response.queued",
            ResponseStreamEvent::Created { .. } => "response.created",
            ResponseStreamEvent::InProgress { .. } => "response.in_progress",
            ResponseStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
            ResponseStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
            ResponseStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
            ResponseStreamEvent::OutputTextDone { .. } => "response.output_text.done",
            ResponseStreamEvent::ContentPartDone { .. } => "response.content_part.done",
            ResponseStreamEvent::OutputItemDone { .. } => "response.output_item.done",
            ResponseStreamEvent::Completed { .. } => "response.completed",
            ResponseStreamEvent::Failed { .. } => "response.failed",
            ResponseStreamEvent::Incomplete { .. } => "response.incomplete",
            ResponseStreamEvent::Error { .. } => "error",
            ResponseStreamEvent::FunctionCallArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            ResponseStreamEvent::FunctionCallArgumentsDone { .. } => {
                "response.function_call_arguments.done"
            }
            ResponseStreamEvent::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
            ResponseStreamEvent::ReasoningTextDone { .. } => "response.reasoning_text.done",
            ResponseStreamEvent::ReasoningSummaryPartAdded { .. } => {
                "response.reasoning_summary_part.added"
            }
            ResponseStreamEvent::ReasoningSummaryPartDone { .. } => {
                "response.reasoning_summary_part.done"
            }
            ResponseStreamEvent::ReasoningSummaryTextDelta { .. } => {
                "response.reasoning_summary_text.delta"
            }
            ResponseStreamEvent::ReasoningSummaryTextDone { .. } => {
                "response.reasoning_summary_text.done"
            }
            ResponseStreamEvent::RefusalDelta { .. } => "response.refusal.delta",
            ResponseStreamEvent::RefusalDone { .. } => "response.refusal.done",
            ResponseStreamEvent::Other { event_type, .. } => event_type.as_str(),
        }
    }
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
                {"type": "function_call_output", "call_id": "c2", "output": [
                    {"type":"input_text", "text":"Image read successfully"},
                    {"type":"input_image", "image_url":"data:image/png;base64,AAECAw=="}
                ]},
                {"type": "item_reference", "id": "rs_stored_1"},
                {"type": "reasoning", "id": "r1", "summary": ["s"]}
            ]
        }))
        .unwrap();
        let items = match req.input {
            ResponsesInput::Items(items) => items,
            ResponsesInput::Text(_) => panic!("expected Items input"),
        };
        assert_eq!(items.len(), 8);
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
            ResponseInputItem::FunctionCallOutput {
                output: FunctionCallOutputValue::Text(_),
                ..
            }
        ));
        assert!(matches!(
            items[5],
            ResponseInputItem::FunctionCallOutput {
                output: FunctionCallOutputValue::Parts(_),
                ..
            }
        ));
        assert!(matches!(items[6], ResponseInputItem::ItemReference { .. }));
        assert!(matches!(items[7], ResponseInputItem::Reasoning { .. }));
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
    fn documented_create_request_options_are_leniently_accepted() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "include": [
                "file_search_call.results",
                "web_search_call.results",
                "web_search_call.action.sources",
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content"
            ],
            "metadata": {"trace": "abc"},
            "parallel_tool_calls": true,
            "previous_response_id": "resp_prev",
            "prompt_cache_key": "session_123",
            "store": false,
            "stream": true,
            "text": {"format": {"type": "text"}, "verbosity": "low"},
            "truncation": "auto",
            "background": false
        }))
        .unwrap();

        assert_eq!(req.include.as_ref().unwrap().len(), 8);
        assert_eq!(req.previous_response_id.as_deref(), Some("resp_prev"));
        assert_eq!(req.store, Some(false));
        assert_eq!(req.parallel_tool_calls, Some(true));
        assert_eq!(req.extra["prompt_cache_key"], "session_123");
        assert_eq!(req.extra["truncation"], "auto");
        assert_eq!(req.extra["background"], false);
    }

    #[test]
    fn documented_input_item_variants_parse_or_passthrough() {
        let passthrough_types = [
            "file_search_call",
            "computer_call",
            "computer_call_output",
            "web_search_call",
            "tool_search_call",
            "tool_search_output",
            "additional_tools",
            "compaction",
            "compaction_trigger",
            "image_generation_call",
            "code_interpreter_call",
            "local_shell_call",
            "local_shell_call_output",
            "shell_call",
            "shell_call_output",
            "apply_patch_call",
            "apply_patch_call_output",
            "mcp_list_tools",
            "mcp_approval_request",
            "mcp_approval_response",
            "mcp_call",
            "custom_tool_call",
            "custom_tool_call_output",
        ];

        for item_type in passthrough_types {
            let item: ResponseInputItem = serde_json::from_value(json!({
                "type": item_type,
                "id": "item_1",
                "status": "completed"
            }))
            .unwrap();
            match item {
                ResponseInputItem::Other {
                    item_type: actual,
                    fields,
                } => {
                    assert_eq!(actual, item_type);
                    assert_eq!(fields["id"], "item_1");
                }
                _ => panic!("expected passthrough input item for {item_type}"),
            }
        }
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
                    annotations: Vec::new(),
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
    fn documented_hosted_output_items_deserialize_as_passthrough_items() {
        let item_types = [
            "web_search_call",
            "file_search_call",
            "function_call_output",
            "computer_call",
            "computer_call_output",
            "tool_search_call",
            "tool_search_output",
            "additional_tools",
            "compaction",
            "code_interpreter_call",
            "image_generation_call",
            "local_shell_call",
            "local_shell_call_output",
            "shell_call",
            "shell_call_output",
            "apply_patch_call",
            "apply_patch_call_output",
            "mcp_call",
            "mcp_list_tools",
            "mcp_approval_request",
            "mcp_approval_response",
            "custom_tool_call",
            "custom_tool_call_output",
        ];

        for item_type in item_types {
            let item: ResponseOutputItem = serde_json::from_value(json!({
                "type": item_type,
                "id": "item_1",
                "status": "completed",
                "action": {"query": "example"},
                "results": []
            }))
            .unwrap();
            match item {
                ResponseOutputItem::Other {
                    item_type: actual,
                    fields,
                } => {
                    assert_eq!(actual, item_type);
                    assert_eq!(fields["id"], "item_1");
                    assert_eq!(fields["status"], "completed");
                }
                _ => panic!("expected passthrough output item for {item_type}"),
            }
        }
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
            annotations: Vec::new(),
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
                ResponseStreamEvent::Queued {
                    response: dummy_response(),
                    sequence_number: 0,
                },
                "response.queued",
            ),
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
            (
                ResponseStreamEvent::Incomplete {
                    response: dummy_response(),
                    sequence_number: 12,
                },
                "response.incomplete",
            ),
            (
                ResponseStreamEvent::FunctionCallArgumentsDelta {
                    item_id: "fc_1".to_string(),
                    output_index: 0,
                    delta: "{".to_string(),
                    sequence_number: 13,
                },
                "response.function_call_arguments.delta",
            ),
            (
                ResponseStreamEvent::FunctionCallArgumentsDone {
                    item_id: "fc_1".to_string(),
                    output_index: 0,
                    arguments: "{}".to_string(),
                    sequence_number: 14,
                },
                "response.function_call_arguments.done",
            ),
            (
                ResponseStreamEvent::ReasoningSummaryPartAdded {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 15,
                },
                "response.reasoning_summary_part.added",
            ),
            (
                ResponseStreamEvent::ReasoningSummaryTextDelta {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    summary_index: 0,
                    delta: "summary".to_string(),
                    sequence_number: 16,
                },
                "response.reasoning_summary_text.delta",
            ),
            (
                ResponseStreamEvent::ReasoningSummaryTextDone {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    summary_index: 0,
                    text: "summary".to_string(),
                    sequence_number: 17,
                },
                "response.reasoning_summary_text.done",
            ),
            (
                ResponseStreamEvent::ReasoningSummaryPartDone {
                    item_id: "r1".to_string(),
                    output_index: 0,
                    summary_index: 0,
                    sequence_number: 18,
                },
                "response.reasoning_summary_part.done",
            ),
            (
                ResponseStreamEvent::RefusalDelta {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    delta: "no".to_string(),
                    sequence_number: 19,
                },
                "response.refusal.delta",
            ),
            (
                ResponseStreamEvent::RefusalDone {
                    item_id: "msg_1".to_string(),
                    output_index: 0,
                    content_index: 0,
                    refusal: "no".to_string(),
                    sequence_number: 20,
                },
                "response.refusal.done",
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

    #[test]
    fn documented_extra_stream_events_deserialize_without_machine_emission() {
        let raw = json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "delta": "{",
            "sequence_number": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(raw).unwrap();
        assert!(matches!(
            event,
            ResponseStreamEvent::FunctionCallArgumentsDelta { .. }
        ));

        let raw_done = json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "arguments": "{}",
            "sequence_number": 0
        });
        let event: ResponseStreamEvent = serde_json::from_value(raw_done).unwrap();
        assert!(matches!(
            event,
            ResponseStreamEvent::FunctionCallArgumentsDone { .. }
        ));

        let raw_tool_progress = json!({
            "type": "response.web_search_call.searching",
            "item_id": "ws_1",
            "output_index": 0,
            "sequence_number": 2
        });
        let event: ResponseStreamEvent = serde_json::from_value(raw_tool_progress).unwrap();
        match event {
            ResponseStreamEvent::Other { event_type, fields } => {
                assert_eq!(event_type, "response.web_search_call.searching");
                assert_eq!(fields["item_id"], "ws_1");
            }
            _ => panic!("expected passthrough stream event"),
        }
    }

    // =======================================================================
    // COMPATIBILITY / REGRESSION COVERAGE
    //
    // Driven by the official openai-python v2.43.0 (@ e20b6b82) contract. Every
    // documented stream-event type, input-item type, output-item type, tool
    // type, content part, tool_choice variant, reasoning/text option, usage
    // shape, and error envelope gets a serde round-trip and/or behavioral test
    // here so future schema changes cannot silently break opencode / ai-sdk /
    // codex compatibility.
    //
    // Convention: an event/type the gateway's enum models EXPLICITLY is proven
    // by deserialize→match→serialize→retag. An event/type the gateway routes
    // through the untagged `Other`/passthrough arm (because the Bedrock state
    // machine never emits it, but a client/SDK may send or parse it) is proven
    // to round-trip through that arm WITHOUT data loss — preserving wire
    // compatibility. Negative tests lock the intentional divergences
    // (hosted request tools → would-be 400; input_file → 400; no
    // function_call_arguments emission; no [DONE]).
    // =======================================================================

    /// Round-trip a raw stream-event JSON through `ResponseStreamEvent`,
    /// asserting the re-serialized `type` tag matches and the named fields are
    /// preserved. Works for both explicitly-modeled variants and the untagged
    /// `Other` passthrough arm.
    fn assert_stream_event_round_trips(raw: Value, expected_type: &str, required_fields: &[&str]) {
        let event: ResponseStreamEvent = serde_json::from_value(raw.clone())
            .unwrap_or_else(|e| panic!("event `{expected_type}` failed to deserialize: {e}"));
        let back = serde_json::to_value(&event)
            .unwrap_or_else(|e| panic!("event `{expected_type}` failed to serialize: {e}"));
        assert_eq!(
            back["type"], expected_type,
            "event `{expected_type}` round-tripped to wrong type tag: {back}"
        );
        for field in required_fields {
            assert_eq!(
                back.get(*field),
                raw.get(*field),
                "event `{expected_type}` lost/changed field `{field}` on round-trip\nraw:  {raw}\nback: {back}"
            );
        }
    }

    // ----- The official 51-event streaming taxonomy ------------------------
    //
    // Each documented event carries `sequence_number:int` + `type`. Lifecycle
    // envelope events carry a `response`. Item-framing, text, refusal,
    // function-args, reasoning, hosted-tool-progress, and audio events carry
    // their own field sets. This single test exhaustively round-trips every
    // one. Events the gateway models EXPLICITLY survive as their typed variant;
    // events that fall through to `Other` survive as a faithful passthrough.

    /// A minimal valid `response` envelope object for lifecycle events.
    fn envelope_response_json() -> Value {
        json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1_700_000_000_i64,
            "status": "in_progress",
            "output": [],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            "model": "m"
        })
    }

    #[test]
    fn official_51_stream_events_all_round_trip() {
        let resp = envelope_response_json();

        // (raw_json, expected_type, required_fields_to_preserve)
        let cases: Vec<(Value, &str, Vec<&str>)> = vec![
            // --- Lifecycle envelope (carry `response`) ---
            (
                json!({"type":"response.created","sequence_number":0,"response":resp}),
                "response.created",
                vec!["sequence_number", "response"],
            ),
            (
                json!({"type":"response.in_progress","sequence_number":1,"response":resp}),
                "response.in_progress",
                vec!["sequence_number", "response"],
            ),
            (
                json!({"type":"response.queued","sequence_number":2,"response":resp}),
                "response.queued",
                vec!["sequence_number", "response"],
            ),
            (
                json!({"type":"response.completed","sequence_number":3,"response":resp}),
                "response.completed",
                vec!["sequence_number", "response"],
            ),
            (
                json!({"type":"response.incomplete","sequence_number":4,"response":resp}),
                "response.incomplete",
                vec!["sequence_number", "response"],
            ),
            (
                json!({"type":"response.failed","sequence_number":5,"response":resp}),
                "response.failed",
                vec!["sequence_number", "response"],
            ),
            // --- error (FLAT envelope, not nested) ---
            (
                json!({"type":"error","message":"boom","code":"server_error","param":null,"sequence_number":6}),
                "error",
                vec!["message", "code", "sequence_number"],
            ),
            // --- Item framing ---
            (
                json!({"type":"response.output_item.added","output_index":0,"sequence_number":7,
                       "item":{"type":"message","id":"msg_1","status":"in_progress","role":"assistant","content":[]}}),
                "response.output_item.added",
                vec!["output_index", "sequence_number", "item"],
            ),
            (
                json!({"type":"response.output_item.done","output_index":0,"sequence_number":8,
                       "item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[]}}),
                "response.output_item.done",
                vec!["output_index", "sequence_number", "item"],
            ),
            (
                json!({"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,
                       "sequence_number":9,"part":{"type":"output_text","text":"","annotations":[]}}),
                "response.content_part.added",
                vec![
                    "item_id",
                    "output_index",
                    "content_index",
                    "sequence_number",
                    "part",
                ],
            ),
            (
                json!({"type":"response.content_part.done","item_id":"msg_1","output_index":0,"content_index":0,
                       "sequence_number":10,"part":{"type":"output_text","text":"hi","annotations":[]}}),
                "response.content_part.done",
                vec![
                    "item_id",
                    "output_index",
                    "content_index",
                    "sequence_number",
                    "part",
                ],
            ),
            // --- Text ---
            (
                json!({"type":"response.output_text.delta","content_index":0,"delta":"h","item_id":"msg_1",
                       "output_index":0,"sequence_number":11}),
                "response.output_text.delta",
                vec![
                    "content_index",
                    "delta",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.output_text.done","content_index":0,"text":"hello","item_id":"msg_1",
                       "output_index":0,"sequence_number":12}),
                "response.output_text.done",
                vec![
                    "content_index",
                    "text",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"http://x"},
                       "annotation_index":0,"content_index":0,"item_id":"msg_1","output_index":0,"sequence_number":13}),
                "response.output_text.annotation.added",
                vec![
                    "annotation",
                    "annotation_index",
                    "item_id",
                    "sequence_number",
                ],
            ),
            // --- Refusal ---
            (
                json!({"type":"response.refusal.delta","delta":"no","content_index":0,"item_id":"msg_1",
                       "output_index":0,"sequence_number":14}),
                "response.refusal.delta",
                vec![
                    "delta",
                    "content_index",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.refusal.done","refusal":"refused","content_index":0,"item_id":"msg_1",
                       "output_index":0,"sequence_number":15}),
                "response.refusal.done",
                vec![
                    "refusal",
                    "content_index",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            // --- Function-call arguments (schema accepts; machine never emits) ---
            (
                json!({"type":"response.function_call_arguments.delta","delta":"{","item_id":"fc_1",
                       "output_index":0,"sequence_number":16}),
                "response.function_call_arguments.delta",
                vec!["delta", "item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.function_call_arguments.done","arguments":"{}","item_id":"fc_1",
                       "name":"f","output_index":0,"sequence_number":17}),
                "response.function_call_arguments.done",
                vec!["arguments", "item_id", "output_index", "sequence_number"],
            ),
            // --- Reasoning ---
            (
                json!({"type":"response.reasoning_text.delta","content_index":0,"delta":"t","item_id":"rs_1",
                       "output_index":0,"sequence_number":18}),
                "response.reasoning_text.delta",
                vec![
                    "content_index",
                    "delta",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.reasoning_text.done","content_index":0,"text":"thought","item_id":"rs_1",
                       "output_index":0,"sequence_number":19}),
                "response.reasoning_text.done",
                vec![
                    "content_index",
                    "text",
                    "item_id",
                    "output_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,
                       "summary_index":0,"sequence_number":20}),
                "response.reasoning_summary_part.added",
                vec![
                    "item_id",
                    "output_index",
                    "summary_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.reasoning_summary_part.done","item_id":"rs_1","output_index":0,
                       "summary_index":0,"sequence_number":21}),
                "response.reasoning_summary_part.done",
                vec![
                    "item_id",
                    "output_index",
                    "summary_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.reasoning_summary_text.delta","delta":"sum","item_id":"rs_1",
                       "output_index":0,"summary_index":0,"sequence_number":22}),
                "response.reasoning_summary_text.delta",
                vec![
                    "delta",
                    "item_id",
                    "output_index",
                    "summary_index",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.reasoning_summary_text.done","text":"sum","item_id":"rs_1",
                       "output_index":0,"summary_index":0,"sequence_number":23}),
                "response.reasoning_summary_text.done",
                vec![
                    "text",
                    "item_id",
                    "output_index",
                    "summary_index",
                    "sequence_number",
                ],
            ),
            // --- Hosted tool progress: web_search ---
            (
                json!({"type":"response.web_search_call.in_progress","item_id":"ws_1","output_index":0,"sequence_number":24}),
                "response.web_search_call.in_progress",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.web_search_call.searching","item_id":"ws_1","output_index":0,"sequence_number":25}),
                "response.web_search_call.searching",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.web_search_call.completed","item_id":"ws_1","output_index":0,"sequence_number":26}),
                "response.web_search_call.completed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            // --- Hosted tool progress: file_search ---
            (
                json!({"type":"response.file_search_call.in_progress","item_id":"fs_1","output_index":0,"sequence_number":27}),
                "response.file_search_call.in_progress",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.file_search_call.searching","item_id":"fs_1","output_index":0,"sequence_number":28}),
                "response.file_search_call.searching",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.file_search_call.completed","item_id":"fs_1","output_index":0,"sequence_number":29}),
                "response.file_search_call.completed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            // --- Hosted tool progress: code_interpreter ---
            (
                json!({"type":"response.code_interpreter_call.in_progress","item_id":"ci_1","output_index":0,"sequence_number":30}),
                "response.code_interpreter_call.in_progress",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.code_interpreter_call.interpreting","item_id":"ci_1","output_index":0,"sequence_number":31}),
                "response.code_interpreter_call.interpreting",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.code_interpreter_call.completed","item_id":"ci_1","output_index":0,"sequence_number":32}),
                "response.code_interpreter_call.completed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            // NOTE the documented `_code` infix on these two events.
            (
                json!({"type":"response.code_interpreter_call_code.delta","delta":"x=1","item_id":"ci_1","output_index":0,"sequence_number":33}),
                "response.code_interpreter_call_code.delta",
                vec!["delta", "item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.code_interpreter_call_code.done","code":"x=1","item_id":"ci_1","output_index":0,"sequence_number":34}),
                "response.code_interpreter_call_code.done",
                vec!["code", "item_id", "output_index", "sequence_number"],
            ),
            // --- Hosted tool progress: image_generation ---
            (
                json!({"type":"response.image_generation_call.in_progress","item_id":"ig_1","output_index":0,"sequence_number":35}),
                "response.image_generation_call.in_progress",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.image_generation_call.generating","item_id":"ig_1","output_index":0,"sequence_number":36}),
                "response.image_generation_call.generating",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.image_generation_call.partial_image","partial_image_b64":"AAEC",
                       "partial_image_index":0,"item_id":"ig_1","output_index":0,"sequence_number":37}),
                "response.image_generation_call.partial_image",
                vec![
                    "partial_image_b64",
                    "partial_image_index",
                    "item_id",
                    "sequence_number",
                ],
            ),
            (
                json!({"type":"response.image_generation_call.completed","item_id":"ig_1","output_index":0,"sequence_number":38}),
                "response.image_generation_call.completed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            // --- Hosted tool progress: mcp ---
            (
                json!({"type":"response.mcp_call.in_progress","item_id":"mcp_1","output_index":0,"sequence_number":39}),
                "response.mcp_call.in_progress",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_call.completed","item_id":"mcp_1","output_index":0,"sequence_number":40}),
                "response.mcp_call.completed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_call.failed","item_id":"mcp_1","output_index":0,"sequence_number":41}),
                "response.mcp_call.failed",
                vec!["item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_call_arguments.delta","delta":"{","item_id":"mcp_1","output_index":0,"sequence_number":42}),
                "response.mcp_call_arguments.delta",
                vec!["delta", "item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_call_arguments.done","arguments":"{}","item_id":"mcp_1","output_index":0,"sequence_number":43}),
                "response.mcp_call_arguments.done",
                vec!["arguments", "item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_list_tools.in_progress","item_id":"mcp_1","output_index":0,"sequence_number":44}),
                "response.mcp_list_tools.in_progress",
                vec!["item_id", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_list_tools.completed","item_id":"mcp_1","output_index":0,"sequence_number":45}),
                "response.mcp_list_tools.completed",
                vec!["item_id", "sequence_number"],
            ),
            (
                json!({"type":"response.mcp_list_tools.failed","item_id":"mcp_1","output_index":0,"sequence_number":46}),
                "response.mcp_list_tools.failed",
                vec!["item_id", "sequence_number"],
            ),
            // --- Hosted tool progress: custom_tool_call_input ---
            (
                json!({"type":"response.custom_tool_call_input.delta","delta":"x","item_id":"ct_1","output_index":0,"sequence_number":47}),
                "response.custom_tool_call_input.delta",
                vec!["delta", "item_id", "output_index", "sequence_number"],
            ),
            (
                json!({"type":"response.custom_tool_call_input.done","input":"x","item_id":"ct_1","output_index":0,"sequence_number":48}),
                "response.custom_tool_call_input.done",
                vec!["input", "item_id", "output_index", "sequence_number"],
            ),
            // --- Audio (completeness; gateway never emits — Other passthrough) ---
            (
                json!({"type":"response.audio.delta","delta":"AAEC","sequence_number":49}),
                "response.audio.delta",
                vec!["delta", "sequence_number"],
            ),
            (
                json!({"type":"response.audio.done","sequence_number":50}),
                "response.audio.done",
                vec!["sequence_number"],
            ),
            (
                json!({"type":"response.audio_transcript.delta","delta":"hi","sequence_number":51}),
                "response.audio_transcript.delta",
                vec!["delta", "sequence_number"],
            ),
            (
                json!({"type":"response.audio_transcript.done","sequence_number":52}),
                "response.audio_transcript.done",
                vec!["sequence_number"],
            ),
        ];

        // Every documented event must survive a serde round-trip (explicit
        // variant or faithful `Other` passthrough). 51 distinct event-type
        // strings from the contract + the 2 extra audio_transcript variants.
        let mut seen_types = std::collections::HashSet::new();
        for (raw, expected_type, required) in &cases {
            assert_stream_event_round_trips(raw.clone(), expected_type, required);
            seen_types.insert(*expected_type);
        }
        assert_eq!(
            seen_types.len(),
            cases.len(),
            "duplicate event-type string in the taxonomy table"
        );
    }

    /// The lifecycle envelope events deserialize into the EXPLICITLY-modeled
    /// variants (not the `Other` passthrough) and preserve the `response`.
    #[test]
    fn lifecycle_envelope_events_are_explicit_variants() {
        let resp = envelope_response_json();
        let created: ResponseStreamEvent = serde_json::from_value(
            json!({"type":"response.created","sequence_number":0,"response":resp}),
        )
        .unwrap();
        assert!(matches!(created, ResponseStreamEvent::Created { .. }));

        let failed: ResponseStreamEvent = serde_json::from_value(
            json!({"type":"response.failed","sequence_number":9,"response":resp}),
        )
        .unwrap();
        match failed {
            ResponseStreamEvent::Failed {
                response,
                sequence_number,
            } => {
                assert_eq!(sequence_number, 9);
                assert_eq!(response.object, "response");
            }
            other => panic!("expected Failed variant, got {other:?}"),
        }
    }

    /// The `error` stream event is FLAT (message/code/param/sequence_number at
    /// the top level — NOT nested under an `error` key).
    #[test]
    fn error_stream_event_is_flat_not_nested() {
        let event: ResponseStreamEvent = serde_json::from_value(json!({
            "type": "error",
            "message": "something failed",
            "code": "server_error",
            "param": null,
            "sequence_number": 3
        }))
        .unwrap();
        match event {
            ResponseStreamEvent::Error {
                ref code,
                ref message,
                ref sequence_number,
                ..
            } => {
                assert_eq!(code.as_deref(), Some("server_error"));
                assert_eq!(message.as_deref(), Some("something failed"));
                assert_eq!(*sequence_number, Some(3));
            }
            other => panic!("expected flat Error event, got {other:?}"),
        }
        let v = serde_json::to_value(&event).unwrap();
        // Flat: message/code at top level, NOT under `error`.
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "something failed");
        assert_eq!(v["code"], "server_error");
        assert!(
            v.get("error").is_none(),
            "error event must be flat, not nested"
        );
    }

    /// Hosted-tool-progress and audio events round-trip through the untagged
    /// `Other` arm with full field preservation (wire compatibility for SDKs
    /// that parse them even though the Bedrock state machine never emits them).
    #[test]
    fn hosted_tool_progress_events_passthrough_via_other() {
        let raw = json!({
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": "AAECAw==",
            "partial_image_index": 2,
            "item_id": "ig_1",
            "output_index": 0,
            "sequence_number": 7
        });
        let event: ResponseStreamEvent = serde_json::from_value(raw.clone()).unwrap();
        match event {
            ResponseStreamEvent::Other {
                ref event_type,
                ref fields,
            } => {
                assert_eq!(event_type, "response.image_generation_call.partial_image");
                assert_eq!(fields["partial_image_b64"], "AAECAw==");
                assert_eq!(fields["partial_image_index"], 2);
            }
            other => panic!("expected Other passthrough, got {other:?}"),
        }
        // Round-trips losslessly.
        let back = serde_json::to_value(&event).unwrap();
        assert_eq!(back, raw);
    }

    // ----- Input items -----------------------------------------------------

    /// EasyInputMessage shorthand: `{role, content}` with NO `type` deserializes
    /// to a Message item; the strict `{type:"message", role, content}` form also
    /// deserializes to a Message item. Roles user/assistant/system/developer.
    #[test]
    fn input_message_easy_and_strict_forms_all_roles() {
        for role in ["user", "assistant", "system", "developer"] {
            // Easy form (no type).
            let easy: ResponseInputItem =
                serde_json::from_value(json!({"role": role, "content": "hi"})).unwrap();
            assert!(
                matches!(easy, ResponseInputItem::Message { .. }),
                "easy form role {role} must be a Message"
            );
            // Strict form (with type).
            let strict: ResponseInputItem =
                serde_json::from_value(json!({"type":"message","role": role, "content": "hi"}))
                    .unwrap();
            assert!(
                matches!(strict, ResponseInputItem::Message { .. }),
                "strict form role {role} must be a Message"
            );
        }
    }

    /// Every documented input content part deserializes correctly:
    /// input_text, input_image (with detail), output_text. input_file parses
    /// into the schema (the 400 is enforced at the TRANSLATION layer, not the
    /// wire boundary — locked in responses_translate.rs::input_file_part_is_bad_request).
    #[test]
    fn input_content_parts_all_variants_deserialize() {
        let item: ResponseInputItem = serde_json::from_value(json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_image", "image_url": "http://x/y.png", "detail": "high"},
                {"type": "output_text", "text": "prior"},
                {"type": "input_file", "file_id": "f1", "filename": "a.pdf"}
            ]
        }))
        .unwrap();
        let parts = match item {
            ResponseInputItem::Message {
                content: ResponsesContent::Parts(parts),
                ..
            } => parts,
            other => panic!("expected message with parts, got {other:?}"),
        };
        assert_eq!(parts.len(), 4);
        assert!(matches!(parts[0], ResponseContentPart::InputText { ref text } if text == "hello"));
        assert!(matches!(
            parts[1],
            ResponseContentPart::InputImage { ref image_url, detail: Some(ref d) }
                if image_url == "http://x/y.png" && d == "high"
        ));
        assert!(
            matches!(parts[2], ResponseContentPart::OutputText { ref text } if text == "prior")
        );
        match &parts[3] {
            ResponseContentPart::InputFile { fields } => {
                assert_eq!(fields["file_id"], "f1");
                assert_eq!(fields["filename"], "a.pdf");
            }
            other => panic!("expected input_file part, got {other:?}"),
        }
    }

    /// input_image accepts every documented `detail` value (low|high|auto|
    /// original) and an absent detail.
    #[test]
    fn input_image_detail_values_all_accepted() {
        for detail in ["low", "high", "auto", "original"] {
            let part: ResponseContentPart = serde_json::from_value(json!({
                "type": "input_image", "image_url": "http://x", "detail": detail
            }))
            .unwrap();
            assert!(
                matches!(part, ResponseContentPart::InputImage { detail: Some(ref d), .. } if d == detail)
            );
        }
        // Absent detail.
        let part: ResponseContentPart =
            serde_json::from_value(json!({"type":"input_image","image_url":"http://x"})).unwrap();
        assert!(matches!(
            part,
            ResponseContentPart::InputImage { detail: None, .. }
        ));
    }

    /// `function_call_output.output` deserializes BOTH as a plain string AND as
    /// an ordered content-part array (opencode preserves screenshot/read tool
    /// results as parts so image data is not JSON-stringified into text).
    #[test]
    fn function_call_output_accepts_string_and_content_parts() {
        // String form.
        let s: ResponseInputItem = serde_json::from_value(json!({
            "type": "function_call_output", "call_id": "c1", "output": "42"
        }))
        .unwrap();
        assert!(matches!(
            s,
            ResponseInputItem::FunctionCallOutput {
                output: FunctionCallOutputValue::Text(ref t), ..
            } if t == "42"
        ));

        // Content-part array form.
        let parts: ResponseInputItem = serde_json::from_value(json!({
            "type": "function_call_output",
            "call_id": "c2",
            "output": [
                {"type": "input_text", "text": "Image read successfully"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAECAw=="}
            ]
        }))
        .unwrap();
        match parts {
            ResponseInputItem::FunctionCallOutput {
                output: FunctionCallOutputValue::Parts(ref p),
                ref call_id,
            } => {
                assert_eq!(call_id, "c2");
                assert_eq!(p.len(), 2);
                assert!(matches!(p[0], ResponseContentPart::InputText { .. }));
                assert!(matches!(p[1], ResponseContentPart::InputImage { .. }));
            }
            other => panic!("expected function_call_output parts, got {other:?}"),
        }
    }

    /// `function_call` input item round-trips with call_id/name/arguments.
    #[test]
    fn function_call_input_item_round_trips() {
        let raw = json!({"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"});
        let item: ResponseInputItem = serde_json::from_value(raw).unwrap();
        match &item {
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "c1");
                assert_eq!(name, "f");
                assert_eq!(arguments, "{\"x\":1}");
            }
            other => panic!("expected function_call, got {other:?}"),
        }
        let back = serde_json::to_value(&item).unwrap();
        assert_eq!(back["type"], "function_call");
        assert_eq!(back["call_id"], "c1");
    }

    /// `reasoning` input item carries id/summary/content/encrypted_content and
    /// round-trips. `encrypted_content` is accepted on input (not round-tripped
    /// to Bedrock — that is locked in responses_translate.rs).
    #[test]
    fn reasoning_input_item_round_trips_with_encrypted_content() {
        let raw = json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "s"}],
            "content": [{"type": "reasoning_text", "text": "c"}],
            "encrypted_content": "OPAQUE"
        });
        let item: ResponseInputItem = serde_json::from_value(raw).unwrap();
        match &item {
            ResponseInputItem::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
            } => {
                assert_eq!(id, "rs_1");
                assert!(summary.is_some());
                assert!(content.is_some());
                assert_eq!(encrypted_content.as_deref(), Some("OPAQUE"));
            }
            other => panic!("expected reasoning item, got {other:?}"),
        }
        let back = serde_json::to_value(&item).unwrap();
        assert_eq!(back["type"], "reasoning");
        assert_eq!(back["encrypted_content"], "OPAQUE");
    }

    /// `item_reference` is accepted (and is dropped at the stateless translation
    /// layer; here we only lock that it DESERIALIZES into the explicit variant).
    #[test]
    fn item_reference_input_item_deserializes() {
        let item: ResponseInputItem =
            serde_json::from_value(json!({"type":"item_reference","id":"rs_stored_1"})).unwrap();
        assert!(matches!(item, ResponseInputItem::ItemReference { ref id } if id == "rs_stored_1"));
    }

    /// EVERY documented hosted input-item type parses as an `Other` passthrough
    /// (the schema does not model them as first-class variants; the translation
    /// layer rejects unknown `Other` types with a 400 — locked separately). This
    /// proves the wire boundary accepts the shapes without a deserialize panic.
    #[test]
    fn all_documented_hosted_input_items_parse_as_passthrough() {
        let hosted = [
            "file_search_call",
            "computer_call",
            "computer_call_output",
            "web_search_call",
            "image_generation_call",
            "code_interpreter_call",
            "local_shell_call",
            "local_shell_call_output",
            "shell_call",
            "shell_call_output",
            "apply_patch_call",
            "apply_patch_call_output",
            "mcp_list_tools",
            "mcp_approval_request",
            "mcp_approval_response",
            "mcp_call",
            "tool_search_call",
            "additional_tools",
            "compaction_trigger",
            "custom_tool_call",
            "custom_tool_call_output",
        ];
        for ty in hosted {
            let item: ResponseInputItem =
                serde_json::from_value(json!({"type": ty, "id": "x_1", "status": "completed"}))
                    .unwrap_or_else(|e| panic!("hosted input item {ty} failed to parse: {e}"));
            match item {
                ResponseInputItem::Other {
                    ref item_type,
                    ref fields,
                } => {
                    assert_eq!(item_type, ty);
                    assert_eq!(fields["id"], "x_1");
                }
                other => panic!("expected Other passthrough for {ty}, got {other:?}"),
            }
        }
    }

    // ----- Output items ----------------------------------------------------

    /// `message` output item with output_text + annotations + logprobs and a
    /// refusal content part both round-trip.
    #[test]
    fn output_message_with_output_text_and_refusal_parts() {
        let raw = json!({
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "hi",
                 "annotations": [{"type":"url_citation","url":"http://x"}],
                 "logprobs": []},
                {"type": "refusal", "refusal": "no"}
            ]
        });
        let item: ResponseOutputItem = serde_json::from_value(raw).unwrap();
        match &item {
            ResponseOutputItem::Message {
                id,
                status,
                role,
                content,
            } => {
                assert_eq!(id, "msg_1");
                assert_eq!(status, "completed");
                assert_eq!(role, "assistant");
                assert_eq!(content.len(), 2);
                assert!(matches!(
                    content[0],
                    OutputContentPart::OutputText { ref text, ref annotations, logprobs: Some(_) }
                        if text == "hi" && !annotations.is_empty()
                ));
                assert!(
                    matches!(content[1], OutputContentPart::Refusal { ref refusal } if refusal == "no")
                );
            }
            other => panic!("expected message output item, got {other:?}"),
        }
        let back = serde_json::to_value(&item).unwrap();
        assert_eq!(back["type"], "message");
        assert_eq!(back["content"][1]["type"], "refusal");
    }

    /// `message` output `status` accepts every documented value
    /// (in_progress|completed|incomplete) — status is a free `String`, so this
    /// guards that none are rejected at the wire boundary.
    #[test]
    fn output_message_status_values_all_accepted() {
        for status in ["in_progress", "completed", "incomplete"] {
            let item: ResponseOutputItem = serde_json::from_value(json!({
                "type":"message","id":"m","status":status,"role":"assistant","content":[]
            }))
            .unwrap();
            assert!(
                matches!(item, ResponseOutputItem::Message { status: ref s, .. } if s == status)
            );
        }
    }

    /// `reasoning` and `function_call` output items round-trip with their
    /// explicit variants.
    #[test]
    fn output_reasoning_and_function_call_round_trip() {
        let reasoning: ResponseOutputItem = serde_json::from_value(json!({
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type":"summary_text","text":"s"}]
        }))
        .unwrap();
        assert!(matches!(reasoning, ResponseOutputItem::Reasoning { .. }));

        let fc: ResponseOutputItem = serde_json::from_value(json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "c1",
            "name": "f",
            "arguments": "{}"
        }))
        .unwrap();
        match &fc {
            ResponseOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
                status,
            } => {
                assert_eq!(id.as_deref(), Some("fc_1"));
                assert_eq!(call_id, "c1");
                assert_eq!(name, "f");
                assert_eq!(arguments, "{}");
                assert_eq!(status.as_deref(), None);
            }
            other => panic!("expected function_call output, got {other:?}"),
        }
    }

    /// Output annotation types (file_citation|url_citation|container_file_citation|
    /// file_path) all parse inside an output_text part's `annotations` array.
    #[test]
    fn output_text_annotation_types_all_parse() {
        for ann in [
            json!({"type":"file_citation","file_id":"f","index":0}),
            json!({"type":"url_citation","url":"http://x","start_index":0,"end_index":1}),
            json!({"type":"container_file_citation","container_id":"c","file_id":"f"}),
            json!({"type":"file_path","file_id":"f","index":0}),
        ] {
            let part: OutputContentPart = serde_json::from_value(json!({
                "type": "output_text", "text": "hi", "annotations": [ann.clone()]
            }))
            .unwrap();
            match part {
                OutputContentPart::OutputText {
                    ref annotations, ..
                } => {
                    assert_eq!(annotations[0]["type"], ann["type"]);
                }
                other => panic!("expected output_text with annotation, got {other:?}"),
            }
        }
    }

    /// An `output_text` part with no annotations must serialize the
    /// `annotations` key as an empty array `[]` (present, never omitted, never
    /// null). @ai-sdk/openai's non-stream parser requires this.
    #[test]
    fn output_text_part_serializes_empty_annotations_array() {
        let part = OutputContentPart::OutputText {
            text: "pong".to_string(),
            annotations: Vec::new(),
            logprobs: None,
        };
        let v = serde_json::to_value(&part).unwrap();
        assert_eq!(v["type"], "output_text");
        assert_eq!(v["text"], "pong");
        assert!(
            v.get("annotations").is_some(),
            "annotations key must always be present"
        );
        assert_eq!(v["annotations"], json!([]));
        assert!(v.get("logprobs").is_none());
        let s = serde_json::to_string(&part).unwrap();
        assert!(s.contains("\"annotations\":[]"), "got: {s}");
    }

    // ----- Tools & tool_choice ---------------------------------------------

    /// The `function` tool uses the FLATTENED Responses shape and round-trips.
    /// (Already covered by `flattened_function_tool_round_trips`; this adds the
    /// minimal-fields form + a `defer_loading` passthrough check via extra.)
    #[test]
    fn function_tool_minimal_form_round_trips() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(json!([
            {"type":"function","name":"f"}
        ]))
        .unwrap();
        assert_eq!(tools.len(), 1);
        let back = serde_json::to_value(&tools[0]).unwrap();
        assert_eq!(back["type"], "function");
        assert_eq!(back["name"], "f");
        assert!(back.get("function").is_none(), "must be flattened");
    }

    /// Every HOSTED / unknown tool type now deserializes into the
    /// [`ResponsesTool::Unknown`] catch-all WITHOUT error — the wire boundary
    /// never 400s on an unrecognized `type`. (Behavior change: these previously
    /// failed deserialization to force a 400; the translation layer now silently
    /// DROPS them so codex sessions that include hosted tools survive. See
    /// responses_translate.rs::filter_and_flatten_tools.)
    #[test]
    fn hosted_request_tools_deserialize_to_unknown() {
        let hosted = [
            "file_search",
            "web_search",
            "web_search_2025_08_26",
            "web_search_preview",
            "computer",
            "computer_use_preview",
            "code_interpreter",
            "image_generation",
            "mcp",
            "local_shell",
            "shell",
            "tool_search",
            "apply_patch",
        ];
        for ty in hosted {
            let tools: Vec<ResponsesTool> = serde_json::from_value(json!([{"type": ty}]))
                .unwrap_or_else(|e| panic!("hosted tool `{ty}` must deserialize to Unknown: {e}"));
            assert_eq!(tools.len(), 1);
            assert!(
                matches!(tools[0], ResponsesTool::Unknown),
                "hosted tool `{ty}` must map to ResponsesTool::Unknown"
            );
        }
    }

    /// A `namespace` tool container with nested `function`(s) deserializes; the
    /// array key is `tools` (not `functions`) and the inner element keeps the
    /// flattened function shape.
    #[test]
    fn namespace_tool_with_nested_functions_deserializes() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(json!([{
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "agent tools",
            "tools": [
                {"type":"function","name":"spawn_agent","description":"d","strict":false,
                 "parameters":{"type":"object","properties":{}}}
            ]
        }]))
        .unwrap();
        assert_eq!(tools.len(), 1);
        let ResponsesTool::Namespace {
            name,
            description,
            tools: inner,
        } = &tools[0]
        else {
            panic!("expected a namespace tool");
        };
        assert_eq!(name, "multi_agent_v1");
        assert_eq!(description, "agent tools");
        assert_eq!(inner.len(), 1);
        let ResponsesNamespaceInner::Function { name, .. } = &inner[0] else {
            panic!("expected a nested function");
        };
        assert_eq!(name, "spawn_agent");
    }

    /// A namespace may carry a nested `custom` inner tool (SDK superset).
    #[test]
    fn namespace_tool_with_nested_custom_deserializes() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(json!([{
            "type": "namespace",
            "name": "ns",
            "description": "d",
            "tools": [{"type":"custom","name":"grammar_tool"}]
        }]))
        .unwrap();
        let ResponsesTool::Namespace { tools: inner, .. } = &tools[0] else {
            panic!("expected namespace");
        };
        assert!(matches!(inner[0], ResponsesNamespaceInner::Custom { .. }));
    }

    /// A top-level `custom` tool deserializes (accepted, modeled).
    #[test]
    fn custom_tool_deserializes() {
        let tools: Vec<ResponsesTool> = serde_json::from_value(json!([{
            "type": "custom",
            "name": "my_custom",
            "description": "free-form",
            "format": {"type":"grammar","syntax":"lark","definition":"start: ..."}
        }]))
        .unwrap();
        assert_eq!(tools.len(), 1);
        let ResponsesTool::Custom { name, .. } = &tools[0] else {
            panic!("expected a custom tool");
        };
        assert_eq!(name, "my_custom");
    }

    /// `tool_choice` accepts every documented form: the string modes
    /// ("none"/"auto"/"required"), the allowed_tools object, hosted-type object,
    /// named-function object, mcp object, and custom object — all via the
    /// String|Object untagged enum.
    #[test]
    fn tool_choice_all_documented_variants_accepted() {
        // String modes.
        for mode in ["none", "auto", "required"] {
            let tc: ResponsesToolChoice = serde_json::from_value(json!(mode)).unwrap();
            assert!(matches!(tc, ResponsesToolChoice::String(ref s) if s == mode));
        }
        // Object variants.
        let objects = [
            json!({"type":"allowed_tools","mode":"auto","tools":[{"type":"function","name":"f"}]}),
            json!({"type":"file_search"}),
            json!({"type":"web_search_preview"}),
            json!({"type":"computer_use_preview"}),
            json!({"type":"image_generation"}),
            json!({"type":"code_interpreter"}),
            json!({"type":"function","name":"f"}),
            json!({"type":"mcp","server_label":"srv","name":"t"}),
            json!({"type":"custom","name":"c"}),
        ];
        for obj in objects {
            let tc: ResponsesToolChoice = serde_json::from_value(obj.clone()).unwrap();
            match tc {
                ResponsesToolChoice::Object(ref v) => assert_eq!(v["type"], obj["type"]),
                ResponsesToolChoice::String(_) => panic!("expected object tool_choice for {obj}"),
            }
        }
    }

    // ----- reasoning / text config -----------------------------------------

    /// `reasoning.effort` accepts every documented level and `summary` accepts
    /// auto|concise|detailed.
    #[test]
    fn reasoning_config_effort_and_summary_values() {
        for effort in ["none", "minimal", "low", "medium", "high", "xhigh"] {
            let cfg: ReasoningConfig = serde_json::from_value(json!({"effort": effort})).unwrap();
            assert_eq!(cfg.effort.as_deref(), Some(effort));
        }
        for summary in ["auto", "concise", "detailed"] {
            let cfg: ReasoningConfig = serde_json::from_value(json!({"summary": summary})).unwrap();
            assert_eq!(cfg.summary.as_deref(), Some(summary));
        }
        // `context` (auto|current_turn|all_turns) is not modeled as a typed
        // field; it must not cause a deserialize failure (extra keys ignored).
        let cfg: ReasoningConfig =
            serde_json::from_value(json!({"effort":"high","summary":"auto","context":"all_turns"}))
                .unwrap();
        assert_eq!(cfg.effort.as_deref(), Some("high"));
    }

    /// `text.format` accepts text|json_object|json_schema and `verbosity`
    /// low|medium|high (verbosity rides along in the flattened TextConfig — it
    /// is accepted, not rejected). format is a free `Value`.
    #[test]
    fn text_config_format_and_verbosity_accepted() {
        for format in [
            json!({"type":"text"}),
            json!({"type":"json_object"}),
            json!({"type":"json_schema","name":"out","schema":{"type":"object"},"strict":true}),
        ] {
            let cfg: TextConfig =
                serde_json::from_value(json!({"format": format.clone()})).unwrap();
            assert_eq!(cfg.format.as_ref().unwrap()["type"], format["type"]);
        }
    }

    // ----- Usage shape -----------------------------------------------------

    /// `ResponsesUsage` carries the Responses field names and the nested
    /// details objects, round-trips, and NEVER serializes chat names.
    #[test]
    fn responses_usage_shape_round_trips() {
        let usage = ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: Some(InputTokensDetails { cached_tokens: 4 }),
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 2,
            }),
            total_tokens: 15,
        };
        let v = serde_json::to_value(&usage).unwrap();
        assert_eq!(v["input_tokens"], 10);
        assert_eq!(v["output_tokens"], 5);
        assert_eq!(v["total_tokens"], 15);
        assert_eq!(v["input_tokens_details"]["cached_tokens"], 4);
        assert_eq!(v["output_tokens_details"]["reasoning_tokens"], 2);
        assert!(v.get("prompt_tokens").is_none());
        assert!(v.get("completion_tokens").is_none());

        // Deserialize the full documented shape (all required on the wire).
        let back: ResponsesUsage = serde_json::from_value(json!({
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 15
        }))
        .unwrap();
        assert_eq!(back.input_tokens, 10);
        assert_eq!(back.input_tokens_details.unwrap().cached_tokens, 4);
        assert_eq!(back.output_tokens_details.unwrap().reasoning_tokens, 2);
    }

    // ----- Response object always-present fields & status enum -------------

    /// The Response `status` field accepts every documented enum value
    /// (completed|failed|in_progress|cancelled|queued|incomplete). status is a
    /// `String`, so this locks that none are rejected when echoed back through
    /// a round-trip.
    #[test]
    fn response_status_enum_values_all_round_trip() {
        for status in [
            "completed",
            "failed",
            "in_progress",
            "cancelled",
            "queued",
            "incomplete",
        ] {
            let resp: ResponsesResponse = serde_json::from_value(json!({
                "id": "resp_1",
                "object": "response",
                "created_at": 1_700_000_000_i64,
                "status": status,
                "output": [],
                "usage": {"input_tokens":0,"output_tokens":0,"total_tokens":0},
                "model": "m"
            }))
            .unwrap();
            assert_eq!(resp.status, status);
            let back = serde_json::to_value(&resp).unwrap();
            assert_eq!(back["status"], status);
        }
    }

    /// `incomplete_details.reason` (max_output_tokens|content_filter|null) and
    /// the `error` object both round-trip on the Response object.
    #[test]
    fn response_incomplete_details_and_error_round_trip() {
        let resp: ResponsesResponse = serde_json::from_value(json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1_700_000_000_i64,
            "status": "incomplete",
            "output": [],
            "usage": {"input_tokens":0,"output_tokens":0,"total_tokens":0},
            "model": "m",
            "incomplete_details": {"reason": "max_output_tokens"},
            "error": {"code": "server_error", "message": "x"}
        }))
        .unwrap();
        let back = serde_json::to_value(&resp).unwrap();
        assert_eq!(back["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(back["error"]["code"], "server_error");
    }

    // ----- Request: lenient acceptance of EVERY documented create field -----

    /// The gateway's `ResponsesRequest` must LENIENTLY accept every documented
    /// create field — modeled fields land in typed fields, all others flow into
    /// the flattened `extra` map (controlled passthrough). NONE may cause a
    /// deserialize rejection. This is the codex/ai-sdk compatibility guard.
    #[test]
    fn every_documented_create_field_is_accepted() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "m",
            "input": "hi",
            "instructions": "be terse",
            "max_output_tokens": 256,
            "max_tool_calls": 4,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_logprobs": 3,
            "tools": [{"type":"function","name":"f"}],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "reasoning": {"effort":"high","summary":"auto"},
            "text": {"format":{"type":"text"},"verbosity":"low"},
            "include": ["reasoning.encrypted_content"],
            "metadata": {"trace":"abc"},
            "store": false,
            "previous_response_id": "resp_prev",
            "conversation": "conv_1",
            "stream": true,
            "stream_options": {"include_obfuscation": false},
            "truncation": "auto",
            "background": false,
            "prompt": {"id":"pmpt_1","version":"1"},
            "prompt_cache_key": "session_123",
            "prompt_cache_retention": "24h",
            "safety_identifier": "user_hash",
            "service_tier": "auto",
            "context_management": {"strategy":"auto"},
            "moderation": {"enabled": true},
            "user": "legacy_user"
        }))
        .expect("every documented create field must be leniently accepted");

        // Modeled fields land in typed slots.
        assert_eq!(req.model, "m");
        assert_eq!(req.instructions.as_deref(), Some("be terse"));
        assert_eq!(req.max_output_tokens, Some(256));
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.parallel_tool_calls, Some(true));
        assert_eq!(req.store, Some(false));
        assert_eq!(req.previous_response_id.as_deref(), Some("resp_prev"));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.include.as_ref().unwrap().len(), 1);
        assert!(req.reasoning.is_some());
        assert!(req.text.is_some());
        assert!(req.tools.is_some());
        assert!(req.tool_choice.is_some());
        assert!(req.metadata.is_some());

        // Unmodeled documented fields flow through `extra` (controlled
        // passthrough) — never rejected, never invented top-level fields.
        for (key, expected) in [
            ("max_tool_calls", json!(4)),
            ("top_logprobs", json!(3)),
            ("conversation", json!("conv_1")),
            ("stream_options", json!({"include_obfuscation": false})),
            ("truncation", json!("auto")),
            ("background", json!(false)),
            ("prompt", json!({"id":"pmpt_1","version":"1"})),
            ("prompt_cache_key", json!("session_123")),
            ("prompt_cache_retention", json!("24h")),
            ("safety_identifier", json!("user_hash")),
            ("service_tier", json!("auto")),
            ("context_management", json!({"strategy":"auto"})),
            ("moderation", json!({"enabled": true})),
            ("user", json!("legacy_user")),
        ] {
            assert_eq!(
                req.extra.get(key),
                Some(&expected),
                "documented create field `{key}` must pass through `extra`"
            );
        }
    }

    /// `truncation` accepts both documented values (auto|disabled).
    #[test]
    fn truncation_values_accepted() {
        for t in ["auto", "disabled"] {
            let req: ResponsesRequest =
                serde_json::from_value(json!({"model":"m","input":"hi","truncation":t})).unwrap();
            assert_eq!(req.extra["truncation"], t);
        }
    }

    /// `service_tier` accepts every documented value (auto|default|flex|scale|
    /// priority).
    #[test]
    fn service_tier_values_accepted() {
        for tier in ["auto", "default", "flex", "scale", "priority"] {
            let req: ResponsesRequest =
                serde_json::from_value(json!({"model":"m","input":"hi","service_tier":tier}))
                    .unwrap();
            assert_eq!(req.extra["service_tier"], tier);
        }
    }

    /// `prompt_cache_retention` accepts both documented values (in_memory|24h).
    #[test]
    fn prompt_cache_retention_values_accepted() {
        for r in ["in_memory", "24h"] {
            let req: ResponsesRequest = serde_json::from_value(
                json!({"model":"m","input":"hi","prompt_cache_retention":r}),
            )
            .unwrap();
            assert_eq!(req.extra["prompt_cache_retention"], r);
        }
    }

    /// Every documented `include` enum value is accepted in the `include` array.
    #[test]
    fn include_enum_values_all_accepted() {
        let includes = json!([
            "file_search_call.results",
            "web_search_call.results",
            "web_search_call.action.sources",
            "message.input_image.image_url",
            "computer_call_output.output.image_url",
            "code_interpreter_call.outputs",
            "reasoning.encrypted_content",
            "message.output_text.logprobs"
        ]);
        let req: ResponsesRequest =
            serde_json::from_value(json!({"model":"m","input":"hi","include":includes})).unwrap();
        assert_eq!(req.include.as_ref().unwrap().len(), 8);
    }

    /// A request with `stream: Literal[True]` (the streaming SDK variant)
    /// deserializes with `stream == Some(true)`.
    #[test]
    fn stream_true_variant_deserializes() {
        let req: ResponsesRequest =
            serde_json::from_value(json!({"model":"m","input":"hi","stream":true})).unwrap();
        assert_eq!(req.stream, Some(true));
    }
}
