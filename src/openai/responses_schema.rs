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
#[path = "responses_schema_tests.rs"]
mod tests;
