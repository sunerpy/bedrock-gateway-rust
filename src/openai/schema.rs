//! OpenAI-compatible wire schema.
//!
//! These types mirror the Python `api/schema.py` Pydantic models EXACTLY and
//! form the API compatibility contract. They are pure data types — no
//! translation/mapping logic lives here.
//!
//! ## Option B guardrail
//!
//! [`ChatResponseMessage::reasoning_content`] is an internal-only field. It is
//! never serialized to the wire (it carries `skip_serializing_if` and must
//! always be `None` when a response is emitted). Reasoning is rendered inline
//! as `<think>...</think>` inside `content`. No non-OpenAI top-level keys ever
//! appear on response payloads.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A single model entry (`Model` in schema.py).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub created: i64,
    #[serde(default = "default_model_object")]
    pub object: String,
    #[serde(default = "default_owned_by")]
    pub owned_by: String,
}

fn default_model_object() -> String {
    "model".to_string()
}

fn default_owned_by() -> String {
    "bedrock".to_string()
}

/// List of models (`Models` in schema.py).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Models {
    #[serde(default = "default_list_object")]
    pub object: String,
    #[serde(default)]
    pub data: Vec<Model>,
}

fn default_list_object() -> String {
    "list".to_string()
}

// ---------------------------------------------------------------------------
// Tool calls / tools
// ---------------------------------------------------------------------------

/// Function payload echoed back inside a tool call (`ResponseFunction`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFunction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

/// A tool call produced by the assistant (`ToolCall`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_function_type")]
    pub r#type: String,
    pub function: ResponseFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

/// Function declaration in a request tool (`Function`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Arbitrary JSON Schema object (`object` in Python).
    pub parameters: Value,
}

/// A tool the model may call (`Tool`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_function_type")]
    pub r#type: String,
    pub function: Function,
}

/// `tool_choice` accepts either a string (e.g. "auto") or an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Object(Value),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::String("auto".to_string())
    }
}

// ---------------------------------------------------------------------------
// Content parts (multimodal)
// ---------------------------------------------------------------------------

/// Text content part (`TextContent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContent {
    #[serde(rename = "type", default = "default_text_type")]
    pub r#type: String,
    pub text: String,
}

fn default_text_type() -> String {
    "text".to_string()
}

/// Image URL payload (`ImageUrl`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default = "default_detail")]
    pub detail: String,
}

fn default_detail() -> String {
    "auto".to_string()
}

/// Image content part (`ImageContent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    #[serde(rename = "type", default = "default_image_type")]
    pub r#type: String,
    pub image_url: ImageUrl,
}

fn default_image_type() -> String {
    "image_url".to_string()
}

/// A single content part: text or image (`TextContent | ImageContent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
}

/// Message content: a plain string or a list of content parts
/// (`str | list[TextContent | ImageContent]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentInput {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Tool message content (`str | list[ToolContent] | list[dict]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolContentInput {
    Text(String),
    Parts(Vec<Value>),
}

// ---------------------------------------------------------------------------
// Request messages
// ---------------------------------------------------------------------------

/// A request message, internally tagged by `role`.
///
/// Mirrors `SystemMessage | UserMessage | AssistantMessage | ToolMessage |
/// DeveloperMessage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        content: String,
    },
    User {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        content: ContentInput,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ContentInput>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: ToolContentInput,
        tool_call_id: String,
    },
    Developer {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        content: String,
    },
}

// ---------------------------------------------------------------------------
// Reasoning effort
// ---------------------------------------------------------------------------

/// Reasoning effort levels.
///
/// The standard OpenAI set is `none | minimal | low | medium | high | xhigh`;
/// `max` is a Bedrock extension. Deserialization is permissive (unknown values
/// are tolerated by the surrounding `Option`/serde defaults at the request
/// level — invalid strings simply fail this enum and the field may be omitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    /// Bedrock extension.
    Max,
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

/// `stop` accepts either a single string or a list of strings
/// (`list[str] | str`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    String(String),
    Vec(Vec<String>),
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

/// Streaming options (`StreamOptions`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default = "default_true")]
    pub include_usage: bool,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Chat request
// ---------------------------------------------------------------------------

/// Chat completion request (`ChatRequest`, `extra="allow"`).
///
/// Unknown top-level fields are captured into [`ChatRequest::extra`] for
/// controlled passthrough (Option B): downstream only honors documented keys
/// and never blindly forwards the captured map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(
        default = "default_max_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrVec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Value>,
    /// Controlled passthrough of unknown top-level fields (`extra="allow"`).
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

fn default_max_tokens() -> Option<i32> {
    Some(2048)
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// Prompt token usage details (`PromptTokensDetails`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: i32,
    #[serde(default)]
    pub audio_tokens: i32,
}

/// Completion token usage details (`CompletionTokensDetails`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i32,
    #[serde(default)]
    pub audio_tokens: i32,
}

/// Token usage (`Usage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

// ---------------------------------------------------------------------------
// Response message / choices
// ---------------------------------------------------------------------------

/// The message inside a response choice (`ChatResponseMessage`).
///
/// All `Option` fields use `skip_serializing_if` to match Python's
/// `exclude_unset` behavior on stream deltas, yielding a clean wire shape.
///
/// `reasoning_content` is internal-only (Option B): it carries
/// `skip_serializing_if` and must always be `None` on the wire — reasoning is
/// rendered inline as `<think>` inside `content`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Internal-only. NEVER serialized with a value. See Option B guardrail.
    ///
    /// `skip_serializing` (unconditional) guarantees the field never reaches
    /// the wire even when populated; reasoning is rendered inline as `<think>`
    /// inside `content`. It still deserializes so internal pipelines can read
    /// upstream reasoning if a provider sends it.
    #[serde(default, skip_serializing)]
    pub reasoning_content: Option<String>,
}

/// A non-streaming choice (`Choice` extends `BaseChoice`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub message: ChatResponseMessage,
}

/// A streaming choice (`ChoiceDelta` extends `BaseChoice`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    #[serde(default)]
    pub index: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub delta: ChatResponseMessage,
}

// ---------------------------------------------------------------------------
// Chat responses
// ---------------------------------------------------------------------------

/// Non-streaming chat completion response (`ChatResponse`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub created: i64,
    pub model: String,
    #[serde(default = "default_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    #[serde(default = "default_chat_completion_object")]
    pub object: String,
    pub usage: Usage,
}

fn default_fingerprint() -> String {
    "fp".to_string()
}

fn default_chat_completion_object() -> String {
    "chat.completion".to_string()
}

/// Streaming chat completion chunk (`ChatStreamResponse`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamResponse {
    pub id: String,
    pub created: i64,
    pub model: String,
    #[serde(default = "default_fingerprint")]
    pub system_fingerprint: String,
    pub choices: Vec<ChoiceDelta>,
    #[serde(default = "default_chat_chunk_object")]
    pub object: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

fn default_chat_chunk_object() -> String {
    "chat.completion.chunk".to_string()
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// Embeddings input (`str | list[str] | Iterable[int | Iterable[int]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    StringArray(Vec<String>),
    IntArray(Vec<i32>),
    IntMatrix(Vec<Vec<i32>>),
}

/// Encoding format for embeddings (`Literal["float", "base64"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

/// Embeddings request (`EmbeddingsRequest`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingInput,
    pub model: String,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// A single embedding's data (`list[float] | bytes`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    Float(Vec<f32>),
    Base64(String),
}

/// A single embedding entry (`Embedding`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(default = "default_embedding_object")]
    pub object: String,
    pub embedding: EmbeddingData,
    pub index: i32,
}

fn default_embedding_object() -> String {
    "embedding".to_string()
}

/// Embeddings usage (`EmbeddingsUsage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: i32,
    pub total_tokens: i32,
}

/// Embeddings response (`EmbeddingsResponse`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    #[serde(default = "default_list_object")]
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

// ---------------------------------------------------------------------------
// Error envelope
// ---------------------------------------------------------------------------

/// Error body (`ErrorMessage` extended with OpenAI's standard fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Error envelope (`Error`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiError {
    pub error: ErrorBody,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Test A: deserialize a realistic OpenAI chat request, re-serialize, and
    /// assert key fields are preserved.
    #[test]
    fn openai_chat_request_roundtrips() {
        let raw = r#"{
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the weather?"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                ]}
            ],
            "temperature": 0.7,
            "tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }}
            ]
        }"#;

        let req: ChatRequest = serde_json::from_str(raw).expect("deserialize request");
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.temperature, Some(0.7));
        let tools = req.tools.as_ref().expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");

        // Re-serialize and parse back; verify key fields survive the round trip.
        let serialized = serde_json::to_string(&req).expect("serialize request");
        let reparsed: ChatRequest = serde_json::from_str(&serialized).expect("reparse request");
        assert_eq!(reparsed.model, "gpt-4o");
        assert_eq!(reparsed.messages.len(), 3);
        assert_eq!(reparsed.temperature, Some(0.7));
        assert_eq!(reparsed.tools.as_ref().expect("tools").len(), 1);
        match &reparsed.messages[0] {
            Message::System { content, .. } => assert_eq!(content, "You are helpful."),
            other => panic!("expected system message, got {other:?}"),
        }
    }

    /// Test A2: unknown top-level fields are captured into `extra` for
    /// controlled passthrough (Option B), not dropped or blindly merged.
    #[test]
    fn unknown_request_fields_captured_in_extra() {
        let raw = r#"{
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "some_vendor_flag": true,
            "another": {"nested": 1}
        }"#;

        let req: ChatRequest = serde_json::from_str(raw).expect("deserialize");
        assert!(req.extra.contains_key("some_vendor_flag"));
        assert!(req.extra.contains_key("another"));
        // Documented fields are NOT captured into extra.
        assert!(!req.extra.contains_key("model"));
        assert!(!req.extra.contains_key("messages"));
    }

    /// Test B: a ChatResponse whose message carries `reasoning_content` must
    /// serialize WITHOUT a `reasoning_content` key and WITHOUT any unknown
    /// top-level keys.
    #[test]
    fn reasoning_content_never_serializes() {
        let response = ChatResponse {
            id: "chatcmpl-123".to_string(),
            created: 1_700_000_000,
            model: "gpt-4o".to_string(),
            system_fingerprint: "fp".to_string(),
            choices: vec![Choice {
                index: 0,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                message: ChatResponseMessage {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                    // Internal reasoning is set, but MUST NOT reach the wire.
                    reasoning_content: Some("secret chain of thought".to_string()),
                },
            }],
            object: "chat.completion".to_string(),
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            },
        };

        let json = serde_json::to_string(&response).expect("serialize response");
        assert!(
            !json.contains("reasoning_content"),
            "reasoning_content leaked to wire: {json}"
        );
        assert!(!json.contains("secret chain of thought"));

        // Verify only OpenAI-recognized top-level keys are present.
        let value: Value = serde_json::from_str(&json).expect("parse json");
        let obj = value.as_object().expect("top-level object");
        let allowed = [
            "id",
            "created",
            "model",
            "system_fingerprint",
            "choices",
            "object",
            "usage",
        ];
        for key in obj.keys() {
            assert!(
                allowed.contains(&key.as_str()),
                "unexpected top-level key on wire: {key}"
            );
        }

        // And the message object itself must not carry reasoning_content.
        let msg = &value["choices"][0]["message"];
        let msg_obj = msg.as_object().expect("message object");
        assert!(!msg_obj.contains_key("reasoning_content"));
    }

    /// Defaults match schema.py.
    #[test]
    fn defaults_match_python() {
        assert!(matches!(ToolChoice::default(), ToolChoice::String(ref s) if s == "auto"));
        assert!(matches!(EncodingFormat::default(), EncodingFormat::Float));
        assert_eq!(default_max_tokens(), Some(2048));
        assert_eq!(default_fingerprint(), "fp");

        let opts: StreamOptions = serde_json::from_str("{}").expect("empty stream options");
        assert!(opts.include_usage);
    }

    /// ReasoningEffort deserializes all variants including the Bedrock `max`.
    #[test]
    fn reasoning_effort_variants() {
        for (raw, expected) in [
            ("\"none\"", ReasoningEffort::None),
            ("\"minimal\"", ReasoningEffort::Minimal),
            ("\"low\"", ReasoningEffort::Low),
            ("\"medium\"", ReasoningEffort::Medium),
            ("\"high\"", ReasoningEffort::High),
            ("\"xhigh\"", ReasoningEffort::Xhigh),
            ("\"max\"", ReasoningEffort::Max),
        ] {
            let got: ReasoningEffort = serde_json::from_str(raw).expect(raw);
            assert_eq!(got, expected);
        }
    }
}
