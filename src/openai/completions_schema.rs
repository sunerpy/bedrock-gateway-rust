use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AppError;
use crate::openai::schema::{StreamOptions, StringOrVec, Usage};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<i64>),
    TokenMatrix(Vec<Vec<i64>>),
}

impl CompletionPrompt {
    pub fn as_single_string(&self) -> Result<String, AppError> {
        match self {
            CompletionPrompt::Text(s) => Ok(s.clone()),
            CompletionPrompt::Texts(v) => Ok(v.join("\n")),
            CompletionPrompt::Tokens(_) | CompletionPrompt::TokenMatrix(_) => {
                Err(AppError::BadRequest(
                    "prompt as token arrays is not supported; send a string".to_string(),
                ))
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrVec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i32,
    pub logprobs: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    #[serde(default = "text_completion_object")]
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<CompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

fn text_completion_object() -> String {
    "text_completion".to_string()
}

#[cfg(test)]
#[path = "completions_schema_tests.rs"]
mod tests;
