//! Bedrock embedding providers and per-family body codecs.
//!
//! This module ports the legacy Python `BedrockEmbeddingsModel` hierarchy
//! (`.legacy-python/src/api/models/bedrock.py` lines 1881-2114) into Rust:
//!
//! - [`CohereCodec`] — Cohere embed family (bedrock.py:1938-1978). Request body
//!   `{"texts":[...], "input_type":"search_document", "truncate":"END"}`,
//!   response `{"embeddings":[[f32,...],...]}`.
//! - [`TitanCodec`] — Amazon Titan embed family (bedrock.py:1981-2011). Request
//!   body `{"inputText": <single string>}`, response
//!   `{"embedding":[f32,...], "inputTextTokenCount":n}` (single vector).
//! - [`NovaCodec`] — Amazon Nova multimodal embeddings (bedrock.py:2014-2095).
//!   Request body per the Nova embeddings schema
//!   (<https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html>),
//!   response `{"embeddings":[{"embeddingType":"TEXT","embedding":[...]}]}`.
//! - [`BedrockEmbeddingProvider`] — implements [`crate::domain::EmbeddingProvider`]:
//!   selects a codec via the externalized [`EmbeddingRegistry`] (no hardcoded
//!   model list), calls `invoke_model`, maps SDK errors via
//!   [`from_bedrock_sdk_error`], and builds an [`EmbeddingsResponse`] honoring
//!   the requested `encoding_format` (`float` → raw `Vec<f32>`; `base64` →
//!   little-endian f32 bytes → base64, matching the NumPy `tobytes()` path at
//!   bedrock.py:1918-1922).
//!
//! DE-HARDCODING CONTRACT: this module contains NO model-id table. The mapping
//! from model id → [`EmbeddingFamily`] lives entirely in the
//! [`EmbeddingRegistry`] (loaded from `config/embeddings.toml`). The per-family
//! body shapes here are *protocol* serialization, not model knowledge.

use async_trait::async_trait;
use aws_smithy_types::Blob;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use tiktoken_rs::cl100k_base_singleton;

use crate::config::embeddings::EmbeddingFamily;
use crate::config::EmbeddingRegistry;
use crate::domain::{EmbeddingBodyCodec, EmbeddingProvider};
use crate::error::{from_bedrock_sdk_error, AppError};
use crate::openai::schema::{
    Embedding, EmbeddingData, EmbeddingInput, EmbeddingsRequest, EmbeddingsResponse,
    EmbeddingsUsage, EncodingFormat,
};

use super::client::BedrockClients;

/// Decode a single token array (`Iterable[int]`) into text using the
/// cl100k_base encoding, mirroring the Python `ENCODER.decode(...)` workaround
/// for encoded embedding inputs (bedrock.py:1947-1958).
fn decode_tokens(tokens: &[i32]) -> Result<String, AppError> {
    let ranks: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    cl100k_base_singleton()
        .decode(&ranks)
        .map_err(|e| AppError::BadRequest(format!("failed to decode token input: {e}")))
}

/// Normalize an [`EmbeddingInput`] into a list of text strings.
///
/// Mirrors the Cohere/Nova input handling (bedrock.py:1939-1958, 2036-2056):
/// strings pass through; integer arrays are treated as a single token sequence
/// decoded via tiktoken; an integer matrix decodes each inner array to one
/// string.
fn inputs_to_texts(input: &EmbeddingInput) -> Result<Vec<String>, AppError> {
    match input {
        EmbeddingInput::String(s) => Ok(vec![s.clone()]),
        EmbeddingInput::StringArray(v) => Ok(v.clone()),
        // Iterable[int]: a single token sequence → one decoded string.
        EmbeddingInput::IntArray(tokens) => Ok(vec![decode_tokens(tokens)?]),
        // Iterable[Iterable[int]]: each inner array decodes to its own string.
        EmbeddingInput::IntMatrix(rows) => rows.iter().map(|row| decode_tokens(row)).collect(),
    }
}

/// Encode embedding vectors into the OpenAI-shaped [`Embedding`] list, honoring
/// the requested `encoding_format`.
///
/// - [`EncodingFormat::Float`] → raw `Vec<f32>` ([`EmbeddingData::Float`]).
/// - [`EncodingFormat::Base64`] → little-endian f32 bytes, base64-encoded
///   ([`EmbeddingData::Base64`]). This matches the Python NumPy
///   `np.array(embedding, dtype=np.float32).tobytes()` → `base64.b64encode`
///   path (bedrock.py:1918-1922).
fn build_data(embeddings: Vec<Vec<f32>>, encoding_format: EncodingFormat) -> Vec<Embedding> {
    embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| {
            let data = match encoding_format {
                EncodingFormat::Float => EmbeddingData::Float(embedding),
                EncodingFormat::Base64 => {
                    let mut bytes = Vec::with_capacity(embedding.len() * 4);
                    for value in &embedding {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    EmbeddingData::Base64(BASE64_STANDARD.encode(&bytes))
                }
            };
            Embedding {
                object: "embedding".to_string(),
                embedding: data,
                index: i as i32,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cohere
// ---------------------------------------------------------------------------

/// Cohere embed family codec (bedrock.py:1938-1978).
#[derive(Debug, Default, Clone, Copy)]
pub struct CohereCodec;

/// Cohere response body shape: `{"embeddings":[[f32,...],...]}`.
#[derive(Debug, Deserialize)]
struct CohereResponse {
    embeddings: Vec<Vec<f32>>,
}

impl EmbeddingBodyCodec for CohereCodec {
    fn encode(&self, req: &EmbeddingsRequest) -> Result<Value, AppError> {
        let texts = inputs_to_texts(&req.input)?;
        Ok(json!({
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END",
        }))
    }

    fn decode(&self, body: &[u8]) -> Result<Vec<Vec<f32>>, AppError> {
        let parsed: CohereResponse = serde_json::from_slice(body)
            .map_err(|e| AppError::Internal(format!("invalid Cohere embeddings response: {e}")))?;
        Ok(parsed.embeddings)
    }
}

// ---------------------------------------------------------------------------
// Titan
// ---------------------------------------------------------------------------

/// Amazon Titan embed family codec (bedrock.py:1981-2011).
#[derive(Debug, Default, Clone, Copy)]
pub struct TitanCodec;

/// Titan response body shape:
/// `{"embedding":[f32,...], "inputTextTokenCount":n}`.
#[derive(Debug, Deserialize)]
struct TitanResponse {
    embedding: Vec<f32>,
}

impl EmbeddingBodyCodec for TitanCodec {
    fn encode(&self, req: &EmbeddingsRequest) -> Result<Value, AppError> {
        // Titan supports only a single input string (bedrock.py:1982-1988).
        let input_text = match &req.input {
            EmbeddingInput::String(s) => s.clone(),
            EmbeddingInput::StringArray(v) if v.len() == 1 => v[0].clone(),
            EmbeddingInput::IntArray(tokens) => decode_tokens(tokens)?,
            EmbeddingInput::IntMatrix(rows) if rows.len() == 1 => decode_tokens(&rows[0])?,
            _ => {
                return Err(AppError::BadRequest(
                    "Amazon Titan Embeddings models support only single strings as input."
                        .to_string(),
                ));
            }
        };
        Ok(json!({ "inputText": input_text }))
    }

    fn decode(&self, body: &[u8]) -> Result<Vec<Vec<f32>>, AppError> {
        let parsed: TitanResponse = serde_json::from_slice(body)
            .map_err(|e| AppError::Internal(format!("invalid Titan embeddings response: {e}")))?;
        // Single vector → Vec<Vec<f32>> of length 1 (bedrock.py:2008).
        Ok(vec![parsed.embedding])
    }
}

// ---------------------------------------------------------------------------
// Nova
// ---------------------------------------------------------------------------

/// Default Nova embedding dimension (bedrock.py:2017).
const NOVA_DEFAULT_DIMENSION: i32 = 3072;
/// Valid Nova embedding dimensions (bedrock.py:2016).
const NOVA_VALID_DIMENSIONS: [i32; 4] = [256, 384, 1024, 3072];

/// Amazon Nova multimodal embeddings codec (bedrock.py:2014-2095).
///
/// Per the Nova embeddings schema, a `SINGLE_EMBEDDING` request embeds one text
/// value. Since the codec abstraction encodes one request body, this codec
/// embeds the **first** text input; the registry-driven provider is responsible
/// for any multi-input fan-out parity if required. The request body assumptions
/// are documented inline and follow the AWS Nova embeddings schema.
#[derive(Debug, Default, Clone, Copy)]
pub struct NovaCodec;

/// Nova response: `{"embeddings":[{"embeddingType":"TEXT","embedding":[...]}]}`.
#[derive(Debug, Deserialize)]
struct NovaResponse {
    embeddings: Vec<NovaEmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct NovaEmbeddingItem {
    embedding: Vec<f32>,
}

impl NovaCodec {
    /// Resolve and validate the requested embedding dimension.
    fn resolve_dimension(dimensions: Option<i32>) -> Result<i32, AppError> {
        let dim = dimensions.unwrap_or(NOVA_DEFAULT_DIMENSION);
        if !NOVA_VALID_DIMENSIONS.contains(&dim) {
            return Err(AppError::BadRequest(format!(
                "Invalid dimensions {dim}. Must be one of {NOVA_VALID_DIMENSIONS:?}"
            )));
        }
        Ok(dim)
    }
}

impl EmbeddingBodyCodec for NovaCodec {
    fn encode(&self, req: &EmbeddingsRequest) -> Result<Value, AppError> {
        let texts = inputs_to_texts(&req.input)?;
        let text = texts
            .into_iter()
            .next()
            .ok_or_else(|| AppError::BadRequest("Input list cannot be empty".to_string()))?;
        let dim = Self::resolve_dimension(req.dimensions)?;
        // Nova SINGLE_EMBEDDING request body (bedrock.py:2019-2033).
        Ok(json!({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dim,
                "text": {
                    "truncationMode": "END",
                    "value": text,
                },
            },
        }))
    }

    fn decode(&self, body: &[u8]) -> Result<Vec<Vec<f32>>, AppError> {
        let parsed: NovaResponse = serde_json::from_slice(body)
            .map_err(|e| AppError::Internal(format!("invalid Nova embeddings response: {e}")))?;
        let first = parsed.embeddings.into_iter().next().ok_or_else(|| {
            AppError::UpstreamBedrock("No embeddings returned from Nova model".to_string())
        })?;
        Ok(vec![first.embedding])
    }
}

// ---------------------------------------------------------------------------
// Codec selection
// ---------------------------------------------------------------------------

/// Select the boxed [`EmbeddingBodyCodec`] for an embedding family.
fn codec_for(family: EmbeddingFamily) -> Box<dyn EmbeddingBodyCodec> {
    match family {
        EmbeddingFamily::Cohere => Box::new(CohereCodec),
        EmbeddingFamily::Titan => Box::new(TitanCodec),
        EmbeddingFamily::Nova => Box::new(NovaCodec),
    }
}

/// Estimate prompt tokens for an input using cl100k_base, mirroring the Python
/// approximation used where the model does not return a token count
/// (bedrock.py:2088). Best-effort: encoding failures contribute 0.
fn estimate_tokens(texts: &[String]) -> i32 {
    let encoder = cl100k_base_singleton();
    texts
        .iter()
        .map(|t| encoder.encode_with_special_tokens(t).len() as i32)
        .sum()
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Bedrock embeddings provider.
///
/// Holds the shared [`BedrockClients`] runtime client plus the externalized
/// [`EmbeddingRegistry`]. On [`embed`](EmbeddingProvider::embed) it looks up the
/// model's [`EmbeddingFamily`] via the registry (no hardcoded model list),
/// selects the matching codec, invokes the model, maps SDK errors, decodes the
/// response, and builds an [`EmbeddingsResponse`].
#[derive(Clone)]
pub struct BedrockEmbeddingProvider {
    clients: BedrockClients,
    registry: EmbeddingRegistry,
}

impl BedrockEmbeddingProvider {
    /// Construct a provider from shared Bedrock clients and the embedding
    /// registry.
    pub fn new(clients: BedrockClients, registry: EmbeddingRegistry) -> Self {
        Self { clients, registry }
    }
}

#[async_trait]
impl EmbeddingProvider for BedrockEmbeddingProvider {
    async fn embed(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse, AppError> {
        // Registry-driven family selection. Unknown models → AppError (no panic),
        // mirroring the Python "Unsupported embedding model id" 400.
        let family = self.registry.family_for(&req.model).ok_or_else(|| {
            AppError::BadRequest(format!("Unsupported embedding model id {}", req.model))
        })?;
        let codec = codec_for(family);

        // Pre-compute texts for token estimation (best-effort token accounting).
        let texts = inputs_to_texts(&req.input)?;
        let prompt_tokens = estimate_tokens(&texts);

        let body = codec.encode(req)?;
        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| AppError::Internal(format!("failed to serialize request body: {e}")))?;

        let response = self
            .clients
            .runtime
            .invoke_model()
            .model_id(&req.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body_bytes))
            .send()
            .await
            .map_err(|e| from_bedrock_sdk_error(&e.into_service_error()))?;

        let embeddings = codec.decode(response.body().as_ref())?;

        Ok(EmbeddingsResponse {
            object: "list".to_string(),
            data: build_data(embeddings, req.encoding_format),
            model: req.model.clone(),
            usage: EmbeddingsUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
        })
    }
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
