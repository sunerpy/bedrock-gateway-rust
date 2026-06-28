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
mod tests {
    use super::*;

    fn str_req(model: &str, text: &str, fmt: EncodingFormat) -> EmbeddingsRequest {
        EmbeddingsRequest {
            input: EmbeddingInput::String(text.to_string()),
            model: model.to_string(),
            encoding_format: fmt,
            dimensions: None,
            user: None,
        }
    }

    /// Cohere encode produces the documented JSON body
    /// (bedrock.py:1961-1965).
    #[test]
    fn cohere_encode_produces_expected_json() {
        let req = str_req(
            "cohere.embed-english-v3",
            "hello world",
            EncodingFormat::Float,
        );
        let body = CohereCodec.encode(&req).expect("encode ok");
        assert_eq!(body["texts"], json!(["hello world"]));
        assert_eq!(body["input_type"], "search_document");
        assert_eq!(body["truncate"], "END");
    }

    /// Cohere encode accepts a string array as multiple texts.
    #[test]
    fn cohere_encode_string_array() {
        let req = EmbeddingsRequest {
            input: EmbeddingInput::StringArray(vec!["a".to_string(), "b".to_string()]),
            model: "cohere.embed-english-v3".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
            user: None,
        };
        let body = CohereCodec.encode(&req).expect("encode ok");
        assert_eq!(body["texts"], json!(["a", "b"]));
    }

    /// Cohere decode parses `{"embeddings":[[...],...]}` into Vec<Vec<f32>>.
    #[test]
    fn cohere_decode_sample_response() {
        let sample = br#"{"embeddings":[[0.1,0.2,0.3],[0.4,0.5,0.6]]}"#;
        let out = CohereCodec.decode(sample).expect("decode ok");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![0.1f32, 0.2, 0.3]);
        assert_eq!(out[1], vec![0.4f32, 0.5, 0.6]);
    }

    /// Titan encode produces `{"inputText": ...}` for a single string
    /// (bedrock.py:1989-1992).
    #[test]
    fn titan_encode_single_string() {
        let req = str_req(
            "amazon.titan-embed-text-v1",
            "embed me",
            EncodingFormat::Float,
        );
        let body = TitanCodec.encode(&req).expect("encode ok");
        assert_eq!(body, json!({ "inputText": "embed me" }));
    }

    /// Titan encode accepts a single-element string array.
    #[test]
    fn titan_encode_single_element_array() {
        let req = EmbeddingsRequest {
            input: EmbeddingInput::StringArray(vec!["only".to_string()]),
            model: "amazon.titan-embed-text-v1".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
            user: None,
        };
        let body = TitanCodec.encode(&req).expect("encode ok");
        assert_eq!(body, json!({ "inputText": "only" }));
    }

    /// Titan rejects multi-element input (bedrock.py:1988).
    #[test]
    fn titan_encode_rejects_multi_input() {
        let req = EmbeddingsRequest {
            input: EmbeddingInput::StringArray(vec!["a".to_string(), "b".to_string()]),
            model: "amazon.titan-embed-text-v1".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
            user: None,
        };
        let err = TitanCodec.encode(&req).expect_err("must reject");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    /// Titan decode wraps the single vector into a length-1 outer vector
    /// (bedrock.py:2008).
    #[test]
    fn titan_decode_sample_response() {
        let sample = br#"{"embedding":[1.0,2.0,3.0],"inputTextTokenCount":5}"#;
        let out = TitanCodec.decode(sample).expect("decode ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![1.0f32, 2.0, 3.0]);
    }

    /// Nova encode produces the SINGLE_EMBEDDING body with default dimension.
    #[test]
    fn nova_encode_default_dimension() {
        let req = str_req(
            "amazon.nova-2-multimodal-embeddings-v1:0",
            "nova text",
            EncodingFormat::Float,
        );
        let body = NovaCodec.encode(&req).expect("encode ok");
        assert_eq!(body["taskType"], "SINGLE_EMBEDDING");
        let params = &body["singleEmbeddingParams"];
        assert_eq!(params["embeddingPurpose"], "GENERIC_INDEX");
        assert_eq!(params["embeddingDimension"], NOVA_DEFAULT_DIMENSION);
        assert_eq!(params["text"]["truncationMode"], "END");
        assert_eq!(params["text"]["value"], "nova text");
    }

    /// Nova rejects invalid dimensions (bedrock.py:2061-2065).
    #[test]
    fn nova_encode_rejects_invalid_dimension() {
        let req = EmbeddingsRequest {
            input: EmbeddingInput::String("x".to_string()),
            model: "amazon.nova-2-multimodal-embeddings-v1:0".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: Some(999),
            user: None,
        };
        let err = NovaCodec.encode(&req).expect_err("must reject");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    /// Nova decode extracts the first embedding item.
    #[test]
    fn nova_decode_sample_response() {
        let sample = br#"{"embeddings":[{"embeddingType":"TEXT","embedding":[0.5,0.6]}]}"#;
        let out = NovaCodec.decode(sample).expect("decode ok");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![0.5f32, 0.6]);
    }

    /// base64 encoding round-trips to identical f32s (little-endian),
    /// matching the NumPy tobytes() path (bedrock.py:1918-1922).
    #[test]
    fn base64_round_trips_to_identical_f32s() {
        let original: Vec<f32> = vec![0.1, -2.5, 3.5, 0.0, 42.0];
        let data = build_data(vec![original.clone()], EncodingFormat::Base64);
        assert_eq!(data.len(), 1);
        let encoded = match &data[0].embedding {
            EmbeddingData::Base64(s) => s.clone(),
            EmbeddingData::Float(_) => panic!("expected base64 data"),
        };
        let bytes = BASE64_STANDARD.decode(&encoded).expect("valid base64");
        assert_eq!(bytes.len(), original.len() * 4);
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, original);
    }

    /// Float format yields raw Vec<f32> without base64.
    #[test]
    fn float_format_yields_raw_vec() {
        let data = build_data(vec![vec![1.0, 2.0]], EncodingFormat::Float);
        match &data[0].embedding {
            EmbeddingData::Float(v) => assert_eq!(*v, vec![1.0f32, 2.0]),
            EmbeddingData::Base64(_) => panic!("expected float data"),
        }
        assert_eq!(data[0].index, 0);
    }

    /// Token-array input decodes to text via tiktoken cl100k (parity with the
    /// Python ENCODER.decode workaround).
    #[test]
    fn cohere_encode_decodes_token_array() {
        // Encode "hello" with cl100k, then ensure encode() round-trips via decode.
        let encoder = cl100k_base_singleton();
        let tokens: Vec<i32> = encoder
            .encode_with_special_tokens("hello")
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let req = EmbeddingsRequest {
            input: EmbeddingInput::IntArray(tokens),
            model: "cohere.embed-english-v3".to_string(),
            encoding_format: EncodingFormat::Float,
            dimensions: None,
            user: None,
        };
        let body = CohereCodec.encode(&req).expect("encode ok");
        assert_eq!(body["texts"], json!(["hello"]));
    }

    /// Unknown model id → AppError (not panic) via the registry-driven provider.
    #[tokio::test]
    async fn unknown_model_yields_app_error_not_panic() {
        // Build a provider with an empty registry so any model is "unknown".
        let settings = crate::config::AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: "us-west-2".to_string(),
            default_model: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
            default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
            enable_cross_region_inference: true,
            enable_application_inference_profiles: true,
            enable_prompt_caching: false,
            api_key: None,
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: None,
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
        };
        let clients = BedrockClients::from_settings(&settings).await;
        let provider = BedrockEmbeddingProvider::new(clients, EmbeddingRegistry::default());
        let req = str_req("nonexistent.model", "hi", EncodingFormat::Float);
        let err = provider.embed(&req).await.expect_err("must error");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    /// Codec selection is registry-family-driven (no hardcoded model table).
    #[test]
    fn codec_selection_by_family() {
        // Each family maps to a codec that encodes its family-specific body.
        let cohere = codec_for(EmbeddingFamily::Cohere);
        let req = str_req("any", "x", EncodingFormat::Float);
        let body = cohere.encode(&req).expect("cohere encode");
        assert!(body.get("texts").is_some());

        let titan = codec_for(EmbeddingFamily::Titan);
        let body = titan.encode(&req).expect("titan encode");
        assert!(body.get("inputText").is_some());

        let nova = codec_for(EmbeddingFamily::Nova);
        let body = nova.encode(&req).expect("nova encode");
        assert_eq!(body["taskType"], "SINGLE_EMBEDDING");
    }
}
