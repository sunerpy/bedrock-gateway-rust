//! Unit and property-based tests for the Bedrock embedding codecs and provider.
//!
//! Relocated out of `embeddings.rs` for code organization (see the
//! `test-coverage-codecov` spec, Task 3.8). Behavior is unchanged; the flat
//! example/unit tests live at the top of this module and the property tests are
//! preserved verbatim in the nested `prop_tests` submodule so the source file
//! references exactly one `mod tests;`.
//!
//! The `prop_tests` submodule sits one level deeper than the implementation
//! module, so its glob import is `use super::super::*;` to keep resolving to the
//! `embeddings` implementation module.

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
        prompt_cache_ttl: "5m".to_string(),
        api_key: None,
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        bind_addr: "0.0.0.0".to_string(),
        port: 8080,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        aws_max_retry_attempts: 8,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
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

/// `BedrockEmbeddingProvider` request construction: for a model registered
/// in the [`EmbeddingRegistry`], the provider selects the matching family
/// codec and builds the family-specific request body. This exercises the
/// exact registry-driven encode path used inside
/// [`EmbeddingProvider::embed`] (registry family lookup → `codec_for` →
/// `encode`) without invoking AWS, so it runs offline with no credentials.
#[test]
fn provider_request_construction_by_family() {
    let registry = EmbeddingRegistry::from_toml_str(
        "[[model]]\nmodel_id = \"cohere.embed-english-v3\"\ndisplay_name = \"C\"\nfamily = \"cohere\"\n\
         [[model]]\nmodel_id = \"amazon.titan-embed-text-v1\"\ndisplay_name = \"T\"\nfamily = \"titan\"\n\
         [[model]]\nmodel_id = \"amazon.nova-2-multimodal-embeddings-v1:0\"\ndisplay_name = \"N\"\nfamily = \"nova\"\n",
    )
    .expect("registry parse");

    // Helper mirroring the provider's internal path for a registered model.
    let build_body = |model: &str, text: &str| {
        let req = str_req(model, text, EncodingFormat::Float);
        let family = registry
            .family_for(&req.model)
            .expect("model must be registered");
        codec_for(family).encode(&req).expect("encode ok")
    };

    // Cohere → texts body.
    let cohere_body = build_body("cohere.embed-english-v3", "hello");
    assert_eq!(cohere_body["texts"], json!(["hello"]));
    assert_eq!(cohere_body["input_type"], "search_document");

    // Titan → single inputText body.
    let titan_body = build_body("amazon.titan-embed-text-v1", "hello");
    assert_eq!(titan_body, json!({ "inputText": "hello" }));

    // Nova → SINGLE_EMBEDDING body.
    let nova_body = build_body("amazon.nova-2-multimodal-embeddings-v1:0", "hello");
    assert_eq!(nova_body["taskType"], "SINGLE_EMBEDDING");
    assert_eq!(nova_body["singleEmbeddingParams"]["text"]["value"], "hello");

    // Unknown model is not registered → provider would return BadRequest.
    assert!(registry.family_for("nonexistent.model").is_none());
}

/// `inputs_to_texts` decodes an integer MATRIX (`Iterable[Iterable[int]]`) into
/// one decoded string per inner token row. Exercised through `CohereCodec`
/// (which fans every text out into the `texts` array), covering the
/// `EmbeddingInput::IntMatrix` arm of `inputs_to_texts`.
#[test]
fn cohere_encode_decodes_token_matrix() {
    let encoder = cl100k_base_singleton();
    let row = |s: &str| -> Vec<i32> {
        encoder
            .encode_with_special_tokens(s)
            .into_iter()
            .map(|t| t as i32)
            .collect()
    };
    let req = EmbeddingsRequest {
        input: EmbeddingInput::IntMatrix(vec![row("hello"), row("world")]),
        model: "cohere.embed-english-v3".to_string(),
        encoding_format: EncodingFormat::Float,
        dimensions: None,
        user: None,
    };
    let body = CohereCodec.encode(&req).expect("encode ok");
    assert_eq!(body["texts"], json!(["hello", "world"]));
}

/// Titan encode decodes a single token ARRAY into the `inputText` field,
/// covering the `EmbeddingInput::IntArray` arm of `TitanCodec::encode`.
#[test]
fn titan_encode_decodes_token_array() {
    let encoder = cl100k_base_singleton();
    let tokens: Vec<i32> = encoder
        .encode_with_special_tokens("titan tokens")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let req = EmbeddingsRequest {
        input: EmbeddingInput::IntArray(tokens),
        model: "amazon.titan-embed-text-v1".to_string(),
        encoding_format: EncodingFormat::Float,
        dimensions: None,
        user: None,
    };
    let body = TitanCodec.encode(&req).expect("encode ok");
    assert_eq!(body, json!({ "inputText": "titan tokens" }));
}

/// Titan encode decodes a SINGLE-ROW token matrix into `inputText`, covering
/// the `EmbeddingInput::IntMatrix if rows.len() == 1` arm of
/// `TitanCodec::encode`.
#[test]
fn titan_encode_decodes_single_row_token_matrix() {
    let encoder = cl100k_base_singleton();
    let tokens: Vec<i32> = encoder
        .encode_with_special_tokens("single row")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let req = EmbeddingsRequest {
        input: EmbeddingInput::IntMatrix(vec![tokens]),
        model: "amazon.titan-embed-text-v1".to_string(),
        encoding_format: EncodingFormat::Float,
        dimensions: None,
        user: None,
    };
    let body = TitanCodec.encode(&req).expect("encode ok");
    assert_eq!(body, json!({ "inputText": "single row" }));
}

/// Titan rejects a MULTI-row token matrix (the `_` reject arm), mirroring the
/// multi-string rejection: only a single input is supported.
#[test]
fn titan_encode_rejects_multi_row_token_matrix() {
    let encoder = cl100k_base_singleton();
    let row: Vec<i32> = encoder
        .encode_with_special_tokens("row")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let req = EmbeddingsRequest {
        input: EmbeddingInput::IntMatrix(vec![row.clone(), row]),
        model: "amazon.titan-embed-text-v1".to_string(),
        encoding_format: EncodingFormat::Float,
        dimensions: None,
        user: None,
    };
    let err = TitanCodec.encode(&req).expect_err("must reject multi-row");
    assert!(matches!(err, AppError::BadRequest(_)));
}

/// Nova decode of an EMPTY `embeddings` array yields an `UpstreamBedrock` error,
/// covering the `ok_or_else` branch of `NovaCodec::decode`.
#[test]
fn nova_decode_empty_embeddings_is_upstream_error() {
    let sample = br#"{"embeddings":[]}"#;
    let err = NovaCodec
        .decode(sample)
        .expect_err("empty embeddings must error");
    assert!(matches!(err, AppError::UpstreamBedrock(_)));
}

/// `estimate_tokens` sums cl100k token counts across every input and returns 0
/// for an empty input list (best-effort token accounting helper).
#[test]
fn estimate_tokens_sums_across_inputs() {
    // Empty list → zero tokens.
    assert_eq!(estimate_tokens(&[]), 0);

    // A non-empty string contributes a positive count.
    let one = estimate_tokens(&["hello world".to_string()]);
    assert!(one > 0);

    // The estimate is additive across the list: two identical inputs yield
    // exactly twice the single-input estimate.
    let two = estimate_tokens(&["hello world".to_string(), "hello world".to_string()]);
    assert_eq!(two, one * 2);
}

/// Property-based tests for the per-family embedding codec round-trips.
///
/// Feature: test-coverage-codecov, Property: embedding-codec-roundtrip
/// (see `.kiro/specs/test-coverage-codecov/design.md`, Task 3.8).
///
/// Validates: Requirements 1.2
mod prop_tests {
    use super::super::*;
    use proptest::prelude::*;

    /// Strategy: a non-empty vector of finite f32 values within a bounded range.
    ///
    /// Bounding to `[-1000, 1000)` keeps every value finite (no NaN / ±Inf /
    /// subnormal edge cases) so the JSON `f32 → f64 → f32` round-trip through
    /// `serde_json` is exact; the small dimensionality keeps the property fast
    /// and fully offline.
    fn embedding_vec() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1_000.0f32..1_000.0f32, 1..16)
    }

    /// Strategy: a non-empty matrix of embedding vectors (multi-vector responses).
    fn embedding_matrix() -> impl Strategy<Value = Vec<Vec<f32>>> {
        prop::collection::vec(embedding_vec(), 1..8)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov, Property: embedding-codec-roundtrip
        ///
        /// For any embedding matrix serialized into the Cohere response body
        /// shape `{"embeddings":[[...],...]}`, decoding with [`CohereCodec`]
        /// yields a semantically equivalent matrix.
        #[test]
        fn cohere_codec_roundtrip(embeddings in embedding_matrix()) {
            let body = serde_json::to_vec(&json!({ "embeddings": embeddings.clone() }))
                .expect("serialize cohere body");
            let decoded = CohereCodec.decode(&body).expect("cohere decode ok");
            prop_assert_eq!(decoded, embeddings);
        }

        /// Feature: test-coverage-codecov, Property: embedding-codec-roundtrip
        ///
        /// For any single embedding vector serialized into the Titan response
        /// body shape `{"embedding":[...], "inputTextTokenCount":n}`, decoding
        /// with [`TitanCodec`] yields that vector wrapped in a length-1 outer
        /// vector.
        #[test]
        fn titan_codec_roundtrip(
            embedding in embedding_vec(),
            token_count in 0i64..1_000_000,
        ) {
            let body = serde_json::to_vec(&json!({
                "embedding": embedding.clone(),
                "inputTextTokenCount": token_count,
            }))
            .expect("serialize titan body");
            let decoded = TitanCodec.decode(&body).expect("titan decode ok");
            prop_assert_eq!(decoded, vec![embedding]);
        }

        /// Feature: test-coverage-codecov, Property: embedding-codec-roundtrip
        ///
        /// For any single embedding vector serialized into the Nova response
        /// body shape `{"embeddings":[{"embeddingType":"TEXT","embedding":[...]}]}`,
        /// decoding with [`NovaCodec`] yields the (only) vector wrapped in a
        /// length-1 outer vector.
        #[test]
        fn nova_codec_roundtrip(embedding in embedding_vec()) {
            let body = serde_json::to_vec(&json!({
                "embeddings": [ { "embeddingType": "TEXT", "embedding": embedding.clone() } ]
            }))
            .expect("serialize nova body");
            let decoded = NovaCodec.decode(&body).expect("nova decode ok");
            prop_assert_eq!(decoded, vec![embedding]);
        }
    }
}
