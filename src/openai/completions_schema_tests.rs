use super::*;
use crate::openai::schema::Usage;
use serde_json::{json, Value};

#[test]
fn completion_prompt_deserializes_supported_shapes() {
    let text: CompletionPrompt = serde_json::from_value(json!("hello")).unwrap();
    let texts: CompletionPrompt = serde_json::from_value(json!(["a", "b"])).unwrap();
    let tokens: CompletionPrompt = serde_json::from_value(json!([1, 2])).unwrap();
    let token_matrix: CompletionPrompt = serde_json::from_value(json!([[1], [2]])).unwrap();

    assert!(matches!(text, CompletionPrompt::Text(_)));
    assert!(matches!(texts, CompletionPrompt::Texts(_)));
    assert!(matches!(tokens, CompletionPrompt::Tokens(_)));
    assert!(matches!(token_matrix, CompletionPrompt::TokenMatrix(_)));
}

#[test]
fn completion_prompt_as_single_string_rejects_tokens() {
    let text = CompletionPrompt::Text("hello".to_string());
    let texts = CompletionPrompt::Texts(vec!["a".to_string(), "b".to_string()]);
    let tokens = CompletionPrompt::Tokens(vec![1, 2]);
    let token_matrix = CompletionPrompt::TokenMatrix(vec![vec![1], vec![2]]);

    assert_eq!(text.as_single_string().unwrap(), "hello");
    assert_eq!(texts.as_single_string().unwrap(), "a\nb");
    // Both token-array shapes map to a 400 (send a string instead).
    assert!(matches!(
        tokens.as_single_string().unwrap_err(),
        AppError::BadRequest(_)
    ));
    assert!(matches!(
        token_matrix.as_single_string().unwrap_err(),
        AppError::BadRequest(_)
    ));
}

/// `suffix` is captured into a typed field at the wire boundary (the actual
/// `400 suffix is not supported` is enforced later in the completions
/// handler; here we only lock that the schema accepts and preserves it).
#[test]
fn completion_request_captures_suffix() {
    let req: CompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "prompt": "hi",
        "suffix": "tail"
    }))
    .unwrap();
    assert_eq!(req.suffix.as_deref(), Some("tail"));
}

/// `logprobs` / `best_of` / `logit_bias` are accepted into typed fields
/// (they are ignored downstream, but MUST NOT cause a deserialize failure).
#[test]
fn completion_request_accepts_logprobs_best_of_logit_bias() {
    let req: CompletionRequest = serde_json::from_value(json!({
        "model": "m",
        "prompt": "hi",
        "logprobs": 5,
        "best_of": 3,
        "logit_bias": {"1234": -100, "50256": 42}
    }))
    .unwrap();
    assert_eq!(req.logprobs, Some(5));
    assert_eq!(req.best_of, Some(3));
    let bias = req.logit_bias.expect("logit_bias present");
    assert_eq!(bias.len(), 2);
    assert_eq!(bias["1234"], json!(-100));
}

#[test]
fn completion_response_serializes_text_completion_shape() {
    let response = CompletionResponse {
        id: "cmpl-test".to_string(),
        object: "text_completion".to_string(),
        created: 1,
        model: "model".to_string(),
        system_fingerprint: None,
        choices: vec![CompletionChoice {
            text: "reply".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
    };

    let value: Value = serde_json::to_value(response).unwrap();
    assert_eq!(value["object"], "text_completion");
    assert_eq!(value["choices"][0]["logprobs"], Value::Null);
    assert!(value["id"]
        .as_str()
        .unwrap_or_default()
        .starts_with("cmpl-"));
}

#[cfg(test)]
mod prop_tests {
    //! Property-based round-trip coverage for the legacy text-completions
    //! wire schema.
    //!
    //! Feature: test-coverage-codecov, Property 1: Schema 序列化往返
    //!
    //! For any valid `CompletionRequest` / `CompletionResponse`, serializing to
    //! JSON then deserializing yields a semantically equivalent value. Proven
    //! via serialization idempotence (`serialize -> deserialize -> serialize`
    //! reproduces the same JSON `Value`) so no `PartialEq` derive is required.
    //!
    //! Validates: Requirements 1.2

    use super::super::*;
    use crate::openai::schema::{StreamOptions, StringOrVec, Usage};
    use proptest::prelude::*;
    use serde::de::DeserializeOwned;
    use serde::Serialize;

    fn assert_json_roundtrip<T>(value: &T) -> Result<(), TestCaseError>
    where
        T: Serialize + DeserializeOwned,
    {
        let v1 = serde_json::to_value(value)
            .map_err(|e| TestCaseError::fail(format!("serialize failed: {e}")))?;
        let back: T = serde_json::from_value(v1.clone())
            .map_err(|e| TestCaseError::fail(format!("deserialize failed: {e}")))?;
        let v2 = serde_json::to_value(&back)
            .map_err(|e| TestCaseError::fail(format!("re-serialize failed: {e}")))?;
        prop_assert_eq!(v1, v2);
        Ok(())
    }

    fn arb_name() -> impl Strategy<Value = String> {
        "[a-zA-Z_][a-zA-Z0-9_]{0,15}"
    }

    fn arb_text() -> impl Strategy<Value = String> {
        "[ -~]{0,24}"
    }

    fn arb_f32() -> impl Strategy<Value = f32> {
        (0u32..=200).prop_map(|n| n as f32 / 100.0)
    }

    /// JSON value without a top-level `null` leaf (keeps `logit_bias` /
    /// `extra` values idempotent across the round-trip).
    fn arb_json() -> impl Strategy<Value = Value> {
        prop_oneof![
            any::<bool>().prop_map(Value::Bool),
            (-1000i64..1000).prop_map(|n| serde_json::json!(n)),
            arb_text().prop_map(Value::String),
        ]
    }

    fn arb_prompt() -> impl Strategy<Value = CompletionPrompt> {
        prop_oneof![
            arb_text().prop_map(CompletionPrompt::Text),
            prop::collection::vec(arb_text(), 0..3).prop_map(CompletionPrompt::Texts),
            prop::collection::vec(-1000i64..1000, 0..3).prop_map(CompletionPrompt::Tokens),
            prop::collection::vec(prop::collection::vec(-1000i64..1000, 0..3), 0..3)
                .prop_map(CompletionPrompt::TokenMatrix),
        ]
    }

    fn arb_stop() -> impl Strategy<Value = StringOrVec> {
        prop_oneof![
            arb_text().prop_map(StringOrVec::String),
            prop::collection::vec(arb_text(), 0..3).prop_map(StringOrVec::Vec),
        ]
    }

    prop_compose! {
        fn arb_request()(
            model in arb_name(),
            prompt in arb_prompt(),
            suffix in prop::option::of(arb_text()),
            max_tokens in prop::option::of(0i32..100_000),
            temperature in prop::option::of(arb_f32()),
            top_p in prop::option::of(arb_f32()),
            n in prop::option::of(1i32..8),
            stream in prop::option::of(any::<bool>()),
            stream_options in prop::option::of(any::<bool>().prop_map(|include_usage| StreamOptions { include_usage })),
            logprobs in prop::option::of(0i32..10),
            echo in prop::option::of(any::<bool>()),
            stop in prop::option::of(arb_stop()),
            presence_penalty in prop::option::of(arb_f32()),
            frequency_penalty in prop::option::of(arb_f32()),
            best_of in prop::option::of(1i32..8),
            logit_bias in prop::option::of(prop::collection::hash_map(arb_name(), arb_json(), 0..3)),
            seed in prop::option::of(0i64..1_000_000),
            user in prop::option::of(arb_name()),
            extra in prop::collection::hash_map(arb_name().prop_map(|k| format!("x_{k}")), arb_json(), 0..3),
        ) -> CompletionRequest {
            CompletionRequest {
                model,
                prompt,
                suffix,
                max_tokens,
                temperature,
                top_p,
                n,
                stream,
                stream_options,
                logprobs,
                echo,
                stop,
                presence_penalty,
                frequency_penalty,
                best_of,
                logit_bias,
                seed,
                user,
                extra,
            }
        }
    }

    fn arb_choice() -> impl Strategy<Value = CompletionChoice> {
        (arb_text(), 0i32..8, prop::option::of(arb_name())).prop_map(
            |(text, index, finish_reason)| CompletionChoice {
                text,
                index,
                logprobs: None,
                finish_reason,
            },
        )
    }

    prop_compose! {
        fn arb_response()(
            id in arb_name(),
            created in 0i64..2_000_000_000,
            model in arb_name(),
            system_fingerprint in prop::option::of(arb_name()),
            choices in prop::collection::vec(arb_choice(), 0..3),
            usage in prop::option::of((0i32..1000, 0i32..1000, 0i32..2000)),
        ) -> CompletionResponse {
            CompletionResponse {
                id,
                object: "text_completion".to_string(),
                created,
                model,
                system_fingerprint,
                choices,
                usage: usage.map(|(prompt_tokens, completion_tokens, total_tokens)| Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                }),
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Property 1: `CompletionRequest` survives a JSON round-trip
        /// (all four `prompt` shapes + flattened `extra` passthrough).
        #[test]
        fn completion_request_round_trips(req in arb_request()) {
            assert_json_roundtrip(&req)?;
        }

        /// Property 1: `CompletionResponse` survives a JSON round-trip and
        /// keeps the `text_completion` object tag.
        #[test]
        fn completion_response_round_trips(resp in arb_response()) {
            assert_json_roundtrip(&resp)?;
        }
    }
}
