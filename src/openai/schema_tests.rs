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
        Message::System { content, .. } => {
            assert!(matches!(content, SystemContentInput::Text(text) if text == "You are helpful."))
        }
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

// -----------------------------------------------------------------------
// Property-based tests
// -----------------------------------------------------------------------
//
// Feature: test-coverage-codecov, Property 1: Schema serialization round-trip
//
// For any well-formed wire value, `serialize -> deserialize -> serialize`
// is semantically stable: the JSON produced from the original value equals
// the JSON produced after a full deserialize round trip. Because these
// schema types do not derive `PartialEq`, JSON-value equality is the
// canonical expression of semantic equivalence. The suite also asserts the
// Option-B guardrail: `reasoning_content` (`#[serde(skip_serializing)]`)
// never reaches the wire, and `#[serde(flatten)] extra` is preserved.
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// `serialize -> deserialize -> serialize` must be JSON-stable.
    ///
    /// Returns the JSON `Value` of the original serialization so callers can
    /// make further assertions (e.g. on top-level keys / flattened extras).
    fn roundtrip_value<T>(value: &T) -> serde_json::Value
    where
        T: Serialize + serde::de::DeserializeOwned,
    {
        let v1 = serde_json::to_value(value).expect("serialize original");
        let back: T = serde_json::from_value(v1.clone()).expect("deserialize");
        let v2 = serde_json::to_value(&back).expect("serialize round-tripped");
        assert_eq!(v1, v2, "round-trip is not JSON-stable");
        v1
    }

    // --- shared leaf/value strategies ---------------------------------

    /// A shallow JSON leaf (no nesting) — stable under a JSON round trip.
    fn arb_leaf() -> impl Strategy<Value = serde_json::Value> {
        prop_oneof![
            Just(serde_json::Value::Null),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(serde_json::Value::from),
            "[a-zA-Z0-9 ]{0,10}".prop_map(serde_json::Value::from),
        ]
    }

    /// A small JSON object (used for tool `parameters`).
    fn arb_obj() -> impl Strategy<Value = serde_json::Value> {
        prop::collection::hash_map("[a-z]{1,5}", arb_leaf(), 0..3)
            .prop_map(|m| serde_json::Value::Object(m.into_iter().collect()))
    }

    /// Unknown top-level fields captured by `#[serde(flatten)] extra`.
    ///
    /// Keys are `vendor_`-prefixed so they never collide with a real
    /// `ChatRequest` field name.
    fn arb_extra() -> impl Strategy<Value = HashMap<String, serde_json::Value>> {
        prop::collection::hash_map("vendor_[a-z]{1,6}", arb_leaf(), 0..4)
    }

    fn arb_reasoning_effort() -> impl Strategy<Value = ReasoningEffort> {
        prop_oneof![
            Just(ReasoningEffort::None),
            Just(ReasoningEffort::Minimal),
            Just(ReasoningEffort::Low),
            Just(ReasoningEffort::Medium),
            Just(ReasoningEffort::High),
            Just(ReasoningEffort::Xhigh),
            Just(ReasoningEffort::Max),
        ]
    }

    // --- message / tool strategies ------------------------------------

    fn arb_message() -> impl Strategy<Value = Message> {
        let name = prop::option::of("[a-z]{1,8}".prop_map(String::from));
        prop_oneof![
            (name.clone(), "[a-zA-Z0-9 ]{0,20}").prop_map(|(name, content)| {
                Message::System {
                    name,
                    content: SystemContentInput::Text(content),
                }
            }),
            (name.clone(), "[a-zA-Z0-9 ]{0,20}").prop_map(|(name, content)| {
                Message::User {
                    name,
                    content: ContentInput::Text(content),
                }
            }),
            (name.clone(), prop::option::of("[a-zA-Z0-9 ]{0,20}")).prop_map(|(name, content)| {
                Message::Assistant {
                    name,
                    content: content.map(ContentInput::Text),
                    tool_calls: None,
                }
            }),
            ("[a-zA-Z0-9 ]{0,20}", "[a-z0-9]{1,10}").prop_map(|(content, id)| {
                Message::Tool {
                    content: ToolContentInput::Text(content),
                    tool_call_id: id,
                }
            }),
            (name, "[a-zA-Z0-9 ]{0,20}").prop_map(|(name, content)| {
                Message::Developer {
                    name,
                    content: SystemContentInput::Text(content),
                }
            }),
        ]
    }

    fn arb_tool() -> impl Strategy<Value = Tool> {
        (
            "[a-z_]{1,10}",
            prop::option::of("[a-zA-Z0-9 ]{0,15}"),
            arb_obj(),
        )
            .prop_map(|(name, description, parameters)| Tool {
                r#type: "function".to_string(),
                function: Function {
                    name,
                    description,
                    parameters,
                },
            })
    }

    fn arb_tool_call() -> impl Strategy<Value = ToolCall> {
        (
            prop::option::of(any::<i32>()),
            prop::option::of("[a-z0-9]{1,8}"),
            "[a-zA-Z0-9 {}\":,]{0,20}",
        )
            .prop_map(|(index, id, arguments)| ToolCall {
                index,
                id,
                r#type: "function".to_string(),
                function: ResponseFunction {
                    name: Some("fn".to_string()),
                    arguments,
                },
            })
    }

    fn arb_stop() -> impl Strategy<Value = Option<StringOrVec>> {
        prop::option::of(prop_oneof![
            "[a-z]{1,6}".prop_map(StringOrVec::String),
            prop::collection::vec("[a-z]{1,6}", 1..3).prop_map(StringOrVec::Vec),
        ])
    }

    // --- top-level request/response strategies ------------------------

    fn arb_chat_request() -> impl Strategy<Value = ChatRequest> {
        (
            prop::collection::vec(arb_message(), 1..4),
            "[a-zA-Z0-9._-]{1,20}",
            prop::option::of(-2.0f32..2.0f32),
            prop::option::of(0.0f32..1.0f32),
            prop::option::of(any::<bool>()),
            0i32..8192,
            prop::option::of(arb_reasoning_effort()),
            prop::option::of(prop::collection::vec(arb_tool(), 1..3)),
            arb_stop(),
            arb_extra(),
        )
            .prop_map(
                |(
                    messages,
                    model,
                    temperature,
                    top_p,
                    stream,
                    max_tokens,
                    reasoning_effort,
                    tools,
                    stop,
                    extra,
                )| ChatRequest {
                    messages,
                    model,
                    frequency_penalty: None,
                    presence_penalty: None,
                    stream,
                    stream_options: None,
                    temperature,
                    top_p,
                    user: None,
                    // Always `Some`: `None` would be dropped on the wire and
                    // re-hydrate to the `default_max_tokens()` value, which is
                    // an intentional (non-round-trip) default, not a bug.
                    max_tokens: Some(max_tokens),
                    max_completion_tokens: None,
                    reasoning_effort,
                    n: None,
                    tools,
                    tool_choice: ToolChoice::default(),
                    stop,
                    response_format: None,
                    extra_body: None,
                    extra,
                },
            )
    }

    fn arb_usage() -> impl Strategy<Value = Usage> {
        (
            0i32..1_000_000,
            0i32..1_000_000,
            0i32..2_000_000,
            prop::option::of((0i32..1000, 0i32..1000)),
            prop::option::of((0i32..1000, 0i32..1000)),
        )
            .prop_map(|(prompt, completion, total, pd, cd)| Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: total,
                prompt_tokens_details: pd.map(|(cached_tokens, audio_tokens)| {
                    PromptTokensDetails {
                        cached_tokens,
                        audio_tokens,
                    }
                }),
                completion_tokens_details: cd.map(|(reasoning_tokens, audio_tokens)| {
                    CompletionTokensDetails {
                        reasoning_tokens,
                        audio_tokens,
                    }
                }),
            })
    }

    fn arb_response_message() -> impl Strategy<Value = ChatResponseMessage> {
        (
            prop::option::of("[a-z]{1,8}"),
            prop::option::of("[a-zA-Z0-9 ]{0,20}"),
            prop::option::of(prop::collection::vec(arb_tool_call(), 1..3)),
            prop::option::of("[a-zA-Z0-9 ]{0,20}"),
        )
            .prop_map(|(role, content, tool_calls, reasoning_content)| {
                ChatResponseMessage {
                    role,
                    content,
                    tool_calls,
                    reasoning_content,
                }
            })
    }

    fn arb_chat_response() -> impl Strategy<Value = ChatResponse> {
        (
            "chatcmpl-[a-z0-9]{1,10}",
            0i64..2_000_000_000,
            "[a-zA-Z0-9._-]{1,20}",
            "[a-z0-9]{1,10}",
            prop::collection::vec(
                (
                    0i32..8,
                    prop::option::of("[a-z_]{1,10}"),
                    arb_response_message(),
                )
                    .prop_map(|(index, finish_reason, message)| Choice {
                        index,
                        finish_reason,
                        logprobs: None,
                        message,
                    }),
                1..3,
            ),
            arb_usage(),
        )
            .prop_map(|(id, created, model, system_fingerprint, choices, usage)| {
                ChatResponse {
                    id,
                    created,
                    model,
                    system_fingerprint,
                    choices,
                    object: "chat.completion".to_string(),
                    usage,
                }
            })
    }

    fn arb_chat_stream_response() -> impl Strategy<Value = ChatStreamResponse> {
        (
            "chatcmpl-[a-z0-9]{1,10}",
            0i64..2_000_000_000,
            "[a-zA-Z0-9._-]{1,20}",
            "[a-z0-9]{1,10}",
            prop::collection::vec(
                (
                    0i32..8,
                    prop::option::of("[a-z_]{1,10}"),
                    arb_response_message(),
                )
                    .prop_map(|(index, finish_reason, delta)| ChoiceDelta {
                        index,
                        finish_reason,
                        logprobs: None,
                        delta,
                    }),
                1..3,
            ),
            prop::option::of(arb_usage()),
        )
            .prop_map(|(id, created, model, system_fingerprint, choices, usage)| {
                ChatStreamResponse {
                    id,
                    created,
                    model,
                    system_fingerprint,
                    choices,
                    object: "chat.completion.chunk".to_string(),
                    usage,
                }
            })
    }

    // --- embeddings / model strategies --------------------------------

    fn arb_embedding_input() -> impl Strategy<Value = EmbeddingInput> {
        prop_oneof![
            "[a-zA-Z0-9 ]{0,20}".prop_map(EmbeddingInput::String),
            prop::collection::vec("[a-zA-Z0-9 ]{0,10}", 1..4).prop_map(EmbeddingInput::StringArray),
            prop::collection::vec(any::<i32>(), 1..4).prop_map(EmbeddingInput::IntArray),
            prop::collection::vec(prop::collection::vec(any::<i32>(), 1..3), 1..3)
                .prop_map(EmbeddingInput::IntMatrix),
        ]
    }

    fn arb_embeddings_request() -> impl Strategy<Value = EmbeddingsRequest> {
        (
            arb_embedding_input(),
            "[a-zA-Z0-9._-]{1,20}",
            prop_oneof![Just(EncodingFormat::Float), Just(EncodingFormat::Base64)],
            prop::option::of(1i32..4096),
            prop::option::of("[a-z]{1,8}"),
        )
            .prop_map(|(input, model, encoding_format, dimensions, user)| {
                EmbeddingsRequest {
                    input,
                    model,
                    encoding_format,
                    dimensions,
                    user,
                }
            })
    }

    fn arb_embedding_data() -> impl Strategy<Value = EmbeddingData> {
        prop_oneof![
            // Finite, bounded f32 values: no NaN/Inf (invalid JSON) and the
            // f32->f64->f32 round trip is lossless.
            prop::collection::vec(-1000.0f32..1000.0f32, 1..8).prop_map(EmbeddingData::Float),
            "[A-Za-z0-9+/=]{0,16}".prop_map(EmbeddingData::Base64),
        ]
    }

    fn arb_embeddings_response() -> impl Strategy<Value = EmbeddingsResponse> {
        (
            prop::collection::vec(
                (arb_embedding_data(), 0i32..64).prop_map(|(embedding, index)| Embedding {
                    object: "embedding".to_string(),
                    embedding,
                    index,
                }),
                1..4,
            ),
            "[a-zA-Z0-9._-]{1,20}",
            (0i32..1_000_000, 0i32..2_000_000),
        )
            .prop_map(
                |(data, model, (prompt_tokens, total_tokens))| EmbeddingsResponse {
                    object: "list".to_string(),
                    data,
                    model,
                    usage: EmbeddingsUsage {
                        prompt_tokens,
                        total_tokens,
                    },
                },
            )
    }

    fn arb_model() -> impl Strategy<Value = Model> {
        (
            "[a-zA-Z0-9._-]{1,20}",
            0i64..2_000_000_000,
            "[a-z]{1,8}",
            "[a-z]{1,10}",
        )
            .prop_map(|(id, created, object, owned_by)| Model {
                id,
                created,
                object,
                owned_by,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn chat_request_roundtrip(req in arb_chat_request()) {
            let v1 = roundtrip_value(&req);
            // `#[serde(flatten)] extra` keys survive the round trip verbatim.
            for key in req.extra.keys() {
                prop_assert!(v1.get(key).is_some(), "flattened extra key missing: {key}");
            }
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn chat_response_roundtrip(resp in arb_chat_response()) {
            let v1 = roundtrip_value(&resp);
            // Option-B guardrail: `reasoning_content` NEVER reaches the wire,
            // even when populated internally.
            let s = serde_json::to_string(&resp).expect("serialize response");
            prop_assert!(!s.contains("reasoning_content"), "reasoning_content leaked: {s}");
            // Only OpenAI-recognized top-level keys appear.
            let allowed = [
                "id",
                "created",
                "model",
                "system_fingerprint",
                "choices",
                "object",
                "usage",
            ];
            for key in v1.as_object().expect("object").keys() {
                prop_assert!(allowed.contains(&key.as_str()), "unexpected key: {key}");
            }
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn chat_stream_response_roundtrip(resp in arb_chat_stream_response()) {
            roundtrip_value(&resp);
            let s = serde_json::to_string(&resp).expect("serialize stream response");
            prop_assert!(!s.contains("reasoning_content"), "reasoning_content leaked: {s}");
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn usage_roundtrip(usage in arb_usage()) {
            roundtrip_value(&usage);
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn embeddings_request_roundtrip(req in arb_embeddings_request()) {
            roundtrip_value(&req);
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn embeddings_response_roundtrip(resp in arb_embeddings_response()) {
            roundtrip_value(&resp);
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn model_roundtrip(model in arb_model()) {
            roundtrip_value(&model);
        }

        /// Feature: test-coverage-codecov, Property 1
        #[test]
        fn models_list_roundtrip(models in prop::collection::vec(arb_model(), 0..4)) {
            let list = Models { object: "list".to_string(), data: models };
            roundtrip_value(&list);
        }
    }
}

// -----------------------------------------------------------------------
// Default-value helper coverage
// -----------------------------------------------------------------------
//
// The property tests above always populate every field, so serde's
// `default = "..."` helper fns (which fire ONLY when a field is omitted
// during deserialization) are never exercised. These flat tests deserialize
// JSON that deliberately omits each defaulted field and asserts the helper's
// value is applied — driving every `default_*` fn on the deser path.

/// `Model.object` / `Model.owned_by` fall back to their defaults when omitted.
#[test]
fn model_defaults_applied_when_omitted() {
    let model: Model = serde_json::from_str(r#"{"id":"m","created":1}"#).expect("deserialize");
    assert_eq!(model.object, "model"); // default_model_object
    assert_eq!(model.owned_by, "bedrock"); // default_owned_by
}

/// `Models.object` defaults to "list" and `data` to empty when omitted.
#[test]
fn models_list_defaults_applied_when_omitted() {
    let models: Models = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(models.object, "list"); // default_list_object
    assert!(models.data.is_empty());
}

/// `ToolCall.type` defaults to "function" when omitted.
#[test]
fn tool_call_type_default_applied_when_omitted() {
    let call: ToolCall =
        serde_json::from_str(r#"{"function":{"arguments":"{}"}}"#).expect("deserialize");
    assert_eq!(call.r#type, "function"); // default_function_type
    assert_eq!(call.function.arguments, "{}");
    assert!(call.function.name.is_none());
}

/// `Tool.type` defaults to "function" when omitted.
#[test]
fn tool_type_default_applied_when_omitted() {
    let tool: Tool =
        serde_json::from_str(r#"{"function":{"name":"f","parameters":{"type":"object"}}}"#)
            .expect("deserialize");
    assert_eq!(tool.r#type, "function"); // default_function_type
    assert_eq!(tool.function.name, "f");
}

/// `TextContent.type` defaults to "text" when omitted.
#[test]
fn text_content_type_default_applied_when_omitted() {
    let text: TextContent = serde_json::from_str(r#"{"text":"hello"}"#).expect("deserialize");
    assert_eq!(text.r#type, "text"); // default_text_type
    assert_eq!(text.text, "hello");
}

/// `ImageUrl.detail` defaults to "auto" and `ImageContent.type` to
/// "image_url" when omitted.
#[test]
fn image_content_defaults_applied_when_omitted() {
    let img: ImageContent =
        serde_json::from_str(r#"{"image_url":{"url":"https://x/y.png"}}"#).expect("deserialize");
    assert_eq!(img.r#type, "image_url"); // default_image_type
    assert_eq!(img.image_url.url, "https://x/y.png");
    assert_eq!(img.image_url.detail, "auto"); // default_detail
}

/// `StreamOptions.include_usage` defaults to true when omitted.
#[test]
fn stream_options_default_true_when_omitted() {
    let opts: StreamOptions = serde_json::from_str("{}").expect("deserialize");
    assert!(opts.include_usage); // default_true
}

/// Token limits remain absent when omitted, while `tool_choice` uses its
/// OpenAI default ("auto").
#[test]
fn chat_request_defaults_applied_when_omitted() {
    let req: ChatRequest =
        serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
            .expect("deserialize");
    assert_eq!(req.max_tokens, None);
    assert!(matches!(req.tool_choice, ToolChoice::String(ref s) if s == "auto"));
    assert!(req.extra.is_empty());
}

/// `ChatResponse.object` / `.system_fingerprint` fall back to their defaults
/// when omitted.
#[test]
fn chat_response_defaults_applied_when_omitted() {
    let raw = r#"{
        "id":"chatcmpl-1",
        "created":1,
        "model":"m",
        "choices":[{"message":{"role":"assistant","content":"hi"}}],
        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
    }"#;
    let resp: ChatResponse = serde_json::from_str(raw).expect("deserialize");
    assert_eq!(resp.object, "chat.completion"); // default_chat_completion_object
    assert_eq!(resp.system_fingerprint, "fp"); // default_fingerprint
    assert_eq!(resp.choices[0].index, 0);
}

/// `ChatStreamResponse.object` / `.system_fingerprint` fall back to their
/// defaults when omitted.
#[test]
fn chat_stream_response_defaults_applied_when_omitted() {
    let raw = r#"{
        "id":"chatcmpl-1",
        "created":1,
        "model":"m",
        "choices":[{"delta":{"content":"hi"}}]
    }"#;
    let resp: ChatStreamResponse = serde_json::from_str(raw).expect("deserialize");
    assert_eq!(resp.object, "chat.completion.chunk"); // default_chat_chunk_object
    assert_eq!(resp.system_fingerprint, "fp"); // default_fingerprint
    assert!(resp.usage.is_none());
}

/// `Embedding.object` defaults to "embedding" and `EmbeddingsResponse.object`
/// to "list" when omitted.
#[test]
fn embeddings_response_defaults_applied_when_omitted() {
    let raw = r#"{
        "data":[{"embedding":[0.1,0.2],"index":0}],
        "model":"m",
        "usage":{"prompt_tokens":1,"total_tokens":1}
    }"#;
    let resp: EmbeddingsResponse = serde_json::from_str(raw).expect("deserialize");
    assert_eq!(resp.object, "list"); // default_list_object
    assert_eq!(resp.data[0].object, "embedding"); // default_embedding_object
    assert_eq!(resp.data[0].index, 0);
}

/// Usage token detail sub-structs default their fields to 0 when omitted.
#[test]
fn usage_token_details_default_to_zero() {
    let details: PromptTokensDetails = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(details.cached_tokens, 0);
    assert_eq!(details.audio_tokens, 0);

    let cdetails: CompletionTokensDetails = serde_json::from_str("{}").expect("deserialize");
    assert_eq!(cdetails.reasoning_tokens, 0);
    assert_eq!(cdetails.audio_tokens, 0);
}

// -----------------------------------------------------------------------
// Untagged enum arm coverage
// -----------------------------------------------------------------------
//
// Each untagged enum's serde arms only execute when a value of that shape is
// actually deserialized. These tests round-trip every arm so all match arms
// in the generated `Deserialize`/`Serialize` impls run.

/// `ContentInput` round-trips both the `Text` and `Parts` arms.
#[test]
fn content_input_untagged_arms_roundtrip() {
    let text: ContentInput = serde_json::from_str(r#""just a string""#).expect("text arm");
    assert!(matches!(text, ContentInput::Text(ref s) if s == "just a string"));

    let parts: ContentInput = serde_json::from_str(
        r#"[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"u"}}]"#,
    )
    .expect("parts arm");
    match parts {
        ContentInput::Parts(ref p) => {
            assert_eq!(p.len(), 2);
            assert!(matches!(p[0], ContentPart::Text(_)));
            assert!(matches!(p[1], ContentPart::Image(_)));
        }
        other => panic!("expected Parts, got {other:?}"),
    }

    // Serialize each arm back to confirm the Serialize arms also run.
    assert_eq!(
        serde_json::to_value(&text).expect("ser text"),
        serde_json::json!("just a string")
    );
    assert!(serde_json::to_value(&parts).expect("ser parts").is_array());
}

/// `SystemContentInput` round-trips both the `Text` and `Parts` arms.
#[test]
fn system_content_input_untagged_arms_roundtrip() {
    let text: SystemContentInput = serde_json::from_str(r#""sys""#).expect("text arm");
    assert!(matches!(text, SystemContentInput::Text(ref s) if s == "sys"));

    let parts: SystemContentInput =
        serde_json::from_str(r#"[{"type":"text","text":"a"}]"#).expect("parts arm");
    assert!(matches!(parts, SystemContentInput::Parts(ref p) if p.len() == 1));

    assert!(serde_json::to_value(&text).expect("ser").is_string());
    assert!(serde_json::to_value(&parts).expect("ser").is_array());
}

/// `ToolContentInput` round-trips both the `Text` and `Parts` arms.
#[test]
fn tool_content_input_untagged_arms_roundtrip() {
    let text: ToolContentInput = serde_json::from_str(r#""result""#).expect("text arm");
    assert!(matches!(text, ToolContentInput::Text(ref s) if s == "result"));

    let parts: ToolContentInput =
        serde_json::from_str(r#"[{"any":"json"},{"n":1}]"#).expect("parts arm");
    assert!(matches!(parts, ToolContentInput::Parts(ref p) if p.len() == 2));

    assert!(serde_json::to_value(&text).expect("ser").is_string());
    assert!(serde_json::to_value(&parts).expect("ser").is_array());
}

/// `StringOrVec` round-trips both the `String` and `Vec` arms.
#[test]
fn string_or_vec_untagged_arms_roundtrip() {
    let single: StringOrVec = serde_json::from_str(r#""STOP""#).expect("string arm");
    assert!(matches!(single, StringOrVec::String(ref s) if s == "STOP"));

    let many: StringOrVec = serde_json::from_str(r#"["a","b"]"#).expect("vec arm");
    assert!(matches!(many, StringOrVec::Vec(ref v) if v.len() == 2));

    assert!(serde_json::to_value(&single).expect("ser").is_string());
    assert!(serde_json::to_value(&many).expect("ser").is_array());
}

/// `EmbeddingInput` round-trips all four arms.
#[test]
fn embedding_input_untagged_arms_roundtrip() {
    let s: EmbeddingInput = serde_json::from_str(r#""hi""#).expect("string arm");
    assert!(matches!(s, EmbeddingInput::String(_)));

    let sa: EmbeddingInput = serde_json::from_str(r#"["a","b"]"#).expect("string array arm");
    assert!(matches!(sa, EmbeddingInput::StringArray(ref v) if v.len() == 2));

    let ia: EmbeddingInput = serde_json::from_str(r#"[1,2,3]"#).expect("int array arm");
    assert!(matches!(ia, EmbeddingInput::IntArray(ref v) if v.len() == 3));

    let im: EmbeddingInput = serde_json::from_str(r#"[[1,2],[3,4]]"#).expect("int matrix arm");
    assert!(matches!(im, EmbeddingInput::IntMatrix(ref v) if v.len() == 2));

    for value in [&s, &sa, &ia, &im] {
        // Each arm serializes without panicking.
        let _ = serde_json::to_value(value).expect("serialize embedding input");
    }
}

/// `EmbeddingData` round-trips both the `Float` and `Base64` arms.
#[test]
fn embedding_data_untagged_arms_roundtrip() {
    let floats: EmbeddingData = serde_json::from_str(r#"[0.1,0.2,0.3]"#).expect("float arm");
    assert!(matches!(floats, EmbeddingData::Float(ref v) if v.len() == 3));

    let b64: EmbeddingData = serde_json::from_str(r#""AAAA""#).expect("base64 arm");
    assert!(matches!(b64, EmbeddingData::Base64(ref s) if s == "AAAA"));

    assert!(serde_json::to_value(&floats).expect("ser").is_array());
    assert!(serde_json::to_value(&b64).expect("ser").is_string());
}

/// `ContentPart` round-trips both the `Text` and `Image` arms directly.
#[test]
fn content_part_untagged_arms_roundtrip() {
    let text: ContentPart = serde_json::from_str(r#"{"type":"text","text":"a"}"#).expect("text");
    assert!(matches!(text, ContentPart::Text(_)));

    let image: ContentPart =
        serde_json::from_str(r#"{"type":"image_url","image_url":{"url":"u","detail":"low"}}"#)
            .expect("image");
    match image {
        ContentPart::Image(ref c) => assert_eq!(c.image_url.detail, "low"),
        other => panic!("expected Image, got {other:?}"),
    }

    assert!(serde_json::to_value(&text).expect("ser").is_object());
    assert!(serde_json::to_value(&image).expect("ser").is_object());
}

/// `ResponseFormat` round-trips all three internally-tagged variants.
#[test]
fn response_format_variants_roundtrip() {
    let text: ResponseFormat = serde_json::from_str(r#"{"type":"text"}"#).expect("text");
    assert!(matches!(text, ResponseFormat::Text));

    let obj: ResponseFormat = serde_json::from_str(r#"{"type":"json_object"}"#).expect("obj");
    assert!(matches!(obj, ResponseFormat::JsonObject));

    let schema: ResponseFormat = serde_json::from_str(
        r#"{"type":"json_schema","json_schema":{"name":"s","strict":true,"schema":{"type":"object"}}}"#,
    )
    .expect("schema");
    match schema {
        ResponseFormat::JsonSchema { ref json_schema } => {
            assert_eq!(json_schema.name.as_deref(), Some("s"));
            assert_eq!(json_schema.strict, Some(true));
            assert!(json_schema.schema.is_some());
        }
        other => panic!("expected JsonSchema, got {other:?}"),
    }

    // Serialize each variant back so the Serialize arms execute.
    assert_eq!(
        serde_json::to_value(&text).expect("ser text"),
        serde_json::json!({"type":"text"})
    );
    assert_eq!(
        serde_json::to_value(&obj).expect("ser obj"),
        serde_json::json!({"type":"json_object"})
    );
    assert!(
        serde_json::to_value(&schema).expect("ser schema")["json_schema"]
            .get("name")
            .is_some()
    );
}

/// `ToolChoice` round-trips both the `String` and `Object` arms.
#[test]
fn tool_choice_untagged_arms_roundtrip() {
    let s: ToolChoice = serde_json::from_str(r#""auto""#).expect("string arm");
    assert!(matches!(s, ToolChoice::String(ref v) if v == "auto"));

    let o: ToolChoice =
        serde_json::from_str(r#"{"type":"function","function":{"name":"f"}}"#).expect("object arm");
    assert!(matches!(o, ToolChoice::Object(_)));

    assert!(serde_json::to_value(&s).expect("ser").is_string());
    assert!(serde_json::to_value(&o).expect("ser").is_object());
}

// -----------------------------------------------------------------------
// Default derives
// -----------------------------------------------------------------------

/// `ChatResponseMessage::default()` yields an all-`None` message that
/// serializes to an empty object (every field is `skip_serializing_if`).
#[test]
fn chat_response_message_default_is_empty() {
    let msg = ChatResponseMessage::default();
    assert!(msg.role.is_none());
    assert!(msg.content.is_none());
    assert!(msg.tool_calls.is_none());
    assert!(msg.reasoning_content.is_none());

    let json = serde_json::to_value(&msg).expect("serialize default message");
    assert_eq!(json, serde_json::json!({}));
}

/// `EncodingFormat::default()` is `Float`, and both variants round-trip.
#[test]
fn encoding_format_default_and_variants() {
    assert!(matches!(EncodingFormat::default(), EncodingFormat::Float));

    let float: EncodingFormat = serde_json::from_str(r#""float""#).expect("float");
    let base64: EncodingFormat = serde_json::from_str(r#""base64""#).expect("base64");
    assert_eq!(float, EncodingFormat::Float);
    assert_eq!(base64, EncodingFormat::Base64);

    assert_eq!(
        serde_json::to_value(EncodingFormat::Base64).expect("ser"),
        serde_json::json!("base64")
    );
}

/// `EmbeddingsRequest.encoding_format` defaults to `Float` when omitted.
#[test]
fn embeddings_request_encoding_format_default_when_omitted() {
    let req: EmbeddingsRequest =
        serde_json::from_str(r#"{"input":"hi","model":"m"}"#).expect("deserialize");
    assert_eq!(req.encoding_format, EncodingFormat::Float);
    assert!(req.dimensions.is_none());
}
