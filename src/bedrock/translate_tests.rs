//! Unit tests for [`crate::bedrock::translate`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec).
//!
//! The source file declares this via a `#[path]` mod tests, so the
//! top-level `use super::*;` resolves to the implementation module.

use super::*;
use crate::bedrock::capabilities::ConfigModelCapabilities;
use crate::config::ModelCapabilityConfig;
use crate::openai::schema::{
    ContentInput, ContentPart, ImageContent, ImageUrl, JsonSchemaSpec, Message, ResponseFormat,
    SystemContentInput, TextContent,
};
use std::collections::HashMap;

const MODELS_TOML: &str = "config/models.toml";

fn caps() -> ConfigModelCapabilities {
    let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load models.toml");
    ConfigModelCapabilities::new(config)
}

/// A test resolver that never hits the network. `supports_image` is a flag;
/// `fetch` returns canned bytes so the remote path is exercised offline.
struct TestResolver {
    image_ok: bool,
    canned: Option<(Vec<u8>, String)>,
}

#[async_trait::async_trait]
impl ImageResolver for TestResolver {
    fn supports_image(&self, _model_id: &str) -> bool {
        self.image_ok
    }
    async fn fetch(&self, _url: &str) -> Result<(Vec<u8>, String), AppError> {
        self.canned
            .clone()
            .ok_or_else(|| AppError::Internal("no canned image".to_string()))
    }
}

fn resolver(image_ok: bool) -> TestResolver {
    TestResolver {
        image_ok,
        canned: None,
    }
}

fn base_request(model: &str, messages: Vec<Message>) -> ChatRequest {
    ChatRequest {
        messages,
        model: model.to_string(),
        frequency_penalty: None,
        presence_penalty: None,
        stream: None,
        stream_options: None,
        temperature: None,
        top_p: None,
        user: None,
        max_tokens: Some(2048),
        max_completion_tokens: None,
        reasoning_effort: None,
        n: None,
        tools: None,
        tool_choice: Default::default(),
        stop: None,
        response_format: None,
        extra_body: None,
        extra: HashMap::new(),
    }
}

fn user_text(text: &str) -> Message {
    Message::User {
        name: None,
        content: ContentInput::Text(text.to_string()),
    }
}

#[tokio::test]
async fn text_message_translation() {
    let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("Hello!")]);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    assert_eq!(args.model_id, "anthropic.claude-3-sonnet-v1:0");
    let msgs = args.messages.as_array().expect("messages array");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"][0]["text"], "Hello!");
    assert_eq!(args.inference_config["maxTokens"], 2048);
    // No optional params set.
    assert!(args.inference_config.get("temperature").is_none());
    assert!(args.inference_config.get("topP").is_none());
    assert!(args.additional_model_request_fields.is_none());
    assert!(args.tool_config.is_none());
}

#[tokio::test]
async fn system_and_developer_become_system_blocks() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::System {
                name: None,
                content: SystemContentInput::Text("You are helpful.".to_string()),
            },
            Message::Developer {
                name: None,
                content: SystemContentInput::Text("Be terse.".to_string()),
            },
            user_text("Hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let sys = args.system.as_array().expect("system array");
    assert_eq!(sys.len(), 2);
    assert_eq!(sys[0]["text"], "You are helpful.");
    assert_eq!(sys[1]["text"], "Be terse.");
    // System/developer messages do NOT appear in messages.
    let msgs = args.messages.as_array().expect("messages array");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
}

#[tokio::test]
async fn empty_system_text_is_skipped() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::System {
                name: None,
                content: SystemContentInput::Text("".to_string()),
            },
            Message::Developer {
                name: None,
                content: SystemContentInput::Text("Be terse.".to_string()),
            },
            user_text("Hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let sys = args.system.as_array().expect("system array");
    assert_eq!(sys.len(), 1);
    assert_eq!(sys[0]["text"], "Be terse.");
}

#[tokio::test]
async fn user_empty_text_is_bad_request() {
    let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("")]);
    let c = caps();
    let r = resolver(false);
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("empty user text must reject");

    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("message content must contain at least"))
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn mixed_empty_text_parts_keep_only_non_empty_text() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::User {
            name: None,
            content: ContentInput::Parts(vec![
                ContentPart::Text(TextContent {
                    r#type: "text".to_string(),
                    text: "".to_string(),
                }),
                ContentPart::Text(TextContent {
                    r#type: "text".to_string(),
                    text: "keep me".to_string(),
                }),
            ]),
        }],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let content = args.messages[0]["content"].as_array().expect("content");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["text"], "keep me");
}

#[tokio::test]
async fn array_system_content_flattens_with_newline() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::System {
                name: None,
                content: SystemContentInput::Parts(vec![
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "a".to_string(),
                    }),
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "b".to_string(),
                    }),
                ]),
            },
            user_text("hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let sys = args.system.as_array().expect("system array");
    assert_eq!(sys.len(), 1);
    assert_eq!(sys[0]["text"], "a\nb");
}

#[tokio::test]
async fn image_part_in_system_content_is_rejected() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::System {
                name: None,
                content: SystemContentInput::Parts(vec![ContentPart::Image(ImageContent {
                    r#type: "image_url".to_string(),
                    image_url: ImageUrl {
                        url: "https://example.com/x.png".to_string(),
                        detail: "auto".to_string(),
                    },
                })]),
            },
            user_text("hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("image part in system content must reject");

    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("does not accept non-text content parts"))
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_system_array_is_skipped() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::System {
                name: None,
                content: SystemContentInput::Parts(vec![]),
            },
            Message::Developer {
                name: None,
                content: SystemContentInput::Text("Be terse.".to_string()),
            },
            user_text("Hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let sys = args.system.as_array().expect("system array");
    assert_eq!(sys.len(), 1);
    assert_eq!(sys[0]["text"], "Be terse.");
}

#[tokio::test]
async fn developer_array_content_flattens() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::Developer {
                name: None,
                content: SystemContentInput::Parts(vec![
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "one".to_string(),
                    }),
                    ContentPart::Text(TextContent {
                        r#type: "text".to_string(),
                        text: "two".to_string(),
                    }),
                ]),
            },
            user_text("hi"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");

    let sys = args.system.as_array().expect("system array");
    assert_eq!(sys.len(), 1);
    assert_eq!(sys[0]["text"], "one\ntwo");
}

#[tokio::test]
async fn stop_string_becomes_singleton_sequence() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::String("STOP".to_string()));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let seqs = args.inference_config["stopSequences"]
        .as_array()
        .expect("stopSequences array");
    assert_eq!(seqs.len(), 1);
    assert_eq!(seqs[0], "STOP");
}

#[tokio::test]
async fn blank_stop_list_is_omitted() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::Vec(vec!["\n".to_string()]));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert!(args.inference_config.get("stopSequences").is_none());
}

#[tokio::test]
async fn blank_stop_string_is_omitted() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::String("\n".to_string()));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert!(args.inference_config.get("stopSequences").is_none());
}

#[tokio::test]
async fn blank_stop_list_entries_are_filtered() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::Vec(vec![
        "".to_string(),
        "\n".to_string(),
        "END".to_string(),
    ]));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let seqs = args.inference_config["stopSequences"]
        .as_array()
        .expect("array");
    assert_eq!(seqs.len(), 1);
    assert_eq!(seqs[0], "END");
}

#[tokio::test]
async fn non_blank_stop_list_preserves_order() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::Vec(vec!["a".to_string(), "b".to_string()]));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let seqs = args.inference_config["stopSequences"]
        .as_array()
        .expect("array");
    assert_eq!(seqs.len(), 2);
    assert_eq!(seqs[0], "a");
    assert_eq!(seqs[1], "b");
}

#[tokio::test]
async fn stop_list_passes_through() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.stop = Some(StringOrVec::Vec(vec!["a".to_string(), "b".to_string()]));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let seqs = args.inference_config["stopSequences"]
        .as_array()
        .expect("array");
    assert_eq!(seqs.len(), 2);
    assert_eq!(seqs[1], "b");
}

#[tokio::test]
async fn topp_conflict_drops_topp_when_temperature_present() {
    // claude-sonnet-4-5 has temperature_topp_conflict.
    let mut req = base_request(
        "global.anthropic.claude-sonnet-4-5-20250101-v1:0",
        vec![user_text("hi")],
    );
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let temp = args.inference_config["temperature"].as_f64().unwrap();
    assert!((temp - 0.7).abs() < 1e-6, "temperature ~0.7, got {temp}");
    assert!(
        args.inference_config.get("topP").is_none(),
        "topP must be dropped on conflict"
    );
}

#[tokio::test]
async fn topp_kept_when_no_conflict() {
    // A model WITHOUT the conflict keeps both.
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let temp = args.inference_config["temperature"].as_f64().unwrap();
    assert!((temp - 0.7).abs() < 1e-6, "temperature ~0.7, got {temp}");
    let topp = args.inference_config["topP"].as_f64().unwrap();
    assert!((topp - 0.9).abs() < 1e-6, "topP ~0.9, got {topp}");
}

#[tokio::test]
async fn drop_sampling_params_strips_both_temperature_and_topp() {
    // Opus 4.7+ / Sonnet 5 / Fable/Mythos 5 reject any non-default sampling
    // param with HTTP 400, so drop_sampling_params must strip BOTH.
    for model in [
        "us.anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "claude-mythos-5",
        "claude-fable-5",
    ] {
        let mut req = base_request(model, vec![user_text("hi")]);
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        let c = caps();
        let r = resolver(false);
        let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
            .await
            .expect("translate");
        assert!(
            args.inference_config.get("temperature").is_none(),
            "{model}: temperature must be dropped"
        );
        assert!(
            args.inference_config.get("topP").is_none(),
            "{model}: topP must be dropped"
        );
    }
}

#[tokio::test]
async fn drop_sampling_params_strips_temperature_even_without_topp() {
    // Regression for issue #248: a lone temperature (no top_p) still 400s on
    // Opus 4.7, so it must be stripped even when top_p is absent.
    let mut req = base_request("us.anthropic.claude-opus-4-7", vec![user_text("hi")]);
    req.temperature = Some(0.7);
    req.top_p = None;
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert!(
        args.inference_config.get("temperature").is_none(),
        "lone temperature must be dropped on drop_sampling_params models"
    );
}

#[tokio::test]
async fn opus_4_6_keeps_sampling_params() {
    // Opus 4.6 predates the deprecation and still accepts temperature/top_p —
    // it must NOT carry drop_sampling_params.
    let mut req = base_request("us.anthropic.claude-opus-4-6", vec![user_text("hi")]);
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let temp = args.inference_config["temperature"].as_f64().unwrap();
    assert!(
        (temp - 0.7).abs() < 1e-6,
        "4.6 keeps temperature, got {temp}"
    );
    let topp = args.inference_config["topP"].as_f64().unwrap();
    assert!((topp - 0.9).abs() < 1e-6, "4.6 keeps topP, got {topp}");
}

#[tokio::test]
async fn data_uri_image_decodes_to_image_block() {
    // "hi" base64 = "aGk=".
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::User {
            name: None,
            content: ContentInput::Parts(vec![
                ContentPart::Text(TextContent {
                    r#type: "text".to_string(),
                    text: "look".to_string(),
                }),
                ContentPart::Image(ImageContent {
                    r#type: "image_url".to_string(),
                    image_url: ImageUrl {
                        url: "data:image/png;base64,aGk=".to_string(),
                        detail: "auto".to_string(),
                    },
                }),
            ]),
        }],
    );
    let c = caps();
    let r = resolver(true);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let content = args.messages[0]["content"].as_array().expect("content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["text"], "look");
    assert_eq!(content[1]["image"]["format"], "png");
    let bytes = content[1]["image"]["source"]["bytes"]
        .as_array()
        .expect("bytes array");
    // "hi" = [104, 105].
    assert_eq!(bytes.len(), 2);
    assert_eq!(bytes[0], 104);
    assert_eq!(bytes[1], 105);
}

#[tokio::test]
async fn image_to_non_image_model_is_bad_request() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::User {
            name: None,
            content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                r#type: "image_url".to_string(),
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGk=".to_string(),
                    detail: "auto".to_string(),
                },
            })]),
        }],
    );
    let c = caps();
    let r = resolver(false); // model lacks IMAGE modality
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("must reject image on non-IMAGE model");
    assert!(matches!(err, AppError::BadRequest(_)));
}

#[tokio::test]
async fn remote_image_uses_resolver() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::User {
            name: None,
            content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                r#type: "image_url".to_string(),
                image_url: ImageUrl {
                    url: "https://example.com/cat.jpg".to_string(),
                    detail: "auto".to_string(),
                },
            })]),
        }],
    );
    let c = caps();
    let r = TestResolver {
        image_ok: true,
        canned: Some((vec![1, 2, 3], "jpeg".to_string())),
    };
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let content = args.messages[0]["content"].as_array().expect("content");
    assert_eq!(content[0]["image"]["format"], "jpeg");
    let bytes = content[0]["image"]["source"]["bytes"]
        .as_array()
        .expect("bytes");
    assert_eq!(bytes.len(), 3);
}

#[tokio::test]
async fn invalid_base64_image_is_bad_request() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::User {
            name: None,
            content: ContentInput::Parts(vec![ContentPart::Image(ImageContent {
                r#type: "image_url".to_string(),
                image_url: ImageUrl {
                    url: "data:image/png;base64,!!!notbase64!!!".to_string(),
                    detail: "auto".to_string(),
                },
            })]),
        }],
    );
    let c = caps();
    let r = resolver(true);
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("invalid base64");
    assert!(matches!(err, AppError::BadRequest(_)));
}

#[tokio::test]
async fn contiguous_same_role_users_merge() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![user_text("Hello"), user_text("Who are you?")],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    // Two user messages merge into ONE user turn with two text blocks.
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    let content = msgs[0]["content"].as_array().expect("content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["text"], "Hello");
    assert_eq!(content[1]["text"], "Who are you?");
}

#[tokio::test]
async fn no_assistant_prefill_appends_continuation() {
    // claude-opus-4-8 has no_assistant_prefill.
    let req = base_request(
        "global.anthropic.claude-opus-4-8-20251101-v1:0",
        vec![
            user_text("hi"),
            Message::Assistant {
                name: None,
                content: Some(ContentInput::Text("partial".to_string())),
                tool_calls: None,
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    // user, assistant, then an appended user continuation.
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(
        msgs[2]["content"][0]["text"],
        "Please continue your response from where you left off."
    );
}

#[tokio::test]
async fn no_continuation_when_model_supports_prefill() {
    // A plain model (no no_assistant_prefill) ending on assistant: no append.
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            user_text("hi"),
            Message::Assistant {
                name: None,
                content: Some(ContentInput::Text("partial".to_string())),
                tool_calls: None,
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1]["role"], "assistant");
}

#[tokio::test]
async fn passthrough_allowlist_only_forwards_whitelisted_fields() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.extra
        .insert("thinking".to_string(), json!({"type": "x"}));
    req.extra
        .insert("anthropic_beta".to_string(), json!(["foo"]));
    // Not whitelisted — must NOT pass through.
    req.extra
        .insert("prompt_cache_key".to_string(), json!("secret"));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let fields = args
        .additional_model_request_fields
        .expect("additional fields present");
    assert!(fields.get("thinking").is_some());
    assert!(fields.get("anthropic_beta").is_some());
    assert!(
        fields.get("prompt_cache_key").is_none(),
        "non-whitelisted field leaked to Bedrock"
    );
}

#[tokio::test]
async fn extra_body_passes_through_minus_prompt_caching() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.extra_body = Some(json!({
        "thinking": {"type": "enabled"},
        "prompt_caching": {"system": true}
    }));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let fields = args
        .additional_model_request_fields
        .expect("fields present");
    assert!(fields.get("thinking").is_some());
    assert!(
        fields.get("prompt_caching").is_none(),
        "prompt_caching is a control field, must not reach Bedrock"
    );
}

#[tokio::test]
async fn thinking_field_drops_topp() {
    let mut req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    req.top_p = Some(0.9);
    req.extra_body = Some(json!({ "thinking": {"type": "enabled"} }));
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert!(
        args.inference_config.get("topP").is_none(),
        "topP must be dropped when thinking is present"
    );
}

#[tokio::test]
async fn context_1m_beta_auto_injected_from_config_header() {
    // claude-sonnet-4-6 has context_1m_beta. The config model entry does not
    // set per-model beta_headers, so beta_headers() is empty and nothing is
    // injected — this asserts the de-hardcoded behavior: injection sources
    // ONLY from caps.beta_headers(), never a literal. We additionally verify
    // the merge helper directly below.
    let req = base_request(
        "global.anthropic.claude-sonnet-4-6-20250601-v1:0",
        vec![user_text("hi")],
    );
    let c = caps();
    assert!(c.has(&req.model, Capability::Context1mBeta));
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    // beta_headers is empty in the shipped config, so no anthropic_beta key.
    let injected = args
        .additional_model_request_fields
        .as_ref()
        .and_then(|f| f.get("anthropic_beta"))
        .is_some();
    assert!(
        !injected,
        "no per-model beta header configured, so nothing injected"
    );
}

#[test]
fn merge_anthropic_beta_handles_absent_string_and_list() {
    // Absent → singleton list.
    let mut m = Map::new();
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["ctx-1m"]));

    // String (different) → both, in order.
    let mut m = Map::new();
    m.insert("anthropic_beta".to_string(), json!("existing"));
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["existing", "ctx-1m"]));

    // String (same) → single.
    let mut m = Map::new();
    m.insert("anthropic_beta".to_string(), json!("ctx-1m"));
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["ctx-1m"]));

    // List without header → appended.
    let mut m = Map::new();
    m.insert("anthropic_beta".to_string(), json!(["a"]));
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["a", "ctx-1m"]));

    // List with header → unchanged (no dup).
    let mut m = Map::new();
    m.insert("anthropic_beta".to_string(), json!(["ctx-1m", "b"]));
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["ctx-1m", "b"]));
}

#[tokio::test]
async fn reasoning_seam_fields_merge_into_additional() {
    let req = base_request("anthropic.claude-3-sonnet-v1:0", vec![user_text("hi")]);
    let c = caps();
    let r = resolver(false);
    let extras = ConverseExtras {
        reasoning_fields: Some(json!({"reasoning_config": {"type": "enabled"}})),
        tool_config: Some(json!({"tools": []})),
    };
    let args = to_converse_args(&req, &c, &r, &extras)
        .await
        .expect("translate");
    let fields = args
        .additional_model_request_fields
        .expect("fields present");
    assert!(fields.get("reasoning_config").is_some());
    // Tool config is placed verbatim into the slot (task-17 seam).
    assert_eq!(args.tool_config, Some(json!({"tools": []})));
}

#[tokio::test]
async fn assistant_tool_calls_become_tool_use_blocks() {
    use crate::openai::schema::{ResponseFunction, ToolCall};
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            user_text("call a tool"),
            Message::Assistant {
                name: None,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    index: None,
                    id: Some("call_1".to_string()),
                    r#type: "function".to_string(),
                    function: ResponseFunction {
                        name: Some("get_weather".to_string()),
                        arguments: r#"{"city":"SF"}"#.to_string(),
                    },
                }]),
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    let assistant = &msgs[1];
    assert_eq!(assistant["role"], "assistant");
    let tu = &assistant["content"][0]["toolUse"];
    assert_eq!(tu["toolUseId"], "call_1");
    assert_eq!(tu["name"], "get_weather");
    assert_eq!(tu["input"]["city"], "SF");
}

#[tokio::test]
async fn tool_message_becomes_user_tool_result() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::Tool {
            content: ToolContentInput::Text("72F sunny".to_string()),
            tool_call_id: "call_1".to_string(),
        }],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    assert_eq!(msgs[0]["role"], "user");
    let tr = &msgs[0]["content"][0]["toolResult"];
    assert_eq!(tr["toolUseId"], "call_1");
    assert_eq!(tr["content"][0]["text"], "72F sunny");
}

#[tokio::test]
async fn tool_result_and_text_user_turns_split() {
    // A toolResult user turn followed by a normal-text user turn must NOT
    // merge (bedrock.py:1740-1742).
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::Tool {
                content: ToolContentInput::Text("result".to_string()),
                tool_call_id: "call_1".to_string(),
            },
            user_text("now do this"),
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    // Two separate user turns (split, not merged).
    assert_eq!(msgs.len(), 2);
    assert!(msgs[0]["content"][0].get("toolResult").is_some());
    assert_eq!(msgs[1]["content"][0]["text"], "now do this");
}

#[test]
fn parse_image_data_uri_rejects_non_data() {
    assert_eq!(parse_image_data_uri("https://x/y.png").expect("ok"), None);
}

#[tokio::test]
async fn response_format_json_object_builds_output_config() {
    let mut req = base_request("claude-sonnet-4-5", vec![user_text("hi")]);
    req.response_format = Some(ResponseFormat::JsonObject);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let oc = args.output_config.expect("output_config present");
    assert!(oc["textFormat"].is_object());
    assert_eq!(oc["textFormat"]["type"], "json_schema");
    // The synthesized schema is STRINGIFIED into
    // outputConfig.textFormat.structure.jsonSchema.schema. Bedrock rejects any
    // `object`-type schema that omits `additionalProperties: false`, so the
    // synthesized json_object schema MUST carry it.
    let schema = &oc["textFormat"]["structure"]["jsonSchema"]["schema"];
    let schema_str = schema
        .as_str()
        .unwrap_or_else(|| panic!("schema must be a string, got: {schema}"));
    let parsed: Value =
        serde_json::from_str(schema_str).expect("stringified schema must be valid JSON");
    assert_eq!(parsed["type"], "object");
    assert_eq!(parsed["additionalProperties"], false);
}

#[tokio::test]
async fn response_format_json_schema_stringifies_schema() {
    let mut req = base_request("claude-sonnet-4-5", vec![user_text("hi")]);
    req.response_format = Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaSpec {
            name: Some("x".to_string()),
            description: None,
            strict: None,
            schema: Some(json!({"type": "object", "properties": {"a": {"type": "string"}}})),
        },
    });
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let oc = args.output_config.expect("output_config present");
    let schema = &oc["textFormat"]["structure"]["jsonSchema"]["schema"];
    assert!(
        schema.is_string(),
        "schema must be stringified, got: {schema}"
    );
    assert_eq!(oc["textFormat"]["structure"]["jsonSchema"]["name"], "x");
}

#[tokio::test]
async fn response_format_unsupported_model_is_400() {
    let mut req = base_request("deepseek.v3", vec![user_text("hi")]);
    req.response_format = Some(ResponseFormat::JsonObject);
    let c = caps();
    let r = resolver(false);
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("must reject unsupported model");
    assert!(matches!(err, AppError::BadRequest(_)));
}

#[tokio::test]
async fn response_format_absent_is_noop() {
    let req = base_request("claude-sonnet-4-5", vec![user_text("hi")]);
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert!(args.output_config.is_none());
}

// ---- Property: message-order-preserving --------------------------------
//
// Feature: test-coverage-codecov, Property: message-order-preserving
// (see `.kiro/specs/test-coverage-codecov/design.md`, translate.rs row).
//
// Validates: Requirements 1.2

use futures::executor::block_on;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Feature: test-coverage-codecov, Property: message-order-preserving.
    ///
    /// For ANY sequence of user/assistant text messages, `to_converse_args`
    /// preserves order and role:
    /// - the output turn roles equal the run-length collapse of the input
    ///   roles (contiguous same-role text turns merge into one turn),
    /// - every text block appears in the original input order across turns.
    ///
    /// The model is one WITHOUT `no_assistant_prefill` / `drop_sampling_params`,
    /// so no continuation turn is appended and no sampling param is stripped —
    /// isolating the ordering behavior.
    #[test]
    fn prop_message_order_preserving(
        turns in prop::collection::vec((any::<bool>(), "[a-zA-Z0-9]{1,12}"), 1..8),
    ) {
        let messages: Vec<Message> = turns
            .iter()
            .map(|(is_user, text)| {
                if *is_user {
                    Message::User {
                        name: None,
                        content: ContentInput::Text(text.clone()),
                    }
                } else {
                    Message::Assistant {
                        name: None,
                        content: Some(ContentInput::Text(text.clone())),
                        tool_calls: None,
                    }
                }
            })
            .collect();

        let req = base_request("anthropic.claude-3-sonnet-v1:0", messages);
        let c = caps();
        let r = resolver(false);
        let args = block_on(to_converse_args(&req, &c, &r, &ConverseExtras::default()))
            .expect("translate");
        let msgs = args.messages.as_array().expect("messages array");

        // Expected roles = run-length collapse of the input roles.
        let mut expected_roles: Vec<&str> = Vec::new();
        for (is_user, _) in &turns {
            let role = if *is_user { "user" } else { "assistant" };
            if expected_roles.last() != Some(&role) {
                expected_roles.push(role);
            }
        }
        let actual_roles: Vec<&str> = msgs
            .iter()
            .map(|m| m["role"].as_str().expect("role string"))
            .collect();
        prop_assert_eq!(actual_roles, expected_roles);

        // Every text block is preserved in the original input order.
        let mut actual_texts: Vec<String> = Vec::new();
        for m in msgs {
            for block in m["content"].as_array().expect("content array") {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    actual_texts.push(t.to_string());
                }
            }
        }
        let expected_texts: Vec<String> = turns.iter().map(|(_, t)| t.clone()).collect();
        prop_assert_eq!(actual_texts, expected_texts);
    }
}

// ---- Coverage-deepening tests (test-coverage-codecov) -------------------
//
// These flat tests target branches in `to_converse_args` and its helpers that
// were previously unexercised: `ConverseArgs::to_value`, the `data:`-URI
// edge cases, tool-message parts extraction, invalid tool-call arguments,
// same-role tool merge/split, empty-message reframing, the response-format
// schema-default path, the beta-header injection loop, and the
// `merge_anthropic_beta` defensive arm. All remain fully offline (no AWS, no
// network, no sleep) and keep the `TestResolver` pattern.

use crate::config::{BudgetRatios, ReasoningPath};
use crate::domain::ResponsesBackend;
use crate::openai::schema::{ResponseFunction, ToolCall, ToolContentInput};

/// A minimal [`ModelCapabilities`] stub that reports exactly one capability and
/// a fixed beta-header list. Lets tests exercise the de-hardcoded beta-header
/// injection loop in `to_converse_args` without depending on the shipped
/// `models.toml` (which configures no per-model beta headers).
struct StubCaps {
    enabled: Capability,
    beta: Vec<String>,
}

impl ModelCapabilities for StubCaps {
    fn has(&self, _model: &str, cap: Capability) -> bool {
        cap == self.enabled
    }
    fn resolve_foundation(&self, model_or_profile: &str) -> String {
        model_or_profile.to_string()
    }
    fn budget_ratios(&self, _model: &str) -> Option<BudgetRatios> {
        None
    }
    fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
        None
    }
    fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
        None
    }
    fn beta_headers(&self, _model: &str) -> Vec<String> {
        self.beta.clone()
    }
    fn reasoning_path(&self, _model: &str) -> ReasoningPath {
        ReasoningPath::None
    }
    fn responses_backend(&self, _model: &str) -> ResponsesBackend {
        ResponsesBackend::Converse
    }
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
    }
    fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
        None
    }
}

#[test]
fn to_value_renders_all_present_slots() {
    let args = ConverseArgs {
        model_id: "m".to_string(),
        messages: json!([{ "role": "user", "content": [{ "text": "hi" }] }]),
        system: json!([{ "text": "sys" }]),
        inference_config: json!({ "maxTokens": 10 }),
        additional_model_request_fields: Some(json!({ "thinking": {} })),
        tool_config: Some(json!({ "tools": [] })),
        output_config: Some(json!({ "textFormat": {} })),
    };
    let v = args.to_value();
    assert_eq!(v["modelId"], "m");
    assert_eq!(v["messages"][0]["role"], "user");
    assert_eq!(v["system"][0]["text"], "sys");
    assert_eq!(v["inferenceConfig"]["maxTokens"], 10);
    assert!(v.get("additionalModelRequestFields").is_some());
    assert!(v.get("toolConfig").is_some());
    assert!(v.get("outputConfig").is_some());
}

#[test]
fn to_value_omits_absent_optional_slots() {
    let args = ConverseArgs {
        model_id: "m".to_string(),
        messages: json!([]),
        system: json!([]),
        inference_config: json!({ "maxTokens": 1 }),
        additional_model_request_fields: None,
        tool_config: None,
        output_config: None,
    };
    let v = args.to_value();
    assert!(v.get("additionalModelRequestFields").is_none());
    assert!(v.get("toolConfig").is_none());
    assert!(v.get("outputConfig").is_none());
}

#[test]
fn parse_image_data_uri_non_image_mime_returns_none() {
    // A `data:` URI whose MIME is not `image/*` is not decoded here.
    assert_eq!(
        parse_image_data_uri("data:text/plain;base64,aGk=").expect("ok"),
        None
    );
}

#[test]
fn parse_image_data_uri_without_base64_marker_returns_none() {
    // A `data:` URI without the `;base64,` marker is not an inline image.
    assert_eq!(
        parse_image_data_uri("data:image/png,rawbytes").expect("ok"),
        None
    );
}

#[test]
fn parse_image_data_uri_trims_leading_payload_whitespace() {
    // The `\s*` after the comma is stripped before base64 decoding.
    let decoded = parse_image_data_uri("data:image/png;base64,   aGk=")
        .expect("ok")
        .expect("decoded");
    assert_eq!(decoded.format, "png");
    assert_eq!(decoded.bytes, vec![104, 105]);
}

#[tokio::test]
async fn assistant_parts_content_is_parsed() {
    // Assistant content given as a parts array exercises the
    // `assistant_has_text(Parts)` branch and parses each text part.
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            user_text("hi"),
            Message::Assistant {
                name: None,
                content: Some(ContentInput::Parts(vec![ContentPart::Text(TextContent {
                    r#type: "text".to_string(),
                    text: "assistant part".to_string(),
                })])),
                tool_calls: None,
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["content"][0]["text"], "assistant part");
}

#[tokio::test]
async fn tool_message_parts_content_covers_all_item_shapes() {
    // A tool message whose content is a parts array exercises every branch of
    // `extract_tool_content`: object+valid-JSON-text (pretty-printed),
    // object+invalid-JSON-text, object+plain-text, object+non-string-text,
    // object without a `text` key, a bare string item, and a non-object/
    // non-string item.
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::Tool {
            content: ToolContentInput::Parts(vec![
                json!({ "text": "{\"a\": 1}" }),
                json!({ "text": "{oops}" }),
                json!({ "text": "plain text" }),
                json!({ "text": 42 }),
                json!({ "note": "no text key" }),
                json!("bare string"),
                json!(99),
            ]),
            tool_call_id: "call_1".to_string(),
        }],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let text = args.messages[0]["content"][0]["toolResult"]["content"][0]["text"]
        .as_str()
        .expect("tool result text");
    // Valid embedded JSON is pretty-printed.
    assert!(text.contains("\"a\": 1"), "pretty JSON missing: {text}");
    // Invalid JSON-looking text is passed through verbatim.
    assert!(text.contains("{oops}"), "invalid-JSON passthrough: {text}");
    assert!(text.contains("plain text"), "plain text: {text}");
    // Non-string `text` value is stringified.
    assert!(text.contains("42"), "non-string text: {text}");
    // Object without a `text` key is JSON-serialized.
    assert!(text.contains("no text key"), "no-text object: {text}");
    assert!(text.contains("bare string"), "bare string item: {text}");
    assert!(text.contains("99"), "scalar item: {text}");
}

#[tokio::test]
async fn invalid_tool_call_arguments_json_is_bad_request() {
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            user_text("go"),
            Message::Assistant {
                name: None,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    index: None,
                    id: Some("call_1".to_string()),
                    r#type: "function".to_string(),
                    function: ResponseFunction {
                        name: Some("f".to_string()),
                        arguments: "not-json".to_string(),
                    },
                }]),
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let err = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect_err("invalid tool_call arguments must reject");
    match err {
        AppError::BadRequest(message) => {
            assert!(message.contains("invalid tool_call arguments JSON"))
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn contiguous_tool_results_merge_into_one_user_turn() {
    // Two adjacent tool messages (both toolResult) MUST merge into a single
    // user turn (bedrock.py:1734-1736 same-role merge).
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            Message::Tool {
                content: ToolContentInput::Text("r1".to_string()),
                tool_call_id: "call_1".to_string(),
            },
            Message::Tool {
                content: ToolContentInput::Text("r2".to_string()),
                tool_call_id: "call_2".to_string(),
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    assert_eq!(msgs.len(), 1, "two tool results must merge into one turn");
    let content = msgs[0]["content"].as_array().expect("content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["toolResult"]["toolUseId"], "call_1");
    assert_eq!(content[1]["toolResult"]["toolUseId"], "call_2");
}

#[tokio::test]
async fn contiguous_assistant_tool_uses_merge_into_one_turn() {
    // Two adjacent assistant messages that each carry a toolUse MUST merge.
    let mk_call = |id: &str, name: &str| Message::Assistant {
        name: None,
        content: None,
        tool_calls: Some(vec![ToolCall {
            index: None,
            id: Some(id.to_string()),
            r#type: "function".to_string(),
            function: ResponseFunction {
                name: Some(name.to_string()),
                arguments: "{}".to_string(),
            },
        }]),
    };
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![user_text("go"), mk_call("c1", "a"), mk_call("c2", "b")],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    // user, then a single merged assistant turn with two toolUse blocks.
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1]["role"], "assistant");
    let content = msgs[1]["content"].as_array().expect("content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["toolUse"]["toolUseId"], "c1");
    assert_eq!(content[1]["toolUse"]["toolUseId"], "c2");
}

#[tokio::test]
async fn assistant_tool_use_then_text_splits() {
    // A toolUse-only assistant turn followed by a text-only assistant turn must
    // NOT merge (`should_split_same_role_merge`, assistant branch).
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![
            user_text("go"),
            Message::Assistant {
                name: None,
                content: None,
                tool_calls: Some(vec![ToolCall {
                    index: None,
                    id: Some("c1".to_string()),
                    r#type: "function".to_string(),
                    function: ResponseFunction {
                        name: Some("a".to_string()),
                        arguments: "{}".to_string(),
                    },
                }]),
            },
            Message::Assistant {
                name: None,
                content: Some(ContentInput::Text("done".to_string())),
                tool_calls: None,
            },
        ],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let msgs = args.messages.as_array().expect("messages");
    // user, assistant(toolUse), assistant(text) — three distinct turns.
    assert_eq!(msgs.len(), 3);
    assert!(msgs[1]["content"][0].get("toolUse").is_some());
    assert_eq!(msgs[2]["content"][0]["text"], "done");
}

#[tokio::test]
async fn only_system_message_yields_empty_messages() {
    // A request with no user/assistant/tool turns reframes to an empty message
    // array (the `reformatted.last()` None path).
    let req = base_request(
        "anthropic.claude-3-sonnet-v1:0",
        vec![Message::System {
            name: None,
            content: SystemContentInput::Text("only system".to_string()),
        }],
    );
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    assert_eq!(args.messages.as_array().expect("messages").len(), 0);
    assert_eq!(args.system[0]["text"], "only system");
}

#[tokio::test]
async fn response_format_json_schema_without_schema_defaults_to_object() {
    // A json_schema response_format that omits `schema` defaults to
    // `{"type":"object"}` (stringified into the jsonSchema slot).
    let mut req = base_request("claude-sonnet-4-5", vec![user_text("hi")]);
    req.response_format = Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaSpec {
            name: None,
            description: None,
            strict: None,
            schema: None,
        },
    });
    let c = caps();
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let oc = args.output_config.expect("output_config present");
    let schema_str = oc["textFormat"]["structure"]["jsonSchema"]["schema"]
        .as_str()
        .expect("schema string");
    let parsed: Value = serde_json::from_str(schema_str).expect("valid JSON");
    assert_eq!(parsed["type"], "object");
    // No name was supplied, so the jsonSchema slot omits `name`.
    assert!(oc["textFormat"]["structure"]["jsonSchema"]
        .get("name")
        .is_none());
}

#[tokio::test]
async fn context_1m_beta_injects_configured_header() {
    // With a model that both declares Context1mBeta AND has a configured beta
    // header, the injection loop merges the header into anthropic_beta. Uses a
    // stub capability source (the shipped config sets no per-model headers).
    let req = base_request("stub-model", vec![user_text("hi")]);
    let c = StubCaps {
        enabled: Capability::Context1mBeta,
        beta: vec!["ctx-1m-2025".to_string()],
    };
    let r = resolver(false);
    let args = to_converse_args(&req, &c, &r, &ConverseExtras::default())
        .await
        .expect("translate");
    let fields = args
        .additional_model_request_fields
        .expect("fields present");
    assert_eq!(fields["anthropic_beta"], json!(["ctx-1m-2025"]));
}

#[test]
fn merge_anthropic_beta_replaces_non_string_non_list_value() {
    // Defensive arm: a non-string/non-list existing value is replaced with a
    // singleton header list.
    let mut m = Map::new();
    m.insert("anthropic_beta".to_string(), json!(42));
    merge_anthropic_beta(&mut m, "ctx-1m");
    assert_eq!(m["anthropic_beta"], json!(["ctx-1m"]));
}

#[test]
fn reqwest_image_resolver_construct_predicate_and_debug() {
    // The network-backed resolver's constructor, modality predicate, and Debug
    // impl are exercised without any network access.
    let resolver = ReqwestImageResolver::new(|m| m == "img-model");
    assert!(resolver.supports_image("img-model"));
    assert!(!resolver.supports_image("other"));
    let dbg = format!("{resolver:?}");
    assert!(dbg.contains("ReqwestImageResolver"));
}
