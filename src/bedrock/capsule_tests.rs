use std::collections::HashMap;

use serde_json::{json, Value};

use super::{
    decode_capsule, decode_responses_capsule, encode_capsule, encode_responses_capsule, is_capsule,
    is_responses_capsule, resolve_capsule_runtime, CapsuleKeyring, MAX_CAPSULE_WIRE_BYTES,
};
use crate::config::AppSettings;
use crate::error::AppError;

fn keyring() -> CapsuleKeyring {
    CapsuleKeyring::new(
        HashMap::from([("current".to_string(), b"test-capsule-key".to_vec())]),
        Some("current".to_string()),
    )
}

fn signed_text_blocks() -> Vec<Value> {
    vec![json!({
        "reasoningText": {
            "text": "private reasoning",
            "signature": "provider-signature"
        }
    })]
}

fn encoded(blocks: &[Value]) -> String {
    encode_capsule("tool-123", blocks, &keyring()).expect("capsule encodes")
}

fn runtime_settings(
    encoder_enabled: bool,
    active_kid: Option<&str>,
    keys: Option<&str>,
) -> AppSettings {
    AppSettings {
        api_route_prefix: "/api/v1".to_string(),
        debug: false,
        aws_region: "us-west-2".to_string(),
        default_model: "model".to_string(),
        default_embedding_model: "embedding".to_string(),
        enable_cross_region_inference: false,
        enable_application_inference_profiles: false,
        enable_prompt_caching: false,
        prompt_cache_ttl: "5m".to_string(),
        chat_reasoning_capsule_enabled: encoder_enabled,
        chat_reasoning_capsule_active_kid: active_kid.map(str::to_string),
        chat_reasoning_capsule_keys: keys.map(str::to_string),
        api_key: None,
        api_key_secret_arn: None,
        api_key_param_name: None,
        bedrock_api_key: None,
        disable_mantle: false,
        bind_addr: "127.0.0.1".to_string(),
        port: 8080,
        log_level: "info".to_string(),
        aws_connect_timeout_secs: 60,
        aws_read_timeout_secs: 900,
        responses_stream_idle_timeout_secs: 180,
        aws_max_retry_attempts: 8,
        max_body_size_mb: 20,
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
    }
}

#[test]
fn runtime_allows_disabled_encoder_without_keys() {
    let settings = runtime_settings(false, None, None);

    let runtime = resolve_capsule_runtime(&settings).expect("disabled runtime resolves");

    assert!(!runtime.encoder_enabled);
    assert_eq!(runtime.keyring.active_kid(), None);
    assert_eq!(runtime.keyring.key_for("unknown"), None);
}

#[test]
fn runtime_allows_enabled_encoder_with_active_key() {
    let settings = runtime_settings(true, Some("current"), Some("current:c2VjcmV0"));

    let runtime = resolve_capsule_runtime(&settings).expect("enabled runtime resolves");

    assert!(runtime.encoder_enabled);
    assert_eq!(runtime.keyring.active_kid(), Some("current"));
    assert_eq!(
        runtime.keyring.key_for("current"),
        Some(b"secret".as_slice())
    );
}

#[test]
fn runtime_rejects_enabled_encoder_when_active_key_is_missing() {
    let settings = runtime_settings(true, Some("current"), Some("previous:cHJldmlvdXM"));

    let error = resolve_capsule_runtime(&settings)
        .err()
        .expect("missing active key must fail");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn runtime_rejects_enabled_encoder_without_active_kid() {
    let settings = runtime_settings(true, None, Some("current:c2VjcmV0"));

    let error = resolve_capsule_runtime(&settings)
        .err()
        .expect("missing active kid must fail");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn runtime_parses_multiple_keyring_entries() {
    let settings = runtime_settings(false, None, Some("current:c2VjcmV0,previous:cHJldmlvdXM"));

    let runtime = resolve_capsule_runtime(&settings).expect("keyring resolves");

    assert_eq!(
        runtime.keyring.key_for("current"),
        Some(b"secret".as_slice())
    );
    assert_eq!(
        runtime.keyring.key_for("previous"),
        Some(b"previous".as_slice())
    );
}

#[test]
fn runtime_rejects_malformed_keyring_entry() {
    let settings = runtime_settings(false, None, Some("missing-separator"));

    let error = resolve_capsule_runtime(&settings)
        .err()
        .expect("malformed keyring must fail");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn runtime_rejects_invalid_base64_key() {
    let settings = runtime_settings(false, None, Some("current:***"));

    let error = resolve_capsule_runtime(&settings)
        .err()
        .expect("invalid key encoding must fail");

    assert!(matches!(error, AppError::Internal(_)));
}

fn replace_segment_byte(capsule: &str, segment_index: usize) -> String {
    let mut segments: Vec<String> = capsule.split('.').map(str::to_string).collect();
    let segment = segments
        .get_mut(segment_index)
        .expect("capsule segment exists");
    let replacement = if segment.starts_with('A') { 'B' } else { 'A' };
    segment.replace_range(0..1, &replacement.to_string());
    segments.join(".")
}

#[test]
fn round_trip_preserves_tool_use_id_and_reasoning_blocks() {
    let blocks = signed_text_blocks();
    let capsule = encoded(&blocks);

    let decoded = decode_capsule(&capsule, &keyring()).expect("capsule decodes");

    assert_eq!(decoded.tool_use_id, "tool-123");
    assert_eq!(decoded.reasoning_blocks, blocks);
}

#[test]
fn round_trip_accepts_redacted_content_block() {
    let blocks = vec![json!({ "redactedContent": "opaque-content" })];
    let capsule = encoded(&blocks);

    let decoded = decode_capsule(&capsule, &keyring()).expect("redacted capsule decodes");

    assert_eq!(decoded.reasoning_blocks, blocks);
}

#[test]
fn tampered_payload_is_rejected() {
    let capsule = encoded(&signed_text_blocks());
    let tampered = replace_segment_byte(&capsule, 1);

    let error = decode_capsule(&tampered, &keyring()).expect_err("tampered payload must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn tampered_tag_is_rejected_by_hmac_verification() {
    let capsule = encoded(&signed_text_blocks());
    let tampered = replace_segment_byte(&capsule, 2);

    let error = decode_capsule(&tampered, &keyring()).expect_err("tampered tag must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn unknown_kid_is_rejected() {
    let capsule = encoded(&signed_text_blocks());
    let unknown_keyring = CapsuleKeyring::new(HashMap::new(), None);

    let error = decode_capsule(&capsule, &unknown_keyring).expect_err("unknown kid must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn capsule_without_tag_is_rejected() {
    let capsule = encoded(&signed_text_blocks());
    let without_tag = capsule.rsplit_once('.').expect("tag segment exists").0;

    let error = decode_capsule(without_tag, &keyring()).expect_err("two-part capsule must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn non_capsule_is_not_recognized_and_is_rejected_by_decoder() {
    let candidate = "ordinary-tool-id";

    assert!(!is_capsule(candidate));
    assert!(matches!(
        decode_capsule(candidate, &keyring()),
        Err(AppError::BadRequest(_))
    ));
}

#[test]
fn wire_size_limit_is_checked_before_decoding() {
    let candidate = "x".repeat(MAX_CAPSULE_WIRE_BYTES + 1);

    let error = decode_capsule(&candidate, &keyring()).expect_err("oversize wire must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn encoder_rejects_capsule_above_wire_size_limit() {
    let blocks = vec![json!({
        "reasoningText": {
            "text": "x".repeat(MAX_CAPSULE_WIRE_BYTES),
            "signature": "provider-signature"
        }
    })];
    let error = encode_capsule("tool-123", &blocks, &keyring())
        .expect_err("encoder must not mint an oversized capsule");

    assert!(matches!(error, AppError::Internal(_)));
}

#[test]
fn invalid_reasoning_block_is_rejected() {
    let capsule = encoded(&[json!({ "text": "unsigned" })]);

    let error =
        decode_capsule(&capsule, &keyring()).expect_err("invalid reasoning block must fail");

    assert!(matches!(error, AppError::BadRequest(_)));
}

#[test]
fn responses_capsule_round_trip_and_namespace_are_independent() {
    let items = vec![json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque"
    })];
    let capsule = encode_responses_capsule("call-1", &items, &keyring()).expect("capsule encodes");

    assert!(is_responses_capsule(&capsule));
    assert!(!is_capsule(&capsule));
    let decoded =
        decode_responses_capsule(&capsule, &keyring()).expect("responses capsule decodes");
    assert_eq!(decoded.call_id, "call-1");
    assert_eq!(decoded.reasoning_items, items);
    assert!(decode_capsule(&capsule, &keyring()).is_err());
}

#[test]
fn responses_capsule_rejects_missing_encrypted_content() {
    let error = encode_responses_capsule(
        "call-1",
        &[json!({"type": "reasoning", "id": "rs_1"})],
        &keyring(),
    )
    .expect_err("unsigned Responses reasoning must fail");
    assert!(matches!(error, AppError::Internal(_)));
}
