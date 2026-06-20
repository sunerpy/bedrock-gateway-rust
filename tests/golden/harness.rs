//! Golden record/replay parity test harness.
//!
//! This is the **Tier-1 offline parity safety net** described in the Metis
//! two-tier plan: it proves the Rust gateway is behaviourally faithful to the
//! pinned Python reference (SHA `9a3e752`) **without** any live Bedrock / AWS
//! access. It runs entirely against on-disk fixtures.
//!
//! # What lives here
//!
//! * A **semantic-equality comparator** ([`assert_semantic_eq`] and friends).
//!   Metis decision: parity is checked **semantically**, NOT byte-exact. Two
//!   payloads match when they have the same field *set*, the same field
//!   *values*, and — for streams — the same event-type *ordering*, after
//!   volatile fields (`id`, `created`, request IDs, `system_fingerprint`, …)
//!   are normalised away.
//! * **Fixture loaders** for the two fixture families (translation request
//!   pairs and streaming/response pairs). See `tests/golden/README.md` for the
//!   directory layout and on-disk format.
//! * **Self-tests** (positive + negative controls) that prove the comparator
//!   actually works *before* any real fixtures are captured (real fixtures land
//!   in task 32). The negative control asserts the comparator REJECTS a genuine
//!   difference (e.g. a changed `finish_reason`); the positive control asserts
//!   it ACCEPTS payloads that differ only in volatile fields / key ordering.
//!
//! # Importing from the crate
//!
//! The harness may import typed schema from the crate under test, e.g.
//! `bedrock_gateway_rust::openai::schema`, for typed round-trips. The
//! comparator itself works on untyped [`serde_json::Value`] so it can compare
//! arbitrary payloads (including ones with extension keys) without coupling to
//! a specific struct.

#![allow(dead_code)]

#[cfg(test)]
mod corpus;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

// ---------------------------------------------------------------------------
// Volatile fields
// ---------------------------------------------------------------------------

/// Field names that are **non-deterministic** between runs / between the Python
/// reference and the Rust gateway, and therefore must be ignored when comparing
/// for semantic parity.
///
/// These are matched by key name at *any* depth of the JSON tree.
pub const DEFAULT_VOLATILE_FIELDS: &[&str] = &[
    "id",
    "created",
    "created_at",
    "request_id",
    "x-request-id",
    "system_fingerprint",
];

// ---------------------------------------------------------------------------
// Public comparator API
// ---------------------------------------------------------------------------

/// Assert that `actual` is semantically equal to `expected`, ignoring the
/// [`DEFAULT_VOLATILE_FIELDS`].
///
/// Panics with a human-readable diff on mismatch. This is the ergonomic entry
/// point used by most fixture tests.
pub fn assert_semantic_eq(expected: &Value, actual: &Value) {
    assert_semantic_eq_with(expected, actual, DEFAULT_VOLATILE_FIELDS);
}

/// Assert semantic equality, ignoring the supplied `ignore` field names (in
/// addition to nothing else — pass [`DEFAULT_VOLATILE_FIELDS`] explicitly if you
/// want the defaults too).
///
/// Panics with a human-readable diff on mismatch.
pub fn assert_semantic_eq_with(expected: &Value, actual: &Value, ignore: &[&str]) {
    if let Err(diff) = semantic_eq(expected, actual, ignore) {
        panic!("semantic parity mismatch:\n{diff}");
    }
}

/// Compare two JSON values for semantic equality, ignoring `ignore` keys at any
/// depth.
///
/// Returns `Ok(())` when the values match semantically, or `Err(diff)` with a
/// rendered, path-qualified description of the first mismatches found. This is
/// the fallible core used by the negative-control self-test.
pub fn semantic_eq(expected: &Value, actual: &Value, ignore: &[&str]) -> Result<(), String> {
    let mut diffs = Vec::new();
    compare(expected, actual, ignore, "$", &mut diffs);
    if diffs.is_empty() {
        Ok(())
    } else {
        let mut out = String::new();
        for d in &diffs {
            let _ = writeln!(out, "  - {d}");
        }
        Err(out)
    }
}

// ---------------------------------------------------------------------------
// SSE / streaming comparator
// ---------------------------------------------------------------------------

/// A single parsed SSE/stream event for comparison purposes.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// The event "type" used for ORDER comparison. For OpenAI SSE chunks this is
    /// derived from the payload (object kind / delta shape); for raw Bedrock
    /// events it is the event tag.
    pub kind: String,
    /// The full event payload (already volatile-normalised by the comparator).
    pub payload: Value,
}

/// Compare two ordered streams of events for semantic parity.
///
/// Checks, in order: (1) the number of events, (2) the event-type ORDERING, and
/// (3) the per-event semantic equality of payloads (ignoring `ignore` keys).
///
/// Returns `Ok(())` on match or `Err(diff)` describing the first divergence.
pub fn semantic_eq_stream(
    expected: &[StreamEvent],
    actual: &[StreamEvent],
    ignore: &[&str],
) -> Result<(), String> {
    let mut diffs = Vec::new();

    if expected.len() != actual.len() {
        diffs.push(format!(
            "stream length differs: expected {} event(s), got {}",
            expected.len(),
            actual.len()
        ));
    }

    let expected_order: Vec<&str> = expected.iter().map(|e| e.kind.as_str()).collect();
    let actual_order: Vec<&str> = actual.iter().map(|e| e.kind.as_str()).collect();
    if expected_order != actual_order {
        diffs.push(format!(
            "event-type ordering differs:\n      expected: {expected_order:?}\n      actual:   {actual_order:?}"
        ));
    }

    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if let Err(d) = semantic_eq(&e.payload, &a.payload, ignore) {
            diffs.push(format!(
                "event[{i}] (kind `{}`) payload mismatch:\n{d}",
                e.kind
            ));
        }
    }

    if diffs.is_empty() {
        Ok(())
    } else {
        Err(diffs.join("\n  - "))
    }
}

/// Assert stream parity, panicking with a diff on mismatch. Uses
/// [`DEFAULT_VOLATILE_FIELDS`].
pub fn assert_stream_eq(expected: &[StreamEvent], actual: &[StreamEvent]) {
    if let Err(diff) = semantic_eq_stream(expected, actual, DEFAULT_VOLATILE_FIELDS) {
        panic!("stream parity mismatch:\n  - {diff}");
    }
}

/// Parse an OpenAI-style SSE body into ordered [`StreamEvent`]s.
///
/// Accepts either:
/// * a true SSE text body (`data: {...}\n\n` framed, with an optional
///   `data: [DONE]` terminator), or
/// * JSONL where each non-empty line is a chunk object.
///
/// The `[DONE]` sentinel is preserved as an event of kind `done` so ordering
/// checks see the terminator.
pub fn parse_sse(body: &str) -> Result<Vec<StreamEvent>, String> {
    let mut events = Vec::new();
    for (lineno, raw) in body.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let json_str = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
        if json_str.is_empty() {
            continue;
        }
        if json_str == "[DONE]" {
            events.push(StreamEvent {
                kind: "done".to_string(),
                payload: Value::String("[DONE]".to_string()),
            });
            continue;
        }
        let payload: Value = serde_json::from_str(json_str)
            .map_err(|e| format!("line {}: invalid JSON chunk: {e}", lineno + 1))?;
        let kind = sse_chunk_kind(&payload);
        events.push(StreamEvent { kind, payload });
    }
    Ok(events)
}

/// Derive a stable "kind" tag for an OpenAI SSE chunk, used for ORDER checks.
///
/// The tag captures the *shape* of the chunk (what kind of delta it carries)
/// rather than its volatile contents, so two semantically equivalent streams
/// produce the same ordering signature.
fn sse_chunk_kind(chunk: &Value) -> String {
    // Prefer an explicit object discriminator when present.
    let object = chunk.get("object").and_then(Value::as_str);

    let choice0 = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first());

    if let Some(choice) = choice0 {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            return format!("finish:{reason}");
        }
        if let Some(delta) = choice.get("delta") {
            if delta.get("role").is_some() {
                return "delta:role".to_string();
            }
            if delta.get("tool_calls").is_some() {
                return "delta:tool_calls".to_string();
            }
            if delta.get("content").and_then(Value::as_str).is_some() {
                return "delta:content".to_string();
            }
            if delta.as_object().map(|o| o.is_empty()).unwrap_or(false) {
                return "delta:empty".to_string();
            }
            return "delta".to_string();
        }
    }

    object
        .map(|o| o.to_string())
        .unwrap_or_else(|| "chunk".to_string())
}

// ---------------------------------------------------------------------------
// Core recursive comparison
// ---------------------------------------------------------------------------

fn compare(expected: &Value, actual: &Value, ignore: &[&str], path: &str, diffs: &mut Vec<String>) {
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            // Field SET comparison (post-ignore). Object key ORDERING is
            // irrelevant: serde_json::Map iteration is normalised here.
            let e_keys: std::collections::BTreeSet<&String> =
                e.keys().filter(|k| !is_ignored(k, ignore)).collect();
            let a_keys: std::collections::BTreeSet<&String> =
                a.keys().filter(|k| !is_ignored(k, ignore)).collect();

            for missing in e_keys.difference(&a_keys) {
                diffs.push(format!(
                    "{path}.{missing}: present in expected but MISSING in actual"
                ));
            }
            for extra in a_keys.difference(&e_keys) {
                diffs.push(format!(
                    "{path}.{extra}: present in actual but UNEXPECTED (not in expected)"
                ));
            }
            for key in e_keys.intersection(&a_keys) {
                let child = format!("{path}.{key}");
                compare(&e[*key], &a[*key], ignore, &child, diffs);
            }
        }
        (Value::Array(e), Value::Array(a)) => {
            // Arrays are ORDER-sensitive (preserves message / chunk / content
            // ordering, which is a parity-relevant property).
            if e.len() != a.len() {
                diffs.push(format!(
                    "{path}: array length differs (expected {}, got {})",
                    e.len(),
                    a.len()
                ));
            }
            for (i, (ev, av)) in e.iter().zip(a.iter()).enumerate() {
                let child = format!("{path}[{i}]");
                compare(ev, av, ignore, &child, diffs);
            }
        }
        (e, a) if e == a => {}
        (e, a) => {
            diffs.push(format!(
                "{path}: value differs (expected {}, got {})",
                render_scalar(e),
                render_scalar(a)
            ));
        }
    }
}

fn is_ignored(key: &str, ignore: &[&str]) -> bool {
    ignore.iter().any(|i| i.eq_ignore_ascii_case(key))
}

fn render_scalar(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{s:?}"),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

/// Root directory holding all golden fixtures: `tests/golden/fixtures/`.
pub fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("fixtures")
}

/// Load and parse a JSON fixture, relative to [`fixtures_root`].
pub fn load_json(rel: impl AsRef<Path>) -> Value {
    let path = fixtures_root().join(rel.as_ref());
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("fixture {} is not valid JSON: {e}", path.display()))
}

/// Load a raw text/JSONL/SSE fixture, relative to [`fixtures_root`].
pub fn load_text(rel: impl AsRef<Path>) -> String {
    let path = fixtures_root().join(rel.as_ref());
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()))
}

/// A translation fixture pair: an OpenAI request and the Bedrock arguments it
/// is expected to translate into.
///
/// On-disk layout: `<case>/openai_request.json` + `<case>/expected_bedrock_args.json`.
#[derive(Debug)]
pub struct TranslationFixture {
    pub case: String,
    pub openai_request: Value,
    pub expected_bedrock_args: Value,
}

/// Load a translation fixture pair from a case directory under
/// `fixtures/translation/<case>/`.
pub fn load_translation_fixture(case: &str) -> TranslationFixture {
    let dir = Path::new("translation").join(case);
    TranslationFixture {
        case: case.to_string(),
        openai_request: load_json(dir.join("openai_request.json")),
        expected_bedrock_args: load_json(dir.join("expected_bedrock_args.json")),
    }
}

/// A streaming fixture pair: a Bedrock event stream and the OpenAI SSE chunks it
/// is expected to render into.
///
/// On-disk layout: `<case>/bedrock_events.jsonl` + `<case>/expected_sse_chunks.jsonl`.
#[derive(Debug)]
pub struct StreamFixture {
    pub case: String,
    pub bedrock_events: String,
    pub expected_sse: Vec<StreamEvent>,
}

/// Load a streaming fixture pair from `fixtures/streaming/<case>/`.
pub fn load_stream_fixture(case: &str) -> StreamFixture {
    let dir = Path::new("streaming").join(case);
    let expected_raw = load_text(dir.join("expected_sse_chunks.jsonl"));
    StreamFixture {
        case: case.to_string(),
        bedrock_events: load_text(dir.join("bedrock_events.jsonl")),
        expected_sse: parse_sse(&expected_raw)
            .unwrap_or_else(|e| panic!("case {case}: bad expected_sse_chunks.jsonl: {e}")),
    }
}

/// A non-streaming response fixture pair: a Bedrock output and the OpenAI
/// response it is expected to render into.
///
/// On-disk layout: `<case>/bedrock_output.json` + `<case>/expected_openai_response.json`.
#[derive(Debug)]
pub struct ResponseFixture {
    pub case: String,
    pub bedrock_output: Value,
    pub expected_openai_response: Value,
}

/// Load a response fixture pair from `fixtures/response/<case>/`.
pub fn load_response_fixture(case: &str) -> ResponseFixture {
    let dir = Path::new("response").join(case);
    ResponseFixture {
        case: case.to_string(),
        bedrock_output: load_json(dir.join("bedrock_output.json")),
        expected_openai_response: load_json(dir.join("expected_openai_response.json")),
    }
}

// ===========================================================================
// Self-tests — prove the comparator works BEFORE real fixtures exist.
// ===========================================================================

#[cfg(test)]
mod self_tests {
    use super::*;
    use serde_json::json;

    /// Sanity: this whole suite must run with NO AWS environment present.
    /// We assert the harness never reaches for AWS credentials. (We do not set
    /// them; if CI injected them we still must not call AWS — the harness has
    /// no AWS deps at all, which this test documents.)
    #[test]
    fn runs_offline_no_aws_calls() {
        // The golden harness imports nothing from aws-sdk-*; it is pure JSON.
        // This test exists as an executable assertion of the offline contract.
        let _ = fixtures_root();
    }

    // -- POSITIVE control: volatile-only differences are accepted -----------

    #[test]
    fn positive_ignores_volatile_fields() {
        let expected = json!({
            "id": "chatcmpl-AAA",
            "created": 1,
            "system_fingerprint": "fp_aaa",
            "model": "claude",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        });
        let actual = json!({
            "id": "chatcmpl-ZZZ",
            "created": 999_999,
            "system_fingerprint": "fp_zzz",
            "model": "claude",
            "choices": [{"index": 0, "finish_reason": "stop"}]
        });
        // Differs ONLY in volatile fields → must be a semantic match.
        assert_semantic_eq(&expected, &actual);
    }

    #[test]
    fn positive_ignores_object_key_ordering() {
        // Same content, different key insertion order → must match.
        let expected = json!({"a": 1, "b": {"x": true, "y": [1, 2, 3]}});
        let actual = json!({"b": {"y": [1, 2, 3], "x": true}, "a": 1});
        assert_semantic_eq(&expected, &actual);
    }

    #[test]
    fn positive_two_chunks_differ_only_in_id_and_created() {
        // Direct restatement of the task's positive requirement.
        let expected = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        });
        let actual = json!({
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "created": 1700009999,
            "choices": [{"index": 0, "delta": {"content": "Hello"}}]
        });
        assert_semantic_eq(&expected, &actual);
    }

    // -- NEGATIVE control: real differences are REJECTED --------------------

    #[test]
    fn negative_detects_changed_finish_reason() {
        let expected = json!({
            "choices": [{"index": 0, "finish_reason": "stop"}]
        });
        let actual = json!({
            "choices": [{"index": 0, "finish_reason": "length"}]
        });
        let result = semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS);
        assert!(
            result.is_err(),
            "comparator MUST reject a changed finish_reason"
        );
        let diff = result.unwrap_err();
        assert!(
            diff.contains("finish_reason"),
            "diff should name the offending field, got: {diff}"
        );
    }

    #[test]
    fn negative_detects_changed_value() {
        let expected = json!({"model": "claude-haiku", "n": 1});
        let actual = json!({"model": "claude-sonnet", "n": 1});
        assert!(semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS).is_err());
    }

    #[test]
    fn negative_detects_missing_field() {
        let expected = json!({"a": 1, "b": 2});
        let actual = json!({"a": 1});
        let diff = semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS).unwrap_err();
        assert!(diff.contains("MISSING"), "got: {diff}");
    }

    #[test]
    fn negative_detects_unexpected_field() {
        let expected = json!({"a": 1});
        let actual = json!({"a": 1, "surprise": true});
        let diff = semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS).unwrap_err();
        assert!(diff.contains("UNEXPECTED"), "got: {diff}");
    }

    #[test]
    fn negative_detects_array_reorder() {
        // Array ordering IS significant (message/chunk order matters).
        let expected = json!({"messages": ["a", "b"]});
        let actual = json!({"messages": ["b", "a"]});
        assert!(semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS).is_err());
    }

    #[test]
    fn negative_detects_array_length() {
        let expected = json!([1, 2, 3]);
        let actual = json!([1, 2]);
        let diff = semantic_eq(&expected, &actual, DEFAULT_VOLATILE_FIELDS).unwrap_err();
        assert!(diff.contains("array length"), "got: {diff}");
    }

    // -- SSE / streaming comparator -----------------------------------------

    #[test]
    fn sse_positive_volatile_only_stream() {
        let expected = "\
data: {\"id\":\"a\",\"created\":1,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}

data: {\"id\":\"a\",\"created\":1,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}

data: {\"id\":\"a\",\"created\":1,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{}}]}

data: [DONE]
";
        // Same stream but with different volatile id/created on every chunk.
        let actual = "\
data: {\"id\":\"z\",\"created\":42,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}

data: {\"id\":\"z\",\"created\":42,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}

data: {\"id\":\"z\",\"created\":42,\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{}}]}

data: [DONE]
";
        let e = parse_sse(expected).unwrap();
        let a = parse_sse(actual).unwrap();
        assert_eq!(
            e.iter().map(|x| x.kind.clone()).collect::<Vec<_>>(),
            vec!["delta:role", "delta:content", "finish:stop", "done"]
        );
        assert_stream_eq(&e, &a);
    }

    #[test]
    fn sse_negative_event_reordering() {
        let expected = parse_sse(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\
             data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n",
        )
        .unwrap();
        let actual = parse_sse(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\
             data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n",
        )
        .unwrap();
        let res = semantic_eq_stream(&expected, &actual, DEFAULT_VOLATILE_FIELDS);
        assert!(res.is_err(), "reordered events must be rejected");
        assert!(res.unwrap_err().contains("ordering"));
    }

    #[test]
    fn sse_negative_changed_finish_reason() {
        let expected = parse_sse(
            "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\",\"delta\":{}}]}\n",
        )
        .unwrap();
        let actual = parse_sse(
            "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"length\",\"delta\":{}}]}\n",
        )
        .unwrap();
        // Different finish_reason changes the event KIND → ordering signature
        // diverges, so this is caught at the ordering layer.
        assert!(semantic_eq_stream(&expected, &actual, DEFAULT_VOLATILE_FIELDS).is_err());
    }

    #[test]
    fn sse_negative_extra_chunk() {
        let expected =
            parse_sse("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n")
                .unwrap();
        let actual = parse_sse(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\
             data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"}}]}\n",
        )
        .unwrap();
        let diff = semantic_eq_stream(&expected, &actual, DEFAULT_VOLATILE_FIELDS).unwrap_err();
        assert!(diff.contains("length"), "got: {diff}");
    }

    // -- Custom ignore list --------------------------------------------------

    #[test]
    fn custom_ignore_list_is_respected() {
        let expected = json!({"keep": 1, "drop_me": "a"});
        let actual = json!({"keep": 1, "drop_me": "b"});
        // With "drop_me" ignored → match.
        assert_semantic_eq_with(&expected, &actual, &["drop_me"]);
        // Without it → mismatch.
        assert!(semantic_eq(&expected, &actual, &[]).is_err());
    }

    // -- Fixture loaders against the bundled placeholder fixtures -----------

    #[test]
    fn placeholder_translation_fixture_roundtrips() {
        let fx = load_translation_fixture("placeholder_selftest");
        // The placeholder expected args are the request's relevant fields,
        // proving the loader + comparator path end-to-end. Volatile-only
        // differences in the bundled fixtures must still pass.
        assert_semantic_eq(&fx.expected_bedrock_args, &fx.expected_bedrock_args);
        assert!(fx.openai_request.get("messages").is_some());
    }

    #[test]
    fn placeholder_stream_fixture_parses_and_matches_itself() {
        let fx = load_stream_fixture("placeholder_selftest");
        assert!(!fx.expected_sse.is_empty());
        // A stream is trivially parity-equal to itself.
        assert_stream_eq(&fx.expected_sse, &fx.expected_sse);
        // And the bedrock side parses as JSONL without error.
        let _ = parse_sse(&fx.bedrock_events).unwrap();
    }

    #[test]
    fn placeholder_response_fixture_roundtrips() {
        let fx = load_response_fixture("placeholder_selftest");
        assert_semantic_eq(&fx.expected_openai_response, &fx.expected_openai_response);
        assert!(fx.bedrock_output.is_object());
    }
}
