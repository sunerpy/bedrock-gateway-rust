//! Unit tests for the OTLP (`otel`) telemetry module.
//!
//! Relocated out of `otel.rs` for code organization (see the
//! `test-coverage-codecov` spec). Behavior is unchanged; `use super::*`
//! continues to resolve to the parent `otel` module since this file is
//! declared as its child `mod tests;`. The whole `otel` module (and hence
//! this file) is compiled only under the `otel` Cargo feature.

use super::*;

#[test]
fn span_attrs_redacted_by_default_carries_no_content() {
    let metadata = vec![
        ("model", "some-model".to_string()),
        ("stream", "false".to_string()),
        ("prompt_tokens", "12".to_string()),
    ];
    let attrs = request_span_attrs(SpanContentPolicy::Redacted, metadata, || {
        vec![
            (ATTR_PROMPT, "secret prompt text".to_string()),
            (ATTR_COMPLETION, "secret completion text".to_string()),
        ]
    });

    let keys: Vec<&str> = attrs.iter().map(|(k, _)| *k).collect();
    assert!(keys.contains(&"model"));
    assert!(keys.contains(&"prompt_tokens"));
    assert!(
        !keys.contains(&ATTR_PROMPT),
        "prompt content must be absent by default"
    );
    assert!(
        !keys.contains(&ATTR_COMPLETION),
        "completion content must be absent by default"
    );
}

#[test]
fn span_attrs_capture_content_includes_content_only_when_opted_in() {
    let metadata = vec![("model", "some-model".to_string())];
    let attrs = request_span_attrs(SpanContentPolicy::CaptureContent, metadata, || {
        vec![(ATTR_PROMPT, "prompt text".to_string())]
    });
    let keys: Vec<&str> = attrs.iter().map(|(k, _)| *k).collect();
    assert!(keys.contains(&ATTR_PROMPT), "opt-in must include content");
}

#[test]
fn policy_from_capture_flag_defaults_redacted() {
    assert_eq!(
        SpanContentPolicy::from_capture_flag(false),
        SpanContentPolicy::Redacted
    );
    assert_eq!(
        SpanContentPolicy::from_capture_flag(true),
        SpanContentPolicy::CaptureContent
    );
}

#[test]
fn request_metric_labels_are_exactly_model_and_finish_reason() {
    let labels = RequestMetricLabels::new("anthropic.claude", Some("stop".to_string()));
    let keys = labels.label_keys();
    assert_eq!(keys, vec![LABEL_MODEL, LABEL_FINISH_REASON]);
    assert!(
        !keys.contains(&"request_id"),
        "request_id must NEVER be a metric label (cardinality)"
    );

    let kvs = labels.key_values();
    let kv_keys: Vec<&str> = kvs.iter().map(|kv| kv.key.as_str()).collect();
    assert_eq!(kv_keys, vec![LABEL_MODEL, LABEL_FINISH_REASON]);
    assert!(!kv_keys.iter().any(|k| k.contains("request")));
}

#[test]
fn request_metric_labels_omit_finish_reason_when_absent() {
    let labels = RequestMetricLabels::new("m", None);
    assert_eq!(labels.label_keys(), vec![LABEL_MODEL]);
    assert_eq!(labels.key_values().len(), 1);
}

/// `record_request_metrics` is safe to call even when no endpoint was ever
/// configured: the global no-op meter swallows the records. Exercises the
/// counter plus all three histogram instrument paths, including the
/// negative-count clamp (`prompt_tokens.max(0)`).
#[test]
fn record_request_metrics_no_panic_with_noop_meter() {
    let labels = RequestMetricLabels::new("m", None);
    record_request_metrics(&labels, 0, -5, -1); // negatives clamp to 0
    record_request_metrics(&labels, 123, 456, 789);
}

/// The `resource()` builder is a pure, offline helper: it assembles the OTLP
/// `Resource` from the crate's service-name/version constants with no network
/// or provider init. Covers the builder body without touching `init_otel`.
#[test]
fn resource_builder_carries_service_identity() {
    let r = resource();
    // At minimum the service name + version attributes must be present.
    assert!(
        r.len() >= 2,
        "resource must include service name and version attributes"
    );
    let has_service_name = r
        .iter()
        .any(|(k, _)| k.as_str() == opentelemetry_semantic_conventions::resource::SERVICE_NAME);
    let has_service_version = r
        .iter()
        .any(|(k, _)| k.as_str() == opentelemetry_semantic_conventions::resource::SERVICE_VERSION);
    assert!(has_service_name, "service.name attribute must be present");
    assert!(
        has_service_version,
        "service.version attribute must be present"
    );
}

/// `RequestMetricLabels` derives (Clone / PartialEq / Debug) plus the
/// `finish_reason = Some(..)` `key_values` arm carrying the reason value.
#[test]
fn request_metric_labels_clone_eq_debug_and_values() {
    let labels = RequestMetricLabels::new("m", Some("length".to_string()));
    let cloned = labels.clone();
    assert_eq!(labels, cloned);

    let other = RequestMetricLabels::new("m", None);
    assert_ne!(labels, other);

    // Debug formatting must name the type and not panic.
    assert!(format!("{labels:?}").contains("RequestMetricLabels"));

    // The Some(finish_reason) arm renders the reason value into the KeyValue set.
    let kvs = labels.key_values();
    assert_eq!(kvs.len(), 2);
    let reason_value = kvs
        .iter()
        .find(|kv| kv.key.as_str() == LABEL_FINISH_REASON)
        .map(|kv| kv.value.as_str().into_owned());
    assert_eq!(reason_value.as_deref(), Some("length"));
}

/// `SpanContentPolicy` derives (Clone / Copy / Debug) exercised alongside the
/// `Redacted` short-circuit: under `Redacted` the content closure is NEVER
/// invoked (proven via a side-effecting flag), so no content work is done.
#[test]
fn span_content_policy_derives_and_redacted_skips_closure() {
    let policy = SpanContentPolicy::Redacted;
    let copied = policy; // Copy
    assert_eq!(policy, copied);
    assert!(format!("{policy:?}").contains("Redacted"));

    let mut closure_called = false;
    let attrs = request_span_attrs(policy, vec![("model", "m".to_string())], || {
        closure_called = true;
        vec![(ATTR_PROMPT, "text".to_string())]
    });
    assert!(
        !closure_called,
        "Redacted policy must not invoke the content closure"
    );
    assert_eq!(attrs.len(), 1);
}
