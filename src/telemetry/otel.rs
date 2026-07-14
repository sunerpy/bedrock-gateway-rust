//! OTLP traces + metrics, compiled ONLY under the `otel` Cargo feature.
//!
//! This module owns three concerns, kept deliberately separate so the
//! privacy- and cardinality-sensitive pieces are unit-testable without a live
//! collector:
//!
//! 1. **Provider init** ([`init_otel`]) — build an OTLP tracer + meter provider
//!    (HTTP/protobuf transport) when an endpoint is configured, register the
//!    meter globally, and hand back an [`OpenTelemetryLayer`] to be composed
//!    into the existing `tracing_subscriber::Registry`.
//! 2. **Redaction** ([`SpanContentPolicy`] / [`request_span_attrs`]) — the
//!    span-attribute builder. By default it attaches ONLY metadata; message
//!    content is attached exclusively when content capture is opted in. This is
//!    the single place the logging-privacy contract is relaxed.
//! 3. **Metric labels** ([`RequestMetricLabels`]) — the label set for a request
//!    metric is EXACTLY `{model, finish_reason}`. `request_id` is used for
//!    logs/spans but is NEVER a metric label (cardinality guard).

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::TracerProvider;
use opentelemetry_sdk::Resource;

use crate::telemetry::{SERVICE_NAME, SERVICE_VERSION};

/// The concrete Bedrock-gateway OTLP tracer type.
pub type GatewayTracer = opentelemetry_sdk::trace::Tracer;

/// Metric-name constants. These are OTLP instrument names (protocol strings,
/// not model knowledge), analogous to the allowed SSE/OpenAI constants.
const METRIC_REQUEST_COUNT: &str = "gateway.requests";
const METRIC_REQUEST_DURATION: &str = "gateway.request.duration";
const METRIC_PROMPT_TOKENS: &str = "gateway.tokens.prompt";
const METRIC_COMPLETION_TOKENS: &str = "gateway.tokens.completion";

/// Label KEYS for request metrics — the ONLY two dimensions permitted.
const LABEL_MODEL: &str = "model";
const LABEL_FINISH_REASON: &str = "finish_reason";

/// Whether span attributes may carry prompt/completion text.
///
/// `Redacted` (the default) permits metadata/counts only. `CaptureContent` is
/// reached exclusively via the explicit `OTEL_CAPTURE_CONTENT=true` opt-in and
/// is the single relaxation of the logging-privacy contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanContentPolicy {
    Redacted,
    CaptureContent,
}

impl SpanContentPolicy {
    /// Derive the policy from the opt-in flag. `capture == false` ⇒ `Redacted`.
    pub fn from_capture_flag(capture: bool) -> Self {
        if capture {
            Self::CaptureContent
        } else {
            Self::Redacted
        }
    }

    fn captures_content(self) -> bool {
        matches!(self, Self::CaptureContent)
    }
}

/// Immutable, cardinality-safe label set for a per-request metric.
///
/// Constructed only from `{model, finish_reason}`. There is intentionally no
/// constructor arm that accepts a `request_id`, so a high-cardinality label
/// cannot be attached by accident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestMetricLabels {
    model: String,
    finish_reason: Option<String>,
}

impl RequestMetricLabels {
    pub fn new(model: impl Into<String>, finish_reason: Option<String>) -> Self {
        Self {
            model: model.into(),
            finish_reason,
        }
    }

    /// Render the OpenTelemetry `KeyValue` label array. Exactly `model` plus, if
    /// known, `finish_reason` — never a request id.
    pub fn key_values(&self) -> Vec<KeyValue> {
        let mut kvs = vec![KeyValue::new(LABEL_MODEL, self.model.clone())];
        if let Some(reason) = &self.finish_reason {
            kvs.push(KeyValue::new(LABEL_FINISH_REASON, reason.clone()));
        }
        kvs
    }

    /// The label KEY set, for verification/testing.
    pub fn label_keys(&self) -> Vec<&'static str> {
        let mut keys = vec![LABEL_MODEL];
        if self.finish_reason.is_some() {
            keys.push(LABEL_FINISH_REASON);
        }
        keys
    }
}

/// A single span attribute: a static key and an owned string value.
pub type SpanAttr = (&'static str, String);

/// Attribute keys reserved for opted-in content capture. Used by the redaction
/// test to assert these never appear by default.
pub const ATTR_PROMPT: &str = "gen_ai.prompt";
pub const ATTR_COMPLETION: &str = "gen_ai.completion";

/// Build the span attributes for a request under the given content policy.
///
/// `metadata` (model, streaming flag, token counts, latency, finish_reason) is
/// ALWAYS included. `content` (prompt/completion text) is included ONLY when the
/// policy is [`SpanContentPolicy::CaptureContent`]. Under `Redacted` the result
/// provably contains no [`ATTR_PROMPT`]/[`ATTR_COMPLETION`] key.
pub fn request_span_attrs(
    policy: SpanContentPolicy,
    metadata: Vec<SpanAttr>,
    content: impl FnOnce() -> Vec<SpanAttr>,
) -> Vec<SpanAttr> {
    let mut attrs = metadata;
    if policy.captures_content() {
        attrs.extend(content());
    }
    attrs
}

/// Initialized OTLP providers plus the tracer used to build the tracing layer.
/// The providers are held by the caller so their background export pipelines
/// stay alive for the process lifetime.
pub struct OtelInit {
    pub tracer: GatewayTracer,
    pub tracer_provider: TracerProvider,
    pub meter_provider: SdkMeterProvider,
}

fn resource() -> Resource {
    Resource::new(vec![
        KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_NAME,
            SERVICE_NAME,
        ),
        KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_VERSION,
            SERVICE_VERSION,
        ),
    ])
}

/// Build OTLP tracer + meter providers pointed at `endpoint`. The meter provider
/// is registered globally so [`record_request_metrics`] can create instruments
/// anywhere. Returns the tracer so the caller can build a generic
/// `OpenTelemetryLayer`. `Err` if either exporter fails to build.
pub fn init_otel(endpoint: &str) -> anyhow::Result<OtelInit> {
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .build()?;
    let tracer_provider = TracerProvider::builder()
        .with_resource(resource())
        .with_batch_exporter(span_exporter, runtime::Tokio)
        .build();

    let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .build()?;
    let reader = PeriodicReader::builder(metric_exporter, runtime::Tokio).build();
    let meter_provider = SdkMeterProvider::builder()
        .with_resource(resource())
        .with_reader(reader)
        .build();

    global::set_meter_provider(meter_provider.clone());

    let tracer = tracer_provider.tracer(SERVICE_NAME);

    Ok(OtelInit {
        tracer,
        tracer_provider,
        meter_provider,
    })
}

/// Record a completed request into the OTLP metrics, labeled ONLY by
/// `{model, finish_reason}`. A no-op-safe helper: instruments are created from
/// the global meter provider, so if `init_otel` was never called (no endpoint)
/// the global no-op meter swallows the records.
pub fn record_request_metrics(
    labels: &RequestMetricLabels,
    duration_ms: u64,
    prompt_tokens: i32,
    completion_tokens: i32,
) {
    let meter = global::meter(SERVICE_NAME);
    let kvs = labels.key_values();

    meter.u64_counter(METRIC_REQUEST_COUNT).build().add(1, &kvs);
    meter
        .u64_histogram(METRIC_REQUEST_DURATION)
        .with_unit("ms")
        .build()
        .record(duration_ms, &kvs);
    meter
        .u64_histogram(METRIC_PROMPT_TOKENS)
        .build()
        .record(prompt_tokens.max(0) as u64, &kvs);
    meter
        .u64_histogram(METRIC_COMPLETION_TOKENS)
        .build()
        .record(completion_tokens.max(0) as u64, &kvs);
}

#[cfg(test)]
#[path = "otel_tests.rs"]
mod tests;
