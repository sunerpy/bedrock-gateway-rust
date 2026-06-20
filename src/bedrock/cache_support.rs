//! Negative cache + shared error predicate for prompt-caching support.
//!
//! Some Bedrock foundation models reject `cachePoint` blocks at `send()` time
//! rather than silently ignoring them (confirmed live: `AccessDeniedException`
//! HTTP 403 with the message *"You invoked an unsupported model or your request
//! did not allow prompt caching."*). This module provides the foundation for a
//! self-healing safety net (consumed by a later step): a process-local
//! **negative cache** of foundation ids that proved unsupported, plus the single
//! shared predicate that classifies whether a raw Bedrock error is a
//! caching-unsupported rejection (as opposed to a genuine authorization failure,
//! throttling, or a 5xx).
//!
//! ## Why a `Mutex<HashSet>` (not `DashMap`)
//!
//! The key space is bounded by the number of distinct foundation ids the gateway
//! ever sends — small and finite — so no TTL or eviction is needed. Lookups are
//! brief and rare on the hot path, so a `std::sync::Mutex<HashSet<String>>` is
//! sufficient and avoids pulling in a new concurrency dependency. The lock is
//! held only for the duration of a single `contains` / `insert`.
//!
//! ## Poison tolerance
//!
//! A poisoned negative-cache mutex must never crash a request. All lock
//! acquisitions use the poison-recovery pattern
//! `lock().unwrap_or_else(PoisonError::into_inner)` so a panic in some unrelated
//! holder degrades to "treat the set as usable" rather than propagating.

use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};

use aws_smithy_types::error::metadata::ProvideErrorMetadata;

use crate::error::{from_bedrock_sdk_error, AppError};

/// Process-local negative cache of foundation ids that Bedrock has rejected for
/// prompt caching.
///
/// The registry stores already-normalized foundation id strings verbatim — it
/// does no normalization itself (callers pass the normalized id). It is shared
/// (behind an `Arc`) across both the chat and Responses providers so a rejection
/// observed on one surface suppresses cache-point injection on the other.
#[derive(Debug, Default)]
pub struct CacheSupportRegistry {
    /// Foundation ids known to reject `cachePoint`. Bounded by model count, so
    /// no eviction policy is required.
    unsupported: Mutex<HashSet<String>>,
}

impl CacheSupportRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            unsupported: Mutex::new(HashSet::new()),
        }
    }

    /// Return `true` if `normalized_model` has previously been marked as not
    /// supporting prompt caching.
    ///
    /// Poison-tolerant: a poisoned lock recovers the inner set rather than
    /// panicking on the request path.
    #[must_use]
    pub fn is_unsupported(&self, normalized_model: &str) -> bool {
        let guard = self
            .unsupported
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        guard.contains(normalized_model)
    }

    /// Record `normalized_model` as not supporting prompt caching.
    ///
    /// Idempotent. Poison-tolerant (see [`Self::is_unsupported`]).
    pub fn mark_unsupported(&self, normalized_model: &str) {
        let mut guard = self
            .unsupported
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        guard.insert(normalized_model.to_string());
    }
}

/// Classify whether a raw Bedrock error is a *prompt-caching-unsupported*
/// rejection.
///
/// This is the **single shared gate**: it is the only place that decides whether
/// a Bedrock error means "this model does not support `cachePoint`" (so the
/// safety net should strip cache points and retry, and mark the model in the
/// negative cache). It must be called on the RAW service error — i.e. the
/// `into_service_error()` output that still carries `.code()`/`.message()` via
/// `ProvideErrorMetadata` — BEFORE [`crate::error::from_bedrock_sdk_error`] maps
/// `AccessDeniedException` to the message-discarding `Unauthorized` unit variant.
///
/// ## Contract (refined by live findings)
///
/// Returns `true` only when BOTH a caching-rejection error code AND a
/// cache-vocabulary message substring are present — never on the code alone,
/// because `AccessDeniedException` is overloaded (it also covers genuine IAM
/// denials, which must NOT be misclassified):
///
/// - `AccessDeniedException` AND the lowercased message contains
///   `"prompt caching"` — the live, authoritative form (anchor substring of
///   *"did not allow prompt caching"*).
/// - `ValidationException` AND the lowercased message contains any of
///   `"cachepoint"`, `"prompt cache"`, `"prompt caching"`, or both
///   `"does not support"` and `"cache"` — defensive coverage for other Bedrock
///   variants/regions (e.g. an *"extraneous key [cachePoint] not permitted"*
///   wording).
///
/// Returns `false` for everything else, including:
/// - `ThrottlingException` (any message),
/// - `AccessDeniedException` carrying an IAM message with no cache vocabulary
///   (e.g. *"not authorized to perform bedrock:InvokeModel"*),
/// - `ServiceUnavailableException` / other 5xx.
#[must_use]
pub fn is_cache_unsupported_error(code: &str, msg: &str) -> bool {
    let lower = msg.to_lowercase();
    match code {
        // Live-confirmed form: unsupported model rejects prompt caching with a
        // 403 AccessDeniedException. REQUIRE the cache-vocabulary substring so a
        // real credential/authorization denial is not misclassified.
        "AccessDeniedException" => lower.contains("prompt caching"),
        // Defensive coverage for other Bedrock variants/regions that surface the
        // rejection as a ValidationException.
        "ValidationException" => {
            lower.contains("cachepoint")
                || lower.contains("prompt cache")
                || lower.contains("prompt caching")
                || (lower.contains("does not support") && lower.contains("cache"))
        }
        _ => false,
    }
}

/// Failure of a single Converse / ConverseStream attempt, distinguishing a raw
/// Bedrock *service error* (whose `.code()`/`.message()` the cache predicate must
/// inspect) from a gateway-side [`AppError`] raised before the wire call (e.g.
/// re-assembly or JSON→SDK build failure on the strip path).
pub(crate) enum SendError<E> {
    /// Raw Bedrock service error (`into_service_error()` output) — still carries
    /// metadata for [`is_cache_unsupported_error`].
    Service(E),
    /// A gateway-side error produced before/around the send (never a cache
    /// rejection); propagated unchanged.
    App(AppError),
}

/// Run a Converse / ConverseStream `send()` with the prompt-cache self-healing
/// safety net, shared by all four send points (chat + Responses × stream +
/// non-stream).
///
/// Each closure yields a [`SendError`] on failure: `Service(e)` carries the raw
/// service error (the caller maps the `SdkError` with `.into_service_error()` so
/// `.code()`/`.message()` survive — [`crate::error::from_bedrock_sdk_error`]
/// collapses `AccessDeniedException` into a message-discarding `Unauthorized`, so
/// the predicate MUST see the raw service error first); `App(e)` is a
/// gateway-side failure that propagates unchanged.
///
/// On the first error, retry is taken EXACTLY ONCE and ONLY when the failure is
/// a raw service error AND both `cache_points_injected` and
/// [`is_cache_unsupported_error`] hold (the false-positive guard: a
/// cache-vocabulary error on a request that injected no cachePoints propagates
/// unchanged). The retry marks the negative cache, WARNs, and runs `retry_send`
/// (built by the caller with caching forced OFF). Any other error maps and
/// propagates the ORIGINAL error. No backoff, no second retry.
pub(crate) async fn send_with_cache_strip_retry<O, E, Fut1, Fut2>(
    cache_support: &Arc<CacheSupportRegistry>,
    normalized_model: &str,
    cache_points_injected: bool,
    first_send: impl FnOnce() -> Fut1,
    retry_send: impl FnOnce() -> Fut2,
) -> Result<O, AppError>
where
    E: ProvideErrorMetadata + std::fmt::Display,
    Fut1: Future<Output = Result<O, SendError<E>>>,
    Fut2: Future<Output = Result<O, SendError<E>>>,
{
    match first_send().await {
        Ok(output) => Ok(output),
        Err(SendError::App(app)) => Err(app),
        Err(SendError::Service(err)) => {
            let code = err.code().unwrap_or_default();
            let msg = err.message().unwrap_or_default();
            if cache_points_injected && is_cache_unsupported_error(code, msg) {
                cache_support.mark_unsupported(normalized_model);
                tracing::warn!(
                    model = %normalized_model,
                    "prompt caching unsupported by model, stripped and retried"
                );
                match retry_send().await {
                    Ok(output) => Ok(output),
                    Err(SendError::App(app)) => Err(app),
                    Err(SendError::Service(retry_err)) => Err(from_bedrock_sdk_error(&retry_err)),
                }
            } else {
                Err(from_bedrock_sdk_error(&err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Registry mark / is_unsupported ------------------------------------

    #[test]
    fn registry_starts_empty_then_marks_and_reports() {
        let reg = CacheSupportRegistry::new();
        assert!(!reg.is_unsupported("anthropic.claude-3-5-sonnet"));
        reg.mark_unsupported("anthropic.claude-3-5-sonnet");
        assert!(reg.is_unsupported("anthropic.claude-3-5-sonnet"));
        // A different id is unaffected.
        assert!(!reg.is_unsupported("amazon.nova-lite"));
    }

    #[test]
    fn registry_mark_is_idempotent() {
        let reg = CacheSupportRegistry::new();
        reg.mark_unsupported("qwen.qwen3-235b");
        reg.mark_unsupported("qwen.qwen3-235b");
        assert!(reg.is_unsupported("qwen.qwen3-235b"));
    }

    // --- Predicate: positive cases -----------------------------------------

    #[test]
    fn predicate_true_for_access_denied_with_prompt_caching_message() {
        // The live, authoritative wording.
        let msg = "You invoked an unsupported model or your request did not allow \
                   prompt caching. See the documentation for more information.";
        assert!(is_cache_unsupported_error("AccessDeniedException", msg));
    }

    #[test]
    fn predicate_true_for_access_denied_message_case_insensitive() {
        assert!(is_cache_unsupported_error(
            "AccessDeniedException",
            "Request did not allow PROMPT CACHING."
        ));
    }

    #[test]
    fn predicate_true_for_validation_exception_cachepoint_wording() {
        assert!(is_cache_unsupported_error(
            "ValidationException",
            "extraneous key [cachePoint] not permitted"
        ));
    }

    #[test]
    fn predicate_true_for_validation_exception_does_not_support_cache() {
        assert!(is_cache_unsupported_error(
            "ValidationException",
            "This model does not support prompt cache checkpoints."
        ));
    }

    #[test]
    fn predicate_true_for_validation_exception_does_not_support_plus_cache() {
        assert!(is_cache_unsupported_error(
            "ValidationException",
            "The selected model does not support the requested cache feature."
        ));
    }

    // --- Predicate: false-positive guards ----------------------------------

    #[test]
    fn predicate_false_for_access_denied_iam_message() {
        // A genuine authorization failure with no cache vocabulary must NOT be
        // misclassified as a caching-unsupported rejection.
        let msg = "User: arn:aws:iam::123:user/x is not authorized to perform: \
                   bedrock:InvokeModel on resource: ...";
        assert!(!is_cache_unsupported_error("AccessDeniedException", msg));
    }

    #[test]
    fn predicate_false_for_throttling_any_message() {
        assert!(!is_cache_unsupported_error(
            "ThrottlingException",
            "Too many requests, please slow down."
        ));
        // Even if the throttling message somehow mentioned caching, the code
        // gate rejects it (only AccessDenied/Validation codes are considered).
        assert!(!is_cache_unsupported_error(
            "ThrottlingException",
            "prompt caching limit exceeded"
        ));
    }

    #[test]
    fn predicate_false_for_service_unavailable() {
        assert!(!is_cache_unsupported_error(
            "ServiceUnavailableException",
            "The service is temporarily unavailable."
        ));
    }

    #[test]
    fn predicate_false_for_validation_exception_unrelated_message() {
        // A normal validation error (no cache vocabulary) is not a cache
        // rejection.
        assert!(!is_cache_unsupported_error(
            "ValidationException",
            "messages.0.content: at least one content block is required"
        ));
    }

    #[test]
    fn predicate_false_for_empty_code_and_message() {
        assert!(!is_cache_unsupported_error("", ""));
    }

    // --- send_with_cache_strip_retry --------------------------------------

    use aws_smithy_types::error::metadata::ErrorMetadata;
    use std::cell::Cell;

    type TestSend = Result<i32, SendError<ErrorMetadata>>;

    fn svc_err(code: &str, message: &str) -> SendError<ErrorMetadata> {
        SendError::Service(ErrorMetadata::builder().code(code).message(message).build())
    }

    const CACHE_REJECT_MSG: &str =
        "You invoked an unsupported model or your request did not allow prompt caching.";

    #[tokio::test]
    async fn cache_error_with_injection_strips_retries_once_and_memoizes() {
        let reg = Arc::new(CacheSupportRegistry::new());
        let retried = Cell::new(false);

        let out: Result<i32, AppError> = send_with_cache_strip_retry(
            &reg,
            "qwen.qwen3-235b",
            true,
            || async { TestSend::Err(svc_err("AccessDeniedException", CACHE_REJECT_MSG)) },
            || async {
                retried.set(true);
                TestSend::Ok(7)
            },
        )
        .await;

        assert_eq!(out.expect("retry succeeds"), 7);
        assert!(retried.get(), "retry_send must run exactly once");
        assert!(
            reg.is_unsupported("qwen.qwen3-235b"),
            "model must be memoized as unsupported"
        );
    }

    #[tokio::test]
    async fn non_cache_error_propagates_without_retry_or_memoize() {
        let reg = Arc::new(CacheSupportRegistry::new());
        let retried = Cell::new(false);

        let out: Result<i32, AppError> = send_with_cache_strip_retry(
            &reg,
            "anthropic.claude-sonnet-4-5",
            true,
            || async { TestSend::Err(svc_err("ThrottlingException", "Too many requests")) },
            || async {
                retried.set(true);
                TestSend::Ok(7)
            },
        )
        .await;

        assert!(matches!(out, Err(AppError::Throttled(_))));
        assert!(!retried.get(), "non-cache error must NOT retry");
        assert!(
            !reg.is_unsupported("anthropic.claude-sonnet-4-5"),
            "non-cache error must NOT memoize"
        );
    }

    #[tokio::test]
    async fn cache_error_without_injection_propagates_no_memoize() {
        // False-positive guard: a cache-vocabulary error on a request that
        // injected NO cachePoints must propagate unchanged and never memoize.
        let reg = Arc::new(CacheSupportRegistry::new());
        let retried = Cell::new(false);

        let out: Result<i32, AppError> = send_with_cache_strip_retry(
            &reg,
            "qwen.qwen3-235b",
            false,
            || async { TestSend::Err(svc_err("AccessDeniedException", CACHE_REJECT_MSG)) },
            || async {
                retried.set(true);
                TestSend::Ok(7)
            },
        )
        .await;

        assert!(matches!(out, Err(AppError::Unauthorized)));
        assert!(!retried.get(), "no injection ⇒ no retry");
        assert!(
            !reg.is_unsupported("qwen.qwen3-235b"),
            "no injection ⇒ no memoize (false-positive guard)"
        );
    }

    #[tokio::test]
    async fn first_attempt_success_skips_retry() {
        let reg = Arc::new(CacheSupportRegistry::new());
        let retried = Cell::new(false);

        let out: Result<i32, AppError> = send_with_cache_strip_retry(
            &reg,
            "amazon.nova-lite",
            true,
            || async { TestSend::Ok(42) },
            || async {
                retried.set(true);
                TestSend::Ok(0)
            },
        )
        .await;

        assert_eq!(out.expect("first send ok"), 42);
        assert!(!retried.get(), "success path must not retry");
        assert!(!reg.is_unsupported("amazon.nova-lite"));
    }

    #[tokio::test]
    async fn read_gate_skips_injection_on_second_request() {
        // Read-gate contract at the registry level: once a model is memoized,
        // a caller computes cache_points_injected=false (caching forced off),
        // so even a cache-vocabulary error would not re-strip/re-mark — the
        // request just propagates.
        let reg = Arc::new(CacheSupportRegistry::new());
        reg.mark_unsupported("qwen.qwen3-235b");
        assert!(reg.is_unsupported("qwen.qwen3-235b"));

        let retried = Cell::new(false);
        let out: Result<i32, AppError> = send_with_cache_strip_retry(
            &reg,
            "qwen.qwen3-235b",
            false,
            || async { TestSend::Err(svc_err("AccessDeniedException", CACHE_REJECT_MSG)) },
            || async {
                retried.set(true);
                TestSend::Ok(7)
            },
        )
        .await;

        assert!(matches!(out, Err(AppError::Unauthorized)));
        assert!(!retried.get(), "memoized model never re-injects ⇒ no retry");
    }
}
