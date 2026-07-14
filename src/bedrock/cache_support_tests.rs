//! Tests for [`crate::bedrock::cache_support`] — negative-cache registry, the
//! shared cache-unsupported error predicate, and the self-healing retry wrapper.
//!
//! Test code lives in this sibling file (Option A: no inline `#[cfg(test)]`
//! blocks in the source module). `use super::*;` resolves to the
//! `cache_support` implementation module via the
//! `#[path = "cache_support_tests.rs"] mod tests;` declaration there.

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
