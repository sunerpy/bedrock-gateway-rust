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
#[path = "cache_support_tests.rs"]
mod tests;
