//! Reasoning token estimation for OpenAI API compatibility.
//!
//! This module provides token counting for reasoning content blocks.
//! Uses tiktoken cl100k_base encoding (OpenAI-compatible).
//!
//! ⚠️ IMPORTANT: tiktoken is ONLY for reasoning_tokens (OpenAI field) estimates.
//! Do NOT use for prompt/completion accounting—Bedrock and Claude use their own tokenizers.
//! Mixing leads to mismatched billing and incorrect usage metrics.

use tiktoken_rs::cl100k_base_singleton;

/// OpenAI-shaped token usage counts rebuilt from Bedrock's per-component
/// counts.
///
/// This is the single source of truth for prompt/completion/total/cached token
/// accounting, shared by the non-streaming response mapper
/// ([`crate::bedrock::response::from_converse_output`]) and the streaming usage
/// chunk ([`crate::bedrock::stream::StreamState::map_event`]).
///
/// `reasoning_tokens` is intentionally NOT part of this struct: it is a separate
/// tiktoken estimate (see [`estimate_reasoning_tokens`]), not derived from
/// Bedrock's reported counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsageCounts {
    /// OpenAI `prompt_tokens`. INCLUDES cached (read + write) tokens —
    /// `input + cacheRead + cacheWrite`. OpenAI semantics: prompt_tokens is the
    /// full input billed, cached or not.
    pub prompt_tokens: i32,
    /// OpenAI `completion_tokens` = Bedrock `outputTokens`.
    pub completion_tokens: i32,
    /// OpenAI `total_tokens` = `input + cacheRead + cacheWrite + output`.
    pub total_tokens: i32,
    /// OpenAI `prompt_tokens_details.cached_tokens` = `cacheRead` ONLY (read
    /// side). `cacheWrite` is acknowledged in the prompt/total math but never
    /// surfaced as its own wire field (no standard OpenAI field exists).
    pub cached_tokens: i32,
}

/// Rebuild OpenAI token usage from Bedrock's four per-component counts.
///
/// Pure function (no I/O). This is the **rebuild-from-parts** formula, which is
/// explicit and robust regardless of whether a given Bedrock response folds
/// cache tokens into `totalTokens`:
///
/// - `prompt_tokens   = input + cacheRead + cacheWrite`
/// - `completion_tokens = output`
/// - `total_tokens    = input + cacheRead + cacheWrite + output`
/// - `cached_tokens   = cacheRead` (read side only)
///
/// Verified live against Bedrock: `totalTokens == input + cacheRead +
/// cacheWrite + output`, where `input` is NON-cached input only, so this
/// rebuild agrees numerically with the older `totalTokens - output` shortcut on
/// real data — but it does not depend on `totalTokens` at all.
///
/// # Example
/// ```
/// use bedrock_gateway_rust::bedrock::tokens::compute_token_usage;
///
/// let u = compute_token_usage(100, 20, 500, 300);
/// assert_eq!(u.prompt_tokens, 900); // 100 + 500 + 300
/// assert_eq!(u.completion_tokens, 20);
/// assert_eq!(u.total_tokens, 920); // 900 + 20
/// assert_eq!(u.cached_tokens, 500); // cacheRead only
///
/// let no_cache = compute_token_usage(100, 20, 0, 0);
/// assert_eq!(no_cache.prompt_tokens, 100);
/// assert_eq!(no_cache.total_tokens, 120);
/// assert_eq!(no_cache.cached_tokens, 0);
/// ```
#[must_use]
pub fn compute_token_usage(
    input_tokens: i32,
    output_tokens: i32,
    cache_read_tokens: i32,
    cache_write_tokens: i32,
) -> TokenUsageCounts {
    let prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens;
    TokenUsageCounts {
        prompt_tokens,
        completion_tokens: output_tokens,
        total_tokens: prompt_tokens + output_tokens,
        cached_tokens: cache_read_tokens,
    }
}

/// Estimate reasoning tokens from text using OpenAI's cl100k_base encoding.
///
/// Returns token count as u32 for OpenAI `reasoning_tokens` field compatibility.
/// Empty or whitespace-only text returns 0.
///
/// # Encoding
/// Uses `tiktoken_rs::cl100k_base_singleton()` (lazy, preloaded at first use).
/// Matches Python tiktoken behavior for OpenAI model token counting.
///
/// # Example
/// ```
/// use bedrock_gateway_rust::bedrock::tokens::estimate_reasoning_tokens;
///
/// let tokens = estimate_reasoning_tokens("What is 2+2?");
/// assert!(tokens > 0);
///
/// let empty = estimate_reasoning_tokens("");
/// assert_eq!(empty, 0);
/// ```
pub fn estimate_reasoning_tokens(text: &str) -> u32 {
    // Trim whitespace; empty/whitespace-only text always returns 0.
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0;
    }

    // Get cl100k_base singleton (lazy init; preloaded at first call).
    let encoder = cl100k_base_singleton();

    // Encode with special tokens (matches Python tiktoken.encode(text)).
    // Returns token IDs; len() gives token count.
    encoder.encode_with_special_tokens(trimmed).len() as u32
}

#[cfg(test)]
#[path = "tokens_tests.rs"]
mod tests;
