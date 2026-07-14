//! Unit and property-based tests for the token-accounting helpers.
//!
//! Relocated out of `tokens.rs` for code organization (see the
//! `test-coverage-codecov` spec). Behavior is unchanged; the two original
//! test modules (`tests` and `prop_tests`) are preserved verbatim as nested
//! submodules so the source file references exactly one `mod tests;`.
//!
//! Because the modules are now nested one level deeper than before, their
//! `use super::*;` becomes `use super::super::*;` to keep resolving to the
//! implementation `tokens` module.

mod unit_tests {
    use super::super::*;

    #[test]
    fn test_empty_text() {
        assert_eq!(estimate_reasoning_tokens(""), 0);
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(estimate_reasoning_tokens("   \t\n  "), 0);
    }

    #[test]
    fn test_simple_text() {
        // Known sample: "What is 2+2?" should yield stable nonzero token count.
        let tokens = estimate_reasoning_tokens("What is 2+2?");
        assert!(tokens > 0, "Simple text should produce nonzero tokens");
        // cl100k_base typically tokenizes this to ~4-5 tokens.
        assert!(
            tokens <= 10,
            "Simple text should not produce excessive tokens"
        );
    }

    #[test]
    fn test_longer_text() {
        let text = "Reasoning about this complex problem requires careful analysis of all edge cases and potential failure modes in the system.";
        let tokens = estimate_reasoning_tokens(text);
        assert!(tokens > 0, "Longer text should produce tokens");
        // Should be roughly 20+ tokens.
        assert!(
            tokens >= 15,
            "Longer text should produce meaningful token count"
        );
    }

    #[test]
    fn test_consistency() {
        // Same text should always produce same token count.
        let text = "The quick brown fox jumps over the lazy dog";
        let tokens1 = estimate_reasoning_tokens(text);
        let tokens2 = estimate_reasoning_tokens(text);
        assert_eq!(tokens1, tokens2, "Token count should be deterministic");
    }

    #[test]
    fn compute_token_usage_rebuilds_from_parts_with_cache() {
        let u = compute_token_usage(100, 20, 500, 300);
        assert_eq!(u.prompt_tokens, 900);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 920);
        assert_eq!(u.cached_tokens, 500);
    }

    #[test]
    fn compute_token_usage_no_cache_equals_input() {
        let u = compute_token_usage(100, 20, 0, 0);
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 120);
        assert_eq!(u.cached_tokens, 0);
    }

    #[test]
    fn compute_token_usage_cache_write_only_folds_into_prompt() {
        let u = compute_token_usage(10, 5, 0, 40);
        assert_eq!(u.prompt_tokens, 50);
        assert_eq!(u.total_tokens, 55);
        assert_eq!(u.cached_tokens, 0);
    }

    #[test]
    fn compute_token_usage_prompt_never_negative_without_total() {
        let u = compute_token_usage(7, 16, 0, 0);
        assert_eq!(u.prompt_tokens, 7);
        assert!(u.prompt_tokens >= 0);
        assert_eq!(u.total_tokens, 23);
    }
}

/// Property-based tests for the token-accounting invariants.
///
/// Feature: test-coverage-codecov, Property 2: Token 计账不变量
/// (see `.kiro/specs/test-coverage-codecov/design.md`).
///
/// Validates: Requirements 1.2
mod prop_tests {
    use super::super::*;
    use proptest::prelude::*;

    // Bound each component so that the four-way sum (input + cacheRead +
    // cacheWrite + output) can never overflow i32::MAX (~2.147e9). With each
    // value capped at 1e8, the worst-case total is 4e8, comfortably in range.
    const MAX_COMPONENT: i32 = 100_000_000;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// Feature: test-coverage-codecov, Property 2: Token 计账不变量.
        ///
        /// For any non-negative input/output/cacheRead/cacheWrite:
        /// - prompt_tokens   == input + cacheRead + cacheWrite
        /// - total_tokens    == prompt_tokens + output
        /// - cached_tokens   == cacheRead
        /// - completion_tokens == output
        #[test]
        fn prop_compute_token_usage_accounting_invariants(
            input in 0i32..=MAX_COMPONENT,
            output in 0i32..=MAX_COMPONENT,
            cache_read in 0i32..=MAX_COMPONENT,
            cache_write in 0i32..=MAX_COMPONENT,
        ) {
            let u = compute_token_usage(input, output, cache_read, cache_write);

            prop_assert_eq!(u.prompt_tokens, input + cache_read + cache_write);
            prop_assert_eq!(u.total_tokens, u.prompt_tokens + output);
            prop_assert_eq!(u.cached_tokens, cache_read);
            prop_assert_eq!(u.completion_tokens, output);

            // Sanity: with non-negative inputs the derived counts stay non-negative.
            prop_assert!(u.prompt_tokens >= 0);
            prop_assert!(u.total_tokens >= 0);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Feature: test-coverage-codecov, Property 2: Token 计账不变量
        /// (`estimate_reasoning_tokens` monotonic / non-negative half).
        ///
        /// For any non-whitespace base string `a`:
        /// - the estimate is at least 1 (non-empty input yields >= 1 token),
        /// - the estimate never exceeds the trimmed byte length (each token
        ///   consumes at least one byte),
        /// - appending more space-separated content never decreases the count
        ///   (monotonic under extension; the space forces a clean tiktoken
        ///   pre-token boundary so `a`'s tokens are preserved as a prefix).
        #[test]
        fn prop_estimate_reasoning_tokens_monotone_and_bounded(
            a in "[a-zA-Z0-9]{1,64}",
            b in "[a-zA-Z0-9 ]{0,64}",
        ) {
            let base = estimate_reasoning_tokens(&a);

            // Non-negative + non-empty non-whitespace input => at least one token.
            prop_assert!(base >= 1);

            // Upper bound: token count can never exceed the trimmed byte length.
            prop_assert!(base as usize <= a.trim().len());

            // Monotonic under space-separated extension.
            let combined = format!("{a} {b}");
            prop_assert!(estimate_reasoning_tokens(&combined) >= base);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Feature: test-coverage-codecov, Property 2: Token 计账不变量
        /// (`estimate_reasoning_tokens` determinism + whitespace => 0).
        #[test]
        fn prop_estimate_reasoning_tokens_deterministic_and_zero_on_blank(
            s in "[a-zA-Z0-9 ]{0,80}",
            blank in "[ \t\r\n]{0,16}",
        ) {
            // Determinism: identical input always yields the same estimate.
            prop_assert_eq!(estimate_reasoning_tokens(&s), estimate_reasoning_tokens(&s));

            // Whitespace-only (or empty) input always estimates to zero.
            prop_assert_eq!(estimate_reasoning_tokens(&blank), 0);
        }
    }
}
