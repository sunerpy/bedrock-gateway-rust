//! Reasoning / extended-thinking configuration mapping.
//!
//! This module is the PURE translation of an OpenAI `reasoning_effort` into the
//! Bedrock Converse `additionalModelRequestFields` shape, plus the side-signals
//! the caller must honor (an explicit `maxTokens` and dropping `topP`).
//!
//! It implements the four distinct reasoning paths from the legacy Python
//! `.legacy-python/src/api/models/bedrock.py` lines 1152-1189:
//!
//! 1. Adaptive thinking (Claude w/ `adaptive_thinking`, bedrock.py:1168-1172):
//!    `{ "thinking": {"type":"adaptive"}, "output_config": {"effort": <effort>} }`
//!    plus set `maxTokens = max_completion_tokens || max_tokens` and drop `topP`.
//! 2. Budget tokens (Claude non-adaptive, bedrock.py:1173-1177):
//!    `{ "reasoning_config": {"type":"enabled", "budget_tokens": <calc>} }`
//!    plus set `maxTokens` and drop `topP`. The budget is derived from
//!    [`crate::config::BudgetRatios`] (bedrock.py:1679-1689).
//! 3. DeepSeek string (bedrock.py:1178-1185):
//!    `{ "reasoning_config": "<effort string>" }`. No `maxTokens`, no `topP` drop.
//! 4. None / unsupported (bedrock.py:1186-1189): no reasoning fields at all;
//!    `reasoning_effort` is ignored.
//!
//! DE-HARDCODING CONTRACT: every path/ratio decision is read from
//! [`crate::domain::ModelCapabilities`] (config), never from model-id literals
//! or inline magic numbers. The only constants here are the Bedrock *wire* field
//! names (e.g. `"thinking"`, `"reasoning_config"`), which are protocol shape,
//! not model knowledge.

use serde_json::{json, Map, Value};

use crate::config::{BudgetRatios, ReasoningPath};
use crate::domain::ModelCapabilities;
use crate::openai::schema::ReasoningEffort;

/// Protocol floor for the Claude thinking `budget_tokens` when a model declares
/// no `min_budget_tokens` in config. Anthropic rejects a thinking budget below
/// 1024 with HTTP 400; this is a provider WIRE constraint, not model knowledge,
/// hence a code constant (config overrides it via `min_budget_tokens`).
const DEFAULT_MIN_BUDGET_TOKENS: i32 = 1024;

/// Spare completion tokens kept above the thinking budget when raising the
/// Bedrock `maxTokens` to satisfy `maxTokens > budget_tokens`. Anthropic treats
/// the thinking budget as a subset of `maxTokens`, so a request whose
/// `max_output_tokens` is at or below the (clamped) budget would otherwise have
/// zero room for the answer. This is a protocol headroom constant, not model
/// knowledge.
const COMPLETION_HEADROOM_TOKENS: i32 = 256;

/// The result of mapping a `reasoning_effort` for a given model.
///
/// This is what a translation layer (task 15) consumes: the Bedrock
/// `additionalModelRequestFields` object (empty when reasoning is unsupported),
/// an optional `maxTokens` override the caller must apply to `inferenceConfig`,
/// and a `drop_top_p` flag instructing the caller to remove `topP`
/// (bedrock.py:1166, 1267 — extended thinking rejects `topP`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningOutcome {
    /// Fields to merge into the Bedrock request's
    /// `additionalModelRequestFields`. Empty when the model does not support
    /// reasoning (path 4) — callers should add nothing in that case.
    pub additional_model_request_fields: Map<String, Value>,
    /// When `Some`, the caller MUST set `inferenceConfig.maxTokens` to this
    /// value (paths 1 & 2 set `max_completion_tokens || max_tokens`).
    pub max_tokens: Option<i32>,
    /// When `true`, the caller MUST drop `topP` from `inferenceConfig`
    /// (paths 1 & 2 — extended thinking cannot accept both temperature & topP).
    pub drop_top_p: bool,
}

impl ReasoningOutcome {
    /// `true` when no reasoning fields were produced (path 4 / no
    /// `reasoning_effort`). Convenience for callers that branch on "did we add
    /// anything?".
    pub fn is_empty(&self) -> bool {
        self.additional_model_request_fields.is_empty()
            && self.max_tokens.is_none()
            && !self.drop_top_p
    }
}

/// Map a [`ReasoningEffort`] to its Bedrock wire string.
///
/// Mirrors the Python code passing `chat_request.reasoning_effort` straight
/// through as the string value for both the adaptive `output_config.effort`
/// (bedrock.py:1171) and the DeepSeek `reasoning_config` (bedrock.py:1182). The
/// full enum is accepted, including the Bedrock `max` extension.
pub fn effort_str(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

/// Compute the Claude `budget_tokens` value for a reasoning effort.
///
/// Pure port of `_calc_budget_tokens` (bedrock.py:1679-1689): `low` and
/// `medium` scale `max_tokens` by their configured ratios; every other effort
/// (`high`/`xhigh`/`max`, and — matching the Python `else` branch —
/// `none`/`minimal`) yields `max_tokens - 1`.
///
/// The ratios come entirely from `ratios` (config); there are no inline
/// numbers. A non-negative ratio is applied as `(max_tokens as f32 * ratio)`
/// truncated toward zero (Python `int(...)`). The `high` field's `-1.0`
/// sentinel is never consulted for `low`/`medium`; the high/else branch always
/// uses the `max_tokens - 1` rule.
///
/// # Examples
///
/// ```
/// use bedrock_gateway_rust::bedrock::reasoning::calc_budget_tokens;
/// use bedrock_gateway_rust::config::BudgetRatios;
/// use bedrock_gateway_rust::openai::schema::ReasoningEffort;
///
/// let ratios = BudgetRatios { low: 0.3, medium: 0.6, high: -1.0 };
/// assert_eq!(calc_budget_tokens(1000, ReasoningEffort::Low, ratios), 300);
/// assert_eq!(calc_budget_tokens(1000, ReasoningEffort::Medium, ratios), 600);
/// assert_eq!(calc_budget_tokens(1000, ReasoningEffort::High, ratios), 999);
/// ```
pub fn calc_budget_tokens(max_tokens: i32, effort: ReasoningEffort, ratios: BudgetRatios) -> i32 {
    match effort {
        ReasoningEffort::Low => scale(max_tokens, ratios.low),
        ReasoningEffort::Medium => scale(max_tokens, ratios.medium),
        // high / xhigh / max (and none / minimal, per the Python `else`):
        // max_tokens - 1.
        _ => max_tokens - 1,
    }
}

/// Apply a non-negative ratio to `max_tokens`, truncating toward zero like the
/// Python `int(max_tokens * ratio)`.
fn scale(max_tokens: i32, ratio: f32) -> i32 {
    (max_tokens as f32 * ratio) as i32
}

/// Build the reasoning configuration for `(model, reasoning_effort)`.
///
/// Returns a [`ReasoningOutcome`]. The path is selected exclusively by
/// `caps.reasoning_path(model)`; no model-id literals are inspected here.
///
/// - [`ReasoningPath::AdaptiveThinking`] → `thinking`+`output_config`, set
///   `maxTokens = max_completion_tokens || max_tokens`, drop `topP`.
/// - [`ReasoningPath::BudgetTokens`] → `reasoning_config` enabled with
///   `budget_tokens` from [`calc_budget_tokens`] (ratios via
///   `caps.budget_ratios(model)`), set `maxTokens`, drop `topP`.
/// - [`ReasoningPath::DeepseekString`] → `reasoning_config = "<effort>"`.
/// - [`ReasoningPath::None`] → empty outcome (effort ignored).
///
/// `max_tokens` and `max_completion_tokens` are the request's respective fields
/// (OpenAI `max_tokens` / `max_completion_tokens`). `caps` is the config-driven
/// capability resolver.
pub fn build_reasoning_config(
    model: &str,
    reasoning_effort: ReasoningEffort,
    max_tokens: Option<i32>,
    max_completion_tokens: Option<i32>,
    caps: &dyn ModelCapabilities,
) -> ReasoningOutcome {
    match caps.reasoning_path(model) {
        ReasoningPath::AdaptiveThinking => {
            let effective = max_completion_tokens.or(max_tokens);
            let mut fields = Map::new();
            fields.insert("thinking".to_string(), json!({ "type": "adaptive" }));
            fields.insert(
                "output_config".to_string(),
                json!({ "effort": effort_str(reasoning_effort) }),
            );
            ReasoningOutcome {
                additional_model_request_fields: fields,
                max_tokens: effective,
                drop_top_p: true,
            }
        }
        ReasoningPath::BudgetTokens => {
            let effective = max_completion_tokens.or(max_tokens);
            // Ratios are config-supplied; fall back to a zeroed ratio set with
            // the `-1.0` sentinel so the high/else branch still yields
            // `max_tokens - 1` when no config is present (defensive — the
            // resolver supplies a `default` entry in practice).
            let ratios = caps.budget_ratios(model).unwrap_or(BudgetRatios {
                low: 0.0,
                medium: 0.0,
                high: -1.0,
            });
            // `_calc_budget_tokens(max_tokens, ...)` (bedrock.py:1174). When no
            // max is supplied, there is nothing meaningful to scale; default to
            // 0 so the arithmetic is well-defined (callers always supply one).
            let budget = calc_budget_tokens(effective.unwrap_or(0), reasoning_effort, ratios);
            // Clamp the ratio-scaled budget UP to the configured floor (default
            // 1024) so a small max_tokens never violates Anthropic's hard
            // minimum. Then ensure the Bedrock maxTokens strictly exceeds the
            // budget by a completion-headroom margin (thinking budget is a
            // subset of maxTokens). This only adjusts the maxTokens sent to
            // Bedrock — no client-facing wire field changes.
            let min_budget = caps
                .min_budget_tokens(model)
                .map(|v| v as i32)
                .unwrap_or(DEFAULT_MIN_BUDGET_TOKENS);
            let budget = budget.max(min_budget);
            let required_max = budget + COMPLETION_HEADROOM_TOKENS;
            let max_tokens = Some(effective.unwrap_or(0).max(required_max));
            let mut fields = Map::new();
            fields.insert(
                "reasoning_config".to_string(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
            ReasoningOutcome {
                additional_model_request_fields: fields,
                max_tokens,
                drop_top_p: true,
            }
        }
        ReasoningPath::DeepseekString => {
            let mut fields = Map::new();
            fields.insert(
                "reasoning_config".to_string(),
                Value::String(effort_str(reasoning_effort).to_string()),
            );
            ReasoningOutcome {
                additional_model_request_fields: fields,
                max_tokens: None,
                drop_top_p: false,
            }
        }
        ReasoningPath::None => ReasoningOutcome::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test double for [`ModelCapabilities`] that returns a fixed reasoning
    /// path and optional budget ratios. This keeps the unit tests free of any
    /// real model-id knowledge while exercising all four paths and the budget
    /// math against a controlled [`BudgetRatios`].
    struct StubCaps {
        path: ReasoningPath,
        ratios: Option<BudgetRatios>,
        min_budget: Option<u32>,
    }

    impl StubCaps {
        fn new(path: ReasoningPath, ratios: Option<BudgetRatios>) -> Self {
            Self {
                path,
                ratios,
                min_budget: None,
            }
        }

        fn with_min_budget(
            path: ReasoningPath,
            ratios: Option<BudgetRatios>,
            min_budget: Option<u32>,
        ) -> Self {
            Self {
                path,
                ratios,
                min_budget,
            }
        }
    }

    impl ModelCapabilities for StubCaps {
        fn has(&self, _model: &str, _cap: crate::config::Capability) -> bool {
            false
        }
        fn resolve_foundation(&self, model_or_profile: &str) -> String {
            model_or_profile.to_string()
        }
        fn budget_ratios(&self, _model: &str) -> Option<BudgetRatios> {
            self.ratios
        }
        fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
            self.min_budget
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
            Vec::new()
        }
        fn reasoning_path(&self, _model: &str) -> ReasoningPath {
            self.path
        }
        fn responses_backend(&self, _model: &str) -> crate::domain::ResponsesBackend {
            crate::domain::ResponsesBackend::Converse
        }
        fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
            None
        }
    }

    /// A controlled ratio set mirroring the config defaults, used to assert the
    /// budget math without depending on `config/models.toml`.
    const TEST_RATIOS: BudgetRatios = BudgetRatios {
        low: 0.3,
        medium: 0.6,
        high: -1.0,
    };

    // -- Path 1: adaptive thinking -----------------------------------------

    #[test]
    fn adaptive_path_emits_thinking_and_output_config() {
        // bedrock.py:1168-1172. Effort string flows into output_config.effort;
        // maxTokens = max_completion_tokens || max_tokens; topP dropped.
        let caps = StubCaps::new(ReasoningPath::AdaptiveThinking, None);
        let outcome = build_reasoning_config(
            "any-model",
            ReasoningEffort::High,
            Some(4096),
            Some(8192),
            &caps,
        );

        assert_eq!(
            outcome.additional_model_request_fields["thinking"],
            json!({ "type": "adaptive" })
        );
        assert_eq!(
            outcome.additional_model_request_fields["output_config"],
            json!({ "effort": "high" })
        );
        // max_completion_tokens wins over max_tokens.
        assert_eq!(outcome.max_tokens, Some(8192));
        assert!(outcome.drop_top_p);
        // No reasoning_config on the adaptive path.
        assert!(!outcome
            .additional_model_request_fields
            .contains_key("reasoning_config"));
    }

    #[test]
    fn adaptive_falls_back_to_max_tokens_when_no_completion_tokens() {
        let caps = StubCaps::new(ReasoningPath::AdaptiveThinking, None);
        let outcome = build_reasoning_config("m", ReasoningEffort::Medium, Some(2048), None, &caps);
        assert_eq!(outcome.max_tokens, Some(2048));
        assert_eq!(
            outcome.additional_model_request_fields["output_config"],
            json!({ "effort": "medium" })
        );
    }

    // -- Path 2: budget tokens ---------------------------------------------

    #[test]
    fn budget_path_low_uses_ratio_low() {
        // low = int(max * 0.3). With max=10000 the ratio result (3000) is above
        // the 1024 floor, so the floor does not engage and max_tokens is
        // unchanged.
        let caps = StubCaps::new(ReasoningPath::BudgetTokens, Some(TEST_RATIOS));
        let outcome = build_reasoning_config("m", ReasoningEffort::Low, Some(10000), None, &caps);
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            json!({ "type": "enabled", "budget_tokens": 3000 })
        );
        assert_eq!(outcome.max_tokens, Some(10000));
        assert!(outcome.drop_top_p);
    }

    #[test]
    fn budget_path_medium_uses_ratio_medium() {
        // medium = int(max * 0.6). max=10000 → 6000, above the 1024 floor.
        let caps = StubCaps::new(ReasoningPath::BudgetTokens, Some(TEST_RATIOS));
        let outcome =
            build_reasoning_config("m", ReasoningEffort::Medium, Some(10000), None, &caps);
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            json!({ "type": "enabled", "budget_tokens": 6000 })
        );
        assert_eq!(outcome.max_tokens, Some(10000));
    }

    #[test]
    fn budget_path_high_uses_max_minus_one_sentinel() {
        // high/xhigh/max → max_tokens - 1. max=10000 → 9999, above the floor.
        let caps = StubCaps::new(ReasoningPath::BudgetTokens, Some(TEST_RATIOS));
        for effort in [
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ] {
            let outcome = build_reasoning_config("m", effort, Some(10000), None, &caps);
            assert_eq!(
                outcome.additional_model_request_fields["reasoning_config"],
                json!({ "type": "enabled", "budget_tokens": 9999 }),
                "effort {effort:?} should yield max_tokens - 1"
            );
        }
    }

    #[test]
    fn budget_path_prefers_completion_tokens_for_calc_and_max() {
        // maxTokens = max_completion_tokens || max_tokens, and the budget calc
        // uses that same effective value. max_completion_tokens=20000 → budget
        // int(20000*0.3)=6000 (above floor), max_tokens unchanged at 20000.
        let caps = StubCaps::new(ReasoningPath::BudgetTokens, Some(TEST_RATIOS));
        let outcome =
            build_reasoning_config("m", ReasoningEffort::Low, Some(10000), Some(20000), &caps);
        assert_eq!(outcome.max_tokens, Some(20000));
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            json!({ "type": "enabled", "budget_tokens": 6000 })
        );
    }

    #[test]
    fn budget_path_clamps_small_budget_to_default_floor() {
        // Regression for the 400 bug: a tiny max_tokens (50) with low effort
        // yields a ratio budget of 15, which violates Anthropic's 1024 floor.
        // The budget is clamped UP to 1024 (no config override → default), and
        // the Bedrock maxTokens is raised to 1024 + 256 headroom = 1280 so the
        // thinking budget fits with completion room.
        let caps = StubCaps::new(ReasoningPath::BudgetTokens, Some(TEST_RATIOS));
        let outcome = build_reasoning_config("m", ReasoningEffort::Low, Some(50), None, &caps);
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            json!({ "type": "enabled", "budget_tokens": 1024 })
        );
        assert_eq!(outcome.max_tokens, Some(1280));
        assert!(outcome.drop_top_p);
    }

    #[test]
    fn budget_path_respects_config_min_budget_override() {
        // A config-supplied min_budget_tokens overrides the protocol default.
        // max_tokens=50, low effort → ratio budget 15, clamped to the override
        // 2048, maxTokens raised to 2048 + 256 = 2304.
        let caps =
            StubCaps::with_min_budget(ReasoningPath::BudgetTokens, Some(TEST_RATIOS), Some(2048));
        let outcome = build_reasoning_config("m", ReasoningEffort::Low, Some(50), None, &caps);
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            json!({ "type": "enabled", "budget_tokens": 2048 })
        );
        assert_eq!(outcome.max_tokens, Some(2304));
    }

    // -- Path 3: deepseek string -------------------------------------------

    #[test]
    fn deepseek_path_emits_string_reasoning_config() {
        // bedrock.py:1178-1185: reasoning_config is the bare effort string.
        let caps = StubCaps::new(ReasoningPath::DeepseekString, None);
        let outcome = build_reasoning_config("m", ReasoningEffort::High, Some(1000), None, &caps);
        assert_eq!(
            outcome.additional_model_request_fields["reasoning_config"],
            Value::String("high".to_string())
        );
        // No maxTokens override and no topP drop on the DeepSeek path.
        assert_eq!(outcome.max_tokens, None);
        assert!(!outcome.drop_top_p);
    }

    #[test]
    fn deepseek_path_passes_all_effort_levels_through() {
        let caps = StubCaps::new(ReasoningPath::DeepseekString, None);
        for (effort, expected) in [
            (ReasoningEffort::None, "none"),
            (ReasoningEffort::Minimal, "minimal"),
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::Xhigh, "xhigh"),
            (ReasoningEffort::Max, "max"),
        ] {
            let outcome = build_reasoning_config("m", effort, Some(1000), None, &caps);
            assert_eq!(
                outcome.additional_model_request_fields["reasoning_config"],
                Value::String(expected.to_string())
            );
        }
    }

    // -- Path 4: none / unsupported ----------------------------------------

    #[test]
    fn none_path_emits_nothing() {
        // bedrock.py:1186-1189: reasoning_effort is ignored entirely.
        let caps = StubCaps::new(ReasoningPath::None, Some(TEST_RATIOS));
        let outcome =
            build_reasoning_config("m", ReasoningEffort::High, Some(1000), Some(2000), &caps);
        assert!(outcome.additional_model_request_fields.is_empty());
        assert_eq!(outcome.max_tokens, None);
        assert!(!outcome.drop_top_p);
        assert!(outcome.is_empty());
    }

    // -- Budget math helper (direct) ---------------------------------------

    #[test]
    fn calc_budget_tokens_matches_python_ratios() {
        // Direct unit test of the helper against a controlled ratio set
        // (bedrock.py:1679-1689).
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::Low, TEST_RATIOS),
            300
        );
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::Medium, TEST_RATIOS),
            600
        );
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::High, TEST_RATIOS),
            999
        );
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::Xhigh, TEST_RATIOS),
            999
        );
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::Max, TEST_RATIOS),
            999
        );
        // none / minimal follow the Python `else` branch → max - 1.
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::None, TEST_RATIOS),
            999
        );
        assert_eq!(
            calc_budget_tokens(1000, ReasoningEffort::Minimal, TEST_RATIOS),
            999
        );
    }

    #[test]
    fn calc_budget_tokens_truncates_toward_zero() {
        // int(333 * 0.3) == int(99.9) == 99 (Python truncation parity).
        assert_eq!(
            calc_budget_tokens(333, ReasoningEffort::Low, TEST_RATIOS),
            99
        );
    }

    #[test]
    fn effort_str_covers_full_enum() {
        assert_eq!(effort_str(ReasoningEffort::None), "none");
        assert_eq!(effort_str(ReasoningEffort::Minimal), "minimal");
        assert_eq!(effort_str(ReasoningEffort::Low), "low");
        assert_eq!(effort_str(ReasoningEffort::Medium), "medium");
        assert_eq!(effort_str(ReasoningEffort::High), "high");
        assert_eq!(effort_str(ReasoningEffort::Xhigh), "xhigh");
        assert_eq!(effort_str(ReasoningEffort::Max), "max");
    }
}
