//! Unit tests for [`crate::bedrock::reasoning`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec).
//!
//! The source file declares this via a `#[path]` mod tests, so the
//! top-level `use super::*;` resolves to the implementation module.

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
    fn chat_backend(&self, _model: &str) -> crate::domain::ChatBackend {
        crate::domain::ChatBackend::Converse
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
    let outcome = build_reasoning_config("m", ReasoningEffort::Medium, Some(10000), None, &caps);
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
    let outcome = build_reasoning_config("m", ReasoningEffort::High, Some(1000), Some(2000), &caps);
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
