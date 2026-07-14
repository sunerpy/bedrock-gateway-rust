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
#[path = "reasoning_tests.rs"]
mod tests;
