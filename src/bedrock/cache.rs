//! Config-driven prompt caching (`cachePoint` insertion) — pure functions.
//!
//! This module ports the prompt-caching half of the legacy Python request
//! builder into small, pure, testable functions that *decorate* the already-built
//! Bedrock Converse `system` / `messages` arrays produced by
//! [`crate::bedrock::translate`].
//!
//! Ported logic (with provenance line ranges from
//! `.legacy-python/src/api/models/bedrock.py`):
//! - `_parse_system_prompts` cache branch (709-772) → [`decorate_system_blocks`]
//! - `_parse_messages` cache branch          (996-1036) → [`decorate_messages`]
//! - `_supports_prompt_caching`              (419-452) → [`supports_caching`]
//! - `_get_max_cache_tokens` (Nova 20000)    (454-477) → `caps.max_cache_tokens`
//! - extra_body `prompt_caching` strip        (1238-1241) →
//!   [`PromptCachingControl::extract_and_strip`]
//!
//! ## De-hardcoding contract
//!
//! There are NO cache magic numbers in this file. Every limit/minimum/ceiling
//! flows from the [`ModelCapabilities`] accessors backed by `config/models.toml`:
//! - `caps.cache_min_tokens(model)` — minimum tokens before caching is worthwhile
//!   (gates insertion; `None` ⇒ no minimum to clear).
//! - `caps.max_cache_tokens(model)` — Nova-style ceiling (warn-only, like Python:
//!   still inserts and lets Bedrock enforce the real limit).
//! - `caps.max_cache_checkpoints(model)` — ceiling on the number of `cachePoint`
//!   blocks inserted across system + messages.
//! - capability gate ([`supports_caching`]): a model "supports caching" iff its
//!   config entry declares cache parameters (`cache_min_tokens` or
//!   `max_cache_tokens`). No model id is named in code.
//!
//! ## The control field never reaches Bedrock
//!
//! The per-request enablement object lives under `extra_body.prompt_caching`. It
//! is a GATEWAY control field, not a Bedrock field.
//! [`PromptCachingControl::extract_and_strip`] parses it out of `extra_body` AND
//! removes the `prompt_caching` key, so the caller can hand the cleaned
//! `extra_body` to the translate layer (which forwards `extra_body` into
//! `additionalModelRequestFields`). This mirrors bedrock.py:1238-1241.
//!
//! ## Purity
//!
//! All decorators take input arrays + flags and return decorated arrays. They
//! never call Bedrock, never read globals, and never mutate the request.

use serde_json::{json, Value};

use crate::domain::{Capability, ModelCapabilities};

/// The Bedrock cache-checkpoint block appended into a content/system array.
///
/// Mirrors the Python literal `{"cachePoint": {"type": "default"}}`
/// (bedrock.py:767, 721 example output). When `ttl` is `Some`, the checkpoint
/// carries a `ttl` key (`"5m"` | `"1h"`) so Bedrock applies the requested cache
/// lifetime; when `ttl` is `None`, NO `ttl` key is emitted — the output is
/// byte-identical to the historical `{"cachePoint": {"type": "default"}}`,
/// preserving backward compatibility for any path that opts out.
fn cache_point(ttl: Option<&str>) -> Value {
    match ttl {
        Some(ttl) => json!({ "cachePoint": { "type": "default", "ttl": ttl } }),
        None => json!({ "cachePoint": { "type": "default" } }),
    }
}

/// Per-request prompt-caching control, parsed from `extra_body.prompt_caching`.
///
/// This is the OpenAI-sanctioned `extra_body` extension (the README's "Option
/// B"): `{"prompt_caching": {"system"?: bool, "messages"?: bool}}`. Each field
/// is tri-state:
/// - `Some(true)`  — caching for this scope is explicitly enabled by the request.
/// - `Some(false)` — caching for this scope is explicitly disabled by the request.
/// - `None`        — the request did not mention this scope; fall back to the
///   global [`crate::config::AppSettings::enable_prompt_caching`] default.
///
/// This precedence matches the Python override logic (bedrock.py:744-748,
/// 1004-1007): `cache_enabled = ENABLE_PROMPT_CACHING`, then if the key is
/// present in `extra_body.prompt_caching`, the request value wins.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptCachingControl {
    /// Explicit per-request control for system-prompt caching, if present.
    pub system: Option<bool>,
    /// Explicit per-request control for message caching, if present.
    pub messages: Option<bool>,
    /// Explicit per-request cache TTL override (`"5m"` | `"1h"`), if present.
    /// This is the Option-B per-request knob under `extra_body.prompt_caching`;
    /// it takes precedence over the env `PROMPT_CACHE_TTL` default. `None` ⇒ the
    /// request did not request a TTL and the settings default applies.
    pub ttl: Option<String>,
}

impl PromptCachingControl {
    /// Parse the `prompt_caching` control object out of `extra_body` AND strip
    /// the `prompt_caching` key from it.
    ///
    /// `extra_body` is the request's optional `extra_body` value. After this
    /// call, the `prompt_caching` key is GONE from `extra_body` (if it was an
    /// object), so the cleaned `extra_body` is safe to forward to Bedrock — the
    /// control field can never leak into `additionalModelRequestFields`.
    ///
    /// Returns the parsed control. A missing/non-object `extra_body`, or a
    /// missing/non-object `prompt_caching`, yields the all-`None` default.
    ///
    /// Mirrors bedrock.py:744-748 / 1004-1007 (parse) and 1238-1241 (strip).
    ///
    /// # Example
    /// ```
    /// use serde_json::json;
    /// use bedrock_gateway_rust::bedrock::cache::PromptCachingControl;
    ///
    /// let mut extra = Some(json!({
    ///     "thinking": {"type": "enabled"},
    ///     "prompt_caching": {"system": true}
    /// }));
    /// let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
    /// assert_eq!(ctrl.system, Some(true));
    /// assert_eq!(ctrl.messages, None);
    /// // The control field is gone; only the Bedrock-bound field remains.
    /// let obj = extra.unwrap();
    /// assert!(obj.get("prompt_caching").is_none());
    /// assert!(obj.get("thinking").is_some());
    /// ```
    pub fn extract_and_strip(extra_body: &mut Option<Value>) -> Self {
        let Some(Value::Object(map)) = extra_body.as_mut() else {
            return Self::default();
        };
        // Remove (not just read) so the control field never reaches Bedrock.
        let removed = map.remove("prompt_caching");
        match removed {
            Some(Value::Object(pc)) => Self {
                system: pc.get("system").and_then(Value::as_bool),
                messages: pc.get("messages").and_then(Value::as_bool),
                ttl: pc.get("ttl").and_then(Value::as_str).map(str::to_string),
            },
            _ => Self::default(),
        }
    }

    /// Parse the control object from a borrowed `extra_body` WITHOUT mutating it.
    ///
    /// Useful when the caller only needs the enablement decision and strips the
    /// field elsewhere (e.g. the translate layer already drops `prompt_caching`).
    pub fn parse(extra_body: Option<&Value>) -> Self {
        let Some(Value::Object(map)) = extra_body else {
            return Self::default();
        };
        match map.get("prompt_caching") {
            Some(Value::Object(pc)) => Self {
                system: pc.get("system").and_then(Value::as_bool),
                messages: pc.get("messages").and_then(Value::as_bool),
                ttl: pc.get("ttl").and_then(Value::as_str).map(str::to_string),
            },
            _ => Self::default(),
        }
    }

    /// Resolve whether system-prompt caching is enabled, given the global
    /// default. The per-request value wins when present (bedrock.py:744-748).
    pub fn system_enabled(&self, global_default: bool) -> bool {
        self.system.unwrap_or(global_default)
    }

    /// Resolve whether message caching is enabled, given the global default.
    /// The per-request value wins when present (bedrock.py:1004-1007).
    pub fn messages_enabled(&self, global_default: bool) -> bool {
        self.messages.unwrap_or(global_default)
    }
}

/// Does the model support prompt caching, per config?
///
/// De-hardcoded replacement for the Python `_supports_prompt_caching`
/// substring whitelist (bedrock.py:419-452). Instead of naming Claude/Nova in
/// code, a model "supports caching" iff its `config/models.toml` entry declares
/// cache parameters — either a `cache_min_tokens` minimum or a
/// `max_cache_tokens` ceiling. Adding/removing caching support for a model is a
/// TOML-only edit.
pub fn supports_caching(model: &str, caps: &dyn ModelCapabilities) -> bool {
    caps.cache_min_tokens(model).is_some() || caps.max_cache_tokens(model).is_some()
}

/// Rough token estimate for the text carried by a slice of `{"text": ...}`
/// blocks, used only for the `cache_min_tokens` gate.
///
/// Mirrors the Python heuristic (bedrock.py:754-755): join the text and apply
/// `words * 1.3`. This is intentionally cheap and approximate — it gates whether
/// caching is *worthwhile*, not billing.
fn estimate_block_tokens(blocks: &[Value]) -> u32 {
    let mut words: u64 = 0;
    for b in blocks {
        if let Some(text) = b.get("text").and_then(Value::as_str) {
            words += text.split_whitespace().count() as u64;
        }
    }
    // words * 1.3, saturating into u32.
    ((words as f64) * 1.3).round().min(u32::MAX as f64) as u32
}

/// Decorate the Bedrock `toolConfig.tools` array with a trailing `cachePoint`,
/// when enabled + supported + the array is non-empty + budget remains.
///
/// Pure: takes the `tools` array (the `toolConfig.tools` value, where each entry
/// is a `{"toolSpec": ...}` block) plus the running checkpoint budget, and
/// returns a possibly-extended array.
///
/// Tools are the FIRST zone in the Bedrock cache order (tools → system →
/// messages). Long, stable tool definitions (e.g. opencode/codex agent tool
/// specs) are the highest-value cache prefix, so the tools cachePoint is placed
/// at the array tail unconditionally when supported+enabled, subject only to the
/// shared checkpoint ceiling. Unlike [`decorate_system_blocks`], there is NO
/// `cache_min_tokens` floor here: tool definitions are cached on presence, not
/// length (Bedrock still enforces the real minimum and silently no-ops a
/// too-small prefix).
///
/// - empty tools ⇒ unchanged (never inject when no tools are present).
/// - model doesn't support caching ⇒ unchanged.
/// - `enabled` false ⇒ unchanged.
/// - `already_used` checkpoints ≥ `max_checkpoints` ⇒ no room, unchanged.
/// - already ends with a `cachePoint` ⇒ unchanged (no double-insert).
/// - otherwise append `{"cachePoint": {"type": "default"}}` to the tail.
///
/// `tools` is expected to be a JSON array (the `toolConfig.tools` value produced
/// by [`crate::bedrock::tools::build_tool_config`]). A non-array value is
/// returned unchanged.
pub fn decorate_tools(
    tools: Value,
    model: &str,
    caps: &dyn ModelCapabilities,
    enabled: bool,
    already_used: u32,
    max_checkpoints: Option<u32>,
    ttl: Option<&str>,
) -> Value {
    let Value::Array(mut specs) = tools else {
        return tools;
    };

    // Empty tools ⇒ nothing to cache (and the contract forbids injecting a
    // tools cachePoint when tools are absent/empty).
    if specs.is_empty() {
        return Value::Array(specs);
    }
    if !supports_caching(model, caps) {
        return Value::Array(specs);
    }
    if !enabled {
        return Value::Array(specs);
    }
    // Shared checkpoint ceiling from config (NO literal). `None` ⇒ unbounded.
    if let Some(max_cp) = max_checkpoints {
        if already_used >= max_cp {
            return Value::Array(specs);
        }
    }
    // Never double-insert if the tail is already a cachePoint.
    let ends_with_cp = specs
        .last()
        .and_then(Value::as_object)
        .is_some_and(|o| o.contains_key("cachePoint"));
    if ends_with_cp {
        return Value::Array(specs);
    }

    specs.push(cache_point(ttl));
    Value::Array(specs)
}

/// Decorate the Bedrock `system` blocks with a trailing `cachePoint`, when
/// enabled + supported + over the configured minimum.
///
/// Pure: takes the system array and flags, returns a possibly-extended array.
/// Ports the cache branch of `_parse_system_prompts` (bedrock.py:733-772):
/// - empty system ⇒ unchanged (bedrock.py:733-734).
/// - model doesn't support caching ⇒ unchanged (bedrock.py:737-738).
/// - `enabled` is the already-resolved decision (global default OR per-request
///   `prompt_caching.system`); `false` ⇒ unchanged (bedrock.py:750-751).
/// - over `caps.max_cache_tokens` ⇒ still inserts; Bedrock enforces the real
///   limit (bedrock.py:759-764, "Still add cachePoint").
/// - below `caps.cache_min_tokens` ⇒ NOT worthwhile, skip insertion.
/// - otherwise append `{"cachePoint": {"type": "default"}}` (bedrock.py:767).
///
/// `system` is expected to be a JSON array (as produced by
/// `translate::parse_system_prompts`). A non-array value is returned unchanged.
pub fn decorate_system_blocks(
    system: Value,
    model: &str,
    caps: &dyn ModelCapabilities,
    enabled: bool,
    ttl: Option<&str>,
) -> Value {
    let Value::Array(mut blocks) = system else {
        return system;
    };

    // Empty system prompts ⇒ nothing to cache (bedrock.py:733-734).
    if blocks.is_empty() {
        return Value::Array(blocks);
    }
    // Capability gate (bedrock.py:737-738).
    if !supports_caching(model, caps) {
        return Value::Array(blocks);
    }
    // Resolved enablement (bedrock.py:750-751).
    if !enabled {
        return Value::Array(blocks);
    }

    // cache_min_tokens floor: only cache when it is worthwhile. `None` ⇒ no
    // floor to clear (config-supplied; no literal here).
    if let Some(min_tokens) = caps.cache_min_tokens(model) {
        let estimated = estimate_block_tokens(&blocks);
        if estimated < min_tokens {
            return Value::Array(blocks);
        }
    }

    // max_cache_tokens is warn-only in the Python parity: still insert and let
    // Bedrock enforce the real ceiling (bedrock.py:759-764). We don't log here
    // (pure fn); the ceiling is read but does not block insertion.
    let _ = caps.max_cache_tokens(model);

    blocks.push(cache_point(ttl));
    Value::Array(blocks)
}

/// Decorate the Bedrock `messages` array with a trailing `cachePoint` on the
/// last user turn, when enabled + supported, respecting the checkpoint ceiling.
///
/// Pure: takes the messages array (+ how many checkpoints were already spent on
/// system caching + the configured checkpoint ceiling) and flags, returns a
/// possibly-modified array.
///
/// Ports the cache branch of `_parse_messages` (bedrock.py:996-1036),
/// simplified to the primary checkpoint (CP2 — last eligible user turn). The
/// Python midpoint CP3 is an additional optimization; the count ceiling here is
/// supplied by the caller via `max_checkpoints` (read from
/// `config/models.toml`'s `max_cache_checkpoints`) so it can never exceed
/// config. `None` ⇒ no configured ceiling.
/// - no messages ⇒ unchanged (bedrock.py:997).
/// - model doesn't support caching ⇒ unchanged (bedrock.py:998-999).
/// - `enabled` false ⇒ unchanged (bedrock.py:1009 inverse).
/// - `already_used` checkpoints ≥ `max_checkpoints` ⇒ no room, unchanged.
/// - otherwise append a `cachePoint` block to the last user turn's content that
///   does not already end with one. `toolResult`-only turns are skipped
///   (bedrock.py:1018-1019: Bedrock requires toolResult-only content in that
///   turn).
///
/// `messages` is expected to be a JSON array of `{"role","content":[...]}`
/// objects. A non-array value is returned unchanged.
pub fn decorate_messages(
    messages: Value,
    model: &str,
    caps: &dyn ModelCapabilities,
    enabled: bool,
    already_used: u32,
    max_checkpoints: Option<u32>,
    ttl: Option<&str>,
) -> Value {
    let Value::Array(mut turns) = messages else {
        return messages;
    };

    if turns.is_empty() {
        return Value::Array(turns);
    }
    if !supports_caching(model, caps) {
        return Value::Array(turns);
    }
    if !enabled {
        return Value::Array(turns);
    }
    // Checkpoint ceiling from config (NO literal). `None` ⇒ unbounded, but we
    // still only insert a single message checkpoint here.
    if let Some(max_cp) = max_checkpoints {
        if already_used >= max_cp {
            return Value::Array(turns);
        }
    }

    // Find the last eligible user turn: role == "user" and its content does not
    // consist solely of toolResult blocks (bedrock.py:1018-1019) and does not
    // already end with a cachePoint.
    let last_eligible = turns.iter().rposition(|turn| {
        let is_user = turn.get("role").and_then(Value::as_str) == Some("user");
        if !is_user {
            return false;
        }
        match turn.get("content").and_then(Value::as_array) {
            Some(content) if !content.is_empty() => {
                let all_tool_result = content
                    .iter()
                    .all(|b| b.as_object().is_some_and(|o| o.contains_key("toolResult")));
                let ends_with_cp = content
                    .last()
                    .and_then(Value::as_object)
                    .is_some_and(|o| o.contains_key("cachePoint"));
                !all_tool_result && !ends_with_cp
            }
            _ => false,
        }
    });

    if let Some(idx) = last_eligible {
        if let Some(content) = turns[idx].get_mut("content").and_then(Value::as_array_mut) {
            content.push(cache_point(ttl));
        }
    }

    Value::Array(turns)
}

/// The canonical 1-hour prompt-cache TTL literal.
const TTL_1H: &str = "1h";
/// The canonical 5-minute prompt-cache TTL literal (always allowed).
const TTL_5M: &str = "5m";

/// The outcome of resolving the effective prompt-cache TTL for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTtl {
    /// The TTL string threaded into every `cachePoint` for this request
    /// (uniform across tools/system/messages), e.g. `"5m"` or `"1h"`.
    pub effective: String,
    /// The TTL the caller requested before the capability gate (per-request
    /// override or the settings default).
    pub requested: String,
    /// `true` when a requested `1h` was silently downgraded to `5m` because the
    /// resolved model lacks [`Capability::CacheTtl1h`]. The caller emits a
    /// metadata-only WARN when this is set.
    pub downgraded: bool,
}

/// Resolve the effective, UNIFORM prompt-cache TTL for a request.
///
/// Precedence: the per-request `extra_body.prompt_caching.ttl` (`ctrl_ttl`) wins
/// over the settings `PROMPT_CACHE_TTL` default. `1h` is honored only when the
/// resolved model declares [`Capability::CacheTtl1h`]; otherwise it is silently
/// downgraded to `5m` (always allowed for any caching-capable model). Any value
/// other than `1h` passes through unchanged (no gate) — `5m` is the documented
/// default, and an unrecognized value is forwarded verbatim for Bedrock to
/// validate.
pub fn resolve_cache_ttl(
    ctrl_ttl: Option<&str>,
    settings_default: &str,
    model: &str,
    caps: &dyn ModelCapabilities,
) -> ResolvedTtl {
    let requested = ctrl_ttl.unwrap_or(settings_default).to_string();
    if requested == TTL_1H && !caps.has(model, Capability::CacheTtl1h) {
        return ResolvedTtl {
            effective: TTL_5M.to_string(),
            requested,
            downgraded: true,
        };
    }
    ResolvedTtl {
        effective: requested.clone(),
        requested,
        downgraded: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bedrock::capabilities::ConfigModelCapabilities;
    use crate::config::ModelCapabilityConfig;

    const MODELS_TOML: &str = "config/models.toml";

    // Nova has max_cache_tokens=20000 configured ⇒ supports caching.
    const NOVA: &str = "us.amazon.nova-pro-v1:0";
    // Claude opus-4-8 has NO cache params in the shipped config ⇒ per our
    // config-driven gate, it does NOT "support caching" today. This is correct
    // de-hardcoded behavior: caching support is a TOML edit, not code.
    const CLAUDE_NO_CACHE: &str = "global.anthropic.claude-opus-4-8-20251101-v1:0";

    fn caps() -> ConfigModelCapabilities {
        let config = ModelCapabilityConfig::load(MODELS_TOML).expect("load models.toml");
        ConfigModelCapabilities::new(config)
    }

    /// A test-only capabilities impl with tunable cache params so the gating
    /// logic can be exercised without depending on shipped TOML values. The
    /// checkpoint ceiling is NOT a trait accessor — it is passed to
    /// `decorate_messages` directly — so it is stored separately and read via
    /// [`FakeCaps::checkpoints`].
    struct FakeCaps {
        min: Option<u32>,
        max: Option<u32>,
        checkpoints: Option<u32>,
    }

    impl ModelCapabilities for FakeCaps {
        fn has(&self, _model: &str, _cap: crate::domain::Capability) -> bool {
            false
        }
        fn resolve_foundation(&self, m: &str) -> String {
            m.to_string()
        }
        fn budget_ratios(&self, _model: &str) -> Option<crate::domain::BudgetRatios> {
            None
        }
        fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
            None
        }
        fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
            self.max
        }
        fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
            self.min
        }
        fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
            self.checkpoints
        }
        fn beta_headers(&self, _model: &str) -> Vec<String> {
            Vec::new()
        }
        fn reasoning_path(&self, _model: &str) -> crate::domain::ReasoningPath {
            crate::domain::ReasoningPath::None
        }
        fn responses_backend(&self, _model: &str) -> crate::domain::ResponsesBackend {
            crate::domain::ResponsesBackend::Converse
        }
        fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
            None
        }
    }

    impl FakeCaps {
        fn supports(min: Option<u32>, max: Option<u32>, checkpoints: Option<u32>) -> Self {
            Self {
                min,
                max,
                checkpoints,
            }
        }

        fn checkpoints(&self) -> Option<u32> {
            self.checkpoints
        }
    }

    fn sys(texts: &[&str]) -> Value {
        Value::Array(texts.iter().map(|t| json!({ "text": t })).collect())
    }

    fn long_text(words: usize) -> String {
        vec!["word"; words].join(" ")
    }

    // ---- PromptCachingControl: parse + strip ----

    #[test]
    fn control_extract_strips_field_and_parses_flags() {
        let mut extra = Some(json!({
            "thinking": {"type": "enabled"},
            "prompt_caching": {"system": true, "messages": false}
        }));
        let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
        assert_eq!(ctrl.system, Some(true));
        assert_eq!(ctrl.messages, Some(false));

        // CRITICAL: the control field is removed and never reaches Bedrock.
        let obj = extra.expect("extra_body retained");
        assert!(
            obj.get("prompt_caching").is_none(),
            "prompt_caching control field must be stripped"
        );
        // Non-control fields survive for Bedrock passthrough.
        assert!(obj.get("thinking").is_some());
    }

    #[test]
    fn control_strip_when_only_control_field_present() {
        let mut extra = Some(json!({ "prompt_caching": {"system": true} }));
        let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
        assert_eq!(ctrl.system, Some(true));
        // The object remains (now empty) but prompt_caching is gone.
        let obj = extra.expect("extra retained");
        assert!(obj.get("prompt_caching").is_none());
        assert_eq!(obj.as_object().map(|m| m.len()), Some(0));
    }

    #[test]
    fn control_absent_extra_body_is_default() {
        let mut none: Option<Value> = None;
        assert_eq!(
            PromptCachingControl::extract_and_strip(&mut none),
            PromptCachingControl::default()
        );
        // Non-object extra_body: default, untouched.
        let mut arr = Some(json!([1, 2, 3]));
        assert_eq!(
            PromptCachingControl::extract_and_strip(&mut arr),
            PromptCachingControl::default()
        );
    }

    #[test]
    fn control_parse_is_non_mutating() {
        let extra = json!({ "prompt_caching": {"messages": true} });
        let ctrl = PromptCachingControl::parse(Some(&extra));
        assert_eq!(ctrl.messages, Some(true));
        assert_eq!(ctrl.system, None);
        // parse does not strip; the field is still present on the borrowed value.
        assert!(extra.get("prompt_caching").is_some());
    }

    #[test]
    fn control_precedence_per_request_beats_global() {
        // Per-request explicit wins over global default (bedrock.py:744-748).
        let ctrl = PromptCachingControl {
            system: Some(false),
            messages: Some(true),
            ttl: None,
        };
        // Global true, but request disables system.
        assert!(!ctrl.system_enabled(true));
        // Global false, but request enables messages.
        assert!(ctrl.messages_enabled(false));

        // Absent per-request ⇒ fall back to global.
        let empty = PromptCachingControl::default();
        assert!(empty.system_enabled(true));
        assert!(!empty.system_enabled(false));
        assert!(empty.messages_enabled(true));
    }

    // ---- supports_caching: config-driven capability gate ----

    #[test]
    fn supports_caching_from_config_nova_yes_claude_yes() {
        let c = caps();
        // Nova has max_cache_tokens=20000 in config ⇒ supported.
        assert!(supports_caching(NOVA, &c));
        // Claude entries now declare cache_min_tokens=1024 in config ⇒ supported.
        assert!(supports_caching(CLAUDE_NO_CACHE, &c));
        // Unknown model ⇒ not supported.
        assert!(!supports_caching("vendor.unknown-v1:0", &c));
    }

    #[test]
    fn nova_max_cache_tokens_read_from_config() {
        // The Nova 20000 ceiling is config-sourced, not hardcoded here.
        let c = caps();
        assert_eq!(c.max_cache_tokens(NOVA), Some(20_000));
    }

    #[test]
    fn nova_cache_floor_applies_from_config() {
        let c = caps();
        assert_eq!(c.cache_min_tokens(NOVA), Some(1024));

        // Below the 1024-token floor (estimate = words * 1.3) ⇒ no cachePoint.
        let small = decorate_system_blocks(sys(&[&long_text(100)]), NOVA, &c, true, None);
        let small_arr = small.as_array().unwrap();
        assert_eq!(small_arr.len(), 1, "below Nova floor ⇒ no cachePoint");
        assert!(small_arr[0].get("cachePoint").is_none());

        // Above the floor ⇒ cachePoint injected.
        let large = decorate_system_blocks(sys(&[&long_text(2000)]), NOVA, &c, true, None);
        let large_arr = large.as_array().unwrap();
        assert_eq!(large_arr.len(), 2, "above Nova floor ⇒ cachePoint");
        assert_eq!(large_arr[1], cache_point(None));
    }

    // ---- decorate_system_blocks ----

    #[test]
    fn system_cachepoint_inserted_when_enabled_supported_over_min() {
        // min=4 tokens; supply ~13 estimated tokens (10 words * 1.3).
        let caps = FakeCaps::supports(Some(4), Some(20_000), Some(4));
        let blocks = sys(&[&long_text(10)]);
        let out = decorate_system_blocks(blocks, "m", &caps, true, None);
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 2, "text block + cachePoint");
        assert!(arr[0]["text"].is_string());
        assert_eq!(arr[1], cache_point(None));
    }

    #[test]
    fn system_cachepoint_not_inserted_when_under_min() {
        // min very high; a single short word can't clear it.
        let caps = FakeCaps::supports(Some(100_000), None, None);
        let blocks = sys(&["hi"]);
        let out = decorate_system_blocks(blocks, "m", &caps, true, None);
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 1, "below min ⇒ no cachePoint");
        assert!(arr[0].get("cachePoint").is_none());
    }

    #[test]
    fn system_cachepoint_inserted_over_max_warn_only() {
        // Over max_cache_tokens still inserts (Python parity: warn-only).
        let caps = FakeCaps::supports(Some(1), Some(2), Some(4));
        let blocks = sys(&[&long_text(50)]); // ~65 tokens >> max 2
        let out = decorate_system_blocks(blocks, "m", &caps, true, None);
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1], cache_point(None));
    }

    #[test]
    fn system_not_inserted_for_unsupported_model() {
        // No cache params ⇒ supports_caching == false ⇒ unchanged.
        let caps = FakeCaps::supports(None, None, None);
        let blocks = sys(&[&long_text(100)]);
        let out = decorate_system_blocks(blocks.clone(), "m", &caps, true, None);
        assert_eq!(out, blocks, "unsupported model: no cachePoint");
    }

    #[test]
    fn system_not_inserted_when_disabled() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let blocks = sys(&[&long_text(100)]);
        let out = decorate_system_blocks(blocks.clone(), "m", &caps, false, None);
        assert_eq!(out, blocks, "disabled: no cachePoint");
    }

    #[test]
    fn system_empty_is_unchanged() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let blocks = Value::Array(vec![]);
        let out = decorate_system_blocks(blocks.clone(), "m", &caps, true, None);
        assert_eq!(out, blocks);
    }

    #[test]
    fn system_no_min_configured_inserts_when_enabled() {
        // cache_min_tokens None but max present ⇒ supported, no floor to clear.
        let caps = FakeCaps::supports(None, Some(20_000), Some(4));
        let blocks = sys(&["short"]);
        let out = decorate_system_blocks(blocks, "m", &caps, true, None);
        assert_eq!(out.as_array().unwrap().len(), 2);
    }

    // ---- decorate_messages ----

    fn user_turn(text: &str) -> Value {
        json!({ "role": "user", "content": [{ "text": text }] })
    }
    fn assistant_turn(text: &str) -> Value {
        json!({ "role": "assistant", "content": [{ "text": text }] })
    }

    #[test]
    fn messages_cachepoint_appended_to_last_user_turn() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![
            user_turn("first"),
            assistant_turn("reply"),
            user_turn("second"),
        ]);
        let out = decorate_messages(messages, "m", &caps, true, 0, caps.checkpoints(), None);
        let turns = out.as_array().unwrap();
        // Last user turn gets the cachePoint appended.
        let last_content = turns[2]["content"].as_array().unwrap();
        assert_eq!(last_content.len(), 2);
        assert_eq!(last_content[1], cache_point(None));
        // Earlier user turn is untouched.
        assert_eq!(turns[0]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn messages_not_inserted_for_unsupported_model() {
        let caps = FakeCaps::supports(None, None, None);
        let messages = Value::Array(vec![user_turn("hi")]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            true,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages);
    }

    #[test]
    fn messages_not_inserted_when_disabled() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![user_turn("hi")]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            false,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages);
    }

    #[test]
    fn messages_respect_checkpoint_ceiling() {
        // already_used == max_cache_checkpoints ⇒ no room.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(2));
        let messages = Value::Array(vec![user_turn("hi")]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            true,
            2,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages, "no room left under the checkpoint ceiling");

        // One under the ceiling ⇒ inserts.
        let out2 = decorate_messages(messages, "m", &caps, true, 1, caps.checkpoints(), None);
        let turns = out2.as_array().unwrap();
        assert_eq!(turns[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn messages_skip_tool_result_only_turn() {
        // A user turn whose content is only toolResult must NOT get a cachePoint
        // (bedrock.py:1018-1019). With no other eligible user turn, unchanged.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![
            assistant_turn("calling tool"),
            json!({
                "role": "user",
                "content": [{ "toolResult": { "toolUseId": "t1", "content": [] } }]
            }),
        ]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            true,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages, "toolResult-only turn must be skipped");
    }

    #[test]
    fn messages_pick_prior_user_when_last_is_tool_result() {
        // Last user turn is toolResult-only ⇒ fall back to the prior normal user.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![
            user_turn("real question"),
            assistant_turn("calling tool"),
            json!({
                "role": "user",
                "content": [{ "toolResult": { "toolUseId": "t1", "content": [] } }]
            }),
        ]);
        let out = decorate_messages(messages, "m", &caps, true, 0, caps.checkpoints(), None);
        let turns = out.as_array().unwrap();
        // The normal user turn (index 0) receives the cachePoint.
        let first = turns[0]["content"].as_array().unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[1], cache_point(None));
        // The toolResult-only turn is untouched.
        assert_eq!(turns[2]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn messages_no_double_cachepoint() {
        // A user turn already ending in a cachePoint is not eligible again.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![json!({
            "role": "user",
            "content": [{ "text": "hi" }, cache_point(None)]
        })]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            true,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages, "must not append a second cachePoint");
    }

    #[test]
    fn messages_empty_unchanged() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![]);
        let out = decorate_messages(
            messages.clone(),
            "m",
            &caps,
            true,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(out, messages);
    }

    // ---- combined: control parse → strip → decorate, asserting absence ----

    #[test]
    fn end_to_end_control_stripped_and_decoration_applied() {
        // Simulate the gateway flow: parse+strip the control field, then use the
        // resolved decision to decorate. Assert the control field is ABSENT from
        // the (Bedrock-bound) extra_body afterwards.
        let mut extra = Some(json!({
            "anthropic_beta": ["x"],
            "prompt_caching": {"system": true, "messages": true}
        }));
        let ctrl = PromptCachingControl::extract_and_strip(&mut extra);

        // Control field gone — would-be Bedrock args do not carry it.
        let bedrock_extra = extra.unwrap();
        assert!(bedrock_extra.get("prompt_caching").is_none());
        assert!(bedrock_extra.get("anthropic_beta").is_some());

        // Global default false, but per-request enables both.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let system = decorate_system_blocks(
            sys(&[&long_text(10)]),
            "m",
            &caps,
            ctrl.system_enabled(false),
            None,
        );
        assert_eq!(system.as_array().unwrap().len(), 2);

        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &caps,
            ctrl.messages_enabled(false),
            1, // one checkpoint already spent on system
            caps.checkpoints(),
            None,
        );
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
    }

    // ---- T4: master-switch default semantics (provider.rs global_default) ----

    fn resolved(ctrl: PromptCachingControl, master_on: bool) -> (bool, bool) {
        (
            ctrl.system_enabled(master_on),
            ctrl.messages_enabled(master_on),
        )
    }

    #[test]
    fn t4_master_on_no_extra_body_auto_injects_system_and_messages() {
        let ctrl = PromptCachingControl::parse(None);
        let (sys_on, msg_on) = resolved(ctrl, true);

        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, sys_on, None);
        assert_eq!(
            system.as_array().unwrap().len(),
            2,
            "master on + no extra_body ⇒ system cachePoint"
        );

        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &caps,
            msg_on,
            count_cp(&system),
            caps.checkpoints(),
            None,
        );
        assert_eq!(
            messages[0]["content"].as_array().unwrap().len(),
            2,
            "master on + no extra_body ⇒ messages cachePoint"
        );
    }

    #[test]
    fn t4_extra_body_messages_false_disables_messages_only() {
        let extra = json!({ "prompt_caching": { "messages": false } });
        let ctrl = PromptCachingControl::parse(Some(&extra));
        let (sys_on, msg_on) = resolved(ctrl, true);

        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, sys_on, None);
        assert_eq!(
            system.as_array().unwrap().len(),
            2,
            "system still cached (override only touches messages)"
        );

        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &caps,
            msg_on,
            count_cp(&system),
            caps.checkpoints(),
            None,
        );
        assert_eq!(
            messages[0]["content"].as_array().unwrap().len(),
            1,
            "messages=false override ⇒ no messages cachePoint"
        );
    }

    #[test]
    fn t4_unsupported_model_zero_cachepoints_even_with_master_on() {
        let ctrl = PromptCachingControl::parse(None);
        let (sys_on, msg_on) = resolved(ctrl, true);

        let caps = FakeCaps::supports(None, None, None);
        let system = decorate_system_blocks(sys(&[&long_text(100)]), "m", &caps, sys_on, None);
        assert_eq!(
            count_cp(&system),
            0,
            "unsupported model ⇒ no system cachePoint"
        );

        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &caps,
            msg_on,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(
            count_cp(&messages),
            0,
            "unsupported model ⇒ no messages cachePoint"
        );
    }

    #[test]
    fn t4_master_off_no_extra_body_zero_cachepoints_supported_model() {
        let ctrl = PromptCachingControl::parse(None);
        let (sys_on, msg_on) = resolved(ctrl, false);

        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let system = decorate_system_blocks(sys(&[&long_text(100)]), "m", &caps, sys_on, None);
        assert_eq!(count_cp(&system), 0, "master off ⇒ no system cachePoint");

        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &caps,
            msg_on,
            0,
            caps.checkpoints(),
            None,
        );
        assert_eq!(
            count_cp(&messages),
            0,
            "master off ⇒ no messages cachePoint"
        );

        let extra = json!({ "prompt_caching": { "system": true } });
        let forced = PromptCachingControl::parse(Some(&extra));
        let system_forced = decorate_system_blocks(
            sys(&[&long_text(100)]),
            "m",
            &caps,
            forced.system_enabled(false),
            None,
        );
        assert_eq!(
            count_cp(&system_forced),
            1,
            "master off but extra_body forces system on"
        );
    }

    fn count_cp(value: &Value) -> u32 {
        match value {
            Value::Array(arr) => arr
                .iter()
                .map(|item| match item {
                    Value::Object(o) if o.contains_key("cachePoint") => 1,
                    _ => item.get("content").map(count_cp).unwrap_or(0),
                })
                .sum(),
            _ => 0,
        }
    }

    // ---- T6: decorate_tools + shared tools→system→messages budget ----

    fn tool_specs(names: &[&str]) -> Value {
        Value::Array(
            names
                .iter()
                .map(|n| {
                    json!({
                        "toolSpec": {
                            "name": n,
                            "inputSchema": { "json": { "type": "object", "properties": {} } }
                        }
                    })
                })
                .collect(),
        )
    }

    fn tools_tail_is_cache_point(tools: &Value) -> bool {
        tools
            .as_array()
            .and_then(|a| a.last())
            .and_then(Value::as_object)
            .is_some_and(|o| o.contains_key("cachePoint"))
    }

    /// Drive the full assemble() zone order (tools → system → messages) against
    /// the pure decorators with ONE shared running budget, exactly as
    /// `provider.rs::assemble` does. Returns each decorated zone plus the grand
    /// total of cachePoints placed.
    #[allow(clippy::too_many_arguments)]
    fn run_three_zones(
        tools: Value,
        system: Value,
        messages: Value,
        model: &str,
        caps: &dyn ModelCapabilities,
        enabled: bool,
        max_cp: Option<u32>,
        ttl: Option<&str>,
    ) -> (Value, Value, Value, u32) {
        let mut used: u32 = 0;

        let dt = decorate_tools(tools, model, caps, enabled, used, max_cp, ttl);
        used += count_cp(&dt);

        let ds = decorate_system_blocks(system, model, caps, enabled, ttl);
        used += count_cp(&ds);

        let dm = decorate_messages(messages, model, caps, enabled, used, max_cp, ttl);
        let grand = count_cp(&dt) + count_cp(&ds) + count_cp(&dm);
        (dt, ds, dm, grand)
    }

    #[test]
    fn t6_tools_system_history_budget4_tail_cp_and_total_le_4() {
        // tools + system + long history, max_checkpoints=4 → ≤4 total AND tools
        // tail carries a cachePoint.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![
            user_turn("first long turn"),
            assistant_turn("reply"),
            user_turn(&long_text(50)),
        ]);
        let (dt, ds, dm, grand) = run_three_zones(
            tool_specs(&["get_weather", "search"]),
            sys(&[&long_text(10)]),
            messages,
            "m",
            &caps,
            true,
            Some(4),
            None,
        );
        assert!(grand <= 4, "grand total {grand} must be ≤ 4");
        assert!(
            tools_tail_is_cache_point(&dt),
            "tools array tail must carry a cachePoint"
        );
        // tools(1) + system(1) + messages(1) all fit under 4.
        assert_eq!(count_cp(&dt), 1, "one tools cachePoint");
        assert_eq!(count_cp(&ds), 1, "one system cachePoint");
        assert_eq!(count_cp(&dm), 1, "one messages cachePoint");
        assert_eq!(grand, 3);
    }

    #[test]
    fn t6_no_tools_system_messages_behavior_unchanged_regression() {
        // NO tools → no tools cachePoint; system/messages identical to the
        // pre-T6 (system_checkpoints-seeded) behavior.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![user_turn("q")]);

        // T6 path: empty tools array (no toolConfig) + shared budget.
        let (dt, ds_new, dm_new, grand) = run_three_zones(
            Value::Array(vec![]),
            sys(&[&long_text(10)]),
            messages.clone(),
            "m",
            &caps,
            true,
            Some(4),
            None,
        );
        assert_eq!(count_cp(&dt), 0, "no tools ⇒ zero tools cachePoint");

        // Legacy path (pre-T6): system decorated, then messages seeded with the
        // system checkpoint count.
        let ds_old = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, true, None);
        let sys_cp = count_cp(&ds_old);
        let dm_old = decorate_messages(messages, "m", &caps, true, sys_cp, Some(4), None);

        assert_eq!(ds_new, ds_old, "system behavior must be unchanged");
        assert_eq!(dm_new, dm_old, "messages behavior must be unchanged");
        assert_eq!(grand, count_cp(&ds_old) + count_cp(&dm_old));
    }

    #[test]
    fn t6_budget2_places_exactly_two_tools_and_system_messages_skipped() {
        // max_checkpoints=2 → exactly 2 placed (tools + system); messages skipped
        // because the budget is exhausted by the time we reach zone 3.
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(2));
        let messages = Value::Array(vec![user_turn(&long_text(20))]);
        let (dt, ds, dm, grand) = run_three_zones(
            tool_specs(&["t1"]),
            sys(&[&long_text(10)]),
            messages,
            "m",
            &caps,
            true,
            Some(2),
            None,
        );
        assert_eq!(grand, 2, "exactly two cachePoints under a budget of 2");
        assert_eq!(count_cp(&dt), 1, "tools consumes slot 1");
        assert_eq!(count_cp(&ds), 1, "system consumes slot 2");
        assert_eq!(
            count_cp(&dm),
            0,
            "messages skipped: budget exhausted before zone 3"
        );
        assert!(tools_tail_is_cache_point(&dt));
    }

    #[test]
    fn t6_unsupported_or_disabled_zero_cachepoints_anywhere() {
        let messages = Value::Array(vec![user_turn(&long_text(20))]);

        // (a) Unsupported model: zero cache params ⇒ zero cachePoints in all
        // three zones, including tools.
        let unsupported = FakeCaps::supports(None, None, None);
        let (dt, ds, dm, grand) = run_three_zones(
            tool_specs(&["t1", "t2"]),
            sys(&[&long_text(50)]),
            messages.clone(),
            "m",
            &unsupported,
            true,
            Some(4),
            None,
        );
        assert_eq!(grand, 0, "unsupported model ⇒ zero cachePoints");
        assert_eq!(count_cp(&dt), 0);
        assert_eq!(count_cp(&ds), 0);
        assert_eq!(count_cp(&dm), 0);
        assert!(!tools_tail_is_cache_point(&dt));

        // (b) Caching disabled (all zone flags false) on a SUPPORTED model ⇒
        // still zero cachePoints anywhere, including tools.
        let supported = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let (dt2, ds2, dm2, grand2) = run_three_zones(
            tool_specs(&["t1", "t2"]),
            sys(&[&long_text(50)]),
            messages,
            "m",
            &supported,
            false,
            Some(4),
            None,
        );
        assert_eq!(grand2, 0, "disabled ⇒ zero cachePoints");
        assert_eq!(count_cp(&dt2), 0);
        assert_eq!(count_cp(&ds2), 0);
        assert_eq!(count_cp(&dm2), 0);
        assert!(!tools_tail_is_cache_point(&dt2));
    }

    #[test]
    fn t6_decorate_tools_empty_and_double_insert_guard() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        // Empty tools ⇒ unchanged.
        let empty = decorate_tools(Value::Array(vec![]), "m", &caps, true, 0, Some(4), None);
        assert_eq!(empty, Value::Array(vec![]));

        // Already ends with a cachePoint ⇒ no double-insert.
        let already = Value::Array(vec![
            json!({ "toolSpec": { "name": "t", "inputSchema": { "json": {} } } }),
            cache_point(None),
        ]);
        let out = decorate_tools(already.clone(), "m", &caps, true, 0, Some(4), None);
        assert_eq!(out, already, "must not append a second tools cachePoint");

        // Budget exhausted ⇒ unchanged.
        let full = decorate_tools(tool_specs(&["t"]), "m", &caps, true, 4, Some(4), None);
        assert_eq!(count_cp(&full), 0, "no room under the ceiling");
    }

    // ---- PR-G: cachePoint.ttl (5m/1h) ----

    /// A test-only capabilities impl that reports `Capability::CacheTtl1h` for a
    /// model, so the 1h gate can be exercised in a `.rs` test WITHOUT naming a
    /// model in production code (zero-hardcoding: the flag is the gate).
    struct TtlCaps {
        supports_1h: bool,
    }

    impl ModelCapabilities for TtlCaps {
        fn has(&self, _model: &str, cap: Capability) -> bool {
            matches!(cap, Capability::CacheTtl1h) && self.supports_1h
        }
        fn resolve_foundation(&self, m: &str) -> String {
            m.to_string()
        }
        fn budget_ratios(&self, _model: &str) -> Option<crate::domain::BudgetRatios> {
            None
        }
        fn min_budget_tokens(&self, _model: &str) -> Option<u32> {
            None
        }
        fn max_cache_tokens(&self, _model: &str) -> Option<u32> {
            Some(20_000)
        }
        fn cache_min_tokens(&self, _model: &str) -> Option<u32> {
            Some(1)
        }
        fn max_cache_checkpoints(&self, _model: &str) -> Option<u32> {
            Some(4)
        }
        fn beta_headers(&self, _model: &str) -> Vec<String> {
            Vec::new()
        }
        fn reasoning_path(&self, _model: &str) -> crate::domain::ReasoningPath {
            crate::domain::ReasoningPath::None
        }
        fn responses_backend(&self, _model: &str) -> crate::domain::ResponsesBackend {
            crate::domain::ResponsesBackend::Converse
        }
        fn model_regions(&self, _model: &str) -> Option<Vec<String>> {
            None
        }
    }

    #[test]
    fn cache_point_emits_ttl_when_present() {
        assert_eq!(
            cache_point(Some("1h")),
            json!({ "cachePoint": { "type": "default", "ttl": "1h" } })
        );
        assert_eq!(
            cache_point(Some("5m")),
            json!({ "cachePoint": { "type": "default", "ttl": "5m" } })
        );
        // None ⇒ NO ttl key: byte-identical to the historical output.
        assert_eq!(
            cache_point(None),
            json!({ "cachePoint": { "type": "default" } })
        );
        let none = serde_json::to_string(&cache_point(None)).unwrap();
        assert!(!none.contains("ttl"), "None must not emit a ttl key");
    }

    #[test]
    fn decorate_system_blocks_threads_ttl() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let out = decorate_system_blocks(sys(&[&long_text(10)]), "m", &caps, true, Some("1h"));
        let arr = out.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1], cache_point(Some("1h")));
        assert_eq!(arr[1]["cachePoint"]["ttl"], "1h");
    }

    #[test]
    fn decorate_tools_threads_ttl() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let out = decorate_tools(
            tool_specs(&["t1"]),
            "m",
            &caps,
            true,
            0,
            Some(4),
            Some("1h"),
        );
        assert!(tools_tail_is_cache_point(&out));
        let tail = out.as_array().unwrap().last().unwrap();
        assert_eq!(tail, &cache_point(Some("1h")));
        assert_eq!(tail["cachePoint"]["ttl"], "1h");
    }

    #[test]
    fn decorate_messages_threads_ttl() {
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![user_turn("q")]);
        let out = decorate_messages(
            messages,
            "m",
            &caps,
            true,
            0,
            caps.checkpoints(),
            Some("1h"),
        );
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[1], cache_point(Some("1h")));
        assert_eq!(content[1]["cachePoint"]["ttl"], "1h");
    }

    #[test]
    fn prompt_caching_ttl_parsed_from_extra_body() {
        let mut extra = Some(json!({
            "thinking": {"type": "enabled"},
            "prompt_caching": {"system": true, "ttl": "1h"}
        }));
        let ctrl = PromptCachingControl::extract_and_strip(&mut extra);
        assert_eq!(ctrl.ttl.as_deref(), Some("1h"));
        assert_eq!(ctrl.system, Some(true));

        // The control field is stripped; the Bedrock-bound field survives.
        let obj = extra.expect("extra retained");
        assert!(
            obj.get("prompt_caching").is_none(),
            "prompt_caching (incl. ttl) must be stripped from extra_body"
        );
        assert!(obj.get("thinking").is_some());

        // Absent ttl ⇒ None (unchanged behavior).
        let mut no_ttl = Some(json!({ "prompt_caching": {"system": true} }));
        let ctrl2 = PromptCachingControl::extract_and_strip(&mut no_ttl);
        assert_eq!(ctrl2.ttl, None);
    }

    #[test]
    fn one_hour_downgraded_to_5m_on_unsupported_model() {
        // Model LACKS cache_ttl_1h: a requested 1h resolves to 5m + downgrade flag.
        let unsupported = TtlCaps { supports_1h: false };
        let resolved = resolve_cache_ttl(Some("1h"), "5m", "m", &unsupported);
        assert_eq!(resolved.effective, "5m");
        assert_eq!(resolved.requested, "1h");
        assert!(
            resolved.downgraded,
            "1h on unsupported model must downgrade"
        );

        // The emitted cachePoints (all zones) carry 5m — assert via decorated args.
        let ttl = Some(resolved.effective.as_str());
        let system = decorate_system_blocks(sys(&[&long_text(10)]), "m", &unsupported, true, ttl);
        assert_eq!(system.as_array().unwrap()[1]["cachePoint"]["ttl"], "5m");
        let messages = decorate_messages(
            Value::Array(vec![user_turn("q")]),
            "m",
            &unsupported,
            true,
            0,
            Some(4),
            ttl,
        );
        assert_eq!(
            messages[0]["content"].as_array().unwrap()[1]["cachePoint"]["ttl"],
            "5m"
        );

        // Model SUPPORTS cache_ttl_1h: 1h is honored, no downgrade.
        let supported = TtlCaps { supports_1h: true };
        let ok = resolve_cache_ttl(Some("1h"), "5m", "m", &supported);
        assert_eq!(ok.effective, "1h");
        assert!(!ok.downgraded);

        // 5m is always allowed on any caching-capable model (no downgrade).
        let five = resolve_cache_ttl(Some("5m"), "5m", "m", &unsupported);
        assert_eq!(five.effective, "5m");
        assert!(!five.downgraded);
    }

    #[test]
    fn resolve_cache_ttl_precedence_per_request_over_default() {
        let supported = TtlCaps { supports_1h: true };
        // Per-request override wins over the settings default.
        let r = resolve_cache_ttl(Some("1h"), "5m", "m", &supported);
        assert_eq!(r.effective, "1h");
        // No per-request ttl ⇒ settings default applies.
        let d = resolve_cache_ttl(None, "5m", "m", &supported);
        assert_eq!(d.effective, "5m");
        let d1h = resolve_cache_ttl(None, "1h", "m", &supported);
        assert_eq!(d1h.effective, "1h");
    }

    #[test]
    fn ttl_is_uniform_across_all_zones_in_one_request() {
        // Every cachePoint placed across tools → system → messages carries the
        // SAME ttl (no longer-before-shorter ordering).
        let caps = FakeCaps::supports(Some(1), Some(20_000), Some(4));
        let messages = Value::Array(vec![
            user_turn("first"),
            assistant_turn("reply"),
            user_turn(&long_text(50)),
        ]);
        let (dt, ds, dm, grand) = run_three_zones(
            tool_specs(&["get_weather", "search"]),
            sys(&[&long_text(10)]),
            messages,
            "m",
            &caps,
            true,
            Some(4),
            Some("1h"),
        );
        assert_eq!(grand, 3);
        let ttls = collect_ttls(&dt)
            .into_iter()
            .chain(collect_ttls(&ds))
            .chain(collect_ttls(&dm))
            .collect::<Vec<_>>();
        assert_eq!(ttls.len(), 3, "one ttl per zone");
        assert!(
            ttls.iter().all(|t| t == "1h"),
            "all cachePoint ttls must be uniform (1h), got {ttls:?}"
        );
    }

    /// Recursively gather every `cachePoint.ttl` value present in a decorated
    /// zone (arrays of blocks or message turns with nested `content` arrays).
    fn collect_ttls(value: &Value) -> Vec<String> {
        let mut out = Vec::new();
        if let Value::Array(arr) = value {
            for item in arr {
                if let Some(ttl) = item
                    .get("cachePoint")
                    .and_then(|cp| cp.get("ttl"))
                    .and_then(Value::as_str)
                {
                    out.push(ttl.to_string());
                }
                if let Some(content) = item.get("content") {
                    out.extend(collect_ttls(content));
                }
            }
        }
        out
    }

    /// Live integration test — skipped unless `BEDROCK_INTEGRATION` is set.
    /// Proves (C1/A3) that Nova ACCEPTS an explicit cachePoint and reports a
    /// real cache hit: a large system prefix sent twice yields
    /// cacheRead > 0 on the second identical call, with no error.
    /// Bearer auth via `BEDROCK_API_KEY`, region us-east-2,
    /// `us.amazon.nova-lite-v1:0`.
    #[tokio::test]
    #[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
    async fn nova_live_cache_read_on_second_call() {
        use aws_sdk_bedrockruntime::types::{
            CachePointBlock, CachePointType, ContentBlock, ConversationRole, Message,
            SystemContentBlock,
        };

        if std::env::var("BEDROCK_INTEGRATION").is_err() {
            return;
        }

        let settings = crate::config::AppSettings::load().expect("settings load");
        let clients = crate::bedrock::client::BedrockClients::from_settings(&settings).await;

        // A large, byte-stable prefix that comfortably clears the Nova floor.
        let prefix = vec!["The quick brown fox jumps over the lazy dog."; 400].join(" ");
        let system = vec![
            SystemContentBlock::Text(prefix),
            SystemContentBlock::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .expect("cache point"),
            ),
        ];
        let user = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(
                "Reply with the single word: ok".to_string(),
            ))
            .build()
            .expect("user message");

        let call = || async {
            clients
                .runtime
                .converse()
                .model_id("us.amazon.nova-lite-v1:0")
                .set_system(Some(system.clone()))
                .messages(user.clone())
                .customize()
                .config_override(crate::bedrock::client::region_config_override("us-east-2"))
                .send()
                .await
        };

        let first = call().await.expect("first Nova converse should succeed");
        let _ = first
            .usage()
            .map(|u| u.cache_write_input_tokens())
            .expect("usage present on first call");

        let second = call().await.expect("second Nova converse should succeed");
        let cache_read = second
            .usage()
            .and_then(|u| u.cache_read_input_tokens())
            .expect("usage present on second call");

        assert!(
            cache_read > 0,
            "second identical Nova call must report cacheRead > 0, got {cache_read}"
        );
    }
}
