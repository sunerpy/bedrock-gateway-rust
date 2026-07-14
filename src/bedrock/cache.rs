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
#[path = "cache_tests.rs"]
mod tests;
