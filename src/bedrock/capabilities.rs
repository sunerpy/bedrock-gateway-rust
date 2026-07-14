//! Config-driven model capability resolver.
//!
//! This is the RUNTIME ENGINE for model capability detection. It implements the
//! [`crate::domain::ModelCapabilities`] trait over the loaded
//! [`crate::config::ModelCapabilityConfig`] (task 4's `config/models.toml`).
//!
//! DE-HARDCODING CONTRACT (mirrors `src/config/capabilities.rs`):
//! This file contains ONLY the matching/resolution ALGORITHM. It MUST NOT
//! contain any model IDs, capability flag literals, or per-model magic numbers.
//! ALL such DATA lives in `config/models.toml` and is reached exclusively via
//! `self.config`. Adding/tuning a model is a TOML-only edit.
//!
//! Algorithm provenance — `.legacy-python/src/api/models/bedrock.py`:
//! - 170-172 `_has_model_capability`: substring-match the lowercased resolved
//!   foundation model id against the capability table —
//!   `any(cap in caps for substr, caps if substr in model_lower)`.
//! - 397-417 `_resolve_to_foundation_model`: profile/ARN → underlying
//!   foundation model id via dictionary lookup; unknown ids pass through
//!   unchanged.
//! - 1130-1138 conflict detection usage (`temperature_topp_conflict`).
//! - 454-477 `_get_max_cache_tokens` (Nova 20K ceiling, sourced from config).
//! - 1679-1689 `_calc_budget_tokens` (budget ratios, sourced from config).

use std::collections::HashMap;

use crate::config::capabilities::ModelEntry;
use crate::config::{BudgetRatios, Capability, ModelCapabilityConfig, ReasoningPath};
use crate::domain::{ModelCapabilities, ResponsesBackend};

/// The exact-key name of the fallback entry in `config/models.toml` that
/// supplies default tunables (e.g. budget ratios) when a matched model entry
/// does not override them. This is a CONFIG KEY, not a model id — it is the
/// name the data file uses to label its own defaults row.
const DEFAULT_ENTRY_KEY: &str = "default";

/// Geo/cross-region inference-profile prefixes, normalized away on the
/// capability-MATCHING path only (NOT resolve_foundation; does NOT affect the
/// model id sent to Bedrock — that keeps its original prefix).
///
/// This is the COMPLETE set as of 2026-06-20, obtained by scanning all 34 AWS
/// commercial regions:
///   aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED
/// → prefixes: us. global. eu. apac. jp. au. ca.
/// If AWS introduces a new geographic prefix, ADD it here. (A missing prefix
/// only degrades capability matching for that prefixed form — cache/reasoning
/// lookups fall back; the actual Bedrock call still works since the original
/// prefixed id is sent unchanged.)
const GEO_PREFIXES: &[&str] = &["us.", "global.", "eu.", "apac.", "jp.", "au.", "ca."];

/// Lowercase `id` and strip a single leading cross-region geo prefix from the
/// enumerated [`GEO_PREFIXES`] set.
///
/// Used ONLY on the capability-matching path so all cross-region forms of a
/// model and the bare form substring-match the same `ModelEntry.match`. This
/// does NOT change `resolve_foundation` — it is a match-time-only comparison
/// form.
///
/// Algorithm: lowercase, then strip the first matching prefix from
/// [`GEO_PREFIXES`] if present; otherwise return the lowercased id unchanged.
/// Bare vendor forms (`amazon.nova-…`, `deepseek.v3-…`) are naturally safe —
/// `amazon.`/`deepseek.` are not in [`GEO_PREFIXES`], so `strip_prefix` never
/// matches and the id passes through.
///
/// `pub(crate)` so the cache safety net keys its negative cache on the SAME
/// normalized form the capability/cache layers match on.
pub(crate) fn normalize_for_match(id: &str) -> String {
    let lower = id.to_lowercase();
    for prefix in GEO_PREFIXES {
        if let Some(stripped) = lower.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    lower
}

/// Build the `from → to` alias lookup from a loaded config's `[[alias]]` tables.
fn alias_map(config: &ModelCapabilityConfig) -> HashMap<String, String> {
    config
        .aliases
        .iter()
        .map(|a| (a.from.clone(), a.to.clone()))
        .collect()
}

/// Config-driven implementation of [`ModelCapabilities`].
///
/// Wraps a loaded [`ModelCapabilityConfig`] plus a profile→foundation map. All
/// capability decisions are read from the wrapped config using the Python-parity
/// substring algorithm; no model knowledge is baked into this type.
///
/// The `profile_map` maps an inference-profile id / ARN to its underlying
/// foundation model id. It is populated at runtime (task 23's model listing);
/// constructing without it (an empty map) is valid and means "no profiles known
/// yet — pass ids through unchanged".
#[derive(Debug, Clone)]
pub struct ConfigModelCapabilities {
    /// The externalized model-capability registry (task 4 data).
    config: ModelCapabilityConfig,
    /// Inference-profile-id/ARN → underlying foundation model id.
    profile_map: HashMap<String, String>,
    /// Client-facing model name → canonical resolved id, from the config's
    /// `[[alias]]` tables. Consulted BEFORE `profile_map` so an alias resolves
    /// without any runtime-seeded inference-profile catalog.
    aliases: HashMap<String, String>,
}

impl ConfigModelCapabilities {
    /// Construct from a loaded config with an empty profile map.
    pub fn new(config: ModelCapabilityConfig) -> Self {
        let aliases = alias_map(&config);
        Self {
            config,
            profile_map: HashMap::new(),
            aliases,
        }
    }

    /// Construct from a loaded config and a seeded profile→foundation map.
    pub fn with_profiles(
        config: ModelCapabilityConfig,
        profile_map: HashMap<String, String>,
    ) -> Self {
        let aliases = alias_map(&config);
        Self {
            config,
            profile_map,
            aliases,
        }
    }

    /// Borrow the wrapped configuration.
    pub fn config(&self) -> &ModelCapabilityConfig {
        &self.config
    }

    /// First entry whose `match` pattern is a SUBSTRING of `model_lower`.
    ///
    /// This is the per-entry half of the Python `_has_model_capability`
    /// predicate (`substr in model_lower`) reused for parameter lookups. The
    /// synthetic `default` entry is never returned here — it is consulted
    /// explicitly as a fallback by callers that want defaults.
    fn matching_entry(&self, model_lower: &str) -> Option<&ModelEntry> {
        let normalized = normalize_for_match(model_lower);
        self.config
            .models
            .iter()
            .filter(|e| e.match_pattern != DEFAULT_ENTRY_KEY)
            .find(|e| normalized.contains(e.match_pattern.as_str()))
    }
}

impl ModelCapabilities for ConfigModelCapabilities {
    fn has(&self, model: &str, cap: Capability) -> bool {
        // Resolve profile→foundation first, then normalize (lowercase + strip
        // any cross-region geo prefix) for matching (Python parity + C2).
        let resolved = self.resolve_foundation(model);
        let normalized = normalize_for_match(&resolved);

        // any(cap in caps for substr, caps in TABLE if substr in model_lower)
        // (bedrock.py:170-172). DATA comes entirely from self.config.
        self.config
            .models
            .iter()
            .filter(|e| e.match_pattern != DEFAULT_ENTRY_KEY)
            .filter(|e| normalized.contains(e.match_pattern.as_str()))
            .any(|e| e.has_capability(cap))
    }

    fn resolve_foundation(&self, model_or_profile: &str) -> String {
        // Aliases (config `[[alias]]`) win first; then the runtime profile map
        // (bedrock.py:415-417); unknown ids pass through unchanged.
        if let Some(canonical) = self.aliases.get(model_or_profile) {
            return canonical.clone();
        }
        self.profile_map
            .get(model_or_profile)
            .cloned()
            .unwrap_or_else(|| model_or_profile.to_string())
    }

    fn budget_ratios(&self, model: &str) -> Option<BudgetRatios> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();

        // Prefer a matched model's override; otherwise fall back to the
        // config-supplied `default` entry (bedrock.py:1679-1689 defaults).
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.budget_ratios)
            .or_else(|| {
                self.config
                    .entry_for_match(DEFAULT_ENTRY_KEY)
                    .and_then(|e| e.params.budget_ratios)
            })
    }

    fn min_budget_tokens(&self, model: &str) -> Option<u32> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.min_budget_tokens)
            .or_else(|| {
                self.config
                    .entry_for_match(DEFAULT_ENTRY_KEY)
                    .and_then(|e| e.params.min_budget_tokens)
            })
    }

    fn max_cache_tokens(&self, model: &str) -> Option<u32> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.max_cache_tokens)
    }

    fn cache_min_tokens(&self, model: &str) -> Option<u32> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.cache_min_tokens)
    }

    fn max_cache_checkpoints(&self, model: &str) -> Option<u32> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.max_cache_checkpoints)
    }

    fn beta_headers(&self, model: &str) -> Vec<String> {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.beta_headers.clone())
            .unwrap_or_default()
    }

    fn reasoning_path(&self, model: &str) -> ReasoningPath {
        let resolved = self.resolve_foundation(model);
        let model_lower = resolved.to_lowercase();
        self.matching_entry(&model_lower)
            .and_then(|e| e.params.reasoning_path)
            .unwrap_or(ReasoningPath::None)
    }

    fn responses_backend(&self, model: &str) -> ResponsesBackend {
        let canonical = self.resolve_foundation(model);
        match self
            .config
            .entry_for_match(&canonical)
            .and_then(|e| e.params.responses_backend.as_deref())
        {
            Some("mantle") => ResponsesBackend::Mantle,
            _ => ResponsesBackend::Converse,
        }
    }

    fn model_regions(&self, model: &str) -> Option<Vec<String>> {
        let canonical = self.resolve_foundation(model);
        self.config
            .entry_for_match(&canonical)
            .and_then(|e| e.params.available_regions.clone())
    }
}

#[cfg(test)]
#[path = "capabilities_tests.rs"]
mod tests;
