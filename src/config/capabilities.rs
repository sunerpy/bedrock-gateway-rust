//! Model-capability configuration schema + TOML loader.
//!
//! This module is the SCHEMA + LOADER for externalized model knowledge. It
//! replaces the Python `MODEL_CAPABILITIES` substring dict (see
//! `.legacy-python/src/api/models/bedrock.py` lines 147-157) and the scattered
//! per-model magic numbers (max_cache_tokens, budget ratios, reasoning paths)
//! with a single declarative `config/models.toml`.
//!
//! IMPORTANT — DE-HARDCODING CONTRACT:
//! This file defines ONLY the schema and a typed loader. It MUST NOT contain
//! any model IDs, capability flag values, or per-model magic numbers. ALL such
//! data lives in `config/models.toml`. Adding a new model is a TOML-only edit
//! and requires NO recompile.
//!
//! The matching / resolution ALGORITHM (substring vs exact, precedence, merge)
//! is intentionally NOT implemented here — that is a separate task. This module
//! exposes only simple typed access to the loaded registry.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single model capability flag.
///
/// Mirrors the Python flag set documented at
/// `.legacy-python/src/api/models/bedrock.py` lines 141-146. Serialized in TOML
/// as snake_case strings (e.g. `temperature_topp_conflict`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Drop `topP` when `temperature` is also present.
    TemperatureToppConflict,
    /// Append a user continuation prompt instead of an assistant prefill.
    NoAssistantPrefill,
    /// Only supports `thinking.type=adaptive` + `output_config.effort`
    /// (rejects legacy `reasoning_config` budget_tokens with HTTP 400).
    AdaptiveThinking,
    /// Drop BOTH `temperature` and `topP` from `inferenceConfig` — this model
    /// deprecates all sampling parameters and returns HTTP 400 if any
    /// non-default value is sent (Claude Opus 4.7+, Sonnet 5, Fable/Mythos 5).
    /// Stronger than [`Capability::TemperatureToppConflict`], which only drops
    /// `topP` and only when `temperature` is also present.
    DropSamplingParams,
    /// Auto-inject the 1M-context beta header.
    #[serde(rename = "context_1m_beta")]
    Context1mBeta,
    /// This model supports native structured output via Bedrock Converse
    /// `outputConfig.textFormat` (grammar-constrained decoding). When set, the
    /// gateway honors an OpenAI `response_format` (`json_object` / `json_schema`)
    /// by emitting an `outputConfig`; when a model lacks this flag, a
    /// `response_format` request is rejected with HTTP 400 rather than silently
    /// ignored.
    StructuredOutput,
    /// This model supports a 1-hour prompt-cache `cachePoint.ttl`. When set, a
    /// per-request or configured `1h` cache TTL is honored; when a model lacks
    /// this flag, a requested `1h` TTL is silently downgraded to `5m` (which is
    /// always allowed for any caching-capable model). The gate is config-driven
    /// via this flag — never a model-name check in code.
    #[serde(rename = "cache_ttl_1h")]
    CacheTtl1h,
}

/// The strategy used to express reasoning/extended-thinking to Bedrock.
///
/// Mirrors the branches in
/// `.legacy-python/src/api/models/bedrock.py` lines 1168-1189.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningPath {
    /// `additionalModelRequestFields.thinking = {type: "adaptive"}` +
    /// `output_config.effort` (bedrock.py:1168-1172).
    AdaptiveThinking,
    /// `reasoning_config = {type: "enabled", budget_tokens: N}` where `N` is
    /// derived from `budget_ratios` (bedrock.py:1173-1177).
    BudgetTokens,
    /// DeepSeek v3 string form: `reasoning_config = "low"|"medium"|"high"`
    /// (bedrock.py:1178-1185).
    DeepseekString,
    /// Reasoning effort is ignored / unsupported (bedrock.py:1186-1189).
    None,
}

/// Budget-token ratios per reasoning effort level.
///
/// Defaults documented at `.legacy-python/src/api/models/bedrock.py`
/// lines 1679-1689: low = 30%, medium = 60%, high/xhigh/max = `max_tokens - 1`.
/// The `high` field uses a sentinel of `-1.0` to mean "max_tokens - 1" (the
/// algorithm in the resolution task interprets the sentinel); any non-negative
/// value is treated as a literal ratio.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BudgetRatios {
    /// Ratio of `max_tokens` for `low` effort (Python default 0.3).
    pub low: f32,
    /// Ratio of `max_tokens` for `medium` effort (Python default 0.6).
    pub medium: f32,
    /// Ratio for `high`/`xhigh`/`max` effort. Sentinel `-1.0` means
    /// `max_tokens - 1` (Python default behavior).
    pub high: f32,
}

/// Per-model tunable parameters. All optional; `None` means "not specified for
/// this entry" and the resolution layer falls back to defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelParams {
    /// Maximum cacheable tokens (e.g. Nova = 20000; bedrock.py:454,470).
    pub max_cache_tokens: Option<u32>,
    /// Minimum tokens required to enable prompt caching for this model
    /// (model-specific: 1024 or 4096).
    pub cache_min_tokens: Option<u32>,
    /// Maximum number of cache checkpoints supported (e.g. 4).
    pub max_cache_checkpoints: Option<u32>,
    /// Cache time-to-live, e.g. "5m" or "1h".
    pub cache_ttl: Option<String>,
    /// Beta headers to attach for this model.
    pub beta_headers: Option<Vec<String>>,
    /// Budget-token ratios for reasoning effort levels.
    pub budget_ratios: Option<BudgetRatios>,
    /// Minimum thinking `budget_tokens` floor for the `budget_tokens` reasoning
    /// path. The ratio-scaled budget is clamped UP to this value so a small
    /// `max_tokens` never produces a budget below the provider's hard minimum
    /// (Anthropic rejects `budget_tokens < 1024` with HTTP 400). Defaults to
    /// 1024 via the `default` entry when an entry does not override it.
    pub min_budget_tokens: Option<u32>,
    /// The reasoning strategy for this model.
    pub reasoning_path: Option<ReasoningPath>,
    /// Which backend serves this model's `/responses` requests. `None` means the
    /// default Bedrock Converse path; a value such as `"mantle"` routes the model
    /// to the bedrock-mantle OpenAI-compatible upstream instead. This is a CONFIG
    /// DATA flag — the resolution layer reads it; no model-name branching in code.
    pub responses_backend: Option<String>,
    /// Regions where this model is available. `None` means "no region gate"
    /// (available everywhere); a non-empty list restricts the model to those
    /// regions, enabling startup validation and per-request 400 on a mismatch.
    pub available_regions: Option<Vec<String>>,
}

/// A single declarative model entry from `config/models.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model-id substring (or exact id) matched against the lowercased resolved
    /// foundation model id (Python semantics: bedrock.py:140, 170-172).
    #[serde(rename = "match")]
    pub match_pattern: String,

    /// Capability flags enabled for this entry.
    #[serde(default)]
    pub capabilities: Vec<Capability>,

    /// Per-model tunable parameters.
    #[serde(default)]
    pub params: ModelParams,
}

impl ModelEntry {
    /// Convenience: does this entry declare the given capability?
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }
}

/// A single declarative model-id alias from `config/models.toml`.
///
/// Maps a client-facing model name (`from`) to a canonical resolved id (`to`).
/// The resolution layer consults aliases BEFORE the runtime profile map, so an
/// alias works without any seeded inference-profile catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelAlias {
    /// Client-facing model name to rewrite.
    pub from: String,
    /// Canonical resolved foundation/profile id the alias maps to.
    pub to: String,
}

/// The top-level model-capability registry, deserialized from
/// `config/models.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCapabilityConfig {
    /// Global 1M-context beta header value
    /// (Python `CONTEXT_1M_BETA_HEADER`, bedrock.py:159).
    #[serde(default = "default_context_1m_beta_header")]
    pub context_1m_beta_header: String,

    /// The list of model entries (TOML `[[model]]` tables).
    #[serde(default, rename = "model")]
    pub models: Vec<ModelEntry>,

    /// The list of model-id aliases (TOML `[[alias]]` tables). Consulted before
    /// the runtime profile map during foundation resolution.
    #[serde(default, rename = "alias")]
    pub aliases: Vec<ModelAlias>,

    /// Optional registry-level allow-list of model-id substrings (TOML
    /// top-level `allowed_models = [...]`). Applied at catalog-build time to
    /// filter both `GET /models` and `GET /models/{id}`. Overridden by the
    /// `ALLOWED_MODELS` env var when that is set. Empty ⇒ allow all. This is a
    /// registry-level list, NOT a per-model `Capability` flag.
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

/// Default for `context_1m_beta_header` when omitted from TOML.
/// The literal default value is documented in `config/models.toml`; this is the
/// schema-level fallback required by serde.
fn default_context_1m_beta_header() -> String {
    "context-1m-2025-08-07".to_string()
}

impl Default for ModelCapabilityConfig {
    fn default() -> Self {
        Self {
            context_1m_beta_header: default_context_1m_beta_header(),
            models: Vec::new(),
            aliases: Vec::new(),
            allowed_models: Vec::new(),
        }
    }
}

/// The model-capability TOML embedded into the binary at compile time.
///
/// `include_str!` resolves relative to THIS source file (`src/config/`), so
/// `../../config/models.toml` reaches the repo-root `config/models.toml`. This
/// guarantees the single binary ships with a non-empty default registry even
/// when no external `config/` directory is present at runtime — the root cause
/// fix for "binary copied elsewhere silently degrades every capability".
const EMBEDDED_MODELS_TOML: &str = include_str!("../../config/models.toml");

impl ModelCapabilityConfig {
    /// Load and parse a model-capability config from a TOML file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read model config file: {}", path.display()))?;
        Self::from_toml_str(&raw)
            .with_context(|| format!("failed to parse model config file: {}", path.display()))
    }

    /// Parse the compile-time-embedded model config.
    ///
    /// The embedded string is the repo's authored `config/models.toml`, so it is
    /// known-valid at build time. A parse failure here means the committed TOML
    /// is malformed — a build/release defect, not a runtime condition — hence the
    /// `expect`. This runs only at startup initialization, never on a request
    /// path.
    pub fn load_embedded() -> Self {
        Self::from_toml_str(EMBEDDED_MODELS_TOML)
            .expect("embedded config/models.toml must be valid")
    }

    /// Load with the external-over-embedded fallback strategy.
    ///
    /// Priority: a valid external file wins; otherwise the compile-time-embedded
    /// default is used. This NEVER degrades to an empty [`Self::default`] (which
    /// would disable every model capability) — that was the prior design flaw.
    ///
    /// - `Some(path)` that exists and parses → external (INFO log, with path).
    /// - `Some(path)` missing → embedded default (DEBUG log).
    /// - `Some(path)` present but invalid → embedded default (WARN log + error).
    /// - `None` → embedded default directly.
    pub fn load_with_fallback(external_path: Option<&Path>) -> Self {
        let Some(path) = external_path else {
            return Self::load_embedded();
        };
        if !path.exists() {
            tracing::debug!(
                path = %path.display(),
                "model-capability config file absent; using embedded default"
            );
            return Self::load_embedded();
        }
        match Self::load(path) {
            Ok(cfg) => {
                tracing::info!(
                    path = %path.display(),
                    "loaded external model-capability config"
                );
                cfg
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "external model-capability config invalid; using embedded default"
                );
                Self::load_embedded()
            }
        }
    }

    /// Parse a model-capability config from a TOML string.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let parsed: Self = toml::from_str(raw).context("invalid TOML for model capabilities")?;
        Ok(parsed)
    }

    /// Simple typed access: find the first entry whose `match` pattern equals
    /// the given key exactly. (This is exact-key lookup only — substring
    /// resolution is a separate task.)
    pub fn entry_for_match(&self, match_pattern: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|e| e.match_pattern == match_pattern)
    }

    /// The client-facing alias names whose canonical id is served by the mantle
    /// backend.
    ///
    /// For each `[[alias]]`, resolve its `to` (canonical id) against the model
    /// entries: if any entry whose `match` is a substring of the canonical id
    /// declares `responses_backend = "mantle"`, surface the alias `from`. These
    /// mantle models are absent from the Bedrock control-plane catalog, so the
    /// `/models` listing injects these bare alias names. Derived purely from
    /// config — no model-name string literals.
    pub fn mantle_alias_names(&self) -> Vec<String> {
        const MANTLE_BACKEND: &str = "mantle";
        self.aliases
            .iter()
            .filter(|alias| {
                self.models.iter().any(|entry| {
                    alias.to.contains(&entry.match_pattern)
                        && entry.params.responses_backend.as_deref() == Some(MANTLE_BACKEND)
                })
            })
            .map(|alias| alias.from.clone())
            .collect()
    }
}

#[cfg(test)]
#[path = "capabilities_tests.rs"]
mod tests;
