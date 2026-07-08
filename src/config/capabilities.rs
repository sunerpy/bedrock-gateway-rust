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
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Path to the project's authored model config, relative to the crate root.
    const MODELS_TOML: &str = "config/models.toml";

    fn load_project_config() -> ModelCapabilityConfig {
        ModelCapabilityConfig::load(MODELS_TOML).expect("config/models.toml must load and parse")
    }

    #[test]
    fn test_loads_project_models_toml() {
        let cfg = load_project_config();
        assert!(
            !cfg.models.is_empty(),
            "expected at least one [[model]] entry in {}",
            MODELS_TOML
        );
        // Global header is present (either authored or defaulted).
        assert!(!cfg.context_1m_beta_header.is_empty());
    }

    #[test]
    fn test_context_1m_beta_header_value() {
        let cfg = load_project_config();
        assert_eq!(cfg.context_1m_beta_header, "context-1m-2025-08-07");
    }

    #[test]
    fn test_opus_4_8_capabilities() {
        // Opus 4.7+ deprecate all sampling params (drop_sampling_params) on top
        // of the adaptive-thinking + no-prefill flags.
        let cfg = load_project_config();
        let entry = cfg
            .entry_for_match("claude-opus-4-8")
            .expect("claude-opus-4-8 entry must exist");
        let caps: HashSet<Capability> = entry.capabilities.iter().copied().collect();
        let expected: HashSet<Capability> = [
            Capability::NoAssistantPrefill,
            Capability::AdaptiveThinking,
            Capability::DropSamplingParams,
        ]
        .into_iter()
        .collect();
        assert_eq!(caps, expected);
    }

    #[test]
    fn test_sonnet_4_5_capabilities() {
        // Parity with Python MODEL_CAPABILITIES (bedrock.py:148):
        // "claude-sonnet-4-5": {"temperature_topp_conflict"}
        let cfg = load_project_config();
        let entry = cfg
            .entry_for_match("claude-sonnet-4-5")
            .expect("claude-sonnet-4-5 entry must exist");
        let caps: HashSet<Capability> = entry.capabilities.iter().copied().collect();
        let expected: HashSet<Capability> =
            [Capability::TemperatureToppConflict].into_iter().collect();
        assert_eq!(caps, expected);
    }

    #[test]
    fn test_sonnet_4_6_has_context_1m_beta() {
        // bedrock.py:149: {"temperature_topp_conflict", "context_1m_beta"}
        let cfg = load_project_config();
        let entry = cfg
            .entry_for_match("claude-sonnet-4-6")
            .expect("claude-sonnet-4-6 entry must exist");
        assert!(entry.has_capability(Capability::TemperatureToppConflict));
        assert!(entry.has_capability(Capability::Context1mBeta));
    }

    #[test]
    fn test_nova_max_cache_tokens() {
        // bedrock.py:454,470: Nova models have a 20,000 token caching limit.
        let cfg = load_project_config();
        let entry = cfg
            .entry_for_match("amazon.nova")
            .expect("amazon.nova entry must exist");
        assert_eq!(entry.params.max_cache_tokens, Some(20_000));
    }

    #[test]
    fn test_default_budget_ratios_present() {
        // bedrock.py:1679-1689: low=0.3, medium=0.6, high=max_tokens-1 (sentinel -1).
        let cfg = load_project_config();
        let entry = cfg
            .entry_for_match("default")
            .expect("default entry must exist for budget_ratios");
        let ratios = entry
            .params
            .budget_ratios
            .expect("default entry must define budget_ratios");
        assert_eq!(ratios.low, 0.3);
        assert_eq!(ratios.medium, 0.6);
        assert_eq!(ratios.high, -1.0);
    }

    #[test]
    fn test_reasoning_path_hints() {
        let cfg = load_project_config();

        // claude + adaptive_thinking → AdaptiveThinking (bedrock.py:1168-1172)
        let opus = cfg.entry_for_match("claude-opus-4-8").unwrap();
        assert_eq!(
            opus.params.reasoning_path,
            Some(ReasoningPath::AdaptiveThinking)
        );

        // claude non-adaptive → BudgetTokens (bedrock.py:1173-1177)
        let sonnet = cfg.entry_for_match("claude-sonnet-4-5").unwrap();
        assert_eq!(
            sonnet.params.reasoning_path,
            Some(ReasoningPath::BudgetTokens)
        );

        // deepseek.v3 → DeepseekString (bedrock.py:1178-1185)
        let deepseek = cfg.entry_for_match("deepseek.v3").unwrap();
        assert_eq!(
            deepseek.params.reasoning_path,
            Some(ReasoningPath::DeepseekString)
        );
    }

    #[test]
    fn test_extension_requires_toml_only() {
        // Proves a NEW model can be added by editing TOML alone — no code change,
        // no recompile of the schema. We append a [[model]] table to a copy of
        // the authored config and assert it resolves.
        let base = std::fs::read_to_string(MODELS_TOML).unwrap();
        let extended = format!(
            "{base}\n\n[[model]]\nmatch = \"vendor.future-model-99\"\ncapabilities = [\"adaptive_thinking\", \"context_1m_beta\"]\n\n[model.params]\nmax_cache_tokens = 12345\nreasoning_path = \"adaptive_thinking\"\n"
        );
        let cfg =
            ModelCapabilityConfig::from_toml_str(&extended).expect("extended config must parse");
        let entry = cfg
            .entry_for_match("vendor.future-model-99")
            .expect("newly added entry must resolve");
        assert!(entry.has_capability(Capability::AdaptiveThinking));
        assert!(entry.has_capability(Capability::Context1mBeta));
        assert_eq!(entry.params.max_cache_tokens, Some(12345));
        assert_eq!(
            entry.params.reasoning_path,
            Some(ReasoningPath::AdaptiveThinking)
        );
    }

    #[test]
    fn test_from_toml_str_minimal() {
        // Schema-level: a minimal entry with only `match` parses (capabilities
        // and params default to empty).
        let cfg = ModelCapabilityConfig::from_toml_str("[[model]]\nmatch = \"x.y\"\n").unwrap();
        let entry = cfg.entry_for_match("x.y").unwrap();
        assert!(entry.capabilities.is_empty());
        assert_eq!(entry.params, ModelParams::default());
        // Header defaults when omitted.
        assert_eq!(cfg.context_1m_beta_header, "context-1m-2025-08-07");
    }

    #[test]
    fn test_load_missing_file_errors() {
        let err = ModelCapabilityConfig::load("config/__does_not_exist__.toml");
        assert!(err.is_err());
    }

    #[test]
    fn test_load_embedded_is_non_empty_and_has_claude() {
        let cfg = ModelCapabilityConfig::load_embedded();
        assert!(
            !cfg.models.is_empty(),
            "embedded models.toml must be non-empty"
        );
        assert!(
            cfg.models
                .iter()
                .any(|m| m.match_pattern.contains("claude")),
            "embedded registry must include a Claude entry"
        );
        assert!(
            cfg.entry_for_match("default").is_some(),
            "embedded registry must include the family-fallback default entry"
        );
    }

    #[test]
    fn test_load_with_fallback_none_returns_embedded() {
        let cfg = ModelCapabilityConfig::load_with_fallback(None);
        assert!(!cfg.models.is_empty());
        assert_eq!(cfg, ModelCapabilityConfig::load_embedded());
    }

    #[test]
    fn test_load_with_fallback_missing_path_returns_embedded() {
        let missing = Path::new("config/__does_not_exist__.toml");
        let cfg = ModelCapabilityConfig::load_with_fallback(Some(missing));
        assert!(!cfg.models.is_empty());
    }

    #[test]
    fn test_load_with_fallback_external_file_wins() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("bgw_models_test_{}.toml", std::process::id()));
        std::fs::write(
            &path,
            "[[model]]\nmatch = \"external.only-model\"\ncapabilities = [\"adaptive_thinking\"]\n",
        )
        .unwrap();
        let cfg = ModelCapabilityConfig::load_with_fallback(Some(&path));
        std::fs::remove_file(&path).ok();
        let entry = cfg
            .entry_for_match("external.only-model")
            .expect("external file content must be loaded");
        assert!(entry.has_capability(Capability::AdaptiveThinking));
    }

    #[test]
    fn test_aliases_and_responses_backend_and_regions_parse() {
        let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n\n[[alias]]\nfrom = \"gpt-5.4\"\nto = \"openai.gpt-5.4\"\n\n[[model]]\nmatch = \"openai.gpt-5.5\"\n[model.params]\nresponses_backend = \"mantle\"\navailable_regions = [\"us-east-2\"]\n";
        let cfg =
            ModelCapabilityConfig::from_toml_str(raw).expect("config with aliases must parse");

        assert_eq!(cfg.aliases.len(), 2);
        assert_eq!(cfg.aliases[0].from, "gpt-5.5");
        assert_eq!(cfg.aliases[0].to, "openai.gpt-5.5");
        assert_eq!(cfg.aliases[1].from, "gpt-5.4");
        assert_eq!(cfg.aliases[1].to, "openai.gpt-5.4");

        let entry = cfg
            .entry_for_match("openai.gpt-5.5")
            .expect("gpt-5.5 model entry must resolve");
        assert_eq!(entry.params.responses_backend.as_deref(), Some("mantle"));
        assert_eq!(
            entry.params.available_regions.as_deref(),
            Some(["us-east-2".to_string()].as_slice())
        );
    }

    #[test]
    fn test_mantle_alias_names_derived_from_config() {
        // Two gpt aliases whose `to` resolves to mantle model entries → both
        // bare alias names are surfaced. A non-mantle alias is NOT surfaced.
        let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n\n[[alias]]\nfrom = \"gpt-5.4\"\nto = \"openai.gpt-5.4\"\n\n[[alias]]\nfrom = \"sonnet\"\nto = \"anthropic.claude-sonnet-4-5\"\n\n[[model]]\nmatch = \"openai.gpt-5.5\"\n[model.params]\nresponses_backend = \"mantle\"\n\n[[model]]\nmatch = \"openai.gpt-5.4\"\n[model.params]\nresponses_backend = \"mantle\"\n\n[[model]]\nmatch = \"anthropic.claude-sonnet-4-5\"\n";
        let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config must parse");

        let names: HashSet<String> = cfg.mantle_alias_names().into_iter().collect();
        let expected: HashSet<String> = ["gpt-5.4".to_string(), "gpt-5.5".to_string()]
            .into_iter()
            .collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn test_mantle_alias_names_empty_without_mantle_entries() {
        // Aliases that do not resolve to any mantle-backed entry surface nothing.
        let raw = "[[alias]]\nfrom = \"sonnet\"\nto = \"anthropic.claude-sonnet-4-5\"\n\n[[model]]\nmatch = \"anthropic.claude-sonnet-4-5\"\n";
        let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config must parse");
        assert!(cfg.mantle_alias_names().is_empty());
    }
}
