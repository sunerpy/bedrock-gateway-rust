//! Per-model region-routing configuration schema + TOML loader.
//!
//! This module is the SCHEMA + LOADER for externalized region routing. It
//! replaces the Python `MODEL_REGION_ROUTING` dict (see
//! `.legacy-python/src/api/models/bedrock.py` lines 79-119) — a map from an
//! incoming model id to `(target_region, rewritten_model_id)` — with a single
//! declarative `config/regions.toml`.
//!
//! Like the Python source, this ships EMPTY by default: with no entries every
//! model uses the gateway's home-region client unchanged. An override is added
//! only when a model's global profile is unreachable from the home region and a
//! geo profile in another region works.
//!
//! IMPORTANT — DE-HARDCODING CONTRACT:
//! This file defines ONLY the schema and a typed loader. It MUST NOT contain any
//! region names or model IDs. ALL such data lives in `config/regions.toml`.
//! Adding a route is a TOML-only edit and requires NO recompile.
//!
//! The resolution ALGORITHM and the runtime-client cache (Python
//! `_route_model` / `_get_runtime_client`, bedrock.py:92-119) are intentionally
//! NOT implemented here — those are separate tasks. This module exposes only
//! simple typed access to the loaded table.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single region-routing override.
///
/// Mirrors the Python tuple value `(target_region, rewritten_model_id)` from
/// `MODEL_REGION_ROUTING` (bedrock.py:80-85). The keying model id is stored on
/// the enclosing [`RouteEntry`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteOverride {
    /// Target AWS region the model must run in (e.g. a geo cross-region
    /// profile's region). Python tuple element 0.
    pub region: String,
    /// The rewritten profile / model id to send to that region. Python tuple
    /// element 1.
    pub rewritten_model_id: String,
}

/// A single `[[route]]` table from `config/regions.toml`: an incoming model id
/// plus its [`RouteOverride`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteEntry {
    /// The incoming model id this route matches (exact key — Python dict key).
    pub model_id: String,
    /// Target region the matched model is routed to.
    pub region: String,
    /// Rewritten model / profile id used in the target region.
    pub rewritten_model_id: String,
}

/// The top-level region-routing table, deserialized from
/// `config/regions.toml`. Empty by default (Python parity: `{}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionRoutingConfig {
    /// The list of route entries (TOML `[[route]]` tables). Empty means no
    /// overrides — every model uses the home-region client unchanged.
    #[serde(default, rename = "route")]
    pub routes: Vec<RouteEntry>,
}

/// The region-routing TOML embedded into the binary at compile time.
///
/// `include_str!` resolves relative to this source file (`src/config/`), so
/// `../../config/regions.toml` reaches the repo-root file. Ships empty by
/// default (Python parity), but embedding it keeps the loader's contract
/// uniform with the other two config modules.
const EMBEDDED_REGIONS_TOML: &str = include_str!("../../config/regions.toml");

impl RegionRoutingConfig {
    /// Load and parse a region-routing config from a TOML file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read region config file: {}", path.display()))?;
        Self::from_toml_str(&raw)
            .with_context(|| format!("failed to parse region config file: {}", path.display()))
    }

    /// Parse the compile-time-embedded region-routing config. Embedded TOML is
    /// build-time-validated; a parse failure is a release defect, hence `expect`.
    pub fn load_embedded() -> Self {
        Self::from_toml_str(EMBEDDED_REGIONS_TOML)
            .expect("embedded config/regions.toml must be valid")
    }

    /// Load with external-over-embedded fallback (see
    /// `ModelCapabilityConfig::load_with_fallback` for the shared contract).
    pub fn load_with_fallback(external_path: Option<&Path>) -> Self {
        let Some(path) = external_path else {
            return Self::load_embedded();
        };
        if !path.exists() {
            tracing::debug!(
                path = %path.display(),
                "region-routing config file absent; using embedded default"
            );
            return Self::load_embedded();
        }
        match Self::load(path) {
            Ok(cfg) => {
                tracing::info!(
                    path = %path.display(),
                    "loaded external region-routing config"
                );
                cfg
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "external region-routing config invalid; using embedded default"
                );
                Self::load_embedded()
            }
        }
    }

    /// Parse a region-routing config from a TOML string.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let parsed: Self = toml::from_str(raw).context("invalid TOML for region routing")?;
        Ok(parsed)
    }

    /// Simple typed access: find the override for an exact model id.
    ///
    /// Returns `None` when no entry matches — the caller then preserves the home
    /// region and original model id (Python `_route_model` "route is None"
    /// branch, bedrock.py:115-117). This is exact-key lookup only — the routing
    /// algorithm and client cache are separate tasks.
    pub fn route_for(&self, model_id: &str) -> Option<RouteOverride> {
        self.routes
            .iter()
            .find(|e| e.model_id == model_id)
            .map(|e| RouteOverride {
                region: e.region.clone(),
                rewritten_model_id: e.rewritten_model_id.clone(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the project's authored region config, relative to the crate root.
    const REGIONS_TOML: &str = "config/regions.toml";

    #[test]
    fn test_project_regions_toml_is_empty_by_default() {
        // Python parity: MODEL_REGION_ROUTING = {} (bedrock.py:85). The shipped
        // file must contain no active routes — only the commented example.
        let cfg = RegionRoutingConfig::load(REGIONS_TOML)
            .expect("config/regions.toml must load and parse");
        assert!(
            cfg.routes.is_empty(),
            "config/regions.toml must ship empty (no [[route]] entries)"
        );
    }

    #[test]
    fn test_empty_config_route_for_returns_none() {
        // Default (empty) → any lookup returns None → home region preserved.
        let cfg = RegionRoutingConfig::default();
        assert_eq!(cfg.route_for("anything"), None);

        let cfg = RegionRoutingConfig::from_toml_str("").unwrap();
        assert_eq!(cfg.route_for("anything"), None);
    }

    #[test]
    fn test_populated_entry_returns_override() {
        // One entry model="x" region="eu-central-1" rewritten="y" → route_for("x") == Some.
        let raw = "\
[[route]]
model_id = \"x\"
region = \"eu-central-1\"
rewritten_model_id = \"y\"
";
        let cfg = RegionRoutingConfig::from_toml_str(raw).expect("must parse");
        let route = cfg.route_for("x").expect("route_for(\"x\") must be Some");
        assert_eq!(route.region, "eu-central-1");
        assert_eq!(route.rewritten_model_id, "y");

        // Non-matching key still returns None (home region preserved).
        assert_eq!(cfg.route_for("z"), None);
    }

    #[test]
    fn test_extension_requires_toml_only() {
        // Proves a NEW route can be added by editing TOML alone — no code change.
        let raw = "\
[[route]]
model_id = \"vendor.future-model-99\"
region = \"ap-northeast-1\"
rewritten_model_id = \"apac.vendor.future-model-99\"
";
        let cfg = RegionRoutingConfig::from_toml_str(raw).unwrap();
        let route = cfg.route_for("vendor.future-model-99").unwrap();
        assert_eq!(route.region, "ap-northeast-1");
        assert_eq!(route.rewritten_model_id, "apac.vendor.future-model-99");
    }

    #[test]
    fn test_load_missing_file_errors() {
        let err = RegionRoutingConfig::load("config/__does_not_exist__.toml");
        assert!(err.is_err());
    }

    #[test]
    fn test_load_embedded_parses_ok() {
        let cfg = RegionRoutingConfig::load_embedded();
        assert_eq!(cfg, RegionRoutingConfig::load_embedded());
        assert_eq!(cfg.route_for("anything"), None);
    }

    #[test]
    fn test_load_with_fallback_none_returns_embedded() {
        let cfg = RegionRoutingConfig::load_with_fallback(None);
        assert_eq!(cfg, RegionRoutingConfig::load_embedded());
    }

    #[test]
    fn test_load_with_fallback_missing_path_returns_embedded() {
        let missing = Path::new("config/__does_not_exist__.toml");
        let cfg = RegionRoutingConfig::load_with_fallback(Some(missing));
        assert_eq!(cfg, RegionRoutingConfig::load_embedded());
    }

    #[test]
    fn test_load_with_fallback_external_file_wins() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("bgw_regions_test_{}.toml", std::process::id()));
        std::fs::write(
            &path,
            "[[route]]\nmodel_id = \"x\"\nregion = \"eu-central-1\"\nrewritten_model_id = \"y\"\n",
        )
        .unwrap();
        let cfg = RegionRoutingConfig::load_with_fallback(Some(&path));
        std::fs::remove_file(&path).ok();
        let route = cfg.route_for("x").expect("external route must be loaded");
        assert_eq!(route.region, "eu-central-1");
        assert_eq!(route.rewritten_model_id, "y");
    }
}
