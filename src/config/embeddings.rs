//! Embedding-model registry config schema + TOML loader.
//!
//! This module is the SCHEMA + LOADER for the externalized embedding-model
//! registry. It replaces the Python `SUPPORTED_BEDROCK_EMBEDDING_MODELS` dict
//! (see `.legacy-python/src/api/models/bedrock.py` lines 122-130) with a single
//! declarative `config/embeddings.toml`.
//!
//! IMPORTANT — DE-HARDCODING CONTRACT:
//! This file defines ONLY the schema and a typed loader. It MUST NOT contain any
//! model IDs, display names, or family values. ALL such data lives in
//! `config/embeddings.toml`. Adding a new embedding model is a TOML-only edit and
//! requires NO recompile.
//!
//! The `family` field selects which Bedrock request/response body codec to use
//! (Cohere: bedrock.py:1938-1978; Titan: bedrock.py:1981+; Nova multimodal). The
//! body-shape / codec logic itself is intentionally NOT implemented here — that
//! is a separate task. This module exposes only simple typed access to the loaded
//! registry.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The body-codec family for an embedding model.
///
/// Selects which Bedrock request/response body shape applies. Serialized in TOML
/// as lowercase strings (`cohere`, `titan`, `nova`).
///
/// Provenance (`.legacy-python/src/api/models/bedrock.py`):
///   - Cohere body codec: lines 1938-1978
///   - Titan body codec:  lines 1981+
///   - Nova multimodal body codec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingFamily {
    /// Cohere embed family (bedrock.py:1938-1978).
    Cohere,
    /// Amazon Titan embed family (bedrock.py:1981+).
    Titan,
    /// Amazon Nova multimodal embed family.
    Nova,
}

/// A single declarative embedding-model entry from `config/embeddings.toml`.
///
/// Authored as a TOML array-of-tables (`[[model]]`) with an explicit `model_id`
/// string field. Using a string field (rather than a bare TOML key) keeps model
/// ids containing a colon — e.g. `amazon.titan-embed-text-v2:0` — round-tripping
/// cleanly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelEntry {
    /// The exact Bedrock embedding model id (e.g. `cohere.embed-english-v3`).
    pub model_id: String,

    /// Human-friendly display name (parity with the Python dict value).
    pub display_name: String,

    /// The body-codec family for this model.
    pub family: EmbeddingFamily,
}

/// The top-level embedding-model registry, deserialized from
/// `config/embeddings.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRegistry {
    /// The list of model entries (TOML `[[model]]` tables).
    #[serde(default, rename = "model")]
    pub models: Vec<EmbeddingModelEntry>,
}

/// The embedding-registry TOML embedded into the binary at compile time.
///
/// `include_str!` resolves relative to this source file (`src/config/`), so
/// `../../config/embeddings.toml` reaches the repo-root file. Guarantees the
/// single binary ships with a non-empty embedding registry even with no
/// external `config/` directory present at runtime.
const EMBEDDED_EMBEDDINGS_TOML: &str = include_str!("../../config/embeddings.toml");

impl EmbeddingRegistry {
    /// Load and parse an embedding registry from a TOML file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read embedding config file: {}", path.display()))?;
        Self::from_toml_str(&raw)
            .with_context(|| format!("failed to parse embedding config file: {}", path.display()))
    }

    /// Parse the compile-time-embedded embedding registry. Embedded TOML is
    /// build-time-validated; a parse failure is a release defect, hence `expect`.
    pub fn load_embedded() -> Self {
        Self::from_toml_str(EMBEDDED_EMBEDDINGS_TOML)
            .expect("embedded config/embeddings.toml must be valid")
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
                "embedding-registry config file absent; using embedded default"
            );
            return Self::load_embedded();
        }
        match Self::load(path) {
            Ok(cfg) => {
                tracing::info!(
                    path = %path.display(),
                    "loaded external embedding-registry config"
                );
                cfg
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "external embedding-registry config invalid; using embedded default"
                );
                Self::load_embedded()
            }
        }
    }

    /// Parse an embedding registry from a TOML string.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let parsed: Self = toml::from_str(raw).context("invalid TOML for embedding registry")?;
        Ok(parsed)
    }

    /// Find the entry whose `model_id` exactly equals the given id.
    pub fn entry_for(&self, model_id: &str) -> Option<&EmbeddingModelEntry> {
        self.models.iter().find(|e| e.model_id == model_id)
    }

    /// Resolve the body-codec family for the given model id, if registered.
    /// Returns `None` for unknown models (no panic).
    pub fn family_for(&self, model_id: &str) -> Option<EmbeddingFamily> {
        self.entry_for(model_id).map(|e| e.family)
    }
}

#[cfg(test)]
#[path = "embeddings_tests.rs"]
mod tests;
