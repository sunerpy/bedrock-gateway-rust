//! Bedrock model listing + refresh (control-plane).
//!
//! Ports `list_bedrock_models` from `.legacy-python/src/api/models/bedrock.py`
//! (lines 194-302) to the `aws-sdk-bedrock` control-plane client. It builds a
//! unified model map plus `profile_metadata` that feeds the capability
//! resolver's `with_profiles` constructor (task 11).
//!
//! ## Design
//!
//! The async [`ModelCatalog::refresh`] only ADAPTS SDK output types into plain
//! Rust values, then delegates ALL filtering/assembly to the pure, fully unit
//! tested [`assemble_catalog`]. This keeps the de-hardcoding contract honest:
//! there are **no** model IDs or model-specific magic numbers in this file —
//! every model comes from the live Bedrock API, and the only filter constants
//! are the documented status / inference-type / streaming gates.
//!
//! ## Python parity (bedrock.py:194-302)
//!
//! - `enable_cross_region_inference` → paginate `list_inference_profiles`
//!   (`typeEquals=SYSTEM_DEFINED`, `maxResults=1000`); record `profile_id →
//!   first model's foundation id` (ARN tail).
//! - `enable_application_inference_profiles` → paginate `list_inference_profiles`
//!   (`typeEquals=APPLICATION`); record `profile_arn → first model's underlying
//!   id` (ARN tail).
//! - `list_foundation_models(byOutputModality=TEXT)` → keep models where
//!   `responseStreamingSupported` AND `status ∈ {ACTIVE, LEGACY}` AND
//!   `inferenceTypesSupported` contains `ON_DEMAND`; record their input
//!   modalities. Also surface every inference profile whose underlying model id
//!   matches a kept foundation model.
//! - Empty result → insert `default_model` with modalities `[TEXT, IMAGE]`.

use std::collections::HashMap;

use aws_sdk_bedrock::types::{
    FoundationModelLifecycleStatus, InferenceProfileType, InferenceType, ModelModality,
};

use crate::config::AppSettings;
use crate::error::AppError;
use crate::openai::schema::{Model, Models};

/// The two foundation-model lifecycle states the gateway exposes
/// (bedrock.py:270 — `status not in ["ACTIVE", "LEGACY"]`). Anything else
/// (e.g. rerank/preview-only models) is filtered out.
const ALLOWED_STATUSES: [FoundationModelLifecycleStatus; 2] = [
    FoundationModelLifecycleStatus::Active,
    FoundationModelLifecycleStatus::Legacy,
];

/// Per-model facts needed to decide inclusion and record modalities.
///
/// This is the plain-data shape the async `refresh` lowers SDK
/// `FoundationModelSummary` values into, so the assembly logic is testable
/// without constructing SDK types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundationModelFacts {
    /// The Bedrock foundation model id (e.g. `anthropic.claude-...-v1:0`).
    pub model_id: String,
    /// Input modalities reported by the API (e.g. `TEXT`, `IMAGE`).
    pub input_modalities: Vec<String>,
    /// Inference types reported by the API (e.g. `ON_DEMAND`).
    pub inference_types: Vec<String>,
    /// Whether the model supports streamed responses.
    pub response_streaming_supported: bool,
    /// Lifecycle status string (`ACTIVE`, `LEGACY`, ...).
    pub status: String,
}

/// A profile-metadata entry: a profile id/ARN mapped to its underlying
/// foundation model id (bedrock.py:221-224, 252-256).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileEntry {
    /// The key used to address this profile — a profile id (SYSTEM_DEFINED) or
    /// a profile ARN (APPLICATION).
    pub key: String,
    /// The underlying foundation model id (ARN tail of the first model).
    pub underlying_model_id: String,
}

/// Capabilities/modality info for one model in the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// Input modalities for the model (e.g. `["TEXT", "IMAGE"]`).
    pub modalities: Vec<String>,
}

/// The assembled model registry produced by a refresh.
///
/// `models` is the unified map of usable model ids (foundation models +
/// matching inference profiles) → [`ModelInfo`]. `profile_metadata` maps every
/// discovered profile id/ARN → its underlying foundation model id; it feeds
/// [`crate::bedrock::capabilities::ConfigModelCapabilities::with_profiles`] so
/// capability resolution works for profiles.
#[derive(Debug, Clone, Default)]
pub struct ModelCatalog {
    models: HashMap<String, ModelInfo>,
    profile_metadata: HashMap<String, String>,
    extra_model_ids: Vec<String>,
}

impl ModelCatalog {
    /// Refresh the catalog from the live Bedrock control-plane API.
    ///
    /// Mirrors `list_bedrock_models` (bedrock.py:194-302): optionally paginates
    /// system-defined and application inference profiles (gated by settings),
    /// lists TEXT-output foundation models, and assembles the unified map via
    /// [`assemble_catalog`]. On an empty result, falls back to
    /// `settings.default_model` with `[TEXT, IMAGE]` modalities.
    ///
    /// Transient Bedrock faults are surfaced as [`AppError`] via
    /// [`crate::error::from_bedrock_sdk_error`]; callers may retry.
    pub async fn refresh(
        control: &aws_sdk_bedrock::Client,
        settings: &AppSettings,
    ) -> Result<Self, AppError> {
        let mut profiles: Vec<ProfileEntry> = Vec::new();

        if settings.enable_cross_region_inference {
            profiles.extend(
                collect_profiles(control, InferenceProfileType::SystemDefined, |p| {
                    // SYSTEM_DEFINED profiles are addressed by profile id.
                    p.inference_profile_id().to_string()
                })
                .await?,
            );
        }

        if settings.enable_application_inference_profiles {
            profiles.extend(
                collect_profiles(control, InferenceProfileType::Application, |p| {
                    // APPLICATION profiles are addressed by ARN.
                    p.inference_profile_arn().to_string()
                })
                .await?,
            );
        }

        let response = control
            .list_foundation_models()
            .by_output_modality(ModelModality::Text)
            .send()
            .await
            .map_err(|e| crate::error::from_bedrock_sdk_error(&e))?;

        let foundation_models: Vec<FoundationModelFacts> = response
            .model_summaries()
            .iter()
            .map(|m| FoundationModelFacts {
                model_id: m.model_id().to_string(),
                input_modalities: m
                    .input_modalities()
                    .iter()
                    .map(|mod_| mod_.as_str().to_string())
                    .collect(),
                inference_types: m
                    .inference_types_supported()
                    .iter()
                    .map(|t| t.as_str().to_string())
                    .collect(),
                // Python defaults missing streaming flag to True (bedrock.py:266).
                response_streaming_supported: m.response_streaming_supported().unwrap_or(true),
                // Python defaults missing status to ACTIVE (bedrock.py:267).
                status: m
                    .model_lifecycle()
                    .map(|lc| lc.status().as_str().to_string())
                    .unwrap_or_else(|| FoundationModelLifecycleStatus::Active.as_str().to_string()),
            })
            .collect();

        Ok(assemble_catalog(&foundation_models, &profiles, settings))
    }

    /// Borrow the unified model map (model id / profile id-ARN → [`ModelInfo`]).
    pub fn models(&self) -> &HashMap<String, ModelInfo> {
        &self.models
    }

    /// The profile-metadata map (profile id/ARN → underlying foundation id).
    ///
    /// Pass this to
    /// [`crate::bedrock::capabilities::ConfigModelCapabilities::with_profiles`].
    pub fn profile_metadata(&self) -> &HashMap<String, String> {
        &self.profile_metadata
    }

    /// Attach display-only model ids (e.g. mantle-backed GPT aliases) that are
    /// absent from the Bedrock control-plane catalog. They surface in `list()`
    /// and `get()` but are NOT added to `models` or `profile_metadata`, so
    /// capability resolution and routing are unaffected.
    pub fn with_extra_models(mut self, ids: Vec<String>) -> Self {
        self.extra_model_ids = ids;
        self
    }

    /// Retain only models whose id contains any `patterns` entry as a
    /// case-insensitive substring, filtering BOTH the control-plane `models`
    /// map and the `extra_model_ids` (mantle aliases). Empty `patterns` is the
    /// identity — the catalog is returned unchanged (allow-all, backward
    /// compatible). `profile_metadata` is left untouched so capability/routing
    /// resolution is unaffected; only the surfaces `list()`/`get()` iterate over
    /// are narrowed. Applied once at catalog-build time, so `GET /models` and
    /// `GET /models/{id}` are both filtered with zero per-request cost.
    pub fn apply_allow_list(mut self, patterns: &[String]) -> Self {
        if patterns.is_empty() {
            return self;
        }
        let matches = |id: &str| {
            let lower = id.to_ascii_lowercase();
            patterns
                .iter()
                .any(|p| lower.contains(&p.to_ascii_lowercase()))
        };
        self.models.retain(|id, _| matches(id));
        self.extra_model_ids.retain(|id| matches(id));
        self
    }

    /// Lookup a single model in OpenAI `Model` shape, if present.
    pub fn get(&self, id: &str) -> Option<Model> {
        if self.models.contains_key(id) || self.extra_model_ids.iter().any(|e| e == id) {
            Some(make_model(id))
        } else {
            None
        }
    }

    /// Render the catalog as the OpenAI `Models` list (sorted by id for stable
    /// output). Every entry uses `object="model"`, `owned_by="bedrock"` and a
    /// fixed `created` epoch (parity with the Python `Model` defaults).
    pub fn list(&self) -> Models {
        let mut ids: Vec<&str> = self.models.keys().map(String::as_str).collect();
        ids.extend(self.extra_model_ids.iter().map(String::as_str));
        ids.sort_unstable();
        ids.dedup();
        Models {
            object: "list".to_string(),
            data: ids.into_iter().map(make_model).collect(),
        }
    }
}

/// Build an OpenAI [`Model`] descriptor for a model id.
///
/// `created` is a fixed epoch (`0`) — the Python schema hardcodes a constant
/// creation time and the value is not semantically meaningful for Bedrock
/// models; keeping it constant yields deterministic output.
fn make_model(id: &str) -> Model {
    Model {
        id: id.to_string(),
        created: 0,
        object: "model".to_string(),
        owned_by: "bedrock".to_string(),
    }
}

/// Paginate `list_inference_profiles` for one profile `type_equals`, lowering
/// each summary into a [`ProfileEntry`] keyed by `key_of`.
///
/// Mirrors bedrock.py:207-224 / 228-259: take the FIRST model's `modelArn`,
/// derive the underlying foundation id as the ARN tail (`split('/').last()`),
/// and skip profiles with no key or no models.
async fn collect_profiles<F>(
    control: &aws_sdk_bedrock::Client,
    type_equals: InferenceProfileType,
    key_of: F,
) -> Result<Vec<ProfileEntry>, AppError>
where
    F: Fn(&aws_sdk_bedrock::types::InferenceProfileSummary) -> String,
{
    let mut entries = Vec::new();
    let mut stream = control
        .list_inference_profiles()
        .max_results(1000)
        .type_equals(type_equals)
        .into_paginator()
        .send();

    while let Some(page) = stream
        .next()
        .await
        .transpose()
        .map_err(|e| crate::error::from_bedrock_sdk_error(&e))?
    {
        for profile in page.inference_profile_summaries() {
            let key = key_of(profile);
            if key.is_empty() {
                continue;
            }
            // First model only — all models in the array are regional
            // instances of the same underlying model (bedrock.py:242-243).
            let Some(model_arn) = profile.models().first().and_then(|m| m.model_arn()) else {
                continue;
            };
            if model_arn.is_empty() {
                continue;
            }
            entries.push(ProfileEntry {
                key,
                underlying_model_id: underlying_id_from_arn(model_arn),
            });
        }
    }

    Ok(entries)
}

/// Extract a foundation model id from a model ARN by taking the segment after
/// the last `/` (bedrock.py:220, 249). ARNs without a `/` pass through
/// unchanged.
fn underlying_id_from_arn(model_arn: &str) -> String {
    model_arn
        .rsplit('/')
        .next()
        .unwrap_or(model_arn)
        .to_string()
}

/// Whether a foundation model is callable as a bare on-demand id: it must
/// stream, sit in an [`ALLOWED_STATUSES`] lifecycle state, AND advertise
/// `ON_DEMAND` inference (bedrock.py:270-276). Bare ids that lack `ON_DEMAND`
/// (e.g. `INFERENCE_PROFILE`-only models such as `anthropic.claude-sonnet-4-5`)
/// cannot be invoked directly and must NOT enter the catalog under their bare id.
fn foundation_directly_callable(fm: &FoundationModelFacts) -> bool {
    let status_ok = ALLOWED_STATUSES
        .iter()
        .any(|s| s.as_str().eq_ignore_ascii_case(&fm.status));
    let on_demand_ok = fm
        .inference_types
        .iter()
        .any(|t| t == InferenceType::OnDemand.as_str());
    fm.response_streaming_supported && status_ok && on_demand_ok
}

/// Whether a foundation model is usable AS THE BACKING of an inference profile:
/// it must stream AND sit in an allowed lifecycle state. `ON_DEMAND` is NOT
/// required here — invoking through a profile *is* the `INFERENCE_PROFILE`
/// call path, so a profile-only foundation (no `ON_DEMAND`) is still a valid
/// backing for its cross-region profiles.
fn foundation_eligible_for_profile(fm: &FoundationModelFacts) -> bool {
    let status_ok = ALLOWED_STATUSES
        .iter()
        .any(|s| s.as_str().eq_ignore_ascii_case(&fm.status));
    fm.response_streaming_supported && status_ok
}

/// PURE assembly of the unified catalog from plain data (no SDK / no network).
///
/// This is the heart of `list_bedrock_models` and the unit-test surface:
///
/// 1. `profile_metadata` is built from every discovered profile entry (id/ARN →
///    underlying foundation id).
/// 2. A foundation model is included under its BARE id iff it is directly
///    callable — it streams, its status is in [`ALLOWED_STATUSES`], AND it
///    supports `ON_DEMAND` inference (see [`foundation_directly_callable`]).
///    `INFERENCE_PROFILE`-only foundations (no `ON_DEMAND`) are intentionally
///    excluded under their bare id because the bare id is not invocable.
/// 3. Inference profiles are surfaced INDEPENDENTLY of step 2's `ON_DEMAND`
///    gate: a profile enters the map iff its underlying foundation streams and
///    has an allowed status (see [`foundation_eligible_for_profile`]). This is
///    what lets cross-region profiles of an `INFERENCE_PROFILE`-only model
///    (e.g. `us.`/`global.anthropic.claude-sonnet-4-5`) appear in the catalog
///    while the bare `anthropic.claude-sonnet-4-5` correctly does not.
/// 4. If the resulting map is empty, fall back to `settings.default_model` with
///    `[TEXT, IMAGE]` modalities (bedrock.py:287-289).
pub fn assemble_catalog(
    foundation_models: &[FoundationModelFacts],
    profiles: &[ProfileEntry],
    settings: &AppSettings,
) -> ModelCatalog {
    let profile_metadata: HashMap<String, String> = profiles
        .iter()
        .map(|p| (p.key.clone(), p.underlying_model_id.clone()))
        .collect();

    // Index foundation facts by model id so profile inclusion can look up its
    // backing foundation's eligibility + modalities without an O(n*m) scan.
    let foundation_index: HashMap<&str, &FoundationModelFacts> = foundation_models
        .iter()
        .map(|fm| (fm.model_id.as_str(), fm))
        .collect();

    let mut models: HashMap<String, ModelInfo> = HashMap::new();

    // Step 2 — directly callable foundations enter under their bare id.
    for fm in foundation_models {
        if !foundation_directly_callable(fm) {
            continue;
        }
        models.insert(
            fm.model_id.clone(),
            ModelInfo {
                modalities: fm.input_modalities.clone(),
            },
        );
    }

    // Step 3 — profiles enter independently of the ON_DEMAND gate, as long as
    // their underlying foundation streams + has an allowed status. The profile
    // inherits that foundation's modalities. This admits cross-region profiles
    // of INFERENCE_PROFILE-only foundations (e.g. claude) that step 2 skips.
    for (profile_key, underlying) in &profile_metadata {
        let Some(fm) = foundation_index.get(underlying.as_str()) else {
            continue;
        };
        if !foundation_eligible_for_profile(fm) {
            continue;
        }
        models.insert(
            profile_key.clone(),
            ModelInfo {
                modalities: fm.input_modalities.clone(),
            },
        );
    }

    if models.is_empty() {
        // Stack-not-updated fallback (bedrock.py:287-289).
        models.insert(
            settings.default_model.clone(),
            ModelInfo {
                modalities: vec![
                    ModelModality::Text.as_str().to_string(),
                    ModelModality::Image.as_str().to_string(),
                ],
            },
        );
    }

    ModelCatalog {
        models,
        profile_metadata,
        extra_model_ids: Vec::new(),
    }
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
