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
mod tests {
    use super::*;

    /// Minimal `AppSettings` for assembly tests. Only `default_model` and the
    /// two profile gates matter for these unit tests; the rest are inert.
    fn settings(default_model: &str) -> AppSettings {
        AppSettings {
            api_route_prefix: "/api/v1".to_string(),
            debug: false,
            aws_region: "us-west-2".to_string(),
            default_model: default_model.to_string(),
            default_embedding_model: "cohere.embed-multilingual-v3".to_string(),
            enable_cross_region_inference: true,
            enable_application_inference_profiles: true,
            enable_prompt_caching: false,
            api_key: None,
            api_key_secret_arn: None,
            api_key_param_name: None,
            bedrock_api_key: None,
            disable_mantle: false,
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            log_level: "info".to_string(),
            aws_connect_timeout_secs: 60,
            aws_read_timeout_secs: 900,
            aws_max_retry_attempts: 8,
            mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1"
                .to_string(),
            allowed_models: None,
        }
    }

    fn fm(
        id: &str,
        modalities: &[&str],
        inference: &[&str],
        streaming: bool,
        status: &str,
    ) -> FoundationModelFacts {
        FoundationModelFacts {
            model_id: id.to_string(),
            input_modalities: modalities.iter().map(|s| s.to_string()).collect(),
            inference_types: inference.iter().map(|s| s.to_string()).collect(),
            response_streaming_supported: streaming,
            status: status.to_string(),
        }
    }

    #[test]
    fn on_demand_streaming_active_model_is_included() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.model-a-v1:0",
            &["TEXT", "IMAGE"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let catalog = assemble_catalog(&models, &[], &s);

        assert!(catalog.models().contains_key("vendor.model-a-v1:0"));
        assert_eq!(
            catalog.models()["vendor.model-a-v1:0"].modalities,
            vec!["TEXT".to_string(), "IMAGE".to_string()]
        );
        // Profile metadata is empty when no profiles supplied.
        assert!(catalog.profile_metadata().is_empty());
    }

    #[test]
    fn legacy_status_is_included() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.legacy-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "LEGACY",
        )];
        let catalog = assemble_catalog(&models, &[], &s);
        assert!(catalog.models().contains_key("vendor.legacy-v1:0"));
    }

    #[test]
    fn non_streaming_model_is_filtered_out() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.no-stream-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            false,
            "ACTIVE",
        )];
        let catalog = assemble_catalog(&models, &[], &s);
        // Empty after filter → falls back to default model only.
        assert!(!catalog.models().contains_key("vendor.no-stream-v1:0"));
        assert!(catalog.models().contains_key("fallback.model-v1:0"));
    }

    #[test]
    fn disallowed_status_is_filtered_out() {
        let s = settings("fallback.model-v1:0");
        // A streaming, on-demand model but in an unsupported lifecycle state.
        let models = [
            fm(
                "vendor.preview-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "PREVIEW",
            ),
            fm(
                "vendor.keep-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
        ];
        let catalog = assemble_catalog(&models, &[], &s);
        assert!(!catalog.models().contains_key("vendor.preview-v1:0"));
        assert!(catalog.models().contains_key("vendor.keep-v1:0"));
    }

    #[test]
    fn non_on_demand_model_is_filtered_out() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.provisioned-v1:0",
            &["TEXT"],
            &["PROVISIONED"],
            true,
            "ACTIVE",
        )];
        let catalog = assemble_catalog(&models, &[], &s);
        assert!(!catalog.models().contains_key("vendor.provisioned-v1:0"));
        // Empty → fallback.
        assert!(catalog.models().contains_key("fallback.model-v1:0"));
    }

    #[test]
    fn empty_input_falls_back_to_default_model_with_text_image() {
        let s = settings("my.default-v1:0");
        let catalog = assemble_catalog(&[], &[], &s);

        assert_eq!(catalog.models().len(), 1);
        let info = &catalog.models()["my.default-v1:0"];
        assert_eq!(
            info.modalities,
            vec!["TEXT".to_string(), "IMAGE".to_string()]
        );
    }

    #[test]
    fn system_defined_profile_metadata_and_matching_model_surfaced() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.base-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let profiles = [ProfileEntry {
            key: "us.vendor.base-v1:0".to_string(),
            underlying_model_id: "vendor.base-v1:0".to_string(),
        }];
        let catalog = assemble_catalog(&models, &profiles, &s);

        // profile_metadata records the mapping.
        assert_eq!(
            catalog.profile_metadata()["us.vendor.base-v1:0"],
            "vendor.base-v1:0"
        );
        // Both the foundation model and its matching profile are in the map.
        assert!(catalog.models().contains_key("vendor.base-v1:0"));
        assert!(catalog.models().contains_key("us.vendor.base-v1:0"));
        // The profile inherits the foundation model's modalities.
        assert_eq!(
            catalog.models()["us.vendor.base-v1:0"].modalities,
            vec!["TEXT".to_string()]
        );
    }

    #[test]
    fn application_profile_arn_recorded_even_without_matching_model() {
        // An application profile whose underlying model is NOT in the foundation
        // list still appears in profile_metadata (for capability resolution),
        // but is NOT added to the usable model map.
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.present-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123";
        let profiles = [ProfileEntry {
            key: arn.to_string(),
            underlying_model_id: "vendor.absent-v1:0".to_string(),
        }];
        let catalog = assemble_catalog(&models, &profiles, &s);

        // Metadata recorded.
        assert_eq!(catalog.profile_metadata()[arn], "vendor.absent-v1:0");
        // But the ARN is NOT a usable model (its underlying model is absent).
        assert!(!catalog.models().contains_key(arn));
        // The present foundation model is still there.
        assert!(catalog.models().contains_key("vendor.present-v1:0"));
    }

    #[test]
    fn application_profile_surfaced_when_underlying_present() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.base-v1:0",
            &["TEXT", "IMAGE"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/p1";
        let profiles = [ProfileEntry {
            key: arn.to_string(),
            underlying_model_id: "vendor.base-v1:0".to_string(),
        }];
        let catalog = assemble_catalog(&models, &profiles, &s);

        assert!(catalog.models().contains_key(arn));
        assert_eq!(
            catalog.models()[arn].modalities,
            vec!["TEXT".to_string(), "IMAGE".to_string()]
        );
    }

    #[test]
    fn inference_profile_only_foundation_surfaces_profile_not_bare_id() {
        // The claude root-cause case: a foundation with ONLY INFERENCE_PROFILE
        // (no ON_DEMAND). Its bare id is not directly invocable, so it must NOT
        // enter the catalog; but its cross-region profile MUST, since calling
        // through the profile is the INFERENCE_PROFILE call path.
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "anthropic.claude-sonnet-4-5-v1:0",
            &["TEXT", "IMAGE"],
            &["INFERENCE_PROFILE"],
            true,
            "ACTIVE",
        )];
        let profiles = [
            ProfileEntry {
                key: "us.anthropic.claude-sonnet-4-5-v1:0".to_string(),
                underlying_model_id: "anthropic.claude-sonnet-4-5-v1:0".to_string(),
            },
            ProfileEntry {
                key: "global.anthropic.claude-sonnet-4-5-v1:0".to_string(),
                underlying_model_id: "anthropic.claude-sonnet-4-5-v1:0".to_string(),
            },
        ];
        let catalog = assemble_catalog(&models, &profiles, &s);

        assert!(
            !catalog
                .models()
                .contains_key("anthropic.claude-sonnet-4-5-v1:0"),
            "bare INFERENCE_PROFILE-only foundation must not be directly listed"
        );
        assert!(catalog
            .models()
            .contains_key("us.anthropic.claude-sonnet-4-5-v1:0"));
        assert!(catalog
            .models()
            .contains_key("global.anthropic.claude-sonnet-4-5-v1:0"));
        assert_eq!(
            catalog.models()["us.anthropic.claude-sonnet-4-5-v1:0"].modalities,
            vec!["TEXT".to_string(), "IMAGE".to_string()]
        );
    }

    #[test]
    fn on_demand_foundation_lists_both_bare_id_and_profile() {
        // Original behavior preserved: an ON_DEMAND foundation lists under its
        // bare id AND surfaces its profile.
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.both-v1:0",
            &["TEXT"],
            &["ON_DEMAND", "INFERENCE_PROFILE"],
            true,
            "ACTIVE",
        )];
        let profiles = [ProfileEntry {
            key: "us.vendor.both-v1:0".to_string(),
            underlying_model_id: "vendor.both-v1:0".to_string(),
        }];
        let catalog = assemble_catalog(&models, &profiles, &s);

        assert!(catalog.models().contains_key("vendor.both-v1:0"));
        assert!(catalog.models().contains_key("us.vendor.both-v1:0"));
    }

    #[test]
    fn profile_excluded_when_underlying_foundation_fails_stream_or_status() {
        // A profile only enters when its underlying foundation streams AND has
        // an allowed status. A non-streaming or disallowed-status backing keeps
        // the profile out, even though ON_DEMAND is not required.
        let s = settings("fallback.model-v1:0");
        let models = [
            fm(
                "vendor.no-stream-v1:0",
                &["TEXT"],
                &["INFERENCE_PROFILE"],
                false,
                "ACTIVE",
            ),
            fm(
                "vendor.preview-v1:0",
                &["TEXT"],
                &["INFERENCE_PROFILE"],
                true,
                "PREVIEW",
            ),
        ];
        let profiles = [
            ProfileEntry {
                key: "us.vendor.no-stream-v1:0".to_string(),
                underlying_model_id: "vendor.no-stream-v1:0".to_string(),
            },
            ProfileEntry {
                key: "us.vendor.preview-v1:0".to_string(),
                underlying_model_id: "vendor.preview-v1:0".to_string(),
            },
        ];
        let catalog = assemble_catalog(&models, &profiles, &s);

        assert!(!catalog.models().contains_key("us.vendor.no-stream-v1:0"));
        assert!(!catalog.models().contains_key("us.vendor.preview-v1:0"));
        // Nothing usable → fallback only.
        assert!(catalog.models().contains_key("fallback.model-v1:0"));
    }

    #[test]
    fn underlying_id_from_arn_takes_last_segment() {
        assert_eq!(
            underlying_id_from_arn(
                "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-v1:0"
            ),
            "anthropic.claude-3-v1:0"
        );
        // No slash → passthrough.
        assert_eq!(
            underlying_id_from_arn("vendor.model-v1:0"),
            "vendor.model-v1:0"
        );
    }

    #[test]
    fn list_renders_openai_model_shape_sorted() {
        let s = settings("fallback.model-v1:0");
        let models = [
            fm(
                "vendor.zeta-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
            fm(
                "vendor.alpha-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
        ];
        let catalog = assemble_catalog(&models, &[], &s);
        let listed = catalog.list();

        assert_eq!(listed.object, "list");
        assert_eq!(listed.data.len(), 2);
        // Sorted by id.
        assert_eq!(listed.data[0].id, "vendor.alpha-v1:0");
        assert_eq!(listed.data[1].id, "vendor.zeta-v1:0");
        // OpenAI shape defaults.
        assert_eq!(listed.data[0].object, "model");
        assert_eq!(listed.data[0].owned_by, "bedrock");
    }

    #[test]
    fn get_returns_model_when_present_and_none_otherwise() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.base-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let catalog = assemble_catalog(&models, &[], &s);

        let got = catalog.get("vendor.base-v1:0").expect("model present");
        assert_eq!(got.id, "vendor.base-v1:0");
        assert_eq!(got.object, "model");
        assert_eq!(got.owned_by, "bedrock");

        assert!(catalog.get("nope.absent-v1:0").is_none());
    }

    #[test]
    fn extra_model_ids_listed_merged_sorted_deduped() {
        let s = settings("fallback.model-v1:0");
        let models = [
            fm(
                "vendor.zeta-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
            fm(
                "vendor.alpha-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
        ];
        let catalog = assemble_catalog(&models, &[], &s).with_extra_models(vec![
            "gpt-5.5".to_string(),
            "gpt-5.4".to_string(),
            "vendor.zeta-v1:0".to_string(),
        ]);
        let listed = catalog.list();

        let ids: Vec<String> = listed.data.iter().map(|m| m.id.clone()).collect();
        assert_eq!(
            ids,
            vec![
                "gpt-5.4".to_string(),
                "gpt-5.5".to_string(),
                "vendor.alpha-v1:0".to_string(),
                "vendor.zeta-v1:0".to_string(),
            ]
        );
        assert_eq!(listed.data[0].object, "model");
        assert_eq!(listed.data[0].owned_by, "bedrock");
    }

    #[test]
    fn get_resolves_extra_model_ids() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.base-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let catalog =
            assemble_catalog(&models, &[], &s).with_extra_models(vec!["gpt-5.5".to_string()]);

        let got = catalog.get("gpt-5.5").expect("extra model present");
        assert_eq!(got.id, "gpt-5.5");
        assert_eq!(got.object, "model");
        assert!(catalog.get("vendor.base-v1:0").is_some());
        assert!(catalog.get("nope.absent-v1:0").is_none());
        assert!(!catalog.profile_metadata().contains_key("gpt-5.5"));
    }

    #[test]
    fn status_match_is_case_insensitive() {
        // Defensive: the SDK enum renders uppercase, but the pure fn should not
        // care about case if a lowercase value ever reaches it.
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "vendor.lower-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "active",
        )];
        let catalog = assemble_catalog(&models, &[], &s);
        assert!(catalog.models().contains_key("vendor.lower-v1:0"));
    }

    #[test]
    fn apply_allow_list_empty_is_identity() {
        let s = settings("fallback.model-v1:0");
        let models = [
            fm(
                "anthropic.claude-3-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
            fm(
                "amazon.nova-pro-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
        ];
        let catalog = assemble_catalog(&models, &[], &s);
        let before = catalog.list().data.len();

        let filtered = catalog.apply_allow_list(&[]);

        assert_eq!(filtered.list().data.len(), before);
        assert!(filtered.get("anthropic.claude-3-v1:0").is_some());
        assert!(filtered.get("amazon.nova-pro-v1:0").is_some());
    }

    #[test]
    fn apply_allow_list_substring_filters_list_and_get() {
        let s = settings("fallback.model-v1:0");
        let models = [
            fm(
                "anthropic.claude-3-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
            fm(
                "amazon.nova-pro-v1:0",
                &["TEXT"],
                &["ON_DEMAND"],
                true,
                "ACTIVE",
            ),
        ];
        let filtered = assemble_catalog(&models, &[], &s).apply_allow_list(&["claude".to_string()]);

        let ids: Vec<String> = filtered.list().data.iter().map(|m| m.id.clone()).collect();
        assert_eq!(ids, vec!["anthropic.claude-3-v1:0".to_string()]);
        assert!(filtered.get("amazon.nova-pro-v1:0").is_none());
        assert!(filtered.get("anthropic.claude-3-v1:0").is_some());
    }

    #[test]
    fn apply_allow_list_matches_extra_model_ids() {
        let s = settings("fallback.model-v1:0");
        let models = [fm(
            "anthropic.claude-3-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        )];
        let catalog = assemble_catalog(&models, &[], &s)
            .with_extra_models(vec!["gpt-5.5".to_string(), "gpt-5.4".to_string()]);

        // A pattern that matches a mantle alias retains it; the non-matching
        // alias and the non-matching foundation model are dropped.
        let filtered = catalog.apply_allow_list(&["gpt-5.5".to_string()]);
        let ids: Vec<String> = filtered.list().data.iter().map(|m| m.id.clone()).collect();
        assert_eq!(ids, vec!["gpt-5.5".to_string()]);
        assert!(filtered.get("gpt-5.5").is_some());
        assert!(filtered.get("gpt-5.4").is_none());
        assert!(filtered.get("anthropic.claude-3-v1:0").is_none());
    }

    /// Live integration test — skipped unless `BEDROCK_INTEGRATION` is set.
    /// Exercises the real `refresh` against AWS (requires credentials + network).
    #[tokio::test]
    #[ignore = "requires live AWS Bedrock access; gated by BEDROCK_INTEGRATION"]
    async fn refresh_against_live_bedrock() {
        if std::env::var("BEDROCK_INTEGRATION").is_err() {
            return;
        }
        let settings = AppSettings::load().expect("settings load");
        let clients = crate::bedrock::client::BedrockClients::from_settings(&settings).await;
        let catalog = ModelCatalog::refresh(&clients.control, &settings)
            .await
            .expect("refresh should succeed against live Bedrock");
        assert!(
            !catalog.models().is_empty(),
            "live refresh should return at least the fallback model"
        );
    }
}
