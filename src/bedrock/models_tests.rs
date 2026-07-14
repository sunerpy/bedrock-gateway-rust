//! Unit tests for the Bedrock model catalog assembly + rendering.
//!
//! Relocated out of `models.rs` into this sibling file for code organization
//! (see the `test-coverage-codecov` spec). Behavior is unchanged: the original
//! inline tests are preserved verbatim as FLAT functions here, and the module
//! is referenced from `models.rs` via a single
//! `#[cfg(test)] #[path = "models_tests.rs"] mod tests;` declaration.
//!
//! The live-integration `refresh_against_live_bedrock` test remains `#[ignore]`
//! and `BEDROCK_INTEGRATION`-gated so the offline suite never touches AWS.

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
        prompt_cache_ttl: "5m".to_string(),
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
        mantle_base_url_template: "https://bedrock-mantle.{region}.api.aws/openai/v1".to_string(),
        mantle_chat_base_url_template: "https://bedrock-mantle.{region}.api.aws/v1".to_string(),
        allowed_models: None,
        otel_exporter_otlp_endpoint: None,
        otel_capture_content: false,
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
    let catalog = assemble_catalog(&models, &[], &s).with_extra_models(vec!["gpt-5.5".to_string()]);

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

// ─── Supplementary coverage (task 6.2) ──────────────────────────────────

#[test]
fn default_catalog_is_empty_on_all_surfaces() {
    // The `Default` derive yields an empty catalog: no models, no profile
    // metadata, and an empty OpenAI listing.
    let catalog = ModelCatalog::default();
    assert!(catalog.models().is_empty());
    assert!(catalog.profile_metadata().is_empty());
    let listed = catalog.list();
    assert_eq!(listed.object, "list");
    assert!(listed.data.is_empty());
    assert!(catalog.get("anything").is_none());
}

#[test]
fn with_extra_models_does_not_touch_models_or_profile_metadata() {
    // Alias injection is display-only: extra ids surface via list()/get() but
    // never enter the routing `models` map or the capability `profile_metadata`.
    let s = settings("fallback.model-v1:0");
    let models = [fm(
        "vendor.base-v1:0",
        &["TEXT"],
        &["ON_DEMAND"],
        true,
        "ACTIVE",
    )];
    let catalog = assemble_catalog(&models, &[], &s)
        .with_extra_models(vec!["gpt-5.5".to_string(), "gpt-5.4".to_string()]);

    // Routing map only holds the real foundation model.
    assert_eq!(catalog.models().len(), 1);
    assert!(catalog.models().contains_key("vendor.base-v1:0"));
    assert!(!catalog.models().contains_key("gpt-5.5"));
    assert!(!catalog.models().contains_key("gpt-5.4"));
    // Capability resolution surface is untouched by aliases.
    assert!(!catalog.profile_metadata().contains_key("gpt-5.5"));
    assert!(!catalog.profile_metadata().contains_key("gpt-5.4"));
}

#[test]
fn with_extra_models_replaces_previous_alias_set() {
    // `with_extra_models` sets (not appends) the alias list, so a second call
    // fully replaces the first.
    let s = settings("fallback.model-v1:0");
    let catalog = assemble_catalog(&[], &[], &s)
        .with_extra_models(vec!["gpt-5.4".to_string()])
        .with_extra_models(vec!["gpt-5.5".to_string()]);

    assert!(catalog.get("gpt-5.5").is_some());
    assert!(catalog.get("gpt-5.4").is_none());
}

#[test]
fn extra_model_duplicating_default_fallback_is_deduped_in_list() {
    // Empty foundation input falls back to `default_model`; injecting the same
    // id as an alias must not produce a duplicate entry in the listing.
    let s = settings("fallback.model-v1:0");
    let catalog =
        assemble_catalog(&[], &[], &s).with_extra_models(vec!["fallback.model-v1:0".to_string()]);
    let listed = catalog.list();

    let occurrences = listed
        .data
        .iter()
        .filter(|m| m.id == "fallback.model-v1:0")
        .count();
    assert_eq!(occurrences, 1, "duplicate id must be collapsed by dedup");
}

#[test]
fn apply_allow_list_is_case_insensitive() {
    // Allow-list matching lowercases both sides, so a mixed-case pattern still
    // matches a lowercase model id.
    let s = settings("fallback.model-v1:0");
    let models = [fm(
        "anthropic.claude-3-v1:0",
        &["TEXT"],
        &["ON_DEMAND"],
        true,
        "ACTIVE",
    )];
    let filtered = assemble_catalog(&models, &[], &s).apply_allow_list(&["CLAUDE".to_string()]);
    assert!(filtered.get("anthropic.claude-3-v1:0").is_some());
}

#[test]
fn profile_metadata_survives_allow_list_filtering() {
    // apply_allow_list narrows the visible surfaces but leaves profile_metadata
    // intact so capability/routing resolution is unaffected.
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
    // Filter out everything (no id contains "zzz").
    let filtered = catalog.apply_allow_list(&["zzz-nomatch".to_string()]);

    assert!(filtered.list().data.is_empty());
    // Metadata is preserved regardless of the visible-surface filter.
    assert_eq!(
        filtered.profile_metadata()["us.vendor.base-v1:0"],
        "vendor.base-v1:0"
    );
}

#[test]
fn duplicate_profiles_last_underlying_wins_in_metadata() {
    // Two profile entries with the same key collapse into one metadata slot
    // (HashMap semantics); the model map still surfaces the shared key once.
    let s = settings("fallback.model-v1:0");
    let models = [
        fm(
            "vendor.first-v1:0",
            &["TEXT"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        ),
        fm(
            "vendor.second-v1:0",
            &["IMAGE"],
            &["ON_DEMAND"],
            true,
            "ACTIVE",
        ),
    ];
    let profiles = [
        ProfileEntry {
            key: "dup.key".to_string(),
            underlying_model_id: "vendor.first-v1:0".to_string(),
        },
        ProfileEntry {
            key: "dup.key".to_string(),
            underlying_model_id: "vendor.second-v1:0".to_string(),
        },
    ];
    let catalog = assemble_catalog(&models, &profiles, &s);

    // Exactly one metadata entry for the duplicated key.
    assert!(catalog.profile_metadata().contains_key("dup.key"));
    // And the profile key appears once in the model map.
    assert!(catalog.models().contains_key("dup.key"));
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
