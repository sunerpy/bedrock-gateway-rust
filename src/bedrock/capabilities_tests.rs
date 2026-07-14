//! Unit tests for [`crate::bedrock::capabilities`], relocated out of the source
//! module for code organization (see the `test-coverage-codecov` spec).
//!
//! The source file declares this via a `#[path]` mod tests, so the
//! top-level `use super::*;` resolves to the implementation module.

use super::*;

/// Path to the project's authored model config, relative to the crate root.
const MODELS_TOML: &str = "config/models.toml";

fn caps() -> ConfigModelCapabilities {
    let config =
        ModelCapabilityConfig::load(MODELS_TOML).expect("config/models.toml must load and parse");
    ConfigModelCapabilities::new(config)
}

// A realistic full Bedrock model id embedding the opus-4-8 substring. The
// substring is the only thing the algorithm keys on; the surrounding id is
// arbitrary test input, not model knowledge encoded in production code.
const FULL_OPUS_4_8: &str = "global.anthropic.claude-opus-4-8-20251101-v1:0";
const FULL_SONNET_4_5: &str = "us.anthropic.claude-sonnet-4-5-20250101-v1:0";
const FULL_NOVA: &str = "us.amazon.nova-pro-v1:0";
const FULL_DEEPSEEK_V3: &str = "us.deepseek.v3-v1:0";

#[test]
fn opus_4_8_has_adaptive_thinking() {
    // Parity with Python MODEL_CAPABILITIES (bedrock.py:154):
    // "claude-opus-4-8": {"no_assistant_prefill", "adaptive_thinking"}
    let c = caps();
    assert!(c.has(FULL_OPUS_4_8, Capability::AdaptiveThinking));
    assert!(c.has(FULL_OPUS_4_8, Capability::NoAssistantPrefill));
    // opus-4-8 does NOT have temperature_topp_conflict.
    assert!(!c.has(FULL_OPUS_4_8, Capability::TemperatureToppConflict));
}

#[test]
fn sonnet_4_5_conflict_not_adaptive() {
    // Parity with Python MODEL_CAPABILITIES (bedrock.py:148):
    // "claude-sonnet-4-5": {"temperature_topp_conflict"}
    let c = caps();
    assert!(c.has(FULL_SONNET_4_5, Capability::TemperatureToppConflict));
    assert!(!c.has(FULL_SONNET_4_5, Capability::AdaptiveThinking));
    assert!(!c.has(FULL_SONNET_4_5, Capability::NoAssistantPrefill));
}

#[test]
fn sonnet_4_6_has_context_1m_beta() {
    // bedrock.py:149: {"temperature_topp_conflict", "context_1m_beta"}
    let c = caps();
    let id = "anthropic.claude-sonnet-4-6-20250601-v1:0";
    assert!(c.has(id, Capability::TemperatureToppConflict));
    assert!(c.has(id, Capability::Context1mBeta));
}

#[test]
fn full_capability_table_parity() {
    // Exhaustive parity with the Python MODEL_CAPABILITIES table
    // (bedrock.py:147-157). Each tuple is (substring, expected flag set).
    // The substrings here are TEST INPUT mirroring the Python keys; they
    // exercise the algorithm against the config DATA.
    use Capability::*;
    let c = caps();
    let table: &[(&str, &[Capability])] = &[
        ("claude-sonnet-4-5", &[TemperatureToppConflict]),
        (
            "claude-sonnet-4-6",
            &[TemperatureToppConflict, Context1mBeta],
        ),
        ("claude-haiku-4-5", &[TemperatureToppConflict]),
        ("claude-opus-4-5", &[TemperatureToppConflict]),
        (
            "claude-opus-4-6",
            &[Context1mBeta, NoAssistantPrefill, AdaptiveThinking],
        ),
        (
            "claude-opus-4-7",
            &[NoAssistantPrefill, AdaptiveThinking, DropSamplingParams],
        ),
        (
            "claude-opus-4-8",
            &[NoAssistantPrefill, AdaptiveThinking, DropSamplingParams],
        ),
        ("claude-mythos-5", &[AdaptiveThinking, DropSamplingParams]),
        (
            "claude-fable-5",
            &[NoAssistantPrefill, AdaptiveThinking, DropSamplingParams],
        ),
    ];
    let all = [
        TemperatureToppConflict,
        NoAssistantPrefill,
        AdaptiveThinking,
        DropSamplingParams,
        Context1mBeta,
    ];
    for (substr, expected) in table {
        // Wrap the substring in a realistic-looking id to prove substring
        // matching against arbitrary surrounding text.
        let full = format!("global.anthropic.{substr}-20250101-v1:0");
        for cap in all {
            let want = expected.contains(&cap);
            let got = c.has(&full, cap);
            assert_eq!(
                got, want,
                "capability {cap:?} for {full}: expected {want}, got {got}"
            );
        }
    }
}

#[test]
fn unknown_model_has_no_capabilities() {
    let c = caps();
    let id = "some.unknown.model-v1:0";
    assert!(!c.has(id, Capability::AdaptiveThinking));
    assert!(!c.has(id, Capability::TemperatureToppConflict));
    assert!(!c.has(id, Capability::Context1mBeta));
    assert!(!c.has(id, Capability::NoAssistantPrefill));
}

#[test]
fn resolve_foundation_passthrough_when_unknown() {
    // bedrock.py:415-417: unknown ids pass through unchanged.
    let c = caps();
    let id = "anthropic.claude-opus-4-8-20251101-v1:0";
    assert_eq!(c.resolve_foundation(id), id);
}

#[test]
fn alias_resolves_foundation_without_seeded_profile_map() {
    let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n\n[[alias]]\nfrom = \"gpt-5.4\"\nto = \"openai.gpt-5.4\"\n";
    let config = ModelCapabilityConfig::from_toml_str(raw).expect("alias config must parse");
    let c = ConfigModelCapabilities::new(config);

    assert_eq!(c.resolve_foundation("gpt-5.5"), "openai.gpt-5.5");
    assert_eq!(c.resolve_foundation("gpt-5.4"), "openai.gpt-5.4");
    assert_eq!(c.resolve_foundation("not-aliased"), "not-aliased");
}

#[test]
fn alias_wins_over_profile_map() {
    let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n";
    let config = ModelCapabilityConfig::from_toml_str(raw).expect("alias config must parse");
    let mut profiles = HashMap::new();
    profiles.insert("gpt-5.5".to_string(), "profile.wrong-target".to_string());
    let c = ConfigModelCapabilities::with_profiles(config, profiles);

    assert_eq!(c.resolve_foundation("gpt-5.5"), "openai.gpt-5.5");
}

#[test]
fn profile_map_resolves_to_foundation_and_drives_lookup() {
    // Seed a profile id → foundation id mapping; capability lookup must use
    // the resolved foundation id (bedrock.py:415-417, 1130-1131).
    let config = ModelCapabilityConfig::load(MODELS_TOML).unwrap();
    let mut profiles = HashMap::new();
    profiles.insert("some.profile.id".to_string(), FULL_OPUS_4_8.to_string());
    let c = ConfigModelCapabilities::with_profiles(config, profiles);

    // resolve_foundation maps the profile to the underlying model.
    assert_eq!(c.resolve_foundation("some.profile.id"), FULL_OPUS_4_8);

    // Capability lookup on the PROFILE id resolves through to opus-4-8.
    assert!(c.has("some.profile.id", Capability::AdaptiveThinking));
    assert!(c.has("some.profile.id", Capability::NoAssistantPrefill));
    assert!(!c.has("some.profile.id", Capability::TemperatureToppConflict));

    // reasoning_path also follows the resolved foundation model.
    assert_eq!(
        c.reasoning_path("some.profile.id"),
        ReasoningPath::AdaptiveThinking
    );
}

#[test]
fn budget_ratios_returns_configured_defaults() {
    // bedrock.py:1679-1689: low=0.3, medium=0.6, high=max_tokens-1 (sentinel
    // -1). No model entry overrides budget_ratios, so the `default` entry
    // supplies them for every model.
    let c = caps();
    let ratios = c
        .budget_ratios(FULL_SONNET_4_5)
        .expect("budget ratios must resolve from the default entry");
    assert_eq!(ratios.low, 0.3);
    assert_eq!(ratios.medium, 0.6);
    assert_eq!(ratios.high, -1.0);

    // Even an unknown model falls back to the config defaults.
    let unknown = c
        .budget_ratios("vendor.totally-unknown-v1:0")
        .expect("unknown model still gets default budget ratios");
    assert_eq!(unknown.low, 0.3);
}

#[test]
fn max_cache_tokens_nova_ceiling_from_config() {
    // bedrock.py:454,470: Nova models have a 20,000-token caching limit.
    let c = caps();
    assert_eq!(c.max_cache_tokens(FULL_NOVA), Some(20_000));
    // Claude has no configured ceiling.
    assert_eq!(c.max_cache_tokens(FULL_OPUS_4_8), None);
    // Unknown model: None.
    assert_eq!(c.max_cache_tokens("vendor.unknown-v1:0"), None);
}

#[test]
fn reasoning_path_parity_from_config() {
    let c = caps();
    // claude + adaptive_thinking → AdaptiveThinking (bedrock.py:1168-1172).
    assert_eq!(
        c.reasoning_path(FULL_OPUS_4_8),
        ReasoningPath::AdaptiveThinking
    );
    // claude non-adaptive → BudgetTokens (bedrock.py:1173-1177).
    assert_eq!(
        c.reasoning_path(FULL_SONNET_4_5),
        ReasoningPath::BudgetTokens
    );
    // deepseek.v3 → DeepseekString (bedrock.py:1178-1185).
    assert_eq!(
        c.reasoning_path(FULL_DEEPSEEK_V3),
        ReasoningPath::DeepseekString
    );
    // Unknown → None.
    assert_eq!(c.reasoning_path("vendor.unknown-v1:0"), ReasoningPath::None);
}

#[test]
fn cache_min_tokens_and_beta_headers_from_config() {
    // Claude Opus 4-8 configures a 4096 cache_min_tokens floor in config.
    let c = caps();
    assert_eq!(c.cache_min_tokens(FULL_OPUS_4_8), Some(4096));
    assert!(c.beta_headers(FULL_OPUS_4_8).is_empty());
}

#[test]
fn max_cache_checkpoints_set_returns_value_unset_returns_none() {
    let raw = "[[model]]\nmatch = \"vendor.capped\"\n[model.params]\nmax_cache_checkpoints = 3\n\n[[model]]\nmatch = \"vendor.uncapped\"\n";
    let config = ModelCapabilityConfig::from_toml_str(raw).expect("inline config must parse");
    let c = ConfigModelCapabilities::new(config);

    assert_eq!(
        c.max_cache_checkpoints("global.vendor.capped-v1:0"),
        Some(3)
    );
    assert_eq!(c.max_cache_checkpoints("global.vendor.uncapped-v1:0"), None);
    assert_eq!(c.max_cache_checkpoints("vendor.unknown-v1:0"), None);
}

#[test]
fn min_budget_tokens_default_and_override_from_config() {
    // The `default` entry supplies the 1024 protocol floor for any model
    // that does not override it (parity with build_reasoning_config's
    // fallback). A per-entry override wins over the default.
    let c = caps();
    assert_eq!(c.min_budget_tokens(FULL_SONNET_4_5), Some(1024));
    assert_eq!(
        c.min_budget_tokens("vendor.totally-unknown-v1:0"),
        Some(1024)
    );

    let raw = "[[model]]\nmatch = \"default\"\n[model.params]\nmin_budget_tokens = 1024\n\n[[model]]\nmatch = \"vendor.bigfloor\"\n[model.params]\nmin_budget_tokens = 4096\n";
    let config = ModelCapabilityConfig::from_toml_str(raw).expect("inline config must parse");
    let c2 = ConfigModelCapabilities::new(config);
    assert_eq!(
        c2.min_budget_tokens("global.vendor.bigfloor-v1:0"),
        Some(4096)
    );
    // A model with no override falls back to the default entry.
    assert_eq!(c2.min_budget_tokens("global.vendor.other-v1:0"), Some(1024));
}

#[test]
fn usable_via_dyn_trait_object() {
    // Object-safety: the resolver works behind the domain trait object.
    let c = caps();
    let dynamic: &dyn ModelCapabilities = &c;
    assert!(dynamic.has(FULL_OPUS_4_8, Capability::AdaptiveThinking));
}

#[test]
fn normalize_for_match_strips_geo_prefix_and_lowercases() {
    let bare = "anthropic.claude-opus-4-8-20251101-v1:0";
    // The complete enumerated GEO_PREFIXES set all strip to the same bare
    // form (the leading geo prefix is removed; the vendor segment stays).
    for prefix in ["us.", "eu.", "apac.", "global.", "jp.", "au.", "ca."] {
        let prefixed = format!("{prefix}{bare}");
        assert_eq!(normalize_for_match(&prefixed), bare);
        // Uppercased input is lowercased and still stripped.
        assert_eq!(normalize_for_match(&prefixed.to_uppercase()), bare);
    }
    // A bare id has no geo prefix: lowercased, otherwise unchanged.
    assert_eq!(normalize_for_match(bare), bare);
    assert_eq!(
        normalize_for_match("ANTHROPIC.CLAUDE-X"),
        "anthropic.claude-x"
    );
}

#[test]
fn normalize_for_match_covers_full_real_prefix_set_to_same_foundation() {
    // 34-region live scan (2026-06-20): global/us/eu/apac/jp/au/ca — the
    // complete enumerated set — all normalize a cross-region claude profile
    // down to the identical bare foundation id.
    let bare = "anthropic.claude-sonnet-4-5-20250101-v1:0";
    for prefix in ["global.", "us.", "eu.", "apac.", "jp.", "au.", "ca."] {
        assert_eq!(
            normalize_for_match(&format!("{prefix}{bare}")),
            bare,
            "{prefix} must normalize to the bare foundation id"
        );
    }
}

#[test]
fn normalize_for_match_never_strips_bare_provider_segment() {
    // Bare vendor forms must pass through unchanged — the leading segment
    // is a provider vendor, not a GEO_PREFIXES entry, so nothing is stripped.
    assert_eq!(
        normalize_for_match("amazon.nova-pro-v1:0"),
        "amazon.nova-pro-v1:0"
    );
    assert_eq!(normalize_for_match("deepseek.v3-v1:0"), "deepseek.v3-v1:0");
    assert_eq!(
        normalize_for_match("anthropic.claude-x"),
        "anthropic.claude-x"
    );
    assert_eq!(normalize_for_match("cohere.embed-v3"), "cohere.embed-v3");
}

#[test]
fn normalize_for_match_leaves_unlisted_prefix_intact() {
    // A prefix NOT in the enumerated GEO_PREFIXES set is left intact
    // (lowercased only). The set is a closed enumeration: any future AWS geo
    // prefix must be added to GEO_PREFIXES to be normalized.
    assert_eq!(
        normalize_for_match("xx.anthropic.claude-future-v1:0"),
        "xx.anthropic.claude-future-v1:0"
    );
    assert_eq!(normalize_for_match("foo.bar.baz"), "foo.bar.baz");
}

#[test]
fn matching_entry_identical_across_all_cross_region_forms() {
    let c = caps();
    let bare = "anthropic.claude-opus-4-8-20251101-v1:0";
    let baseline = c
        .matching_entry(bare)
        .expect("bare claude-opus-4-8 must match an entry")
        .match_pattern
        .clone();
    for prefix in ["us.", "eu.", "apac.", "global.", "jp.", "au.", "ca."] {
        let prefixed = format!("{prefix}{bare}");
        let got = c
            .matching_entry(&prefixed)
            .expect("cross-region form must match the same entry")
            .match_pattern
            .clone();
        assert_eq!(got, baseline, "form {prefixed} matched a different entry");
    }
}

#[test]
fn resolve_foundation_unchanged_by_normalization() {
    // Regression lock: C2 normalizes ONLY the match path. resolve_foundation
    // still passes unknown ids through verbatim (geo prefix preserved).
    let c = caps();
    let cross_region = "us.anthropic.claude-opus-4-8-20251101-v1:0";
    assert_eq!(c.resolve_foundation(cross_region), cross_region);
    assert_eq!(c.resolve_foundation(FULL_NOVA), FULL_NOVA);
}

#[test]
fn cache_min_tokens_per_claude_version_floors() {
    // C3 regression lock: the per-version cache_min_tokens floors. The
    // 4.5-gen Sonnet/Opus/Haiku carry the real AWS-doc 4096 floor; opus-4-6
    // (synthetic) takes the conservative 4096; sonnet-4-6 keeps 1024.
    let c = caps();
    let floor = |substr: &str| {
        let full = format!("global.anthropic.{substr}-20250101-v1:0");
        c.cache_min_tokens(&full)
    };
    assert_eq!(floor("claude-sonnet-4-5"), Some(4096));
    assert_eq!(floor("claude-opus-4-5"), Some(4096));
    assert_eq!(floor("claude-haiku-4-5"), Some(4096));
    assert_eq!(floor("claude-opus-4-6"), Some(4096));
    // sonnet-4-6 is intentionally left at the 1024 floor.
    assert_eq!(floor("claude-sonnet-4-6"), Some(1024));
}

#[test]
fn unlisted_claude_id_hits_family_catch_all() {
    // C3 family catch-all: an unlisted anthropic.claude-* id falls through
    // to the `anthropic.claude` family entry, gaining caching support with
    // the conservative 4096 floor. supports_caching is the production gate.
    use crate::bedrock::cache::supports_caching;
    let c = caps();
    for id in [
        "anthropic.claude-sonnet-5-0-future-20260101-v1:0",
        "anthropic.claude-future-99-v1:0",
    ] {
        assert!(
            supports_caching(id, &c),
            "{id} should support caching via family catch-all"
        );
        assert_eq!(c.cache_min_tokens(id), Some(4096), "{id} family floor");
    }
}

#[test]
fn non_claude_models_get_no_family_false_positive() {
    // C3: the family catch-all must NOT widen caching to non-Claude models.
    use crate::bedrock::cache::supports_caching;
    let c = caps();
    for id in ["zai.glm-5", "deepseek.v3", "qwen.qwen3-235b-a22b-2507-v1:0"] {
        assert!(
            !supports_caching(id, &c),
            "{id} must NOT support caching (no family false-positive)"
        );
        assert_eq!(c.cache_min_tokens(id), None, "{id} must have no floor");
    }
}

#[test]
fn cross_region_unlisted_claude_hits_family_catch_all() {
    // C3 + C2: every cross-region form of an unlisted Claude id normalizes
    // to `anthropic.claude-…` and hits the family entry.
    use crate::bedrock::cache::supports_caching;
    let c = caps();
    let bare = "anthropic.claude-future-99-v1:0";
    for prefix in ["us.", "eu.", "apac.", "global."] {
        let prefixed = format!("{prefix}{bare}");
        assert!(
            supports_caching(&prefixed, &c),
            "{prefixed} should hit the family catch-all"
        );
        assert_eq!(
            c.cache_min_tokens(&prefixed),
            Some(4096),
            "{prefixed} family floor"
        );
    }
}

#[test]
fn responses_backend_mantle_via_canonical_and_alias() {
    use crate::domain::ResponsesBackend;
    let c = caps();
    // Canonical id and the bare alias both route to Mantle. The alias path
    // transitively re-tests T1's resolve_foundation("gpt-5.5") resolution.
    assert_eq!(
        c.responses_backend("openai.gpt-5.5"),
        ResponsesBackend::Mantle
    );
    assert_eq!(c.responses_backend("gpt-5.5"), ResponsesBackend::Mantle);
    assert_eq!(
        c.responses_backend("openai.gpt-5.4"),
        ResponsesBackend::Mantle
    );
    assert_eq!(c.responses_backend("gpt-5.4"), ResponsesBackend::Mantle);
}

#[test]
fn responses_backend_converse_for_non_mantle_and_unknown() {
    use crate::domain::ResponsesBackend;
    let c = caps();
    assert_eq!(
        c.responses_backend(FULL_SONNET_4_5),
        ResponsesBackend::Converse
    );
    assert_eq!(c.responses_backend(FULL_NOVA), ResponsesBackend::Converse);
    assert_eq!(
        c.responses_backend("vendor.totally-unknown-v1:0"),
        ResponsesBackend::Converse
    );
}

#[test]
fn model_regions_from_config_via_canonical_and_alias() {
    let c = caps();
    // gpt-5.5 is gated to a single region; the alias resolves first.
    assert_eq!(
        c.model_regions("gpt-5.5"),
        Some(vec!["us-east-2".to_string()])
    );
    // gpt-5.4 allows two regions, including us-west-2.
    let regions = c
        .model_regions("gpt-5.4")
        .expect("gpt-5.4 must declare a region allow-list");
    assert!(regions.contains(&"us-west-2".to_string()));
    // A model with no region gate, and an unknown model, both return None.
    assert_eq!(c.model_regions(FULL_SONNET_4_5), None);
    assert_eq!(c.model_regions("vendor.totally-unknown-v1:0"), None);
}
