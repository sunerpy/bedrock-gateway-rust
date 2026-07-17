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
        Capability::StructuredOutput,
    ]
    .into_iter()
    .collect();
    assert_eq!(caps, expected);
}

#[test]
fn test_sonnet_4_5_capabilities() {
    // Parity with Python MODEL_CAPABILITIES (bedrock.py:148):
    // "claude-sonnet-4-5": {"temperature_topp_conflict"}. The 4.5-gen family
    // also declares cache_ttl_1h (PR-G): 1h prompt-cache retention support.
    let cfg = load_project_config();
    let entry = cfg
        .entry_for_match("claude-sonnet-4-5")
        .expect("claude-sonnet-4-5 entry must exist");
    let caps: HashSet<Capability> = entry.capabilities.iter().copied().collect();
    let expected: HashSet<Capability> = [
        Capability::TemperatureToppConflict,
        Capability::StructuredOutput,
        Capability::CacheTtl1h,
    ]
    .into_iter()
    .collect();
    assert_eq!(caps, expected);
}

#[test]
fn cache_ttl_1h_capability_parses() {
    let cfg = ModelCapabilityConfig::from_toml_str(
        "[[model]]\nmatch = \"x.y\"\ncapabilities = [\"cache_ttl_1h\"]\n",
    )
    .expect("config with cache_ttl_1h must parse");
    let entry = cfg.entry_for_match("x.y").unwrap();
    assert!(entry.has_capability(Capability::CacheTtl1h));
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
fn reasoning_path_signature_replay_requirements() {
    assert!(ReasoningPath::AdaptiveThinking.requires_signature_replay());
    assert!(ReasoningPath::BudgetTokens.requires_signature_replay());
    assert!(!ReasoningPath::DeepseekString.requires_signature_replay());
    assert!(!ReasoningPath::None.requires_signature_replay());
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
    let cfg = ModelCapabilityConfig::from_toml_str(&extended).expect("extended config must parse");
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
fn structured_output_capability_parses() {
    let cfg = ModelCapabilityConfig::from_toml_str(
        "[[model]]\nmatch = \"x.y\"\ncapabilities = [\"structured_output\"]\n",
    )
    .expect("config with structured_output must parse");
    let entry = cfg.entry_for_match("x.y").unwrap();
    assert!(entry.has_capability(Capability::StructuredOutput));
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
fn test_load_with_fallback_invalid_external_returns_embedded() {
    // External file present but malformed (`match` must be a string) ⇒ the
    // loader logs a WARN and falls back to the embedded default, never to an
    // empty registry that would silently disable every capability.
    let dir = std::env::temp_dir();
    let path = dir.join(format!("bgw_models_invalid_{}.toml", std::process::id()));
    std::fs::write(&path, "[[model]]\nmatch = 123\n").unwrap();
    let cfg = ModelCapabilityConfig::load_with_fallback(Some(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        !cfg.models.is_empty(),
        "must fall back to non-empty embedded"
    );
    assert_eq!(cfg, ModelCapabilityConfig::load_embedded());
}

#[test]
fn test_aliases_and_responses_backend_and_regions_parse() {
    let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n\n[[alias]]\nfrom = \"gpt-5.4\"\nto = \"openai.gpt-5.4\"\n\n[[model]]\nmatch = \"openai.gpt-5.5\"\n[model.params]\nresponses_backend = \"mantle\"\navailable_regions = [\"us-east-2\"]\n";
    let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config with aliases must parse");

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
fn mantle_alias_names_includes_chat_only_model() {
    // A chat-only mantle model (chat_backend, NO responses_backend) surfaces its
    // bare alias name.
    let raw = "[[alias]]\nfrom = \"gpt-oss-120b\"\nto = \"openai.gpt-oss-120b\"\n\n[[model]]\nmatch = \"openai.gpt-oss-120b\"\n[model.params]\nchat_backend = \"mantle\"\n";
    let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config must parse");
    let names: HashSet<String> = cfg.mantle_alias_names().into_iter().collect();
    assert!(names.contains("gpt-oss-120b"));
}

#[test]
fn mantle_alias_names_still_includes_responses_models() {
    // Regression: a responses-backend mantle model is still surfaced.
    let raw = "[[alias]]\nfrom = \"gpt-5.5\"\nto = \"openai.gpt-5.5\"\n\n[[model]]\nmatch = \"openai.gpt-5.5\"\n[model.params]\nresponses_backend = \"mantle\"\n";
    let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config must parse");
    let names: HashSet<String> = cfg.mantle_alias_names().into_iter().collect();
    assert!(names.contains("gpt-5.5"));
}

#[test]
fn test_mantle_alias_names_empty_without_mantle_entries() {
    // Aliases that do not resolve to any mantle-backed entry surface nothing.
    let raw = "[[alias]]\nfrom = \"sonnet\"\nto = \"anthropic.claude-sonnet-4-5\"\n\n[[model]]\nmatch = \"anthropic.claude-sonnet-4-5\"\n";
    let cfg = ModelCapabilityConfig::from_toml_str(raw).expect("config must parse");
    assert!(cfg.mantle_alias_names().is_empty());
}

#[test]
fn test_gpt_5_6_trio_aliases_and_region_gate() {
    let cfg = load_project_config();
    for (alias, canonical, regions) in [
        (
            "gpt-5.6-sol",
            "openai.gpt-5.6-sol",
            &["us-east-1", "us-east-2"][..],
        ),
        (
            "gpt-5.6-terra",
            "openai.gpt-5.6-terra",
            &["us-east-1", "us-east-2", "us-west-2"][..],
        ),
        (
            "gpt-5.6-luna",
            "openai.gpt-5.6-luna",
            &["us-east-1", "us-east-2", "us-west-2"][..],
        ),
    ] {
        let resolved = cfg
            .aliases
            .iter()
            .find(|a| a.from == alias)
            .map(|a| a.to.as_str())
            .unwrap_or_else(|| panic!("{alias} alias must exist"));
        assert_eq!(resolved, canonical);
        let entry = cfg
            .entry_for_match(canonical)
            .unwrap_or_else(|| panic!("{canonical} model entry must resolve"));
        assert_eq!(entry.params.responses_backend.as_deref(), Some("mantle"));
        let available = entry
            .params
            .available_regions
            .as_deref()
            .expect("available_regions must be set");
        let got: HashSet<&str> = available.iter().map(String::as_str).collect();
        let want: HashSet<&str> = regions.iter().copied().collect();
        assert_eq!(got, want);
        assert!(
            !got.contains("eu-west-1"),
            "region gate must reject an unlisted region"
        );
    }
}
