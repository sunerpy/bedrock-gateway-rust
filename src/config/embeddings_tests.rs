use super::*;

/// Path to the project's authored embedding config, relative to crate root.
const EMBEDDINGS_TOML: &str = "config/embeddings.toml";

fn load_project_config() -> EmbeddingRegistry {
    EmbeddingRegistry::load(EMBEDDINGS_TOML).expect("config/embeddings.toml must load and parse")
}

#[test]
fn test_loads_project_embeddings_toml() {
    let cfg = load_project_config();
    assert!(
        !cfg.models.is_empty(),
        "expected at least one [[model]] entry in {}",
        EMBEDDINGS_TOML
    );
}

#[test]
fn test_cohere_family() {
    // bedrock.py:123 "cohere.embed-multilingual-v3" → Cohere body codec.
    let cfg = load_project_config();
    assert_eq!(
        cfg.family_for("cohere.embed-multilingual-v3"),
        Some(EmbeddingFamily::Cohere)
    );
    assert_eq!(
        cfg.family_for("cohere.embed-english-v3"),
        Some(EmbeddingFamily::Cohere)
    );
}

#[test]
fn test_titan_family_with_colon_id() {
    // bedrock.py:126 "amazon.titan-embed-text-v2:0" — id contains a colon and
    // must round-trip through TOML cleanly.
    let cfg = load_project_config();
    assert_eq!(
        cfg.family_for("amazon.titan-embed-text-v2:0"),
        Some(EmbeddingFamily::Titan)
    );
    assert_eq!(
        cfg.family_for("amazon.titan-embed-text-v1"),
        Some(EmbeddingFamily::Titan)
    );
}

#[test]
fn test_nova_family() {
    // bedrock.py:129 "amazon.nova-2-multimodal-embeddings-v1:0" → Nova codec.
    let cfg = load_project_config();
    assert_eq!(
        cfg.family_for("amazon.nova-2-multimodal-embeddings-v1:0"),
        Some(EmbeddingFamily::Nova)
    );
}

#[test]
fn test_unknown_model_is_none_no_panic() {
    let cfg = load_project_config();
    assert_eq!(cfg.family_for("nonexistent.model"), None);
    assert!(cfg.entry_for("nonexistent.model").is_none());
}

#[test]
fn test_entry_for_carries_display_name() {
    let cfg = load_project_config();
    let entry = cfg
        .entry_for("amazon.titan-embed-text-v1")
        .expect("titan v1 entry must exist");
    assert_eq!(entry.display_name, "Titan Embeddings G1 - Text");
    assert_eq!(entry.family, EmbeddingFamily::Titan);
}

#[test]
fn test_all_python_models_present() {
    // Parity with SUPPORTED_BEDROCK_EMBEDDING_MODELS (bedrock.py:122-130).
    let cfg = load_project_config();
    for id in [
        "cohere.embed-multilingual-v3",
        "cohere.embed-english-v3",
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-text-v2:0",
        "amazon.nova-2-multimodal-embeddings-v1:0",
    ] {
        assert!(
            cfg.entry_for(id).is_some(),
            "expected model id {id} to be registered"
        );
    }
}

#[test]
fn test_family_serde_lowercase() {
    // Family round-trips as lowercase strings in TOML.
    let cfg = EmbeddingRegistry::from_toml_str(
        "[[model]]\nmodel_id = \"x.y:0\"\ndisplay_name = \"X\"\nfamily = \"nova\"\n",
    )
    .unwrap();
    assert_eq!(cfg.family_for("x.y:0"), Some(EmbeddingFamily::Nova));
}

#[test]
fn test_extension_requires_toml_only() {
    // A NEW embedding model can be added by editing TOML alone — no code
    // change, no recompile of the schema.
    let base = std::fs::read_to_string(EMBEDDINGS_TOML).unwrap();
    let extended = format!(
            "{base}\n\n[[model]]\nmodel_id = \"vendor.future-embed-v9:0\"\ndisplay_name = \"Future Embed\"\nfamily = \"titan\"\n"
        );
    let cfg = EmbeddingRegistry::from_toml_str(&extended).expect("extended config must parse");
    assert_eq!(
        cfg.family_for("vendor.future-embed-v9:0"),
        Some(EmbeddingFamily::Titan)
    );
}

#[test]
fn test_load_missing_file_errors() {
    let err = EmbeddingRegistry::load("config/__does_not_exist__.toml");
    assert!(err.is_err());
}

#[test]
fn test_load_embedded_is_non_empty_with_family() {
    let cfg = EmbeddingRegistry::load_embedded();
    assert!(
        !cfg.models.is_empty(),
        "embedded embeddings.toml must be non-empty"
    );
    assert_eq!(
        cfg.family_for("cohere.embed-multilingual-v3"),
        Some(EmbeddingFamily::Cohere)
    );
}

#[test]
fn test_load_with_fallback_none_returns_embedded() {
    let cfg = EmbeddingRegistry::load_with_fallback(None);
    assert_eq!(cfg, EmbeddingRegistry::load_embedded());
}

#[test]
fn test_load_with_fallback_missing_path_returns_embedded() {
    let missing = Path::new("config/__does_not_exist__.toml");
    let cfg = EmbeddingRegistry::load_with_fallback(Some(missing));
    assert!(!cfg.models.is_empty());
}

#[test]
fn test_load_with_fallback_external_file_wins() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("bgw_embeddings_test_{}.toml", std::process::id()));
    std::fs::write(
            &path,
            "[[model]]\nmodel_id = \"external.only-embed:0\"\ndisplay_name = \"Ext\"\nfamily = \"nova\"\n",
        )
        .unwrap();
    let cfg = EmbeddingRegistry::load_with_fallback(Some(&path));
    std::fs::remove_file(&path).ok();
    assert_eq!(
        cfg.family_for("external.only-embed:0"),
        Some(EmbeddingFamily::Nova)
    );
}

#[test]
fn test_load_with_fallback_invalid_external_returns_embedded() {
    // Malformed external file (unknown `family` variant) ⇒ WARN + embedded
    // fallback (non-empty), never an empty registry.
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "bgw_embeddings_invalid_{}.toml",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "[[model]]\nmodel_id = \"x.y:0\"\ndisplay_name = \"X\"\nfamily = \"unknown_family\"\n",
    )
    .unwrap();
    let cfg = EmbeddingRegistry::load_with_fallback(Some(&path));
    std::fs::remove_file(&path).ok();
    assert!(
        !cfg.models.is_empty(),
        "must fall back to non-empty embedded"
    );
    assert_eq!(cfg, EmbeddingRegistry::load_embedded());
}
