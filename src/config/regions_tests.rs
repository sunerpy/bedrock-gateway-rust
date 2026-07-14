use super::*;

/// Path to the project's authored region config, relative to the crate root.
const REGIONS_TOML: &str = "config/regions.toml";

#[test]
fn test_project_regions_toml_is_empty_by_default() {
    // Python parity: MODEL_REGION_ROUTING = {} (bedrock.py:85). The shipped
    // file must contain no active routes — only the commented example.
    let cfg =
        RegionRoutingConfig::load(REGIONS_TOML).expect("config/regions.toml must load and parse");
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

#[test]
fn test_load_with_fallback_invalid_external_returns_embedded() {
    // Malformed external file (`model_id` must be a string) ⇒ WARN + embedded
    // fallback (which is empty for regions, Python parity).
    let dir = std::env::temp_dir();
    let path = dir.join(format!("bgw_regions_invalid_{}.toml", std::process::id()));
    std::fs::write(&path, "[[route]]\nmodel_id = 123\n").unwrap();
    let cfg = RegionRoutingConfig::load_with_fallback(Some(&path));
    std::fs::remove_file(&path).ok();
    assert_eq!(cfg, RegionRoutingConfig::load_embedded());
    assert_eq!(cfg.route_for("anything"), None);
}
