//! Regression contract for long-lived SSE behind ECS Service Connect.
//!
//! AWS applies a 15-second HTTP per-request timeout when Service Connect is
//! enabled and `perRequestTimeoutSeconds` is omitted. That proxy timeout can
//! end a valid Responses stream while a tool call is still being generated.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn validate(path: &Path) -> Output {
    Command::new("bash")
        .arg(root().join("scripts/check-ecs-service-connect-timeouts.sh"))
        .arg(path)
        .output()
        .expect("run Service Connect timeout validator")
}

fn model_match_patterns(path: &Path) -> BTreeSet<String> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read model registry {}: {error}", path.display()));
    let registry: toml::Value = toml::from_str(&contents)
        .unwrap_or_else(|error| panic!("parse model registry {}: {error}", path.display()));
    let models = registry
        .get("model")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("model registry {} has no [[model]] entries", path.display()));

    models
        .iter()
        .map(|model| {
            model
                .get("match")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "model registry {} has a [[model]] entry without a string `match`",
                        path.display()
                    )
                })
                .to_owned()
        })
        .collect()
}

#[test]
fn rejects_service_connects_implicit_fifteen_second_request_timeout() {
    let output = validate(&root().join("tests/fixtures/ecs_service_connect_unsafe.json"));

    assert_eq!(
        output.status.code(),
        Some(1),
        "unsafe config must fail the policy check, not the validator runtime: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("15-second HTTP default"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("perRequestTimeoutSeconds"),
        "stderr: {stderr}"
    );
}

#[test]
fn accepts_versioned_streaming_safe_service_connect_config() {
    let output = validate(&root().join("deployment/service-connect-streaming.json"));

    assert!(
        output.status.success(),
        "safe config failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("total request timeout is disabled"));
}

#[test]
fn helm_model_registry_contains_every_config_model_match_pattern() {
    let config_path = root().join("config/models.toml");
    let helm_path = root().join("helm/bedrock-gateway/files/models.toml");
    let config_patterns = model_match_patterns(&config_path);
    let helm_patterns = model_match_patterns(&helm_path);
    let missing: Vec<_> = config_patterns
        .difference(&helm_patterns)
        .cloned()
        .collect();

    assert!(
        missing.is_empty(),
        "{} is missing model `match` pattern(s) declared in {}:\n  - {}\nUpdate the Helm model registry whenever config/models.toml declares a model.",
        helm_path.display(),
        config_path.display(),
        missing.join("\n  - ")
    );
}
