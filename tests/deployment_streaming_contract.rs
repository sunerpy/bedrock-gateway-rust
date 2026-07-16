//! Regression contract for long-lived SSE behind ECS Service Connect.
//!
//! AWS applies a 15-second HTTP per-request timeout when Service Connect is
//! enabled and `perRequestTimeoutSeconds` is omitted. That proxy timeout can
//! end a valid Responses stream while a tool call is still being generated.

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
