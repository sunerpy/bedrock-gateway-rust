//! The binary's own flags, run as a subprocess the way an operator or a
//! container health check runs them.

use std::process::Command;

/// `--version` and `-V` print the binary name and the crate version and exit
/// with 0 before any configuration is read, so they work without an API key.
/// The release workflow runs `--version` on every packaged binary and compares
/// the line with the version being released.
#[test]
fn version_flags_print_the_name_and_version_without_configuration() {
    for flag in ["--version", "-V"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bedrock-gateway"))
            .arg(flag)
            .env_remove("API_KEY")
            .env_remove("API_KEY_SECRET_ARN")
            .env_remove("API_KEY_PARAM_NAME")
            .output()
            .unwrap_or_else(|error| panic!("run bedrock-gateway {flag}: {error}"));
        assert!(
            output.status.success(),
            "{flag} exited with {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim_end(),
            format!("bedrock-gateway {}", env!("CARGO_PKG_VERSION")),
            "{flag}"
        );
    }
}
