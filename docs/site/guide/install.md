# Install

bedrock-gateway ships as a container image, as a binary for each platform, and as a crate. All three run the same
program with the same configuration. Pick the one that fits where it will run.

## Container image

The image is published to Docker Hub and to Amazon ECR Public for `linux/amd64` and `linux/arm64`. It is a distroless
image that holds the binary and its configuration files.

| Registry         | Image                                                  |
| ---------------- | ------------------------------------------------------ |
| Docker Hub       | `sunerpy/bedrock-gateway-rust`                         |
| Amazon ECR Public | `public.ecr.aws/k5i7b1e0/bedrock-gateway-rust`        |

Each release is tagged with its version, such as `0.17.0`, and with `latest`. Pin a version in production:

```sh
docker pull sunerpy/bedrock-gateway-rust:0.17.0
```

ECR Public needs no login, so an ECS task or Lambda function can pull from it without Docker Hub's rate limits. The
image listens on port 8080 and runs as a non-root user.

## Binary

Every [GitHub release](https://github.com/sunerpy/bedrock-gateway-rust/releases) has an archive per platform. Each
holds one file, the `bedrock-gateway` binary (`bedrock-gateway.exe` on Windows); its configuration is built in.

| Platform            | Archive                                              |
| ------------------- | ---------------------------------------------------- |
| Linux x64           | `bedrock-gateway-<version>-x86_64-unknown-linux-musl.tar.gz`  |
| Linux ARM64         | `bedrock-gateway-<version>-aarch64-unknown-linux-musl.tar.gz` |
| macOS Intel         | `bedrock-gateway-<version>-x86_64-apple-darwin.tar.gz`        |
| macOS Apple Silicon | `bedrock-gateway-<version>-aarch64-apple-darwin.tar.gz`       |
| Windows x64         | `bedrock-gateway-<version>-x86_64-pc-windows-msvc.zip`        |

With the GitHub CLI, download the latest one for your platform and unpack it:

```sh
gh release download --repo sunerpy/bedrock-gateway-rust --pattern '*-x86_64-unknown-linux-musl.tar.gz'
tar -xzf bedrock-gateway-*-x86_64-unknown-linux-musl.tar.gz
./bedrock-gateway
```

The Linux binaries are statically linked and run on any distribution.

## Cargo

```sh
cargo install bedrock-gateway-rust
```

This builds the `bedrock-gateway` binary into `~/.cargo/bin`. It needs a Rust toolchain; the version the project
builds with is pinned in [`rust-toolchain.toml`](../../../rust-toolchain.toml).

## Build from source

```sh
git clone https://github.com/sunerpy/bedrock-gateway-rust
cd bedrock-gateway-rust
cargo build --release
./target/release/bedrock-gateway
```

To export traces and metrics with OpenTelemetry, build with `--features otel`; see
[Observability](../operate/observability.md). The release binaries and images are built without it.

## Configuration files

The binary carries a built-in copy of its configuration: the model registry (`models.toml`), the region routes
(`regions.toml`) and the embedding models (`embeddings.toml`). At start it looks for these files in `CONFIG_DIR`,
which is `config` in the working directory unless you set it. A file that exists and parses replaces the built-in
copy; a missing or broken file leaves the built-in copy in use. The image sets its working directory to
`/etc/bedrock-gateway`, so it reads `/etc/bedrock-gateway/config/`. [Models](models.md#add-or-change-a-model)
explains the model registry.

## Update

Pull the new image tag, or download the new archive and replace the binary, then restart the gateway. It keeps no
state between runs, so nothing needs to be migrated.

## Remove

Stop the gateway and delete the binary or the image. It writes no files of its own.
