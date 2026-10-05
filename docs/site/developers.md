# Contributing

This page is for working on bedrock-gateway itself: the layout of the repository, the checks a change has to pass,
and how releases and this site are made.

## The repository

bedrock-gateway is written in Rust with axum, tokio and the AWS SDK for Rust. `rust-toolchain.toml` pins the
toolchain, and `rustup` installs it on the first build.

| Path             | Holds                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `src/server/`    | The HTTP server: routes, authentication, and the dispatchers that pick a backend for each model         |
| `src/openai/`    | The OpenAI wire types of Chat Completions, Responses and Completions                                    |
| `src/bedrock/`   | Translation to and from Bedrock Converse, streaming, caching, reasoning, embeddings and the OpenAI-compatible backend |
| `src/config/`    | Settings and the loaders of the model registry, the region routes and the embedding models              |
| `src/domain/`    | The traits the server calls, free of AWS types                                                           |
| `src/telemetry/` | Logging and the optional OpenTelemetry export                                                            |
| `config/`        | `models.toml`, `regions.toml`, `embeddings.toml` and `app.toml`: everything the gateway knows about models |
| `tests/golden/`  | Recorded requests and answers that the translation is checked against                                    |
| `deployment/`, `helm/` | The CloudFormation templates and the Helm chart                                                   |
| `docs/`          | The references; `docs/site/` is this site                                                                |

[`AGENTS.md`](../../AGENTS.md) holds the rules every change follows, for people and coding agents alike. Two of them
shape most changes: model knowledge lives in `config/models.toml`, never in the code, and features that exist only on
Bedrock reach it through `extra_body`, never through new top-level request fields.

## Build and test

```sh
git clone https://github.com/sunerpy/bedrock-gateway-rust
cd bedrock-gateway-rust
make hooks                                                # run the three checks below before every git push
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

`--all-features` compiles the `otel` feature as well. CI runs the same three commands and `cargo audit`, so a branch
that passes the hook passes CI.

The tests run offline, with no AWS credentials. A change to a translation comes with a fixture in `tests/golden/`;
the golden tests compare answers by meaning, not byte for byte. Tests that call Bedrock are skipped unless asked for,
and need access to Bedrock in `us-east-2`:

```sh
BEDROCK_INTEGRATION=1 AWS_PROFILE=<your profile> cargo test -- --ignored
```

## Adding a model

A new model is a change to `config/models.toml` only; [Models](guide/models.md#add-or-change-a-model) describes the
entries. The Helm chart carries a copy in `helm/bedrock-gateway/files/models.toml`, and a test fails when the two
differ in any entry, so change both.

## Pull requests and releases

- Commit messages and pull request titles follow Conventional Commits, for example
  `fix: preserve image support in Responses translation`. release-please reads them to choose the next version.
- Pull requests are squash-merged into `main`; nothing is pushed to `main` directly.
- Releases are cut by release-please: its pull request bumps the version and the changelog, and merging it builds the
  binaries for five platforms, the images on Docker Hub and Amazon ECR Public, and the crate on crates.io.

## This site

The pages of this site live in `docs/site/`, English at the root and Chinese under `zh/`, next to the code they
describe, so a change that alters what users see updates its page in the same pull request.
[`docs/site/README.md`](README.md) lists the rules for these pages. The site's theme and build live in
[sunerpy/firlab](https://github.com/sunerpy/firlab), which copies the pages in after every change to `main`.
