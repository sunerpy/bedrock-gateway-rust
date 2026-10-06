# Contributing

Thanks for helping improve bedrock-gateway-rust. This page covers what a change needs before it can be merged.

## Reporting bugs

Open a [GitHub issue](https://github.com/sunerpy/bedrock-gateway-rust/issues) after checking that it has not been reported already. Include:

- the gateway version or commit, and how you run it (binary, Docker, ECS, Lambda, Helm);
- the endpoint, the Bedrock model ID, and the AWS region;
- a minimal request that reproduces the problem, and what you expected instead.

Remove API keys, AWS credentials, and private prompt content before posting.

## Security issues

Never report a vulnerability in a public issue. Follow [SECURITY.md](SECURITY.md) instead.

## Development setup

The Rust toolchain is pinned in `rust-toolchain.toml`, and rustup selects it automatically. After cloning, enable the pre-push hook once:

```bash
make hooks
```

Then run the same checks CI runs before you push:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

The suite runs offline and needs no AWS credentials. [AGENTS.md](../AGENTS.md) describes the architecture and the rules a change must follow.

## Making changes

- Model knowledge (model IDs, capability flags, token limits, cache thresholds) goes in `config/models.toml`, never in Rust code. Copy the file to `helm/bedrock-gateway/files/models.toml` too; a contract test checks that the two match.
- A change to a translation or rendering path needs a golden fixture under `tests/golden/`. See [tests/golden/README.md](../tests/golden/README.md).
- Unit tests live in sidecar `*_tests.rs` files next to the module they test.
- Update the docs when behavior or configuration changes.

## Commits and pull requests

Use [Conventional Commits](https://www.conventionalcommits.org/) with an English, imperative subject and no scope:

```text
fix: preserve image support in Responses translation
feat: add Nova embedding support
```

Pull requests are squash-merged, and the PR title becomes the commit on `main`, so the title must follow the same format. The version bump and the release notes are derived from these commit messages. Keep each pull request focused on one change.

## License

Contributions are licensed under MIT-0, the project's license. See [LICENSE](../LICENSE).
