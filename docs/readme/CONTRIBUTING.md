# Contributing Guidelines

Thank you for your interest in contributing to **bedrock-gateway-rust**. Whether it's a
bug report, new feature, correction, or additional documentation, we greatly value
feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure
we have all the necessary information to effectively respond to your bug report or
contribution.

## Reporting Bugs / Feature Requests

We use the GitHub issue tracker for bugs and feature suggestions.

When filing an issue, please check existing open (and recently closed) issues to avoid
duplicates. Include as much context as you can:

- A reproducible test case or series of steps
- The binary version or git commit hash
- The Bedrock model ID and region you are targeting
- Any modifications you've made relevant to the bug
- Anything unusual about your environment or deployment

## Contributing via Pull Requests

Before sending a pull request, please ensure that:

1. You are working against the latest source on the `main` branch.
2. You've checked existing open and recently merged pull requests to avoid overlap.
3. You've opened an issue to discuss any significant work beforehand.

To send a pull request:

1. Fork the repository and create a feature branch from `main`.
2. Focus your changes on the specific problem you are solving. Mixed-concern PRs are
   hard to review.
3. Run the pre-commit gate before pushing:
   ```bash
   cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
   ```
4. If you add a new translation path, add a golden fixture alongside the implementation.
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/) format with
   a Chinese subject line (e.g. `feat: 添加 Nova embedding 支持`).
6. Open the pull request and stay engaged with any CI feedback or reviewer comments.

GitHub has helpful guides on
[forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Finding Something to Work On

Browse existing issues — labels like `help wanted` and `good first issue` are good
starting points. The [AGENTS.md](../../AGENTS.md) file describes the architecture and
contributor conventions in detail.

## Zero-Hardcoding Contract

All model knowledge lives in `config/*.toml`. Adding a new model or adjusting a
capability flag never requires a code change. See the
[Zero-Hardcoding Contract](../../AGENTS.md#3-零硬编码契约critical) section in AGENTS.md.

## Code of Conduct

This project has adopted the
[Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). For more
information see the
[Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security Issue Notifications

If you discover a potential security issue, please notify AWS/Amazon Security via the
[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/).
Do **not** create a public GitHub issue for security vulnerabilities.

## Licensing

See the [LICENSE](../../LICENSE) file for our project's licensing. We will ask you to
confirm the licensing of your contribution.
