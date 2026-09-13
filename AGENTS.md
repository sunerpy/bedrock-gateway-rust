# AGENTS.md

OpenAI-compatible HTTP gateway for AWS Bedrock. Rust, `axum + tokio + aws-sdk-bedrockruntime`. Crate `bedrock-gateway-rust`, binary `bedrock-gateway`, image `sunerpy/bedrock-gateway-rust`.

`README.md` already documents endpoints, env vars, quick start, and deployment. This file holds only what an agent would otherwise get wrong.

## Verification gate

Run all three, in order, before any commit. `--all-features` is not optional: it is the only thing that compiles the `otel` feature, and CI runs it.

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

- CI (`.github/workflows/ci.yml`) runs exactly those three plus `cargo audit`. `ci-success` requires only `test` + `audit`; `coverage` is informational and never blocks.
- `make hooks` once per clone points `core.hooksPath` at `.githooks/`, enabling a pre-push hook that runs the same three checks. It gates `git push` only — `git commit` stays free for WIP. Passing pre-push implies passing CI. `core.hooksPath` is local config and is never inherited by a fresh clone.
- Toolchain is pinned to 1.91.1 (`rust-toolchain.toml`). Do not add a `rustup override`.
- `make fmt` also runs `oxfmt` over TOML when installed, so it can produce unrelated `config/*.toml` diffs. Use `cargo fmt --all` when you only mean Rust.
- `cargo audit` reads `.cargo/audit.toml`, which carries four ignores with a written non-exploitability argument each (legacy `rustls 0.21` / `h2 0.3` pulled transitively by the AWS SDK). Do not delete them to make a red build green. After any `aws-config` / `aws-sdk-*` bump, re-run `cargo tree -i rustls-webpki` and `cargo tree -i h2@0.3.27` and drop an ignore only once its root is gone.
- Release profile sets `panic = "abort"`, but that applies to release only; the test profile still unwinds, so `catch_unwind` in `tests/golden` works.

## Where things live

```
src/
  domain/mod.rs      provider-agnostic traits, no AWS types
  openai/            wire schemas: schema.rs (chat), responses_schema.rs, completions_schema.rs
  config/            settings.rs, capabilities.rs, regions.rs, embeddings.rs
  bedrock/           Converse path + mantle_* (raw upstream) + responses_* (second surface)
  server/            auth.rs, state.rs (build_app_state), composite*.rs, routers/mod.rs
config/              models.toml, regions.toml, embeddings.toml, app.toml — all model knowledge
```

Config resolution: an external config file that exists and parses wins over the compile-time embedded default. A missing or unparseable external file falls back to the embedded default — it never degrades to empty config. The Docker image embeds `config/`, so a config-only change needs an image rebuild to reach a container.

Two composite dispatchers pick a backend per request, so a model can be mantle on one surface and Converse on the other:

- `server/composite.rs` — `caps.responses_backend(model)` → `MantleResponsesProvider` or `BedrockResponsesProvider`.
- `server/composite_chat.rs` — `caps.chat_backend(model)` → Converse, native mantle chat, or `ResponsesChatProvider`.

`chat_backend` and `responses_backend` are independent fields and must never cross-contaminate.

## Hard rules

**Zero hardcoding.** All model knowledge lives in `config/*.toml`; `.rs` holds only the matching algorithm. Protocol constants (`data: `, `[DONE]`, `chat.completion`, `chatcmpl-`, `finish_reason` values) are fine. Model IDs, name-derived capability flags, context-window magic numbers, and any `if model.contains("...")` are not. If you reach for one, add a capability flag to `config/models.toml` and read it via `ModelCapabilities::has(Capability::...)`.

The single documented exception is `skip_tool_choice_for` in `src/bedrock/provider.rs` (inspects `meta.llama3-1-`). It is annotated in-code and flagged for replacement. Do not add another without the same in-code justification.

**No invented top-level fields.** Bedrock-only features are surfaced through OpenAI's `extra_body` mechanism, parsed via `#[serde(flatten)] extra: HashMap<String, Value>` on `ChatRequest`. Never add a Bedrock concept as a new top-level field on `ChatRequest` / `ChatResponse`.

**One token formula.** `compute_token_usage(input, output, cacheRead, cacheWrite)` in `src/bedrock/tokens.rs` is the only place it lives: `prompt_tokens = input + cacheRead + cacheWrite`, `total_tokens = prompt_tokens + output`, `cached_tokens = cacheRead` only. `cacheWriteInputTokens` folds into `prompt_tokens` and is never a separate wire field. All four render paths (`response.rs`, `stream.rs`, `responses_response.rs`, `responses_stream.rs`) call it — do not re-derive.

**Three reasoning render paths, never unified.**

| Surface | Render |
| --- | --- |
| `/chat/completions` (Converse) | inline `<think>...</think>` inside `content`; `reasoning_content` is `#[serde(skip_serializing)]` and never hits the wire |
| `/responses` | structured `reasoning` item in the `output` array, not wrapped in `<think>` |
| `/chat/completions` (mantle, gpt-oss) | raw passthrough: upstream `delta.reasoning` + per-chunk `obfuscation` survive verbatim, never rewritten to `<think>`, never dropped |

Touching one means verifying the other two are unchanged.

**Raw-lane contract.** Raw lanes return `Result<Option<...>, AppError>`: `Ok(Some(stream))` selects passthrough, `Ok(None)` means unsupported, `Err(e)` preserves the first pre-stream failure and must not trigger a second typed request. The one asymmetry: the raw **chat** lane appends `data: [DONE]\n\n` at the tail (mantle omits it, OpenAI chat clients expect it); the raw **responses** lane does not, and terminates on `response.completed`.

**Responses surface is stateless.** `store` and `previous_response_id` are accepted and ignored. Unknown tool types deserialize into `ResponsesTool`'s `#[serde(other)] Unknown` and hosted server tools with no Bedrock equivalent are silently dropped, never 400 — codex bundles them unconditionally, and a 400 would kill the whole session including the caller's real function tools.

**Cache placement.** Auto-injection is default-ON (`enable_prompt_caching`). Order is tools → system → messages, sharing a budget of `max_cache_checkpoints` total points. A model supports caching iff its `config/models.toml` entry has `cache_min_tokens` — config gate, no name inspection. Cache hits need byte-stable prefixes: changing any segment before a `cachePoint` invalidates every later point in that request.

**Logging.** `info` = access log plus business metadata (model, streaming flag, `finish_reason`, token counts). `debug` adds resolved model and target region. At no level are bodies, message content, prompt/completion text, raw token values, or the bearer token ever logged. Use structured `tracing` fields; never `Debug`-print a whole request or response struct. Handler failures: `error!` for 5xx, `warn!` for 4xx, split by `AppError::is_server_error()`.

## Adding a model

Config only, no `src/` edits. Append to `config/models.toml`:

```toml
[[model]]
match = "provider.model-id"                 # prefix or exact
capabilities = ["TemperatureToppConflict"]  # zero or more Capability variants
[model.params]
max_tokens = 8192
context_window = 200000
# cache_min_tokens = 1024        # presence enables prompt caching
# reasoning_path = "BudgetTokens"
# responses_backend = "mantle"   # /responses via mantle upstream
# chat_backend = "mantle"        # /chat/completions via mantle upstream (raw)
# chat_backend = "responses"     # /chat/completions via the Responses adapter
# available_regions = ["us-east-2"]  # absent = everywhere; non-empty = per-request 400 elsewhere
```

`[[alias]]` entries must sit **above the first `[[model]]`** — a TOML positional constraint, not a style preference. Cross-region routing goes in `config/regions.toml`; embedding models in `config/embeddings.toml` with a `family`.

Mantle upstream URLs come from `MANTLE_BASE_URL_TEMPLATE` (default `.../openai/v1`, responses) and `MANTLE_CHAT_BASE_URL_TEMPLATE` (default `.../v1`, **no** `/openai` prefix — the routes genuinely differ). `{region}` is substituted with `AWS_REGION` at call time.

Startup degrades rather than fails: a mantle-backed model with no `bedrock_api_key` (`AWS_BEARER_TOKEN_BEDROCK` / `BEDROCK_API_KEY`), or `DISABLE_MANTLE=true`, disables only those models with a WARN. Region mismatches WARN at boot and 400 per request. The gateway still starts and every other model keeps working.

## Tests

Unit tests are sidecar files, not inline modules: `foo.rs` ends with `#[cfg(test)] #[path = "foo_tests.rs"] mod tests;` and the tests live in `foo_tests.rs`. There are 33 such files. Follow the pattern; do not add an inline `mod tests`.

Golden record/replay lives in `tests/golden/`, fixtures under `fixtures/<group>/<case>/` (`translation`, `response`, `streaming`, `responses_*`, `embeddings`). Helpers in `harness.rs`: `assert_semantic_eq`, `assert_semantic_eq_with`, `assert_stream_eq`. Assertions are semantic, not byte-equal, so field ordering can drift. Add a fixture alongside any new translation path. Runs offline in CI with no AWS credentials.

Live integration tests are `#[ignore]`d and env-gated (`src/bedrock/cache_tests.rs`, `src/bedrock/models_tests.rs`):

```bash
BEDROCK_INTEGRATION=1 AWS_PROFILE=us cargo test -- --ignored   # needs real Bedrock access in us-east-2
```

Coverage is tracked but never blocking: `make coverage`, `make coverage-html`, `make coverage-lcov`. Target 95%; see `docs/coverage.md`.

Transient Bedrock 503/500 in CI: retry. Before re-running a task that may already have committed, check `git status` and `git log` first.

## Conventions

- **Commits:** Conventional Commits with a Chinese, imperative subject — `feat: 添加 Nova embedding 支持`, `fix: 修复流式响应 finish_reason 映射`.
- **Do not edit `src/api/`** — Python reference artifact from the original gateway, not live code.
- **Two unrelated auth directions.** Client → gateway is `API_KEY` / `API_KEY_SECRET_ARN` / `API_KEY_PARAM_NAME`, resolved SSM → Secrets Manager → env, enforced in `server/auth.rs` (`AppSettings::api_key`). Gateway → Bedrock is `AWS_BEARER_TOKEN_BEDROCK` / `BEDROCK_API_KEY` with SigV4 fallback, injected in `bedrock::client::build_aws_config` (`AppSettings::bedrock_api_key`). Never conflate them.
- **Env vars:** both `APP_`-prefixed and bare Python-parity names are accepted; bare names win. Allow-list is `apply_bare_env_overrides` in `src/config/settings.rs`.
- **Lambda:** never set `AWS_REGION` in the Lambda environment — reserved variable, cfn-lint `E3663`. The runtime injects it.
- **Docs layout:** root holds only `README.md` and `AGENTS.md`. Everything else goes under `docs/readme/` (README_CN, CONTRIBUTING, CODE_OF_CONDUCT) or `docs/deploy/`. Do not add `.md` files to the root.
- **axum stays.** actix-web was evaluated and rejected (SSE path is axum-native, `FromRequestParts`/`IntoResponse` encode the 401-vs-405 error contract, graceful shutdown is built in; the service is IO-bound). Closed decision — reopen only with a benchmark showing axum as the bottleneck.

## Deeper reference

- `docs/caching-and-reasoning.md` — per-model `cache_min_tokens`, reasoning budget ratios, cross-region inference profiles, config precedence.
- `docs/openai-protocol-compatibility.md` — streaming contract per surface, tool handling audit, required regression matrix.
- `docs/gpt-responses-chat-adapter.md` — Responses→Chat adapter, stateless reasoning + tool replay, stream ordering.
- `docs/chat-reasoning-tool-replay-incident.md` — `rsc_v1` / `CHAT_REASONING_CAPSULE_*` keyring incident history.
- `docs/deploy/` — docker, ecs, lambda, migration guides. `docs/coverage.md` — coverage policy.
