# AGENTS.md — Contributor & Agent Guide

> Bilingual guide (English + 中文). Both sections carry equivalent information. When you update one, keep the other in sync.

---

## English

### 1. Project Overview

`bedrock-gateway-rust` is an OpenAI-compatible HTTP gateway for AWS Bedrock, written in Rust. It replaces an earlier Python/FastAPI implementation while preserving wire-exact compatibility with the OpenAI REST API. The runtime stack is **axum + tokio + aws-sdk-bedrockruntime**. Docker image: `sunerpy/bedrock-gateway-rust`.

Supported endpoints (prefix configurable via `API_ROUTE_PREFIX`, default `/api/v1`):

| Endpoint                        | Notes                                                       |
| ------------------------------- | ----------------------------------------------------------- |
| `POST /api/v1/chat/completions` | Streaming (SSE) + non-streaming                             |
| `POST /api/v1/responses`        | OpenAI Responses API surface (stateless; required by codex) |
| `POST /api/v1/embeddings`       | Cohere / Titan / Nova                                       |
| `GET  /api/v1/models`           | Catalog refresh from Bedrock control plane                  |
| `GET  /api/v1/models/{id}`      | Single model lookup                                         |
| `GET  /api/v1/health`           | Liveness probe                                              |

---

### 2. Architecture

The codebase is layered. Dependencies flow strictly downward.

```
src/
├── main.rs              # tokio::main, AppSettings::load → telemetry::init → server::serve
├── lib.rs               # crate root, re-exports
├── error.rs             # AppError (thiserror), OpenAI error envelope, HTTP status mapping
├── telemetry.rs         # tracing subscriber, ReloadHandle for dynamic log-level
│
├── openai/
│   ├── schema.rs        # Wire types: ChatRequest, ChatResponse, ChatStreamResponse,
│   │                    #   ChatResponseMessage, Usage, Embeddings*, Model(s), OpenAiError
│   └── responses_schema.rs  # Responses surface types: ResponsesRequest, ResponsesResponse,
│                            #   ResponseInputItem, ResponseOutputItem, ResponseStreamEvent,
│                            #   ResponsesUsage
│
├── domain/
│   └── mod.rs           # Provider-agnostic traits:
│                        #   ChatProvider, EmbeddingProvider  (async_trait)
│                        #   ResponsesProvider  (async_trait)
│                        #   ModelCapabilities, RegionRouter, EmbeddingBodyCodec  (sync)
│                        #   NormalizedChatRequest { request, resolved_model }
│                        #   NormalizedResponsesRequest { request, resolved_model }
│                        #   ChatStream = BoxStream<'static, Result<ChatStreamResponse, AppError>>
│                        #   ResponsesStream = BoxStream<'static, Result<ResponseStreamEvent, AppError>>
│
├── config/
│   ├── settings.rs      # AppSettings::load, layered env (APP_ prefix + bare override list)
│   ├── capabilities.rs  # ModelCapabilityConfig::load/from_toml_str, Capability enum,
│   │                    #   ReasoningPath enum, BudgetRatios
│   ├── regions.rs       # RegionRoutingConfig::load, RouteOverride { region, rewritten_model_id }
│   └── embeddings.rs    # EmbeddingRegistry::load, EmbeddingFamily { Cohere, Titan, Nova }
│
├── bedrock/
│   ├── capabilities.rs  # ConfigModelCapabilities implements domain::ModelCapabilities
│   ├── client.rs        # BedrockClients { runtime, control }, build_aws_config,
│   │                    #   region_config_override for per-request region override
│   ├── tokens.rs        # estimate_reasoning_tokens(&str) -> u32
│   │                    #   compute_token_usage(input, output, cacheRead, cacheWrite) -> Usage
│   ├── translate.rs     # to_converse_args: ChatRequest → ConverseArgs + ConverseExtras seam
│   ├── reasoning.rs     # build_reasoning_config → ReasoningOutcome; 4 paths via ReasoningPath
│   ├── tools.rs         # OpenAI tool_use ↔ Bedrock toolConfig translation
│   ├── cache.rs         # Prompt-caching cache_point injection (Claude + Nova);
│   │                    #   decorate_tools/system/messages with shared budget ≤ max_cache_checkpoints
│   ├── response.rs      # from_converse_output: ConverseOutput → ChatResponse,
│   │                    #   <think> inline rendering, usage mapping
│   ├── stream.rs        # StreamState machine + converse_stream_to_openai async_stream wrapper
│   ├── embeddings.rs    # CohereCodec / TitanCodec / NovaCodec implement EmbeddingBodyCodec;
│   │                    #   BedrockEmbeddingProvider implements EmbeddingProvider
│   ├── models.rs        # ModelCatalog { models, profile_metadata }, refresh via control plane
│   ├── provider.rs      # BedrockChatProvider implements ChatProvider — composes
│   │                    #   translate + reasoning + tools + cache → converse/converse_stream
│   │                    #   → response/stream mapping
│   ├── responses_translate.rs  # to_responses_converse_input: ResponsesRequest → Bedrock messages/system;
│   │                           #   reasoning_outcome reuses build_reasoning_config
│   ├── responses_response.rs   # from_converse_output_to_responses: ConverseOutput → ResponsesResponse;
│   │                           #   reasoning → structured reasoning output item (NOT <think>)
│   ├── responses_stream.rs     # ResponsesStreamState + converse_stream_to_openai_responses wrapper;
│   │                           #   full lifecycle events, monotonic sequence_number, NO [DONE] sentinel
│   └── responses_provider.rs   # BedrockResponsesProvider implements ResponsesProvider — composes
│                                #   responses_translate + reasoning + cache → converse/converse_stream
│                                #   → responses_response/responses_stream mapping
│
└── server/
    ├── auth.rs          # Bearer-token middleware
    ├── state.rs         # AppState, build_app_state assembles all components
    ├── mod.rs           # serve(AppSettings) entrypoint, apply_layers (TraceLayer + CorsLayer)
    └── routers/
        └── mod.rs       # build_router: axum Router wiring all endpoints
```

Config files (NOT code):

```
config/
├── models.toml      # All model capability declarations
├── regions.toml     # Cross-region routing rules
├── embeddings.toml  # Embedding model registry
└── app.toml         # Application defaults (overridden by env)
```

#### ADR: HTTP framework — axum (evaluated, retained)

The HTTP framework is **axum** (tokio + tower + tower-http). Replacing it with actix-web was evaluated and **rejected**. Reasons to retain axum:

1. **SSE streaming backbone** — the streaming path is axum-native (`axum::response::Sse`); migrating would require rewriting the entire `server/` layer with no correctness gain.
2. **Custom OpenAI error envelope + auth semantics** — axum's `FromRequestParts` + `IntoResponse` cleanly encode the 401-vs-405 distinction required by the OpenAI error contract; actix middleware achieves the same only with more boilerplate.
3. **Graceful shutdown** — axum's `serve(...).with_graceful_shutdown(...)` integrates with tokio's signal handling out of the box.

This service is IO-bound (Bedrock proxy); actix-web offers no measurable throughput advantage. Lambda Web Adapter is framework-neutral, so the Lambda deployment path is unaffected. This decision is closed — do not re-open it without a concrete benchmark showing axum as the bottleneck.

#### Responses surface

`POST /api/v1/responses` is a **second OpenAI surface** — the OpenAI Responses API — implemented entirely separately from chat completions. It has its own provider trait (`ResponsesProvider` in `src/domain/mod.rs`), its own schema (`src/openai/responses_schema.rs`), and a dedicated four-module stack under `src/bedrock/`:

| Module                   | Role                                                                                                                                         |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `responses_translate.rs` | Parse `ResponsesRequest` input items → Bedrock messages/system; reuse `build_reasoning_config` for thinking budget                           |
| `responses_response.rs`  | Map `ConverseOutput` → `ResponsesResponse`; reasoning → structured `reasoning` output item                                                   |
| `responses_stream.rs`    | `ResponsesStreamState` + `converse_stream_to_openai_responses`; full lifecycle events, monotonic `sequence_number`, **no `[DONE]` sentinel** |
| `responses_provider.rs`  | `BedrockResponsesProvider` implements `ResponsesProvider`; composes the above three + cache injection                                        |

The surface is **stateless**: `store` and `previous_response_id` are accepted and silently ignored (codex sends `store: false`). It reuses the same Converse call layer and the shared `compute_token_usage` helper from `src/bedrock/tokens.rs`. codex requires this surface (`wire_api = "responses"` only).

**Limits / rejection matrix:**

- Built-in server tools (`web_search`, `file_search`, `code_interpreter`, `mcp`, `computer`, `image_generation`) → 400.
- `encrypted_content` is not round-tripped.
- No `function_call_arguments.delta` stream events (codex-leniency driven).
- `input_file` parts → 400 (no Bedrock document-block mapping).

#### Cache placement contract

Cache-point auto-injection is **default-ON** (master switch `enable_prompt_caching`, default `true` in `config/app.toml` and `settings.rs`). The placement order is **tools → system → messages**, with a shared budget of at most `max_cache_checkpoints` total cache points across all three positions. `max_cache_checkpoints` is config-driven via `ModelCapabilities::max_cache_checkpoints` (default constant 4).

A model "supports caching" (`supports_caching` in `cache.rs`) if and only if its entry in `config/models.toml` includes a `cache_min_tokens` param. This is the config gate — no model name inspection in code.

**Byte-stable-prefix discipline:** cache hits depend on deterministic serialization. Changing any segment before a `cachePoint` invalidates all later cache points in that request. Keep early segments stable across turns.

**Token accounting** is done by the single `compute_token_usage(input, output, cacheRead, cacheWrite)` helper in `src/bedrock/tokens.rs`:

- `prompt_tokens` = `input + cacheRead + cacheWrite`
- `total_tokens` = `prompt_tokens + output`
- `cached_tokens` = `cacheRead` only

`cacheWriteInputTokens` from Bedrock folds into `prompt_tokens` but is **never a separate wire field** (no standard OpenAI field for write-side cache accounting). Both `response.rs` and `stream.rs` (chat surface) and `responses_response.rs` / `responses_stream.rs` (Responses surface) all call this same helper — do not duplicate the formula.

For per-model `cache_min_tokens` thresholds, reasoning budget behavior, and cross-region inference profile rules, see [`docs/caching-and-reasoning.md`](docs/caching-and-reasoning.md).

#### Two reasoning render paths (architectural rule)

Reasoning output takes **different forms on the two surfaces** and must never be unified:

| Surface                    | Reasoning render                                                                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat (`/chat/completions`) | Inline `<think>...</think>` inside the `content` string. `reasoning_content` in `ChatResponseMessage` carries `#[serde(skip_serializing)]` and never appears on the wire. |
| Responses (`/responses`)   | A structured `reasoning` output item in the `output` array. Not wrapped in `<think>`.                                                                                     |

If you touch either rendering path, verify the other is unchanged. Do not merge them.

---

### 3. Zero-Hardcoding Contract (CRITICAL)

**ALL model knowledge lives in `config/*.toml`. Rust code holds only the matching algorithm.**

| Allowed in `.rs`                                        | Forbidden in `.rs`                                 |
| ------------------------------------------------------- | -------------------------------------------------- |
| SSE protocol constants (`data: `, `[DONE]`)             | Model IDs (`anthropic.claude-*`, etc.)             |
| OpenAI object type strings (`chat.completion`, etc.)    | Capability flags tied to a model name              |
| `chatcmpl-` id prefix                                   | Magic numbers for context windows or token budgets |
| `finish_reason` values (`stop`, `length`, `tool_calls`) | Any `if model.contains("...")` logic               |

**One documented exception:** `src/bedrock/provider.rs` contains a `skip_tool_choice_for` check that inspects `meta.llama3-1-`. This is explicitly documented in-code and flagged for replacement with a capability flag in `models.toml`. Do not add similar exceptions without documenting them the same way.

If you find yourself writing `if model_id.contains("claude")` in Rust, stop. Add a capability flag to `config/models.toml` instead, then read it through `ModelCapabilities::has(Capability::...)`.

---

### 4. Option-B Compliance

The gateway presents a byte-exact OpenAI wire shape. Bedrock-specific features are surfaced **only** through the OpenAI-sanctioned `extra_body` mechanism — never as invented top-level request fields.

**Reasoning / extended thinking:** rendered inline as `<think>...</think>` inside the `content` string. The `reasoning_content` field in `ChatResponseMessage` carries `#[serde(skip_serializing)]` unconditionally — it never appears on the wire, even if populated internally.

**Prompt caching:** requested via `extra_body: { "prompt_caching": { "system": true, "messages": true } }`. The `cached_tokens` field in `PromptTokensDetails` reflects **cache-read** tokens only. `cacheWriteInputTokens` from Bedrock is acknowledged but intentionally not mapped (no standard OpenAI field exists for write-side cache accounting).

**Rule:** if you add any Bedrock-only feature, route it through `extra_body` parsing in `openai::schema::ChatRequest` (via `#[serde(flatten)] extra: HashMap<String, Value>`). Never add a new top-level field to `ChatRequest` or `ChatResponse` for Bedrock concepts.

#### Logging / observability

`info` level emits a per-request access log (method/path/status/latency, via the axum `TraceLayer` configured at INFO in `server/mod.rs`) plus key business metadata from the handlers (`model`, streaming flag, `finish_reason`, token **counts**). `debug` level additionally logs upstream Bedrock call details (resolved model, target region) from `bedrock/provider.rs`. At **no** level (not even `debug`) are request/response bodies, message content, prompt/completion text, raw token values, or the `API_KEY`/bearer token ever logged — only metadata. When adding logs, use structured `tracing` fields; never `Debug`-print a whole request/response struct.

---

### 5. How to Add a New Model

No code change needed. Edit `config/models.toml`:

```toml
[[model]]
match = "your-provider.your-model-id"   # prefix or exact string
capabilities = ["TemperatureToppConflict"]  # zero or more Capability variants
[model.params]
max_tokens = 8192
context_window = 200000
# reasoning_path = "BudgetTokens"  # if model supports extended thinking
```

For cross-region routing, add an entry to `config/regions.toml`. For a new embedding model, add to `config/embeddings.toml` with its `family` field.

No recompile required for config-only changes when the binary reads config at startup from disk. (The Docker image embeds the config files; rebuild the image to pick up config changes in containerized deployments.)

---

### 6. Trait Extension Points

To add a non-Bedrock backend, implement the traits in `src/domain/mod.rs`:

| Trait                | Sync/Async | Responsibility                                                                    |
| -------------------- | ---------- | --------------------------------------------------------------------------------- |
| `ChatProvider`       | async      | Translate `NormalizedChatRequest` → `ChatResponse` or `ChatStream`                |
| `EmbeddingProvider`  | async      | Translate embedding request → `EmbeddingsResponse`                                |
| `ResponsesProvider`  | async      | Translate `NormalizedResponsesRequest` → `ResponsesResponse` or `ResponsesStream` |
| `ModelCapabilities`  | sync       | Query capabilities and routing metadata for a model ID                            |
| `RegionRouter`       | sync       | Return `RouteOverride` for a given model ID                                       |
| `EmbeddingBodyCodec` | sync       | Encode/decode embedding request/response bytes for a specific model family        |

Currently only the Bedrock backend is implemented (`src/bedrock/`). The traits carry no AWS types — they're provider-agnostic by design.

Wire your new provider into `src/server/state.rs` inside `build_app_state`, following the same Arc-wrapping pattern as `BedrockChatProvider`.

---

### 7. Build / Test / Deploy Commands

```bash
# Development
cargo build                                              # debug build
cargo build --release                                   # release binary → target/release/bedrock-gateway
cargo test                                              # all tests (unit + golden + doctests)
cargo clippy --all-targets --all-features -- -D warnings  # must be warning-free
cargo fmt                                               # format check / apply

# Makefile shortcuts
make help                                               # list all targets

# Docker (local)
docker build -t bedrock-gateway-rust .                  # distroless image from root Dockerfile

# Run locally (no real AWS creds needed for health check)
API_KEY=testkey cargo run
curl http://localhost:8080/api/v1/health
```

**Pre-commit gate (mandatory before every commit):**

```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

**Deployment targets:**

| Target                | Files                                                                       |
| --------------------- | --------------------------------------------------------------------------- |
| ECS/Fargate (ALB)     | `deployment/BedrockGatewayFargate.template` + root `Dockerfile`             |
| Lambda (Function URL) | `deployment/BedrockGatewayLambda.template` + `deployment/lambda/Dockerfile` |
| Lambda docs           | `docs/deploy/lambda.md`                                                     |

Both CloudFormation templates accept bare env-var names (`API_KEY`, `AWS_REGION`, `DEFAULT_MODEL`, etc.). See the full allow-list in `src/config/settings.rs` → `apply_bare_env_overrides`.

Lambda note: do NOT set `AWS_REGION` in the Lambda environment — it is a Lambda reserved variable and cfn-lint will flag it as `E3663`. The Lambda runtime injects it automatically.

---

### 8. Parity / Golden-Replay Workflow

Tests are two-tier:

**Tier 1 — Offline golden record/replay** (`tests/golden/`):

- Fixtures are pinned against Python HEAD `9a3e752`
- Assertion helpers: `assert_semantic_eq` (unordered field comparison) and `assert_stream_eq`
- Run automatically in CI with no AWS credentials needed
- `cargo test` runs these by default

**Tier 2 — Live integration tests** (gated):

```bash
BEDROCK_INTEGRATION=1 AWS_PROFILE=us cargo test -- --ignored
```

- Requires real AWS credentials and Bedrock access in `us-east-2`
- Skipped by default in CI
- Use the `us` profile / `us-east-2` region

When you add a new translation path, add a golden fixture alongside the implementation. The fixture represents the expected Bedrock-side JSON; the test asserts semantic equivalence (not byte equality) to tolerate field ordering differences.

---

### 9. Documented Divergences from the Python Gateway

| Behavior                                   | Python                                              | Rust                                                                                                                        |
| ------------------------------------------ | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Error responses                            | Sometimes returned plain text (non-JSON) on 4xx/5xx | Always returns full OpenAI error envelope: `{ "error": { "message": ..., "type": ..., "code": ... } }`                      |
| Cache-write token accounting               | Mapped `cacheWriteInputTokens` to a usage field     | Intentionally not mapped — no standard OpenAI field for write-side cache; `cached_tokens` reflects reads only               |
| Environment variable names                 | Required `APP_` prefix for most settings            | Accepts both `APP_` prefix and bare Python-parity names (`API_KEY`, `AWS_REGION`, `PORT`, etc.); bare names win on conflict |
| `reasoning_content`                        | Exposed as a top-level response field               | Never serialized to the wire (`#[serde(skip_serializing)]`); reasoning rendered as `<think>...</think>` inline in `content` |
| Responses `store` / `previous_response_id` | N/A (surface did not exist)                         | Accepted and silently ignored — this surface is stateless                                                                   |
| Responses stream `[DONE]` sentinel         | N/A                                                 | Not emitted — the Responses stream terminates with a `response.completed` event                                             |
| Responses `function_call_arguments.delta`  | N/A                                                 | Not emitted (codex does not require it; omitting it keeps the event set minimal)                                            |
| Responses built-in server tools            | N/A                                                 | Rejected with 400 (`web_search`, `file_search`, `code_interpreter`, `mcp`, `computer`, `image_generation`)                  |

---

### 10. Conventions

**Commits:** Conventional Commits format, Chinese subject line, imperative mood.
Examples: `feat: 添加 Nova embedding 支持`, `fix: 修复流式响应 finish_reason 映射`, `docs: 更新 AGENTS.md`

**Pre-commit (all three, in order):**

```
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

**No `src/` edits for model additions.** Config only.

**No `.legacy-python/` or `src/api/` edits** — those paths are reference artifacts.

**503/500 from Bedrock in CI:** transient. Retry the failing test. Verify `git status` and `git log` before re-running a task that may have already committed successfully.

**Two completely separate authentication directions — never mix them up:**

- **Client → gateway** (`API_KEY` / `API_KEY_SECRET_ARN` / `API_KEY_PARAM_NAME`): the bearer token that callers present to this proxy. Resolved in priority order: SSM Parameter Store → Secrets Manager → plain env var. Enforced in `server/auth.rs`.
- **Gateway → Bedrock** (`AWS_BEARER_TOKEN_BEDROCK` / `BEDROCK_API_KEY` alias, or SigV4 fallback): how the gateway authenticates with AWS. Set `AWS_BEARER_TOKEN_BEDROCK` to use a Bedrock API Key (bearer token, recommended for new deployments); leave it unset to fall back to the standard SigV4 credential chain (access key/secret, `AWS_PROFILE`, IMDS, ECS task role). Injected in `bedrock::client::build_aws_config` — zero branching, SDK-native. The internal field is `AppSettings::bedrock_api_key`; it is completely unrelated to `AppSettings::api_key`.

**Documentation layout:** the root directory contains only `README.md` and `AGENTS.md`. All other docs live under:

- `docs/readme/` — `README_CN.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `docs/deploy/` — deployment-specific guides (e.g. `lambda.md`)

New documentation files must follow this layout. Do not add `.md` files to the root.

---

---

## 中文

### 1. 项目概述

`bedrock-gateway-rust` 是一个兼容 OpenAI API 的 HTTP 网关，后端对接 AWS Bedrock，使用 Rust 编写。它替代了早期的 Python/FastAPI 实现，在保持与 OpenAI REST API 字节级兼容的同时大幅提升性能。运行时栈：**axum + tokio + aws-sdk-bedrockruntime**。Docker 镜像：`sunerpy/bedrock-gateway-rust`。

已支持端点（路径前缀通过 `API_ROUTE_PREFIX` 配置，默认 `/api/v1`）：

| 端点                            | 说明                                            |
| ------------------------------- | ----------------------------------------------- |
| `POST /api/v1/chat/completions` | 流式（SSE）+ 非流式                             |
| `POST /api/v1/responses`        | OpenAI Responses API 接口（无状态；codex 必需） |
| `POST /api/v1/embeddings`       | Cohere / Titan / Nova                           |
| `GET  /api/v1/models`           | 从 Bedrock 控制面刷新模型目录                   |
| `GET  /api/v1/models/{id}`      | 单个模型查询                                    |
| `GET  /api/v1/health`           | 存活探针                                        |

---

### 2. 架构说明

代码库采用严格分层结构，依赖关系只向下流动。

```
src/
├── main.rs              # tokio::main，AppSettings::load → telemetry::init → server::serve
├── lib.rs               # crate 根，重导出
├── error.rs             # AppError（thiserror），OpenAI 错误信封，HTTP 状态码映射
├── telemetry.rs         # tracing subscriber，ReloadHandle 用于动态调整日志级别
│
├── openai/
│   └── schema.rs        # 协议类型：ChatRequest、ChatResponse、ChatStreamResponse、
│                        #   ChatResponseMessage、Usage、Embeddings*、Model(s)、OpenAiError
│   └── responses_schema.rs  # Responses 接口类型：ResponsesRequest、ResponsesResponse、
│                            #   ResponseInputItem、ResponseOutputItem、ResponseStreamEvent、
│                            #   ResponsesUsage
│
├── domain/
│   └── mod.rs           # 与提供商无关的 trait 定义：
│                        #   ChatProvider、EmbeddingProvider（async_trait）
│                        #   ResponsesProvider（async_trait）
│                        #   ModelCapabilities、RegionRouter、EmbeddingBodyCodec（同步）
│                        #   NormalizedChatRequest { request, resolved_model }
│                        #   NormalizedResponsesRequest { request, resolved_model }
│                        #   ChatStream = BoxStream<'static, Result<ChatStreamResponse, AppError>>
│                        #   ResponsesStream = BoxStream<'static, Result<ResponseStreamEvent, AppError>>
│
├── config/
│   ├── settings.rs      # AppSettings::load，分层 env（APP_ 前缀 + 裸名覆盖列表）
│   ├── capabilities.rs  # ModelCapabilityConfig::load/from_toml_str、Capability 枚举、
│   │                    #   ReasoningPath 枚举、BudgetRatios
│   ├── regions.rs       # RegionRoutingConfig::load，RouteOverride { region, rewritten_model_id }
│   └── embeddings.rs    # EmbeddingRegistry::load，EmbeddingFamily { Cohere, Titan, Nova }
│
├── bedrock/
│   ├── capabilities.rs  # ConfigModelCapabilities 实现 domain::ModelCapabilities
│   ├── client.rs        # BedrockClients { runtime, control }，build_aws_config，
│   │                    #   region_config_override 用于单请求级别的 region 覆盖
│   ├── tokens.rs        # estimate_reasoning_tokens(&str) -> u32
│   │                    #   compute_token_usage(input, output, cacheRead, cacheWrite) -> Usage
│   ├── translate.rs     # to_converse_args：ChatRequest → ConverseArgs + ConverseExtras 接缝
│   ├── reasoning.rs     # build_reasoning_config → ReasoningOutcome；通过 ReasoningPath 支持 4 条路径
│   ├── tools.rs         # OpenAI tool_use ↔ Bedrock toolConfig 互转
│   ├── cache.rs         # Prompt 缓存 cache_point 注入（Claude + Nova）
│   │                    #   decorate_tools/system/messages 共享预算 ≤ max_cache_checkpoints
│   ├── response.rs      # from_converse_output：ConverseOutput → ChatResponse，
│   │                    #   <think> 内联渲染，usage 映射
│   ├── stream.rs        # StreamState 状态机 + converse_stream_to_openai async_stream 包装器
│   ├── embeddings.rs    # CohereCodec / TitanCodec / NovaCodec 实现 EmbeddingBodyCodec；
│   │                    #   BedrockEmbeddingProvider 实现 EmbeddingProvider
│   ├── models.rs        # ModelCatalog { models, profile_metadata }，通过控制面刷新
│   ├── provider.rs      # BedrockChatProvider 实现 ChatProvider，组合
│   │                    #   translate + reasoning + tools + cache → converse/converse_stream
│   │                    #   → response/stream 映射
│   ├── responses_translate.rs  # to_responses_converse_input：ResponsesRequest → Bedrock messages/system；
│   │                           #   reasoning_outcome 复用 build_reasoning_config
│   ├── responses_response.rs   # from_converse_output_to_responses：ConverseOutput → ResponsesResponse；
│   │                           #   推理 → 结构化 reasoning 输出项（非 <think>）
│   ├── responses_stream.rs     # ResponsesStreamState + converse_stream_to_openai_responses 包装器；
│   │                           #   完整生命周期事件，单调递增 sequence_number，无 [DONE] 哨兵
│   └── responses_provider.rs   # BedrockResponsesProvider 实现 ResponsesProvider，组合
│                                #   responses_translate + reasoning + cache → converse/converse_stream
│                                #   → responses_response/responses_stream 映射
│
└── server/
    ├── auth.rs          # Bearer token 中间件
    ├── state.rs         # AppState，build_app_state 组装所有组件
    ├── mod.rs           # serve(AppSettings) 入口，apply_layers（TraceLayer + CorsLayer）
    └── routers/
        └── mod.rs       # build_router：axum Router 配置所有端点
```

配置文件（不是代码）：

```
config/
├── models.toml      # 所有模型能力声明
├── regions.toml     # 跨区域路由规则
├── embeddings.toml  # Embedding 模型注册表
└── app.toml         # 应用默认值（可被环境变量覆盖）
```

#### ADR：HTTP 框架选型 — axum（已评估，保留）

HTTP 框架选用 **axum**（tokio + tower + tower-http）。曾评估替换为 actix-web，结论是**保留 axum**。保留理由：

1. **SSE 流式主干** — 流式路径原生基于 axum（`axum::response::Sse`）；迁移需重写整个 `server/` 层，无正确性收益。
2. **自定义 OpenAI 错误信封 + 鉴权语义** — axum 的 `FromRequestParts` + `IntoResponse` 能清晰编码 OpenAI 错误契约所要求的 401-vs-405 区分；actix 中间件实现同等语义需要更多样板代码。
3. **优雅关闭** — axum 的 `serve(...).with_graceful_shutdown(...)` 开箱即用地与 tokio 信号处理集成。

本服务是 IO 密集型 Bedrock 代理，actix-web 无可感知的吞吐量优势。Lambda Web Adapter 对框架中立，Lambda 部署路径不受影响。此决策已关闭，不应在没有明确 axum 瓶颈基准测试的情况下重新讨论。

#### Responses 接口

`POST /api/v1/responses` 是**第二个 OpenAI 接口层** — OpenAI Responses API — 与 chat completions 完全分离实现。它有独立的 provider trait（`src/domain/mod.rs` 中的 `ResponsesProvider`）、独立的协议类型（`src/openai/responses_schema.rs`），以及 `src/bedrock/` 下专属的四模块栈：

| 模块                     | 职责                                                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `responses_translate.rs` | 解析 `ResponsesRequest` 输入项 → Bedrock messages/system；复用 `build_reasoning_config` 处理思考预算                               |
| `responses_response.rs`  | 将 `ConverseOutput` 映射为 `ResponsesResponse`；推理 → 结构化 `reasoning` 输出项                                                   |
| `responses_stream.rs`    | `ResponsesStreamState` + `converse_stream_to_openai_responses`；完整生命周期事件，单调递增 `sequence_number`，**无 `[DONE]` 哨兵** |
| `responses_provider.rs`  | `BedrockResponsesProvider` 实现 `ResponsesProvider`；组合以上三模块 + 缓存注入                                                     |

该接口**无状态**：`store` 和 `previous_response_id` 接受但静默忽略（codex 发送 `store: false`）。它复用同一 Converse 调用层以及 `src/bedrock/tokens.rs` 中的共享 `compute_token_usage` helper。codex 仅支持此接口（`wire_api = "responses"`）。

**限制 / 拒绝矩阵：**

- 内置服务端工具（`web_search`、`file_search`、`code_interpreter`、`mcp`、`computer`、`image_generation`）→ 400。
- `encrypted_content` 不做透传。
- 无 `function_call_arguments.delta` 流事件（codex 宽容性决策）。
- `input_file` 部分 → 400（暂无 Bedrock 文档块映射）。

#### 缓存放置契约

缓存点自动注入**默认开启**（主开关 `enable_prompt_caching`，在 `config/app.toml` 和 `settings.rs` 中默认为 `true`）。放置顺序为 **tools → system → messages**，三个位置共享最多 `max_cache_checkpoints` 个缓存点的预算。`max_cache_checkpoints` 通过 `ModelCapabilities::max_cache_checkpoints` 由配置驱动（默认常量 4）。

一个模型"支持缓存"（`cache.rs` 中的 `supports_caching`），当且仅当其在 `config/models.toml` 中的条目包含 `cache_min_tokens` 参数。这是配置门控 — 代码中不做任何模型名称检查。

**字节稳定前缀规则：** 缓存命中依赖确定性序列化。修改 `cachePoint` 之前的任何段都会使该请求中后续所有缓存点失效。保持早期段在多轮对话中的稳定性。

**Token 计账**由 `src/bedrock/tokens.rs` 中的单一 `compute_token_usage(input, output, cacheRead, cacheWrite)` helper 完成：

- `prompt_tokens` = `input + cacheRead + cacheWrite`
- `total_tokens` = `prompt_tokens + output`
- `cached_tokens` = 仅 `cacheRead`

Bedrock 返回的 `cacheWriteInputTokens` 折入 `prompt_tokens`，但**永不作为独立协议字段**（OpenAI 协议无写侧缓存计费字段）。`response.rs` 和 `stream.rs`（chat 接口）以及 `responses_response.rs` / `responses_stream.rs`（Responses 接口）全部调用同一个 helper — 不要重复这个公式。

逐模型 `cache_min_tokens` 阈值、reasoning budget 行为和跨区域 inference profile 规则，详见 [`docs/caching-and-reasoning.md`](docs/caching-and-reasoning.md)。

#### 两条推理渲染路径（架构规则）

推理输出在两个接口层上采用**不同形式**，绝不能统一：

| 接口                        | 推理渲染方式                                                                                                                                          |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat（`/chat/completions`） | 内联 `<think>...</think>` 嵌入 `content` 字符串。`ChatResponseMessage` 中的 `reasoning_content` 带有 `#[serde(skip_serializing)]`，永不出现在协议层。 |
| Responses（`/responses`）   | `output` 数组中的结构化 `reasoning` 输出项。不包裹在 `<think>` 中。                                                                                   |

修改任一渲染路径时，请确认另一路径未受影响。不要合并两者。

---

### 3. 零硬编码契约（关键规则）

**所有模型知识只存在于 `config/*.toml`。Rust 代码只包含匹配算法。**

| `.rs` 中允许的内容                                   | `.rs` 中禁止的内容                   |
| ---------------------------------------------------- | ------------------------------------ |
| SSE 协议常量（`data: `、`[DONE]`）                   | 模型 ID（`anthropic.claude-*` 等）   |
| OpenAI 对象类型字符串（`chat.completion` 等）        | 与模型名称绑定的能力标志             |
| `chatcmpl-` ID 前缀                                  | 上下文窗口或 token 预算的魔法数字    |
| `finish_reason` 值（`stop`、`length`、`tool_calls`） | 任何 `if model.contains("...")` 逻辑 |

**唯一已记录的例外：** `src/bedrock/provider.rs` 中有一个 `skip_tool_choice_for` 检查，用于检测 `meta.llama3-1-`。此处已在代码中明确注释，并标记为待替换为 `models.toml` 中的能力标志。不要在没有同等记录的情况下新增类似例外。

如果你发现自己在 Rust 里写 `if model_id.contains("claude")`，停下来。改为在 `config/models.toml` 中添加能力标志，然后通过 `ModelCapabilities::has(Capability::...)` 读取它。

---

### 4. Option-B 合规性

网关对外呈现字节级兼容的 OpenAI 协议格式。Bedrock 专属特性**只**通过 OpenAI 官方认可的 `extra_body` 机制暴露，不引入任何自定义的顶层请求字段。

**推理 / 扩展思考：** 渲染为 `<think>...</think>` 内联在 `content` 字符串中。`ChatResponseMessage` 中的 `reasoning_content` 字段带有无条件的 `#[serde(skip_serializing)]`，即使内部有值也绝不出现在协议层。

**Prompt 缓存：** 通过 `extra_body: { "prompt_caching": { "system": true, "messages": true } }` 请求。`PromptTokensDetails` 中的 `cached_tokens` 只反映**缓存读取**的 token 数。Bedrock 返回的 `cacheWriteInputTokens` 已知但有意不映射（OpenAI 协议中无对应的写侧缓存计费字段）。

**规则：** 添加任何 Bedrock 专属功能，都要走 `openai::schema::ChatRequest` 中的 `extra_body` 解析路径（通过 `#[serde(flatten)] extra: HashMap<String, Value>`）。不要为 Bedrock 概念在 `ChatRequest` 或 `ChatResponse` 上新增顶层字段。

#### 日志 / 可观测性

`info` 级别记录每个请求的访问日志（method/path/status/latency，由 `server/mod.rs` 中配置为 INFO 级的 axum `TraceLayer` 输出）以及 handler 的关键业务元数据（`model`、是否流式、`finish_reason`、token **数量**）。`debug` 级别额外记录上游 Bedrock 调用细节（解析后的 model、目标 region，来自 `bedrock/provider.rs`）。**任何**级别（即便 `debug`）都**绝不**记录请求/响应 body、消息内容、prompt/completion 文本、token 明文值或 `API_KEY`/bearer token —— 只记元数据。新增日志时使用结构化 `tracing` 字段；切勿 `Debug` 打印整个 request/response 结构体。

---

### 5. 如何添加新模型

无需修改代码。编辑 `config/models.toml`：

```toml
[[model]]
match = "your-provider.your-model-id"   # 前缀或精确字符串
capabilities = ["TemperatureToppConflict"]  # 零个或多个 Capability 变体
[model.params]
max_tokens = 8192
context_window = 200000
# reasoning_path = "BudgetTokens"  # 如果模型支持扩展思考
```

跨区域路由在 `config/regions.toml` 中添加条目。新的 Embedding 模型在 `config/embeddings.toml` 中添加，并指定对应的 `family` 字段。

对于从磁盘读取配置的部署方式，纯配置变更无需重新编译。容器化部署中配置文件已打包进镜像，需重新构建镜像才能生效。

---

### 6. Trait 扩展点

如需接入非 Bedrock 的后端，实现 `src/domain/mod.rs` 中定义的 trait：

| Trait                | 同步/异步 | 职责                                                                            |
| -------------------- | --------- | ------------------------------------------------------------------------------- |
| `ChatProvider`       | 异步      | 将 `NormalizedChatRequest` 转换为 `ChatResponse` 或 `ChatStream`                |
| `EmbeddingProvider`  | 异步      | 将 Embedding 请求转换为 `EmbeddingsResponse`                                    |
| `ResponsesProvider`  | 异步      | 将 `NormalizedResponsesRequest` 转换为 `ResponsesResponse` 或 `ResponsesStream` |
| `ModelCapabilities`  | 同步      | 查询指定模型 ID 的能力与路由元数据                                              |
| `RegionRouter`       | 同步      | 返回指定模型 ID 的 `RouteOverride`                                              |
| `EmbeddingBodyCodec` | 同步      | 对特定模型系列的 Embedding 请求/响应字节进行编解码                              |

目前只有 Bedrock 后端实现（`src/bedrock/`）。这些 trait 设计上不含任何 AWS 类型，是提供商无关的抽象。

在 `src/server/state.rs` 的 `build_app_state` 中接入新提供商，遵循 `BedrockChatProvider` 的 `Arc` 包装模式。

---

### 7. 构建 / 测试 / 部署命令

```bash
# 开发
cargo build                                              # debug 构建
cargo build --release                                   # 发布版二进制 → target/release/bedrock-gateway
cargo test                                              # 所有测试（单元 + golden + doctest）
cargo clippy --all-targets --all-features -- -D warnings  # 必须零警告
cargo fmt                                               # 格式检查 / 应用格式

# Makefile 快捷方式
make help                                               # 列出所有目标

# Docker（本地）
docker build -t bedrock-gateway-rust .                  # 从根 Dockerfile 构建 distroless 镜像

# 本地运行（健康检查无需真实 AWS 凭证）
API_KEY=testkey cargo run
curl http://localhost:8080/api/v1/health
```

**提交前检查（每次提交前必须按序执行）：**

```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

**部署目标：**

| 部署方式               | 文件                                                                        |
| ---------------------- | --------------------------------------------------------------------------- |
| ECS/Fargate（ALB）     | `deployment/BedrockGatewayFargate.template` + 根目录 `Dockerfile`           |
| Lambda（Function URL） | `deployment/BedrockGatewayLambda.template` + `deployment/lambda/Dockerfile` |
| Lambda 文档            | `docs/deploy/lambda.md`                                                     |

两个 CloudFormation 模板均使用裸环境变量名（`API_KEY`、`AWS_REGION`、`DEFAULT_MODEL` 等）。完整允许列表见 `src/config/settings.rs` 中的 `apply_bare_env_overrides`。

Lambda 注意事项：**不要**在 Lambda 环境中设置 `AWS_REGION`，这是 Lambda 保留变量，cfn-lint 会报 `E3663` 错误。Lambda 运行时会自动注入该变量。

---

### 8. 等价性验证 / Golden Replay 工作流

测试分两层：

**第一层：离线 golden record/replay**（`tests/golden/`）：

- Fixture 固定对齐 Python HEAD `9a3e752`
- 断言辅助函数：`assert_semantic_eq`（无序字段比较）和 `assert_stream_eq`
- 在 CI 中自动运行，无需 AWS 凭证
- `cargo test` 默认执行这些测试

**第二层：实时集成测试**（需显式开启）：

```bash
BEDROCK_INTEGRATION=1 AWS_PROFILE=us cargo test -- --ignored
```

- 需要真实 AWS 凭证和 Bedrock 访问权限，区域为 `us-east-2`
- CI 默认跳过
- 使用 `us` profile / `us-east-2` 区域

添加新的翻译路径时，请随实现一并添加 golden fixture。Fixture 表示预期的 Bedrock 侧 JSON，测试断言语义等价性（而非字节相等），以容忍字段顺序差异。

---

### 9. 与 Python 网关的已记录差异

| 行为                                       | Python                                       | Rust                                                                                                        |
| ------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 错误响应格式                               | 部分 4xx/5xx 返回纯文本（非 JSON）           | 始终返回完整 OpenAI 错误信封：`{ "error": { "message": ..., "type": ..., "code": ... } }`                   |
| 缓存写入 token 计账                        | 将 `cacheWriteInputTokens` 映射到 usage 字段 | 有意不映射，OpenAI 协议无写侧缓存计费字段；`cached_tokens` 只反映读取侧                                     |
| 环境变量名                                 | 大多数配置需要 `APP_` 前缀                   | 同时接受 `APP_` 前缀和 Python 兼容的裸变量名（`API_KEY`、`AWS_REGION`、`PORT` 等），裸变量名优先            |
| `reasoning_content` 字段                   | 作为响应的顶层字段暴露                       | 永不序列化到协议层（`#[serde(skip_serializing)]`）；推理内容以 `<think>...</think>` 内联在 `content` 中呈现 |
| Responses `store` / `previous_response_id` | N/A（接口不存在）                            | 接受但静默忽略 — 该接口无状态                                                                               |
| Responses 流 `[DONE]` 哨兵                 | N/A                                          | 不发送 — Responses 流以 `response.completed` 事件结束                                                       |
| Responses `function_call_arguments.delta`  | N/A                                          | 不发送（codex 不要求；省略保持事件集最小化）                                                                |
| Responses 内置服务端工具                   | N/A                                          | 返回 400（`web_search`、`file_search`、`code_interpreter`、`mcp`、`computer`、`image_generation`）          |

---

### 10. 开发规范

**提交信息：** Conventional Commits 格式，中文主题行，祈使句风格。
示例：`feat: 添加 Nova embedding 支持`，`fix: 修复流式响应 finish_reason 映射`，`docs: 更新 AGENTS.md`

**提交前三步（按序执行）：**

```
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

**添加模型只改配置，不改 `src/`。**

**不修改 `.legacy-python/` 或 `src/api/`** — 这些路径是参考制品，不是活跃代码。

**CI 中遇到 Bedrock 503/500：** 属于间歇性故障，重试即可。在重跑任务前先执行 `git status` 和 `git log` 确认是否已提交成功，避免重复执行已完成的工作。

**两个完全独立的鉴权方向，绝不混淆：**

- **客户端→网关**（`API_KEY` / `API_KEY_SECRET_ARN` / `API_KEY_PARAM_NAME`）：调用方向本代理出示的 bearer token。按优先级解析：SSM Parameter Store → Secrets Manager → 裸环境变量。在 `server/auth.rs` 中执行。
- **网关→Bedrock**（`AWS_BEARER_TOKEN_BEDROCK` / 别名 `BEDROCK_API_KEY`，或 SigV4 回退）：网关向 AWS 鉴权的方式。设置 `AWS_BEARER_TOKEN_BEDROCK` 即使用 Bedrock API Key（bearer token，新部署推荐）；不设置则自动回退到标准 SigV4 凭证链（access key/secret、`AWS_PROFILE`、IMDS、ECS task role）。注入点在 `bedrock::client::build_aws_config`，零分支，SDK 原生支持。内部字段为 `AppSettings::bedrock_api_key`，与 `AppSettings::api_key` 完全无关。

**文档布局约定：** 根目录只保留 `README.md` 和 `AGENTS.md`。其余文档位于：

- `docs/readme/` — `README_CN.md`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md`
- `docs/deploy/` — 部署专项文档（如 `lambda.md`）

新增文档必须遵循此布局。不得在根目录添加其他 `.md` 文件。
