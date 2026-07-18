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
| `POST /api/v1/completions`      | OpenAI legacy text completions (Zed edit-prediction); reuses the Converse path, wire shape `text_completion` |
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
│   ├── responses_provider.rs   # BedrockResponsesProvider implements ResponsesProvider — composes
│   │                           #   responses_translate + reasoning + cache → converse/converse_stream
│   │                           #   → responses_response/responses_stream mapping
│   ├── responses_chat_provider.rs # ResponsesChatProvider adapts ResponsesProvider to ChatProvider;
│   │                              #   Chat↔Responses mapping, incremental SSE decoding, rsc_v1 replay
│   ├── mantle_client.rs        # MantleClient: raw HTTP client for the bedrock-mantle OpenAI-compatible
│   │                           #   upstream (bedrock-mantle.{region}.api.aws); byte-level SSE passthrough
│   │                           #   via responses_nonstream / responses_stream; pre-stream errors mapped
│   │                           #   to AppError; mid-stream errors truncate (no envelope after 200+headers)
│   ├── mantle_provider.rs      # MantleResponsesProvider implements ResponsesProvider for GPT-5.x models;
│   │                            #   region gate → model rewrite (only "model" field patched) →
│   │                            #   responds_raw_stream override (Some(raw)) for streaming verbatim
│   │                            #   passthrough; respond for non-stream; respond_stream as typed fallback
│   └── mantle_chat_provider.rs  # MantleChatProvider implements ChatProvider for gpt-oss chat models;
│                                #   chat surface analogue of mantle_provider; raw Bytes end to end
│                                #   (chat_raw_stream / chat_raw_nonstream); typed chat/chat_stream are
│                                #   unreachable fallbacks (AppError::Internal); no [DONE] appended here
│
└── server/
    ├── auth.rs          # Bearer-token middleware
    ├── state.rs         # AppState, build_app_state assembles all components
    ├── composite.rs     # CompositeResponsesProvider: single Arc<dyn ResponsesProvider> that
    │                    #   dispatches to Converse (BedrockResponsesProvider) or Mantle
    │                    #   (MantleResponsesProvider) by caps.responses_backend(model);
    │                    #   resolve_mantle_enabled: disables mantle if no bedrock_api_key;
    │                    #   soft WARN for region mismatches at boot
    ├── composite_chat.rs # CompositeChatProvider: single Arc<dyn ChatProvider> that dispatches to
    │                    #   Converse, native Mantle Chat, or ResponsesChatProvider by
    │                    #   caps.chat_backend(model); overrides both raw chat lanes;
    │                    #   resolve_mantle_chat_enabled mirrors resolve_mantle_enabled
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

**Composite dispatcher:** `AppState.responses` holds a single `Arc<dyn ResponsesProvider>` — in practice a `CompositeResponsesProvider` (`src/server/composite.rs`) that picks the right backend at request time by calling `caps.responses_backend(model)`. Models with `responses_backend = "mantle"` in `config/models.toml` go to `MantleResponsesProvider` (raw byte passthrough); all others go to `BedrockResponsesProvider` (Converse path). The composite overrides `respond_raw_stream` so the mantle streaming lane fires correctly; the Converse provider inherits the default (`None`), keeping its existing typed-stream path unchanged.

**Tool support / rejection matrix:**

- User-defined tools are SUPPORTED and translated to Bedrock `toolConfig`:
  - `function` (flattened Responses shape) → one `toolSpec` keeping its bare name.
  - `custom` → one `toolSpec` with a reversible string-input adapter (name + description; the `format` grammar has no Bedrock slot and is dropped).
  - `namespace` (a named container of inner tools) → FLATTENED internally: one `toolSpec` per inner tool, with each Bedrock name prefixed `{namespace_name}__{inner_name}` (double-underscore delimiter) so different namespaces can't collide. Inner `function` and `custom` tools are both supported. Responses restore the client-visible bare `name` plus `namespace`; continuation requests join them again before Bedrock.
  - Client-executed `local_shell`, `shell`, and `apply_patch` → one typed `toolSpec` each; output and continuation items retain their original Responses item type and current AI SDK field shape.
- Hosted OpenAI server tools with no Bedrock equivalent (`web_search`, `file_search`, `code_interpreter`, `tool_search`, `mcp`, `computer`, `image_generation`) and ANY unrecognized/future tool type are now **SILENTLY DROPPED** (skipped from `toolConfig`, never a 400). codex unconditionally bundles some hosted tools; a 400 there would kill the whole session including the user's real function tools. The `ResponsesTool` enum carries a `#[serde(other)] Unknown` catch-all so an unrecognized tool `type` NEVER fails deserialization at the wire boundary.
- Signed Bedrock reasoning is carried through an opaque, versioned `reasoning.encrypted_content` envelope and reconstructed unchanged on stateless continuation requests.
- Function tools emit `function_call_arguments.delta/.done` stream events (including `name` on `.done`) before `response.output_item.done`.
- `input_file` parts → 400 (no Bedrock document-block mapping).
- **toolConfig synthesis for tool-continuation requests:** when a request carries prior `toolUse` / `toolResult` history in its messages but omits the `tools` array, the gateway synthesizes a minimal `toolConfig` automatically — one placeholder `toolSpec` per distinct tool name seen in the history, using a fixed description string and an empty-object input schema (`{"type":"object"}`). This satisfies Bedrock's validation rule ("The toolConfig field must be defined when using toolUse and toolResult content blocks") without requiring the client to re-send the original tool definitions on every turn. Synthesis fires only when no real `toolConfig` was supplied and at least one `toolUse` block exists in the message history. Applies to both `/chat/completions` (via `synthesize_tool_config_from_messages` in `bedrock/tools.rs`, called from `bedrock/provider.rs`) and `/responses` (called from `bedrock/responses_provider.rs`).

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
| Mantle chat (`/chat/completions`, gpt-oss) | A THIRD, passthrough-only case: reasoning survives verbatim as the upstream non-standard `delta.reasoning` field (plus per-chunk `obfuscation`). The gateway NEVER rewrites it to `<think>` and NEVER drops it — the mantle chat lane is raw bytes end to end, so the fields simply pass through unparsed. |

If you touch either rendering path, verify the others are unchanged. Do not merge them.

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

**Request-handler error severity:** handler failures are logged at `error!` for 5xx errors (`UpstreamBedrock` / `Internal`) and at `warn!` for 4xx errors (`BadRequest` / `Unsupported` / `Unauthorized` / `Throttled`). The distinction is implemented via `AppError::is_server_error()` in `error.rs`, with five branching arms in `server/routers/mod.rs`.

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

#### Model-ID aliases (`[[alias]]` table)

To let clients use a short name that resolves to a canonical model ID before any capability or region lookup, add an entry to the `[[alias]]` table at the **top** of `config/models.toml` (before the first `[[model]]` entry — TOML positional constraint):

```toml
[[alias]]
from = "short-name"      # what the client sends
to   = "provider.full-id"  # the canonical id the gateway resolves it to
```

Alias resolution runs before the runtime inference-profile map, so it works even with no live Bedrock catalog. The current GPT aliases include `gpt-5.4`, `gpt-5.5`, and `gpt-5.6-sol/terra/luna`, each mapped to its `openai.*` canonical id.

#### GPT-5.x via bedrock-mantle (`responses_backend = "mantle"`)

Models served through AWS Bedrock's mantle OpenAI-compatible upstream use a different backend than the standard Converse path. Three `[model.params]` fields control this:

| Field               | Type              | Meaning                                                                                           |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------- |
| `responses_backend` | `"mantle"`        | Routes this model to `MantleResponsesProvider` instead of `BedrockResponsesProvider`              |
| `chat_backend`      | `"responses"`     | Exposes Chat Completions through the stateless `ResponsesChatProvider` protocol adapter            |
| `available_regions` | array of strings  | Region allow-list; absent = available everywhere; non-empty = per-request 400 if region not listed |

Example (from `config/models.toml`):

```toml
[[model]]
match = "openai.gpt-5.5"
capabilities = []
[model.params]
responses_backend = "mantle"
chat_backend = "responses"
available_regions = ["us-east-2"]
```

**Startup behavior:** if a model declares `responses_backend = "mantle"` but no `bedrock_api_key` (env `AWS_BEARER_TOKEN_BEDROCK` / `BEDROCK_API_KEY`) is set, the gateway logs a WARN and disables ONLY the GPT-5.x mantle models — it does NOT fail to start, and all other models keep working. Requests for a disabled mantle model return a clear 400. `DISABLE_MANTLE=true` explicitly disables the mantle backend the same way. (This replaced the earlier hard-fail behavior.) Region mismatches between the running instance's `AWS_REGION` and a model's `available_regions` emit a WARN at boot but don't hard-fail (the per-request gate returns 400 instead).

**Mantle endpoint template:** the upstream URL is controlled by `MANTLE_BASE_URL_TEMPLATE` (default `https://bedrock-mantle.{region}.api.aws/openai/v1`). The literal `{region}` placeholder is replaced with the gateway's `AWS_REGION` at call time. Change this env var to point at a private or test mantle endpoint without recompiling.

**Constraints for mantle models:**
- `/responses` remains raw SSE byte passthrough: no `[DONE]`; the stream terminates on mantle's `response.completed`.
- `/chat/completions` is supported through `chat_backend = "responses"`; it maps Chat requests and Responses output without calling mantle's unsupported native Chat route.
- Reasoning summaries become inline `<think>` text; raw chain-of-thought is never exposed. Reasoning + tool continuation requires the shared `CHAT_REASONING_CAPSULE_*` keyring and uses authenticated `rsc_v1` tool ids.
- `/completions` remains unsupported.
- Listed in `GET /models` under bare aliases including `gpt-5.4`, `gpt-5.5`, and `gpt-5.6-sol/terra/luna`, injected from config because the control plane omits mantle models.
- Auth reuses the same `bedrock_api_key` bearer (`AWS_BEARER_TOKEN_BEDROCK`) the gateway uses for Converse calls.

#### gpt-oss chat via bedrock-mantle (`chat_backend = "mantle"`)

The gpt-oss family (`gpt-oss-120b` / `gpt-oss-20b`) is served on the **chat** surface through the mantle upstream, using a SEPARATE, independent field from `responses_backend`:

| Field           | Type              | Meaning                                                                                     |
| --------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| `chat_backend`  | `"mantle"`        | Routes this model's `/chat/completions` to `MantleChatProvider` instead of the Converse path |
| `available_regions` | array of strings | Region allow-list; absent = everywhere; non-empty = per-request 400 if region not listed    |

`chat_backend` and `responses_backend` never cross-contaminate — a model may be mantle on one surface and Converse on the other. gpt-oss declares ONLY `chat_backend = "mantle"` (chat-only this round).

**Chat endpoint template:** the chat upstream URL is controlled by `MANTLE_CHAT_BASE_URL_TEMPLATE` (default `https://bedrock-mantle.{region}.api.aws/v1` — note NO `/openai` prefix, unlike the responses `MANTLE_BASE_URL_TEMPLATE`). The mantle chat route (`/chat/completions`) differs from the responses route (`/openai/v1/responses`); this is why the two templates are distinct. `{region}` is substituted with `AWS_REGION` at call time.

**Constraints for gpt-oss chat models:**
- `/chat/completions` only — `/completions` returns a clean 400.
- Raw byte passthrough (stream + non-stream) — no Converse translation, no usage re-normalization. The non-standard `delta.reasoning` and per-chunk `obfuscation` fields survive verbatim.
- The gateway's raw chat SSE lane APPENDS `data: [DONE]\n\n` at the stream tail (mantle chat does not emit it; OpenAI chat clients expect it). This is the ONE behavioral difference from the responses raw lane, which does NOT append `[DONE]`.
- Listed in `GET /models` under their bare alias names (`gpt-oss-120b` / `gpt-oss-20b`), surfaced from `[[alias]]` config (`mantle_alias_names` matches `chat_backend == "mantle"` too).
- Startup: if configured but no `bedrock_api_key` is set (or `DISABLE_MANTLE=true`), ONLY the gpt-oss chat models are disabled with a WARN; the gateway still starts and other models work. A disabled model returns a clear 400.

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

# Git hooks — run ONCE after clone to enable the pre-push quality gate
make hooks                                              # git config core.hooksPath .githooks

# Docker (local)
docker build -t bedrock-gateway-rust .                  # distroless image from root Dockerfile

# Run locally (no real AWS creds needed for health check)
API_KEY=testkey cargo run
curl http://localhost:8080/api/v1/health
```

**Pre-push gate (recommended — gates `git push`, never `git commit`):**

Run `make hooks` once after cloning to point `core.hooksPath` at the version-controlled `.githooks/` directory. This enables `.githooks/pre-push`, which runs the exact same three checks as CI (`cargo fmt --all -- --check` → `cargo clippy --all-targets --all-features -- -D warnings` → `cargo test --all-features`) and aborts the push on any failure with a clear (Chinese) message. Plain `git commit` is intentionally left unblocked so you can freely commit WIP; only `git push` is gated. "Passes pre-push" therefore implies "passes CI".

> `core.hooksPath` is a local git setting and is NOT applied automatically on clone — each contributor must run `make hooks` (or `make setup-hooks`) once.

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
| Responses `function_call_arguments.delta`  | N/A                                                 | Emits ordered `delta` / `done` events; `.done` includes the function name and precedes `response.output_item.done` |
| Responses `namespace` / `custom` / client tools | N/A                                            | SUPPORTED — reversible `custom` string input; namespace names are flattened only toward Bedrock and restored as `namespace` + bare `name`; `local_shell` / `shell` / `apply_patch` retain their Responses item types |
| Responses hosted server tools              | N/A                                                 | SILENTLY DROPPED (`web_search` / `file_search` / `code_interpreter` / `tool_search` / `mcp` / `computer` / `image_generation` + any unknown type) — never a 400, so codex sessions bundling hosted tools survive; `ResponsesTool` has a `#[serde(other)] Unknown` catch-all |
| GPT-5.x (`gpt-5.4` / `gpt-5.5` / `gpt-5.6-*`) models | N/A | Served via AWS bedrock-mantle Responses (`responses_backend = "mantle"`). `/responses` is raw passthrough; `/chat/completions` uses the stateless adapter (`chat_backend = "responses"`) with summary rendering, reasoning-token mapping, and authenticated `rsc_v1` tool replay. `/completions` remains unsupported. Aliases and region gates live in `config/models.toml`. |
| Legacy text completions (`POST /completions`) | N/A (surface did not exist)                    | Added for Zed edit-prediction. Reuses the Chat/Converse path (prompt wrapped as one user message), wire shape `text_completion`. `suffix` → 400 (no Bedrock mapping); token-array `prompt` → 400 (send a string); `logprobs` / `best_of` / `logit_bias` accepted and ignored. |
| gpt-oss chat via mantle (`gpt-oss-120b` / `gpt-oss-20b`) | N/A                                    | Served via AWS bedrock-mantle on the CHAT surface (`chat_backend = "mantle"`, an independent field from `responses_backend`), **`/chat/completions` only** — `/completions` returns 400. Byte-level raw SSE passthrough; no Converse translation, no usage re-normalization. The gateway APPENDS `data: [DONE]` at the stream tail (the ONE difference from the responses raw lane, which does not). Non-standard `delta.reasoning` + per-chunk `obfuscation` are preserved verbatim (never mapped to `<think>`, never dropped). Chat base template `MANTLE_CHAT_BASE_URL_TEMPLATE` = `https://bedrock-mantle.{region}.api.aws/v1` (no `/openai` prefix). Listed in `GET /models` by bare alias name (`gpt-oss-120b` / `gpt-oss-20b`). Region-gated: `us-east-1` / `us-east-2` / `us-west-2`. |
 
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
| `POST /api/v1/completions`      | OpenAI 传统文本补全（Zed edit-prediction）；复用 Converse 路径，协议形状 `text_completion` |
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
│   ├── responses_provider.rs   # BedrockResponsesProvider 实现 ResponsesProvider，组合
│   │                           #   responses_translate + reasoning + cache → converse/converse_stream
│   │                           #   → responses_response/responses_stream 映射
│   ├── responses_chat_provider.rs # ResponsesChatProvider 将 ResponsesProvider 适配为 ChatProvider；
│   │                              #   Chat↔Responses 映射、增量 SSE 解码、rsc_v1 续轮
│   ├── mantle_client.rs        # MantleClient：bedrock-mantle OpenAI 兼容上游的原始 HTTP 客户端
│   │                           #   （bedrock-mantle.{region}.api.aws）；字节级 SSE 透传，通过
│   │                           #   responses_nonstream / responses_stream；流前错误映射为 AppError；
│   │                           #   流中错误截断（200+headers 已发送后无法封装错误信封）
│   ├── mantle_provider.rs      # MantleResponsesProvider 实现 ResponsesProvider，处理 GPT-5.x 模型；
│   │                            #   区域门控 → 模型名称改写（仅改写 "model" 字段）→
│   │                            #   respond_raw_stream 覆盖（返回 Some(raw)）用于流式字节透传；
│   │                            #   respond 用于非流式；respond_stream 为有类型兜底路径
│   └── mantle_chat_provider.rs  # MantleChatProvider 实现 ChatProvider，处理 gpt-oss chat 模型；
│                                #   mantle_provider 的 chat 对应物；stream 与 non-stream 全程 raw Bytes
│                                #   （chat_raw_stream / chat_raw_nonstream）；有类型 chat/chat_stream 为
│                                #   不可达兜底（AppError::Internal）；此处不追加 [DONE]
│
└── server/
    ├── auth.rs          # Bearer token 中间件
    ├── state.rs         # AppState，build_app_state 组装所有组件
    ├── composite.rs     # CompositeResponsesProvider：单个 Arc<dyn ResponsesProvider>，
    │                    #   通过 caps.responses_backend(model) 在请求时分发至
    │                    #   Converse（BedrockResponsesProvider）或 Mantle（MantleResponsesProvider）；
    │                    #   resolve_mantle_enabled：若缺少 bedrock_api_key 则禁用 mantle；
    │                    #   区域不匹配时启动 WARN（不硬失败）
    ├── composite_chat.rs # CompositeChatProvider：单个 Arc<dyn ChatProvider>，通过
    │                    #   caps.chat_backend(model) 分发至 Converse、原生 Mantle Chat
    │                    #   或 ResponsesChatProvider；覆盖两条 raw chat 通道；
    │                    #   resolve_mantle_chat_enabled 与 resolve_mantle_enabled 对应
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

**复合调度器：** `AppState.responses` 持有一个 `Arc<dyn ResponsesProvider>` — 实际上是 `CompositeResponsesProvider`（`src/server/composite.rs`），它在请求时通过 `caps.responses_backend(model)` 选择后端。`config/models.toml` 中设置了 `responses_backend = "mantle"` 的模型走 `MantleResponsesProvider`（字节级透传）；其余模型走 `BedrockResponsesProvider`（Converse 路径）。Composite 覆盖了 `respond_raw_stream` 以确保 mantle 流式通道正确触发；Converse provider 继承默认实现（`None`），保持原有有类型流路径不变。

**工具支持 / 拒绝矩阵：**

- 用户定义的工具均**支持**，翻译为 Bedrock `toolConfig`：
  - `function`（扁平化 Responses 形态）→ 一个 `toolSpec`，保留其裸名称。
  - `custom` → 一个带可逆字符串输入适配器的 `toolSpec`（name + description；`format` 语法在 Bedrock 无对应槽位，丢弃）。
  - `namespace`（命名的内部工具容器）→ 仅在 Bedrock 侧**扁平化**：每个内部工具生成一个 `toolSpec`，Bedrock 名称加前缀 `{namespace_name}__{inner_name}`（双下划线分隔符），以保证不同 namespace 之间不冲突。内部的 `function` 与 `custom` 均支持。响应恢复客户端可见的裸 `name` + `namespace`；续轮请求在发送 Bedrock 前重新拼接。
  - 客户端执行的 `local_shell`、`shell`、`apply_patch` → 各生成一个有类型的 `toolSpec`；输出和续轮项保留原 Responses 项类型及当前 AI SDK 字段形态。
- 无 Bedrock 对应的内置服务端工具（`web_search`、`file_search`、`code_interpreter`、`tool_search`、`mcp`、`computer`、`image_generation`）以及**任何**未识别 / 未来的工具类型，现在一律**静默丢弃**（从 `toolConfig` 跳过，绝不返回 400）。codex 会无条件捆绑部分内置工具；此处返回 400 会连同用户的真实 function 工具一起断掉整个会话。`ResponsesTool` 枚举带有 `#[serde(other)] Unknown` 兜底变体，因此未识别的工具 `type` 在协议边界**永不**反序列化失败。
- 已签名的 Bedrock 推理通过不透明、带版本的 `reasoning.encrypted_content` 信封携带；无状态续轮时原样重建。
- function 工具在 `response.output_item.done` 前发送 `function_call_arguments.delta/.done` 流事件（`.done` 含 `name`）。
- `input_file` 部分 → 400（暂无 Bedrock 文档块映射）。
- **toolConfig 自动合成（工具续轮请求）：** 当请求消息历史中包含先前的 `toolUse` / `toolResult` 块但 `tools` 数组缺失时，网关自动合成一个最小 `toolConfig` —— 对历史中出现的每个不同工具名各生成一个占位 `toolSpec`，使用固定描述字符串和空对象输入 schema（`{"type":"object"}`）。这满足了 Bedrock 的校验规则（"The toolConfig field must be defined when using toolUse and toolResult content blocks"），无需客户端在每轮都重复发送原始工具定义。合成仅在未提供真实 `toolConfig` 且消息历史中存在至少一个 `toolUse` 块时触发。同时适用于 `/chat/completions`（通过 `bedrock/tools.rs` 中的 `synthesize_tool_config_from_messages`，由 `bedrock/provider.rs` 调用）和 `/responses`（由 `bedrock/responses_provider.rs` 调用）。

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
| Mantle chat（`/chat/completions`，gpt-oss） | 第三种、仅透传的情况：推理以上游非标准 `delta.reasoning` 字段（以及每 chunk 的 `obfuscation`）原样保留。网关既不改写为 `<think>` 也不丢弃 —— mantle chat 通道全程 raw bytes，字段直接原样透传。 |

修改任一渲染路径时，请确认其他路径未受影响。不要合并它们。

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

**请求 handler 错误严重级别：** handler 失败时，5xx 错误（`UpstreamBedrock` / `Internal`）记录为 `error!`，4xx 错误（`BadRequest` / `Unsupported` / `Unauthorized` / `Throttled`）记录为 `warn!`。该区分通过 `error.rs` 中的 `AppError::is_server_error()` 实现，`server/routers/mod.rs` 中有五个分支处理各种情况。

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

#### 模型 ID 别名（`[[alias]]` 表）

要让客户端使用短名称，并在任何能力或区域查找之前解析为规范模型 ID，在 `config/models.toml` **顶部**（第一个 `[[model]]` 条目之前 — TOML 位置约束）添加 `[[alias]]` 条目：

```toml
[[alias]]
from = "short-name"      # 客户端发送的名称
to   = "provider.full-id"  # 网关解析为的规范 ID
```

别名解析先于运行时 inference-profile 映射，因此即使没有实时 Bedrock 目录也能生效。当前 GPT 别名包括 `gpt-5.4`、`gpt-5.5` 和 `gpt-5.6-sol/terra/luna`，分别映射到对应的 `openai.*` 规范 ID。

#### GPT-5.x 通过 bedrock-mantle（`responses_backend = "mantle"`）

通过 AWS Bedrock 的 mantle OpenAI 兼容上游提供服务的模型使用与标准 Converse 路径不同的后端。三个 `[model.params]` 字段控制此行为：

| 字段                | 类型              | 含义                                                                                           |
| ------------------- | ----------------- | ---------------------------------------------------------------------------------------------- |
| `responses_backend` | `"mantle"`        | 将该模型路由到 `MantleResponsesProvider` 而非 `BedrockResponsesProvider`                       |
| `chat_backend`      | `"responses"`     | 通过无状态 `ResponsesChatProvider` 协议适配器提供 Chat Completions                            |
| `available_regions` | 字符串数组        | 区域允许列表；缺失 = 全区域可用；非空 = 请求区域不在列表时返回 400                            |

示例（来自 `config/models.toml`）：

```toml
[[model]]
match = "openai.gpt-5.5"
capabilities = []
[model.params]
responses_backend = "mantle"
chat_backend = "responses"
available_regions = ["us-east-2"]
```

**启动行为：** 如果模型声明了 `responses_backend = "mantle"`，但未设置 `bedrock_api_key`（环境变量 `AWS_BEARER_TOKEN_BEDROCK` / `BEDROCK_API_KEY`），网关会记录 WARN 并且只禁用 GPT-5.x mantle 模型 —— 不会启动失败，其他模型会继续工作。请求已禁用的 mantle 模型会返回清晰的 400。`DISABLE_MANTLE=true` 会以同样方式显式禁用 mantle 后端。（这取代了早先的硬失败行为。）运行实例的 `AWS_REGION` 与模型 `available_regions` 不匹配时，启动时发出 WARN 但不硬失败（每请求门控返回 400）。

**Mantle 端点模板：** 上游 URL 由 `MANTLE_BASE_URL_TEMPLATE` 控制（默认 `https://bedrock-mantle.{region}.api.aws/openai/v1`）。字面占位符 `{region}` 在调用时替换为网关的 `AWS_REGION`。修改此环境变量可指向私有或测试 mantle 端点，无需重新编译。

**mantle 模型的限制：**
- `/responses` 保持字节级原始 SSE 透传，不追加 `[DONE]`，以 mantle 的 `response.completed` 结束。
- `/chat/completions` 通过 `chat_backend = "responses"` 支持；适配器不会调用 mantle 不支持的原生 Chat 路由。
- reasoning summary 映射为内联 `<think>`，不暴露原始思考链；reasoning 与工具续轮需要共享 `CHAT_REASONING_CAPSULE_*` keyring，并使用带认证的 `rsc_v1` 工具 ID。
- `/completions` 仍不支持。
- 在 `GET /models` 中以 `gpt-5.4`、`gpt-5.5`、`gpt-5.6-sol/terra/luna` 等裸别名列出，由配置注入。
- 鉴权复用网关用于 Converse 调用的同一 `bedrock_api_key` bearer（`AWS_BEARER_TOKEN_BEDROCK`）。

#### gpt-oss chat 通过 bedrock-mantle（`chat_backend = "mantle"`）

gpt-oss 系列（`gpt-oss-120b` / `gpt-oss-20b`）在 **chat** 接口上通过 mantle 上游提供服务，使用与 `responses_backend` **独立**的字段：

| 字段            | 类型              | 含义                                                                                     |
| --------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `chat_backend`  | `"mantle"`        | 将该模型的 `/chat/completions` 路由到 `MantleChatProvider` 而非 Converse 路径             |
| `available_regions` | 字符串数组     | 区域允许列表；缺失 = 全区域；非空 = 请求区域不在列表时返回 400                          |

`chat_backend` 与 `responses_backend` 互不干扰 —— 一个模型可以在一个接口走 mantle、另一个接口走 Converse。gpt-oss 只设 `chat_backend = "mantle"`（本次仅 chat）。

**chat 端点模板：** chat 上游 URL 由 `MANTLE_CHAT_BASE_URL_TEMPLATE` 控制（默认 `https://bedrock-mantle.{region}.api.aws/v1` —— 注意**无** `/openai` 前缀，与 responses 的 `MANTLE_BASE_URL_TEMPLATE` 不同）。mantle chat 路由（`/chat/completions`）与 responses 路由（`/openai/v1/responses`）不同，这正是两个模板分离的原因。`{region}` 在调用时替换为 `AWS_REGION`。

**gpt-oss chat 模型的限制：**
- 仅支持 `/chat/completions` — `/completions` 返回清晰的 400。
- 原始字节透传（流式 + 非流式）—— 无 Converse 翻译、不重算 usage。非标准 `delta.reasoning` 与每 chunk 的 `obfuscation` 原样保留。
- 网关的 raw chat SSE 通道在流尾**追加** `data: [DONE]\n\n`（mantle chat 不发送，OpenAI chat 客户端期望它）。这是与 responses raw 通道的**唯一**行为差异（后者不追加 `[DONE]`）。
- 在 `GET /models` 中以裸别名（`gpt-oss-120b` / `gpt-oss-20b`）列出，由 `[[alias]]` 配置注入（`mantle_alias_names` 也匹配 `chat_backend == "mantle"`）。
- 启动：若已配置但未设置 `bedrock_api_key`（或 `DISABLE_MANTLE=true`），仅禁用 gpt-oss chat 模型并记录 WARN；网关仍正常启动，其他模型不受影响。被禁用的模型返回清晰的 400。

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

# Git 钩子 —— 克隆后执行一次，启用推送前质量门禁
make hooks                                              # git config core.hooksPath .githooks

# Docker（本地）
docker build -t bedrock-gateway-rust .                  # 从根 Dockerfile 构建 distroless 镜像

# 本地运行（健康检查无需真实 AWS 凭证）
API_KEY=testkey cargo run
curl http://localhost:8080/api/v1/health
```

**推送前门禁（推荐 —— 仅拦截 `git push`，绝不拦截 `git commit`）：**

克隆后执行一次 `make hooks`，将 `core.hooksPath` 指向版本化的 `.githooks/` 目录，即可启用 `.githooks/pre-push`。该钩子运行与 CI 完全一致的三项检查（`cargo fmt --all -- --check` → `cargo clippy --all-targets --all-features -- -D warnings` → `cargo test --all-features`），任一失败即以中文提示中止推送。`git commit` 有意不拦截，可自由提交 WIP；只有 `git push` 受门禁约束。因此「通过 pre-push」即可推断「通过 CI」。

> `core.hooksPath` 是本地 git 设置，克隆时**不会**自动应用 —— 每位贡献者需手动执行一次 `make hooks`（或 `make setup-hooks`）。

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
| Responses `function_call_arguments.delta`  | N/A                                          | 发送有序 `delta` / `done` 事件；`.done` 含函数名，并先于 `response.output_item.done` |
| Responses `namespace` / `custom` / 客户端工具 | N/A                                       | 支持 —— 可逆的 `custom` 字符串输入；namespace 名称仅在 Bedrock 侧扁平化，响应恢复 `namespace` + 裸 `name`；`local_shell` / `shell` / `apply_patch` 保留 Responses 项类型 |
| Responses 内置服务端工具                   | N/A                                          | 静默丢弃（`web_search` / `file_search` / `code_interpreter` / `tool_search` / `mcp` / `computer` / `image_generation` 及任何未知类型）—— 绝不返回 400，捆绑内置工具的 codex 会话得以存活；`ResponsesTool` 带 `#[serde(other)] Unknown` 兜底 |
| GPT-5.x（`gpt-5.4` / `gpt-5.5` / `gpt-5.6-*`）模型 | N/A | 通过 AWS bedrock-mantle Responses 提供（`responses_backend = "mantle"`）。`/responses` 原始透传；`/chat/completions` 使用无状态适配器（`chat_backend = "responses"`），支持摘要渲染、reasoning token 映射和带认证的 `rsc_v1` 工具续轮。`/completions` 仍不支持。别名和区域门控均在 `config/models.toml`。 |
| 传统文本补全（`POST /completions`）        | N/A（接口不存在）                            | 为 Zed edit-prediction 新增。复用 Chat/Converse 路径（prompt 包装为单条 user 消息），协议形状 `text_completion`。`suffix` → 400（无 Bedrock 映射）；token 数组形式的 `prompt` → 400（需发送字符串）；`logprobs` / `best_of` / `logit_bias` 接受但忽略。 |
| gpt-oss chat 通过 mantle（`gpt-oss-120b` / `gpt-oss-20b`） | N/A                          | 通过 AWS bedrock-mantle 在 CHAT 接口提供（`chat_backend = "mantle"`，与 `responses_backend` 独立的字段），**仅支持 `/chat/completions`** —— `/completions` 返回 400。字节级原始 SSE 透传；无 Converse 翻译、不重算 usage。网关在流尾**追加** `data: [DONE]`（与 responses raw 通道的唯一差异，后者不追加）。非标准 `delta.reasoning` + 每 chunk `obfuscation` 原样保留（不映射 `<think>`、不丢弃）。chat 基础模板 `MANTLE_CHAT_BASE_URL_TEMPLATE` = `https://bedrock-mantle.{region}.api.aws/v1`（无 `/openai` 前缀）。在 `GET /models` 中以裸别名（`gpt-oss-120b` / `gpt-oss-20b`）列出。区域门控：`us-east-1` / `us-east-2` / `us-west-2`。 |
 
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
