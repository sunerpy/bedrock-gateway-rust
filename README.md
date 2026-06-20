# bedrock-gateway-rust

[![CI](https://github.com/sunerpy/bedrock-gateway-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/sunerpy/bedrock-gateway-rust/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/sunerpy/bedrock-gateway-rust)](https://hub.docker.com/r/sunerpy/bedrock-gateway-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A 100% Rust, OpenAI-compatible API gateway for AWS Bedrock — drop-in, single-binary, blazingly fast.**

> 中文文档: [docs/readme/README_CN.md](docs/readme/README_CN.md)

Point any OpenAI SDK, tool, or agent at this gateway and it routes requests to AWS Bedrock without a single line of client-side changes. The runtime is **axum + tokio + aws-sdk-bedrockruntime** — fully async, no GC pauses, no Python dependency. It replaces an earlier Python/FastAPI implementation while preserving wire-exact OpenAI API compatibility.

---

## Highlights

- **100% Rust** — single statically-linked binary, distroless Docker image (~12 MB compressed), no GC, low memory footprint, high concurrency under load.
- **Wire-exact OpenAI compatibility** — existing OpenAI SDKs, clients, and agents connect with zero code changes. No invented top-level fields; Bedrock-only features go through the standard `extra_body` mechanism.
- **OpenAI Responses API** — full support for `POST /api/v1/responses`, including streaming. Required by `codex` (`wire_api = "responses"`). Stateless; `store` and `previous_response_id` are accepted and silently ignored.
- **Automatic prompt caching** — cache-point injection is on by default. The gateway places cache points across tools, system prompt, and messages automatically (up to `max_cache_checkpoints` per model). No client changes needed. Thresholds and limits are config-driven per model; a family fallback entry covers new Claude models automatically.
- **Extended thinking / reasoning** — Claude `budget_tokens`, `adaptive_thinking`, and DeepSeek string-form reasoning all supported. Map OpenAI's `reasoning_effort` levels (`low` / `medium` / `high` / `xhigh` / `max`) through `extra_body`; the gateway picks the right Bedrock wire format per model.
- **Cross-region inference profiles** — all seven geographic prefixes (`us.` / `eu.` / `apac.` / `jp.` / `au.` / `ca.` / `global.`) work transparently. Capability matching strips the prefix; Bedrock calls always use the original model ID.
- **Zero-hardcoding** — all model knowledge lives in `config/models.toml`. Adding a new model or tuning cache thresholds never requires a recompile.
- **Four deployment targets** — standalone binary, Docker, ECS/Fargate + ALB (CloudFormation one-click), and Lambda + Function URL (Lambda Web Adapter, no Lambda-specific Rust code).
- **Dual auth, dual direction** — client-to-gateway via Bearer token (SSM / Secrets Manager / env); gateway-to-Bedrock via Bedrock API Key bearer or SigV4 credential chain.
- **Structured observability** — per-request `request_id`, `cached_tokens`, `cache_hit`, `ttfb_ms`, `duration_ms` in structured JSON logs. No prompt text or secrets ever logged.

---

## Supported Endpoints

All endpoints are prefixed by `API_ROUTE_PREFIX` (default `/api/v1`).

| Method | Path                       | Description                                                   |
| ------ | -------------------------- | ------------------------------------------------------------- |
| `POST` | `/api/v1/chat/completions` | Chat completions — streaming (SSE) and non-streaming          |
| `POST` | `/api/v1/responses`        | OpenAI Responses API — stateless, streaming and non-streaming |
| `POST` | `/api/v1/embeddings`       | Embeddings — Cohere, Titan, and Nova families                 |
| `GET`  | `/api/v1/models`           | Live model catalog from Bedrock control plane                 |
| `GET`  | `/api/v1/models/{id}`      | Single model lookup (supports inference profile IDs)          |
| `GET`  | `/api/v1/health`           | Liveness probe — returns `200 OK`                             |

---

## Client Compatibility

| Client          | Wire API                | Endpoint                        | Status       |
| --------------- | ----------------------- | ------------------------------- | ------------ |
| **opencode**    | OpenAI Chat Completions | `POST /api/v1/chat/completions` | ✅ Supported |
| **hermes**      | OpenAI Chat Completions | `POST /api/v1/chat/completions` | ✅ Supported |
| **codex**       | OpenAI Responses API    | `POST /api/v1/responses`        | ✅ Supported |
| **claude code** | Anthropic Messages      | `POST /v1/messages`             | ⏳ Roadmap   |

---

## Quick Start

### Prerequisites

- AWS credentials available (instance role, `AWS_PROFILE`, or access key pair)
- Bedrock model access enabled in your target region
- Rust 1.80+ (to build from source) or Docker

### 30-second start with Docker

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -e AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-... \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

Using an EC2 instance or ECS task with an IAM role? Omit `AWS_BEARER_TOKEN_BEDROCK` and the access key pair — the SDK picks up instance credentials automatically.

### Build from source

```bash
cargo build --release
API_KEY=sk-my-secret-key AWS_REGION=us-east-1 ./target/release/bedrock-gateway
```

### Local dev

```bash
API_KEY=testkey cargo run
```

### Verify

```bash
curl http://localhost:8080/api/v1/health
# OK

curl http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Hello!"}]}'
```

### OpenAI Python SDK — drop-in replacement

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/api/v1",
    api_key="sk-my-secret-key",
)

response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Explain serverless in one sentence."}],
)
print(response.choices[0].message.content)
```

Streaming, tool calling, vision, and embeddings work the same way — no special configuration.

---

## Deployment

Four targets share the same binary and the same environment-variable interface.

| Target                | Guide                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| Standalone binary     | Run `cargo build --release`, ship `target/release/bedrock-gateway` + `config/`                   |
| Docker (distroless)   | `docker pull sunerpy/bedrock-gateway-rust` — see [docs/deploy/docker.md](docs/deploy/docker.md)  |
| ECS / Fargate + ALB   | CloudFormation one-click — see [docs/deploy/ecs.md](docs/deploy/ecs.md)                          |
| Lambda + Function URL | Lambda Web Adapter, no Lambda-specific code — see [docs/deploy/lambda.md](docs/deploy/lambda.md) |

### ECS / Fargate quick deploy

```bash
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayFargate.template \
  --stack-name bedrock-gateway \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ApiKey=sk-my-secret-key \
    VpcId=vpc-... \
    SubnetIds=subnet-...,subnet-...
```

### Lambda quick deploy

```bash
# Build the Lambda container image
docker build -f deployment/lambda/Dockerfile -t bedrock-gateway:lambda .
# Push to ECR, then:
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayLambda.template \
  --stack-name bedrock-gateway-lambda \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ImageUri=<ECR_URI>:latest \
    ApiKeySecretArn=arn:aws:secretsmanager:...
```

> Lambda has a 10-minute maximum timeout. For long-running streaming sessions, prefer ECS/Fargate.

---

## Configuration

All settings are environment variables. The gateway accepts both an `APP_` prefix and bare names (e.g., `API_KEY` or `APP_API_KEY`); bare names take precedence.

### The three required variables

| Variable                   | Purpose                                                                           |
| -------------------------- | --------------------------------------------------------------------------------- |
| `API_KEY`                  | Bearer token your clients send to the gateway. Pick any string.                   |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API Key the gateway presents to AWS (recommended). Or omit and use SigV4. |
| `AWS_REGION`               | AWS region for Bedrock calls. Defaults to `us-west-2`.                            |

**These two keys go in opposite directions — don't mix them up:**

- `API_KEY` is what your clients send _to_ the gateway.
- `AWS_BEARER_TOKEN_BEDROCK` is what the gateway sends _to_ AWS Bedrock.

### AWS credential options

The gateway auto-selects — no switch needed:

1. `AWS_BEARER_TOKEN_BEDROCK` (alias `BEDROCK_API_KEY`) — Bedrock API Key, sent as `Authorization: Bearer`. Bypasses SigV4 entirely. Recommended for new deployments.
2. SigV4 credential chain — resolves `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, then `AWS_PROFILE`, then IMDS / ECS task role.

### Production secret management

Prefer fetching `API_KEY` from a secrets store. Priority order:

1. `API_KEY_PARAM_NAME` — SSM Parameter Store
2. `API_KEY_SECRET_ARN` — Secrets Manager (secret must contain an `api_key` field)
3. `API_KEY` — plaintext env var

### All optional variables

<details>
<summary>Show all optional variables</summary>

**Model defaults**

| Variable                  | Default                                     | Description                                  |
| ------------------------- | ------------------------------------------- | -------------------------------------------- |
| `DEFAULT_MODEL`           | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Model used when the client omits `model`     |
| `DEFAULT_EMBEDDING_MODEL` | `cohere.embed-multilingual-v3`              | Embedding model used when `model` is omitted |

**Feature flags**

| Variable                                | Default  | Description                                                  |
| --------------------------------------- | -------- | ------------------------------------------------------------ |
| `ENABLE_CROSS_REGION_INFERENCE`         | `true`   | Transparently route cross-region inference profiles          |
| `ENABLE_APPLICATION_INFERENCE_PROFILES` | `true`   | Surface inference profile metadata in `GET /models`          |
| `ENABLE_PROMPT_CACHING`                 | `true`   | Auto-inject cache points for Claude and Nova models          |
| `CONFIG_DIR`                            | `config` | External config directory path (overrides embedded defaults) |

**Server**

| Variable           | Default   | Description                   |
| ------------------ | --------- | ----------------------------- |
| `PORT`             | `8080`    | HTTP listen port              |
| `BIND_ADDR`        | `0.0.0.0` | Network interface to bind     |
| `API_ROUTE_PREFIX` | `/api/v1` | Path prefix for all endpoints |

**Logging**

| Variable    | Default | Description                                              |
| ----------- | ------- | -------------------------------------------------------- |
| `LOG_LEVEL` | `info`  | Verbosity: `trace` / `debug` / `info` / `warn` / `error` |
| `DEBUG`     | `false` | Enable verbose error details in responses                |

**Timeouts and retries**

| Variable                   | Default | Description                                               |
| -------------------------- | ------- | --------------------------------------------------------- |
| `AWS_CONNECT_TIMEOUT_SECS` | `60`    | TCP connection timeout to AWS                             |
| `AWS_READ_TIMEOUT_SECS`    | `900`   | Response read timeout (15 min, accommodates long streams) |
| `AWS_MAX_RETRY_ATTEMPTS`   | `8`     | Retries on transient throttling or 5xx failures           |

</details>

### Config files (no recompile needed)

Config files live in `config/` alongside the binary and are read at startup. They are also embedded in the binary as a fallback — the gateway always starts, even if the files are absent.

| File                     | Purpose                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| `config/models.toml`     | Model capability registry: capability flags, reasoning paths, cache thresholds |
| `config/regions.toml`    | Cross-region routing rules                                                     |
| `config/embeddings.toml` | Embedding model registry (Cohere / Titan / Nova families)                      |
| `config/app.toml`        | Application defaults — lowest priority                                         |

**Adding a new model** requires only a `config/models.toml` entry:

```toml
[[model]]
match = "your-provider.your-model-id"
capabilities = []
[model.params]
max_tokens = 8192
context_window = 200000
# reasoning_path = "budget_tokens"  # uncomment if the model supports extended thinking
```

---

## API Usage

All endpoints require `Authorization: Bearer <API_KEY>`.

Base URL: `http://localhost:8080/api/v1`

### Streaming chat

```bash
curl http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Count to 5 slowly."}],"stream":true}'
```

### Embeddings

```bash
curl http://localhost:8080/api/v1/embeddings \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"cohere.embed-multilingual-v3","input":["Hello world","Bedrock gateway"]}'
```

### Extended thinking

```python
response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Solve step by step: ..."}],
    extra_body={"reasoning_effort": "high"},  # none|minimal|low|medium|high|xhigh|max
)
# Reasoning appears inline as <think>...</think> inside content
```

### codex configuration

Add to `~/.codex/config.toml`:

```toml
[model_providers.bgw]
name = "Bedrock Gateway"
base_url = "http://localhost:8080/api/v1"
wire_api = "responses"
requires_openai_auth = false

[model_providers.bgw.env_key]
name = "BGW_API_KEY"
description = "Bearer token for the Bedrock gateway"
```

```bash
export BGW_API_KEY=sk-my-gateway-key
codex --provider bgw --model anthropic.claude-3-5-sonnet-20241022-v2:0
```

---

## Architecture

```
src/
├── main.rs / lib.rs         # Entry point, crate root
├── error.rs                 # AppError, OpenAI error envelope
├── telemetry.rs             # tracing subscriber, dynamic log level
├── openai/                  # Wire types: ChatRequest, ChatResponse, ResponsesRequest, ...
├── domain/                  # Provider-agnostic traits: ChatProvider, ResponsesProvider, ...
├── config/                  # Settings, ModelCapabilityConfig, RegionRoutingConfig, ...
├── bedrock/                 # AWS Bedrock backend: translate, cache, reasoning, stream, ...
└── server/                  # axum router, auth middleware, AppState
```

The codebase is strictly layered — dependencies flow downward only. The Bedrock backend implements the domain traits; adding a non-Bedrock backend means implementing those traits without touching any existing code.

For the full architecture reference, contract rules, and contributor guidelines, see [AGENTS.md](AGENTS.md).

For caching behavior, reasoning budget paths, and cross-region inference profile details, see [docs/caching-and-reasoning.md](docs/caching-and-reasoning.md).

---

## Supported Models

The authoritative list is `config/models.toml` and the live `GET /api/v1/models` endpoint. The registry currently covers:

- **Claude** — Sonnet 4.x, Haiku 4.x, Opus 4.x (via Bedrock model IDs and cross-region inference profiles)
- **Amazon Nova** — multimodal and text models
- **DeepSeek** — v3 (string-form reasoning path)
- Any Bedrock foundation model or inference profile accessible in your account — the catalog refreshes from the control plane at startup

Adding a model requires only a `config/models.toml` entry and no recompile.

---

## Build, Test, and Contributing

```bash
cargo build --release          # release binary → target/release/bedrock-gateway
cargo test                     # unit + golden replay tests (no AWS credentials needed)
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
```

**Pre-commit gate (mandatory before every commit):**

```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

Golden fixtures in `tests/golden/` are pinned offline and run in CI without AWS credentials. Live integration tests require real credentials:

```bash
BEDROCK_INTEGRATION=1 AWS_PROFILE=us cargo test -- --ignored
```

See [docs/readme/CONTRIBUTING.md](docs/readme/CONTRIBUTING.md) for contributor guidelines.

---

## Links

| Resource                   | Link                                                             |
| -------------------------- | ---------------------------------------------------------------- |
| 中文文档                   | [docs/readme/README_CN.md](docs/readme/README_CN.md)             |
| Contributing               | [docs/readme/CONTRIBUTING.md](docs/readme/CONTRIBUTING.md)       |
| Code of Conduct            | [docs/readme/CODE_OF_CONDUCT.md](docs/readme/CODE_OF_CONDUCT.md) |
| Caching and reasoning      | [docs/caching-and-reasoning.md](docs/caching-and-reasoning.md)   |
| Docker deployment          | [docs/deploy/docker.md](docs/deploy/docker.md)                   |
| ECS/Fargate deployment     | [docs/deploy/ecs.md](docs/deploy/ecs.md)                         |
| Lambda deployment          | [docs/deploy/lambda.md](docs/deploy/lambda.md)                   |
| Architecture and contracts | [AGENTS.md](AGENTS.md)                                           |

---

## License

[MIT](LICENSE)

---

## Acknowledgements

This project is inspired by the Python gateway concept in
[aws-samples/bedrock-access-gateway](https://github.com/aws-samples/bedrock-access-gateway).
The Rust implementation is a full rewrite with wire-exact OpenAI compatibility, automatic
prompt caching, Responses API support, and production-grade observability.
