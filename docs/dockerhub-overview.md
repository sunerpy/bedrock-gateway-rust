# bedrock-gateway-rust

**A 100% Rust, OpenAI-compatible API gateway for AWS Bedrock — drop-in, single-binary, blazingly fast.**

[![GitHub](https://img.shields.io/badge/GitHub-sunerpy%2Fbedrock--gateway--rust-blue?logo=github)](https://github.com/sunerpy/bedrock-gateway-rust)
[![crates.io](https://img.shields.io/crates/v/bedrock-gateway-rust.svg?logo=rust)](https://crates.io/crates/bedrock-gateway-rust)
[![License: MIT-0](https://img.shields.io/badge/license-MIT--0-green.svg)](https://github.com/sunerpy/bedrock-gateway-rust/blob/main/LICENSE)

> This file is the canonical source for the Docker Hub repository overview
> (`sunerpy/bedrock-gateway-rust`). Keep it in sync with the repo's README and
> paste it into Docker Hub → repository settings → Description.

Point any OpenAI SDK or client at AWS Bedrock with **zero code changes**. Built on `axum + tokio + aws-sdk-bedrockruntime` — no Python, no GC, low memory, high concurrency.

## Highlights

- **100% Rust** — single static binary, distroless image, multi-arch (amd64 + arm64)
- **Wire-exact OpenAI compatibility** — existing OpenAI SDKs work unchanged
- **OpenAI Responses API** — the newest surface, codex-compatible (stateless)
- **Automatic prompt caching** — config-driven thresholds, family fallback, runtime safety net
- **Extended thinking / reasoning** — Claude adaptive thinking + `reasoning_effort` mapping
- **Cross-region inference profiles** — `us.` / `eu.` / `apac.` / `jp.` / `au.` / `ca.` / `global.`
- **Zero-hardcoding** — all model knowledge in `config/*.toml`, add models without recompiling
- **Multiple deploy targets** — single binary, Docker, AWS ECS/Fargate + ALB, AWS Lambda
- **Structured observability** — JSON logs with `request_id`, `cached_tokens`, `cache_hit`, `ttfb_ms`

## Quick Start

```bash
docker run -d -p 8080:8080 \
  -e API_KEY=your-gateway-key \
  -e AWS_REGION=us-east-2 \
  -e AWS_BEARER_TOKEN_BEDROCK=your-bedrock-api-key \
  sunerpy/bedrock-gateway-rust:latest

# health check
curl http://localhost:8080/api/v1/health

# OpenAI-compatible call
curl http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"us.anthropic.claude-sonnet-4-5-20250929-v1:0","messages":[{"role":"user","content":"hi"}]}'
```

## Supported Endpoints

| Endpoint                                        | Notes                               |
| ----------------------------------------------- | ----------------------------------- |
| `POST /api/v1/chat/completions`                 | Streaming (SSE) + non-streaming     |
| `POST /api/v1/responses`                        | OpenAI Responses API (codex)        |
| `POST /api/v1/embeddings`                       | Cohere / Titan / Nova               |
| `GET /api/v1/models`, `GET /api/v1/models/{id}` | Catalog incl. cross-region profiles |
| `GET /api/v1/health`                            | Liveness probe                      |

## Tags

- `latest` — most recent release
- `0.1.0` — pinned version (amd64 + arm64)

## Configuration

| Variable                   | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| `API_KEY`                  | Bearer token clients present to the gateway         |
| `AWS_REGION`               | Bedrock region (e.g. `us-east-2`)                   |
| `AWS_BEARER_TOKEN_BEDROCK` | Bedrock API key (or use the SigV4 credential chain) |
| `DEFAULT_MODEL`            | Fallback model id                                   |
| `ENABLE_PROMPT_CACHING`    | Auto prompt-caching (default `true`)                |

Config is embedded in the binary (`include_str!`); mount `CONFIG_DIR` to override.

## Documentation & Source

**[github.com/sunerpy/bedrock-gateway-rust](https://github.com/sunerpy/bedrock-gateway-rust)**

License: MIT-0
