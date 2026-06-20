# bedrock-gateway-rust

[![CI](https://github.com/sunerpy/bedrock-gateway-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/sunerpy/bedrock-gateway-rust/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/sunerpy/bedrock-gateway-rust)](https://hub.docker.com/r/sunerpy/bedrock-gateway-rust)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

**100% Rust 实现的 OpenAI 兼容 API 网关，后端对接 AWS Bedrock。单二进制、零 GC、高并发。**

> 📖 [English Documentation / 英文文档](../../README.md)

将本网关部署在 AWS Bedrock 前，任何 OpenAI SDK、工具或 Agent 无需修改客户端代码即可直接使用。运行时栈为 **axum + tokio + aws-sdk-bedrockruntime** — 全异步，无 GC 暂停，无 Python 依赖。替代了早期的 Python/FastAPI 实现，同时保持与 OpenAI REST API 的字节级兼容。

---

## 项目亮点

- **100% Rust** — 单静态链接二进制，distroless Docker 镜像（~12 MB 压缩），无 GC，低内存，高并发。
- **字节级 OpenAI 兼容** — 现有 OpenAI SDK、客户端和 Agent 零改造接入。不引入自定义顶层字段；Bedrock 专属特性走标准 `extra_body` 机制。
- **OpenAI Responses API** — 完整支持 `POST /api/v1/responses`，含流式输出。`codex` 必需（`wire_api = "responses"`）。无状态；`store` 和 `previous_response_id` 接受但静默忽略。
- **自动 Prompt 缓存** — 缓存点注入默认开启。网关自动在 tools、system prompt、messages 区注入缓存点（每模型最多 `max_cache_checkpoints` 个），客户端无感知。阈值按模型配置；兜底条目自动覆盖新 Claude 模型。
- **扩展思考 / Reasoning** — 支持 Claude `budget_tokens`、`adaptive_thinking` 及 DeepSeek 字符串形式推理。通过 `extra_body` 映射 OpenAI `reasoning_effort` 级别；网关按模型自动选择正确的 Bedrock wire 格式。
- **跨区域 Inference Profile** — 7 个地理前缀（`us.` / `eu.` / `apac.` / `jp.` / `au.` / `ca.` / `global.`）全部透明支持。能力匹配去掉前缀；发往 Bedrock 的调用始终使用原始模型 ID。
- **零硬编码** — 所有模型知识在 `config/models.toml`。新增模型或调整缓存阈值均无需重新编译。
- **四种部署方式** — 单二进制、Docker、ECS/Fargate + ALB（CloudFormation 一键部署）、Lambda + Function URL（Lambda Web Adapter，无需 Lambda 专属代码）。
- **双向鉴权** — 客户端→网关使用 Bearer Token（SSM / Secrets Manager / 环境变量）；网关→Bedrock 使用 Bedrock API Key bearer 或 SigV4 凭证链。
- **结构化可观测性** — 每请求 `request_id`、`cached_tokens`、`cache_hit`、`ttfb_ms`、`duration_ms` 结构化 JSON 日志。任何级别均不打印 prompt 内容或密钥。

---

## 支持的端点

所有端点以 `API_ROUTE_PREFIX` 为前缀（默认 `/api/v1`）。

| 方法   | 路径                       | 说明                                        |
| ------ | -------------------------- | ------------------------------------------- |
| `POST` | `/api/v1/chat/completions` | 聊天补全 — 流式（SSE）和非流式              |
| `POST` | `/api/v1/responses`        | OpenAI Responses API — 无状态，流式和非流式 |
| `POST` | `/api/v1/embeddings`       | 嵌入向量 — Cohere、Titan 和 Nova 系列       |
| `GET`  | `/api/v1/models`           | 从 Bedrock 控制面实时获取模型目录           |
| `GET`  | `/api/v1/models/{id}`      | 单个模型查询（支持 inference profile ID）   |
| `GET`  | `/api/v1/health`           | 存活探针 — 返回 `200 OK`                    |

---

## 客户端兼容矩阵

| 客户端          | Wire API                | 端点                            | 状态      |
| --------------- | ----------------------- | ------------------------------- | --------- |
| **opencode**    | OpenAI Chat Completions | `POST /api/v1/chat/completions` | ✅ 已支持 |
| **hermes**      | OpenAI Chat Completions | `POST /api/v1/chat/completions` | ✅ 已支持 |
| **codex**       | OpenAI Responses API    | `POST /api/v1/responses`        | ✅ 已支持 |
| **claude code** | Anthropic Messages      | `POST /v1/messages`             | ⏳ 规划中 |

---

## 快速开始

### 前置条件

- AWS 凭证（实例角色、`AWS_PROFILE`，或访问密钥对）
- 目标区域已开启 Bedrock 模型访问权限
- Rust 1.80+（源码构建）或 Docker

### 30 秒 Docker 启动

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -e AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-... \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

在带 IAM 角色的 EC2 实例或 ECS 任务上运行？省略 `AWS_BEARER_TOKEN_BEDROCK` 和访问密钥对 — SDK 会自动获取实例凭证。

### 源码构建

```bash
cargo build --release
API_KEY=sk-my-secret-key AWS_REGION=us-east-1 ./target/release/bedrock-gateway
```

### 本地开发

```bash
API_KEY=testkey cargo run
```

### 验证服务

```bash
curl http://localhost:8080/api/v1/health
# OK

curl http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"你好！"}]}'
```

### OpenAI Python SDK — 即插即用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/api/v1",
    api_key="sk-my-secret-key",
)

response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "用一句话解释什么是无服务器架构。"}],
)
print(response.choices[0].message.content)
```

流式输出、工具调用、视觉和 Embedding 同理 — 无需额外配置。

---

## 部署方式

四种部署方式共享同一个二进制文件和同一套环境变量接口。

| 方式                  | 文档                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------- |
| 独立二进制            | `cargo build --release`，部署 `target/release/bedrock-gateway` + `config/`            |
| Docker（distroless）  | 拉取 `sunerpy/bedrock-gateway-rust` — [部署 → Docker](../../docs/deploy/docker.md)    |
| ECS / Fargate + ALB   | CloudFormation 一键部署 — [部署 → ECS / Fargate](../../docs/deploy/ecs.md)            |
| Lambda + Function URL | Lambda Web Adapter，无 Lambda 专属代码 — [部署 → Lambda](../../docs/deploy/lambda.md) |

### ECS / Fargate 快速部署

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

### Lambda 快速部署

```bash
# 构建 Lambda 容器镜像
docker build -f deployment/lambda/Dockerfile -t bedrock-gateway:lambda .
# 推送到 ECR，然后：
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayLambda.template \
  --stack-name bedrock-gateway-lambda \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ImageUri=<ECR_URI>:latest \
    ApiKeySecretArn=arn:aws:secretsmanager:...
```

> Lambda 最长超时时间为 10 分钟。对于长时间流式会话，推荐使用 ECS/Fargate。

---

## 配置参考

所有配置均通过环境变量传入。网关同时支持 `APP_` 前缀和裸变量名；两者同时存在时裸变量名优先。

### 三个核心变量

| 变量                       | 说明                                                          |
| -------------------------- | ------------------------------------------------------------- |
| `API_KEY`                  | 客户端发给网关的 Bearer Token，随意选一个字符串               |
| `AWS_BEARER_TOKEN_BEDROCK` | 网关向 AWS Bedrock 出示的 API Key（推荐）。或省略，改用 SigV4 |
| `AWS_REGION`               | Bedrock 调用所在区域，默认 `us-west-2`                        |

**这两个密钥流向相反，切勿混淆：**

- `API_KEY` 是客户端发*给*网关的。
- `AWS_BEARER_TOKEN_BEDROCK` 是网关发*给* AWS Bedrock 的。

### AWS 凭证选项

网关自动选择，无需配置切换标志：

1. `AWS_BEARER_TOKEN_BEDROCK`（别名 `BEDROCK_API_KEY`）— Bedrock API Key，以 `Authorization: Bearer` 发给 AWS，完全绕过 SigV4。新部署推荐使用。
2. SigV4 凭证链 — 依次解析 `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`，然后 `AWS_PROFILE`，再是 IMDS / ECS 任务角色。

### 生产环境密钥管理

推荐从密钥管理服务获取 `API_KEY`，避免明文出现在容器环境变量中。优先级顺序：

1. `API_KEY_PARAM_NAME` — SSM Parameter Store
2. `API_KEY_SECRET_ARN` — Secrets Manager（密钥须包含 `api_key` 字段）
3. `API_KEY` — 明文环境变量

### 全部可选变量

<details>
<summary>展开查看所有可选变量</summary>

**模型默认值**

| 变量                      | 默认值                                      | 说明                                 |
| ------------------------- | ------------------------------------------- | ------------------------------------ |
| `DEFAULT_MODEL`           | `anthropic.claude-3-5-sonnet-20241022-v2:0` | 客户端未指定 `model` 时的默认模型    |
| `DEFAULT_EMBEDDING_MODEL` | `cohere.embed-multilingual-v3`              | 省略 `model` 时的默认 Embedding 模型 |

**功能开关**

| 变量                                    | 默认值   | 说明                                      |
| --------------------------------------- | -------- | ----------------------------------------- |
| `ENABLE_CROSS_REGION_INFERENCE`         | `true`   | 透明支持跨区域 inference profile          |
| `ENABLE_APPLICATION_INFERENCE_PROFILES` | `true`   | 在 `GET /models` 中展示推理配置文件元数据 |
| `ENABLE_PROMPT_CACHING`                 | `true`   | 为 Claude 和 Nova 模型自动注入缓存点      |
| `CONFIG_DIR`                            | `config` | 外部 config 目录路径（覆盖嵌入默认值）    |

**服务**

| 变量               | 默认值    | 说明               |
| ------------------ | --------- | ------------------ |
| `PORT`             | `8080`    | HTTP 监听端口      |
| `BIND_ADDR`        | `0.0.0.0` | 网络绑定地址       |
| `API_ROUTE_PREFIX` | `/api/v1` | 所有端点的路径前缀 |

**日志**

| 变量        | 默认值  | 说明                                                    |
| ----------- | ------- | ------------------------------------------------------- |
| `LOG_LEVEL` | `info`  | 日志级别：`trace` / `debug` / `info` / `warn` / `error` |
| `DEBUG`     | `false` | 在响应中启用详细错误信息                                |

**超时与重试**

| 变量                       | 默认值 | 说明                                        |
| -------------------------- | ------ | ------------------------------------------- |
| `AWS_CONNECT_TIMEOUT_SECS` | `60`   | 与 AWS 建立 TCP 连接的超时秒数              |
| `AWS_READ_TIMEOUT_SECS`    | `900`  | 响应读取超时秒数（15 分钟，适应长流式会话） |
| `AWS_MAX_RETRY_ATTEMPTS`   | `8`    | 瞬时限流或 5xx 故障时的最大重试次数         |

</details>

### TOML 配置文件（无需重新编译）

配置文件位于与二进制文件同级的 `config/` 目录下，启动时读取。同时嵌入二进制作为兜底 — 即使文件缺失网关也能正常启动。

| 文件                     | 用途                                               |
| ------------------------ | -------------------------------------------------- |
| `config/models.toml`     | 模型能力注册表：能力标志、推理路径、缓存阈值       |
| `config/regions.toml`    | 跨区域路由规则                                     |
| `config/embeddings.toml` | Embedding 模型注册表（Cohere / Titan / Nova 系列） |
| `config/app.toml`        | 应用默认值，优先级最低                             |

**添加新模型**只需编辑 `config/models.toml`，无需修改代码：

```toml
[[model]]
match = "your-provider.your-model-id"
capabilities = []
[model.params]
max_tokens = 8192
context_window = 200000
# reasoning_path = "budget_tokens"  # 如果模型支持扩展思考，取消注释
```

---

## API 使用示例

所有端点均需要 `Authorization: Bearer <API_KEY>` 请求头。

基础 URL：`http://localhost:8080/api/v1`

### 流式聊天

```bash
curl http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"从 1 数到 5。"}],"stream":true}'
```

### Embedding 嵌入向量

```bash
curl http://localhost:8080/api/v1/embeddings \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"cohere.embed-multilingual-v3","input":["你好世界","Bedrock 网关"]}'
```

### 扩展思考

```python
response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "请一步步推理解决这个问题：..."}],
    extra_body={"reasoning_effort": "high"},  # none|minimal|low|medium|high|xhigh|max
)
# 推理内容以 <think>...</think> 内联在 content 中
```

### codex 配置

在 `~/.codex/config.toml` 中添加：

```toml
[model_providers.bgw]
name = "Bedrock Gateway"
base_url = "http://localhost:8080/api/v1"
wire_api = "responses"
requires_openai_auth = false

[model_providers.bgw.env_key]
name = "BGW_API_KEY"
description = "Bedrock 网关的 Bearer Token"
```

```bash
export BGW_API_KEY=sk-my-gateway-key
codex --provider bgw --model anthropic.claude-3-5-sonnet-20241022-v2:0
```

---

## 架构简述

```
src/
├── main.rs / lib.rs         # 入口点，crate 根
├── error.rs                 # AppError，OpenAI 错误信封
├── telemetry.rs             # tracing subscriber，动态日志级别
├── openai/                  # 协议类型：ChatRequest、ChatResponse、ResponsesRequest...
├── domain/                  # Provider 无关 trait：ChatProvider、ResponsesProvider...
├── config/                  # 配置加载：Settings、ModelCapabilityConfig、RegionRoutingConfig...
├── bedrock/                 # AWS Bedrock 后端：translate、cache、reasoning、stream...
└── server/                  # axum 路由、鉴权中间件、AppState
```

代码库严格分层，依赖关系只向下流动。Bedrock 后端实现 domain trait；添加非 Bedrock 后端只需实现这些 trait，不动任何现有代码。

完整架构参考、契约规则和贡献者指南，见 [架构与契约指南](../../AGENTS.md)。

缓存行为、Reasoning Budget 路径和跨区域 Inference Profile 详情，见 [深入 → 缓存与推理](../../docs/caching-and-reasoning.md)。

---

## 支持的模型

权威列表在 `config/models.toml` 和实时 `GET /api/v1/models` 端点。注册表当前覆盖：

- **Claude** — Sonnet 4.x、Haiku 4.x、Opus 4.x 系列（通过 Bedrock 模型 ID 和跨区域 inference profile）
- **Amazon Nova** — 多模态和文本模型
- **DeepSeek** — v3（字符串形式推理路径）
- 账户中可访问的任何 Bedrock 基础模型或 inference profile — 模型目录在启动时从控制面刷新

添加新模型只需一条 `config/models.toml` 条目，无需重新编译。

---

## 构建、测试与贡献

```bash
cargo build --release          # release 二进制 → target/release/bedrock-gateway
cargo test                     # 单元测试 + golden replay（无需 AWS 凭证）
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt
```

**提交前检查（每次提交前必须按序执行）：**

```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

`tests/golden/` 中的 golden fixtures 为离线固定测试，在 CI 中无需 AWS 凭证即可运行。实时集成测试需要真实凭证：

```bash
BEDROCK_INTEGRATION=1 AWS_PROFILE=us cargo test -- --ignored
```

贡献规范见 [贡献指南](CONTRIBUTING.md)。

---

## 相关链接

| 资源             | 链接                                                     |
| ---------------- | -------------------------------------------------------- |
| English README   | [Read in English](../../README.md)                       |
| 贡献指南         | [贡献指南](CONTRIBUTING.md)                              |
| 行为准则         | [行为准则](CODE_OF_CONDUCT.md)                           |
| 缓存与推理详解   | [深入 → 缓存与推理](../../docs/caching-and-reasoning.md) |
| Docker 部署      | [部署 → Docker](../../docs/deploy/docker.md)             |
| ECS/Fargate 部署 | [部署 → ECS / Fargate](../../docs/deploy/ecs.md)         |
| Lambda 部署      | [部署 → Lambda](../../docs/deploy/lambda.md)             |
| 架构与契约       | [架构与契约指南](../../AGENTS.md)                        |

---

## 许可证

[MIT](../../LICENSE)

---

## 致谢

本项目的灵感来源于 [aws-samples/bedrock-access-gateway](https://github.com/aws-samples/bedrock-access-gateway) 的 Python 网关思路。Rust 实现是从零开始的完整重写，具备字节级 OpenAI 兼容性、自动 Prompt 缓存、Responses API 支持和生产级可观测性。
