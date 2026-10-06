# Codex CLI

Codex CLI 使用 Responses API，网关也提供这个接口，因此 Codex 可以把网关作为自定义 model provider 接入。你的账号能调用的每个模型都可以使用，包括 Claude 和 Nova。

## 添加 provider

把 provider 写入 `~/.codex/config.toml`：

```toml
model = "global.anthropic.claude-sonnet-5-5"
model_provider = "bgw"

[model_providers.bgw]
name = "bedrock-gateway"
base_url = "http://localhost:8080/api/v1"
env_key = "BGW_API_KEY"
wire_api = "responses"
```

在启动 Codex 的 shell 中导出网关的密钥，然后照常启动 Codex。`-m` 为单次运行指定另一个模型：

```sh
export BGW_API_KEY="<the gateway's API_KEY>"
codex
codex exec -m us.amazon.nova-2-lite-v1:0 "Reply with exactly: BEDROCK_OK"
```

最后一条命令会打印 `BEDROCK_OK`。

## 各项设置的作用

| 设置             | 原因                                                                |
| ---------------- | ------------------------------------------------------------------- |
| `model_provider` | 把 Codex 的请求发给下面定义的 provider，而不是发给 OpenAI。         |
| `base_url`       | 网关的基础 URL。Codex 会在后面加上 `/responses`。                   |
| `env_key`        | 保存密钥的环境变量。Codex 以 `Authorization: Bearer` 的形式发送它。 |
| `wire_api`       | `responses`，即网关提供、Codex 使用的接口。                         |
| `model`          | `GET /api/v1/models` 列出的任意 ID，或 `gpt-5.5` 这样的简称。       |

## 推理

`model_reasoning_effort` 设置 Codex 请求的推理强度，网关会把它转换为模型接受的形式，见[缓存与推理](../guide/caching-and-reasoning.md#推理)：

```toml
model_reasoning_effort = "high"
```

Codex 不依赖服务器端保存的状态：每个请求都带上完整的对话，模型的推理以加密内容的形式返回给 Codex，Codex 在下一轮再把它发回来。这样，推理模型的思考内容在工具调用前后都能完整保留，网关不需要任何设置。

## 上下文窗口

Codex 只知道 OpenAI 自家模型的上下文窗口。对其他 ID，它会提示没有该模型的元数据，并使用默认值。请告诉它模型的上下文窗口，让它在合适的时机压缩对话，例如对 Claude Sonnet 5.5：

```toml
model_context_window = 1000000
```

## 工具

Codex 自己的工具，例如 shell 和文件编辑，是由 Codex 自己执行的函数工具和自定义工具，它们会原样通过网关。由 OpenAI 在其服务器上运行的工具，例如联网搜索，在 Bedrock 上没有对应功能；网关会把它们从请求中去掉，而不是让请求失败，因此模型在没有这些工具的情况下照常工作。
