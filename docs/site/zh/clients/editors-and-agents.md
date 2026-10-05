# 编辑器与 Agent

允许添加 OpenAI 兼容 provider 的编辑器和编码 Agent，只需要基础 URL、密钥和你要使用的模型 ID 就能连接网关。本页以 OpenCode 和 Zed 为例；其他带有这类 provider 的客户端需要的也是这三样东西。

## OpenCode

[OpenCode](https://opencode.ai) 通过它的 `@ai-sdk/openai-compatible` 包连接网关，这个包使用 Chat Completions。把 provider 写入 `~/.config/opencode/opencode.json` 或项目根目录的 `opencode.json`，并列出你要选用的模型：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bgw": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "bedrock-gateway",
      "options": {
        "baseURL": "http://localhost:8080/api/v1",
        "apiKey": "{env:BGW_API_KEY}"
      },
      "models": {
        "global.anthropic.claude-sonnet-5-5": {
          "name": "Claude Sonnet 5.5",
          "limit": { "context": 1000000, "output": 128000 }
        },
        "us.amazon.nova-2-lite-v1:0": {
          "name": "Amazon Nova 2 Lite"
        }
      }
    }
  },
  "model": "bgw/global.anthropic.claude-sonnet-5-5"
}
```

在启动 OpenCode 的 shell 中导出密钥。`/models` 在列出的模型之间切换，`-m` 为单次运行指定模型：

```sh
export BGW_API_KEY="<the gateway's API_KEY>"
opencode
opencode run -m bgw/us.amazon.nova-2-lite-v1:0 "Reply with exactly: BEDROCK_OK"
```

最后一条命令会打印 `BEDROCK_OK`。`limit` 告诉 OpenCode 规划时使用的上下文和输出大小，数值请参考该模型在 Bedrock 上的文档。

OpenCode 的每一轮都带有工具。如果希望 Claude 这样的推理模型在工具调用之间进行思考，请开启[签名密钥](../guide/configuration.md#跨工具调用保留推理)；没有它时，网关会在关闭推理的情况下处理这些请求。

## Zed

Zed 在两个地方连接网关：它的 Agent 通过 OpenAI 兼容的语言模型 provider 连接，编辑预测则通过 Completions 路由连接。

### Agent

在 Zed 的 `settings.json` 中，把 provider 加到 `language_models` 下：

```json
{
  "language_models": {
    "openai_compatible": {
      "bedrock-gateway": {
        "api_url": "http://localhost:8080/api/v1",
        "available_models": [
          {
            "name": "global.anthropic.claude-sonnet-5-5",
            "display_name": "Claude Sonnet 5.5",
            "max_tokens": 1000000,
            "max_output_tokens": 128000,
            "capabilities": {
              "tools": true,
              "images": true,
              "parallel_tool_calls": true,
              "prompt_cache_key": false
            }
          }
        ]
      }
    }
  }
}
```

Zed 从以 provider 命名的环境变量 `BEDROCK_GATEWAY_API_KEY` 读取密钥，也可以使用你在 Agent 设置中填入的密钥。`prompt_cache_key` 保持 `false`：网关会自行放置缓存点。

### 编辑预测

Zed 的 OpenAI 兼容编辑预测会向 Completions 路由发送提示，并读取返回的文本：

```json
{
  "edit_predictions": {
    "provider": "open_ai_compatible_api",
    "open_ai_compatible_api": {
      "api_url": "http://localhost:8080/api/v1/completions",
      "model": "us.amazon.nova-2-lite-v1:0"
    }
  }
}
```

这里的 `api_url` 是路由的完整 URL，而不是基础 URL。Zed 从 `ZED_OPEN_AI_COMPATIBLE_EDIT_PREDICTION_API_KEY` 读取密钥，也可以使用你在编辑预测设置中填入的密钥。选用小而快的模型，预测会更及时。

## 其他客户端

要求填写“OpenAI 兼容”或“自定义 OpenAI”provider 的客户端，通常需要以下这些值：

| 字段                             | 值                                                          |
| -------------------------------- | ----------------------------------------------------------- |
| 基础 URL（Base URL 或 API base） | `http://localhost:8080/api/v1`                              |
| API 密钥（API key）              | 网关的 `API_KEY`                                            |
| 模型（Model）                    | `GET /api/v1/models` 列出的某个 ID，或 `gpt-5.5` 这样的简称 |

客户端要求填写完整 URL 而不是基础 URL 时，请加上路由，例如 `/chat/completions`。
