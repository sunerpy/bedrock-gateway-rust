# 模型

本页介绍请求中可以使用哪些模型 ID、网关如何知道每个模型接受什么，以及如何添加模型。“我能调用哪些模型”这个问题，始终以模型列表的实时结果为准：

```sh
curl -s http://localhost:8080/api/v1/models -H "Authorization: Bearer $API_KEY" | jq -r '.data[].id'
```

列表包含你的账号在网关所在区域可以调用的基础模型和推理配置文件，以及下文介绍的简称。基础模型和推理配置文件在启动时从 Bedrock 读取，之后每隔 `MODEL_CATALOG_REFRESH_SECS` 秒重新读取一次，因此网关运行期间 Bedrock 新推出的模型无需重启就会出现在列表中。

## 模型 ID

请求可以用 Bedrock 接受的任意一种 ID 指定模型：

- 基础模型 ID，例如 `amazon.nova-pro-v1:0`，适用于你所在区域按需提供的模型；
- 跨区域推理配置文件 ID，例如 `us.anthropic.claude-sonnet-4-5-20250929-v1:0` 或 `global.anthropic.claude-sonnet-5-5`，Bedrock 会在多个区域之间路由这类请求。大多数较新的模型只能这样调用；
- 你账号中的应用程序推理配置文件 ARN；
- 网关提供的简称，例如 `gpt-5.5`，见下文。

网关把 ID 按你写的原样发送给 Bedrock，推理配置文件因此保留它的路由方式。`us.`、`eu.`、`apac.`、`jp.`、`au.`、`ca.` 和 `global.` 这些前缀都可以使用。

## Claude

Claude Sonnet、Opus、Haiku 和 Fable 通过 Bedrock 的 Converse API 提供，Chat Completions 和 Responses 都可以使用。较新的版本通过推理配置文件调用，例如 `global.anthropic.claude-sonnet-5-5`、`us.anthropic.claude-opus-5-5`、`global.anthropic.claude-haiku-5-5` 或 `global.anthropic.claude-fable-5-1`。

各版本之间的差异由网关处理。对拒绝 `temperature` 和 `top_p` 的版本，网关会去掉这两个参数。对使用自适应思考的版本，`reasoning_effort` 会转换为模型所需的推理强度。某个版本不接受以 assistant 消息结尾的对话时，网关会追加一条简短的用户消息，请模型接着回答。

`response_format` 和 Responses 的 `text.format` 在所有版本上都会作为结构化输出发给 Bedrock，由 Bedrock 判断该版本是否支持；不支持的版本（例如 Fable 5）返回 HTTP 400。在 Claude 上，JSON schema 里的每个对象都必须设置 `additionalProperties: false`。

## OpenAI GPT

Bedrock 上的 OpenAI 模型使用简称调用：

| 名称                                                    | 提供方式                           | Chat Completions | Responses |
| ------------------------------------------------------- | ---------------------------------- | ---------------- | --------- |
| `gpt-5.4`、`gpt-5.5`                                    | Bedrock 的 OpenAI 兼容端点         | 支持             | 支持      |
| `gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna`          | Bedrock 的 OpenAI 兼容端点         | 支持             | 支持      |
| `gpt-6.1-sol`、`gpt-6-sol`、`gpt-6-luna`、`gpt-6-astra` | Bedrock Converse，全球推理配置文件 | 支持             | 支持      |
| `gpt-oss-120b`、`gpt-oss-20b`                           | Bedrock 的 OpenAI 兼容端点         | 支持             | 不支持    |

- **GPT-5.x 和 gpt-oss** 发往 Bedrock 的 OpenAI 兼容端点。这个端点需要 Bedrock API 密钥（`AWS_BEARER_TOKEN_BEDROCK`），并且只在少数区域可用：`us-east-1`、`us-east-2` 和 `us-west-2`，具体取决于模型。在其他区域，网关返回 HTTP 400。在 Chat Completions 上，网关把 GPT-5.x 的请求转换为 Responses API 请求，再把回答转换回来，推理摘要出现在 `<think>` 标签中。
- **GPT-6.x** 通过 Bedrock Converse 和 `global.openai.*` 推理配置文件调用，因此不需要 Bedrock API 密钥，在该推理配置文件可用的地方都能使用。直接使用 `us.openai.*` 和 `global.openai.*` 形式的 ID 也可以。`reasoning_effort` 会转换为 `reasoning.effort`；这些模型拒绝 `temperature` 和 `top_p`，网关因此去掉这两个参数；结构化输出使用这些模型支持的严格 JSON schema。

## 其他模型

Amazon Nova、DeepSeek 以及你的模型目录中的其他模型都通过 Bedrock Converse 提供，无需额外设置。xAI Grok 4 和 Moonshot Kimi K3 拒绝 `temperature` 和 `top_p`，网关会去掉这两个参数。模型的限制以 Bedrock 为准：模型不支持图片、工具、流式输出或结构化输出时，Bedrock 的错误会返回给客户端。

Embeddings 使用单独的注册表，支持的模型系列见 [API 参考](../reference/api.md#embeddings)。

## 网关如何了解每个模型

每个模型接受什么，记录在注册表 [`config/models.toml`](../../../../config/models.toml) 中，而不是写在代码里。每个条目匹配模型 ID 中的一个片段，可以设置的内容例如：

- 模型是否拒绝 `temperature` 和 `top_p`；
- 模型接受哪种形式的推理参数；
- 值得缓存的最短提示前缀，以及是否支持 1 小时缓存；
- 由哪个后端提供服务，以及在哪些区域可用。

没有对应条目的模型使用默认设置：不带推理参数，也不使用提示缓存（Claude 的 ID 除外，它会得到一个保守的缓存阈值）。

## 添加或修改模型

1. 从你正在运行的版本中取出 `config/models.toml`，复制到一个目录中，例如 `/etc/bedrock-gateway/config/`。
2. 添加或修改条目：

   ```toml
   [[model]]
   match = "provider.model-name"
   capabilities = ["drop_sampling_params", "no_assistant_prefill"]
   [model.params]
   cache_min_tokens = 1024
   reasoning_path = "adaptive_thinking"
   ```

3. 启动网关，并把 `CONFIG_DIR` 指向该目录。

一个 ID 同时匹配两个条目时，参数取自文件中靠前的那个，因此请把更具体的条目（例如 `claude-sonnet-5-5`）放在更短的条目（例如 `claude-sonnet-5`）之前。所有匹配条目的能力标志会叠加在一起。`models.toml` 开头的注释说明了每个标志和参数。

简称是一个 `[[alias]]` 条目，必须写在第一个 `[[model]]` 之前：

```toml
[[alias]]
from = "my-model"
to = "global.provider.model-name"
```
