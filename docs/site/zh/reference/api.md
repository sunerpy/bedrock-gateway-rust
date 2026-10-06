# API 参考

网关在 `API_ROUTE_PREFIX` 下提供以下路由，未修改时这个前缀为 `/api/v1`。每个路由接收 OpenAI 客户端发送的请求，并以 OpenAI 的格式返回。本页列出每个路由接受哪些内容、与 OpenAI 有哪些不同；流式与工具方面的详细约定见[协议兼容性](../../../openai-protocol-compatibility.md)参考。

| 方法   | 路由                       | 用途                                  |
| ------ | -------------------------- | ------------------------------------- |
| `POST` | `/api/v1/chat/completions` | [Chat Completions](#chat-completions) |
| `POST` | `/api/v1/responses`        | [Responses](#responses)               |
| `POST` | `/api/v1/completions`      | [Completions](#completions)           |
| `POST` | `/api/v1/embeddings`       | [Embeddings](#embeddings)             |
| `GET`  | `/api/v1/models`           | [模型列表](#模型)                     |
| `GET`  | `/api/v1/models/{id}`      | [单个模型](#模型)                     |
| `GET`  | `/api/v1/health`           | [存活检查](#健康检查)                 |

## 请求

除 `/health` 外，每个路由都需要 `Authorization: Bearer <API_KEY>`。请求体为 JSON，包括 base64 图片在内，大小不超过 `MAX_BODY_SIZE_MB`（默认 20 MB）。

客户端可以在 `x-request-id` 中为请求附上自己的 ID，网关为这个请求写的日志行就会带上它。网关响应来自任何源的跨域请求，因此持有密钥的浏览器页面也可以调用它。

## 错误

错误以 OpenAI 的格式返回：

```json
{"error": {"message": "model `nope` not found", "type": "invalid_request_error", "code": "bad_request"}}
```

| 状态码 | `code`                | 场景                                                                                                                      |
| ------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 400    | `bad_request`         | 请求格式错误、指定了未知模型、要求模型提供它不具备的功能、指定了网关所在区域不提供的模型，或者 Bedrock 认为请求无效而拒绝 |
| 400    | `unsupported`         | 请求使用了网关不支持的字段，例如 Completions 上的 `suffix`                                                                |
| 401    | `unauthorized`        | 密钥缺失或错误，或者 Bedrock 拒绝网关的 AWS 身份访问该模型                                                                |
| 405    |                       | 路由存在，但不支持这个方法                                                                                                |
| 413    |                       | 请求体超过 `MAX_BODY_SIZE_MB`                                                                                             |
| 429    | `rate_limit_exceeded` | 网关重试之后，请求仍被 Bedrock 限流                                                                                       |
| 500    | `internal_error`      | 网关自身出错，例如无法下载图片 URL                                                                                        |
| 502    | `upstream_error`      | Bedrock 报告模型或服务故障，例如模型尚未就绪                                                                              |

流开始之后发生的错误会在流中返回：在 Chat Completions 上是一个错误事件；在 Responses 上是一个 `response.failed` 事件，它的 `error.code` 是 Codex CLI 能据此处理的值之一：`rate_limit_exceeded`、`server_is_overloaded`、`context_length_exceeded` 或 `server_error`。来自 Bedrock OpenAI 兼容端点（提供 GPT-5.x 和 gpt-oss）的错误，会保留该端点返回的状态码和错误对象。

## Chat Completions

`POST /api/v1/chat/completions` 接收 OpenAI 的 chat 请求：

| 字段                                                 | 处理方式                                                                               |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `model`                                              | 任意模型 ID 或简称，见[模型](../guide/models.md)                                       |
| `messages`                                           | `system`、`developer`、`user`、`assistant` 和 `tool` 消息；支持文本和 `image_url` 部分 |
| `stream`、`stream_options`                           | Server-sent events；`include_usage` 会在 `data: [DONE]` 之前加一个用量 chunk           |
| `max_completion_tokens`、`max_tokens`                | 输出上限；两者都设置时以 `max_completion_tokens` 为准                                  |
| `temperature`、`top_p`、`stop`                       | 传给 Bedrock，拒绝采样参数的模型除外                                                   |
| `tools`、`tool_choice`                               | 函数工具；`tool_choice` 可以是 `auto`、`none`、`required` 或某一个函数                 |
| `response_format`                                    | 在支持结构化输出的模型上支持 `json_schema` 和 `json_object`；其他模型返回 HTTP 400     |
| `reasoning_effort`                                   | 映射为每个模型的推理方式，见[缓存与推理](../guide/caching-and-reasoning.md#推理)       |
| `extra_body`                                         | `prompt_caching` 控制缓存；其他所有键随请求一起发给 Bedrock                            |
| `n`、`frequency_penalty`、`presence_penalty`、`user` | 接受但忽略                                                                             |

图片可以是 `data:` URL，也可以是 `http` 或 `https` URL，后者由网关下载。回答是一个 `chat.completion`，或者一系列以 `data: [DONE]` 结尾的 `chat.completion.chunk` 事件。模型的推理出现在 `content` 中，位于 `<think>` 和 `</think>` 之间；对 GPT-5.x，出现在这里的是推理摘要。

`extra_body` 中除 `prompt_caching` 以外的键，以及顶层字段 `thinking`、`output_config` 和 `anthropic_beta`，会作为附加的模型请求字段（additional model request fields）发给 Bedrock，用于 OpenAI 请求中没有对应字段的模型选项。

## Responses

`POST /api/v1/responses` 接收 OpenAI 的 Responses 请求，服务器端不保存状态：

| 字段                            | 处理方式                                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `model`                         | 任意模型 ID 或简称                                                                                                                |
| `input`、`instructions`         | 字符串或条目列表：包含文本和图片的消息、工具调用及其输出、推理                                                                    |
| `tools`、`tool_choice`          | `function` 和 `custom` 工具、带命名空间的工具，以及由客户端执行的 `local_shell`、`shell` 和 `apply_patch` 工具                    |
| `stream`                        | 带类型的事件，从 `response.created` 开始，以 `response.completed`、`response.incomplete` 或 `response.failed` 结束，没有 `[DONE]` |
| `max_output_tokens`             | 输出上限；因它而截断的回答以 `incomplete` 结束                                                                                    |
| `temperature`、`top_p`          | 传给 Bedrock，拒绝这两个参数的模型除外                                                                                            |
| `reasoning.effort`              | 映射为每个模型的推理方式                                                                                                          |
| `text.format`                   | 在支持结构化输出的模型上支持 `json_schema` 和 `json_object`                                                                       |
| `include`                       | `reasoning.encrypted_content` 以客户端可以在下一轮发回的形式返回推理内容                                                          |
| `store`、`previous_response_id` | 接受但忽略：客户端每次都发送完整对话                                                                                              |

由 OpenAI 在其服务器上运行的工具，例如联网搜索、文件搜索、代码解释器和图片生成，在 Bedrock 上没有对应功能；网关会把它们从请求中去掉，保留客户端自己的工具。推理模型的推理内容是 `output` 中的一个 `reasoning` 条目，位于消息之前。

## Completions

`POST /api/v1/completions` 提供旧版文本补全，供代码补全功能仍在使用它的编辑器调用。`prompt` 是一个字符串，或者一个字符串列表，网关会用换行符把列表拼接起来；token 数组和 `suffix` 会被拒绝。`max_tokens`、`temperature`、`top_p`、`stop` 和 `stream` 的用法与 Chat Completions 相同，回答是一个 `text_completion`。网关把提示作为一条用户消息发给模型。这个路由不提供 GPT-5.x 和 gpt-oss。

## Embeddings

`POST /api/v1/embeddings` 提供嵌入模型注册表 [`config/embeddings.toml`](../../../../config/embeddings.toml) 中的嵌入模型：

| 模型系列 | 模型 ID                                                      | 输入               |
| -------- | ------------------------------------------------------------ | ------------------ |
| Cohere   | `cohere.embed-multilingual-v3`、`cohere.embed-english-v3`    | 字符串或列表       |
| Titan    | `amazon.titan-embed-text-v1`、`amazon.titan-embed-text-v2:0` | 每个请求一个字符串 |
| Nova     | `amazon.nova-2-multimodal-embeddings-v1:0`                   | 每个请求一个字符串 |

Titan 和 Nova 收到多于一个输入的列表时返回 HTTP 400。`encoding_format` 可以是 `float` 或 `base64`。Nova 接受的 `dimensions` 为 256、384、1024 或 3072（默认）。token 数组会先用 `cl100k_base` 编码解码为文本。注册表之外的模型返回 HTTP 400；要提供这样的模型，请把它连同所属的模型系列一起加入注册表。

## 模型

`GET /api/v1/models` 以 OpenAI `model` 对象的形式，列出你的账号在网关所在区域可以调用的基础模型和推理配置文件，以及简称。网关在启动时以及之后每隔 `MODEL_CATALOG_REFRESH_SECS` 秒从 Bedrock 读取这个列表，并从内存中提供。`GET /api/v1/models/{id}` 返回单个条目；ID 不在列表中时返回 HTTP 400。

## 健康检查

`GET /api/v1/health` 以 HTTP 200 返回 `OK`，不需要密钥。它只表明进程在运行，不代表 Bedrock 可以访问。
