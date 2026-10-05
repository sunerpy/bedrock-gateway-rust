# 缓存与推理

本页介绍网关在每个请求上替你做的两件事：放置提示缓存点，以及把 OpenAI 的推理设置转换为每个模型需要的形式。每个模型的阈值见[缓存与推理详解](../../../caching-and-reasoning.md)。

## 提示缓存

Bedrock 可以缓存提示的开头部分，对重复出现的开头按更低的费率计费。这需要请求中带有缓存点，而 OpenAI 客户端不会发送缓存点，因此由网关添加。这项功能默认开启（`ENABLE_PROMPT_CACHING=true`）。

在支持缓存的模型上，网关依次在工具定义之后、系统提示之后以及消息中放置缓存点，数量不超过模型在单个请求中允许的上限（Claude 为四个）。只有当缓存点之前的提示长度达到模型的最小值时，网关才会放置这个缓存点；这个最小值因模型而异，在 512 到 4,096 个 token 之间。

缓存命中要求前缀在前后两个请求之间保持不变。修改工具描述、在系统提示中加入时间戳或者调整工具顺序，都会产生新的缓存条目。

响应中的用量信息会体现缓存的效果。`prompt_tokens` 统计全部输入 token，无论是否来自缓存；`prompt_tokens_details.cached_tokens` 统计其中从缓存读取的部分。在 Responses API 上，对应的字段是 `input_tokens` 和 `input_tokens_details.cached_tokens`。

### 按请求控制

请求体中 `extra_body` 对象里的 `prompt_caching` 对象，可以为单个请求覆盖默认设置。OpenAI SDK 会把自己的 `extra_body` 选项合并到请求的顶层，因此在 Python SDK 中要再嵌套一层：

```python
client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=messages,
    extra_body={"extra_body": {"prompt_caching": {"system": True, "messages": False, "ttl": "1h"}}},
)
```

这样，请求体中带有：

```json
"extra_body": {"prompt_caching": {"system": true, "messages": false, "ttl": "1h"}}
```

| 字段       | 含义                                            |
| ---------- | ----------------------------------------------- |
| `system`   | 设为 `false` 时，系统提示不放置缓存点           |
| `messages` | 设为 `false` 时，消息中不放置缓存点             |
| `ttl`      | `5m` 或 `1h`；为这个请求覆盖 `PROMPT_CACHE_TTL` |

在 Responses API 上，只有 `ttl` 生效。缓存条目的存活时间是五分钟；在支持的模型（例如较新的 Claude 版本）上设置 `ttl: "1h"` 时为一小时。在其他模型上，请求的一小时会变成五分钟，网关会记录一条警告。网关在调用 Bedrock 之前会移除 `prompt_caching`。

## 推理

请求以 OpenAI 的方式开启推理：Chat Completions 上用 `reasoning_effort`，Responses 上用 `reasoning.effort`。可用的值有 `none`、`minimal`、`low`、`medium`、`high`、`xhigh` 和 `max`；每个模型只接受其中一部分，模型不接受的值会被 Bedrock 拒绝。

```python
client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Solve step by step: 17 * 23"}],
    reasoning_effort="high",
)
```

网关把推理强度转换为模型接受的形式：

| 模型                                                   | 模型收到的内容                     |
| ------------------------------------------------------ | ---------------------------------- |
| Claude Sonnet 4.5 和 4.6、Haiku 4.5、Opus 4.5          | 根据 `max_tokens` 计算出的思考预算 |
| Claude Opus 4.6 及更高版本、Sonnet 5 及更高版本、Fable | 使用该推理强度的自适应思考         |
| GPT-6.x                                                | `reasoning.effort`                 |
| GPT-5.x 和 gpt-oss                                     | 客户端发送的原始请求               |
| DeepSeek V3                                            | 字符串形式的推理强度               |

对于思考预算，`low` 是 `max_tokens` 的 30%，`medium` 是 60%，更高的值比 `max_tokens` 少一个 token，且都不低于 1,024 个 token。如果预算给回答留下的空间太少，网关会提高发给 Bedrock 的 `max_tokens`。

### 推理内容出现在哪里

| 路由                             | 回答中的推理内容                                           |
| -------------------------------- | ---------------------------------------------------------- |
| Chat Completions                 | 位于 `content` 中，包在 `<think>` 和 `</think>` 之间       |
| Responses                        | `output` 中的一个 `reasoning` 条目，位于消息之前           |
| 使用 gpt-oss 的 Chat Completions | 每个流式 delta 中的 `reasoning`，与 Bedrock 发送的内容一致 |

模型报告推理 token 数时，用量信息会在 `completion_tokens_details.reasoning_tokens` 或 `output_tokens_details.reasoning_tokens` 中给出。

### 推理与工具

模型先推理再调用工具时，下一个请求必须把这段推理连同它的签名一起带回去。Responses API 通过它的 `reasoning` 条目做到这一点。在 Chat Completions 上，只要你为网关提供签名密钥，网关就可以把推理放在工具调用的 ID 中，见[跨工具调用保留推理](configuration.md#跨工具调用保留推理)。
