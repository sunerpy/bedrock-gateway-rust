# 选择客户端

每个客户端需要的都是同样两样东西：网关的基础 URL，以及网关 `API_KEY` 中的密钥。客户端里不需要安装任何东西。

| 客户端      | 接口                          | 基础 URL                       | 指南                                    |
| ----------- | ----------------------------- | ------------------------------ | --------------------------------------- |
| Codex CLI   | Responses                     | `http://localhost:8080/api/v1` | [Codex CLI](codex.md)                   |
| OpenCode    | Chat Completions              | `http://localhost:8080/api/v1` | [编辑器与 Agent](editors-and-agents.md) |
| Zed         | Chat Completions、Completions | `http://localhost:8080/api/v1` | [编辑器与 Agent](editors-and-agents.md) |
| OpenAI SDK  | Chat Completions、Responses   | `http://localhost:8080/api/v1` | [见下文](#sdk)                          |
| curl 和脚本 | 任意路由                      | `http://localhost:8080/api/v1` | [API 参考](../reference/api.md)         |

客户端会在基础 URL 后面加上路由，例如 `/chat/completions`，因此基础 URL 以 `/api/v1` 结尾，也就是默认的 `API_ROUTE_PREFIX`。请把 `localhost:8080` 换成你的客户端访问网关时使用的地址。

## 密钥

客户端以 `Authorization: Bearer <key>` 的形式发送密钥，OpenAI SDK 和大多数其他客户端在配置了 API 密钥后都是这样发送的。不带密钥或者密钥不对的请求会得到 HTTP 401，错误代码为 `unauthorized`。网关只有一个密钥，持有它的每个客户端都拥有相同的访问权限，因此请像对待密码一样对待它；需要收回某人的访问权限时，请更换密钥。

## 模型名称

`GET /api/v1/models` 列出你的账号在网关所在区域可以调用的模型 ID；网关还接受[模型](../guide/models.md)中介绍的简称，例如 `gpt-5.5`。客户端按列表中的写法发送 ID，网关会原样传给 Bedrock。

有些客户端只认识自家厂商的模型，遇到不认识的名称会发出警告或拒绝使用。这些客户端各自的指南页面说明了在哪里声明模型。

## SDK

OpenAI 的 Python SDK，通过 Responses：

```python
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/api/v1", api_key=os.environ["API_KEY"])
response = client.responses.create(
    model="us.amazon.nova-2-lite-v1:0",
    input="Reply with exactly: BEDROCK_OK",
)
print(response.output_text)
```

OpenAI 的 JavaScript SDK，通过 Chat Completions：

```ts
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:8080/api/v1",
  apiKey: process.env.API_KEY,
});

const completion = await openai.chat.completions.create({
  model: "us.amazon.nova-2-lite-v1:0",
  messages: [{ role: "user", content: "Reply with exactly: BEDROCK_OK" }],
});
console.log(completion.choices[0].message.content);
```

两段代码都会打印 `BEDROCK_OK`。Bedrock 独有的选项，例如[提示缓存控制](../guide/caching-and-reasoning.md#按请求控制)，放在 SDK 的 `extra_body` 选项中。

## 其他客户端

任何支持 OpenAI Chat Completions 或 Responses 的客户端都可以使用同样的基础 URL。如果客户端两者都支持，而对话要让推理模型使用工具，请优先选择 Responses：Responses 自己就能把模型的推理从一轮带到下一轮，Chat Completions 则需要[签名密钥](../guide/configuration.md#跨工具调用保留推理)才能做到。[协议兼容性](../../../openai-protocol-compatibility.md)参考列出了每个路由支持哪些请求字段和工具类型。
