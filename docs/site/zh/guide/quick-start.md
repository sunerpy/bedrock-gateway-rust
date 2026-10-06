# 快速开始

本页从零开始，一步步得到第一个回答：为客户端设定一个密钥，在 Docker 中启动网关，再通过两种主要接口各发一个请求。你需要 Docker，以及一个能够访问 Bedrock 模型的 AWS 账号。

## 1. 为客户端设定密钥

每个客户端都要向网关发送这个密钥，因此请使用足够长的随机值并妥善保管。它不是 AWS 凭据。

```sh
export API_KEY="sk-$(openssl rand -hex 24)"
```

没有密钥时网关拒绝启动。在生产环境中，网关也可以改为从 SSM Parameter Store 或 Secrets Manager 读取密钥，见[配置](configuration.md#客户端发送的密钥)。

## 2. 让网关访问 Bedrock

最简单的方式是使用 Bedrock API 密钥，它可以在 Bedrock 控制台中创建：

```sh
export AWS_BEARER_TOKEN_BEDROCK="<your Bedrock API key>"
```

没有它时，网关使用标准的 AWS 凭据链：访问密钥、profile，或网关所在的 EC2 实例、ECS 任务或 Lambda 函数的角色。不过无论使用哪种方式，GPT-5.x 和 gpt-oss 模型都需要 Bedrock API 密钥。

请在 Bedrock 控制台中确认，你要使用的模型在你所在的区域对你的账号可用。

## 3. 启动网关

```sh
docker run --rm -p 8080:8080 \
  -e API_KEY -e AWS_BEARER_TOKEN_BEDROCK -e AWS_REGION=us-east-1 \
  sunerpy/bedrock-gateway-rust
```

网关监听 8080 端口，所有路由都位于 `/api/v1` 之下。在另一个终端中确认它在运行：

```sh
curl http://localhost:8080/api/v1/health
```

它返回 `OK`，这个请求不需要密钥。

## 4. 发送第一个请求

通过 Chat Completions：

```sh
curl -s http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  --json '{"model": "us.amazon.nova-2-lite-v1:0",
           "messages": [{"role": "user", "content": "Reply with exactly: BEDROCK_OK"}]}'
```

通过 Responses：

```sh
curl -s http://localhost:8080/api/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  --json '{"model": "us.amazon.nova-2-lite-v1:0", "input": "Reply with exactly: BEDROCK_OK"}'
```

可以使用你的账号能够调用的任何模型 ID。要查看有哪些，可以向网关查询：

```sh
curl -s http://localhost:8080/api/v1/models -H "Authorization: Bearer $API_KEY" | jq -r '.data[].id'
```

## 5. 连接客户端

基础 URL 是 `http://localhost:8080/api/v1`，API 密钥就是你的 `API_KEY`。以 OpenAI 的 Python SDK 为例：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/api/v1", api_key="sk-...your API_KEY...")
reply = client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Explain serverless in one sentence."}],
)
print(reply.choices[0].message.content)
```

Codex CLI、Node SDK 和其他工具的接入方法见[客户端](../clients/index.md)。

## 下一步

- 正式[安装](install.md)二进制文件或镜像。
- 阅读[配置](configuration.md)中的各个选项。
- 为长期运行的服务器选择一种[部署方式](../operate/index.md)。
