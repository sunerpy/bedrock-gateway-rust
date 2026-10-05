# Quick start

This page takes you from nothing to a first answer: a key for your clients, the gateway running in Docker, and one
request through each of the two main APIs. It needs Docker and an AWS account with access to a Bedrock model.

## 1. Choose a key for your clients

Every client sends this key to the gateway. Pick a long random value and keep it private. It is not an AWS
credential.

```sh
export API_KEY="sk-$(openssl rand -hex 24)"
```

The gateway refuses to start without a key. In production it can read the key from SSM Parameter Store or Secrets
Manager instead; see [Configuration](configuration.md#the-key-your-clients-send).

## 2. Give the gateway access to Bedrock

The simplest way is a Bedrock API key, which you create in the Bedrock console:

```sh
export AWS_BEARER_TOKEN_BEDROCK="<your Bedrock API key>"
```

Without it the gateway uses the standard AWS credential chain: access keys, a profile, or the role of the EC2
instance, ECS task or Lambda function it runs in. GPT-5.x and gpt-oss models need the Bedrock API key either way.

Check in the Bedrock console that the models you want are available to your account in your region.

## 3. Start the gateway

```sh
docker run --rm -p 8080:8080 \
  -e API_KEY -e AWS_BEARER_TOKEN_BEDROCK -e AWS_REGION=us-east-1 \
  sunerpy/bedrock-gateway-rust
```

The gateway listens on port 8080 and serves every route under `/api/v1`. In a second terminal, check that it is
alive:

```sh
curl http://localhost:8080/api/v1/health
```

It answers `OK`, without a key.

## 4. Send a first request

Through Chat Completions:

```sh
curl -s http://localhost:8080/api/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  --json '{"model": "us.amazon.nova-2-lite-v1:0",
           "messages": [{"role": "user", "content": "Reply with exactly: BEDROCK_OK"}]}'
```

Through Responses:

```sh
curl -s http://localhost:8080/api/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
  --json '{"model": "us.amazon.nova-2-lite-v1:0", "input": "Reply with exactly: BEDROCK_OK"}'
```

Use any model ID your account can call. To see them, ask the gateway:

```sh
curl -s http://localhost:8080/api/v1/models -H "Authorization: Bearer $API_KEY" | jq -r '.data[].id'
```

## 5. Point a client at it

The base URL is `http://localhost:8080/api/v1` and the API key is your `API_KEY`. With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/api/v1", api_key="sk-...your API_KEY...")
reply = client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Explain serverless in one sentence."}],
)
print(reply.choices[0].message.content)
```

[Clients](../clients/index.md) covers Codex CLI, the Node SDK and other tools.

## Next steps

- [Install](install.md) the binary or the image for good.
- Read the [configuration](configuration.md) options.
- Choose a [deployment](../operate/index.md) for a server that stays up.
