# What is bedrock-gateway

bedrock-gateway lets software written for the OpenAI API use the models in Amazon Bedrock. It is one program that you
run in your own environment. Clients send it OpenAI requests; it calls Bedrock with your AWS account and answers in
the OpenAI format.

## What it is for

Many tools and libraries speak the OpenAI API and nothing else: the OpenAI SDKs, Codex CLI, editors with edit
prediction, and agents that let you set a base URL. bedrock-gateway gives them an OpenAI endpoint whose models are
Claude, OpenAI's GPT models on Bedrock, Amazon Nova, DeepSeek and the other models in your Bedrock catalog.

A client changes two settings, the base URL and the API key, and keeps the rest of its code.

## How a request travels

1. A client sends an OpenAI request, such as a Chat Completions or Responses request, to the gateway with the key you
   chose for your clients.
2. The gateway checks the key, translates the request into the form the model needs on Bedrock and sends it with your
   AWS credentials.
3. Bedrock answers. The gateway turns the answer, or the stream of events, back into the OpenAI shape the client asked
   for.

Two credentials are involved, and they go in opposite directions:

| Credential                                | Who sends it          | To whom    |
| ----------------------------------------- | --------------------- | ---------- |
| `API_KEY`                                 | your clients          | the gateway |
| a Bedrock API key or AWS credentials       | the gateway           | Bedrock    |

## What it supports

- The OpenAI routes `/chat/completions`, `/responses`, `/completions`, `/embeddings` and `/models`, streaming and
  non-streaming. The [API reference](../reference/api.md) lists each one.
- Tools, images, JSON output and reasoning effort, on the models that support them.
- Automatic prompt caching on the models that support it, without a change in the client.
- Bedrock's cross-region inference profiles, such as `global.anthropic.claude-sonnet-5-5`.
- A model registry in a configuration file. A model that needs special handling gets an entry there, without a new
  release of the gateway.

## Where it runs

bedrock-gateway is a single binary with no runtime dependencies. It runs on Linux, macOS and Windows, and the same
binary ships as a small container image. The [deployment guide](../operate/index.md) covers running it on your own
machine, on Amazon ECS with Fargate and on AWS Lambda.

## What it is not

- It is not a hosted service. You run it, and it uses your AWS account and your Bedrock quotas.
- It does not keep conversations. The Responses API is served without state.
- It is not an AWS product.

Next: [Quick start](quick-start.md).
