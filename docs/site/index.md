---
layout: home
title: "bedrock-gateway: Amazon Bedrock behind the OpenAI APIs"
titleTemplate: false
description: bedrock-gateway is one Rust binary that serves Amazon Bedrock models through the OpenAI Chat Completions, Responses, Completions and Embeddings APIs, so the OpenAI SDKs, Codex CLI and other agents need only a base URL and a key.

hero:
  name: bedrock-gateway
  text: Amazon Bedrock behind the OpenAI APIs
  tagline: One Rust binary that answers OpenAI Chat Completions, Responses, Completions and Embeddings requests by calling Bedrock with your own AWS account. An OpenAI SDK or an agent needs a base URL and a key, and nothing else changes.
  actions:
    - theme: brand
      text: Install
      link: /guide/install
    - theme: alt
      text: Quick start
      link: /guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/sunerpy/bedrock-gateway-rust

home:
  facts:
    - term: Runs as
      text: A binary for Linux, macOS and Windows, or a container image for Linux, on your own machine, on ECS or on Lambda. It listens on port 8080.
    - term: Your AWS account
      text: Bedrock is called with a Bedrock API key or the standard AWS credential chain. Prompts go to Bedrock in your account.

  visual:
    label: A terminal session. One request goes to the Chat Completions route and one to the Responses route, both with Amazon Nova 2 Lite, and the gateway answers both with BEDROCK_OK.
    transcript:
      - kind: command
        text: export BGW=http://localhost:8080/api/v1
      - kind: command
        text: curl -s $BGW/chat/completions \
      - kind: continuation
        text: "  -H \"Authorization: Bearer $API_KEY\" --json '{"
      - kind: continuation
        text: '    "model": "us.amazon.nova-2-lite-v1:0",'
      - kind: continuation
        text: '    "messages": [{"role": "user",'
      - kind: continuation
        text: "      \"content\": \"Reply with exactly: BEDROCK_OK\"}]}' |"
      - kind: continuation
        text: "  jq -r '.choices[0].message.content'"
      - kind: output
        text: BEDROCK_OK
      - kind: command
        text: curl -s $BGW/responses \
      - kind: continuation
        text: "  -H \"Authorization: Bearer $API_KEY\" --json '{"
      - kind: continuation
        text: '    "model": "us.amazon.nova-2-lite-v1:0",'
      - kind: continuation
        text: "    \"input\": \"Reply with exactly: BEDROCK_OK\"}' |"
      - kind: continuation
        text: "  jq -r '.output[-1].content[0].text'"
      - kind: output
        text: BEDROCK_OK

  index:
    title: What bedrock-gateway does
    intro: Everything below works once the gateway has a key for its clients and AWS credentials; GPT-5.x and gpt-oss also need a Bedrock API key. Reasoning across tool calls on Chat Completions and OpenTelemetry export stay off until you turn them on.
    groups:
      - name: Serve
        items:
          - title: Chat Completions
            body: Streaming and non-streaming chat with tools, images, JSON output and reasoning effort.
            status: available
            link: /reference/api#chat-completions
          - title: Responses
            body: The Responses API that Codex CLI uses, streaming and non-streaming. It keeps no state between requests.
            status: available
            link: /reference/api#responses
          - title: Completions
            body: The legacy text-completion route, for editors whose edit prediction still speaks it.
            status: available
            link: /reference/api#completions
          - title: Embeddings
            body: Cohere, Titan and Nova embedding models behind the OpenAI embeddings request.
            status: available
            link: /reference/api#embeddings
          - title: Model list
            body: GET /api/v1/models lists the models and inference profiles your account can call, read from Bedrock and refreshed while the gateway runs.
            status: available
            link: /guide/models
      - name: Models
        items:
          - title: Claude
            body: Sonnet, Opus, Haiku and Fable, through Bedrock model IDs and cross-region inference profiles.
            status: available
            link: /guide/models#claude
          - title: OpenAI GPT on Bedrock
            body: GPT-5.x, GPT-6.x and gpt-oss, under their short names such as gpt-5.6-sol or gpt-6.1-sol.
            status: available
            link: /guide/models#openai-gpt
          - title: Every other Bedrock model
            body: Amazon Nova, DeepSeek and any model in your account's catalog. A new model is a configuration entry, not a new release.
            status: available
            link: /guide/models#other-models
      - name: Behaviour
        items:
          - title: Prompt caching
            body: Cache points are placed on the tools, the system prompt and the messages without any change in the client, on the models that support it.
            status: available
            link: /guide/caching-and-reasoning#prompt-caching
          - title: Reasoning effort
            body: reasoning_effort becomes the form each model expects, from Claude's adaptive thinking to GPT's reasoning.effort.
            status: available
            link: /guide/caching-and-reasoning#reasoning
          - title: Reasoning across tool calls
            body: Keeps a model's signed reasoning through a Chat Completions tool call and its result. It needs a signing key of your own.
            status: opt-in
            link: /guide/configuration#reasoning-across-tool-calls
      - name: Operate
        items:
          - title: Deploy anywhere
            body: A binary, the Docker image, ECS on Fargate behind a load balancer, or Lambda with a function URL.
            status: available
            link: /operate/
          - title: Keys kept in AWS
            body: The client key can come from SSM Parameter Store or Secrets Manager instead of an environment variable.
            status: available
            link: /guide/configuration#the-key-your-clients-send
          - title: Logs and traces
            body: A log line per request with the model, tokens, cache hits and timing, never prompts or keys. OpenTelemetry export is a build option.
            status: opt-in
            link: /operate/observability

  steps:
    title: From install to the first answer
    items:
      - title: Pull the image
        command: docker pull sunerpy/bedrock-gateway-rust
        body: Or download the binary for your platform from GitHub Releases, or install it with cargo.
      - title: Choose a key for your clients
        command: export API_KEY=sk-replace-with-a-private-random-key
        body: Every client sends this key. It is yours to choose and is not an AWS credential.
      - title: Start the gateway
        command: docker run -p 8080:8080 -e API_KEY -e AWS_REGION=us-east-1 -e AWS_BEARER_TOKEN_BEDROCK sunerpy/bedrock-gateway-rust
        body: AWS_BEARER_TOKEN_BEDROCK is a Bedrock API key. Without it the gateway uses the standard AWS credential chain.
      - title: Point a client at it
        command: http://localhost:8080/api/v1
        body: That is the base URL for the OpenAI SDKs, Codex CLI and other OpenAI-compatible clients.

  routes:
    columns: [Route, Speaks, Notes]
    rows:
      - cells: [POST /api/v1/chat/completions, OpenAI Chat Completions, Streaming and non-streaming]
      - cells: [POST /api/v1/responses, OpenAI Responses, Stateless]
      - cells: [POST /api/v1/completions, OpenAI Completions, Legacy text completion]
      - cells: [POST /api/v1/embeddings, OpenAI Embeddings, "Cohere, Titan and Nova"]
      - cells: [GET /api/v1/models, Model list, Read from Bedrock]
      - cells: [GET /api/v1/health, Liveness, No key needed]
    code: [0]
    caption: Every route except /health asks for your API_KEY, sent as Authorization Bearer. API_ROUTE_PREFIX changes /api/v1.

  clients:
    columns: [Client, API, Set up with]
    rows:
      - cells: [OpenAI Python and Node SDKs, Chat Completions or Responses, A base URL and an API key]
      - cells: [Codex CLI, Responses, A model_provider in config.toml]
      - cells: [Editors and agents, Chat Completions or Completions, An OpenAI-compatible provider]
      - cells: [curl and scripts, Any route, The Authorization header]

  models:
    columns: [Family, Model IDs, Served through]
    rows:
      - cells: [Claude, global.anthropic.claude-sonnet-5-5 us.anthropic.claude-opus-5-5 global.anthropic.claude-fable-5-1, Bedrock Converse]
      - cells: [GPT-5.x, gpt-5.4 gpt-5.5 gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna, OpenAI-compatible endpoint]
      - cells: [GPT-6.x, gpt-6.1-sol gpt-6-sol gpt-6-luna gpt-6-astra, Bedrock Converse]
      - cells: [gpt-oss, gpt-oss-120b gpt-oss-20b, OpenAI-compatible endpoint]
      - cells: [Amazon Nova, us.amazon.nova-2-lite-v1:0 us.amazon.nova-pro-v1:0, Bedrock Converse]
    code: [1]
    caption: DeepSeek and every other model in your catalog are called the same way. The model list shows the IDs your account can call in the gateway's region.

  platforms:
    title: Platforms
    intro: Every release has a binary for each platform below, a multi-architecture image on Docker Hub and Amazon ECR Public, and the crate on crates.io.
    columns: [Platform, Release asset, Container image]
    rows:
      - name: Linux x64
        status: available
        cells: [x86_64-unknown-linux-musl, linux/amd64]
      - name: Linux ARM64
        status: available
        cells: [aarch64-unknown-linux-musl, linux/arm64]
      - name: macOS Intel
        status: available
        cells: [x86_64-apple-darwin, None]
      - name: macOS Apple Silicon
        status: available
        cells: [aarch64-apple-darwin, None]
      - name: Windows x64
        status: available
        cells: [x86_64-pc-windows-msvc, None]
    note: The Linux binaries are static, and the image is distroless with the binary and its configuration only.

  privacy:
    title: What goes where
    intro: bedrock-gateway sits between your clients and your AWS account. Apart from the image URLs clients send and an OpenTelemetry collector you configure, it talks to AWS only.
    sendsLabel: Goes to
    modes:
      - name: Model requests
        sends: Amazon Bedrock, in your account
        detail: The prompt, tools and images of each request go to Bedrock in the region you configure. The log records the model, token counts, cache hits and timing, never prompts, answers or keys.
      - name: Model list
        sends: The Bedrock control plane
        detail: At start and then on a schedule, the gateway lists the foundation models and inference profiles your account can call.
      - name: Image URLs
        sends: The host in the URL
        detail: When a message carries an http or https image URL, the gateway downloads the image and sends its bytes to Bedrock. A data URL is decoded in the gateway.
      - name: Client key
        sends: SSM or Secrets Manager, if you use them
        detail: When the key comes from Parameter Store or Secrets Manager, the gateway reads it there once at start. Otherwise it stays in the environment.

  scope:
    title: What it does not do
    items:
      - It does not store conversations. The Responses API is served without state, so previous_response_id and store are accepted and ignored.
      - It does not speak Anthropic Messages. Claude is served through the OpenAI APIs.
      - It does not run hosted tools such as web search or file search. A client's function tools pass through; hosted tools with no Bedrock counterpart are dropped.
      - It does not share access. Each gateway calls Bedrock with the one AWS account it is configured with.
---

<HomeIndex />

<HomeSteps />

<SplitBlock proof="routes">

## The OpenAI routes your clients already use

Each route takes the request an OpenAI client sends and answers in the same shape, streaming included. Features that
exist only on Bedrock, such as the 1-hour prompt cache, go in an `extra_body` object, which the OpenAI SDKs can add to
any request.

[API reference](reference/api.md) · [Protocol compatibility](../openai-protocol-compatibility.md)

</SplitBlock>

<SplitBlock proof="clients" flip>

## Your client stays as it is

The OpenAI SDKs take the gateway as their base URL. Codex CLI takes it as a custom model provider that speaks
Responses. Editors and agents with an OpenAI-compatible provider take a base URL, a key and a model ID, with no plugin
and no patched client.

[Clients](clients/index.md) · [Codex CLI](clients/codex.md)

</SplitBlock>

<SplitBlock proof="models">

## Bedrock's models, by the names you know

Claude, OpenAI's GPT models on Bedrock, Amazon Nova, DeepSeek and every other model in your catalog are called the same
way. Each model's quirks, such as which sampling parameters it accepts or how it reasons, live in a configuration file,
so a new model needs an entry there and no new release.

[Models](guide/models.md) · [Caching and reasoning](guide/caching-and-reasoning.md)

</SplitBlock>

<HomePlatforms />

<HomePrivacy />

## Install

::: code-group

```sh [Script]
curl -fsSL https://raw.githubusercontent.com/sunerpy/bedrock-gateway-rust/main/scripts/install.sh | sh
```

```sh [Docker]
docker run -p 8080:8080 -e API_KEY -e AWS_REGION=us-east-1 -e AWS_BEARER_TOKEN_BEDROCK sunerpy/bedrock-gateway-rust
```

```sh [Binary]
gh release download --repo sunerpy/bedrock-gateway-rust --pattern '*-x86_64-unknown-linux-musl.tar.gz'
tar -xzf bedrock-gateway-*-x86_64-unknown-linux-musl.tar.gz
```

```sh [Cargo]
cargo install bedrock-gateway-rust
```

:::

The [install guide](guide/install.md) covers every platform, the image tags and building from source.

<HomeScope />

## Feedback

Bug reports and feature requests go to [GitHub Issues](https://github.com/sunerpy/bedrock-gateway-rust/issues).
bedrock-gateway is MIT-0 licensed and is not an AWS product.
