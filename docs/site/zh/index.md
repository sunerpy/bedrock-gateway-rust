---
layout: home
title: "bedrock-gateway：通过 OpenAI 接口使用 Amazon Bedrock"
titleTemplate: false
description: bedrock-gateway 是一个 Rust 二进制文件，通过 OpenAI 的 Chat Completions、Responses、Completions 和 Embeddings 接口提供 Amazon Bedrock 模型，OpenAI SDK、Codex CLI 和其他 Agent 只需配置一个基础 URL 和一个密钥。

hero:
  name: bedrock-gateway
  text: 通过 OpenAI 接口使用 Amazon Bedrock
  tagline: 一个 Rust 二进制文件，用你自己的 AWS 账号调用 Bedrock，响应 OpenAI 的 Chat Completions、Responses、Completions 和 Embeddings 请求。OpenAI SDK 或 Agent 只需配置一个基础 URL 和一个密钥，其余一切保持不变。
  actions:
    - theme: brand
      text: 安装
      link: /zh/guide/install
    - theme: alt
      text: 快速开始
      link: /zh/guide/quick-start
    - theme: alt
      text: GitHub
      link: https://github.com/sunerpy/bedrock-gateway-rust

home:
  facts:
    - term: 运行方式
      text: 适用于 Linux、macOS 和 Windows 的二进制文件，或适用于 Linux 的容器镜像，可以运行在你自己的机器、ECS 或 Lambda 上，监听 8080 端口。
    - term: 你的 AWS 账号
      text: 使用 Bedrock API 密钥或标准 AWS 凭据链调用 Bedrock。提示词发往你账号中的 Bedrock。

  visual:
    label: 一段终端会话。一个请求发往 Chat Completions 路由，另一个发往 Responses 路由，两者都使用 Amazon Nova 2 Lite，网关对两者都返回 BEDROCK_OK。
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
    title: bedrock-gateway 能做什么
    intro: 网关配置好客户端密钥和 AWS 凭据后，下列功能即可使用；GPT-5.x 和 gpt-oss 还需要 Bedrock API 密钥。Chat Completions 上的“跨工具调用保留推理”和 OpenTelemetry 导出默认关闭，需要你自行开启。
    groups:
      - name: 接口
        items:
          - title: Chat Completions
            body: 流式与非流式对话，支持工具、图片、JSON 输出和推理强度。
            status: available
            link: /zh/reference/api#chat-completions
          - title: Responses
            body: Codex CLI 使用的 Responses API，支持流式与非流式，请求之间不保存状态。
            status: available
            link: /zh/reference/api#responses
          - title: Completions
            body: 旧版文本补全路由，供编辑预测功能仍在使用它的编辑器调用。
            status: available
            link: /zh/reference/api#completions
          - title: Embeddings
            body: 通过 OpenAI 的 embeddings 请求使用 Cohere、Titan 和 Nova 嵌入模型。
            status: available
            link: /zh/reference/api#embeddings
          - title: 模型列表
            body: GET /api/v1/models 列出你的账号可以调用的模型和推理配置文件，内容从 Bedrock 读取，并在网关运行期间定期刷新。
            status: available
            link: /zh/guide/models
      - name: 模型
        items:
          - title: Claude
            body: Sonnet、Opus、Haiku 和 Fable，通过 Bedrock 模型 ID 和跨区域推理配置文件调用。
            status: available
            link: /zh/guide/models#claude
          - title: Bedrock 上的 OpenAI GPT
            body: GPT-5.x、GPT-6.x 和 gpt-oss，使用 gpt-5.6-sol、gpt-6.1-sol 这样的简称调用。
            status: available
            link: /zh/guide/models#openai-gpt
          - title: 其他所有 Bedrock 模型
            body: Amazon Nova、DeepSeek 以及你账号模型目录中的任何模型。支持新模型只需添加一个配置条目，不需要发布新版本。
            status: available
            link: /zh/guide/models#其他模型
      - name: 行为
        items:
          - title: 提示缓存
            body: 在支持的模型上，为工具、系统提示和消息放置缓存点，客户端无需任何改动。
            status: available
            link: /zh/guide/caching-and-reasoning#提示缓存
          - title: 推理强度
            body: reasoning_effort 会转换为每个模型需要的形式，从 Claude 的自适应思考到 GPT 的 reasoning.effort。
            status: available
            link: /zh/guide/caching-and-reasoning#推理
          - title: 跨工具调用保留推理
            body: 在 Chat Completions 的工具调用及其结果之间保留模型带签名的推理内容，需要你自己提供签名密钥。
            status: opt-in
            link: /zh/guide/configuration#跨工具调用保留推理
      - name: 运维
        items:
          - title: 多种部署方式
            body: 二进制文件、Docker 镜像、负载均衡器后面运行在 Fargate 上的 ECS，或者使用函数 URL 的 Lambda。
            status: available
            link: /zh/operate/
          - title: 密钥保存在 AWS 中
            body: 客户端密钥可以来自 SSM Parameter Store 或 Secrets Manager，而不是环境变量。
            status: available
            link: /zh/guide/configuration#客户端发送的密钥
          - title: 日志与追踪
            body: 每个请求一行日志，记录模型、token 数、缓存命中和耗时，从不记录提示词或密钥。OpenTelemetry 导出是一个构建选项。
            status: opt-in
            link: /zh/operate/observability

  steps:
    title: 从安装到第一个回答
    items:
      - title: 拉取镜像
        command: docker pull sunerpy/bedrock-gateway-rust
        body: 也可以从 GitHub Releases 下载适合你平台的二进制文件，或者用 cargo 安装。
      - title: 为客户端设定密钥
        command: export API_KEY=sk-replace-with-a-private-random-key
        body: 每个客户端都要发送这个密钥。它由你自己决定，不是 AWS 凭据。
      - title: 启动网关
        command: docker run -p 8080:8080 -e API_KEY -e AWS_REGION=us-east-1 -e AWS_BEARER_TOKEN_BEDROCK sunerpy/bedrock-gateway-rust
        body: AWS_BEARER_TOKEN_BEDROCK 是 Bedrock API 密钥。没有它时，网关使用标准 AWS 凭据链。
      - title: 连接客户端
        command: http://localhost:8080/api/v1
        body: 这是 OpenAI SDK、Codex CLI 和其他 OpenAI 兼容客户端使用的基础 URL。

  routes:
    columns: [路由, 协议, 说明]
    rows:
      - cells: [POST /api/v1/chat/completions, OpenAI Chat Completions, 流式与非流式]
      - cells: [POST /api/v1/responses, OpenAI Responses, 无状态]
      - cells: [POST /api/v1/completions, OpenAI Completions, 旧版文本补全]
      - cells: [POST /api/v1/embeddings, OpenAI Embeddings, Cohere、Titan 和 Nova]
      - cells: [GET /api/v1/models, 模型列表, 从 Bedrock 读取]
      - cells: [GET /api/v1/health, 存活检查, 无需密钥]
    code: [0]
    caption: 除 /health 外，每个路由都要求提供你的 API_KEY，放在 Authorization Bearer 中发送。用 API_ROUTE_PREFIX 可以修改 /api/v1 这个前缀。

  clients:
    columns: [客户端, 接口, 配置方式]
    rows:
      - cells: [OpenAI Python 和 Node SDK, Chat Completions 或 Responses, 一个基础 URL 和一个 API 密钥]
      - cells: [Codex CLI, Responses, config.toml 中的 model_provider]
      - cells: [编辑器与 Agent, Chat Completions 或 Completions, 一个 OpenAI 兼容的 provider]
      - cells: [curl 和脚本, 任意路由, Authorization 请求头]

  models:
    columns: [模型系列, 模型 ID, 提供方式]
    rows:
      - cells: [Claude, global.anthropic.claude-sonnet-5-5 us.anthropic.claude-opus-5-5 global.anthropic.claude-fable-5-1, Bedrock Converse]
      - cells: [GPT-5.x, gpt-5.4 gpt-5.5 gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna, Bedrock 的 OpenAI 兼容端点]
      - cells: [GPT-6.x, gpt-6.1-sol gpt-6-sol gpt-6-luna gpt-6-astra, Bedrock Converse]
      - cells: [gpt-oss, gpt-oss-120b gpt-oss-20b, Bedrock 的 OpenAI 兼容端点]
      - cells: [Amazon Nova, us.amazon.nova-2-lite-v1:0 us.amazon.nova-pro-v1:0, Bedrock Converse]
    code: [1]
    caption: DeepSeek 和你模型目录中的其他所有模型都以同样的方式调用。模型列表会显示你的账号在网关所在区域可以调用的 ID。

  platforms:
    title: 支持的平台
    intro: 每个版本都为下列平台各提供一个二进制文件，在 Docker Hub 和 Amazon ECR Public 上发布多架构镜像，并在 crates.io 上发布 crate。
    columns: [平台, 发布文件, 容器镜像]
    rows:
      - name: Linux x64
        status: available
        cells: [x86_64-unknown-linux-musl, linux/amd64]
      - name: Linux ARM64
        status: available
        cells: [aarch64-unknown-linux-musl, linux/arm64]
      - name: macOS Intel
        status: available
        cells: [x86_64-apple-darwin, 无]
      - name: macOS Apple Silicon
        status: available
        cells: [aarch64-apple-darwin, 无]
      - name: Windows x64
        status: available
        cells: [x86_64-pc-windows-msvc, 无]
    note: Linux 二进制文件是静态链接的；镜像基于 distroless，只包含二进制文件和它的配置。

  privacy:
    title: 数据发往哪里
    intro: bedrock-gateway 位于你的客户端和你的 AWS 账号之间。除了客户端发来的图片 URL 和你配置的 OpenTelemetry 收集器，它只与 AWS 通信。
    sendsLabel: 发往
    modes:
      - name: 模型请求
        sends: 你账号中的 Amazon Bedrock
        detail: 每个请求的提示词、工具和图片发往你所配置区域的 Bedrock。日志记录模型、token 数、缓存命中和耗时，从不记录提示词、回答或密钥。
      - name: 模型列表
        sends: Bedrock 控制面
        detail: 网关在启动时以及之后定期列出你的账号可以调用的基础模型和推理配置文件。
      - name: 图片 URL
        sends: URL 中的主机
        detail: 消息中带有 http 或 https 图片 URL 时，网关会下载图片，再把图片数据发给 Bedrock。data URL 在网关内解码。
      - name: 客户端密钥
        sends: SSM 或 Secrets Manager（如果你使用它们）
        detail: 密钥来自 Parameter Store 或 Secrets Manager 时，网关在启动时从那里读取一次；否则密钥只存在于环境变量中。

  scope:
    title: 它不做什么
    items:
      - 不保存对话。Responses API 以无状态方式提供，因此 previous_response_id 和 store 会被接受，但不起作用。
      - 不提供 Anthropic Messages 接口。Claude 通过 OpenAI 接口提供。
      - 不运行联网搜索、文件搜索这类托管工具。客户端的函数工具会原样传递；没有 Bedrock 对应功能的托管工具会被丢弃。
      - 不共享访问权限。每个网关只用为它配置的那一个 AWS 账号调用 Bedrock。
---

<HomeIndex />

<HomeSteps />

<SplitBlock proof="routes">

## 客户端已在使用的 OpenAI 路由

每个路由都接收 OpenAI 客户端发送的请求，并以同样的格式返回，流式响应也是如此。Bedrock 独有的功能（例如 1 小时提示缓存）放在一个 `extra_body` 对象中，OpenAI SDK 可以在任何请求中加入这个对象。

[API 参考](reference/api.md) · [协议兼容性](../../openai-protocol-compatibility.md)

</SplitBlock>

<SplitBlock proof="clients" flip>

## 客户端无需改动

OpenAI SDK 把网关设为基础 URL 即可。Codex CLI 把它配置为一个使用 Responses 的自定义 model provider。支持 OpenAI 兼容 provider 的编辑器和 Agent 只需填写基础 URL、密钥和模型 ID，不需要插件，也不需要修改客户端。

[客户端](clients/index.md) · [Codex CLI](clients/codex.md)

</SplitBlock>

<SplitBlock proof="models">

## 用熟悉的名称调用 Bedrock 模型

Claude、Bedrock 上的 OpenAI GPT 模型、Amazon Nova、DeepSeek 以及你模型目录中的其他所有模型，调用方式都相同。每个模型的特殊之处，例如接受哪些采样参数、如何推理，都记录在配置文件中，因此支持一个新模型只需在其中添加一个条目，不需要发布新版本。

[模型](guide/models.md) · [缓存与推理](guide/caching-and-reasoning.md)

</SplitBlock>

<HomePlatforms />

<HomePrivacy />

## 安装

::: code-group

```sh [Docker]
docker run -p 8080:8080 -e API_KEY -e AWS_REGION=us-east-1 -e AWS_BEARER_TOKEN_BEDROCK sunerpy/bedrock-gateway-rust
```

```sh [二进制文件]
gh release download --repo sunerpy/bedrock-gateway-rust --pattern '*-x86_64-unknown-linux-musl.tar.gz'
tar -xzf bedrock-gateway-*-x86_64-unknown-linux-musl.tar.gz
```

```sh [Cargo]
cargo install bedrock-gateway-rust
```

:::

[安装指南](guide/install.md)介绍了所有平台、镜像标签以及如何从源码构建。

<HomeScope />

## 反馈

问题报告和功能建议请提交到 [GitHub Issues](https://github.com/sunerpy/bedrock-gateway-rust/issues)。bedrock-gateway 以 MIT-0 许可证发布，不是 AWS 的产品。
