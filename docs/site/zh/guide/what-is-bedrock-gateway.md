# 什么是 bedrock-gateway

bedrock-gateway 让为 OpenAI API 编写的软件能够使用 Amazon Bedrock 中的模型。它是一个运行在你自己环境中的程序：客户端向它发送 OpenAI 请求，它用你的 AWS 账号调用 Bedrock，再以 OpenAI 格式返回回答。

## 用途

许多工具和库只支持 OpenAI API，例如 OpenAI SDK、Codex CLI、带编辑预测功能的编辑器，以及允许设置基础 URL 的 Agent。bedrock-gateway 为它们提供一个 OpenAI 端点，端点背后是 Claude、Bedrock 上的 OpenAI GPT 模型、Amazon Nova、DeepSeek，以及你的 Bedrock 模型目录中的其他模型。

客户端只需修改基础 URL 和 API 密钥这两项设置，其余代码保持不变。

## 一个请求如何流转

1. 客户端向网关发送一个 OpenAI 请求，例如 Chat Completions 或 Responses 请求，并带上你为客户端设定的密钥。
2. 网关校验密钥，把请求转换为该模型在 Bedrock 上所需的形式，再用你的 AWS 凭据发送出去。
3. Bedrock 返回回答。网关把回答或事件流转换回客户端所请求的 OpenAI 格式。

这里涉及两种凭据，方向正好相反：

| 凭据                        | 发送方     | 接收方  |
| --------------------------- | ---------- | ------- |
| `API_KEY`                   | 你的客户端 | 网关    |
| Bedrock API 密钥或 AWS 凭据 | 网关       | Bedrock |

## 支持的功能

- OpenAI 路由 `/chat/completions`、`/responses`、`/completions`、`/embeddings` 和 `/models`，支持流式与非流式。每个路由的说明见 [API 参考](../reference/api.md)。
- 工具、图片、JSON 输出和推理强度，在支持它们的模型上可用。
- 在支持提示缓存的模型上自动使用提示缓存，客户端无需改动。
- Bedrock 的跨区域推理配置文件，例如 `global.anthropic.claude-sonnet-5-5`。
- 位于配置文件中的模型注册表。需要特殊处理的模型在其中添加一个条目即可，网关无需发布新版本。

## 运行环境

bedrock-gateway 是一个没有运行时依赖的二进制文件，可以在 Linux、macOS 和 Windows 上运行，同一个二进制文件也打包成了体积很小的容器镜像。在你自己的机器、使用 Fargate 的 Amazon ECS 以及 AWS Lambda 上运行它的方法，见[部署指南](../operate/index.md)。

## 它不是什么

- 它不是托管服务。它由你自己运行，使用的是你的 AWS 账号和 Bedrock 配额。
- 它不保存对话。Responses API 以无状态方式提供。
- 它不是 AWS 的产品。

下一步：[快速开始](quick-start.md)。
