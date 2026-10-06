# 可观测性

网关在日志中记录它做的事，每个事件一行，还可以通过 OpenTelemetry 导出指标和追踪。两者都从不包含提示词、回答、工具参数、图片或密钥。

## 日志

标准输出上的每一行都是一个 JSON 对象，包含 `timestamp`、`level`、`message`、`target` 以及该事件的各个字段。`DEBUG=true` 会把同样的事件改为输出易读的文本，便于在终端中阅读。

`LOG_LEVEL` 决定写入多少内容：`error`、`warn`、`info`（默认）、`debug` 或 `trace`。它也接受 `RUST_LOG` 格式的过滤指令，例如 `info,tower_http=debug`；设置了 `RUST_LOG` 时，以它为准。

### 每个请求写入的日志

在 `info` 级别，每个请求在到达时写一行，结束时再写一行：

| 事件                                                     | 字段                                                                                                                         |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `chat request received`、`responses request received` 等 | `request_id`、`model`、`stream`；Responses 还会加上 `reasoning_effort`                                                       |
| `chat completed`、`chat streaming completed`             | `prompt_tokens`、`completion_tokens`、`total_tokens`、`cached_tokens`、`cache_hit` 以及耗时                                  |
| `responses completed`、`responses streaming completed`   | `input_tokens`、`output_tokens`、`total_tokens`、`cached_tokens`、`cache_hit`、`reasoning_used`、`reasoning_tokens` 以及耗时 |
| `chat request failed`、`completions request failed` 等   | `request_id`、`model` 以及错误                                                                                               |

非流式回答的耗时记录为 `latency_ms`，流式回答记录为 `duration_ms`。非流式的 chat 回答还会记录 `finish_reason`，Responses 回答还会记录 `status`。

由 Bedrock 的 OpenAI 兼容端点提供的 GPT-5.x 和 gpt-oss，日志的记录方式不同。GPT-5.x 的流以 `responses raw streaming completed` 结束，其中带有 `input_tokens` 和 `output_tokens`，请求经由 Chat Completions 发来时也是如此。Chat Completions 上的 gpt-oss 请求按 Bedrock 发送的内容原样透传，它的日志行不带 token 数。

由客户端引起的失败，例如未知模型或格式错误的请求，记录为 `warn`；网关或 Bedrock 一侧的失败记录为 `error`。

`request_id` 把同一个请求的各行日志关联起来。客户端发送了 `x-request-id` 请求头时，网关直接使用它，客户端因此可以在日志中找到自己的请求；否则网关会自行生成一个。

在 `debug` 级别，网关还会记录每个请求实际发往的模型 ID 和区域，HTTP 层也会记录每个请求的状态和延迟。`debug` 同样作用于 AWS SDK 和各个 HTTP 库，日志会变得很长；使用 `info,bedrock_gateway_rust=debug` 时，只会增加网关自身的日志行。

### 在 AWS 上

在 ECS 和 Lambda 上，标准输出会进入 CloudWatch Logs。由于每一行都是 JSON，CloudWatch Logs Insights 可以直接读取其中的字段，例如统计每个 Converse 模型在 Chat Completions 上使用的 token：

```text
fields @timestamp, model, prompt_tokens, completion_tokens, cached_tokens
| filter message in ["chat completed", "chat streaming completed"]
| stats sum(prompt_tokens), sum(completion_tokens), sum(cached_tokens) by model
```

## OpenTelemetry

发布的二进制文件和镜像在构建时不带 OpenTelemetry。要通过 OTLP 导出，请在构建网关时启用 `otel` 功能：

```sh
cargo build --release --features otel
```

然后把它指向一个接受 HTTP protobuf 协议的 OTLP 收集器，通常在 4318 端口：

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

网关把追踪发送到该地址下的 `/v1/traces`，把指标发送到 `/v1/metrics`，服务名为 `bedrock-gateway`，并附带版本号。设置了这个变量后，标准的 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`、`OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` 和 `OTEL_EXPORTER_OTLP_HEADERS` 变量也会生效。不设置这个变量时，这样的构建不会导出任何内容。

### 指标

| 指标                        | 类型   | 记录内容                |
| --------------------------- | ------ | ----------------------- |
| `gateway.requests`          | 计数器 | 每个请求计一次          |
| `gateway.request.duration`  | 直方图 | 耗时，单位为毫秒        |
| `gateway.tokens.prompt`     | 直方图 | 每个请求的输入 token 数 |
| `gateway.tokens.completion` | 直方图 | 每个请求的输出 token 数 |

每个指标都带有两个标签：`model` 和 `finish_reason`（在 Responses 上为状态）。这些指标只针对 Chat Completions、Completions 和 Responses 上的非流式请求记录；流式请求只在日志中统计。

### 追踪

网关的请求 span 位于 `debug` 级别，而日志级别决定哪些 span 会产生，因此只有在日志级别放行时才会导出追踪。`LOG_LEVEL=info,tower_http=debug` 会为每个 HTTP 请求导出一个 span，包含方法、URI 和耗时，其余日志仍保持在 `info` 级别。
