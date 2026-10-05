# 配置

bedrock-gateway 通过环境变量配置。本页按用途分组列出全部环境变量及其默认值。必需的只有两项：供客户端使用的密钥，以及对 Bedrock 的访问权限。

设置也可以写在工作目录下的 `config/app.toml` 中；像 `PORT` 这样只有一个单词的变量名，也可以通过 `APP_PORT` 设置。本页列出的变量优先于这两种方式。

## 客户端发送的密钥

除 `/health` 外，每个路由都要求以 `Authorization: Bearer <key>` 的形式提供这个密钥。没有密钥时网关拒绝启动。网关按以下顺序查找三个来源，使用第一个已设置的来源：

| 变量                 | 密钥来源                                                          |
| -------------------- | ----------------------------------------------------------------- |
| `API_KEY_PARAM_NAME` | SSM Parameter Store 中以此为名称的参数，读取时解密                |
| `API_KEY_SECRET_ARN` | Secrets Manager 中的一个 secret，其值为带有 `api_key` 字段的 JSON |
| `API_KEY`            | 该变量本身的值                                                    |

网关只在启动时读取一次参数或 secret。更换密钥时，先更新它，再重启网关。

## 访问 Bedrock

| 变量                       | 默认值      | 含义                                                      |
| -------------------------- | ----------- | --------------------------------------------------------- |
| `AWS_REGION`               | `us-west-2` | 网关调用的 Bedrock 端点所在的区域                         |
| `AWS_BEARER_TOKEN_BEDROCK` | 无          | Bedrock API 密钥。也可以用 `BEDROCK_API_KEY` 这个名称设置 |

设置了 Bedrock API 密钥时，网关用它向 Bedrock 进行身份验证。没有设置时，网关使用标准的 AWS 凭据链：先是 `AWS_ACCESS_KEY_ID` 和 `AWS_SECRET_ACCESS_KEY`，然后是 `AWS_PROFILE`，最后是 EC2 实例、ECS 任务或 Lambda 函数的角色。这两个方向彼此独立：`API_KEY` 是客户端发给网关的，Bedrock API 密钥是网关发给 AWS 的。

GPT-5.x 和 gpt-oss 通过 Bedrock 的 OpenAI 兼容端点提供，这个端点只接受 Bedrock API 密钥。没有 Bedrock API 密钥时，网关照常启动并记录一条警告，这些模型的请求会返回错误，其他模型不受影响。

| 变量                            | 默认值                                              | 含义                                                                 |
| ------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------- |
| `DISABLE_MANTLE`                | `false`                                             | 关闭 OpenAI 兼容端点及其模型                                         |
| `MANTLE_BASE_URL_TEMPLATE`      | `https://bedrock-mantle.{region}.api.aws/openai/v1` | GPT-5.x 的 Responses 请求发往的地址；`{region}` 取 `AWS_REGION` 的值 |
| `MANTLE_CHAT_BASE_URL_TEMPLATE` | `https://bedrock-mantle.{region}.api.aws/v1`        | gpt-oss 的 Chat Completions 请求发往的地址                           |

## 服务器

| 变量               | 默认值    | 含义                                          |
| ------------------ | --------- | --------------------------------------------- |
| `PORT`             | `8080`    | 网关监听的端口                                |
| `BIND_ADDR`        | `0.0.0.0` | 绑定的地址；设为 `127.0.0.1` 时只接受本机连接 |
| `API_ROUTE_PREFIX` | `/api/v1` | 所有路由所在的路径                            |
| `MAX_BODY_SIZE_MB` | `20`      | 接受的最大请求体，足以容纳 base64 编码的图片  |

网关只提供明文 HTTP。客户端通过网络访问它时，请在它前面放置一个启用 TLS 的负载均衡器或反向代理。

## 模型

| 变量                                    | 默认值                                      | 含义                                                                   |
| --------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| `ENABLE_CROSS_REGION_INFERENCE`         | `true`                                      | 列出 Bedrock 的跨区域推理配置文件，例如以 `us.` 和 `global.` 开头的 ID |
| `ENABLE_APPLICATION_INFERENCE_PROFILES` | `true`                                      | 列出你账号中的应用程序推理配置文件                                     |
| `ALLOWED_MODELS`                        | 全部                                        | 以逗号分隔的 ID 片段列表；`/models` 只列出包含其中某个片段的 ID        |
| `MODEL_CATALOG_REFRESH_SECS`            | `3600`                                      | 多久从 Bedrock 重新读取一次模型列表；`0` 表示只在启动时读取            |
| `DEFAULT_MODEL`                         | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Bedrock 没有返回网关能够提供的模型时，`/models` 列出的 ID              |
| `CONFIG_DIR`                            | `config`                                    | 存放你自己的 `models.toml`、`regions.toml` 和 `embeddings.toml` 的目录 |

`ALLOWED_MODELS` 只缩小 `/models` 列出的范围。它不会拦截对其他模型的请求，但网关会拒绝这类请求中的图片，因为网关是根据这个列表判断哪些模型接受图片的。模型 ID 和模型注册表的说明见[模型](models.md)。

## 提示缓存

| 变量                    | 默认值 | 含义                                                  |
| ----------------------- | ------ | ----------------------------------------------------- |
| `ENABLE_PROMPT_CACHING` | `true` | 在支持缓存的模型上自动放置缓存点                      |
| `PROMPT_CACHE_TTL`      | `5m`   | 缓存条目的存活时间：`5m`，或者在支持的模型上设为 `1h` |

单个请求可以通过 `extra_body` 覆盖这两项，见[缓存与推理](caching-and-reasoning.md#提示缓存)。

## 跨工具调用保留推理

Claude 的思考内容和 GPT 的推理内容都附带一个签名，对话在工具调用之后继续时，模型需要再次看到这个签名。Chat Completions 格式中没有存放它的字段，因此网关可以把它放进工具调用的 ID 中，并用你自己的密钥签名，使其无法伪造。Responses API 用自己的条目携带推理内容，不需要这项功能。

没有签名密钥时，Chat Completions 上带工具的 Claude 请求会关闭推理后运行，请求仍然成功。Chat Completions 上同时包含推理和工具调用的 GPT-5.x 回答则会失败，错误信息中写明 `CHAT_REASONING_CAPSULE_ENABLED`。如果你的客户端在 Chat Completions 上使用工具，请开启这项功能。

| 变量                                | 默认值  | 含义                                                              |
| ----------------------------------- | ------- | ----------------------------------------------------------------- |
| `CHAT_REASONING_CAPSULE_ENABLED`    | `false` | 把推理内容签名后放入工具调用 ID                                   |
| `CHAT_REASONING_CAPSULE_ACTIVE_KID` | 无      | 为新的工具调用 ID 签名的密钥的 ID                                 |
| `CHAT_REASONING_CAPSULE_KEYS`       | 无      | 以逗号分隔的 `kid:key` 对；每个密钥都是不带填充的 URL 安全 base64 |

生成一个密钥并开启这项功能：

```sh
export CHAT_REASONING_CAPSULE_KEYS="k1:$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=')"
export CHAT_REASONING_CAPSULE_ACTIVE_KID=k1
export CHAT_REASONING_CAPSULE_ENABLED=true
```

同一个负载均衡器后面的所有实例都需要配置相同的密钥。更换密钥时，先添加新的密钥对并把它设为当前密钥，等用到旧密钥的对话都结束后，再删除旧的密钥对。开启这项功能而当前密钥缺失时，网关拒绝启动。

## 超时与重试

| 变量                                 | 默认值 | 含义                                                          |
| ------------------------------------ | ------ | ------------------------------------------------------------- |
| `AWS_CONNECT_TIMEOUT_SECS`           | `60`   | 等待与 AWS 建立连接的时长                                     |
| `AWS_READ_TIMEOUT_SECS`              | `900`  | 等待 AWS 返回数据的时长，足以覆盖较长的流式响应               |
| `RESPONSES_STREAM_IDLE_TIMEOUT_SECS` | `180`  | Responses 流最多可以静默多久，超过后以 `response.failed` 结束 |
| `AWS_MAX_RETRY_ATTEMPTS`             | `8`    | 对 AWS 的调用被限流或失败时最多尝试几次                       |

## 日志

| 变量                          | 默认值  | 含义                                                                            |
| ----------------------------- | ------- | ------------------------------------------------------------------------------- |
| `LOG_LEVEL`                   | `info`  | `error`、`warn`、`info`、`debug` 或 `trace`，也可以是 `RUST_LOG` 格式的过滤指令 |
| `DEBUG`                       | `false` | 以易读的文本输出日志，而不是每行一个 JSON 对象                                  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | 无      | 把追踪和指标发送到这个 OTLP 端点；需要用 `--features otel` 构建                 |

日志会记录哪些内容，见[可观测性](../operate/observability.md)。
