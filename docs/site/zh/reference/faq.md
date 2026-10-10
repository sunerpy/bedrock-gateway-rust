# 常见问题

## 请求

### 请求返回 401 但密钥是正确的

Bedrock 拒绝网关自身的 AWS 身份访问模型时，网关同样返回 401，错误信息与密钥错误时相同。请检查网关运行时使用的角色或用户是否具备该模型及其区域所需的[权限](../operate/index.md#所需的-aws-权限)，并在 Bedrock 控制台中确认该模型已为你的账号启用。两种情况下，日志都把这个失败记录为 `unauthorized`。

### 模型不在模型列表中

列表中是 Bedrock 针对网关所在区域报告的模型，因此请按以下顺序检查：

- `AWS_REGION` 是否是你预期的区域，以及该模型在那里是否对你的账号可用；
- `ENABLE_CROSS_REGION_INFERENCE` 是否没有设为 `false`，因为大多数较新的模型只以 `us.`、`eu.` 或 `global.` 推理配置文件的形式存在；
- `ALLOWED_MODELS` 是否把它排除在外；
- 模型上线之后，网关是否已经重新读取过列表。网关每隔 `MODEL_CATALOG_REFRESH_SECS`（默认一小时）或在重启时重新读取。

对列表之外的模型发出的请求仍会发往 Bedrock，但网关会拒绝其中的图片：网关是根据这个列表判断哪些模型接受图片的。

### GPT-5.x 返回 400

错误信息会说明属于以下哪种情况：

- `requires a Bedrock API key`：GPT-5.x 和 gpt-oss 使用 Bedrock 的 OpenAI 兼容端点，这个端点只接受 Bedrock API 密钥。请设置 `AWS_BEARER_TOKEN_BEDROCK` 并重启。
- `is not available in region`：该模型只在少数区域提供。请在其中一个区域运行网关，见[模型](../guide/models.md#openai-gpt)。
- `is only available on the /responses endpoint` 或 `on the /chat/completions endpoint`：Completions 路由不提供 GPT-5.x 和 gpt-oss。请把请求发到 Chat Completions 或 Responses。

### `response_format` 或 `text.format` 返回 400

网关对所有模型都会把结构化输出发给 Bedrock，所以这个 400 来自 Bedrock。出现的情况有两种：一是模型不支持结构化输出，例如 Amazon Nova 和 Meta Llama（报 `This model doesn't support the outputConfig field`）以及 Claude Fable 5；二是模型无法按这个 schema 约束输出，例如 Claude 要求每个对象都设置 `additionalProperties: false`。遇到这类模型，请去掉 `response_format`，改在 system 提示中要求输出一个 JSON 对象。

### 请求带有工具时 Claude 的回答没有推理内容

在 Chat Completions 上，推理模型在工具调用之后需要拿回它带签名的推理内容，而 Chat Completions 格式中没有存放它的位置。因此，没有签名密钥时，网关会在关闭推理的情况下处理带工具的 Claude 请求。请开启[签名密钥](../guide/configuration.md#跨工具调用保留推理)，或者改用 Responses API，它自己就能携带推理内容。

### HTTP 413

请求体超过了 `MAX_BODY_SIZE_MB`（默认 20 MB）。base64 图片也计入这个大小，带有大量截图的长会话可能会达到上限。请提高上限，或者以 URL 的形式发送图片。

### HTTP 429

Bedrock 对请求进行了限流，网关重试了 `AWS_MAX_RETRY_ATTEMPTS` 次仍未成功。Bedrock 的配额按账号、区域和模型计算。跨区域或 `global.` 推理配置文件可以把请求分散到更多容量上；Service Quotas 控制台中可以查看限额，也可以申请提高。

## 流式响应

### 流在几秒或几分钟后中断

客户端和网关之间的代理或负载均衡器在流没有数据时关闭了连接。请把它的空闲超时提高到超过模型回答中最长的停顿；负载均衡器和 ECS Service Connect 的相应设置见[运维](../operate/index.md#流式响应与超时)。

### Chat Completions 流结束时没有用量信息

请用 `stream_options: {"include_usage": true}` 请求用量信息；网关随后会像 OpenAI 一样，在 `data: [DONE]` 之前发送一个用量 chunk。

## 缓存

### 缓存始终不命中

如果缓存点之前的提示在两次请求之间发生了变化（例如系统提示中带有时间戳），或者它短于模型的最小值，`prompt_tokens_details.cached_tokens` 就会一直为 0。网关在哪里放置缓存点，见[缓存与推理](../guide/caching-and-reasoning.md#提示缓存)。

## 运行

### 可以同时运行多个网关吗？

可以。网关不保存状态，因此一个负载均衡器后面可以运行任意数量的网关。请为它们配置相同的 `API_KEY`；开启签名密钥时，还要配置相同的 `CHAT_REASONING_CAPSULE_KEYS`。

### 它会保存对话吗？

不会。回答发出之后，只留下它的日志行，其中记录模型和 token 数，不含任何内容。Responses API 以无状态方式提供，客户端每次请求都发送完整对话。见[隐私](../privacy.md)。

### 它提供 Anthropic Messages API 吗？

不提供。Claude 通过 OpenAI 接口提供，因此只支持 Anthropic Messages 的客户端（例如 Claude Code）无法使用网关。

### 它是 AWS 的项目吗？

不是。它最初是 [aws-samples/bedrock-access-gateway](https://github.com/aws-samples/bedrock-access-gateway) 的 Rust 重写版，并沿用了它的 MIT-0 许可证，但它是独立维护的，AWS 不为它提供支持。
