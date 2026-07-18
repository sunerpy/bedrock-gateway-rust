# GPT Responses 转 Chat Completions 适配

## 背景

AWS Bedrock Mantle 上的 `openai.gpt-5.4`、`openai.gpt-5.5` 和
`openai.gpt-5.6-{sol,terra,luna}` 只接受 Responses API。直接请求 Mantle 的
`/v1/chat/completions` 会返回 400，因此本项目不能通过改写 URL 获得 Chat 支持。

项目使用配置项 `chat_backend = "responses"` 选择内部协议适配器：

```text
客户端 /chat/completions
  -> ResponsesChatProvider
  -> 现有 CompositeResponsesProvider
  -> Mantle /responses
  -> ResponsesChatProvider
  -> 标准 Chat Completions 响应或 SSE
```

没有按模型名写 Rust 分支。新增同类模型时，只需在 `config/models.toml` 同时声明：

```toml
responses_backend = "mantle"
chat_backend = "responses"
```

## 支持范围

- 流式与非流式文本响应；
- `reasoning_effort` 到 `reasoning.effort` 的映射；
- reasoning summary 以 `<think>...</think>` 放入 Chat `content`；
- `output_tokens_details.reasoning_tokens` 到
  `completion_tokens_details.reasoning_tokens` 的映射；
- Chat function tools、自动或强制 `tool_choice`；
- 并行工具调用；
- 多轮 `assistant.tool_calls` 和 `tool` 结果回传；
- assistant 历史中的纯文本 content parts 数组会规范化为等价字符串；
- `response_format`、`max_completion_tokens`、图片输入和
  `stream_options.include_usage`。

明确限制：

- `n` 只允许 `1`；
- Responses API 没有等价 `stop` 语义，携带 `stop` 时返回 400；
- `/completions` 仍不支持这些 GPT 模型；
- 只能展示上游返回的 reasoning summary，不能展示原始思考链；
- 上游不返回 summary 时，不生成 `<think>` 内容。

## 无状态 reasoning 与工具续轮

工具续轮必须把上游的 `reasoning.encrypted_content` 原样交回 Responses API。标准
Chat 协议没有保存该字段的位置，客户端通常只会回传 `tool_calls[].id`。

适配器将以下内容封装进独立的 `rsc_v1` capsule：

- 原始 Responses `call_id`；
- 本轮完整 reasoning output items，包括 `id` 和 `encrypted_content`。

capsule 使用现有 `CHAT_REASONING_CAPSULE_*` keyring 做 HMAC 认证，与 Converse
路径的 `brtc_v1` 使用同一套密钥运维，但前缀和载荷格式互不兼容。续轮时网关从
标准 Chat 工具 ID 中恢复 reasoning item、function call 和 function output，全程不需要
Redis 或数据库。

reasoning 与工具同时启用时必须配置：

```bash
CHAT_REASONING_CAPSULE_ENABLED=true
CHAT_REASONING_CAPSULE_ACTIVE_KID=current
CHAT_REASONING_CAPSULE_KEYS=current:<base64url-no-pad-key>
```

如果上游已经产生 reasoning 和工具调用，但缺少可重放的 encrypted content，或 capsule
编码未启用，网关会失败关闭，不会返回一个下一轮必然失败的工具 ID。

## 流式顺序保证

Mantle Responses SSE 会经过增量帧解码，HTTP chunk 边界不被当作 SSE 边界。适配器保证：

1. 先发送 `role: assistant`；
2. summary 的 `<think>` 块完整关闭；
3. 每个工具的 `id` 和 `name` 只发送一次；
4. 后续 chunk 只发送 arguments delta；
5. 最后发送 finish reason、可选 usage，由 Chat 路由追加 `[DONE]`。

这避免了客户端在增量拼接时重复工具名、覆盖 capsule，或在 reasoning 尚未结束时收到工具
ID。

Mantle 当前会先发送 `data: {...}`，再发送同一 SSE frame 的 `event: ...`。终止事件观察器
必须保留已从 data 行解析出的 status 和 usage，不能用后到的 event 行空占位覆盖；对应顺序
已有回归测试。

OpenAI Responses 的 `EasyInputMessage` 允许 assistant 历史使用字符串或
`input_text` content 数组，但 Mantle 当前会拒绝后者。WorkBuddy 会把普通 assistant 文本
保存为 `content: [{ "type": "text", ... }]`；适配器必须仅对 assistant 的纯文本 parts
拼接并规范化为字符串。user/system/developer 的 typed input parts 保持不变，混合图片的
assistant content 也不做有损拼接。

## 为什么没有引入第三方库

LiteLLM 等代理项目可以在独立服务中提供多协议入口，但会引入另一套运行时、配置、错误
语义和部署面。OpenAI SDK 与 OpenAI Agents SDK 能分别调用 Chat 或 Responses，并不提供
一个可嵌入 Rust 网关、同时保留加密 reasoning 续轮状态的无损转换层。

本项目已经有两套 wire schema、Responses provider、Mantle 原始 SSE 通道和 capsule
keyring。内部实现只增加协议转换，不增加外部服务或状态存储，且可直接复用现有鉴权、区域
门控、日志和测试体系。

## 真实验证

2026-07-18 使用 TypeScript `openai` SDK 和 OpenAI Agents SDK 对本地源码网关完成验证；
网关通过代理连接真实 Mantle 上游：

- `gpt-5.4`、`gpt-5.5`、`gpt-5.6-sol/terra/luna` 均可通过 Chat 非流式调用；
- `gpt-5.6-sol` 流式 reasoning 与工具调用生成 `rsc_v1`，续轮成功且工具只执行一次；
- WorkBuddy 连续工具会话在第四次续轮加入 assistant 文本 parts 后仍可继续，不再触发
  Mantle 400；
- 工具 `id` 和 `name` 只发送一次，arguments 按 delta 增量发送；
- 客户端 usage 与 CloudWatch 均记录同一真实 reasoning token 数；
- OpenAI Agents SDK 的工具执行次数严格为 1，并正常生成最终回答。

US 环境将在临时镜像部署后单独验证；这一步不等同于 GitHub 发版。
