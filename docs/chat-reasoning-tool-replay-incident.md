# Chat 推理与工具续轮事故记录

状态：实现已收敛，已通过本地测试及真实 Bedrock/WorkBuddy 验证。

日期：2026-07-17

关联历史方案：`docs/chat-reasoning-tool-replay.md`

历史方案保留原文用于追溯，不在本次修改范围内。历史方案最初描述的是未认证的
两段式 token；实际实现采用三段式 HMAC capsule。两者不一致时，以本文和代码为准。

## 一、结论

调试过程中出现过两个相互独立的问题：

1. **网关真实缺陷**：流式响应可能在签名推理块尚未整理完成时，先发送
   `tool_calls[].id`。
2. **本地转发链故障**：一次请求体中嵌套了另一份 HTTP 请求，并被错误的外层
   `Content-Length` 截断。

只有第一个问题需要修改网关业务代码。第二个问题发生在 Axum handler 收到请求之前，
不能通过兼容 BOM、gzip、UTF-16、Base64 等方式在应用层修复。

## 二、事故 A：推理块未完成就发送工具 ID

### 2.1 现象

WorkBuddy 5.2.6 拒绝流式工具调用：

```text
reasoning block was incomplete before the tool call id was emitted
```

### 2.2 根因

Claude 扩展思考返回的 Bedrock `reasoningContent` 不只是可见文本，还包含必须原样
回传的签名。工具续轮必须同时恢复：

- 原始推理文本和签名，或原始 `redactedContent`；
- 原始 Bedrock `toolUseId`；
- 推理块在 `toolUse` 之前的顺序。

OpenAI Chat Completions 没有与 Responses API `reasoning.encrypted_content` 等价的标准
字段。部分客户端只保留标准字段，因此额外增加非标准 sibling 字段不能可靠续传。

原流状态机在收到 `ContentBlockStart::ToolUse` 时立即发送工具 ID。真实 Claude 流中，
工具块开始前不一定先出现推理块的 `ContentBlockStop`。因此网关可能已经收到文本和
签名 delta，却还没有把它们整理为可重放的完整推理块。

工具 ID 一旦通过 SSE 发给客户端就不能撤回或替换，这正是 WorkBuddy 报错的直接原因。

### 2.3 修复方案

网关在标准 `tool_calls[].id` 中放入无状态、带认证的 `brtc_v1` capsule：

```text
brtc_v1.<base64url-no-pad(JSON payload)>.<base64url-no-pad(HMAC-SHA256)>
```

payload 保存：

- 原始 `toolUseId`；
- 完整、按顺序排列的签名推理块；
- 当前签名密钥的 `kid`；
- payload 与推理信封版本。

HMAC 覆盖 capsule 前缀和 payload。客户端可以透明保存并回传 capsule，但不能修改
工具 ID、推理文本、签名或 `kid` 后继续通过网关校验。

### 2.4 流式输出顺序

流式路径必须按以下顺序处理：

1. 累积同一 Bedrock content block 的推理文本、签名或 redacted bytes；
2. 正常收到 `ContentBlockStop` 时完成该推理块；
3. 如果工具块先开始，把工具边界视为推理块的隐式结束；
4. 若只收到推理文本但没有签名，立即失败，不能发送不可重放的工具 ID；
5. 推理块完整时生成 capsule，再发送首个 `tool_calls[].id`；
6. 实际发送 capsule 后，后续参数 delta 不再重复原始 Bedrock ID，避免客户端组装时
   覆盖 capsule；
7. 本轮没有签名推理时继续发送原始工具 ID，保持普通工具流的既有行为。

第 3、4、5 项直接修复 WorkBuddy 的报错。第 6、7 项用于把行为变化严格限制在真正
生成 capsule 的工具调用上。

### 2.5 续轮输入还原

客户端只需按标准 Chat 消息回传：

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "brtc_v1....",
      "type": "function",
      "function": {"name": "lookup", "arguments": "{}"}
    }
  ]
}
```

```json
{
  "role": "tool",
  "tool_call_id": "brtc_v1....",
  "content": "..."
}
```

网关在调用 Bedrock 前执行：

1. 校验 capsule 长度、结构、版本、`kid` 和 HMAC；
2. 还原原始签名推理块；
3. 把 assistant 文本中已经展示过的 `<think>...</think>` 前缀移除，避免把同一段
   推理同时作为普通文本和 `reasoningContent` 重复发送；
4. 在 `toolUse` 前插入原始推理块；
5. 把 assistant `toolUse` 和 user `toolResult` 的 ID 都恢复为原始 `toolUseId`。

网关不保存会话状态，不需要 Redis、数据库或粘性会话。

### 2.6 `tool_choice` 边界

“开启推理后不能使用工具”这个结论不正确。已验证可工作的组合是：

- 签名推理 + 工具定义 + 自动工具选择；
- 签名推理 + 工具历史续轮；
- 普通或 DeepSeek 字符串推理 + 现有工具选择行为。

受限的是 Claude 扩展思考与**强制**工具选择的组合：

- `tool_choice: "required"`；
- 指定某一个 function 的对象形式。

该组合在真实 Bedrock 上会被上游拒绝。网关只在模型使用需要签名重放的推理路径、
且本次确实启用了扩展思考时返回明确的 400；`auto`、未指定 `tool_choice`、DeepSeek
字符串推理和非推理请求不受影响。

## 三、事故 B：本地 HTTP 转发帧损坏

### 3.1 现象

后续一次 WorkBuddy 请求在进入 Chat handler 前失败：

```text
400 expected value at line 1 column 1
```

### 3.2 抓包证据

| 测量项 | 值 |
| --- | ---: |
| Hyper 按外层帧暴露的 body | 150524 字节 |
| 嵌套 HTTP headers | 1445 字节 |
| 内层 JSON 声明长度 | 153830 字节 |
| 实际暴露的内层 JSON | 149079 字节 |
| 缺失 JSON | 4751 字节 |

body 开头不是 `{`，而是另一条完整请求行：

```text
POST http://code-server:18080/api/v1/chat/completions HTTP/1.1
```

因此 JSON 在第 1 列解析失败是正确行为。Hyper 也只能遵守外层 `Content-Length`，
Axum 无法读取外层帧没有暴露的 4751 字节。

证据把问题定位在本地 WorkBuddy/网络转发链，而不是 Bedrock、Chat schema 或 capsule。
具体是哪一个本地转发组件偶发嵌套请求，尚未最终确认。

### 3.3 临时诊断方式

调试期间使用过 `/tmp` 下的原始 TCP 代理，用于观察外层长度之后的字节、拆出内层
HTTP 请求并转发干净 JSON。该代理只用于定位问题，不属于仓库，也不能部署到生产。

### 3.4 明确删除的误诊代码

以下尝试均已从网关代码中删除：

- UTF-8 BOM 兼容；
- gzip、Base64、UTF-16、percent-encoded、SSE 等格式猜测；
- 对畸形 body 的多轮扫描和诊断解码；
- 与本事故无关的推理预算行为修改。

删除原因：

- 抓包证明这些格式都不是本次原因；
- 应用层无法恢复 HTTP framing 已截断的字节；
- 对大 body 反复猜测解码会增加 CPU 和内存开销；
- 接受非标准猜测格式会扩大 OpenAI-compatible 接口的攻击面。

## 四、事故 C：普通工具流重复发送 ID 与名称

### 4.1 生产现象

`0.14.1` 部署到 US 后，WorkBuddy 连续显示同一个工具，例如：

```text
agent-browser
list pages
list pages
list pages
...
```

使用 TypeScript `openai` SDK 6.48.0 对生产接口复现时，一个 Bedrock
`lookup_weather` 工具块最终被 SDK 组装为：

```text
lookup_weatherlookup_weatherlookup_weatherlookup_weatherlookup_weather
```

这证明问题发生在网关的 OpenAI-compatible SSE 映射层，不是 WorkBuddy 客户端自行重复
发起工具。

### 4.2 根因

Bedrock 在 `contentBlockStart(toolUse)` 中提供一次 `toolUseId` 和工具名称，后续
`contentBlockDelta(toolUse)` 只提供参数片段。网关旧实现却在每个参数 delta 中重复发送
相同的 `id` 和 `function.name`。

OpenAI-compatible 客户端会按 `index` 拼接流式字段，因此正确协议应为：

1. start chunk 发送一次 `index + id + type + function.name`；
2. arguments chunk 只发送 `index + function.arguments`；
3. 后续 chunk 不得重复 `id` 或 `function.name`。

此前 capsule 路径为了避免把原始 Bedrock ID 拼到 `brtc_v1` 后面，已经会抑制后续
`id/name`；但没有签名推理块、因而没有生成 capsule 时，仍保留了旧的重复发送行为。
这使缺陷看起来与 capsule 偶发相关，实际是普通 Chat 工具流本身不符合 OpenAI 的增量
组装语义。

### 4.3 Sonnet 5 配置缺口

US 正式镜像中的 `config/models.toml` 没有 `claude-sonnet-5` 专项条目，请求落入
`anthropic.claude` catch-all。该 catch-all 只有通用能力，没有 `reasoning_path`，所以
客户端传入的 `reasoning_effort: "high"` 没有转成 Bedrock effort 配置。

AWS Sonnet 5 模型卡明确说明：

- adaptive thinking 始终开启且不能关闭；
- effort level 可以配置；
- prompt cache 最小前缀为 4096 tokens；
- cache TTL 支持 5 分钟和 1 小时。

因此新增配置使用 `reasoning_path = "adaptive_thinking"`，并声明
`adaptive_thinking`、`drop_sampling_params`、`structured_output` 和
`cache_ttl_1h`。这既让 `reasoning_effort` 真正传到 Bedrock，也保留模型卡声明的缓存
能力。

真实 Converse 探针同时确认：即使 adaptive thinking 已开启，简单和复杂的首轮工具
请求都可能只返回 `toolUse`，不返回可回放的签名 `reasoningContent`。这种响应不需要
capsule，原始工具 ID 是合法结果；但它仍必须遵守只在 start chunk 发送一次 ID 和名称的
规则。只有 Bedrock 实际返回签名推理块时，网关才生成 `brtc_v1` capsule。

### 4.4 修复边界

本次跟进只做两项生产变更：

- Chat 流式 arguments delta 始终省略 `id` 和 `function.name`；
- 在 TOML 中补齐 Sonnet 5 的模型能力和 reasoning path。

不改变 capsule 格式、HMAC、密钥轮换、续轮重建、`/responses` 或非流式 Chat 行为。

## 五、必须保留的代码变更

下表是本次最小实现边界。表中每项删除后都会破坏已验证的续轮闭环。

| 文件或模块 | 保留变更 | 必须保留的原因 |
| --- | --- | --- |
| `Cargo.toml`、`Cargo.lock` | `hmac`、`sha2` 直接依赖 | capsule 必须由网关认证；只依赖 Bedrock 内部签名无法防止客户端替换 capsule 中的工具 ID 或推理块。 |
| `src/bedrock/capsule.rs` | 三段式 capsule、HMAC、版本、keyring、64 KiB 固定上限 | 提供无外部状态的完整性校验、密钥轮换和恶意长 ID 的资源边界。64 KiB 是协议常量，不再暴露可调配置。 |
| `src/config/settings.rs` | enable、active kid、keyring 三个配置项 | enable 用于安全灰度；active kid 与多 key keyring 用于多副本一致签名和滚动轮换。 |
| `src/server/mod.rs` | 启动时解析一次 capsule runtime 并注入 Chat provider | 避免逐请求解析密钥；配置错误在启动阶段失败；所有 Chat 路径共享同一只读 keyring。 |
| `src/config/capabilities.rs` | `requires_signature_replay()` | 只有会产生 Bedrock 签名的推理路径需要 capsule；DeepSeek 字符串推理不能被误伤，也不能在 Rust 中硬编码模型名。 |
| `src/bedrock/provider.rs` | 功能开关、推理与工具组合判断、强制 `tool_choice` 边界、运行时对象传递 | 决定何时真正启用签名推理，并把同一运行时对象交给输入翻译、流式和非流式输出。 |
| `src/bedrock/response.rs` | 非流式 reasoning + toolUse capsule | `/chat/completions` 同时支持 `stream:false`；只修流式会造成同一接口两种续轮语义。 |
| `src/bedrock/stream.rs` | 推理 delta 累积、工具边界完成、先 capsule 后 ID、错误传播；工具 ID/名称只在 start chunk 发送一次 | 前半部分修复 WorkBuddy 首个报错；后半部分防止 OpenAI-compatible SDK 把每个 arguments delta 重复拼成无效工具名。 |
| `src/bedrock/translate.rs` | capsule 解码、`<think>` 去重、reasoning/toolUse/toolResult 重建 | 没有该步骤，客户端回传的是 capsule 而不是原始 `toolUseId`，Bedrock 会拒绝续轮。 |
| `config/models.toml` | Sonnet 5 adaptive reasoning、采样参数和缓存能力 | 没有专项条目时 `reasoning_effort` 会被 catch-all 静默忽略；模型卡声明 adaptive thinking 始终开启且 effort 可配置。 |
| `src/bedrock/mod.rs` | 注册 capsule 模块 | Rust 模块接线所必需。 |
| 对应 `*_tests.rs` | HMAC 篡改、错误 key、流顺序、非流式、并行工具、redacted reasoning、parts/null content、普通 ID 不变 | 覆盖 capsule 的安全边界和 Bedrock 合法返回形态，防止后续“精简”重新引入不可重放工具调用。 |
| 其他测试文件中的 3 行 `AppSettings` 字段 | disabled/None 测试值 | `AppSettings` 是公开 struct literal；新增三个正式配置字段后，现有测试夹具必须补齐，属于机械编译改动，不改变对应模块行为。 |
| `tests/golden/corpus.rs` | mapper 新参数及 `Result` 适配 | 输出 mapper 需要可选 runtime，流 mapper 需要把 capsule 失败传播为错误；golden 基线仍以 `None` 验证旧行为。 |

### 为什么保留并行工具与 redacted reasoning

这两项不是额外产品功能，而是 Bedrock 的合法响应形态：

- 一个 assistant turn 可以包含多个 `toolUse`；每个工具需要自己的原始 ID，但共享同一组
  推理块。
- Bedrock 可以返回 `redactedContent` 而不是 `reasoningText`；它仍必须逐字节续传。

删除任一处理都会让部分正常 Claude 工具续轮变成上游 400。

## 六、本轮额外移除的旁路改动

重新审计后又移除了：

- `/responses` 的 `tool_choice` 行为修改；
- Responses 推理信封对 Chat capsule 校验函数的依赖；
- capsule 大小上限的两个环境变量；
- 只被 Chat 使用的共享 `tools.rs` 校验函数及重复单测；
- 仅验证 getter 或无 `reasoning_effort` 默认值的重复测试。
- `translate.rs` 中 capsule 与普通工具调用各自维护一套转换循环的重复实现；当前两者
  共用一次工具遍历，仅在 ID 确实为 capsule 时增加解码和推理重建。

这些内容都不参与本次 Chat 续轮修复，保留会扩大协议影响面或维护面。

生产复验已证明，普通工具参数 delta 的 `id/name` 抑制并非旁路改动，而是标准 SDK
正确组装工具调用所必需的协议修复，因此在后续补丁中恢复，并增加无 capsule 的回归测试。

## 七、为什么当前差异仍有两千余行

按最终工作区差异的文件口径统计，排除未修改的历史方案文档后，增量大致分为：

| 分类 | 增加 | 删除 | 说明 |
| --- | ---: | ---: | --- |
| 生产实现与启动接线 | 约 730 行 | 约 69 行 | 其中 251 行是独立的 capsule 编解码、安全校验和密钥轮换模块；其余主要是流式状态机、非流式映射、续轮还原和配置接线。该口径还包含 `server/mod.rs` 中约 18 行启动失败测试。 |
| 测试与既有测试适配 | 约 1600 行 | 约 62 行 | 包含 capsule 安全边界、流式事件顺序、非流式、并行工具、redacted reasoning、续轮输入重建、旧行为不变，以及新增 `AppSettings` 字段导致的机械夹具补齐。 |
| 本事故中文文档 | 300 余行 | 0 行 | 记录根因、删除项、保留项、部署约束和验证证据。 |

因此，“数千行”不等于数千行业务实现。生产实现约占三成，测试约占六成。测试数量较大
是因为该功能同时跨越安全边界和有序流协议：HMAC 校验、首个工具 ID 的发送时机以及
Bedrock 续轮块顺序都无法由同一个正常路径测试代替。

`docs/chat-reasoning-tool-replay.md` 当前也是未跟踪文件，共 301 行，但它是此前他人的
历史方案，本轮始终未修改，也不属于本次建议提交的文件集合。建议提交范围只包含本文、
上表列出的 capsule 实现与对应测试。

## 八、安全与部署约束

### 8.1 环境变量

```text
CHAT_REASONING_CAPSULE_ENABLED=true
CHAT_REASONING_CAPSULE_ACTIVE_KID=current
CHAT_REASONING_CAPSULE_KEYS=current:<base64url-no-pad-key>,previous:<base64url-no-pad-key>
```

所有会生成或解析 capsule 的副本必须配置相同 keyring。轮换时先在全部副本加入新 key，
再切换 active kid，最后在旧 capsule 生命周期结束后移除旧 key。

### 8.2 灰度顺序

滚动部署必须按以下顺序：

1. 所有副本先部署解码代码和共享 keyring，但保持 encoder disabled；
2. 确认所有副本均可解码；
3. 再启用 encoder。

否则旧副本可能收到新副本生成的 `brtc_v1` ID 并返回 400。

### 8.3 完整 capsule 重放

HMAC 防止修改，不能阻止把一份合法 capsule 原样放回后续请求。对于无状态 Chat 工具
循环，客户端逐轮回传历史本身就是正常行为，因此不能简单加入一次性 nonce，否则会与
重试、历史重放和多副本无状态部署冲突。

当前边界是：

- capsule 只能还原为它签名时的原始工具 ID 和推理块；
- 网关不会因为 capsule 接受任意新内容；
- Bedrock 仍负责校验推理签名以及消息顺序。

## 九、验证记录

收敛前的清理版本已完成以下真实闭环验证；本轮没有改变 capsule 格式、签名算法、
推理块累积或续轮重建语义：

- Claude 签名推理与自动工具选择同时启用；
- WorkBuddy 只回传标准 `tool_calls[].id`；
- 保存的 150524 字节续轮请求含 2 个 assistant capsule 和 2 个 tool result capsule；
- 网关解码历史 capsule 后，真实 Bedrock 返回 HTTP 200；
- 新一轮流完整结束，包含 `[DONE]` 和 `finish_reason: "tool_calls"`；
- 真实流曾生成 695 字节 `brtc_v1` ID；
- 绕过临时 TCP 代理直连精简后的网关同样返回 HTTP 200；
- TypeScript `openai` SDK 6.48.0 与 OpenAI Agents SDK 0.13.4 保持长工具 ID 不变。

本次最终 diff 重新执行的结果：

| 校验项 | 结果 |
| --- | --- |
| `cargo fmt --all -- --check` | 通过 |
| `cargo clippy --all-targets --all-features -- -D warnings` | 通过 |
| unit tests | 806 passed，2 ignored |
| deployment tests | 2 passed |
| golden tests | 65 passed |
| router integration tests | 31 passed |
| doctests | 5 passed，1 ignored |
| capsule 专项测试 | 40 passed |
| `git diff --check` | 通过 |

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
git diff --check
```

### 9.1 US 临时部署验证（未发版）

2026-07-18 将当前工作区构建为静态 `linux/amd64` musl 二进制，以正式 `0.14.1`
镜像为基础，仅覆盖网关二进制和 `config/models.toml`，推送到 US 私有 ECR。此次验证
没有创建 Git tag、GitHub Release 或正式版本镜像。

- ECS task definition：`bedrock-rust-us-api:47`；
- 临时镜像 digest：
  `sha256:aef3cd75cada21bd317ad1f0a392c5f8d5db3835e0a950140da35ad12dcef323`；
- 两个新任务均为 `RUNNING/HEALTHY`，ALB 只保留两个 healthy target；
- Service Connect 保持 `idleTimeoutSeconds=300`、
  `perRequestTimeoutSeconds=0`，ALB deregistration delay 保持 120 秒；
- TypeScript `openai` SDK 6.48.0 首轮只组装出一个名称严格等于
  `lookup_weather` 的工具调用，续轮正常返回最终答案；
- OpenAI Agents SDK 0.13.4 的工具执行计数严格为 1，并正常返回最终答案；
- 本轮 Bedrock 未返回签名 reasoning block，因此首轮使用普通 `tooluse_*` ID。
  这正好覆盖此前会重复发送 `id/name` 的无 capsule 路径；
- CloudWatch 中上述四次流式请求均记录为 `chat streaming completed`。

## 十、不属于本次提交的内容

- `/tmp` 下的原始 TCP 调试代理；
- 抓取到的 WorkBuddy 请求体和响应体；
- BOM 或其他非标准 body 兼容；
- `/responses` 协议行为调整；
- 推理预算比例调整；
- 发布、ECR 构建和生产部署。
