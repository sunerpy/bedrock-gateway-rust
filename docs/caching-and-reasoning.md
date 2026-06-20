# Prompt Caching、Reasoning Budget 与跨区域 Inference Profile 行为说明

> 受众：运维人员、接入方开发者
>
> 参见 [AGENTS.md](../AGENTS.md) 中"缓存放置契约"和"两条推理渲染路径"段落。

---

## 目录

1. [Prompt Caching 行为](#1-prompt-caching-行为)
   - 1.1 [默认开启与注入顺序](#11-默认开启与注入顺序)
   - 1.2 [支持缓存的判定规则](#12-支持缓存的判定规则)
   - 1.3 [逐模型 cache_min_tokens 阈值](#13-逐模型-cache_min_tokens-阈值)
   - 1.4 [AWS 官方要点摘录](#14-aws-官方要点摘录)
   - 1.5 [usage 字段计账](#15-usage-字段计账)
   - 1.6 [字节稳定前缀规则](#16-字节稳定前缀规则)
2. [Reasoning / Extended Thinking Budget](#2-reasoning--extended-thinking-budget)
   - 2.1 [budget_tokens 计算路径](#21-budget_tokens-计算路径)
   - 2.2 [硬下限与 maxTokens 抬高](#22-硬下限与-maxtokens-抬高)
   - 2.3 [两条推理渲染路径（不可混淆）](#23-两条推理渲染路径不可混淆)
3. [跨区域 Inference Profile](#3-跨区域-inference-profile)
   - 3.1 [带前缀模型 ID 的必要性](#31-带前缀模型-id-的必要性)
   - 3.2 [网关内部双 ID 策略](#32-网关内部双-id-策略)
   - 3.3 [缓存与跨区域共用](#33-缓存与跨区域共用)
   - 3.4 [/models 接口行为（当前部署 region 范围）](#34-models-接口行为当前部署-region-范围)
4. [运维配置指南](#4-运维配置指南)
5. [可观测性与日志](#5-可观测性与日志)
6. [相关源码位置](#6-相关源码位置)

---

## 1. Prompt Caching 行为

### 1.1 默认开启与注入顺序

缓存点自动注入**默认开启**（环境变量 `ENABLE_PROMPT_CACHING` 默认 `true`；配置文件 `config/app.toml` 同样默认 `true`）。

注入顺序固定为：**tools → system → messages**

三个位置共享一个预算，总注入数不超过 `max_cache_checkpoints`（默认 4）。预算从 tools 区开始累计，用完后后续区域跳过注入。

```
tools cachePoint    → system cachePoint → messages cachePoint
  (slot 1)              (slot 2)              (slot 3)
          ↑── 共享 max_cache_checkpoints=4 ──────────────────
```

可通过 `extra_body.prompt_caching` 在单次请求内覆盖全局开关：

```python
# 只缓存 system，跳过 messages
response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[...],
    extra_body={
        "prompt_caching": {
            "system": True,
            "messages": False
        }
    }
)
```

全局关闭：设置 `ENABLE_PROMPT_CACHING=false`。

### 1.2 支持缓存的判定规则

**模型是否支持缓存，完全由 `config/models.toml` 中是否声明 `cache_min_tokens` 参数决定。**
代码（`src/bedrock/cache.rs`）不做任何模型名称判断，这是零硬编码契约的一部分。

判定函数（`supports_caching`）：

```rust
// src/bedrock/cache.rs
pub fn supports_caching(model: &str, caps: &dyn ModelCapabilities) -> bool {
    caps.cache_min_tokens(model).is_some() || caps.max_cache_tokens(model).is_some()
}
```

**Family 兜底机制：** `config/models.toml` 末尾有一个 `match = "anthropic.claude"` 的兜底条目，未在前面单独列出的 Claude 模型 ID 会自动匹配到此条目，获得 `cache_min_tokens = 4096` 的保守默认值，不会因为新模型未录入而静默禁用缓存。

### 1.3 逐模型 cache_min_tokens 阈值

> **常见故障：** 将 `cache_min_tokens` 配置为错误值（例如把 1024 阈值模型配成 4096），会导致实际 prompt 未达阈值时静默不注入 cachePoint，缓存完全不命中，但请求本身正常返回 200——没有任何错误提示。

下表为网关当前配置（来源：`config/models.toml`，数字与 AWS 官方文档对齐）：

| 模型                 | Model ID（foundation id）                 | cache_min_tokens | max_cache_checkpoints | 可缓存字段            | config 条目                |
| -------------------- | ----------------------------------------- | ---------------- | --------------------- | --------------------- | -------------------------- |
| Claude Sonnet 4.5    | anthropic.claude-sonnet-4-5-20250929-v1:0 | 4,096            | 4                     | system/messages/tools | `claude-sonnet-4-5`        |
| Claude Sonnet 4.6    | anthropic.claude-sonnet-4-6               | 1,024            | 4                     | system/messages/tools | `claude-sonnet-4-6`        |
| Claude Haiku 4.5     | anthropic.claude-haiku-4-5-20251001-v1:0  | 4,096            | 4                     | system/messages/tools | `claude-haiku-4-5`         |
| Claude Opus 4.5      | anthropic.claude-opus-4-5-20251101-v1:0   | 4,096            | 4                     | system/messages/tools | `claude-opus-4-5`          |
| Claude Opus 4.6      | anthropic.claude-opus-4-6-v1              | 4,096            | 4                     | system/messages/tools | `claude-opus-4-6`          |
| Claude Opus 4        | anthropic.claude-opus-4-20250514-v1:0     | 1,024            | 4                     | system/messages/tools | `anthropic.claude`（兜底） |
| Claude 3.7 Sonnet    | anthropic.claude-3-7-sonnet-20250219-v1:0 | 1,024            | 4                     | system/messages/tools | `anthropic.claude`（兜底） |
| Claude 3.5 Sonnet v2 | anthropic.claude-3-5-sonnet-20241022-v2:0 | 1,024            | 4                     | system/messages/tools | `anthropic.claude`（兜底） |
| Amazon Nova（所有）  | amazon.nova-\*                            | 1,024            | N/A                   | system/messages/tools | `amazon.nova`              |

**AWS 官方对应表（四列：Model / Model ID / 最小 token/checkpoint / 最大 checkpoints）：**

| Model                | Model ID                                  | Min tokens/checkpoint | Max checkpoints |
| -------------------- | ----------------------------------------- | --------------------- | --------------- |
| Claude Opus 4.5      | anthropic.claude-opus-4-5-20251101-v1:0   | 4,096                 | 4               |
| Claude Sonnet 4.5    | anthropic.claude-sonnet-4-5-20250929-v1:0 | 4,096                 | 4               |
| Claude Haiku 4.5     | anthropic.claude-haiku-4-5-20251001-v1:0  | 4,096                 | 4               |
| Claude Opus 4        | anthropic.claude-opus-4-20250514-v1:0     | 1,024                 | 4               |
| Claude Sonnet 4.6    | anthropic.claude-sonnet-4-6               | 1,024                 | 4               |
| Claude 3.7 Sonnet    | anthropic.claude-3-7-sonnet-20250219-v1:0 | 1,024                 | 4               |
| Claude 3.5 Sonnet v2 | anthropic.claude-3-5-sonnet-20241022-v2:0 | 1,024                 | 4               |

来源：[AWS 文档 - Prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

### 1.4 AWS 官方要点摘录

以下摘录均来自 [AWS Bedrock Prompt Caching 用户指南](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)，均已核实为官方原文。

> "Cache checkpoints have a minimum and maximum number of tokens, dependent on the specific model. You can only create a cache checkpoint if your total prompt prefix meets the minimum number of tokens. **If you try to add a cache checkpoint before meeting the minimum, your inference will still succeed, but your prefix will not be cached.**"

这正是网关 `cache_min_tokens` floor gate 的设计依据：低于阈值时跳过 cachePoint 注入，避免发出无效 cachePoint（Bedrock 会静默忽略但浪费一个 checkpoint 配额）。

> "Prompt caching is only supported for on-demand inference endpoints. **It is not supported with the batch inference API.**"

网关仅调用 Converse/ConverseStream（on-demand），不走 batch，此限制不影响本网关。

> "**These prompt prefixes should be static between requests; alterations to the prompt prefix in subsequent requests will result in cache misses.**"

即字节稳定前缀规则，详见 [1.6 节](#16-字节稳定前缀规则)。

针对 Amazon Nova 模型，官方额外说明：

> "Amazon Nova offers automatic prompt caching for all text prompts... **we recommend opting in to Explicit Prompt Caching.**"

网关已为 Nova 配置 `cache_min_tokens = 1024`，走显式缓存（explicit prompt caching）路径，与官方推荐一致。

**官方链接：**

- Prompt caching 用户指南：https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- Bedrock 定价（缓存读/写费率）：https://aws.amazon.com/bedrock/pricing/

### 1.5 usage 字段计账

网关使用统一的 `compute_token_usage` 函数（`src/bedrock/tokens.rs`）计算所有 usage 字段：

```
prompt_tokens  = input + cacheRead + cacheWrite
total_tokens   = prompt_tokens + output
cached_tokens  = cacheRead（仅读取侧）
```

**关键说明：**

- `cached_tokens` **只反映缓存读取**（cacheRead）。第一次写入缓存时 Bedrock 返回 `cacheWriteInputTokens`，但 OpenAI 协议中没有对应的写侧字段，因此它被折入 `prompt_tokens` 而不单独列出。
- 响应字段位置：
  - Chat Completions：`usage.prompt_tokens_details.cached_tokens`
  - Responses API：`usage.input_tokens_details.cached_tokens`

### 1.6 字节稳定前缀规则

缓存命中依赖确定性序列化。**修改任何 cachePoint 之前的内容都会导致该请求中后续所有缓存点失效。**

实践要点：

- 将稳定内容（大型 system prompt、固定 tools 定义）放在对话靠前的位置。
- 多轮对话中，早期消息一旦确定，不要修改其内容，否则 messages 区 cachePoint 失效。
- 跨区路由时，注入的 `cachePoint` 结构相同；切换区域本身不会破坏前缀稳定性，但官方提示高峰期跨区调用可能增加 cache write 次数（详见 [3.3 节](#33-缓存与跨区域共用)）。

---

## 2. Reasoning / Extended Thinking Budget

### 2.1 budget_tokens 计算路径

使用 `reasoning_effort` 参数触发推理：

```python
# Chat Completions 接口（需显式带 reasoning_effort）
response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "请逐步推理：..."}],
    extra_body={"reasoning_effort": "medium"}
)

# Responses 接口（默认携带 reasoning 字段）
response = client.post("/api/v1/responses", json={
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "..."}]}],
    "reasoning": {"effort": "medium"}
})
```

支持的 effort 级别：`none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`

**BudgetTokens 路径的计算公式（来自 `src/bedrock/reasoning.rs`）：**

```
effective_max = max_completion_tokens ?? max_tokens

low    -> budget = int(effective_max * 0.3)
medium -> budget = int(effective_max * 0.6)
high / xhigh / max -> budget = effective_max - 1
```

比例来自 `config/models.toml` 的 `budget_ratios`（`low=0.3, medium=0.6, high=-1.0 sentinel`），可在 TOML 中逐模型覆盖，**无需改代码**。

四条推理路径（由 `config/models.toml` 的 `reasoning_path` 字段决定）：

| reasoning_path      | 适用模型示例            | Bedrock wire 字段                                        |
| ------------------- | ----------------------- | -------------------------------------------------------- |
| `budget_tokens`     | claude-sonnet-4-x       | `reasoning_config = {type: "enabled", budget_tokens: N}` |
| `adaptive_thinking` | claude-opus-4-6/4-7/4-8 | `thinking = {type: "adaptive"} + output_config.effort`   |
| `deepseek_string`   | deepseek.v3             | `reasoning_config = "low"/"medium"/"high"`               |
| `none`              | 无推理能力模型          | 无（reasoning_effort 被忽略）                            |

**官方示例（Converse API with reasoning）：**
https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_Converse_AnthropicClaudeReasoning_section.html

官方示例中 budget 通过 `additionalModelRequestFields` 传入，网关已按此方式实现。

### 2.2 硬下限与 maxTokens 抬高

> **常见故障：** 当 `max_output_tokens`（或 `max_tokens`）过小时（例如 50），按比例算出的 thinking budget（如 `50 * 0.3 = 15`）会低于 Anthropic 的硬性下限 1024，Bedrock 返回 HTTP 400：
>
> ```
> thinking.enabled.budget_tokens: Input should be greater than or equal to 1024
> ```

网关的修复方案（`src/bedrock/reasoning.rs`，commit c756e79）：

1. **Budget 上取整到下限：** `budget = max(budget, min_budget_tokens)`
   - `min_budget_tokens` 默认值 1024，来自 `config/models.toml` 的 `default` 条目
   - 可逐模型覆盖，仅改 TOML

2. **maxTokens 抬高以容纳 budget：** 同时将发往 Bedrock 的 `maxTokens` 抬高到
   `max(effective_max, budget + 256)`
   - `+256` 是完成余量（`COMPLETION_HEADROOM_TOKENS`），确保 thinking 之外还有空间输出答案
   - Anthropic 要求 `maxTokens > budget_tokens`，此调整满足该约束
   - **这只影响发往 Bedrock 的参数，不改变响应给客户端的任何字段**

示例：`max_output_tokens=50, effort=low`

```
ratio budget = int(50 * 0.3) = 15
clamped budget = max(15, 1024) = 1024
maxTokens sent = max(50, 1024 + 256) = 1280
```

请求成功，响应正常，客户端 `usage.completion_tokens` 基于实际生成量计算。

### 2.3 两条推理渲染路径（不可混淆）

推理输出在两个接口层上形式不同，**不能合并**：

| 接口                | 推理渲染方式                                                  | 理由                                                                                                                   |
| ------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `/chat/completions` | 内联 `<think>...</think>` 嵌入 `content` 字符串               | OpenAI Chat 协议无推理专用字段；`reasoning_content` 内部有值但标记 `#[serde(skip_serializing)]` 永不出现在响应 JSON 中 |
| `/responses`        | `output` 数组中的独立 `reasoning` 输出项（非 `<think>` 包裹） | Responses API 有专用的 reasoning output item 类型                                                                      |

这是架构规则，修改任意一条时必须确认另一条未受影响。详见 AGENTS.md 中"两条推理渲染路径"段。

---

## 3. 跨区域 Inference Profile

### 3.1 带前缀模型 ID 的必要性

AWS Bedrock 的跨区域推理通过 **inference profile** 实现。带地理前缀的模型 ID（如 `us.anthropic.claude-sonnet-4-5-20250929-v1:0`、`eu.anthropic.claude-3-5-sonnet-20241022-v2:0`）是 inference profile 标识符，**必须原样发给 Bedrock**。

如果将前缀去掉使用裸 foundation model ID（如 `anthropic.claude-sonnet-4-5-20250929-v1:0`），Bedrock 会拒绝：

```
HTTP 400 ValidationException:
Invocation of model ID anthropic.claude-sonnet-4-5-20250929-v1:0
with on-demand throughput isn't supported.
Retry with an inference profile.
```

> 这正是本网关修复的第三个 bug：Responses 接口在处理跨区前缀模型时，曾误将 `resolve_foundation()` 返回的裸 foundation ID 作为 model_id 发往 Bedrock，导致 100% 的 /responses 请求对跨区模型 400，而 /chat/completions 用原始模型 ID 所以正常。

支持的前缀（地理/跨区前缀全集）：

| 前缀      | 覆盖区域（示例）                                           |
| --------- | ---------------------------------------------------------- |
| `us.`     | 美国区（us-east-1 / us-east-2 / us-west-2）                |
| `eu.`     | 欧洲区（eu-west-_ / eu-central-_ 等）                      |
| `apac.`   | 亚太区（ap-northeast-_ / ap-southeast-_ / ap-south-\* 等） |
| `jp.`     | 日本（ap-northeast-1）                                     |
| `au.`     | 澳大利亚（ap-southeast-2）                                 |
| `ca.`     | 加拿大（ca-central-1）                                     |
| `global.` | 全球多区（当前主要为 Claude 全系列）                       |

> **来源说明：** 以上为 2026-06-20 扫描全部 34 个 AWS 商业区 `aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED` 得到的完整前缀集（共 7 个，无其他）；代码侧在 `src/bedrock/capabilities.rs` 的 `GEO_PREFIXES` 常量中枚举，AWS 新增地理前缀时需同步更新该常量。

**官方链接：** https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html

### 3.2 网关内部双 ID 策略

网关在以下两个场景中使用不同的模型 ID，两者**绝对不可混用**：

| 用途                                                 | 使用的 ID                                                   | 示例                                           |
| ---------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| 能力匹配（缓存阈值 / reasoning_path / 温度冲突检查） | `resolve_foundation()` 返回的 foundation ID（去前缀、小写） | `anthropic.claude-sonnet-4-5-20250929-v1:0`    |
| 发往 Bedrock 的实际调用（model_id）                  | 原始请求模型 ID（带前缀，原样）                             | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |

能力匹配前先经 `normalize_for_match()`（`src/bedrock/capabilities.rs`）去掉地理前缀并小写，使 `config/models.toml` 的 `match` 字符串只需写一次，同时覆盖 `GEO_PREFIXES` 枚举的全部 7 个前缀变体（`us.`/`global.`/`eu.`/`apac.`/`jp.`/`au.`/`ca.`）。该归一化**仅用于能力匹配**，不改变 `resolve_foundation()` 的返回值，也不改变发往 Bedrock 的 model_id（始终原样带前缀发出）。

Chat 和 Responses 两个接口均已对齐此行为（`src/bedrock/provider.rs` 和 `src/bedrock/responses_provider.rs`）。

### 3.3 缓存与跨区域共用

缓存可以与跨区域推理同时使用，无需额外配置。需注意：

> **AWS 官方提示（来自 [Prompt Caching 文档](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) 中 "Prompt Caching with Cross-region Inference" 段）：**
> 跨区域流量在高峰时段会在多个区域路由，这可能导致同一前缀的 cache write 发生在不同区域，增加 cache write 次数。

实践建议：

- 跨区路由时，稳定的大型 system prompt 仍应放在对话开头，最大化 cache read 机会。
- 若观测到 `cached_tokens` 偶发为 0（正常时应大于 0），可能是跨区切换导致落到了没有该缓存的区域，属正常现象，下次相同区域调用会恢复命中。
- `cached_tokens` 统计的是 cacheRead，不统计 cacheWrite，所以首次调用（写缓存）为 0 是预期行为。

### 3.4 /models 接口行为（当前部署 region 范围）

模型目录的范围由**网关部署所在的 region**决定，不做跨地理区聚合。

- `GET /api/v1/models` 列出**当前部署 region 可用**的模型。网关用 home region（`AWS_REGION`）调 `list-foundation-models` + `list-inference-profiles` 组装目录。因此部署在美国区（如 us-east-2）时，目录会包含 `us.` 前缀的跨区 profiles 与 `global.` 前缀模型，但**不会**聚合 `eu.`/`apac.`/`jp.`/`au.`/`ca.` 等其他地理区的 profiles。
- **INFERENCE_PROFILE-only 的模型**（如 Claude 全系列，其裸 foundation `inferenceTypesSupported` 仅含 `INFERENCE_PROFILE`、无 `ON_DEMAND`）：其**裸 foundation ID 不出现在列表**（不可直调，直接发裸 id 会被 Bedrock 拒绝），但其跨区 profile ID（`us.anthropic.claude-*` / `global.anthropic.claude-*` 等）**会出现在列表**，且可直接用作 `model` 请求参数。
- `GET /api/v1/models/{id}` 支持用 profile ID 查询（如 `GET /api/v1/models/us.anthropic.claude-sonnet-4-5-20250929-v1:0` → 200）。

> 这是现状设计：目录范围 = 部署 region 范围。如需访问其他地理区的模型，需在该地理区单独部署网关实例。目录组装逻辑见 `src/bedrock/models.rs` 的 `assemble_catalog`（裸 foundation 按 `ON_DEMAND` 过滤入目录；其 backing 的 inference profiles 独立纳入，使 INFERENCE_PROFILE-only 模型的跨区 profile 可被发现）。

---

## 4. 运维配置指南

### 4.1 修改缓存阈值或 reasoning 参数

**只改 `config/models.toml`，不改代码，不需要重新编译。**

```toml
# 示例：将某模型的缓存最小 token 改为 1024
[[model]]
match = "claude-sonnet-4-6"
capabilities = ["temperature_topp_conflict", "context_1m_beta"]
[model.params]
cache_min_tokens = 1024      # 修改此值
reasoning_path = "budget_tokens"

# 示例：调整 reasoning budget 比例
[[model]]
match = "default"
capabilities = []
[model.params]
min_budget_tokens = 1024     # Anthropic 硬下限，通常不需要改
[model.params.budget_ratios]
low = 0.3
medium = 0.6
high = -1.0                  # sentinel: max_tokens - 1
```

修改后需要：

- **单二进制部署**：重启进程（config 在启动时读取）。
- **容器部署**：
  - 若使用嵌入 config（`include_str!`）：需重建镜像（`cargo build --release` + `docker build`）。
  - 若通过 `CONFIG_DIR` 环境变量挂载外部 config 目录：只需替换挂载目录下的 TOML 文件并重启容器，**无需重建镜像**。

### 4.2 CONFIG_DIR 外部覆盖

设置 `CONFIG_DIR` 环境变量，可让网关优先从外部目录加载 config，覆盖编译时嵌入的默认值：

```bash
# 使用外部 config 目录
CONFIG_DIR=/etc/bedrock-gateway/config docker run ...

# 优先级：外部文件存在且解析成功 > 编译时嵌入默认
# 外部文件缺失或解析失败 → 自动回退到嵌入默认（不会降级为空配置）
```

### 4.3 环境变量速查

| 变量                    | 默认值                                      | 说明                                         |
| ----------------------- | ------------------------------------------- | -------------------------------------------- |
| `ENABLE_PROMPT_CACHING` | `true`                                      | 全局缓存开关                                 |
| `CONFIG_DIR`            | `config`（相对 WORKDIR）                    | 外部 config 目录路径                         |
| `LOG_LEVEL`             | `info`                                      | 日志级别；设为 `debug` 可看 Bedrock 调用细节 |
| `DEFAULT_MODEL`         | `anthropic.claude-3-5-sonnet-20241022-v2:0` | 默认模型                                     |

---

## 5. 可观测性与日志

每次请求的业务日志字段（`info` 级别）：

**非流式 chat 完成：**

```
chat completed | request_id=req-xxx model=us.anthropic.claude-sonnet-4-5-20250929-v1:0
               | prompt_tokens=1234 completion_tokens=56 total_tokens=1290
               | cached_tokens=1200 cache_hit=true duration_ms=890
```

**流式 chat 开始：**

```
chat streaming started | request_id=req-xxx model=... ttfb_ms=320
```

**流式 chat 完成：**

```
chat streaming completed | request_id=req-xxx prompt_tokens=1234 completion_tokens=56
                         | total_tokens=1290 cached_tokens=1200 cache_hit=true duration_ms=2100
```

Responses 接口日志格式对称（`responses completed` / `responses streaming started` / `responses streaming completed`），字段含义相同，但 token 字段名为 `input_tokens` / `output_tokens`。

**关键字段说明：**

- `cached_tokens`：缓存读取的 token 数；`0` 表示首次写缓存或未命中。
- `cache_hit`：`true` 表示本次有缓存读取；`false` 表示全量 prompt 计算（包括首次建立缓存时）。
- `ttfb_ms`：流式场景下，从收到请求到第一个 token 的延迟（Time To First Byte）。
- `request_id`：网关自生成的 trace ID（格式 `req-{nanos:x}-{seq:x}`），可关联同一请求的所有日志行。若客户端发送 `x-request-id` 头，则使用客户端提供的值。

> 隐私说明：任何日志级别下均**不会**打印 prompt/completion 文本内容、消息正文、API_KEY 或 bearer token。

---

## 6. 相关源码位置

| 功能                                     | 文件                                | 关键函数/位置                                                                            |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------- |
| 缓存 floor gate（低于阈值不注入）        | `src/bedrock/cache.rs`              | `decorate_system_blocks`（约第 290 行 `cache_min_tokens` 判断）                          |
| Reasoning budget 计算与下限钳制          | `src/bedrock/reasoning.rs`          | `build_reasoning_config`（`ReasoningPath::BudgetTokens` 分支，约第 198-204 行）          |
| min_budget_tokens 配置读取               | `src/config/capabilities.rs`        | `ModelParams::min_budget_tokens` 字段定义                                                |
| Responses 接口 outbound model_id         | `src/bedrock/responses_provider.rs` | `send_converse` / `send_converse_stream`（使用原始 `req.request.model` 而非 `resolved`） |
| tools 区 cachePoint 转 SDK 类型          | `src/bedrock/provider.rs`           | `build_sdk_tool_config`（cachePoint 分支，避免 "tool missing toolSpec" 错误）            |
| 能力匹配前缀归一化（去地理前缀）         | `src/bedrock/capabilities.rs`       | `normalize_for_match` + `GEO_PREFIXES`（枚举 7 个实测前缀）                              |
| profile→foundation 解析（不改 model_id） | `src/bedrock/capabilities.rs`       | `ConfigModelCapabilities::resolve_foundation`                                            |
| token usage 计算                         | `src/bedrock/tokens.rs`             | `compute_token_usage`                                                                    |

---

_如发现本文档与代码行为不符，请以代码为准并更新本文档。_
_相关架构决策见 [AGENTS.md](../AGENTS.md) 中"缓存放置契约"、"零硬编码契约"段落。_
