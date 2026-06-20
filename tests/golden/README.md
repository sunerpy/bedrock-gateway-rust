# Golden Record/Replay Parity Fixtures

This directory holds the **Tier-1 offline parity safety net** for the Rust
gateway. It proves the Rust implementation is behaviourally faithful to the
pinned Python reference (SHA `9a3e752`) **without any live Bedrock / AWS
access**. The whole suite runs from on-disk fixtures via:

```bash
cargo test --test golden
```

> The harness + comparator live in [`mod.rs`](./mod.rs). **Real fixtures are
> captured in task 32** — the files under `fixtures/.../placeholder_selftest/`
> are minimal self-test placeholders that exercise the loaders and comparator
> before the real corpus exists.

## Why semantic, not byte-exact?

**Metis decision: parity is checked SEMANTICALLY.** Two payloads are considered
equal when they have:

1. the same field **set** (after volatile fields are removed),
2. the same field **values**, and
3. for streams, the same event-type **ordering**.

Object key insertion order is irrelevant (JSON objects are unordered). Array
order **is** significant (message / chunk / content-block ordering is a
parity-relevant property).

### Volatile (ignored) fields

These are non-deterministic between runs and between Python/Rust, so the
comparator strips them at any depth (case-insensitive). See
`DEFAULT_VOLATILE_FIELDS` in `mod.rs`:

| Field | Why volatile |
|-------|--------------|
| `id` | per-response random id (`chatcmpl-…`) |
| `created` / `created_at` | wall-clock timestamp |
| `request_id` / `x-request-id` | per-request trace id |
| `system_fingerprint` | backend build fingerprint |

You can override the ignore list per assertion with
`assert_semantic_eq_with(expected, actual, &["..."])` /
`semantic_eq(expected, actual, ignore)`.

## Comparator API (in `mod.rs`)

| Function | Purpose |
|----------|---------|
| `assert_semantic_eq(expected, actual)` | panic with diff unless semantically equal (default ignore list) |
| `assert_semantic_eq_with(expected, actual, ignore)` | same, with a custom ignore list |
| `semantic_eq(expected, actual, ignore) -> Result<(), String>` | fallible core; `Err` carries a path-qualified diff |
| `parse_sse(body) -> Vec<StreamEvent>` | parse an SSE/JSONL stream body into ordered events |
| `semantic_eq_stream(expected, actual, ignore)` | compare two event streams: length + event-type ordering + per-event values |
| `assert_stream_eq(expected, actual)` | panic with diff unless streams match (default ignore list) |

## Fixture directory layout

```
tests/golden/fixtures/
├── translation/                # OpenAI request  →  Bedrock invoke args
│   └── <case>/
│       ├── openai_request.json
│       └── expected_bedrock_args.json
├── streaming/                  # Bedrock event stream  →  OpenAI SSE chunks
│   └── <case>/
│       ├── bedrock_events.jsonl
│       └── expected_sse_chunks.jsonl
└── response/                   # Bedrock output  →  OpenAI (non-stream) response
    └── <case>/
        ├── bedrock_output.json
        └── expected_openai_response.json
```

### 1. Translation fixtures — `translation/<case>/`

Proves request translation parity (OpenAI wire request → Bedrock SDK args).

* `openai_request.json` — the inbound OpenAI `/chat/completions` request body.
* `expected_bedrock_args.json` — the Bedrock invoke arguments the gateway is
  expected to produce from that request.

Compared with `assert_semantic_eq(&expected_bedrock_args, &actual_bedrock_args)`.

### 2. Streaming fixtures — `streaming/<case>/`

Proves streaming response parity (Bedrock event stream → OpenAI SSE).

* `bedrock_events.jsonl` — one Bedrock streaming event per line (JSONL).
* `expected_sse_chunks.jsonl` — the expected OpenAI SSE output. Accepts either
  real SSE framing (`data: {…}` lines, optional `data: [DONE]` terminator) or
  plain JSONL (one chunk object per line). `parse_sse` handles both; the
  `[DONE]` sentinel is preserved as a terminal event of kind `done`.

Compared with `assert_stream_eq(&expected_sse, &actual_sse)`, which checks event
count, event-type **ordering**, and per-chunk semantic values.

#### Event-type ordering signature

For ordering checks, each SSE chunk is reduced to a stable *kind* tag derived
from its shape (not its volatile contents):

| Chunk shape | kind |
|-------------|------|
| `choices[0].delta.role` present | `delta:role` |
| `choices[0].delta.tool_calls` present | `delta:tool_calls` |
| `choices[0].delta.content` is a string | `delta:content` |
| `choices[0].delta` is `{}` | `delta:empty` |
| `choices[0].finish_reason` present | `finish:<reason>` |
| `[DONE]` sentinel | `done` |

A changed `finish_reason` therefore changes the ordering signature and is
caught at the ordering layer (in addition to the value layer).

### 3. Response fixtures — `response/<case>/`

Proves non-streaming response parity (Bedrock output → OpenAI response).

* `bedrock_output.json` — the Bedrock `Converse`/`InvokeModel` output payload.
* `expected_openai_response.json` — the expected OpenAI `chat.completion` body.

Compared with `assert_semantic_eq(&expected_openai_response, &actual_openai_response)`.

## Adding a real fixture (task 32)

1. Capture the input/output pair against the pinned Python reference.
2. Drop the pair into a new `<case>/` directory under the matching family.
3. Add a `#[test]` that loads it (`load_translation_fixture("<case>")`, etc.)
   and asserts parity against the Rust gateway's actual output.

No live AWS access is required at test time — fixtures are static files.

## Corpus (task 32) — wired into `cargo test --test golden`

The real corpus lives next to the harness in [`corpus.rs`](./corpus.rs)
(compiled as `golden::corpus`). Every case drives the **real** Rust pipeline
over the static fixture and asserts semantic parity:

- **translation** → `bedrock::translate::to_converse_args` composed with
  `reasoning::build_reasoning_config`, `tools::build_tool_config` /
  `normalize_tool_result_turns` / `inject_placeholder_tool_config`, and
  `cache::{decorate_system_blocks,decorate_messages}` — exactly the order
  `bedrock::provider::BedrockChatProvider::assemble` uses.
- **response** → `bedrock::response::from_converse_output`.
- **streaming** → `bedrock::stream::StreamState::map_event`, fed typed
  `aws_sdk_bedrockruntime` events reconstructed from the compact
  `bedrock_events.jsonl` shape. The router's terminal `data: [DONE]` is appended
  by the test (the state machine never emits it).
- **embeddings** → the public `CohereCodec` / `TitanCodec` / `NovaCodec`
  (`EmbeddingBodyCodec`) `encode`/`decode`, plus the float/base64 `build_data`
  formatting.
- **responses_response** → `bedrock::responses_response::from_converse_output_to_responses`
  over `(bedrock_output.json, openai_request.json) → expected_responses_response.json`.
- **responses_stream** → `bedrock::responses_stream::ResponsesStreamState`
  (`map_event` per Bedrock event + `finish()`) over
  `(bedrock_events.jsonl, openai_request.json) → expected_response_events.jsonl`.
  The Responses protocol has NO `[DONE]` sentinel — the stream ends on
  `response.completed`. Volatile `created_at` is ignored per-event;
  `reasoning_tokens` (a tiktoken estimate) is ignored on the reasoning cases.

### Coverage matrix

| Family      | Case                          | Exercises |
| ----------- | ----------------------------- | --------- |
| translation | `text_basic`                  | plain user text → `{text}` block |
| translation | `system_developer_blocks`     | system+developer → `system` blocks (not in messages) |
| translation | `multimodal_data_uri_image`   | `data:` URI image → `{image:{format,source.bytes}}` |
| translation | `stop_string_singleton`       | `stop` string → singleton `stopSequences` |
| translation | `topp_conflict_drops_topp`    | `temperature_topp_conflict` drops `topP` |
| translation | `reasoning_adaptive_thinking` | adaptive path: `thinking`+`output_config` |
| translation | `reasoning_budget_tokens`     | budget path: `reasoning_config.budget_tokens` (ratio math) |
| translation | `reasoning_deepseek_string`   | deepseek path: `reasoning_config="<effort>"` |
| translation | `reasoning_none_ignored`      | none path: `reasoning_effort` ignored |
| translation | `tools_single_turn_auto`      | tool spec + `toolChoice.auto` |
| translation | `tools_multi_turn_placeholder`| toolUse/toolResult history → placeholder toolConfig |
| translation | `prompt_cache_system_point`   | `extra_body.prompt_caching.system` → system `cachePoint` (Nova) |
| response    | `text_basic`                  | text content + usage |
| response    | `tool_use_single`             | `tool_use` → `tool_calls` + `content:null` |
| response    | `reasoning_inline_think`      | `<think>…</think>` inline; `reasoning_content` never on wire |
| response    | `cache_read_tokens`           | rebuild-from-parts: prompt=input+cacheRead+cacheWrite; `cached_tokens`=cacheRead; cacheWrite never its own field |
| response    | `usage_no_cache_regression`   | no-cache: prompt=inputTokens (never negative); no `prompt_tokens_details` |
| response    | `finish_reason_length`        | `max_tokens` → `length` |
| response    | `finish_reason_content_filter`| `content_filtered` → `content_filter` |
| response    | `usage_fallback_no_total`     | no `totalTokens` → `total = input + output` |
| streaming   | `text_sequence`               | role → content deltas → finish → usage → `[DONE]` |
| streaming   | `reasoning_think_sequence`    | `<think>` open, accumulate, `</think>` on text transition |
| streaming   | `think_open_at_stop_closes`   | `<think>` open at `messageStop` → `</think>`, finish deferred |
| streaming   | `tool_use_sequence`           | tool start + input fragments + `tool_calls` finish + usage |
| embeddings  | `cohere_float`                | Cohere encode body + float decode |
| embeddings  | `cohere_base64`               | Cohere encode + base64 (LE f32 bytes) format |
| embeddings  | `titan_float`                 | Titan encode `{inputText}` + single-vector decode |
| responses_response | `responses_text`        | text → single `message` item (`output_text`); Responses usage field names; no `output_text` wire field; no `<think>` |
| responses_response | `responses_tool_call`   | `tool_use` → `function_call` item (`call_id`/`name`/JSON-string `arguments`) |
| responses_response | `responses_reasoning`   | reasoning → `reasoning` item FIRST then `message`; usage formula with cacheRead+cacheWrite (`input_tokens`=input+read+write, `cached_tokens`=read) |
| responses_stream | `responses_text`          | full text lifecycle order; `sequence_number` monotonic from 0; NO `[DONE]`; `completed` carries full Response |
| responses_stream | `responses_tool`          | `function_call` add+done (full args); state machine emits NO `function_call_arguments.delta` |
| responses_stream | `responses_reasoning`     | reasoning item BEFORE message item; reasoning_text deltas (NOT `<think>`); monotonic seq; NO `[DONE]` |

### Documented intentional divergences (encoded in the corpus)

These are the Metis-sanctioned FIX divergences from the Python gateway; the
fixtures encode the **Rust** (correct) behaviour:

| Divergence | Where it shows | Behaviour |
| ---------- | -------------- | --------- |
| Cache-write tokens never their own field | `response/cache_read_tokens` | `cacheWriteInputTokens` folds into `prompt_tokens`/`total_tokens` (rebuild-from-parts) but is never surfaced as its own wire field; only `cacheReadInputTokens` surfaces as `prompt_tokens_details.cached_tokens`. |
| `reasoning_content` never on the wire | `response/reasoning_inline_think` | reasoning is rendered inline as `<think>…</think>` in `content`; the `reasoning_content` field is never serialized. |
| Error envelope (always JSON) | n/a (asserted in `src` unit tests) | Errors always return the full OpenAI error envelope; not exercised here because the corpus covers the success-path translation/response/stream/embedding mappings. |

### Two volatile-value notes

- `reasoning_tokens` (a tiktoken **estimate**, not a parity-critical wire value)
  is added to the per-assertion ignore list for the reasoning response/stream
  cases. The `<think>` content, `finish_reason`, and prompt/total token math are
  still asserted exactly.
- Float fixtures use values that are **exactly representable in `f32`** (e.g.
  `0.5`, `0.25`, `0.125`) so the `f32 → f64` JSON widening does not introduce a
  spurious mismatch (e.g. `0.7f32` serializes as `0.699999988079071`).
