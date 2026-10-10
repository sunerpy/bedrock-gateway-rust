# Caching and reasoning

This page covers two things the gateway does for you on each request: it places prompt-cache points, and it turns
the OpenAI reasoning setting into what each model expects. [Caching and reasoning in depth](../../caching-and-reasoning.md)
has the thresholds of every model.

## Prompt caching

Bedrock can cache the beginning of a prompt and bill a repeated beginning at a lower rate. It needs cache points in
the request, and OpenAI clients do not send them, so the gateway adds them. It is on by default
(`ENABLE_PROMPT_CACHING=true`).

On a model that supports caching, the gateway places cache points after the tool definitions, after the system
prompt and in the messages, in that order, up to the number of points the model allows in one request (four on
Claude). It places a point only when the prompt before it is at least as long as the model's minimum, which is
between 512 and 4,096 tokens depending on the model.

A cache hit needs a prefix that does not change from one request to the next. A changed tool description, a
timestamp in the system prompt or reordered tools start a new cache entry.

The usage of a response shows the effect. `prompt_tokens` counts every input token, cached or not, and
`prompt_tokens_details.cached_tokens` counts the ones read from the cache; on the Responses API they are
`input_tokens` and `input_tokens_details.cached_tokens`.

### Control it per request

A `prompt_caching` object inside an `extra_body` object in the request body overrides the defaults for one request.
The OpenAI SDKs merge their own `extra_body` option into the top level of the request, so with the Python SDK it is
nested once more:

```python
client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=messages,
    extra_body={"extra_body": {"prompt_caching": {"system": True, "messages": False, "ttl": "1h"}}},
)
```

The request body then carries:

```json
"extra_body": {"prompt_caching": {"system": true, "messages": false, "ttl": "1h"}}
```

| Field      | Meaning                                                                |
| ---------- | ---------------------------------------------------------------------- |
| `system`   | `false` leaves the system prompt without a cache point                 |
| `messages` | `false` leaves the messages without cache points                       |
| `ttl`      | `5m` or `1h`; overrides `PROMPT_CACHE_TTL` for this request            |

On the Responses API only `ttl` applies. A cache entry lives five minutes, or one hour with `ttl: "1h"` on the models
that support it, such as recent Claude versions. On another model a requested hour becomes five minutes, and the
gateway logs a warning. The gateway removes `prompt_caching` before it calls Bedrock.

## Reasoning

A request asks for reasoning the OpenAI way: `reasoning_effort` on Chat Completions, `reasoning.effort` on Responses.
The values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`; each model accepts a subset, and Bedrock
rejects a value the model does not take.

```python
client.chat.completions.create(
    model="global.anthropic.claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Solve step by step: 17 * 23"}],
    reasoning_effort="high",
)
```

The gateway turns the effort into the form the model takes:

| Models                                            | What the model receives                                     |
| ------------------------------------------------- | ----------------------------------------------------------- |
| Claude Sonnet 4.5 and 4.6, Haiku 4.5, Opus 4.5    | a thinking budget, worked out from `max_tokens`             |
| Claude Opus 4.6 and later, Sonnet 5 and later, Haiku 5.5, Fable | adaptive thinking with that effort           |
| GPT-6.x                                           | `reasoning.effort`                                          |
| GPT-5.x and gpt-oss                               | the request as the client sent it                           |
| DeepSeek V3                                       | the effort as a string                                      |

For a thinking budget, `low` is 30 percent of `max_tokens`, `medium` 60 percent and higher values one token less than
`max_tokens`, never below 1,024 tokens. When the budget leaves too little room for the answer, the gateway raises
`max_tokens` for Bedrock.

### Where the reasoning appears

| Route                                  | Reasoning in the answer                                              |
| -------------------------------------- | -------------------------------------------------------------------- |
| Chat Completions                       | inside `content`, between `<think>` and `</think>`                   |
| Responses                              | a `reasoning` item in `output`, before the message                   |
| Chat Completions with gpt-oss          | `reasoning` in each streamed delta, as Bedrock sends it              |

Token usage reports the reasoning tokens in `completion_tokens_details.reasoning_tokens` or
`output_tokens_details.reasoning_tokens` when the model reports them.

### Reasoning and tools

When a model reasons and then calls a tool, the next request has to carry that reasoning back, with its signature.
The Responses API does this with its `reasoning` items. On Chat Completions the gateway can do it inside the tool
call's ID when you give it a signing key; see
[Reasoning across tool calls](configuration.md#reasoning-across-tool-calls).
