# OpenAI protocol compatibility audit and contract

This document records the protocol audit of the gateway's two OpenAI-facing
surfaces and defines the compatibility contract used by regression tests. The
audit was performed against the OpenAI API reference and the current Vercel AI
SDK implementations of `@ai-sdk/openai-compatible` and `@ai-sdk/openai`.

The most important distinction is the backend selected for a request:

| Surface | Backend | Compatibility responsibility |
| --- | --- | --- |
| Chat Completions | Bedrock Converse | The gateway translates every request, chunk, tool call, and stop reason. |
| Chat Completions | bedrock-mantle | The gateway preserves upstream bytes and appends the Chat `[DONE]` sentinel. |
| Responses | Bedrock Converse | The gateway translates the full Responses item and event lifecycle. |
| Responses | bedrock-mantle | The gateway preserves upstream bytes without a `[DONE]` sentinel. |

The translation findings below apply to the Converse paths. Mantle paths must
not deserialize and reconstruct successful upstream responses; their remaining
risk is an upstream or transport failure after response headers have already
been sent.

## Findings that can interrupt an agent loop

### Responses tools were silently removed or changed

The previous implementation accepted every tool type at the JSON boundary but
also treated newer client-executed tools as if they were unavailable hosted
tools. In particular, `local_shell`, `shell`, and `apply_patch` became an
`Unknown` enum variant and disappeared from Bedrock `toolConfig`. A `custom`
free-form tool retained only its name and description; its grammar was
discarded and it was exposed to Bedrock as an empty-object function.

This can make the model print a textual imitation such as `<invoke
name="bash">...</invoke>` instead of producing a structured tool-use block. The
client then sees ordinary assistant text and does not execute the command.

The compatibility contract is:

1. Never silently remove a supported client-executed tool.
2. Translate client-executed tool kinds that can be represented safely.
3. Preserve each tool's original kind so the output item and continuation item
   use the same OpenAI wire type.
4. Silently omit server-hosted and unknown tools that Converse cannot execute;
   Codex may bundle them with real client tools, so rejecting the entire request
   would interrupt an otherwise valid agent turn.
5. Preserve namespace and tool names without inventing a client-visible name.

OpenAI reference:

- <https://developers.openai.com/api/reference/resources/responses/methods/create>
- <https://developers.openai.com/api/reference/resources/responses/streaming-events>

AI SDK reference implementations:

- <https://github.com/vercel/ai/blob/main/packages/openai/src/responses/openai-responses-prepare-tools.ts>
- <https://github.com/vercel/ai/blob/main/packages/openai/src/responses/openai-responses-language-model.ts>

### Bedrock reasoning signatures were not round-tripped

Bedrock extended thinking requires the complete, unmodified thinking block,
including its signature, when a tool-use turn is continued. The previous Chat
stream used the signature only as a signal to close an inline `<think>` block.
The Responses mapper omitted the signature from output and discarded incoming
reasoning items.

This breaks the critical sequence:

```
assistant reasoning + tool call -> client tool result -> next Bedrock request
```

AWS requires the reasoning block and signature to be replayed for this sequence:

- <https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html>
- <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlockDelta.html>

The Responses compatibility contract is to carry an opaque, versioned gateway
envelope in `reasoning.encrypted_content`. Clients already treat this field as
opaque and replay it when `store: false`. The gateway decodes the envelope and
reconstructs the Bedrock `reasoningContent` block without changing its text,
provider signature, or redacted content. Bedrock authenticates the replayed
text/signature pair; malformed envelopes are rejected locally.

Chat Completions has no standard field that can carry a Bedrock reasoning
signature through an arbitrary OpenAI client. Consequently, the gateway must
not claim that extended-thinking tool continuation is supported on Chat. It
must either avoid enabling extended thinking for a tool request or reject the
unsupported combination explicitly. Ordinary Chat tool calling remains
supported.

### Chat streaming could lose the terminal finish reason

When a Bedrock `messageStop` arrived while the inline `<think>` block was open,
the previous state machine emitted only `</think>` and permanently lost the
stop reason. This removed `finish_reason: "tool_calls"`, `"length"`, or
`"stop"` from the stream.

The contract is that every successful Chat stream emits exactly one terminal
choice chunk with a non-null finish reason. Closing `</think>` and publishing
the finish reason may happen in the same chunk.

### Output-token limits could truncate tool JSON unexpectedly

The previous non-reasoning Chat path ignored `max_completion_tokens` and
inserted an unconditional `maxTokens: 2048` when the client supplied no legacy
`max_tokens`. OpenAI defines `max_completion_tokens` as the current output
limit, including reasoning tokens.

The contract is:

- Prefer `max_completion_tokens` over legacy `max_tokens` when both are present.
- Do not introduce an arbitrary 2,048-token cap when the client omitted both.
- Use the config-driven model maximum only where Bedrock requires an explicit
  value.
- A truncated Responses stream must end with `response.incomplete`; an
  unfinished tool input must never be marked `completed`.

### Tool-choice and conversation-state fields were accepted but ignored

Chat previously mapped `tool_choice: "none"` to Bedrock `auto`. Responses
ignored `tool_choice` entirely. Responses accepts `previous_response_id` and
`item_reference` for wire compatibility, but the Converse backend remains
stateless and cannot retrieve context from OpenAI-hosted storage.

The contract is:

- Implement `none`, `auto`, `required`, and a specific function where the
  selected Bedrock model supports them.
- Reject a forced tool choice that Bedrock does not support with extended
  thinking instead of forwarding a request known to fail.
- Do not pretend to implement server-side stored state. Accept and ignore
  `previous_response_id` and unresolved `item_reference` so clients that bundle
  them are not interrupted; callers must still replay full history.
- Full-history stateless requests remain supported.

## Streaming contract

### Chat Completions

- Tool deltas retain a stable `index`, `id`, and name.
- Argument fragments are forwarded verbatim and in order.
- The terminal choice carries `finish_reason`.
- A usage-only chunk is emitted only when `stream_options.include_usage` is
  true, followed by `data: [DONE]`.
- A provider failure after streaming starts is surfaced as an OpenAI error
  event and is logged with the request ID.

### Responses

- `response.created` and `response.in_progress` precede output events.
- Function arguments emit both
  `response.function_call_arguments.delta` and
  `response.function_call_arguments.done`.
- The done event includes the function name.
- `local_shell`, `shell`, and `apply_patch` delay `response.output_item.added`
  until their required `action` or `operation` payload is complete; emitting an
  empty placeholder makes the current AI SDK reject the stream chunk.
- Output items are only marked `completed` after Bedrock closes their content
  block.
- `max_tokens` and content filtering end with `response.incomplete`.
- Provider errors end with `response.failed`.
- The stream has no Chat-style `[DONE]` sentinel.

### Responses tool search (client-executed)

`{"type":"tool_search","execution":"client"}` and the `tool_search_call` /
`tool_search_output` input items are standard Responses types (zuno 0.160.1
sends `tool_search` on every request). Converse has no field for deferred tools
or for search results, so on a model whose `config/models.toml` entry carries
`tool_search` (Claude Sonnet/Haiku/Opus 4.5 and later, Fable/Mythos 5 and 5.1)
such a request goes to `InvokeModel` / `InvokeModelWithResponseStream` with an
Anthropic Messages body (`anthropic_version: bedrock-2023-05-31`, no beta). Every
other request, and every request when `DISABLE_TOOL_SEARCH=true`, stays on
Converse, where `tool_search` is dropped and search items return 400.

| Responses | Messages body |
| --- | --- |
| `tool_search` tool | custom tool `tool_search` (description and parameters verbatim) at the client's position, never deferred |
| `tool_search_call` | assistant `tool_use {id: call_id, name: "tool_search", input: arguments}` |
| `tool_search_output` | user `tool_result` with one `tool_reference {tool_name}` per loaded tool, or the text `No tools matched the query.` |
| tools in every `tool_search_output` of the history | appended after the request's tools with `defer_loading: true` and no `cache_control`; namespaced tools as `{namespace}__{name}`, a top-level function as `{name}__{name}` |

The deferred set is recomputed from the replayed history on every request,
deduplicated by name with the first definition kept, so a tool stays
byte-identical from the turn it was first referenced. That keeps the prompt
cache prefix and the preserved-thinking prefix check intact. Cache points
become `cache_control` on the block they follow; the tools breakpoint therefore
sits on the last non-deferred tool.

Output: Claude's `tool_search` call becomes `{"type":"tool_search_call",
"id":"ts_<id>","call_id":"<id>","execution":"client","status":"completed",
"arguments":{...}}` (arguments stay an object); in a stream it is emitted only as
`response.output_item.done`, after `message_stop`, with no
`function_call_arguments` events. A call of a loaded tool is a normal
`function_call` with the tool's own `name` and `namespace`. A `refusal` stop is
treated as Converse's `content_filtered` (`response.incomplete`, no executable
call); other stop reasons keep Converse's meaning.

The gateway answers 400 before calling Bedrock for an upstream-invalid tool
name, more than 10,000 loaded tools, a loaded tool that collides with a
declared tool, or loaded tools with no non-deferred tool. This lane never
strips prompt caching and retries: its cache points are gateway-generated.

## Request and response coverage

The Converse backend intentionally does not implement OpenAI-hosted execution
for web search, file search, code interpreter, image generation, computer use,
or MCP. These are not client-side function calls and cannot be emulated merely
by changing the wire shape. They are omitted from `toolConfig` without hiding
supported client-executed tools that appear in the same request.

The following client-executed forms are the compatibility target:

- Chat nested `function` tools.
- Responses flattened `function` tools.
- Responses `custom` tools through a reversible string-input adapter.
- Responses namespace tools without changing client-visible names.
- Responses local shell, shell, and apply-patch calls when supplied as
  client-executed tools.

## Required regression matrix

Tests must cover both non-streaming and streaming variants where applicable:

| Area | Required cases |
| --- | --- |
| Chat tools | auto, none, required, specific tool, parallel tool deltas |
| Chat reasoning | terminal finish reason with reasoning text and signature deltas |
| Chat limits | `max_completion_tokens`, legacy `max_tokens`, omitted limit |
| Responses functions | argument delta/done, output-item done, tool result continuation |
| Responses custom tools | free-form input, output type, continuation output |
| Responses shell tools | local shell, shell, apply patch request/output round trips |
| Responses unsupported tools | every hosted/unknown family is omitted while supported sibling tools remain |
| Responses reasoning | encrypted envelope round trip, malformed-envelope rejection, and provider signature replay |
| Responses terminal state | completed, max-token incomplete, content-filter incomplete, failed |
| Responses state | `store`, `previous_response_id`, and unresolved item references remain non-fatal and stateless |
| Responses tool search | the zuno golden requests map to the expected Messages bodies; `tool_search_call` done-only stream; loaded-tool `function_call` with namespace; refusal and max-token stops leave the call incomplete |
| Mantle | byte-exact passthrough and mid-stream truncation behavior |
| AI SDK fixtures | current `@ai-sdk/openai-compatible` Chat and `@ai-sdk/openai` Responses shapes |

Passing legacy Python golden fixtures is necessary but not sufficient. A
fixture that encodes a known protocol divergence must be updated together with
the implementation instead of preserving the divergence as "parity".
