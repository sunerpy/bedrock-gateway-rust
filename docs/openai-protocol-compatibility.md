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
   would interrupt an otherwise valid agent turn. The one hosted tool the
   gateway runs itself is web search for Claude, and only on a request that
   declares it through the `X-Weniq-Web-Search` header or its own history (see
   "Responses web search (gateway-executed)" below); the body's hosted
   `web_search` tool is still omitted.
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

A conversation can switch between a model the gateway translates (Claude on
Converse or InvokeModel) and a bedrock-mantle model (GPT). The client then
replays the reasoning items of both providers in one `input`, and each provider
can read only its own: Bedrock cannot verify OpenAI's ciphertext, and mantle
rejects the gateway envelope with HTTP 400 `invalid encrypted reasoning`. The
gateway therefore drops the other provider's reasoning on both paths:

- Converse and the tool-search lane replay only reasoning items whose
  `encrypted_content` starts with the envelope prefix `bedrock-reasoning-v1:`.
  Every other reasoning item, including one without `encrypted_content`, is
  skipped. A malformed envelope with the right prefix is still rejected with
  HTTP 400.
- The mantle passthrough removes the reasoning items that carry that prefix
  from the `input` array before forwarding and leaves everything else as it
  was. A body without such items is forwarded exactly as before, with only
  `model` patched.

Only reasoning that the receiving provider cannot read is dropped; the messages
and tool calls of the other model's turns are still replayed.

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
- `max_tokens` and content filtering end with `response.incomplete`, except
  that an answer that already ran a gateway-executed web search ends either
  stop with `response.completed` and a gateway message (see "Responses web
  search (gateway-executed)").
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

### Responses web search (gateway-executed)

The gateway can run web search for Claude on `/responses` itself, with Amazon
Bedrock AgentCore Web Search (`WEB_SEARCH_BACKEND=agentcore`). OpenAI's hosted
`web_search` tool in the body plays no part: a request opts in with the request
header `X-Weniq-Web-Search: on` (`off` opts out; any other value is a 400). A
header, not a body field, because the mantle lane forwards the body verbatim.
Mantle-backed models ignore the header.

**Declaration and execution.** A request *declares* the synthetic tool when the
header is `on`, a backend is configured and the model has the `web_search`
capability (`config/models.toml`: Claude Sonnet 5.5 and Opus 5.5), or when its
history carries the gateway's web search marks. It *executes* searches only in
the first case. A marked history keeps the tool declared with the header `off`
or absent (calls answer `disabled_for_this_message`) and with the backend off or
on a model without the capability (calls answer `unavailable`): Claude 5.5 binds
every thinking block to the `tools` it was signed under, so the tool list must
not change once a conversation declared the tool. The first declaration drops
the replayed thinking that has no mark, which was signed without the tool;
removing thinking from the start of a history is allowed. A request without
marks in the middle of a tool loop (its input ends with tool outputs) whose
current answer holds gateway thinking does not declare the tool even when it
could run it: the first declaration would drop that thinking, the thinking of
the loop's last assistant message included. Search then starts with the next
user message. Every other request takes the single-call path unchanged.

**The synthetic tool.** `web_search` with a frozen description and the schema
`{"query": string}`, appended after every client tool (the tool-search lane's
`tool_search` included) and before the tools cache point. A client tool that
translates to the same name is a 400 on a declared request only.

**The loop.** A step whose stop reason is `tool_use` and whose calls are all
`web_search` is executed by the gateway: each call, in block order, runs one
AgentCore search (or gets an error result without a search), and the gateway
calls Bedrock again. A step that also calls client tools runs its searches and
hands the client calls back (`completed`). Any other step is the answer. Each
continuation is built from the step's own output items through the replay
translation, so the client's next request rebuilds the same prefix; only the
remaining `max_output_tokens` and `tool_choice` (`required` becomes `auto` after
the first search step) change between steps. An input that ends with an
assistant item on a model that does not accept prefill gets the continuation
user turn before the first step, and every later step keeps it (the first
step's thinking is bound to it). A later replay does not rebuild that turn
(neither does one without web search), so that thinking can still fail
Bedrock's check on the client's next request. Usage is summed over the steps
through the one token formula.

**Output items.** Per search step: the step's reasoning item (its envelope
carries `"ws": 1`), its message item, one `web_search_call` per executed search
(`id` `ws_bgw_<toolUseId>`, `action: {type: "search", query, queries, sources}`,
`status` `completed` or `failed`; a call that was refused has no item), the
client calls of a mixed step, and last a *carrier*: a reasoning item with an
empty `summary` whose envelope (payload version 2) holds the step's `toolUse`
and `toolResult` blocks and the answer's search count. Replaying the carrier
restores the hidden blocks; replayed `web_search_call` items are skipped like
mantle's. In a mixed step the user message's results are put in the assistant
message's call order. Item ids are unique within a response: step `k` uses
`rs_S_k` / `msg_S_k`, carriers `rs_S_wsk`. Markdown links in the answer that
point at a search result get `url_citation` annotations (character indexes,
half-open); in a stream they arrive as `response.output_text.annotation.added`
before `response.output_text.done`.

**Stream events.** Items are strictly serial (zuno tracks one active item).
Search items are emitted after the step's Bedrock stream ends:
`output_item.added` (with the query), `response.web_search_call.in_progress`,
`response.web_search_call.searching`, `response.web_search_call.completed`
(not for a failed search), `output_item.done`; then the carrier's
`output_item.added` / `output_item.done`. Client calls of a declared request are
emitted after their step, not streamed. `response.created`,
`response.in_progress` and the terminal event appear once per response, and
`sequence_number` counts across all steps.

**Limits and terminal states.** At most `WEB_SEARCH_MAX_CALLS` (default 5)
real searches per answer, counted across the client's continuation requests
through the carriers and reset by a new user message; a gateway
`web_search_call` replayed after the newest carrier (its step's carrier never
reached the client) counts too. Further calls answer `max_uses_exceeded`. At
most `max_calls + 2` Bedrock calls per response. Once the answer has run a real
search, the response never ends `incomplete` (zuno retries `incomplete`,
`content_filter` included) and never ends `completed` without an answer: the
step limit, less than 1,024 output tokens left before a continuation, a
`max_tokens` or content-filter stop, or a final step with neither an answer
nor a client call end `completed` with a gateway-written message
(`msg_S_stop`) that says why; a call cut off by such a stop is not handed back.
Without a real search the single-call terminal states apply, except that the
step limit also ends with the gateway message, which then says that web search
is off for the message, unavailable, or never ran.

**Errors.** AgentCore failures never become HTTP errors: they are error payloads
in the call's `toolResult` (`too_many_requests`, `unavailable`,
`invalid_tool_input`) and a `failed` item. The first Bedrock call of a stream
fails as an HTTP error, as on the single-call path; a later step's failure is
`response.failed`, with `invalid_prompt` when Bedrock rejects a thinking block
bound to another prefix (retrying cannot help) and the shared mapping
otherwise. Non-stream requests return any step's error as the HTTP error.

**Rollback.** A gateway that predates web search answers 400 to a history with
a carrier and stops declaring the tool. Do not roll back past this feature;
turn `WEB_SEARCH_BACKEND` off instead, which keeps searched conversations
working.

## Request and response coverage

The Converse backend intentionally does not implement OpenAI-hosted execution
for web search, file search, code interpreter, image generation, computer use,
or MCP. These are not client-side function calls and cannot be emulated merely
by changing the wire shape. They are omitted from `toolConfig` without hiding
supported client-executed tools that appear in the same request. Web search is
the one exception, and only through the gateway's own tool on a request that
declares it (see "Responses web search (gateway-executed)"); a hosted
`web_search` tool in the body is still omitted.

A hosted tool run can still reach these paths as history. When a conversation
that used hosted tools on a bedrock-mantle (GPT) model continues on a model
served by Converse or InvokeModel, codex-based clients such as zuno replay its
`web_search_call` and `image_generation_call` items in `input`. The search
item carries the action and query but not the results, and the image is not
output of the Bedrock model, so the gateway skips both: they add no Bedrock
block, and the messages around them keep their order, same-role merging, and
reasoning boundaries. The gateway's own `web_search_call` items are skipped
the same way; their blocks come back from the carrier. Going the other way, the
mantle lane drops the gateway's `web_search_call` items (`ws_bgw_` ids) along
with its reasoning envelopes and carriers before forwarding. Input item types that the gateway neither translates nor
skips, including the call items of the other hosted tools, are still rejected
with HTTP 400.

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
| Responses reasoning from another provider | Converse and tool-search lanes skip reasoning without the gateway envelope (OpenAI ciphertext or none) and still reject a malformed envelope; mantle drops gateway-envelope reasoning from `input`, forwards a body without any byte-stable, and leaves a string `input` alone |
| Responses terminal state | completed, max-token incomplete, content-filter incomplete, failed |
| Responses state | `store`, `previous_response_id`, and unresolved item references remain non-fatal and stateless |
| Responses replayed hosted calls | `web_search_call` and `image_generation_call` input items add nothing on the Converse and tool-search lanes; the turns around them keep their order, merging, and reasoning boundaries |
| Responses tool search | the zuno golden requests map to the expected Messages bodies; `tool_search_call` done-only stream; loaded-tool `function_call` with namespace; refusal and max-token stops leave the call incomplete |
| Responses web search | requests that do not declare it keep the single-call bytes; the declaration rules (header, backend, capability, sticky marks, first declaration); the frozen tool's position and the reserved name; search-only, parallel and mixed steps on both lanes, streamed and not; the loop's continuation equals the replay's rebuild (tools, system, message prefix); carrier and marker codecs; refused, failed and invalid calls; the per-answer count across continuations; the step limit, the output budget, `max_tokens`, a content-filter stop and an empty answer after a search end `completed` with the gateway message; the step limit without a search names the gate; a tool loop with unmarked thinking does not declare the tool; the continuation turn of an input that ends on an assistant item stays in every step; gateway searches without a carrier count toward the answer; stream event order and unique ids; `invalid_prompt` for a later binding error; mantle drops the gateway's items |
| Mantle | byte-exact passthrough and mid-stream truncation behavior |
| AI SDK fixtures | current `@ai-sdk/openai-compatible` Chat and `@ai-sdk/openai` Responses shapes |

Passing legacy Python golden fixtures is necessary but not sufficient. A
fixture that encodes a known protocol divergence must be updated together with
the implementation instead of preserving the divergence as "parity".
