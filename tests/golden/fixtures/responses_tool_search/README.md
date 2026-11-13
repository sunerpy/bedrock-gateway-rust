# Responses tool search: zuno 0.160.1 recording

Requests and fake-model answers recorded on 2026-11-13 from zuno 0.160.1 (`zuno app-server`) against a scripted Responses endpoint and a stub MCP server. The recording was sanitized before it was copied here: zuno's base instructions, skills, permissions, environment texts, session/turn/installation IDs and ports are placeholders.

| File | Content |
|---|---|
| `request-1.json` | First request. `tools` holds seven functions, the namespace `mcp__ze_tools` (two direct tools) and `{"type":"tool_search","execution":"client"}`. |
| `model-output-1.json` | The fake model's events: one `tool_search_call` (`arguments` is an object). |
| `request-2.json` | `input` adds that `tool_search_call` and its `tool_search_output`, which loads the namespace `mcp__ze_tools_deferred` with `ticket_search` and `ticket_get`. |
| `request-2-no-match.json` | The same step when the search found nothing (`tools: []`). |
| `model-output-2.json` | The fake model calls a loaded tool: `function_call` with `name: ticket_search`, `namespace: mcp__ze_tools_deferred`. |
| `request-3.json` | `input` adds that `function_call` and its `function_call_output`. |

The gateway tests send these with `model` set to a Claude profile and `max_output_tokens` set, the two fields the client's model broker rewrites in production. They check the InvokeModel Messages body built for each request and the Responses items built from Claude's answers (`src/bedrock/responses_provider_tests.rs`, `src/bedrock/anthropic_messages_tests.rs`).
