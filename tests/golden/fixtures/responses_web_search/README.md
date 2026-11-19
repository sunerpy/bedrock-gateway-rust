# Responses web search (gateway-executed): fixtures

Hand-built fixtures for the web search loop of plan C2 (`src/bedrock/responses_web_search.rs`). They follow the request shape of zuno 0.160.1 (`tests/golden/fixtures/responses_tool_search/`) and the replay shape recorded by the P1 probe (pinned zuno 0.160.1 against a scripted Responses endpoint, 2026-11-19). The model broker's header and hosted tool follow Weniq's `internal/brokers/model/websearch.go`. Signatures are synthetic and the contents are made up or sanitized: none of these bytes came from Bedrock or AgentCore.

The tests load the files with `include_str!` from `src/bedrock/responses_web_search_tests.rs`, `src/bedrock/responses_provider_tests.rs` and `src/bedrock/responses_stream_tests.rs`, not through `corpus.rs`.

| File | Content |
|---|---|
| `request-1.json` | zuno request: functions `get_goal` and `lookup_ticket`, namespace `mcp__ze_tools` (two tools), the hosted `{"type":"web_search","external_web_access":false}` the broker adds, `tool_choice: "auto"`, `parallel_tool_calls: true`, `include: ["reasoning.encrypted_content"]`, `reasoning.effort: "medium"`, `max_output_tokens: 32000`; one user question after the developer and environment messages. |
| `toolsearch-request-1.json` | The same request with a client `tool_search` tool, which puts it on the InvokeModel lane. |
| `converse-1.json`, `converse-stream-1.jsonl` | Step 0 (Converse output, and the same as ConverseStream events): signed thinking, a short text, one `web_search` call; `tool_use`. |
| `agentcore-1.json` | The AgentCore `InvokeWebSearch` body for that call: two results, the second without `publishedAt`. |
| `converse-2.json`, `converse-stream-2.jsonl` | Step 1: signed thinking and an answer with two markdown links, one to a result and one to a page no search returned; `end_turn`. |
| `messages-1.json`, `messages-2.json`, `messages-stream-1.jsonl` | Steps 0 and 1 as InvokeModel Messages responses, and step 0 as Messages stream events. |
| `expected-response.json` | The response's `output` for that scenario, the same on all four paths (Converse and InvokeModel, streamed or not): item ids with the response id replaced by `S`, every `encrypted_content` decoded to its payload. |
| `parallel-converse-1.json` | One step with two `web_search` calls. |
| `mixed-converse-1.json` | One step with signed thinking, the client function `lookup_ticket` and a `web_search` call. |
| `undeclared-requests.json` | Requests that do not declare web search (header absent or `off`, no marks; or `on` for a model without the capability): history with unmarked reasoning, a replayed GPT turn with a hosted search, a client tool named `web_search`, Nova. |
| `undeclared-stream-1.jsonl`, `undeclared-stream-2.jsonl` | ConverseStream events for those requests (a tool turn; a `max_tokens` stop). |
| `undeclared-expected.json` | What the previous code (commit `9430024`, before the loop existed) assembled for `undeclared-requests.json` and for the tool-search goldens on both lanes (Converse slots, tool registry, InvokeModel body bytes), and the stream events it emitted for the two JSONL files. Recorded once by a throwaway test; the tests require the current code to reproduce it exactly. |
