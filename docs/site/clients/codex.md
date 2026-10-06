# Codex CLI

Codex CLI speaks the Responses API, and the gateway serves it, so Codex joins it as a custom model provider. Every
model your account can call works, Claude and Nova included.

## Add the provider

Put the provider in `~/.codex/config.toml`:

```toml
model = "global.anthropic.claude-sonnet-5-5"
model_provider = "bgw"

[model_providers.bgw]
name = "bedrock-gateway"
base_url = "http://localhost:8080/api/v1"
env_key = "BGW_API_KEY"
wire_api = "responses"
```

Export the gateway's key in the shell that starts Codex, then start it as usual. `-m` picks another model for one
run:

```sh
export BGW_API_KEY="<the gateway's API_KEY>"
codex
codex exec -m us.amazon.nova-2-lite-v1:0 "Reply with exactly: BEDROCK_OK"
```

The last command prints `BEDROCK_OK`.

## What the settings do

| Setting          | Why                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `model_provider` | Sends Codex's requests to the provider below instead of to OpenAI.                          |
| `base_url`       | The gateway's base URL. Codex adds `/responses` to it.                                      |
| `env_key`        | The environment variable that holds the key. Codex sends it as `Authorization: Bearer`.     |
| `wire_api`       | `responses`, the API that the gateway serves and Codex speaks.                              |
| `model`          | Any ID from `GET /api/v1/models`, or a short name such as `gpt-5.5`.                        |

## Reasoning

`model_reasoning_effort` sets the effort Codex asks for, and the gateway turns it into what the model takes; see
[Caching and reasoning](../guide/caching-and-reasoning.md#reasoning):

```toml
model_reasoning_effort = "high"
```

Codex runs without stored state on the server: it sends the whole conversation with each request, and the model's
reasoning comes back to it as encrypted content that it sends again on the next turn. That keeps a reasoning model's
thinking intact across tool calls without any setting on the gateway.

## Context window

Codex knows the context windows of OpenAI's own models only. For any other ID it prints that it has no metadata for
the model and uses defaults. Tell it the model's window so that it compacts the conversation at the right point, for
example for Claude Sonnet 5.5:

```toml
model_context_window = 1000000
```

## Tools

Codex's own tools, such as the shell and file edits, are function and custom tools that Codex runs itself, and they
pass through the gateway. Tools that OpenAI runs on its servers, such as web search, have no counterpart on Bedrock;
the gateway leaves them out of the request instead of failing it, so the model works without them.
