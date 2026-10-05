# Editors and agents

Editors and coding agents that let you add an OpenAI-compatible provider reach the gateway with a base URL, a key and
the model IDs you want to use. This page shows OpenCode and Zed; another client with such a provider takes the same
three things.

## OpenCode

[OpenCode](https://opencode.ai) reaches the gateway through its `@ai-sdk/openai-compatible` package, which speaks
Chat Completions. Put the provider in `~/.config/opencode/opencode.json`, or in `opencode.json` at the root of a
project, and list the models you want to pick from:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bgw": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "bedrock-gateway",
      "options": {
        "baseURL": "http://localhost:8080/api/v1",
        "apiKey": "{env:BGW_API_KEY}"
      },
      "models": {
        "global.anthropic.claude-sonnet-5-5": {
          "name": "Claude Sonnet 5.5",
          "limit": { "context": 1000000, "output": 128000 }
        },
        "us.amazon.nova-2-lite-v1:0": {
          "name": "Amazon Nova 2 Lite"
        }
      }
    }
  },
  "model": "bgw/global.anthropic.claude-sonnet-5-5"
}
```

Export the key in the shell that starts OpenCode. `/models` switches between the listed models, and `-m` picks one
for a single run:

```sh
export BGW_API_KEY="<the gateway's API_KEY>"
opencode
opencode run -m bgw/us.amazon.nova-2-lite-v1:0 "Reply with exactly: BEDROCK_OK"
```

The last command prints `BEDROCK_OK`. `limit` gives OpenCode the context and output sizes it plans with; take them
from the model's documentation on Bedrock.

OpenCode works with tools on every turn. When a reasoning model such as Claude should think between tool calls, turn
on the [signing key](../guide/configuration.md#reasoning-across-tool-calls); without it, the gateway serves those
requests with reasoning off.

## Zed

Zed reaches the gateway in two places: its agent, through an OpenAI-compatible language model provider, and its edit
prediction, through the Completions route.

### The agent

Add the provider under `language_models` in Zed's `settings.json`:

```json
{
  "language_models": {
    "openai_compatible": {
      "bedrock-gateway": {
        "api_url": "http://localhost:8080/api/v1",
        "available_models": [
          {
            "name": "global.anthropic.claude-sonnet-5-5",
            "display_name": "Claude Sonnet 5.5",
            "max_tokens": 1000000,
            "max_output_tokens": 128000,
            "capabilities": {
              "tools": true,
              "images": true,
              "parallel_tool_calls": true,
              "prompt_cache_key": false
            }
          }
        ]
      }
    }
  }
}
```

Zed reads the key from the environment variable `BEDROCK_GATEWAY_API_KEY`, named after the provider, or from the key
you enter in the agent's settings. `prompt_cache_key` stays `false`: the gateway places cache points by itself.

### Edit prediction

Zed's OpenAI-compatible edit prediction sends a prompt to a Completions route and reads the text that comes back:

```json
{
  "edit_predictions": {
    "provider": "open_ai_compatible_api",
    "open_ai_compatible_api": {
      "api_url": "http://localhost:8080/api/v1/completions",
      "model": "us.amazon.nova-2-lite-v1:0"
    }
  }
}
```

Here `api_url` is the whole URL of the route, not the base URL. Zed reads the key from
`ZED_OPEN_AI_COMPATIBLE_EDIT_PREDICTION_API_KEY` or from the key you enter in its edit prediction settings. A small,
fast model keeps the predictions quick.

## Other clients

A client that asks for an "OpenAI-compatible" or "custom OpenAI" provider usually takes these values:

| Field                   | Value                                                         |
| ----------------------- | ------------------------------------------------------------- |
| Base URL or API base    | `http://localhost:8080/api/v1`                                |
| API key                 | The gateway's `API_KEY`                                       |
| Model                   | An ID from `GET /api/v1/models`, or a short name like `gpt-5.5` |

When a client asks for a full URL instead of a base URL, add the route, such as `/chat/completions`.
