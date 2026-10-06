# Models

This page explains which model IDs a request can name, how the gateway knows what each model accepts, and how to add
a model. The live answer to "what can I call" is always the model list:

```sh
curl -s http://localhost:8080/api/v1/models -H "Authorization: Bearer $API_KEY" | jq -r '.data[].id'
```

It lists the foundation models and inference profiles your account can call in the gateway's region, read from
Bedrock at start and again every `MODEL_CATALOG_REFRESH_SECS`, plus the short names below. A model that Bedrock
launches while the gateway runs appears there without a restart.

## Model IDs

A request names its model with any ID Bedrock accepts:

- a foundation model ID, such as `amazon.nova-pro-v1:0`, for a model your region serves on demand;
- a cross-region inference profile ID, such as `us.anthropic.claude-sonnet-4-5-20250929-v1:0` or
  `global.anthropic.claude-sonnet-5-5`, which Bedrock routes across regions. Most recent models are available only
  this way;
- an application inference profile ARN from your account;
- one of the gateway's short names, such as `gpt-5.5`, listed below.

The gateway sends the ID to Bedrock as you wrote it, so a profile keeps its routing. The prefixes `us.`, `eu.`,
`apac.`, `jp.`, `au.`, `ca.` and `global.` all work.

## Claude

Claude Sonnet, Opus, Haiku and Fable are served through Bedrock's Converse API, on both Chat Completions and
Responses. Recent versions are reached through inference profiles, for example `global.anthropic.claude-sonnet-5-5`,
`us.anthropic.claude-opus-5-5` or `global.anthropic.claude-fable-5-1`.

The gateway handles the differences between versions for you. On the versions that reject `temperature` and
`top_p`, it drops them. On the versions that think adaptively, `reasoning_effort` becomes the effort the model
expects. When a version does not accept a conversation that ends with an assistant message, the gateway adds a short
user turn that asks the model to continue.

`response_format` and the Responses `text.format` are honoured on Claude Sonnet 4.5 and 4.6, Haiku 4.5 and Opus 4.5
and 4.6, the versions on which Bedrock supports structured output. On the other versions the gateway answers with
HTTP 400 instead of sending a request that Bedrock would refuse.

## OpenAI GPT

OpenAI's models on Bedrock are called by short names:

| Name                                               | Served through                       | Chat Completions | Responses |
| -------------------------------------------------- | ------------------------------------ | ---------------- | --------- |
| `gpt-5.4`, `gpt-5.5`                               | Bedrock's OpenAI-compatible endpoint | Yes              | Yes       |
| `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`     | Bedrock's OpenAI-compatible endpoint | Yes              | Yes       |
| `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra` | Bedrock Converse, global profile | Yes              | Yes       |
| `gpt-oss-120b`, `gpt-oss-20b`                      | Bedrock's OpenAI-compatible endpoint | Yes              | No        |

- **GPT-5.x and gpt-oss** go to Bedrock's OpenAI-compatible endpoint, which needs a Bedrock API key
  (`AWS_BEARER_TOKEN_BEDROCK`) and is available in a few regions only: `us-east-1`, `us-east-2` and `us-west-2`,
  depending on the model. In another region the gateway answers with HTTP 400. On Chat Completions, GPT-5.x is served
  by translating to and from the Responses API, and its reasoning summary appears in `<think>` tags.
- **GPT-6.x** goes through Bedrock Converse with the `global.openai.*` inference profile, so it needs no Bedrock API
  key and works wherever that profile does. The `us.openai.*` and `global.openai.*` IDs work as well.
  `reasoning_effort` becomes `reasoning.effort`, `temperature` and `top_p` are dropped because these models reject
  them, and structured output uses the strict JSON schema the models support.

## Other models

Amazon Nova, DeepSeek and every other model in your catalog are served through Bedrock Converse with no extra setup.
A model's limits are Bedrock's: when a model does not support images, tools or streaming, Bedrock's error comes back
to the client.

Embeddings use their own registry. The [API reference](../reference/api.md#embeddings) lists the supported families.

## How the gateway knows a model

What each model accepts lives in a registry, [`config/models.toml`](../../../config/models.toml), not in the code.
An entry matches a fragment of the model ID and sets, for example:

- whether the model rejects `temperature` and `top_p`;
- which form of reasoning it takes;
- the smallest prompt prefix worth caching, and whether it supports a 1-hour cache;
- whether it supports structured output;
- which backend serves it, and in which regions.

A model with no entry is served with the defaults: no reasoning parameters, no structured output, and no prompt
caching, except for a Claude ID, which gets a conservative cache threshold.

## Add or change a model

1. Copy `config/models.toml` from the release you run into a directory, for example `/etc/bedrock-gateway/config/`.
2. Add or change an entry:

   ```toml
   [[model]]
   match = "provider.model-name"
   capabilities = ["drop_sampling_params", "structured_output"]
   [model.params]
   cache_min_tokens = 1024
   reasoning_path = "adaptive_thinking"
   ```

3. Start the gateway with `CONFIG_DIR` pointing at that directory.

When two entries match one ID, the first one in the file supplies the parameters, so put a more specific entry, such
as `claude-sonnet-5-5`, above a shorter one, such as `claude-sonnet-5`. The capability flags of every matching entry
add up. The comments at the top of `models.toml` describe every flag and parameter.

A short name is an `[[alias]]` entry, which must come before the first `[[model]]`:

```toml
[[alias]]
from = "my-model"
to = "global.provider.model-name"
```
