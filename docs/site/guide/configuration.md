# Configuration

bedrock-gateway is configured with environment variables. This page lists every one, grouped by what it controls,
with its default. Only two things are required: a key for your clients and access to Bedrock.

A setting can also come from `config/app.toml` in the working directory, and a one-word name such as `PORT` also
from `APP_PORT`. The variables on this page take precedence over both.

## The key your clients send

Every route except `/health` asks for this key as `Authorization: Bearer <key>`. The gateway refuses to start
without one. It looks in three places, in this order, and uses the first that is set:

| Variable             | Where the key comes from                                                        |
| -------------------- | ------------------------------------------------------------------------------- |
| `API_KEY_PARAM_NAME` | An SSM Parameter Store parameter with that name, read with decryption            |
| `API_KEY_SECRET_ARN` | A Secrets Manager secret whose value is JSON with an `api_key` field             |
| `API_KEY`            | The variable itself                                                             |

The gateway reads the parameter or secret once, at start. To change the key, update it and restart the gateway.

## Access to Bedrock

| Variable                   | Default     | Meaning                                                                 |
| -------------------------- | ----------- | ----------------------------------------------------------------------- |
| `AWS_REGION`               | `us-west-2` | The region whose Bedrock endpoints the gateway calls                    |
| `AWS_BEARER_TOKEN_BEDROCK` | none        | A Bedrock API key. `BEDROCK_API_KEY` is accepted as another name for it |

With a Bedrock API key, the gateway authenticates to Bedrock with it. Without one, it uses the standard AWS
credential chain: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, then `AWS_PROFILE`, then the role of the EC2
instance, ECS task or Lambda function. The two directions are separate: `API_KEY` is what your clients send to the
gateway, and the Bedrock API key is what the gateway sends to AWS.

GPT-5.x and gpt-oss are served through Bedrock's OpenAI-compatible endpoint, which accepts only a Bedrock API key.
Without one the gateway starts, logs a warning, and those models answer with an error; every other model works.

| Variable                        | Default                                                | Meaning                                                    |
| ------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| `DISABLE_MANTLE`                | `false`                                                | Turn off the OpenAI-compatible endpoint and its models     |
| `MANTLE_BASE_URL_TEMPLATE`      | `https://bedrock-mantle.{region}.api.aws/openai/v1`    | Where GPT-5.x Responses requests go; `{region}` is `AWS_REGION` |
| `MANTLE_CHAT_BASE_URL_TEMPLATE` | `https://bedrock-mantle.{region}.api.aws/v1`           | Where gpt-oss Chat Completions requests go                 |

## Server

| Variable           | Default   | Meaning                                                                  |
| ------------------ | --------- | ------------------------------------------------------------------------ |
| `PORT`             | `8080`    | The port the gateway listens on                                          |
| `BIND_ADDR`        | `0.0.0.0` | The address it binds; `127.0.0.1` keeps it to the local machine         |
| `API_ROUTE_PREFIX` | `/api/v1` | The path every route is served under                                     |
| `MAX_BODY_SIZE_MB` | `20`      | The largest request body accepted, so that base64 images fit             |

The gateway serves plain HTTP. Put a load balancer or a reverse proxy with TLS in front of it when clients reach it
over a network.

## Models

| Variable                                | Default                                     | Meaning                                                                                  |
| --------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ENABLE_CROSS_REGION_INFERENCE`         | `true`                                      | List Bedrock's cross-region inference profiles, such as `us.` and `global.` IDs           |
| `ENABLE_APPLICATION_INFERENCE_PROFILES` | `true`                                      | List the application inference profiles in your account                                  |
| `ALLOWED_MODELS`                        | all                                         | A comma-separated list of ID fragments; `/models` lists only the IDs that contain one     |
| `MODEL_CATALOG_REFRESH_SECS`            | `3600`                                      | How often the model list is read again from Bedrock; `0` reads it only at start           |
| `DEFAULT_MODEL`                         | `anthropic.claude-3-5-sonnet-20241022-v2:0` | The ID `/models` lists when Bedrock returns no model the gateway can serve                |
| `CONFIG_DIR`                            | `config`                                    | The directory with your own `models.toml`, `regions.toml` and `embeddings.toml`           |

`ALLOWED_MODELS` narrows what `/models` lists. It does not block a request for another model, but the gateway
refuses images in such a request, because it learns which models take images from the list.
[Models](models.md) explains the model IDs and the model registry.

## Prompt caching

| Variable                | Default | Meaning                                                                    |
| ----------------------- | ------- | -------------------------------------------------------------------------- |
| `ENABLE_PROMPT_CACHING` | `true`  | Place cache points automatically on the models that support caching        |
| `PROMPT_CACHE_TTL`      | `5m`    | How long a cache entry lives: `5m`, or `1h` on the models that support it   |

A request can override both through `extra_body`; see [Caching and reasoning](caching-and-reasoning.md#prompt-caching).

## Reasoning across tool calls

Claude's thinking and GPT's reasoning come with a signature that the model needs to see again when a conversation
continues after a tool call. The Chat Completions format has no field for it, so the gateway can carry it inside the
tool call's ID, signed with a key of yours so that it cannot be forged. The Responses API carries reasoning in its own
items and does not need this.

Without the key, a Claude request on Chat Completions that has tools runs with reasoning turned off and still
succeeds. A GPT-5.x answer on Chat Completions that combines reasoning with a tool call fails instead, with an error
that names `CHAT_REASONING_CAPSULE_ENABLED`. Turn the feature on when your clients use tools on Chat Completions.

| Variable                            | Default | Meaning                                                                      |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------- |
| `CHAT_REASONING_CAPSULE_ENABLED`    | `false` | Sign reasoning into tool call IDs                                            |
| `CHAT_REASONING_CAPSULE_ACTIVE_KID` | none    | The ID of the key that signs new tool call IDs                               |
| `CHAT_REASONING_CAPSULE_KEYS`       | none    | `kid:key` pairs separated by commas; each key is URL-safe base64 without padding |

Create a key and turn the feature on:

```sh
export CHAT_REASONING_CAPSULE_KEYS="k1:$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=')"
export CHAT_REASONING_CAPSULE_ACTIVE_KID=k1
export CHAT_REASONING_CAPSULE_ENABLED=true
```

Every instance behind the same load balancer needs the same keys. To replace a key, add the new pair, make it the
active one, and remove the old pair once the conversations that used it are over. The gateway refuses to start when
the feature is on and the active key is missing.

## Timeouts and retries

| Variable                             | Default | Meaning                                                                              |
| ------------------------------------ | ------- | ------------------------------------------------------------------------------------ |
| `AWS_CONNECT_TIMEOUT_SECS`           | `60`    | How long to wait for a connection to AWS                                             |
| `AWS_READ_TIMEOUT_SECS`              | `900`   | How long to wait for data from AWS, long enough for a long stream                     |
| `RESPONSES_STREAM_IDLE_TIMEOUT_SECS` | `180`   | How long a Responses stream may stay silent before it ends with `response.failed`     |
| `AWS_MAX_RETRY_ATTEMPTS`             | `8`     | How many times a throttled or failed call to AWS is tried                            |

## Logging

| Variable                      | Default | Meaning                                                                          |
| ----------------------------- | ------- | -------------------------------------------------------------------------------- |
| `LOG_LEVEL`                   | `info`  | `error`, `warn`, `info`, `debug` or `trace`, or filter directives in the `RUST_LOG` form |
| `DEBUG`                       | `false` | Print logs as readable text instead of one JSON object per line                 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none    | Send traces and metrics to this OTLP endpoint; needs a build with `--features otel` |

[Observability](../operate/observability.md) describes what is logged.
