# API reference

The gateway serves these routes under `API_ROUTE_PREFIX`, which is `/api/v1` unless you change it. Each takes the
request an OpenAI client sends and answers in OpenAI's shape. This page lists what each route accepts and how it
differs from OpenAI; the [protocol compatibility](../../openai-protocol-compatibility.md) reference covers the
streaming and tool contracts in depth.

| Method | Route                          | Serves                                        |
| ------ | ------------------------------ | --------------------------------------------- |
| `POST` | `/api/v1/chat/completions`     | [Chat Completions](#chat-completions)         |
| `POST` | `/api/v1/responses`            | [Responses](#responses)                       |
| `POST` | `/api/v1/completions`          | [Completions](#completions)                   |
| `POST` | `/api/v1/embeddings`           | [Embeddings](#embeddings)                     |
| `GET`  | `/api/v1/models`               | [The model list](#models)                     |
| `GET`  | `/api/v1/models/{id}`          | [One model](#models)                          |
| `GET`  | `/api/v1/health`               | [Liveness](#health)                           |

## Requests

Every route except `/health` needs `Authorization: Bearer <API_KEY>`. Bodies are JSON, up to `MAX_BODY_SIZE_MB`
(20 MB by default) including base64 images.

A client can send `x-request-id` with its own ID for the request; the gateway's log lines for that request then carry
it. The gateway answers cross-origin requests from any origin, so a browser page that holds the key can call it.

## Errors

An error comes back in OpenAI's shape:

```json
{"error": {"message": "model `nope` not found", "type": "invalid_request_error", "code": "bad_request"}}
```

| Status | `code`                | When                                                                                                  |
| ------ | --------------------- | ----------------------------------------------------------------------------------------------------- |
| 400    | `bad_request`         | The request is malformed, names an unknown model, asks a model for a feature it lacks, names a model outside the gateway's region, or Bedrock rejects it as invalid |
| 400    | `unsupported`         | The request uses a field the gateway does not serve, such as `suffix` on Completions                  |
| 401    | `unauthorized`        | The key is missing or wrong, or Bedrock denied the gateway's AWS identity access to the model         |
| 405    |                       | The route exists, but not for this method                                                             |
| 413    |                       | The body is larger than `MAX_BODY_SIZE_MB`                                                            |
| 429    | `rate_limit_exceeded` | Bedrock throttled the request after the gateway's retries                                             |
| 500    | `internal_error`      | The gateway failed, for example when an image URL could not be downloaded                             |
| 502    | `upstream_error`      | Bedrock reported a failure of the model or the service, such as a model that is not ready             |

An error after a stream has started arrives in the stream: as an error event on Chat Completions, and as a
`response.failed` event on Responses, whose `error.code` is one that Codex CLI acts on: `rate_limit_exceeded`,
`server_is_overloaded`, `context_length_exceeded` or `server_error`. An error from Bedrock's OpenAI-compatible
endpoint, which serves GPT-5.x and gpt-oss, keeps the status and the error object that endpoint sent.

## Chat Completions

`POST /api/v1/chat/completions` takes OpenAI's chat request:

| Field                                  | Handling                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `model`                                | Any model ID or short name; see [Models](../guide/models.md)                                     |
| `messages`                             | `system`, `developer`, `user`, `assistant` and `tool` messages; text and `image_url` parts        |
| `stream`, `stream_options`             | Server-sent events; `include_usage` adds a usage chunk before `data: [DONE]`                     |
| `max_completion_tokens`, `max_tokens`  | The output limit; `max_completion_tokens` wins when both are set                                 |
| `temperature`, `top_p`, `stop`         | Passed to Bedrock, except on the models that reject sampling parameters                          |
| `tools`, `tool_choice`                 | Function tools; `tool_choice` can be `auto`, `none`, `required` or one function                   |
| `response_format`                      | `json_schema` and `json_object` on the models with structured output; HTTP 400 on the others     |
| `reasoning_effort`                     | Mapped to each model's reasoning; see [Caching and reasoning](../guide/caching-and-reasoning.md#reasoning) |
| `extra_body`                           | `prompt_caching` controls the cache; every other key goes to Bedrock with the request            |
| `n`, `frequency_penalty`, `presence_penalty`, `user` | Accepted and ignored                                                               |

An image is a `data:` URL or an `http` or `https` URL, which the gateway downloads. The answer is a
`chat.completion`, or `chat.completion.chunk` events ending with `data: [DONE]`. A model's reasoning appears inside
`content` between `<think>` and `</think>`; for GPT-5.x the reasoning summary does.

The keys of `extra_body` other than `prompt_caching`, and the top-level fields `thinking`, `output_config` and
`anthropic_beta`, are sent to Bedrock as additional model request fields, for model options the OpenAI request has no
field for.

## Responses

`POST /api/v1/responses` takes OpenAI's Responses request, without state on the server:

| Field                                  | Handling                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `model`                                | Any model ID or short name                                                                        |
| `input`, `instructions`                | A string or a list of items: messages with text and images, tool calls and their outputs, reasoning |
| `tools`, `tool_choice`                 | `function` and `custom` tools, namespaced tools, and the `local_shell`, `shell` and `apply_patch` tools the client runs |
| `stream`                               | Typed events from `response.created` to `response.completed`, `response.incomplete` or `response.failed`, with no `[DONE]` |
| `max_output_tokens`                    | The output limit; an answer cut by it ends as `incomplete`                                        |
| `temperature`, `top_p`                 | Passed to Bedrock, except on the models that reject them                                         |
| `reasoning.effort`                     | Mapped to each model's reasoning                                                                 |
| `text.format`                          | `json_schema` and `json_object` on the models with structured output                             |
| `include`                              | `reasoning.encrypted_content` returns the reasoning in a form the client sends back on the next turn |
| `store`, `previous_response_id`        | Accepted and ignored: the client sends the whole conversation each time                          |

Tools that OpenAI runs on its own servers, such as web search, file search, code interpreter and image generation,
have no counterpart on Bedrock; the gateway leaves them out of the request and keeps the client's own tools. A
reasoning model's reasoning is a `reasoning` item in `output`, before the message.

## Completions

`POST /api/v1/completions` serves the legacy text completion, for editors whose code completion uses it. `prompt` is
a string, or a list of strings that the gateway joins with newlines; token arrays and `suffix` are refused. `max_tokens`,
`temperature`, `top_p`, `stop` and `stream` work as on Chat Completions, and the answer is a `text_completion`. The
gateway sends the prompt to the model as a user message. GPT-5.x and gpt-oss are not served on this route.

## Embeddings

`POST /api/v1/embeddings` serves the embedding models in the embedding registry,
[`config/embeddings.toml`](../../../config/embeddings.toml):

| Family | Model IDs                                                   | Input                     |
| ------ | ----------------------------------------------------------- | ------------------------- |
| Cohere | `cohere.embed-multilingual-v3`, `cohere.embed-english-v3`  | A string or a list        |
| Titan  | `amazon.titan-embed-text-v1`, `amazon.titan-embed-text-v2:0` | One string per request   |
| Nova   | `amazon.nova-2-multimodal-embeddings-v1:0`                   | One string per request   |

Titan and Nova answer HTTP 400 to a list of more than one input. `encoding_format` is `float` or `base64`. Nova
takes `dimensions` of 256, 384, 1024 or 3072, the default. Token arrays are decoded to text with the `cl100k_base`
encoding first. A model outside the registry gets HTTP 400; add it there with its family to serve it.

## Models

`GET /api/v1/models` lists the foundation models and inference profiles your account can call in the gateway's
region, plus the short names, as OpenAI `model` objects. The gateway reads the list from Bedrock at start and every
`MODEL_CATALOG_REFRESH_SECS`, and serves it from memory. `GET /api/v1/models/{id}` returns one entry, or HTTP 400 when
the ID is not in the list.

## Health

`GET /api/v1/health` answers `OK` with HTTP 200 and needs no key. It reports that the process runs, not that Bedrock
is reachable.
