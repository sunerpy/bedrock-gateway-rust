# Observability

The gateway reports what it does in its log, one line per event, and can export metrics and traces over
OpenTelemetry. Neither ever carries a prompt, an answer, a tool argument, an image or a key.

## The log

Each line on standard output is one JSON object with `timestamp`, `level`, `message`, `target` and the fields of the
event. `DEBUG=true` prints the same events as readable text instead, for a terminal.

`LOG_LEVEL` sets how much is written: `error`, `warn`, `info` (the default), `debug` or `trace`. It also takes filter
directives in the `RUST_LOG` form, such as `info,tower_http=debug`, and `RUST_LOG` itself, when set, takes precedence.

### What a request writes

At `info`, each request writes a line when it arrives and one when it ends:

| Event                                                         | Fields                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `chat request received`, `responses request received` and the like | `request_id`, `model`, `stream`; Responses adds `reasoning_effort`                         |
| `chat completed`, `chat streaming completed`                  | `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`, `cache_hit` and the time taken |
| `responses completed`, `responses streaming completed`        | `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`, `cache_hit`, `reasoning_used`, `reasoning_tokens` and the time taken |
| `chat request failed`, `completions request failed` and the like | `request_id`, `model` and the error                                                            |

The time taken is `latency_ms` on a non-streaming answer and `duration_ms` on a stream. A non-streaming chat answer
also logs its `finish_reason`, and a Responses answer its `status`.

GPT-5.x and gpt-oss, which Bedrock's OpenAI-compatible endpoint serves, log differently. A GPT-5.x stream ends with
`responses raw streaming completed`, which carries `input_tokens` and `output_tokens`, also when the request came
through Chat Completions. A gpt-oss request on Chat Completions is passed through as Bedrock sends it, and its lines
carry no token counts.

A failure the client caused, such as an unknown model or a malformed request, is logged at `warn`; a failure on the
gateway's or Bedrock's side, at `error`.

`request_id` ties the lines of one request together. The gateway takes it from the request's `x-request-id` header
when the client sends one, so a client can find its own requests in the log, and makes one up otherwise.

At `debug`, the gateway also logs the model ID and region each request was sent to, and the HTTP layer logs every
request with its status and latency. `debug` applies to the AWS SDK and the HTTP libraries too, which makes the log
long; `info,bedrock_gateway_rust=debug` keeps the extra lines to the gateway's own.

### On AWS

On ECS and Lambda, standard output goes to CloudWatch Logs. Because each line is JSON, CloudWatch Logs Insights reads
the fields directly; for example, the tokens each Converse model used on Chat Completions:

```text
fields @timestamp, model, prompt_tokens, completion_tokens, cached_tokens
| filter message in ["chat completed", "chat streaming completed"]
| stats sum(prompt_tokens), sum(completion_tokens), sum(cached_tokens) by model
```

## OpenTelemetry

The release binaries and images are built without OpenTelemetry. Build the gateway with the `otel` feature to export
over OTLP:

```sh
cargo build --release --features otel
```

Then point it at an OTLP collector that accepts HTTP with protobuf, usually on port 4318:

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

The gateway sends traces to `/v1/traces` and metrics to `/v1/metrics` under that address, as service `bedrock-gateway`
with its version. With that variable set, the standard `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`,
`OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` variables are honoured too. Without it the
build exports nothing.

### Metrics

| Metric                       | Kind      | Records                            |
| ---------------------------- | --------- | ---------------------------------- |
| `gateway.requests`           | counter   | one per request                    |
| `gateway.request.duration`   | histogram | the time taken, in milliseconds    |
| `gateway.tokens.prompt`      | histogram | the input tokens of each request   |
| `gateway.tokens.completion`  | histogram | the output tokens of each request  |

Each carries two labels, `model` and `finish_reason` (the status on Responses). They are recorded for non-streaming
requests on Chat Completions, Completions and Responses; streams are counted in the log only.

### Traces

The gateway's request spans are at the `debug` level, and the log level decides which spans exist, so traces are
exported only when it lets them through. `LOG_LEVEL=info,tower_http=debug` exports a span for each HTTP request, with
its method, URI and duration, and keeps the rest of the log at `info`.
