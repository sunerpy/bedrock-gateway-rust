# Data and network

This page lists every place bedrock-gateway sends data to and what it keeps. It runs in your environment, calls
Bedrock with your AWS account, and reports to no one else.

## What it keeps

Nothing on disk: it writes no files, and the image needs no writable volume. In memory it holds its configuration and
the model list, never a conversation. A request lives only until its answer is sent; afterwards only its log line
remains.

## What it sends, and where

`<region>` is the gateway's `AWS_REGION`.

| Destination                                      | When                                                           | What goes there                                                    |
| ------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| `bedrock-runtime.<region>.amazonaws.com`         | Every model and embedding request                              | The request's messages, tools, images and parameters               |
| `bedrock-mantle.<region>.api.aws`                | Requests for GPT-5.x and gpt-oss                               | The request, as the client sent it                                 |
| `bedrock.<region>.amazonaws.com`                 | At start, then every `MODEL_CATALOG_REFRESH_SECS`              | A listing of the models and inference profiles of your account     |
| `ssm.<region>.amazonaws.com`, `secretsmanager.<region>.amazonaws.com` | At start, when the client key is kept there | A read of that parameter or secret                                 |
| The AWS credential sources of the environment    | When the AWS SDK gets or renews credentials                    | The standard credential exchange, such as with the instance metadata service or STS |
| The host in an image URL                         | When a message carries an `http` or `https` image URL          | A download of the image                                            |
| The OTLP endpoint you set                        | Only in a build with the `otel` feature                        | Metrics and spans, without content                                 |

A request for an inference profile can be served by Bedrock in another region that the profile covers; a `global.`
profile covers regions worldwide. Bedrock's own data handling then applies; see Bedrock's
[data protection](https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html) page.

There is no telemetry and no update check.

## Image URLs

The gateway downloads an image URL from the network it runs in, from any host, and sends the bytes to Bedrock. A
client that holds the key can therefore make it fetch URLs that only that network reaches. When that matters, run
the gateway where outbound requests are limited, or have clients send images as `data:` URLs, which the gateway
decodes without a request.

## Who can reach it

The gateway listens on `0.0.0.0:8080` over plain HTTP by default, so anyone who reaches the port and holds the key can
use your Bedrock account through it. `BIND_ADDR=127.0.0.1` keeps it to the local machine. For clients on a network,
put a load balancer or a reverse proxy with TLS in front of it, and keep the key as private as an AWS credential.

## The log

The log goes to standard output, one JSON object per line: the model, token counts, cache use, timing, the request
ID and, for a failure, the error. It never holds prompts, answers, tool definitions or arguments, images or keys.
[Observability](operate/observability.md) lists the fields.
