# Questions and answers

## Requests

### A request answers 401, but the key is right

The gateway also answers 401 when Bedrock denies its own AWS identity access to the model, and the message is the
same as for a wrong key. Check that the role or user the gateway runs as has the
[permissions](../operate/index.md#what-it-needs-from-aws) for the model and its region, and that the model is enabled
for your account in the Bedrock console. The log records the failure as `unauthorized` in both cases.

### A model is not in the model list

The list holds what Bedrock reports for the gateway's region, so check, in that order:

- that `AWS_REGION` is the region you expect, and that the model is available to your account there;
- that `ENABLE_CROSS_REGION_INFERENCE` is not `false`, since most recent models exist only as `us.`, `eu.` or
  `global.` inference profiles;
- that `ALLOWED_MODELS` does not leave it out;
- that the gateway has read the list since the model launched. It reads it again every
  `MODEL_CATALOG_REFRESH_SECS`, an hour by default, or at a restart.

A request for a model outside the list still goes to Bedrock, but the gateway refuses images in it: it learns which
models take images from the list.

### GPT-5.x answers 400

The message says which of these applies:

- `requires a Bedrock API key`: GPT-5.x and gpt-oss use Bedrock's OpenAI-compatible endpoint, which accepts only a
  Bedrock API key. Set `AWS_BEARER_TOKEN_BEDROCK` and restart.
- `is not available in region`: the model is served in a few regions only. Run the gateway in one of them, see
  [Models](../guide/models.md#openai-gpt).
- `is only available on the /responses endpoint` or `on the /chat/completions endpoint`: the Completions route does
  not serve GPT-5.x and gpt-oss. Send the request to Chat Completions or Responses.

### `response_format` answers 400 on a Claude model

Bedrock supports structured output on Claude Sonnet 4.5 and 4.6, Haiku 4.5 and Opus 4.5 and 4.6 only. On the other
versions the gateway refuses the request instead of sending one that Bedrock would reject. Ask for JSON in the prompt
on those versions, or use a model that has it.

### A Claude answer has no reasoning when tools are in the request

On Chat Completions, a reasoning model needs its signed reasoning back after a tool call, and the Chat Completions
format has no place for it. Without the signing key, the gateway therefore serves Claude requests with tools with
reasoning off. Turn on the [signing key](../guide/configuration.md#reasoning-across-tool-calls), or use the Responses
API, which carries the reasoning by itself.

### HTTP 413

The request body is larger than `MAX_BODY_SIZE_MB`, 20 MB by default. Base64 images count toward it; a long session
with many screenshots can reach it. Raise the limit, or send images as URLs.

### HTTP 429

Bedrock throttled the request, and the gateway's retries, `AWS_MAX_RETRY_ATTEMPTS` of them, did not get through.
Bedrock's quotas are per account, region and model. A cross-region or `global.` inference profile spreads requests
over more capacity, and the Service Quotas console shows the limits and takes increase requests.

## Streaming

### Streams stop after a few seconds or minutes

A proxy or load balancer between the client and the gateway closed the connection while the stream was quiet. Raise
its idle timeout above the longest pause of a model's answer; [Operate](../operate/index.md#streaming-and-timeouts)
lists the settings for the load balancer and ECS Service Connect.

### Chat Completions streams end without usage

Ask for it with `stream_options: {"include_usage": true}`; the gateway then sends a usage chunk before
`data: [DONE]`, as OpenAI does.

## Caching

### The cache never hits

`prompt_tokens_details.cached_tokens` stays 0 when the prompt before a cache point changes between requests, such as a
timestamp in the system prompt, or when it is shorter than the model's minimum. [Caching and
reasoning](../guide/caching-and-reasoning.md#prompt-caching) explains where the gateway places cache points.

## Running it

### Can several gateways run side by side?

Yes. They keep no state, so any number can run behind one load balancer. Give them the same `API_KEY`, and, when the
signing key is on, the same `CHAT_REASONING_CAPSULE_KEYS`.

### Does it store conversations?

No. Once an answer is sent, only its log line remains, with the model and token counts and no content. The Responses
API is served without state, and the client sends the whole conversation with each request. See
[Privacy](../privacy.md).

### Does it serve the Anthropic Messages API?

No. Claude is served through the OpenAI APIs, so a client that speaks only Anthropic Messages, such as Claude Code,
cannot use it.

### Is it an AWS project?

No. It began as a Rust rewrite of [aws-samples/bedrock-access-gateway](https://github.com/aws-samples/bedrock-access-gateway)
and keeps its MIT-0 license, but it is maintained on its own and is not supported by AWS.
