# Choose a client

Every client needs the same two things: the gateway's base URL and the key in its `API_KEY`. Nothing is installed
into the client.

| Client             | API                             | Base URL                       | Guide                                       |
| ------------------ | ------------------------------- | ------------------------------ | ------------------------------------------- |
| Codex CLI          | Responses                       | `http://localhost:8080/api/v1` | [Codex CLI](codex.md)                       |
| OpenCode           | Chat Completions                | `http://localhost:8080/api/v1` | [Editors and agents](editors-and-agents.md) |
| Zed                | Chat Completions, Completions   | `http://localhost:8080/api/v1` | [Editors and agents](editors-and-agents.md) |
| OpenAI SDKs        | Chat Completions, Responses     | `http://localhost:8080/api/v1` | [below](#sdks)                              |
| curl and scripts   | Any route                       | `http://localhost:8080/api/v1` | [API reference](../reference/api.md)        |

Each client adds the route, such as `/chat/completions`, to the base URL, so the base URL ends in `/api/v1`, the
default `API_ROUTE_PREFIX`. Replace `localhost:8080` with the address at which your clients reach the gateway.

## The key

Clients send the key as `Authorization: Bearer <key>`, which is what the OpenAI SDKs and most other clients send
when they are given an API key. A request without the key, or with another one, gets HTTP 401 with the error code
`unauthorized`. The gateway has one key, and every client that holds it has the same access, so treat it like a
password and change it when someone should lose access.

## Model names

`GET /api/v1/models` lists the model IDs your account can call in the gateway's region, and the gateway also accepts
the short names in [Models](../guide/models.md), such as `gpt-5.5`. A client sends the ID as it is listed; the gateway
passes it to Bedrock unchanged.

Some clients know the models of their own vendor only and warn about, or refuse, a name they do not know. Their pages
below show where to declare the model.

## SDKs

The OpenAI SDK for Python, through Responses:

```python
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/api/v1", api_key=os.environ["API_KEY"])
response = client.responses.create(
    model="us.amazon.nova-2-lite-v1:0",
    input="Reply with exactly: BEDROCK_OK",
)
print(response.output_text)
```

The OpenAI SDK for JavaScript, through Chat Completions:

```ts
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:8080/api/v1",
  apiKey: process.env.API_KEY,
});

const completion = await openai.chat.completions.create({
  model: "us.amazon.nova-2-lite-v1:0",
  messages: [{ role: "user", content: "Reply with exactly: BEDROCK_OK" }],
});
console.log(completion.choices[0].message.content);
```

Both print `BEDROCK_OK`. Options that exist only on Bedrock, such as the
[prompt cache controls](../guide/caching-and-reasoning.md#control-it-per-request), go in the SDK's `extra_body`
option.

## Other clients

Any client that speaks OpenAI Chat Completions or Responses can use the same base URL. Prefer Responses when the client
offers both and the conversation uses tools with a reasoning model: Responses carries the model's reasoning from one
turn to the next on its own, while Chat Completions needs the
[signing key](../guide/configuration.md#reasoning-across-tool-calls) for it. The
[protocol compatibility](../../openai-protocol-compatibility.md) reference lists which request fields and tool types
each route handles.
