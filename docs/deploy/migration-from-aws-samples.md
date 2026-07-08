# Migrating from aws-samples/bedrock-access-gateway

This guide is for teams already running the Python gateway
[`aws-samples/bedrock-access-gateway`](https://github.com/aws-samples/bedrock-access-gateway)
who want to move to this Rust reimplementation.

The short version: both projects expose the same OpenAI-compatible client contract. Your
existing clients keep talking to the same `base_url` shape and the same bearer `api_key`.
What changes is the deployment — a new container image and a new CloudFormation stack.
No client-side code changes are required.

---

## TL;DR

Your clients already point `OPENAI_BASE_URL` at something like
`http://<alb-or-api-gateway>/api/v1` and send `OPENAI_API_KEY` as a bearer token. That
does not change. You:

1. Deploy the new stack (`deployment/BedrockGatewayFargate.template` or
   `deployment/BedrockGatewayLambda.template`) — no image build required, the Fargate
   template defaults to a published Docker Hub image.
2. Point clients at the new stack's `APIBaseUrl` output (or repoint DNS at it).
3. Decommission the old stack.

Same key, same URL shape, same request/response bodies for chat completions and
embeddings. The only moving part is which stack answers the request.

---

## What's the same

If you're used to the upstream gateway, this should feel familiar:

- **OpenAI-compatible surface** — `POST /chat/completions`, `POST /embeddings`,
  `GET /models`, `GET /models/{id}`, `GET /health` all exist with the same verbs and paths
  under the route prefix.
- **Same auth model** — the bearer token clients present to the gateway comes from a
  Secrets Manager secret with an `api_key` field (or SSM Parameter Store, or a plain env
  var). You can point the new stack's `ApiKeySecretArn` parameter at the exact same secret
  you already created for the upstream gateway — no new secret needed.
- **Same default-model concept** — `DEFAULT_MODEL` / `DEFAULT_EMBEDDING_MODEL` still mean
  "the model to use when a client request omits `model`."
- **Same default route prefix** — `/api/v1` is the default on both projects' Fargate
  templates, so `curl $APIBaseUrl/health` and `curl $APIBaseUrl/chat/completions` behave
  the same way out of the box (watch the Lambda caveat below, though).

---

## What's better / new in the Rust gateway

- **Responses API** — `POST /responses`, streaming and non-streaming. This is the surface
  `codex` requires (`wire_api = "responses"`); the upstream Python gateway does not have
  this endpoint at all.
- **Legacy text completions** — `POST /completions` (`text_completion` wire shape), useful
  for editors like Zed's edit-prediction feature. Also absent upstream.
- **A uniform OpenAI error envelope.** Every error response is
  `{"error":{"message":...,"type":...,"code":...}}`, at every status code. The upstream
  gateway is inconsistent here — some failures return plain text, others return
  `{"detail":...}`. If any of your client code parses error bodies, this is the one
  behavior change worth testing before you cut over (see the table below).
- **A published Docker Hub image.** `docker.io/sunerpy/bedrock-gateway-rust:latest` is the
  Fargate template's default `ContainerImageUri` — you don't have to build or push
  anything to get started. Upstream ships no prebuilt image; you build and push your own
  to ECR via their `scripts/push-to-ecr.sh`.
- **A single static Rust binary.** No Python runtime, no GC pauses, a much smaller image.
  Same binary runs standalone, in Docker, on ECS/Fargate, and on Lambda.
- **One-click CloudFormation for Fargate.** The Fargate template can create its own VPC,
  subnets, and internet gateway (`CreateNetwork=true`, the default) so you don't need an
  existing network to try it — or set `CreateNetwork=false` and supply your own `VpcId` /
  `Subnets` the way the upstream template always required.

---

## Key differences to watch (migration caveats)

| Area | Upstream (`aws-samples/bedrock-access-gateway`) | Rust gateway | What to do |
| --- | --- | --- | --- |
| Route prefix | Fargate template uses `/api/v1`. **Lambda template overrides it to `/v1`.** | Default is `/api/v1` everywhere (Fargate and Lambda). | If you're migrating off the upstream **Lambda** deployment, either set `API_ROUTE_PREFIX=/v1` on the Rust gateway to keep old client URLs working, or update clients to the new `/api/v1` prefix. Migrating off upstream **Fargate** needs no prefix change. |
| Container image | No prebuilt image. You build `bedrock-proxy-api` (Lambda) or `bedrock-proxy-api-ecs` (Fargate) yourself and push to your own ECR via `scripts/push-to-ecr.sh`. | Fargate defaults to the published `docker.io/sunerpy/bedrock-gateway-rust:latest`. Lambda still requires you to build and push to your own ECR (Lambda needs a container image in ECR either way). | No build step needed for the Fargate path. The Lambda path still needs a `docker build` + ECR push, same as before. |
| Error response shape | Inconsistent: `422` validation errors return plain text; auth failures and most Bedrock errors return `{"detail": "..."}`; not a uniform OpenAI shape. | Always `{"error":{"message":...,"type":...,"code":...}}`. | If any client code branches on `response.json()["detail"]`, switch it to `response.json()["error"]["message"]`. |
| Default model | CFN parameter `DefaultModelId` defaults to `anthropic.claude-3-sonnet-20240229-v1:0` on both upstream templates. | `DefaultModelId` on the Rust templates defaults to an empty string (the binary's own built-in default is `anthropic.claude-3-5-sonnet-20241022-v2:0`). | Pass `DefaultModelId=anthropic.claude-3-sonnet-20240229-v1:0` explicitly if you need to preserve the exact prior default rather than picking up the newer one. |
| GPT-5.x | Not supported — no such models or routing exist upstream. | Supported on `/responses` only, via the AWS Bedrock mantle backend, and it's opt-in. The default one-click Fargate deploy sets `DisableMantle=true` so the published image boots on SigV4 alone with no extra key. | Nothing to do unless you want GPT-5.x — then set `DisableMantle=false` and supply `BedrockApiKeySecretArn` (Fargate) / `BedrockApiKey` (Lambda). See [ecs.md](ecs.md#enabling-gpt-5x-optional). |

---

## CloudFormation parameter mapping

Both upstream templates (`BedrockProxy.template` for Lambda, `BedrockProxyFargate.template`
for Fargate) share exactly four parameters. Here's how they map onto the Rust templates
(`deployment/BedrockGatewayFargate.template` and `deployment/BedrockGatewayLambda.template`):

| Upstream parameter | Rust equivalent | Notes |
| --- | --- | --- |
| `ApiKeySecretArn` | `ApiKeySecretArn` | 1:1. Same secret shape (`api_key` field) — reuse your existing secret as-is. |
| `ContainerImageUri` | `ContainerImageUri` | Same name. Fargate now defaults to the published image, so this is optional there; Lambda still requires your own ECR image. |
| `DefaultModelId` | `DefaultModelId` | Same name, different default — see the caveats table above. |
| `EnablePromptCaching` | `EnablePromptCaching` | Same name, default flipped: upstream defaults to `false`, the Rust templates default to `true`. |
| *(none)* | `DefaultEmbeddingModelId` | New. Default embedding model, mirroring `DefaultModelId` for the embeddings endpoint. |
| *(none)* | `DisableMantle` | New. Gates the GPT-5.x mantle backend; defaults to `"true"` so a default deploy needs no extra Bedrock key. |
| *(none)* | `BedrockApiKeySecretArn` (Fargate) / `BedrockApiKey` (Lambda) | New. Supplies the Bedrock API key used only when GPT-5.x is enabled. Fargate takes a Secrets Manager ARN; Lambda takes a plaintext `NoEcho` parameter (Lambda has no ECS-style secret injection). |
| *(none)* | `LogLevel` | New. `trace` / `debug` / `info` / `warn` / `error`. |
| *(none)* | `CreateNetwork` / `VpcId` / `Subnets` / `VpcCidr` (Fargate only) | New. Lets Fargate create its own VPC/subnets (`CreateNetwork=true`, the default) instead of always requiring an existing network the way upstream did. |
| *(none)* | `DesiredCount` (Fargate only) | New. Number of running Fargate tasks, defaults to `2`. |

---

## Step-by-step migration

1. **Keep your existing API-key secret.** The upstream gateway and this one both read the
   bearer token from a Secrets Manager secret's `api_key` field. Point the new stack's
   `ApiKeySecretArn` parameter at the same secret ARN you already have — nothing to
   recreate.

2. **Deploy the new stack.** For Fargate, the one-click example in
   [ecs.md](ecs.md#deploy--one-click) needs nothing but `ApiKeySecretArn` — no image build:

   ```bash
   aws cloudformation deploy \
     --template-file deployment/BedrockGatewayFargate.template \
     --stack-name bedrock-gateway \
     --capabilities CAPABILITY_IAM \
     --parameter-overrides \
       ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:bedrock-gateway/api-key
   ```

   For Lambda, follow [lambda.md](lambda.md#build-and-deploy) — build the image, push it to
   ECR, then deploy `BedrockGatewayLambda.template` with `ContainerImageUri` pointing at
   your new image.

3. **Point clients at the new stack's `APIBaseUrl` output** (or update DNS to resolve to
   the new ALB / Function URL). The bearer token doesn't change:

   ```bash
   APIBASEURL=$(aws cloudformation describe-stacks \
     --stack-name bedrock-gateway \
     --query "Stacks[0].Outputs[?OutputKey=='APIBaseUrl'].OutputValue" \
     --output text)
   ```

4. **Verify** before cutting traffic over:

   ```bash
   curl "$APIBASEURL/health"
   # OK

   curl "$APIBASEURL/chat/completions" \
     -H "Authorization: Bearer sk-my-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Hello!"}]}'
   ```

5. **Decommission the old stack** once traffic has moved and you've confirmed the new one
   is healthy under real load.

---

## Client configuration is unchanged

Whatever OpenAI-compatible client you're using, the only thing that changes is the URL.
The API key stays the same.

**OpenAI SDK (Python):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<new-stack-APIBaseUrl>",  # was the old stack's URL
    api_key="sk-my-secret-key",                  # unchanged
)
```

**Environment variables (OpenAI SDK, LangChain, LiteLLM, and most OpenAI-compatible
tooling all read these):**

```bash
export OPENAI_BASE_URL="https://<new-stack-APIBaseUrl>"
export OPENAI_API_KEY="sk-my-secret-key"
```

If your client hardcodes the upstream Lambda's `/v1` prefix, either set
`API_ROUTE_PREFIX=/v1` on the new gateway or update the hardcoded path to `/api/v1` — see
the route-prefix row in the caveats table above.
