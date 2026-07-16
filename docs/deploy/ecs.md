# ECS / Fargate Deployment (CloudFormation)

Deploy the gateway as an ECS Fargate service behind an internet-facing Application Load
Balancer using the provided CloudFormation template. The gateway's only backend is Amazon
Bedrock, which is available only in AWS commercial regions — deploy this template in any
AWS commercial region where Amazon Bedrock is available.

Template: `deployment/BedrockGatewayFargate.template`

---

## Two deploy modes

The template supports two networking modes, controlled by the `CreateNetwork` parameter:

- **One-click** (`CreateNetwork=true`, the default) — the stack creates a new VPC, two
  public subnets (one per AZ), an internet gateway, and public routing. Nothing needs to
  exist beforehand; you only need a Secrets Manager secret for the API key.
- **Bring-your-own** (`CreateNetwork=false`) — the stack reuses an existing VPC. You must
  also supply `VpcId` and at least two `Subnets` (in different AZs) for the ALB and the
  Fargate tasks.

---

## Prerequisites

- Bedrock model access enabled in the target commercial region.
- A Secrets Manager secret containing an `api_key` field. This is the bearer token clients
  present to the gateway:

  ```bash
  aws secretsmanager create-secret \
    --name bedrock-gateway/api-key \
    --secret-string '{"api_key":"sk-my-secret-key"}'
  ```

---

## Deploy — one-click

```bash
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayFargate.template \
  --stack-name bedrock-gateway \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:bedrock-gateway/api-key
```

`DisableMantle` defaults to `"true"`, so the default published image
(`docker.io/sunerpy/bedrock-gateway-rust:latest`) boots on SigV4 alone, with no Bedrock API
key required. `CreateNetwork` defaults to `"true"`, so this single command provisions a
new VPC, subnets, ALB, and Fargate service end to end.

## Deploy — bring-your-own network

```bash
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayFargate.template \
  --stack-name bedrock-gateway \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:bedrock-gateway/api-key \
    CreateNetwork=false \
    VpcId=vpc-0123456789abcdef0 \
    Subnets=subnet-0aaa,subnet-0bbb
```

Use public subnets, or private subnets with a NAT gateway for ECR / Bedrock / Secrets
Manager egress.

## Enabling GPT-5.x (optional)

GPT-5.x (`gpt-5.4` / `gpt-5.5`) is served through the AWS Bedrock mantle upstream and
needs a real Bedrock API key. Create a second secret with a `bedrock_api_key` field, then
set `DisableMantle=false` and pass its ARN:

```bash
aws secretsmanager create-secret \
  --name bedrock-gateway/bedrock-api-key \
  --secret-string '{"bedrock_api_key":"<your-bedrock-api-key>"}'

aws cloudformation deploy \
  --template-file deployment/BedrockGatewayFargate.template \
  --stack-name bedrock-gateway \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:bedrock-gateway/api-key \
    DisableMantle=false \
    BedrockApiKeySecretArn=arn:aws:secretsmanager:us-east-1:123456789012:secret:bedrock-gateway/bedrock-api-key
```

Leaving `DisableMantle=true` (or omitting `BedrockApiKeySecretArn`) is fine if you don't
need GPT-5.x — every other model works over the standard SigV4 task role.

---

## Parameters

| Parameter                | Default                                          | Description                                                                                                             |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `ContainerImageUri`      | `docker.io/sunerpy/bedrock-gateway-rust:latest`  | Container image for the gateway. Push your own build to ECR to override.                                               |
| `ApiKeySecretArn`        | *(required)*                                      | Secrets Manager ARN holding the client→gateway bearer token (`api_key` field).                                         |
| `DisableMantle`          | `"true"`                                          | `"true"` skips GPT-5.x mantle startup validation so the default image boots without a Bedrock API key. Set `"false"` only alongside `BedrockApiKeySecretArn`. |
| `BedrockApiKeySecretArn` | `""`                                              | (Optional) Secrets Manager ARN whose `bedrock_api_key` field enables the GPT-5.x mantle backend.                        |
| `DefaultModelId`         | `""`                                              | (Optional) Default Bedrock model ID when a client omits `model`.                                                        |
| `DefaultEmbeddingModelId`| `""`                                              | (Optional) Default embedding model ID.                                                                                  |
| `EnablePromptCaching`    | `"true"`                                          | Auto-inject prompt cache points for Claude and Nova models.                                                             |
| `LogLevel`               | `info`                                            | `trace` / `debug` / `info` / `warn` / `error`.                                                                          |
| `CreateNetwork`          | `"true"`                                          | `"true"` = one-click (new VPC + two public subnets + IGW + routing). `"false"` = bring-your-own (`VpcId` + `Subnets` become required). |
| `VpcCidr`                | `10.240.0.0/16`                                   | CIDR for the VPC created in one-click mode. Ignored when `CreateNetwork=false`. Two `/24` public subnets are carved from it. |
| `VpcId`                  | `""`                                              | Existing VPC ID. Required only when `CreateNetwork=false`.                                                              |
| `Subnets`                | `""`                                              | Comma-separated list of at least two existing subnets in different AZs. Required only when `CreateNetwork=false`.       |
| `DesiredCount`           | `2`                                                | Number of running Fargate tasks (minimum 1).                                                                            |

### Outputs

| Output        | Description                                                          |
| ------------- | ---------------------------------------------------------------------- |
| `APIBaseUrl`  | `http://<ALB-DNS>/api/v1` — use as `OPENAI_API_BASE`.                |
| `ClusterName` | ECS cluster name.                                                    |
| `ServiceName` | ECS service name.                                                    |
| `VpcId`       | The VPC in use (created in one-click mode, or the provided `VpcId`). |
| `Subnets`     | The subnets used by the ALB and Fargate tasks.                       |
| `NetworkMode` | `one-click` or `bring-your-own`.                                     |

---

## After deployment / verify

Grab the `APIBaseUrl` stack output and hit the health check first:

```bash
APIBASEURL=$(aws cloudformation describe-stacks \
  --stack-name bedrock-gateway \
  --query "Stacks[0].Outputs[?OutputKey=='APIBaseUrl'].OutputValue" \
  --output text)

curl "$APIBASEURL/health"
# OK
```

Then send an authed chat request using the `api_key` value you put in `ApiKeySecretArn`:

```bash
curl "$APIBASEURL/chat/completions" \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Hello!"}]}'
```

Some models are only callable through a cross-region inference profile in a given
region — if you get a Bedrock access error for an on-demand model ID, try the same model
with the region's inference-profile prefix (e.g. `us.anthropic.claude-...` or
`apac.anthropic.claude-...`) instead of the bare model ID.

### Service Connect and streaming timeouts

The supplied CloudFormation template does not enable ECS Service Connect. If you add it
to an existing service, configure its HTTP timeouts explicitly. AWS otherwise applies a
**15-second per-request timeout**, which can cut a healthy SSE response in the middle of
a tool call. Clients may then retry the incomplete turn and appear to repeat assistant
text or tool calls indefinitely.

For this streaming gateway, disable the total request deadline and retain an idle
deadline at least as large as the gateway's 180-second upstream idle timeout:

```json
"timeout": {
  "idleTimeoutSeconds": 300,
  "perRequestTimeoutSeconds": 0
}
```

A complete reusable service configuration is checked in at
`deployment/service-connect-streaming.json`. Replace its namespace and aliases for your
environment, then update the service:

```bash
aws ecs update-service \
  --cluster bedrock-gateway \
  --service bedrock-gateway \
  --service-connect-configuration file://deployment/service-connect-streaming.json
```

Validate an existing ECS service before or after deployment:

```bash
aws ecs describe-services \
  --cluster bedrock-gateway \
  --services bedrock-gateway \
  --output json > /tmp/ecs-service.json

scripts/check-ecs-service-connect-timeouts.sh /tmp/ecs-service.json
```

The validator deliberately fails when `perRequestTimeoutSeconds` is absent: omission is
not unlimited; it selects the unsafe AWS 15-second default.

---

## High availability

Set `DesiredCount` to 2 or more for multi-AZ redundancy. The gateway is stateless — all
replicas are identical and can be placed across AZs behind the same ALB with no additional
coordination.

---

## Logging

The task logs to CloudWatch Logs. Set `LogLevel=debug` to see upstream Bedrock call details
(resolved model, target region). At every level the gateway logs metadata only (model,
`cached_tokens`, `cache_hit`, `duration_ms`) — never prompt content, completion text, or the
API key itself.

---

## Notes

- Lambda has a 10-minute maximum timeout. For workloads with long streaming sessions,
  ECS/Fargate is the right choice, provided every HTTP hop is configured for streaming.
  The supplied ALB uses a 600-second idle timeout; if Service Connect is enabled, follow
  the timeout contract above.
