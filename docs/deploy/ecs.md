# ECS / Fargate Deployment (CloudFormation)

Deploy the gateway on AWS ECS Fargate behind an Application Load Balancer using the
provided CloudFormation template. The template provisions everything needed in a single
stack: ALB, ECS service, IAM task role, and health check configuration.

Template: `deployment/BedrockGatewayFargate.template`

---

## Prerequisites

- An existing VPC with at least two subnets (for ALB high-availability)
- Bedrock model access enabled in your target region
- A gateway API key ready (plaintext string or Secrets Manager ARN)

---

## Deploy

```bash
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayFargate.template \
  --stack-name bedrock-gateway \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ApiKey=sk-my-secret-key \
    VpcId=vpc-... \
    SubnetIds=subnet-...,subnet-...
```

### What the template creates

- **Application Load Balancer** — internet-facing, ports 80 (redirect to 443) and 443
- **ECS Fargate service** — pulls `sunerpy/bedrock-gateway-rust`, runs behind the ALB
- **IAM task role** — with `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`,
  `bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles`
- **Security groups** — ALB ingress on 80/443, task ingress on 8080 from the ALB SG
- **Health check** — ALB targets `GET /api/v1/health`, expects `200 OK`

---

## Template Parameters

| Parameter      | Required | Description                                             |
| -------------- | -------- | ------------------------------------------------------- |
| `ApiKey`       | Yes      | Bearer token your clients send to the gateway           |
| `VpcId`        | Yes      | VPC to deploy into                                      |
| `SubnetIds`    | Yes      | Comma-separated subnet IDs (at least two for HA)        |
| `AwsRegion`    | No       | AWS region for Bedrock calls (defaults to stack region) |
| `DefaultModel` | No       | Default model when the client omits `model`             |
| `ImageTag`     | No       | Docker Hub image tag (defaults to `latest`)             |

---

## Custom image

To use a custom ECR image instead of the public Docker Hub image:

1. Build and push:

   ```bash
   docker build -t <ECR_URI>:latest .
   aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URI>
   docker push <ECR_URI>:latest
   ```

2. Override `ImageUri` in the stack parameters (check the template for the exact
   parameter name).

---

## After deployment

Find the ALB DNS name in the CloudFormation stack outputs, then verify:

```bash
curl https://<ALB_DNS>/api/v1/health
# OK

curl https://<ALB_DNS>/api/v1/chat/completions \
  -H "Authorization: Bearer sk-my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"anthropic.claude-3-5-sonnet-20241022-v2:0","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## High availability

Set `DesiredCount` to 2 or more for multi-AZ redundancy. The gateway is stateless —
all replicas are identical and can be placed in different AZs behind the same ALB
without any additional coordination.

---

## Logging

The task logs to CloudWatch Logs. Set `LOG_LEVEL=debug` in the task definition
environment to see upstream Bedrock call details. The default `info` level emits
per-request business metadata (model, cached_tokens, cache_hit, duration_ms) without
any prompt content.

---

## Notes

- Lambda has a 10-minute maximum timeout. For workloads with long streaming sessions,
  ECS/Fargate is the right choice — no timeout cap applies.
- The task role uses `Resource: "*"` for Bedrock actions because model ARN formats
  vary across regions and model families. Scope it down if your security policy requires.
