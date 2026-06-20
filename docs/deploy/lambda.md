# Deploying bedrock-gateway on AWS Lambda (Lambda Web Adapter)

## Overview

The `bedrock-gateway` binary is the same artifact across all four deployment targets:
standalone, Docker, ECS/Fargate, and Lambda. There is no `lambda_http` dependency, no
separate binary, and no `main` function fork for Lambda. AWS Lambda Web Adapter (LWA)
handles the translation layer — the unmodified axum HTTP server runs inside Lambda;
LWA bridges the Lambda event protocol to plain HTTP.

---

## How It Works

LWA ships as a Lambda extension binary at `/opt/extensions/lambda-adapter`. The cold-start
sequence:

1. Lambda boots the container and starts all extensions, including `lambda-adapter`.
2. LWA starts `bedrock-gateway` (via the container `CMD`) as a child process.
3. LWA polls `AWS_LWA_READINESS_CHECK_PATH` (`/api/v1/health`) on `127.0.0.1:$AWS_LWA_PORT`
   until it gets `HTTP 200`.
4. Once healthy, LWA signals Lambda that the function is ready.
5. On each invocation, LWA translates the Function URL or API Gateway event into a plain
   HTTP request, forwards it to the axum server, and streams the response back.

The axum server never knows it is inside Lambda — it sees ordinary HTTP traffic.

---

## Required Environment Variables

### LWA configuration

These are baked into `deployment/lambda/Dockerfile` — you don't need to set them manually
unless overriding.

| Variable                       | Value             | Purpose                                    |
| ------------------------------ | ----------------- | ------------------------------------------ |
| `AWS_LWA_PORT`                 | `8080`            | Port LWA forwards to (must match `PORT`)   |
| `PORT`                         | `8080`            | Port the axum server binds on              |
| `AWS_LWA_INVOKE_MODE`          | `response_stream` | Enables SSE streaming through Function URL |
| `AWS_LWA_READINESS_CHECK_PATH` | `/api/v1/health`  | Path LWA polls before accepting traffic    |
| `AWS_LWA_READINESS_CHECK_PORT` | `8080`            | Port for the readiness probe               |

### Gateway configuration

| Variable             | Purpose                                                              |
| -------------------- | -------------------------------------------------------------------- |
| `API_KEY`            | Static API key (plaintext; use only for testing)                     |
| `API_KEY_SECRET_ARN` | ARN of a Secrets Manager secret with an `api_key` field (production) |
| `API_KEY_PARAM_NAME` | SSM Parameter Store parameter name holding the key (alternative)     |
| `DEFAULT_MODEL`      | Default model ID when the client omits `model`                       |

> Do not set `AWS_REGION` in the Lambda environment — it is a Lambda-reserved variable.
> The runtime injects it automatically. Setting it manually will cause `cfn-lint E3663`.

---

## Build and Deploy

### 1. Build the container image

```bash
docker build \
  -f deployment/lambda/Dockerfile \
  -t bedrock-gateway:lambda \
  .
```

The Dockerfile uses a two-stage musl build (static binary, no glibc) and copies LWA
from `public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1`.

### 2. Push to ECR

```bash
aws ecr create-repository --repository-name bedrock-gateway-lambda

ECR_URI=$(aws ecr describe-repositories \
  --repository-names bedrock-gateway-lambda \
  --query 'repositories[0].repositoryUri' --output text)

aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_URI"
docker tag bedrock-gateway:lambda "$ECR_URI:latest"
docker push "$ECR_URI:latest"
```

### 3. Deploy with CloudFormation (recommended)

```bash
aws cloudformation deploy \
  --template-file deployment/BedrockGatewayLambda.template \
  --stack-name bedrock-gateway-lambda \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ImageUri="$ECR_URI:latest" \
    ApiKeySecretArn=arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:BedrockProxyAPIKey
```

The template provisions a Lambda function, the necessary IAM execution role, and a
Function URL with response streaming enabled.

### 4. Create the function manually (alternative)

```bash
aws lambda create-function \
  --function-name bedrock-gateway \
  --package-type Image \
  --code ImageUri="$ECR_URI:latest" \
  --role arn:aws:iam::ACCOUNT_ID:role/bedrock-gateway-lambda-role \
  --timeout 600 \
  --memory-size 1024 \
  --environment "Variables={
    AWS_LWA_INVOKE_MODE=response_stream,
    API_KEY_SECRET_ARN=arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:BedrockProxyAPIKey
  }"
```

### 5. Grant IAM permissions

The Lambda execution role needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream",
    "bedrock:ListFoundationModels",
    "bedrock:ListInferenceProfiles"
  ],
  "Resource": "*"
}
```

If using Secrets Manager for the API key:

```json
{
  "Effect": "Allow",
  "Action": ["secretsmanager:GetSecretValue"],
  "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:BedrockProxyAPIKey*"
}
```

If using SSM Parameter Store:

```json
{
  "Effect": "Allow",
  "Action": ["ssm:GetParameter"],
  "Resource": "arn:aws:ssm:REGION:ACCOUNT_ID:parameter/YOUR_PARAM_NAME"
}
```

### 6. Create a Function URL

```bash
aws lambda create-function-url-config \
  --function-name bedrock-gateway \
  --auth-type AWS_IAM \
  --invoke-mode RESPONSE_STREAM
```

Function URLs support streaming natively and are the simplest way to expose the gateway
without managing an API Gateway.

---

## Lambda Timeout Limit

Lambda has a **maximum timeout of 10 minutes** (600 seconds). Long-running streaming
completions that exceed this will be cut off. For workloads that require longer sessions,
use ECS/Fargate instead — see [ecs.md](ecs.md).

SSE streaming works correctly within this limit because LWA bridges the HTTP chunked
response to Lambda's response streaming protocol.

---

## Memory Recommendation

512 MB minimum, **1024 MB recommended** for cold-start latency. The binary itself is
small (~12 MB), but Lambda allocates CPU proportionally to memory — more memory means
faster cold starts and better sustained throughput.
