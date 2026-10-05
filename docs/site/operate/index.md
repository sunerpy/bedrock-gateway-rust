# Operate bedrock-gateway

bedrock-gateway keeps no state between requests, so every instance is the same: run one, or several behind a load
balancer, and replace them freely. This page covers where to run it, what it needs from AWS, and how to keep it
healthy.

## Where to run it

| Where                      | How                                                                                 | Guide                                                 |
| -------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------- |
| A server or a workstation  | The binary, or the image with Docker or Docker Compose                              | [Docker](../../deploy/docker.md)                      |
| Amazon ECS on Fargate      | A CloudFormation template with an Application Load Balancer, one-click or in your network | [ECS](../../deploy/ecs.md)                     |
| AWS Lambda                 | The image with the Lambda Web Adapter, behind a function URL                         | [Lambda](../../deploy/lambda.md)                      |
| Kubernetes                 | The Helm chart in `helm/bedrock-gateway`                                             | [values.yaml](../../../helm/bedrock-gateway/values.yaml) |

Coming from the Python gateway in aws-samples? [Migrating from aws-samples](../../deploy/migration-from-aws-samples.md)
maps its settings and template parameters to these.

## What it needs from AWS

The AWS identity the gateway runs as, whether access keys, a profile or the role of its EC2 instance, ECS task or
Lambda function, needs these permissions:

| Action                                                          | For                                                   |
| --------------------------------------------------------------- | ----------------------------------------------------- |
| `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream`  | Every model request, streaming or not, and embeddings |
| `bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles` | The model list                                        |
| `ssm:GetParameter`                                              | The client key, when `API_KEY_PARAM_NAME` is set      |
| `secretsmanager:GetSecretValue`                                 | The client key, when `API_KEY_SECRET_ARN` is set      |

A parameter or secret encrypted with your own KMS key also needs `kms:Decrypt` on that key. The model must be
available to your account in the gateway's region, and a cross-region inference profile needs the permissions in each
region it routes to. With a Bedrock API key, the key's own permissions apply to the model requests instead.

## Health

`GET /api/v1/health` answers `OK` with HTTP 200, without a key, while the process runs. Point the load balancer's or
the orchestrator's health check at it.

The image has no shell, so its built-in health check runs the binary itself: `bedrock-gateway --health-check` calls
`/health` on the local port and exits with 0 or 1.

## Streaming and timeouts

Long answers stream for minutes, so every hop between a client and the gateway must let an idle stream live that
long:

- An Application Load Balancer needs an idle timeout longer than the longest pause in a stream. The ECS template
  sets 600 seconds.
- ECS Service Connect cuts each request after 15 seconds unless you configure its timeouts; the
  [ECS guide](../../deploy/ecs.md#service-connect-and-streaming-timeouts) shows the setting.
- The gateway waits up to `AWS_READ_TIMEOUT_SECS` (900 seconds) for data from Bedrock, and ends a Responses stream
  that stays silent for `RESPONSES_STREAM_IDLE_TIMEOUT_SECS` (180 seconds).

## Updates and shutdown

On `SIGTERM` or Ctrl-C the gateway stops accepting connections and lets the requests in flight finish. Give it time
to do so: the ECS template stops a task 120 seconds after the signal. A rolling update with at least two instances
then replaces them without failed requests, apart from streams longer than that window.

Updating is replacing the image tag or the binary; there is nothing to migrate. Pin a version tag in production and
read the [release notes](https://github.com/sunerpy/bedrock-gateway-rust/releases) before you move to a newer one.

## Logs

The gateway writes one JSON object per line to standard output, with no prompts, answers or keys in it.
[Observability](observability.md) lists what each line holds and how to export metrics.
