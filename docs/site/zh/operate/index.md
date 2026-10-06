# 运维 bedrock-gateway

bedrock-gateway 在请求之间不保存状态，因此每个实例都完全相同：可以只运行一个，也可以在负载均衡器后面运行多个，并随时替换。本页介绍在哪里运行它、它需要 AWS 提供哪些权限，以及如何让它保持健康运行。

## 在哪里运行

| 运行位置                   | 方式                                                                                          | 指南                                                        |
| -------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 服务器或工作站             | 二进制文件，或用 Docker、Docker Compose 运行镜像                                              | [Docker](../../../deploy/docker.md)                         |
| 基于 Fargate 的 Amazon ECS | 带 Application Load Balancer 的 CloudFormation 模板，可以一键部署，也可以部署到你自己的网络中 | [ECS](../../../deploy/ecs.md)                               |
| AWS Lambda                 | 配合 Lambda Web Adapter 使用的镜像，通过函数 URL 访问                                         | [Lambda](../../../deploy/lambda.md)                         |
| Kubernetes                 | `helm/bedrock-gateway` 中的 Helm chart                                                        | [values.yaml](../../../../helm/bedrock-gateway/values.yaml) |

如果你之前使用的是 aws-samples 中的 Python 网关，[从 aws-samples 迁移](../../../deploy/migration-from-aws-samples.md)列出了它的设置和模板参数与这里的对应关系。

## 所需的 AWS 权限

网关运行时使用的 AWS 身份，无论是访问密钥、profile，还是 EC2 实例、ECS 任务或 Lambda 函数的角色，都需要以下权限：

| 操作                                                            | 用途                                           |
| --------------------------------------------------------------- | ---------------------------------------------- |
| `bedrock:InvokeModel`、`bedrock:InvokeModelWithResponseStream`  | 所有模型请求（无论是否流式）以及 embeddings    |
| `bedrock:ListFoundationModels`、`bedrock:ListInferenceProfiles` | 模型列表                                       |
| `ssm:GetParameter`                                              | 设置了 `API_KEY_PARAM_NAME` 时，读取客户端密钥 |
| `secretsmanager:GetSecretValue`                                 | 设置了 `API_KEY_SECRET_ARN` 时，读取客户端密钥 |

用你自己的 KMS 密钥加密的参数或 secret，还需要对该 KMS 密钥的 `kms:Decrypt` 权限。模型必须在网关所在区域对你的账号可用；跨区域推理配置文件还需要它所路由到的每个区域的权限。使用 Bedrock API 密钥时，模型请求改为适用该密钥自身的权限。

## 健康检查

只要进程在运行，`GET /api/v1/health` 就会以 HTTP 200 返回 `OK`，不需要密钥。请把负载均衡器或编排系统的健康检查指向它。

镜像中没有 shell，因此它内置的健康检查直接运行二进制文件：`bedrock-gateway --health-check` 请求本地端口上的 `/health`，并以 0 或 1 退出。

## 流式响应与超时

较长的回答会持续流式输出几分钟，因此客户端和网关之间的每一跳都必须允许空闲的流保持这么久：

- Application Load Balancer 的空闲超时需要长于流中最长的停顿。ECS 模板将它设为 600 秒。
- 除非你配置了它的超时，否则 ECS Service Connect 会在 15 秒后切断每个请求；[ECS 指南](../../../deploy/ecs.md#service-connect-and-streaming-timeouts)给出了相应的设置。
- 网关等待 Bedrock 返回数据的时间最长为 `AWS_READ_TIMEOUT_SECS`（900 秒）；Responses 流静默的时间达到 `RESPONSES_STREAM_IDLE_TIMEOUT_SECS`（180 秒）时，网关会结束这个流。

## 更新与停止

收到 `SIGTERM` 或 Ctrl-C 时，网关停止接受新连接，并让正在处理的请求完成。请给它留出足够的时间：ECS 模板在发出信号 120 秒后才停止任务。这样，至少有两个实例的滚动更新可以在替换实例时不让任何请求失败，只有持续时间超过这个窗口的流例外。

更新就是替换镜像标签或二进制文件，不需要迁移任何东西。生产环境请固定版本标签，升级到新版本之前先阅读[发布说明](https://github.com/sunerpy/bedrock-gateway-rust/releases)。

## 日志

网关向标准输出写日志，每行一个 JSON 对象，其中不含提示词、回答或密钥。每行包含哪些内容，以及如何导出指标，见[可观测性](observability.md)。
