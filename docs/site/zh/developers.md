# 参与开发

本页面向参与 bedrock-gateway 本身开发的人：仓库的结构、改动必须通过的检查，以及版本和本站是怎样发布的。

## 仓库结构

bedrock-gateway 用 Rust 编写，使用 axum、tokio 和 AWS SDK for Rust。`rust-toolchain.toml` 固定了工具链版本，`rustup` 会在第一次构建时安装它。

| 路径                   | 内容                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| `src/server/`          | HTTP 服务器：路由、身份验证，以及为每个模型选择后端的分发器                                |
| `src/openai/`          | Chat Completions、Responses 和 Completions 的 OpenAI 线上数据类型                          |
| `src/bedrock/`         | 与 Bedrock Converse 之间的双向转换、流式处理、缓存、推理、embeddings，以及 OpenAI 兼容后端 |
| `src/config/`          | 设置，以及模型注册表、区域路由和嵌入模型的加载器                                           |
| `src/domain/`          | 服务器调用的 trait，不含 AWS 类型                                                          |
| `src/telemetry/`       | 日志，以及可选的 OpenTelemetry 导出                                                        |
| `config/`              | `models.toml`、`regions.toml`、`embeddings.toml` 和 `app.toml`：网关掌握的全部模型知识     |
| `tests/golden/`        | 录制下来的请求和回答，用来检验转换结果                                                     |
| `deployment/`、`helm/` | CloudFormation 模板和 Helm chart                                                           |
| `docs/`                | 参考文档；`docs/site/` 是本站                                                              |

[`AGENTS.md`](../../../AGENTS.md) 记录了每个改动都要遵循的规则，对开发者和编码 Agent 同样适用。其中两条影响大多数改动：模型知识只写在 `config/models.toml` 中，从不写进代码；Bedrock 独有的功能通过 `extra_body` 传递，从不新增顶层请求字段。

## 构建与测试

```sh
git clone https://github.com/sunerpy/bedrock-gateway-rust
cd bedrock-gateway-rust
make hooks    # 每次 git push 之前运行 make check
make check    # 与 CI 相同：格式检查、全部功能与默认功能各一次 clippy、测试
```

`make check` 会运行两次 clippy：一次带 `--all-features`，把 `otel` 功能一起编译；一次用默认功能，也就是发布的二进制文件和镜像所用的配置。CI 运行 `make check`、Windows 上的测试和 `cargo audit`，`CI Success` 要求三者全部通过，因此通过了这个 hook 的分支也能通过 Linux 上的 CI。

测试离线运行，不需要 AWS 凭据。修改转换逻辑时，要在 `tests/golden/` 中附带一个 fixture；golden 测试按语义比较回答，而不是逐字节比较。调用 Bedrock 的测试默认跳过，只有明确要求时才会运行，并且需要能访问 `us-east-2` 的 Bedrock：

```sh
BEDROCK_INTEGRATION=1 AWS_PROFILE=<your profile> cargo test -- --ignored
```

## 添加模型

添加新模型只需修改 `config/models.toml`，条目的写法见[模型](guide/models.md#添加或修改模型)。Helm chart 在 `helm/bedrock-gateway/files/models.toml` 中保存了一份副本，两者的任何条目不一致时都会有测试失败，因此请同时修改两处。

## Pull request 与发布

- 提交信息和 pull request 标题遵循 Conventional Commits，例如 `fix: preserve image support in Responses translation`。release-please 根据它们确定下一个版本号。
- Pull request 以 squash 方式合并到 `main`，不直接向 `main` 推送任何内容。
- 版本由 release-please 发布：它的 pull request 会更新版本号，合并后创建 tag 和一个草稿 release。同一次运行会检查打 tag 的提交，构建五个平台的二进制文件并逐个运行，连同 `SHA256SUMS` 和证明一起附加到 release，把镜像推送到 Docker Hub 和 Amazon ECR Public，把 crate 发布到 crates.io，最后才公开这个 release。

## 本站

本站的页面放在 `docs/site/` 中，英文在根目录，中文在 `zh/` 下，与它们所描述的代码放在一起，因此改变用户所见内容的改动会在同一个 pull request 中更新对应页面。[`docs/site/README.md`](../README.md) 列出了这些页面的编写规则。站点的主题和构建位于 [sunerpy/firlab](https://github.com/sunerpy/firlab)，每次 `main` 有改动后，它都会把这些页面复制过去。
