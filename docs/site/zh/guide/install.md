# 安装

bedrock-gateway 以三种形式发布：容器镜像、每个平台各一个的二进制文件，以及一个 crate。三者运行的是同一个程序，使用同样的配置。请根据运行环境选择合适的一种。

## 容器镜像

镜像发布在 Docker Hub 和 Amazon ECR Public 上，提供 `linux/amd64` 和 `linux/arm64` 两种架构。它是一个 distroless 镜像，只包含二进制文件和它的配置文件。

| 镜像仓库          | 镜像                                           |
| ----------------- | ---------------------------------------------- |
| Docker Hub        | `sunerpy/bedrock-gateway-rust`                 |
| Amazon ECR Public | `public.ecr.aws/k5i7b1e0/bedrock-gateway-rust` |

每个版本都同时带有版本号标签（例如 `0.17.0`）和 `latest` 标签。生产环境请固定版本：

```sh
docker pull sunerpy/bedrock-gateway-rust:0.17.0
```

从 ECR Public 拉取不需要登录，因此 ECS 任务或 Lambda 函数从这里拉取镜像不受 Docker Hub 的速率限制。镜像监听 8080 端口，以非 root 用户运行。

## 二进制文件

每个 [GitHub release](https://github.com/sunerpy/bedrock-gateway-rust/releases) 都为每个平台提供一个压缩包。压缩包里只有 `bedrock-gateway` 这一个二进制文件（Windows 上为 `bedrock-gateway.exe`），配置已经内置在其中。

| 平台                | 压缩包                                                        |
| ------------------- | ------------------------------------------------------------- |
| Linux x64           | `bedrock-gateway-<version>-x86_64-unknown-linux-musl.tar.gz`  |
| Linux ARM64         | `bedrock-gateway-<version>-aarch64-unknown-linux-musl.tar.gz` |
| macOS Intel         | `bedrock-gateway-<version>-x86_64-apple-darwin.tar.gz`        |
| macOS Apple Silicon | `bedrock-gateway-<version>-aarch64-apple-darwin.tar.gz`       |
| Windows x64         | `bedrock-gateway-<version>-x86_64-pc-windows-msvc.zip`        |

使用 GitHub CLI 下载适合你平台的最新压缩包并解压：

```sh
gh release download --repo sunerpy/bedrock-gateway-rust --pattern '*-x86_64-unknown-linux-musl.tar.gz'
tar -xzf bedrock-gateway-*-x86_64-unknown-linux-musl.tar.gz
./bedrock-gateway
```

Linux 二进制文件是静态链接的，可以在任何发行版上运行。

## Cargo

```sh
cargo install bedrock-gateway-rust
```

这条命令构建 `bedrock-gateway` 二进制文件并放到 `~/.cargo/bin` 中。它需要 Rust 工具链；项目构建所用的版本固定在 [`rust-toolchain.toml`](../../../../rust-toolchain.toml) 中。

## 从源码构建

```sh
git clone https://github.com/sunerpy/bedrock-gateway-rust
cd bedrock-gateway-rust
cargo build --release
./target/release/bedrock-gateway
```

如果要用 OpenTelemetry 导出追踪和指标，构建时加上 `--features otel`，见[可观测性](../operate/observability.md)。发布的二进制文件和镜像在构建时不带这个功能。

## 配置文件

二进制文件内置了一份配置：模型注册表（`models.toml`）、区域路由（`regions.toml`）和嵌入模型（`embeddings.toml`）。启动时，它在 `CONFIG_DIR` 中查找这些文件；这个目录默认是工作目录下的 `config`。文件存在且能正确解析时，会取代内置的那一份；文件缺失或有错误时，继续使用内置的那一份。镜像把工作目录设为 `/etc/bedrock-gateway`，因此读取的是 `/etc/bedrock-gateway/config/`。模型注册表的说明见[模型](models.md#添加或修改模型)。

## 更新

拉取新的镜像标签，或者下载新的压缩包替换二进制文件，然后重启网关。网关在两次运行之间不保存任何状态，因此不需要迁移任何数据。

## 卸载

停止网关，然后删除二进制文件或镜像。网关不会自行写入任何文件。
