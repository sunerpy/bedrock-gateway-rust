# 测试覆盖率指南

## 1. 概览

本项目使用 [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)（基于 LLVM source-based coverage）度量单元测试与离线 golden 测试的行覆盖率，目标为 **95%+**。

覆盖率是「**可追踪但非阻塞**」的信息型指标：它帮助我们观察测试完备度的趋势，但**永不使 CI 或合并变红**。这遵循本项目的铁律——**本地绿灯即 CI 绿灯（local green ⇒ CI green）**。覆盖率的所有门禁（本地 `make`、CI `coverage` 作业、Codecov 状态）都被刻意设计为不阻塞合并，详见下文「门禁策略」。

覆盖率只度量单元测试与离线 golden 测试；需要真实 AWS 凭证的实时集成测试层（Live Integration Tier）被自然排除，见「排除范围」。

## 2. 前置安装

覆盖率度量需要两样东西：`cargo-llvm-cov` 子命令，以及 Rust 工具链的 `llvm-tools-preview` 组件（提供插桩支持）。

```bash
cargo install cargo-llvm-cov --locked
rustup component add llvm-tools-preview
```

若未安装 `cargo-llvm-cov`，任一 `make coverage*` 目标都会先行中止并打印上述安装提示（`cargo install cargo-llvm-cov --locked`），不会静默失败。

## 3. 本地运行

所有覆盖率操作都通过 `Makefile` 目标完成。本项目是单 crate（package `bedrock-gateway-rust`，二进制 `bedrock-gateway`），因此底层 `cargo llvm-cov` 命令均**不带** `--workspace`，统一使用 `--all-features`（与现有 fmt/clippy/test 的特性集一致）。

| 目标 | 用途 | 产物路径 |
| --- | --- | --- |
| `make coverage` | 将覆盖率摘要（line/region/function %）打印到 stdout | 无（仅 stdout） |
| `make coverage-html` | 生成可浏览的 HTML 报告 | `target/llvm-cov/html/index.html` |
| `make coverage-lcov` | 生成 LCOV 报告（CI 上传 Codecov 的产物） | `lcov.info` |
| `make coverage-open` | 先构建 HTML 报告，随后尽力打开它 | `target/llvm-cov/html/index.html` |
| `make coverage-clean` | 清除覆盖率插桩 / profraw 数据 | 无 |

常用流程：

```bash
# 快速查看当前覆盖率百分比
make coverage

# 生成并在浏览器中打开逐行标注的 HTML 报告
make coverage-open

# 重新度量前清理旧的插桩数据
make coverage-clean
```

`make coverage-open` 采用「尽力而为」策略：依次尝试 `xdg-open`（Linux）与 `open`（macOS）；若两者都不可用，则打印报告文件路径供手动打开，**绝不使目标失败**。

## 4. 排除范围

### `#[ignore]` 实时集成测试层的自然排除

实时集成测试（Live Integration Tier，如 `src/bedrock/cache.rs` 与 `src/bedrock/models.rs` 中标注 `#[ignore]` 且由 `BEDROCK_INTEGRATION=1` 门控的测试）需要真实 AWS Bedrock 访问权限。

`cargo llvm-cov` 底层调用 `cargo test`，而 `cargo test` 在**不显式传入 `--ignored`（或 `--include-ignored`）**时不会运行 `#[ignore]` 测试。上述覆盖率命令均不带这些开关，因此这些实时测试**天然不被执行**——无需任何额外过滤，也不会因缺少 AWS 凭证而失败。

### `src/main.rs` 的排除理由

`src/main.rs` 仅是组合胶水（`#[tokio::main]` 引导 + `--health-check` 自探针，内部调用 `process::exit`），无业务逻辑且无法在进程内测试框架下执行。因此它在 Codecov 侧被排除——见 `codecov.yml` 的 `ignore` 列表。（`src/api/` 是遗留 Python 参考制品，不属于 Rust crate，本就不会被编译或插桩，无需特殊处理。）

## 5. 门禁策略

覆盖率**永不阻塞合并**，通过三层设计共同保证：

1. **CI `coverage` 作业不入门禁**：`.github/workflows/ci.yml` 中的 `coverage` 作业是一个独立作业，**刻意不出现在** `ci-success` 汇总作业的 `needs: [test, audit]` 列表中。合并硬门禁只取决于 `test` 与 `audit`，与覆盖率结果完全无关。
2. **Codecov 状态为 report-only**：`codecov.yml` 中 project 与 patch 覆盖率状态均设 `target: 95%` + `informational: true`。即使覆盖率低于 95%，Codecov 也只报告差异、绝不使 PR 变红。
3. **上传错误不影响 CI**：`codecov/codecov-action@v5` 设置了 `fail_ci_if_error: false`，Codecov 上传或处理出错时仅报告错误，不使 `coverage` 作业失败。

这三层与本项目铁律一致——**local green ⇒ CI green**：只要本地 `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features` 通过，CI 的硬门禁就会通过。覆盖率是这之上的一个可观测信号，绝不会成为额外的合并阻碍。

## 6. CI 说明

CI 中的 `coverage` 作业（`.github/workflows/ci.yml`）步骤如下：

1. `actions/checkout@v4` — 检出代码
2. `dtolnay/rust-toolchain@stable`（`components: llvm-tools-preview`）— 安装工具链与插桩组件
3. `Swatinem/rust-cache@v2` — 缓存构建产物
4. `taiki-e/install-action@v2`（`tool: cargo-llvm-cov`）— 安装覆盖率工具
5. 生成 LCOV 报告：

   ```bash
   cargo llvm-cov --all-features --lcov --output-path lcov.info
   ```

6. `codecov/codecov-action@v5` — 上传 `lcov.info` 至 Codecov

关于 Codecov token：公开仓库支持 tokenless 上传，`token` 为可选；私有仓库则需配置 `CODECOV_TOKEN` secret。作业中通过 `token: ${{ secrets.CODECOV_TOKEN }}` 传入，配合 `fail_ci_if_error: false`，即便 token 缺失或上传失败也不会阻塞 CI。

CI 的 `coverage` 作业**不注入任何 AWS 凭证**，因此实时集成测试层同样被自然排除，与本地行为保持一致。
