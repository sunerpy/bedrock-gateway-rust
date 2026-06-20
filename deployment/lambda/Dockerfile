# syntax=docker/dockerfile:1
#
# Lambda Web Adapter (LWA) container image for the bedrock-gateway binary.
#
# This image runs the *exact same* `bedrock-gateway` axum server used by the
# ECS/Fargate deploy target — there is NO Lambda-specific Rust code and NO
# second binary. AWS Lambda Web Adapter is attached as a Lambda extension that
# transparently proxies API Gateway / Function URL events to the plain HTTP
# server listening on PORT (8080). The binary stays identical across all
# deploy targets (local, ECS, Lambda, Function URL).
#
# Build:
#   docker build -f deployment/lambda/Dockerfile -t bedrock-gateway:lambda .
#
# Notes on parity with task 26 (the main distroless ECS image): both share the
# musl static-build convention (rust:<ver>-alpine builder + x86_64 musl target)
# so the produced binary is fully static and runs on the minimal Lambda base.
# This Dockerfile is intentionally self-contained so it builds even if the ECS
# Dockerfile is not present.

# ---------------------------------------------------------------------------
# Stage 1: build a fully static (musl) release binary
# ---------------------------------------------------------------------------
# Pin to the toolchain in rust-toolchain.toml (1.91.1). musl gives a static
# binary with no glibc dependency, so it runs on any minimal runtime base.
FROM rust:1.91.1-alpine AS builder

# musl-dev + build essentials for static linking; pkgconfig for any C deps.
RUN apk add --no-cache musl-dev build-base pkgconfig

WORKDIR /build

# Target the musl ABI for a static binary.
ENV TARGET=x86_64-unknown-linux-musl
RUN rustup target add ${TARGET}

# Cache dependency compilation: copy manifests first, build a stub, then the
# real sources. This keeps layer caching effective across source-only changes.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
    && echo 'fn main() { println!("stub"); }' > src/main.rs \
    && cargo build --release --target ${TARGET} --bin bedrock-gateway || true \
    && rm -rf src

# Copy the real source tree and config defaults the binary loads at runtime.
COPY src ./src
COPY config ./config

# Force a rebuild of our crate (touch ensures cargo notices the new main.rs).
RUN touch src/main.rs \
    && cargo build --release --target ${TARGET} --bin bedrock-gateway \
    && cp target/${TARGET}/release/bedrock-gateway /build/bedrock-gateway \
    && strip /build/bedrock-gateway

# ---------------------------------------------------------------------------
# Stage 2: Lambda runtime image with the Web Adapter extension
# ---------------------------------------------------------------------------
# Use the AWS-provided minimal Lambda base. The static musl binary has no
# dynamic library dependencies, so this base just needs to host the binary and
# the LWA extension.
FROM public.ecr.aws/lambda/provided:al2023

# --- Attach AWS Lambda Web Adapter as a Lambda extension ---------------------
# LWA is an internal Lambda extension dropped into /opt/extensions. It boots
# alongside the function, waits for PORT to be ready, then translates each
# Lambda invocation (API Gateway / Function URL) into a normal HTTP request to
# the local server and streams the response back.
#
# Version is pinned but intentionally swappable — bump the tag to upgrade.
# See https://github.com/awslabs/aws-lambda-web-adapter/releases
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
     /lambda-adapter /opt/extensions/lambda-adapter

# --- LWA configuration (env, no secrets) -------------------------------------
# PORT                  : port the axum server binds to; LWA proxies to it.
# AWS_LWA_PORT          : explicit LWA target port (mirrors PORT; belt & braces).
# AWS_LWA_INVOKE_MODE   : response_stream enables SSE / streaming responses
#                         through Function URL streaming / API GW response
#                         streaming. Use "buffered" for non-streaming gateways.
# AWS_LWA_READINESS_CHECK_PATH : LWA polls this until 200 before forwarding.
# AWS_LWA_READINESS_CHECK_PORT : port for the readiness probe (same as PORT).
# RUST_LOG              : tracing filter (overridable per deployment).
ENV PORT=8080 \
    AWS_LWA_PORT=8080 \
    AWS_LWA_INVOKE_MODE=response_stream \
    AWS_LWA_READINESS_CHECK_PATH=/api/v1/health \
    AWS_LWA_READINESS_CHECK_PORT=8080 \
    RUST_LOG=info

# Runtime config the binary reads (models.toml etc). Loaded best-effort; the
# server boots even if a file is missing.
COPY config /var/task/config
WORKDIR /var/task

# The application binary — identical to the ECS/local artifact.
COPY --from=builder /build/bedrock-gateway /var/task/bedrock-gateway

# LWA requires overriding the Lambda base image entrypoint so the container
# runs the web server directly instead of the Lambda runtime handler.
ENTRYPOINT []
CMD ["/var/task/bedrock-gateway"]
