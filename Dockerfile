# Multi-stage build: static musl binary → distroless runtime
# Multi-arch aware: honors buildx TARGETARCH (amd64 / arm64).
# Uses cargo-zigbuild for reliable cross-compilation to musl targets
# without per-arch QEMU emulation of the whole build.

# ============================================================
# Builder Stage: Compile static musl binary (cross via zig)
# ============================================================
FROM --platform=$BUILDPLATFORM rust:1.91-bookworm AS builder

# buildx-provided args
ARG TARGETARCH

# Install zig + cargo-zigbuild + musl targets for both arches
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        curl \
        xz-utils && \
    rm -rf /var/lib/apt/lists/* && \
    rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl && \
    cargo install --locked cargo-zigbuild && \
    ZIG_VERSION=0.13.0 && \
    curl -sSL "https://ziglang.org/download/${ZIG_VERSION}/zig-linux-$(uname -m)-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt && \
    ln -s "/opt/zig-linux-$(uname -m)-${ZIG_VERSION}/zig" /usr/local/bin/zig

WORKDIR /build

# Resolve TARGETARCH → Rust target triple
RUN case "${TARGETARCH}" in \
        amd64) echo "x86_64-unknown-linux-musl" > /tmp/rust_target ;; \
        arm64) echo "aarch64-unknown-linux-musl" > /tmp/rust_target ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac

# Copy dependency manifests (enables better layer caching)
COPY Cargo.toml Cargo.lock ./

# Fetch dependencies (layer caching)
RUN cargo fetch

# Copy source code
COPY src ./src
COPY config ./config

# Build release binary with static musl linking (cross via zig)
RUN RUST_TARGET="$(cat /tmp/rust_target)" && \
    cargo zigbuild --release --target "${RUST_TARGET}" && \
    cp "/build/target/${RUST_TARGET}/release/bedrock-gateway" /bedrock-gateway

# ============================================================
# Runtime Stage: Distroless with CA certificates
# ============================================================
FROM gcr.io/distroless/static-debian12:nonroot

# Copy static binary from builder
COPY --from=builder /bedrock-gateway /usr/local/bin/bedrock-gateway

# Copy default configuration (can be overridden via mounted volume)
COPY config /etc/bedrock-gateway/config

# Set working directory for config relative-path resolution
WORKDIR /etc/bedrock-gateway

# Expose service port
EXPOSE 8080

# Set environment variables for config override
# (distroless nonroot uid:gid = 65532:65532, no need for USER)
ENV API_ROUTE_PREFIX="/api/v1" \
    PORT=8080 \
    AWS_REGION="us-west-2" \
    LOG_LEVEL="info"

# Health endpoint: /api/v1/health (hardened path)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/bedrock-gateway", "--health-check"]

ENTRYPOINT ["/usr/local/bin/bedrock-gateway"]
