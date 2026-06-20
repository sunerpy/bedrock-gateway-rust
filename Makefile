SHELL = /bin/bash
PROJECT_NAME = bedrock-gateway
PROJECT_ROOT = $(abspath .)
DIST_DIR = dist
DOCKER_IMAGE = sunerpy/bedrock-gateway-rust

GIT_TAG = $(shell git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
BUILD_TIME = $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT_ID = $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

RUST_FILES = $(shell find . -name "*.rs" -not -path "./target/*")
TOML_FILES = $(shell find . -name "*.toml" -not -path "./target/*")

.PHONY: all build build-binaries build-local docker-build docker-release fmt lint test run clean help hooks setup-hooks

all: fmt lint build

build: fmt
	@echo "Building $(PROJECT_NAME) in release mode..."
	@mkdir -p $(DIST_DIR)
	cargo build --release
	@cp target/release/$(PROJECT_NAME) $(DIST_DIR)/$(PROJECT_NAME)
	@echo "Binary: $(DIST_DIR)/$(PROJECT_NAME)"

build-local:
	@echo "Building $(PROJECT_NAME) for local environment..."
	@mkdir -p $(DIST_DIR)
	cargo build
	@cp target/debug/$(PROJECT_NAME) $(DIST_DIR)/$(PROJECT_NAME)-debug
	@echo "Debug binary: $(DIST_DIR)/$(PROJECT_NAME)-debug"

build-binaries: fmt
	@echo "Building multi-platform binaries with cargo-zigbuild..."
	@if ! command -v cargo-zigbuild > /dev/null 2>&1; then \
		echo "cargo-zigbuild not found. Install with:"; \
		echo "  cargo install cargo-zigbuild"; \
		exit 1; \
	fi
	@mkdir -p $(DIST_DIR)
	@echo "Building for x86_64-unknown-linux-musl..."
	cargo zigbuild --release --target x86_64-unknown-linux-musl
	@tar -czf $(DIST_DIR)/$(PROJECT_NAME)-x86_64-linux-musl.tar.gz -C target/x86_64-unknown-linux-musl/release $(PROJECT_NAME)
	@echo "Created: $(DIST_DIR)/$(PROJECT_NAME)-x86_64-linux-musl.tar.gz"
	@echo "Building for aarch64-unknown-linux-musl..."
	cargo zigbuild --release --target aarch64-unknown-linux-musl
	@tar -czf $(DIST_DIR)/$(PROJECT_NAME)-aarch64-linux-musl.tar.gz -C target/aarch64-unknown-linux-musl/release $(PROJECT_NAME)
	@echo "Created: $(DIST_DIR)/$(PROJECT_NAME)-aarch64-linux-musl.tar.gz"

docker-build: build
	@echo "Building Docker image $(DOCKER_IMAGE):$(GIT_TAG)..."
	docker build \
		--build-arg VERSION=$(GIT_TAG) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		--build-arg COMMIT_ID=$(COMMIT_ID) \
		-t $(DOCKER_IMAGE):$(GIT_TAG) \
		-t $(DOCKER_IMAGE):latest .
	@echo "Docker image built: $(DOCKER_IMAGE):$(GIT_TAG)"

docker-release: docker-build
	@echo "Pushing Docker image with buildx (multi-arch)..."
	@if ! command -v docker > /dev/null 2>&1 || ! docker buildx ls > /dev/null 2>&1; then \
		echo "docker buildx not available. Ensure:"; \
		echo "  1. Docker buildx is installed"; \
		echo "  2. You are logged in: docker login"; \
		exit 1; \
	fi
	docker buildx build \
		--push \
		--platform linux/amd64,linux/arm64 \
		--build-arg VERSION=$(GIT_TAG) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		--build-arg COMMIT_ID=$(COMMIT_ID) \
		-t $(DOCKER_IMAGE):$(GIT_TAG) \
		-t $(DOCKER_IMAGE):latest .
	@echo "Multi-arch image pushed to Docker Hub"

fmt: fmt-rust fmt-config
	@echo "Formatting complete."

fmt-rust:
	@echo "Formatting Rust code..."
	@if ! command -v cargo-fmt > /dev/null 2>&1; then \
		echo "cargo-fmt not found. Installing..."; \
		rustup component add rustfmt; \
	fi
	cargo fmt --all

fmt-config:
	@echo "Formatting config files with oxfmt (if available)..."
	@if command -v oxfmt > /dev/null 2>&1; then \
		oxfmt --no-error-on-unmatched-pattern "$(PROJECT_ROOT)" 2>/dev/null || true; \
	else \
		echo "(oxfmt not found; skipping config formatting - install with: cargo install oxfmt)"; \
	fi

lint:
	@echo "Running clippy linter..."
	@if ! command -v cargo-clippy > /dev/null 2>&1; then \
		echo "cargo-clippy not found. Installing..."; \
		rustup component add clippy; \
	fi
	cargo clippy --all-targets --all-features -- -D warnings
	@echo "Clippy check passed."

test:
	@echo "Running tests..."
	@mkdir -p $(DIST_DIR)
	cargo test --all-features
	@echo "All tests passed."

run: build
	@echo "Running $(PROJECT_NAME)..."
	API_KEY=testkey ./$(DIST_DIR)/$(PROJECT_NAME)

# 一次性启用版本化的 git 钩子（克隆后执行一次即可）。
# 将 core.hooksPath 指向 .githooks，从而启用推送前门禁（pre-push）。
# 仅在 push 时校验 fmt + clippy + test，commit 不受影响。
hooks setup-hooks:
	@echo "Enabling version-controlled git hooks (core.hooksPath -> .githooks)..."
	git config core.hooksPath .githooks
	@echo "✅ Done. The pre-push gate is now active (fmt + clippy + test on push)."
	@echo "   Plain 'git commit' stays unblocked; only 'git push' is gated."

clean:
	@echo "Cleaning..."
	cargo clean
	@rm -rf $(DIST_DIR)
	@echo "Clean complete."

help:
	@echo "=== $(PROJECT_NAME) Makefile Targets ==="
	@echo ""
	@echo "Main targets:"
	@echo "  all              - Default: fmt → lint → build"
	@echo "  build            - Build release binary (with fmt pre-check)"
	@echo "  build-local      - Build debug binary for local dev"
	@echo "  build-binaries   - Cross-compile musl binaries (x86_64, aarch64)"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build     - Build Docker image locally"
	@echo "  docker-release   - Push multi-arch image (linux/amd64,arm64) to Docker Hub"
	@echo ""
	@echo "Code quality:"
	@echo "  fmt              - Format Rust + config files (cargo fmt + oxfmt)"
	@echo "  fmt-rust         - Format Rust code only"
	@echo "  fmt-config       - Format config files with oxfmt (optional)"
	@echo "  lint             - Run clippy with -D warnings"
	@echo "  test             - Run all tests (lib + doc)"
	@echo ""
	@echo "Utilities:"
	@echo "  run              - Build and run binary (API_KEY=testkey)"
	@echo "  hooks            - Enable pre-push git hook (run once after clone)"
	@echo "  clean            - Remove build artifacts and dist/"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Environment:"
	@echo "  PROJECT_NAME     = $(PROJECT_NAME)"
	@echo "  DOCKER_IMAGE     = $(DOCKER_IMAGE)"
	@echo "  GIT_TAG          = $(GIT_TAG)"
	@echo "  COMMIT_ID        = $(COMMIT_ID)"
	@echo ""
	@echo "Examples:"
	@echo "  make all                # Default workflow"
	@echo "  make fmt lint test      # Code quality checks"
	@echo "  make build-binaries     # Multi-platform build"
	@echo "  make docker-release     # Push to Docker Hub"
