# Docker Deployment

The root `Dockerfile` builds a minimal distroless image using a two-stage build:
a `rust:bookworm` builder stage that compiles a statically-linked musl binary, and a
`gcr.io/distroless/static-debian12:nonroot` runtime stage with no shell, no package
manager, and no extra attack surface.

Compressed image size is roughly 12 MB.

---

## Pull from Docker Hub

```bash
docker pull sunerpy/bedrock-gateway-rust
```

## Build locally

```bash
docker build -t bedrock-gateway-rust .
```

---

## Run

### Recommended: Bedrock API Key

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -e AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-... \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

### Traditional: SigV4 access key

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=AKIA... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

### IAM instance/task role (EC2 or ECS)

Omit both `AWS_BEARER_TOKEN_BEDROCK` and the access key pair — the SDK picks up
instance credentials via IMDS automatically.

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

---

## Environment Variables

See [../../README.md#configuration](../../README.md#configuration) for the full variable
reference. The minimal set:

| Variable                   | Required    | Description                                   |
| -------------------------- | ----------- | --------------------------------------------- |
| `API_KEY`                  | Yes         | Bearer token your clients send to the gateway |
| `AWS_BEARER_TOKEN_BEDROCK` | Recommended | Bedrock API Key the gateway sends to AWS      |
| `AWS_REGION`               | Recommended | AWS region (defaults to `us-west-2`)          |

---

## Verify

```bash
curl http://localhost:8080/api/v1/health
# OK
```

---

## Config override

The binary embeds `config/*.toml` at compile time as a fallback. To override config
at runtime without rebuilding the image, mount an external config directory and set
`CONFIG_DIR`:

```bash
docker run \
  -e API_KEY=sk-my-secret-key \
  -e AWS_REGION=us-east-1 \
  -e CONFIG_DIR=/etc/bgw/config \
  -v /host/path/to/config:/etc/bgw/config:ro \
  -p 8080:8080 \
  sunerpy/bedrock-gateway-rust
```

---

## Docker Compose example

```yaml
services:
  bedrock-gateway:
    image: sunerpy/bedrock-gateway-rust
    ports:
      - "8080:8080"
    environment:
      API_KEY: sk-my-secret-key
      AWS_REGION: us-east-1
      AWS_BEARER_TOKEN_BEDROCK: bedrock-api-key-...
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8080/api/v1/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## Notes

- The container runs as non-root user `65532` (distroless `nonroot`).
- The default working directory inside the container is `/etc/bedrock-gateway`. Config
  files are copied to `/etc/bedrock-gateway/config/` at build time.
- Port 8080 is the default. Override with `-e PORT=<port>` and adjust the `-p` mapping.
