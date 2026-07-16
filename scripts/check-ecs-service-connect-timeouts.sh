#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to validate ECS Service Connect timeouts" >&2
  exit 2
fi

input="${1:--}"

if [[ "$input" != "-" && ! -r "$input" ]]; then
  echo "error: cannot read ECS service description: $input" >&2
  exit 2
fi

config="$({
  jq -c '
    if has("enabled") then
      .
    elif (.services? | type) == "array" then
      (
        .services[0].deployments[]?
        | select(.status == "PRIMARY")
        | .serviceConnectConfiguration
      ) // .services[0].serviceConnectConfiguration // {enabled: false}
    else
      .
    end
  ' "$input"
} 2>/dev/null)" || {
  echo "error: invalid ECS service/Service Connect JSON: $input" >&2
  exit 2
}

enabled="$(jq -r '.enabled // false' <<<"$config")"
if [[ "$enabled" != "true" ]]; then
  echo "ok: ECS Service Connect is disabled; no proxy request timeout applies"
  exit 0
fi

service_count="$(jq -r '(.services // []) | length' <<<"$config")"
if [[ "$service_count" -eq 0 ]]; then
  echo "error: Service Connect is enabled but contains no service configuration" >&2
  exit 1
fi

failed=0
while IFS=$'\t' read -r port_name per_request idle; do
  if [[ "$per_request" == "missing" ]]; then
    echo "error: Service Connect service '$port_name' omits timeout.perRequestTimeoutSeconds; AWS therefore applies its 15-second HTTP default and can truncate SSE/tool calls" >&2
    failed=1
  elif [[ "$per_request" != "0" ]]; then
    echo "error: Service Connect service '$port_name' sets perRequestTimeoutSeconds=$per_request; streaming endpoints require 0 (disabled)" >&2
    failed=1
  fi

  if [[ "$idle" != "missing" && "$idle" != "0" && "$idle" -lt 180 ]]; then
    echo "error: Service Connect service '$port_name' sets idleTimeoutSeconds=$idle; it must be 0 or at least the gateway's 180-second upstream idle timeout" >&2
    failed=1
  fi
done < <(
  jq -r '
    (.services // [])[]
    | [
        (.portName // "<unknown>"),
        (if .timeout.perRequestTimeoutSeconds == null then "missing" else (.timeout.perRequestTimeoutSeconds | tostring) end),
        (if .timeout.idleTimeoutSeconds == null then "missing" else (.timeout.idleTimeoutSeconds | tostring) end)
      ]
    | @tsv
  ' <<<"$config"
)

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "ok: Service Connect total request timeout is disabled for every service"
