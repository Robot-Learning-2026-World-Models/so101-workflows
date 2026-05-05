#!/usr/bin/env bash
set -euo pipefail

# Mac-side latency test for a Brev-hosted LeRobot async policy server.

BREV_INSTANCE="${BREV_INSTANCE:-}"
BREV_HOST="${BREV_HOST:-}"
BREV_SSH_PORT="${BREV_SSH_PORT:-22}"
BREV_PING_COUNT="${BREV_PING_COUNT:-20}"
PORT_FORWARD="${PORT_FORWARD:-true}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
REMOTE_PORT="${REMOTE_PORT:-8080}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-30}"

if [[ -z "${BREV_INSTANCE}" ]]; then
  if ! command -v brev >/dev/null 2>&1; then
    echo "Could not auto-select Brev instance: brev CLI is not installed." >&2
    echo "Set BREV_INSTANCE=<instance-name> or BREV_HOST=<ssh-host>." >&2
    exit 1
  fi

  if BREV_LS_JSON="$(brev ls --json 2>/dev/null)"; then
    BREV_INSTANCE="$(
      BREV_LS_JSON="${BREV_LS_JSON}" python3 - <<'PY'
import json
import os

data = json.loads(os.environ["BREV_LS_JSON"])
workspaces = data.get("workspaces", [])

if not workspaces:
    raise SystemExit(1)

running = [w for w in workspaces if w.get("status") == "RUNNING"]
candidates = running or workspaces

timestamp_keys = ("created_at", "createdAt", "created", "updated_at", "updatedAt")

def timestamp(workspace):
    for key in timestamp_keys:
        value = workspace.get(key)
        if value:
            return value
    return ""

candidates = sorted(candidates, key=timestamp, reverse=True)
print(candidates[0]["name"])
PY
    )"
  fi

  if [[ -z "${BREV_INSTANCE}" ]]; then
    echo "Could not auto-select Brev instance from 'brev ls --json'." >&2
    echo "Set BREV_INSTANCE=<instance-name> or BREV_HOST=<ssh-host>." >&2
    exit 1
  fi
fi

echo "Using Brev instance: ${BREV_INSTANCE}"

if [[ -z "${BREV_HOST}" ]] && command -v ssh >/dev/null 2>&1; then
  BREV_HOST="$(ssh -G "${BREV_INSTANCE}" 2>/dev/null | awk '/^hostname / { print $2; exit }' || true)"
fi

if [[ -n "${BREV_HOST}" ]]; then
  echo "Measuring TCP connect latency to ${BREV_HOST}:${BREV_SSH_PORT}..."
  BREV_HOST="${BREV_HOST}" \
  BREV_SSH_PORT="${BREV_SSH_PORT}" \
  BREV_PING_COUNT="${BREV_PING_COUNT}" \
  python3 - <<'PY'
import os
import socket
import statistics
import time

host = os.environ["BREV_HOST"]
port = int(os.environ["BREV_SSH_PORT"])
count = int(os.environ["BREV_PING_COUNT"])
samples = []

for i in range(count):
    start = time.perf_counter()
    with socket.create_connection((host, port), timeout=5):
        pass
    elapsed_ms = (time.perf_counter() - start) * 1000
    samples.append(elapsed_ms)
    print(f"{i + 1:02d}: {elapsed_ms:.1f} ms")

print(
    "tcp_connect_min_avg_max_ms: "
    f"{min(samples):.1f}/{statistics.fmean(samples):.1f}/{max(samples):.1f}"
)
PY
else
  echo "Skipping TCP latency probe: could not resolve BREV_HOST from ssh -G ${BREV_INSTANCE}."
  echo "Set BREV_HOST=ec2-...compute.amazonaws.com to enable it."
fi

if [[ "${PORT_FORWARD}" == "true" ]]; then
  brev port-forward "${BREV_INSTANCE}" --port "${LOCAL_PORT}:${REMOTE_PORT}"
fi

export SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:${LOCAL_PORT}}"

echo "Waiting for gRPC policy server at ${SERVER_ADDRESS}..."
SERVER_ADDRESS="${SERVER_ADDRESS}" READY_TIMEOUT_S="${READY_TIMEOUT_S}" uv run python - <<'PY'
import os
import sys
import time

import grpc

server_address = os.environ["SERVER_ADDRESS"]
deadline = time.monotonic() + float(os.environ["READY_TIMEOUT_S"])
last_error = None

while time.monotonic() < deadline:
    channel = grpc.insecure_channel(server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=2)
        channel.close()
        print(f"gRPC policy server is ready at {server_address}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - shell script diagnostic
        last_error = exc
        channel.close()
        time.sleep(1)

print(f"Timed out waiting for gRPC policy server at {server_address}: {last_error}", file=sys.stderr)
print("Make sure brev_latency_test_server.sh is running and says 'PolicyServer started'.", file=sys.stderr)
sys.exit(1)
PY

echo
echo "Starting LeRobot async latency client..."

export PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-outputs/fake_act_so101}"
export POLICY_TYPE="${POLICY_TYPE:-act}"
export POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
export CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
export DURATION_S="${DURATION_S:-30}"
export CAMERA_WIDTH="${CAMERA_WIDTH:-160}"
export CAMERA_HEIGHT="${CAMERA_HEIGHT:-120}"
export VERBOSE="${VERBOSE:-true}"
export ROBOT_LOG_EVERY="${ROBOT_LOG_EVERY:-0}"

bash scripts/fake_async_latency_client.sh
