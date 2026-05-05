#!/usr/bin/env bash
set -euo pipefail

HOST="${ASYNC_SERVER_HOST:-0.0.0.0}"
PORT="${ASYNC_SERVER_PORT:-8080}"
FPS="${ASYNC_SERVER_FPS:-30}"
INFERENCE_LATENCY="${ASYNC_SERVER_INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${ASYNC_SERVER_OBS_QUEUE_TIMEOUT:-2}"

uv run python -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}"

