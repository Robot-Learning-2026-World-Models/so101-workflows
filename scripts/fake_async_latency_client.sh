#!/usr/bin/env bash
set -euo pipefail

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
POLICY_TYPE="${POLICY_TYPE:-act}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
TASK="${TASK:-dummy}"
DURATION_S="${DURATION_S:-30}"
FPS="${FPS:-30}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_KEY="${CAMERA_KEY:-front}"
VERBOSE="${VERBOSE:-true}"
ROBOT_LOG_EVERY="${ROBOT_LOG_EVERY:-30}"

if [[ -z "${PRETRAINED_NAME_OR_PATH}" ]]; then
  echo "Missing PRETRAINED_NAME_OR_PATH. Set it to a local path or HF model id for an ACT/LeRobot policy." >&2
  exit 1
fi

uv run python -m so101_workflows.fake_async_latency_client \
  --server_address="${SERVER_ADDRESS}" \
  --policy_type="${POLICY_TYPE}" \
  --pretrained_name_or_path="${PRETRAINED_NAME_OR_PATH}" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --task="${TASK}" \
  --duration_s="${DURATION_S}" \
  --fps="${FPS}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --verbose="${VERBOSE}" \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=false \
  --robot.type=fake_so101_follower \
  --robot.id=fake_so101 \
  --robot.camera_key="${CAMERA_KEY}" \
  --robot.camera_width="${CAMERA_WIDTH}" \
  --robot.camera_height="${CAMERA_HEIGHT}" \
  --robot.fps="${FPS}" \
  --robot.log_every="${ROBOT_LOG_EVERY}"
