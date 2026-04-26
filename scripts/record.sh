#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
CONFIG_PATH=""
DEFAULT_CONFIG="${ROOT_DIR}/configs/tasks/so101-first-dataset.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/record.sh [--config path/to/config.json]" >&2
      exit 1
      ;;
  esac
done

if [[ -n "${CONFIG_PATH}" && "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
fi

if [[ -z "${CONFIG_PATH}" && -f "${DEFAULT_CONFIG}" ]]; then
  CONFIG_PATH="${DEFAULT_CONFIG}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing .env file at: ${ENV_FILE}" >&2
  exit 1
fi

# Load .env and export all vars
set -a
source "${ENV_FILE}"
set +a

required_vars=(
  FOLLOWER_PORT
  LEADER_PORT
  FOLLOWER_ID
  LEADER_ID
)

for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required env var: ${var_name}" >&2
    exit 1
  fi
done

# Optional task config override via JSON file.
if [[ -n "${CONFIG_PATH}" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi

  # Parse a task config JSON and export known keys as shell assignments.
  eval "$(
    python - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

camera = cfg.get("camera", {})

pairs = {
    "PROFILE_NAME": cfg.get("profile_name"),
    "HF_DATASET_OWNER": cfg.get("hf_dataset_owner"),
    "DATASET_NAME": cfg.get("dataset_name"),
    "TASK": cfg.get("task"),
    "NUM_EPISODES": cfg.get("num_episodes"),
    "EPISODE_TIME_S": cfg.get("episode_time_s"),
    "RESET_TIME_S": cfg.get("reset_time_s"),
    "CAMERA_INDEX": camera.get("fallback_index"),
    "CAMERA_WIDTH": camera.get("width"),
    "CAMERA_HEIGHT": camera.get("height"),
    "CAMERA_FPS": camera.get("fps"),
}

for key, value in pairs.items():
    if value is not None and value != "":
        print(f"{key}={shlex.quote(str(value))}")
PY
  )"
fi

# Optional overrides via env vars
: "${DATASET_NAME:=so101-first-dataset}"
: "${HF_DATASET_OWNER:=}"
: "${TASK:=pick up the object and place it in the target area}"
: "${NUM_EPISODES:=20}"
: "${EPISODE_TIME_S:=25}"
: "${RESET_TIME_S:=10}"
: "${CAMERA_INDEX:=0}"
: "${CAMERA_WIDTH:=640}"
: "${CAMERA_HEIGHT:=480}"
: "${CAMERA_FPS:=30}"

# Ensure HF CLI auth exists (do not force interactive login in script)
if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not logged into Hugging Face CLI." >&2
  echo "Run: hf auth login" >&2
  exit 1
fi

HF_WHOAMI_OUTPUT="$(NO_COLOR=1 hf auth whoami)"
HF_USER="$(
  awk -F': *' '
    /^[[:space:]]*user:[[:space:]]*/ {print $2; exit}
  ' <<<"${HF_WHOAMI_OUTPUT}"
)"
if [[ -z "${HF_USER}" ]]; then
  echo "Could not determine Hugging Face username from 'hf auth whoami'." >&2
  echo "Output was:" >&2
  echo "${HF_WHOAMI_OUTPUT}" >&2
  exit 1
fi

DATASET_OWNER="${HF_DATASET_OWNER:-$HF_USER}"

echo "Recording dataset to: ${DATASET_OWNER}/${DATASET_NAME}"

uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}" \
  --dataset.repo_id="${DATASET_OWNER}/${DATASET_NAME}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --display_data=true