#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
DISPLAY_DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --display-data)
      DISPLAY_DATA="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/teleop.sh [--display-data]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing .env file at: ${ENV_FILE}" >&2
  exit 1
fi

# Export variables loaded from .env so child processes can use them.
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

# Optional camera settings used for teleoperation display.
: "${CAMERA_INDEX:=0}"
: "${CAMERA_WIDTH:=640}"
: "${CAMERA_HEIGHT:=480}"
: "${CAMERA_FPS:=30}"

cmd=(
  uv run lerobot-teleoperate
  --robot.type=so101_follower
  --robot.port="${FOLLOWER_PORT}"
  --robot.id="${FOLLOWER_ID}"
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"
  --teleop.type=so101_leader
  --teleop.port="${LEADER_PORT}"
  --teleop.id="${LEADER_ID}"
)

if [[ -n "${DISPLAY_DATA}" ]]; then
  cmd+=(--display_data="${DISPLAY_DATA}")
fi

"${cmd[@]}"
