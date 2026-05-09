#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
CONFIG_PATH=""
OVERWRITE=0
RESUME=0
CAMERAS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --camera)
      CAMERAS+=("${2:-}")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/record.sh --config path/to/config.json [--overwrite] [--resume]" >&2
      echo "                              [--camera name,index,width,height,fps] ..." >&2
      exit 1
      ;;
  esac
done

if [[ "${OVERWRITE}" -eq 1 && "${RESUME}" -eq 1 ]]; then
  echo "Use only one of --overwrite or --resume." >&2
  exit 1
fi

if [[ -n "${CONFIG_PATH}" && "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
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

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Missing required argument: --config path/to/config.json" >&2
  echo "Usage: bash scripts/record.sh --config path/to/config.json [--overwrite] [--resume]" >&2
  exit 1
fi

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

pairs = {
    "PROFILE_NAME": cfg.get("profile_name"),
    "HF_DATASET_OWNER": cfg.get("hf_dataset_owner"),
    "DATASET_NAME": cfg.get("dataset_name"),
    "TASK": cfg.get("task"),
    "NUM_EPISODES": cfg.get("num_episodes"),
    "EPISODE_TIME_S": cfg.get("episode_time_s"),
    "RESET_TIME_S": cfg.get("reset_time_s"),
}

for key, value in pairs.items():
    if value is not None and value != "":
        print(f"{key}={shlex.quote(str(value))}")

# Build CAMERAS_FROM_CONFIG: prefer "cameras" list, fall back to singular "camera"
cameras_list = cfg.get("cameras")
if cameras_list:
    entries = []
    for cam in cameras_list:
        name  = cam.get("name", "front")
        index = cam.get("fallback_index", 0)
        w     = cam.get("width", 640)
        h     = cam.get("height", 480)
        fps   = cam.get("fps", 30)
        entries.append(f"{name},{index},{w},{h},{fps}")
    print(f"CAMERAS_FROM_CONFIG={shlex.quote('|'.join(entries))}")
else:
    cam = cfg.get("camera", {})
    index = cam.get("fallback_index", 0)
    w     = cam.get("width", 640)
    h     = cam.get("height", 480)
    fps   = cam.get("fps", 30)
    print(f"CAMERAS_FROM_CONFIG={shlex.quote(f'front,{index},{w},{h},{fps}')}")
PY
)"

# Optional overrides via env vars
: "${HF_DATASET_OWNER:=}"
: "${NUM_EPISODES:=20}"
: "${EPISODE_TIME_S:=25}"
: "${RESET_TIME_S:=10}"
: "${HF_DATASET_PRIVATE:=true}"

required_config_vars=(
  DATASET_NAME
  TASK
)

for var_name in "${required_config_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required config field: ${var_name}" >&2
    echo "Add it to your --config JSON file." >&2
    exit 1
  fi
done

# If no --camera flags given, use cameras from config JSON
if [[ ${#CAMERAS[@]} -eq 0 ]]; then
  IFS='|' read -ra CAMERAS <<< "${CAMERAS_FROM_CONFIG}"
fi

# Validate and build the cameras dict string: { name: {type: opencv, ...}, ... }
CAMERAS_STR="{ "
first=1
for cam in "${CAMERAS[@]}"; do
  IFS=',' read -r cam_name cam_index cam_width cam_height cam_fps <<< "${cam}"
  if [[ -z "${cam_name}" || -z "${cam_index}" || -z "${cam_width}" || -z "${cam_height}" || -z "${cam_fps}" ]]; then
    echo "Invalid --camera value '${cam}': expected name,index,width,height,fps" >&2
    exit 1
  fi
  [[ ${first} -eq 0 ]] && CAMERAS_STR+=", "
  CAMERAS_STR+="${cam_name}: {type: opencv, index_or_path: ${cam_index}, width: ${cam_width}, height: ${cam_height}, fps: ${cam_fps}}"
  first=0
done
CAMERAS_STR+=" }"

HF_DATASET_PRIVATE_NORMALIZED="$(printf '%s' "${HF_DATASET_PRIVATE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${HF_DATASET_PRIVATE_NORMALIZED}" == "true" ]]; then
  HF_DATASET_PRIVATE_NORMALIZED="true"
else
  HF_DATASET_PRIVATE_NORMALIZED="false"
fi

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
LOCAL_DATASET_CACHE="${HOME}/.cache/huggingface/lerobot/${DATASET_OWNER}/${DATASET_NAME}"

echo "Recording dataset to: ${DATASET_OWNER}/${DATASET_NAME}"
echo "Dataset visibility: $([[ "${HF_DATASET_PRIVATE_NORMALIZED}" == "true" ]] && echo "private" || echo "public")"

if [[ "${OVERWRITE}" -eq 1 ]]; then
  if [[ -e "${LOCAL_DATASET_CACHE}" ]]; then
    echo "Removing existing local dataset cache path: ${LOCAL_DATASET_CACHE}"
    rm -rf "${LOCAL_DATASET_CACHE}"
  else
    echo "No local dataset cache path found at: ${LOCAL_DATASET_CACHE}"
  fi
fi

# LeRobot's resume path only auto-downloads `meta/` when the local folder is empty.
# A full Hub upload after recording would then omit existing `data/` and `videos/`,
# effectively overwriting the remote dataset. Pull the complete dataset first so
# resume works on any machine (same as a warm local cache).
if [[ "${RESUME}" -eq 1 ]]; then
  echo "Syncing full dataset from Hugging Face Hub into: ${LOCAL_DATASET_CACHE}"
  echo "(Required so resume does not replace Hub data with metadata-only copy.)"
  HF_REPO_ID="${DATASET_OWNER}/${DATASET_NAME}" \
    HF_LOCAL_ROOT="${LOCAL_DATASET_CACHE}" \
    uv run python - <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

repo_id = os.environ["HF_REPO_ID"]
root = Path(os.environ["HF_LOCAL_ROOT"])
revision = os.environ.get("HF_DATASET_REVISION") or CODEBASE_VERSION

root.mkdir(parents=True, exist_ok=True)

try:
    try:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(root),
        )
    except RevisionNotFoundError:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=str(root),
        )
except HfHubHTTPError as exc:
    print(
        f"Could not download dataset '{repo_id}' from the Hub ({exc}). "
        "Resume needs an existing Hub dataset and network access (or run once online).",
        file=sys.stderr,
    )
    sys.exit(1)
PY
fi

cmd=(
  uv run lerobot-record
  --robot.type=so101_follower
  --robot.port="${FOLLOWER_PORT}"
  --robot.id="${FOLLOWER_ID}"
  --teleop.type=so101_leader
  --teleop.port="${LEADER_PORT}"
  --teleop.id="${LEADER_ID}"
  --robot.cameras="${CAMERAS_STR}"
  --dataset.repo_id="${DATASET_OWNER}/${DATASET_NAME}"
  --dataset.single_task="${TASK}"
  --dataset.num_episodes="${NUM_EPISODES}"
  --dataset.episode_time_s="${EPISODE_TIME_S}"
  --dataset.reset_time_s="${RESET_TIME_S}"
  --dataset.private="${HF_DATASET_PRIVATE_NORMALIZED}"
  --display_data=true
)

if [[ "${RESUME}" -eq 1 ]]; then
  cmd+=(--resume=true)
  cmd+=(--dataset.root="${LOCAL_DATASET_CACHE}")
fi

"${cmd[@]}"