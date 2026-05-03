#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
DATASET_ARG=""
EPISODE_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_ARG="${2:-}"
      shift 2
      ;;
    --episode)
      EPISODE_ARG="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/replay.sh --dataset <name|owner/name> --episode <n>" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing .env file at: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

for var_name in FOLLOWER_PORT FOLLOWER_ID; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required env var: ${var_name}" >&2
    exit 1
  fi
done

if [[ -z "${DATASET_ARG}" || -z "${EPISODE_ARG}" ]]; then
  echo "Usage: bash scripts/replay.sh --dataset <name|owner/name> --episode <n>" >&2
  exit 1
fi

if [[ "${DATASET_ARG}" == */* ]]; then
  REPO_ID="${DATASET_ARG}"
else
  if [[ -z "${HF_DATASET_OWNER:-}" ]]; then
    echo "Short --dataset requires HF_DATASET_OWNER in .env (or pass full owner/name)." >&2
    exit 1
  fi
  REPO_ID="${HF_DATASET_OWNER}/${DATASET_ARG}"
fi

uv run lerobot-replay \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.episode="${EPISODE_ARG}" \
  --dataset.fps=30
