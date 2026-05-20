#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAJECTORY_PATH="${1:-}"
EXTRA_ARGS=()
if [[ -n "$TRAJECTORY_PATH" && "$TRAJECTORY_PATH" == --* ]]; then
  EXTRA_ARGS=("$@")
  TRAJECTORY_PATH=""
elif [[ -n "$TRAJECTORY_PATH" ]]; then
  EXTRA_ARGS=("${@:2}")
fi

if [[ -z "$TRAJECTORY_PATH" ]]; then
  TRAJECTORY_PATH="$(find "$REPO_ROOT/outputs/smoke_policy_trajectories" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$TRAJECTORY_PATH" ]]; then
  echo "No trajectory JSON found. Run ./commands/smoke_policy.sh first, or pass a trajectory path." >&2
  exit 1
fi

PYTHON="${LEROBOT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/outputs/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"

"$PYTHON" "$REPO_ROOT/scripts/trajectory_to_lerobot_dataset.py" "$TRAJECTORY_PATH" \
  --repo-id rl26-world-models/inference-viz-1 \
  --output-root "$REPO_ROOT/outputs/hf_trajectory_datasets" \
  --include-video \
  --push-to-hub \
  "${EXTRA_ARGS[@]}"
