#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ROOT="${1:-}"
EXTRA_ARGS=()
if [[ -n "$DATASET_ROOT" && "$DATASET_ROOT" == --* ]]; then
  EXTRA_ARGS=("$@")
  DATASET_ROOT=""
elif [[ -n "$DATASET_ROOT" ]]; then
  EXTRA_ARGS=("${@:2}")
fi

if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT="$(find "$REPO_ROOT/outputs/lerobot_trajectory_datasets" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$DATASET_ROOT" ]]; then
  echo "No converted trajectory dataset found. Run ./commands/trajectory_to_lerobot_dataset.sh first." >&2
  exit 1
fi

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  echo "Not a LeRobot dataset root: $DATASET_ROOT" >&2
  exit 1
fi

SAVE_MODE=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--save" ]]; then
    SAVE_MODE=1
  fi
done

if [[ "$SAVE_MODE" == "0" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && -z "${WAYLAND_SOCKET:-}" ]]; then
  echo "No desktop display detected in this shell." >&2
  echo "Run this from a terminal opened on the Ubuntu desktop, or add --save 1 --output-dir outputs/rerun_trajectory_rrd." >&2
fi

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/outputs/hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"

REPO_ID="local/$(basename "$DATASET_ROOT")"

exec "$REPO_ROOT/.venv/bin/lerobot-dataset-viz" \
  --repo-id "$REPO_ID" \
  --root "$DATASET_ROOT" \
  --episode-index 0 \
  "${EXTRA_ARGS[@]}"
