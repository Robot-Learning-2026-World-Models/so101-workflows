#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAJECTORY_PATH="${1:-}"
if [[ -z "$TRAJECTORY_PATH" ]]; then
  TRAJECTORY_PATH="$(find "$REPO_ROOT/outputs/smoke_policy_trajectories" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$TRAJECTORY_PATH" ]]; then
  echo "No trajectory JSON found. Run ./commands/smoke_policy.sh first, or pass a trajectory path." >&2
  exit 1
fi

PYTHON="${LEROBOT_PYTHON:-$REPO_ROOT/.venv/bin/python}"

"$PYTHON" "$REPO_ROOT/scripts/trajectory_to_lerobot_dataset.py" "$TRAJECTORY_PATH" "${@:2}"
