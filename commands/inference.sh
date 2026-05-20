#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${FOLLOWER_PORT:?Set FOLLOWER_PORT to the SO101 follower serial port.}"
: "${FOLLOWER_ID:?Set FOLLOWER_ID to the SO101 follower id.}"
: "${MIMIC_VIDEO_MODEL_PATH:?Set MIMIC_VIDEO_MODEL_PATH to iter_000002800_fused.pt.}"
: "${MIMIC_ACTION_MODEL_PATH:?Set MIMIC_ACTION_MODEL_PATH to iter_000001850.pt.}"
: "${MIMIC_STATS_PATH:?Set MIMIC_STATS_PATH to so101_task1_2_stats.json.}"
: "${PROMPT_EMBEDDING_PATH:?Set PROMPT_EMBEDDING_PATH to the precomputed prompt embedding .pt file.}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_NAME="${DATASET_NAME:-eval_mimic_video_smoke_$RUN_ID}"
DATASET_ROOT="${DATASET_ROOT:-$PWD/data/$DATASET_NAME}"
mkdir -p "$PWD/data"

PYTHONPATH=$PWD/mimic-video/model:$PWD/lerobot_policy_mimic_video/src \
lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30} }" \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.push_to_hub=false \
  --dataset.single_task="the robot arm gently nudges the white object towards the target circle" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=10 \
  --play_sounds=false \
  --policy.type=mimic_video \
  --policy.device=cuda \
  --policy.video_model_path="$MIMIC_VIDEO_MODEL_PATH" \
  --policy.action_model_path="$MIMIC_ACTION_MODEL_PATH" \
  --policy.stats_path="$MIMIC_STATS_PATH" \
  --policy.prompt_embedding_path="$PROMPT_EMBEDDING_PATH" \
  --policy.training_config=$PWD/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml
