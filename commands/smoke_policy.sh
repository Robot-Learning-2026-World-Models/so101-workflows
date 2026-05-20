#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# CKPT_DIR="$REPO_ROOT/checkpoints/mimic-finetune"
CKPT_DIR="${MIMIC_CKPT_DIR:-$HOME/Desktop/mimic-inference/checkpoints}"
MIMIC_VIDEO_MODEL_PATH="$CKPT_DIR/iter_000002800_fused.pt"
MIMIC_ACTION_MODEL_PATH="$CKPT_DIR/iter_000001850.pt"
MIMIC_STATS_PATH="$CKPT_DIR/dataset_statistics/so101_task1_2_stats.json"
# PROMPT_EMBEDDING_PATH="$CKPT_DIR/prompt_embeddings/so101_task1_2.pt"
PROMPT_EMBEDDING_PATH="${PROMPT_EMBEDDING_PATH:-$REPO_ROOT/checkpoints/inference_t5_embeddings/the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle}"

for path in "$MIMIC_VIDEO_MODEL_PATH" "$MIMIC_ACTION_MODEL_PATH" "$MIMIC_STATS_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$PROMPT_EMBEDDING_PATH" ]]; then
  echo "Missing prompt embedding: $PROMPT_EMBEDDING_PATH" >&2
  echo "Create it on the cloud with scripts/encode_prompt_embedding.py, then copy it here." >&2
  exit 1
fi

PYTHON="${PYTHON:-$REPO_ROOT/mimic-video/model/.venv/bin/python}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$REPO_ROOT/mimic-video/configs/so101/idm_train_dual_task_low_rank.yaml}"

args=(
  "$REPO_ROOT/scripts/smoke_mimic_video_policy.py"
  --video-model-path "$MIMIC_VIDEO_MODEL_PATH"
  --action-model-path "$MIMIC_ACTION_MODEL_PATH"
  --stats-path "$MIMIC_STATS_PATH"
  --training-config "$TRAINING_CONFIG"
  --prompt-embedding-path "$PROMPT_EMBEDDING_PATH"
  --device "${POLICY_DEVICE:-cuda}"
  --seed "${MIMIC_SEED:-0}"
)

if [[ -n "${SMOKE_IMAGE_PATH:-}" ]]; then
  args+=(--image-path "$SMOKE_IMAGE_PATH")
fi

if [[ -n "${SMOKE_NUM_SAMPLING_STEPS:-}" ]]; then
  args+=(--num-sampling-steps "$SMOKE_NUM_SAMPLING_STEPS")
fi

if [[ -n "${SMOKE_STOP_VIDEO_DENOISING_STEP:-}" ]]; then
  args+=(--stop-video-denoising-step "$SMOKE_STOP_VIDEO_DENOISING_STEP")
fi

if [[ "${MIMIC_USE_CUDA_GRAPHS:-1}" == "0" ]]; then
  args+=(--no-use-cuda-graphs)
fi

PYTHONPATH=$REPO_ROOT/mimic-video/model:$REPO_ROOT/lerobot_policy_mimic_video/src \
"$PYTHON" "${args[@]}"
