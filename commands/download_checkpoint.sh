#!/usr/bin/env bash
set -euo pipefail

CKPT_NAME="mimic-finetune"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ID="mimicinferencegobrrrr"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints/$CKPT_NAME"

mkdir -p "$CHECKPOINT_DIR"

source .env

uv run --with huggingface-hub --no-project hf download "$REPO_ID/$CKPT_NAME" \
  --local-dir "$CHECKPOINT_DIR" \
  --exclude "text_encoder/**"

ln -sf "$CHECKPOINT_DIR/video_backbone/iter_000002800_fused.pt" \
  "$CHECKPOINT_DIR/iter_000002800_fused.pt"
ln -sf "$CHECKPOINT_DIR/action_decoder/iter_000001850.pt" \
  "$CHECKPOINT_DIR/iter_000001850.pt"

mkdir -p "$REPO_ROOT/mimic-video/model/checkpoints"
ln -sfn "$CHECKPOINT_DIR/video_backbone" \
  "$REPO_ROOT/mimic-video/model/checkpoints/video_backbone"

echo "Downloaded $REPO_ID/$CKPT_NAME to $CHECKPOINT_DIR"
