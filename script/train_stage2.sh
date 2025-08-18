#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/projects/dev/inverse"
cd "$ROOT"

POSE_ROOT="$ROOT/data/WLBSL/pose_format"
LABELS="$ROOT/data/WLBSL/WLBSL_Labels.csv"
SAVE="$ROOT/checkpoints/wlbs_stage2_words"
FINETUNE="${FINETUNE:-$ROOT/checkpoints/pretrain_wlbs/best.pt}"   # override if different
mkdir -p "$SAVE"

ARGS=(
  --stage 2
  --task ISLR
  --dataset WLASL
  --pose-root "$POSE_ROOT"
  --labels    "$LABELS"
  --save-dir  "$SAVE"
  --device    cuda
)

if [[ -f "$FINETUNE" ]]; then
  echo "Using finetune checkpoint: $FINETUNE"
  ARGS+=(--finetune "$FINETUNE")
else
  echo "⚠️  FINETUNE not found at: $FINETUNE — training Stage-2 from scratch."
fi

PYTHONPATH="$ROOT:$PYTHONPATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 fine_tuning.py "${ARGS[@]}"
