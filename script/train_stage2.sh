#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Allow overriding via environment variables but provide sensible defaults
LABELS="${LABELS:-$ROOT/data/WLBSL/WLBSL_Labels.csv}"
SAVE="${SAVE:-$ROOT/checkpoints/wlbs_stage2_words}"
FINETUNE="${FINETUNE:-$ROOT/checkpoints/pretrain_wlbs/best.pt}"   # override if different

mkdir -p "$SAVE"

ARGS=(
  --stage 2
  --task ISLR
  --dataset WLASL
  --labels     "$LABELS"
  --output_dir "$SAVE"
  --device    cuda
)

if [[ -f "$FINETUNE" ]]; then
  echo "Using finetune checkpoint: $FINETUNE"
  ARGS+=(--finetune "$FINETUNE")
else
  echo "⚠️  FINETUNE not found at: $FINETUNE — training Stage-2 from scratch."
fi

PYTHONPATH="$ROOT:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 fine_tuning.py "${ARGS[@]}"
