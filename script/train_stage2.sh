#!/usr/bin/env bash
set -euo pipefail

# Ensure distutils is available (Python 3.12+ drops it)
python3 - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec('distutils') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'])
# Base python packages required for stage-2 fine-tuning. ``mpi4py`` is
# needed by DeepSpeed for its distributed initialisation even when using a
# single GPU.
req = ['torch', 'torchvision', 'torchaudio', 'timm', 'transformers', 'einops', 'deepspeed', 'mpi4py']
missing = [r for r in req if importlib.util.find_spec(r) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
PY

# Resolve repository root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Allow overriding via environment variables but provide sensible defaults
LABELS="${LABELS:-$ROOT/data/WLBSL/WLBSL_Labels.csv}"
SAVE="${SAVE:-$ROOT/checkpoints/wlbs_stage2_words}"
FINETUNE="${FINETUNE:-$ROOT/checkpoints/pretrain_wlbs/best.pt}"   # override if different

mkdir -p "$SAVE"

if [[ ! -f "$LABELS" ]]; then
  echo "❌  Labels CSV not found at: $LABELS" >&2
  exit 1
fi

ARGS=(
  --stage 2
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
