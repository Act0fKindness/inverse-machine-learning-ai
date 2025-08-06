#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

BASE_DIR="/home/danielharding/projects/dev/inverse"
VENV_DIR="/home/danielharding/mmpose-env"
RTMLIB_REPO="https://github.com/Tau-J/rtmlib.git"
RTMLIB_TAG="0.0.13"   # choose the tag you want
MMCV_SRC_DIR="/home/danielharding/mmcv_full_src"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
MIM="$VENV_DIR/bin/mim"

echo "1️⃣  Create & activate virtualenv at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "2️⃣  Upgrade pip, setuptools, wheel, cython"
$PIP install --upgrade pip setuptools wheel cython

echo "3️⃣  Install project Python requirements"
cd "$BASE_DIR"
$PIP install -r requirements.txt

echo "4️⃣  Clone & install RTMLib@$RTMLIB_TAG"
cd "$BASE_DIR"
if [ -d "rtmlib" ]; then
  echo "   Removing existing rtmlib/"
  rm -rf rtmlib
fi
git clone "$RTMLIB_REPO" rtmlib
cd rtmlib
git fetch --tags
git checkout "refs/tags/$RTMLIB_TAG"
$PIP install -e . || {
  echo "❌ RTMLib install failed. Ensure you have permissions and python3-dev installed."
  exit 1
}

echo "5️⃣  Uninstall any mmcv variants"
$PIP uninstall -y mmcv mmcv-full mmcv-lite || true

echo "6️⃣  Install system build deps for mmcv"
sudo apt-get update
sudo apt-get install -y build-essential git python3-dev pkg-config ninja-build

echo "7️⃣  Clone & install MMCV 2.1.0 from source"
cd /home/danielharding
if [ -d "$MMCV_SRC_DIR" ]; then rm -rf "$MMCV_SRC_DIR"; fi
git clone https://github.com/open-mmlab/mmcv.git "$MMCV_SRC_DIR"
cd "$MMCV_SRC_DIR"
git checkout v2.1.0
$PIP install -e . || {
  echo "❌ MMCV build failed. Check CUDA toolkit, NVIDIA drivers, and Python headers."
  exit 1
}

echo "8️⃣  Install OpenMIM, MMEngine, MMPose"
$PIP install --upgrade openmim
$MIM install mmengine
$MIM install mmpose

echo "9️⃣  Quick sanity imports"
$PYTHON - <<'EOF'
import mmcv
import mmpose
from mmcv.ops import MultiScaleDeformableAttention
print("✅ mmcv:", mmcv.__version__)
print("✅ mmpose:", mmpose.__version__)
print("✅ deformable attention ops loaded")
EOF

echo "🔟 Running extract_pose_from_videos.py"
cd "$BASE_DIR"
$PYTHON extract_pose_from_videos.py || {
  echo "❌ extract_pose script failed. Check the traceback above."
  exit 1
}

echo "🎉 All done — pose extraction completed successfully!"

