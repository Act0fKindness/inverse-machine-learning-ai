#!/usr/bin/env bash
set -euo pipefail

# ==== CONFIG ====
PROJECT="inverse-machine-learning-ai"
INSTANCE_NAME="inverse-box-v2"
MACHINE_TYPE="a2-highgpu-1g"
GPU_TYPE="nvidia-tesla-a100"
GPU_COUNT=1
DISK_SIZE="3000GB"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
PROVISIONING="STANDARD"   # use "SPOT" if you want preemptible
TERMINATION_ACTION="STOP"

# Regions to try
REGIONS=( "europe-west4" "europe-west1" "us-central1" "us-west4" "us-east1" )

# ==== Helpers ====
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1"; exit 1; }; }
need gcloud
gcloud config set project "$PROJECT" >/dev/null

echo "🔍 Looking for free A100 capacity…"
while true; do
  for REGION in "${REGIONS[@]}"; do
    QUOTA=$(gcloud compute regions describe "$REGION" \
      --format="value(quotas[metric:NVIDIA_A100_40GB].available)" 2>/dev/null || echo "")
    [[ -z "$QUOTA" ]] && continue
    AVAIL=${QUOTA:-0}
    echo "Region $REGION has quota: ${AVAIL}"

    if (( ${AVAIL%.*} >= GPU_COUNT )); then
      ZONES=($(gcloud compute zones list --filter="region:($REGION) status:UP" \
        --format="value(name)" | sort))
      for Z in "${ZONES[@]}"; do
        echo "→ Trying zone $Z…"
        set +e
        gcloud compute instances create "$INSTANCE_NAME" \
          --zone="$Z" \
          --machine-type="$MACHINE_TYPE" \
          --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
          --maintenance-policy=TERMINATE \
          --provisioning-model="$PROVISIONING" \
          --instance-termination-action="$TERMINATION_ACTION" \
          --image-family="$IMAGE_FAMILY" \
          --image-project="$IMAGE_PROJECT" \
          --boot-disk-size="$DISK_SIZE" \
          --boot-disk-type="pd-balanced" \
          --scopes="cloud-platform" \
          --metadata="install-nvidia-driver=True" \
          --tags="ssh-allowed" \
          --quiet
        rc=$?
        set -e
        if [[ $rc -eq 0 ]]; then
          echo "✅ Instance created in $Z"
          echo "SSH: gcloud compute ssh $INSTANCE_NAME --zone=$Z --project=$PROJECT"
          exit 0
        else
          echo "   Zone $Z failed. Trying next…"
        fi
      done
    fi
  done
  echo "⏳ No capacity found. Retrying in 60s…"
  sleep 60
done

