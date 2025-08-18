#!/usr/bin/env bash
set -euo pipefail

# ==== CONFIG ====
PROJECT="inverse-machine-learning-ai-v2"
MACHINE_TYPE="a2-ultragpu-4g"          # 4x A100 80GB
PROVISIONING_MODEL="STANDARD"          # on-demand
REAL_NAME="inverse-a100-4g-clone"
IMAGE_NAME="inverse-a100-clone-img"    # custom image in this project
REAL_BOOT_SIZE="5000GB"
BOOT_DISK_TYPE="pd-balanced"           # or pd-ssd
PROBE_BOOT_SIZE="20GB"
LOOP_SECONDS=60
NETWORK="default"
SSH_TAG="ssh-allowed"

PREFERRED_REGIONS="europe-west4 us-central1 us-east4 asia-southeast1"

gcloud config set project "$PROJECT" >/dev/null

# ---- helpers ----
cleanup_probe() {
  local name="${1:-}" zone="${2:-}"
  [[ -z "$name" || -z "$zone" ]] && return 0
  gcloud compute instances delete "$name" --zone="$zone" --quiet >/dev/null 2>&1 || true
}

region_of() { echo "$1" | sed -E 's/-[a-z]$//'; }

have_quota_for_region() {
  local region="$1"
  local q_a10080 q_a2cpus
  q_a10080="$(gcloud compute regions describe "$region" \
      --format="value(quotas[metric=NVIDIA_A100_80GB_GPUS].limit)" 2>/dev/null || echo 0)"
  q_a2cpus="$(gcloud compute regions describe "$region" \
      --format="value(quotas[metric=A2_CPUS].limit)" 2>/dev/null || echo 0)"
  # Need >=4 GPUs and >=48 A2_CPUS
  awk -v g="$q_a10080" -v c="$q_a2cpus" 'BEGIN{exit !(g>=4 && c>=48)}'
}

list_candidate_zones() {
  gcloud compute accelerator-types list \
    --filter="name=nvidia-a100-80gb" \
    --format="value(zone)" | sort -u
}

ensure_ssh_rule() {
  if ! gcloud compute firewall-rules list \
      --filter="name=default-allow-ssh AND network~${NETWORK}" \
      --format="value(name)" | grep -q . ; then
    gcloud compute firewall-rules create default-allow-ssh \
      --allow=tcp:22 --network="$NETWORK" --quiet >/dev/null
  fi
}

in_list() {
  # usage: in_list "needle" "a b c"
  local needle="$1" list="${2:-}" w
  for w in $list; do [[ "$w" == "$needle" ]] && return 0; done
  return 1
}

sort_zones_by_preference() {
  # usage: sort_zones_by_preference "zone1 zone2 ..." -> prints ordered list
  local zones="$1" ordered="" used="" z pref r
  # first by preferred regions
  for pref in $PREFERRED_REGIONS; do
    for z in $zones; do
      r="$(region_of "$z")"
      if [[ "$r" == "$pref" ]]; then
        ordered="$ordered $z"
        used="$used $z"
      fi
    done
  done
  # then any remaining zones
  for z in $zones; do
    if ! in_list "$z" "$used"; then
      ordered="$ordered $z"
    fi
  done
  echo "$ordered"
}

try_zone() {
  local Z="$1"
  local PROBE="probe-a2-ultra-4g-$Z-$$"
  # shellcheck disable=SC2064
  trap "cleanup_probe '$PROBE' '$Z'" RETURN

  echo "== Probing capacity in $Z (model=$PROVISIONING_MODEL) =="

  set +e
  gcloud compute instances create "$PROBE" \
    --zone="$Z" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model="$PROVISIONING_MODEL" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$PROBE_BOOT_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --network="$NETWORK" \
    --tags="$SSH_TAG" \
    --scopes=cloud-platform \
    --quiet
  local RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "✗ No capacity in $Z (or quota denial)."
    return 1
  fi

  local PMODE
  PMODE="$(gcloud compute instances describe "$PROBE" --zone="$Z" --format='value(scheduling.provisioningModel)')"
  echo "✓ Probe succeeded in $Z (provisioning model reported: $PMODE). Creating REAL instance…"

  cleanup_probe "$PROBE" "$Z"

  gcloud compute instances create "${REAL_NAME}-${Z}" \
    --zone="$Z" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model="$PROVISIONING_MODEL" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$REAL_BOOT_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --image="$IMAGE_NAME" \
    --image-project="$PROJECT" \
    --network="$NETWORK" \
    --tags="$SSH_TAG" \
    --metadata=install-nvidia-driver=True \
    --scopes=cloud-platform

  echo
  echo "✅ Created ${REAL_NAME}-${Z} in $Z"
  echo "SSH: gcloud compute ssh ${REAL_NAME}-${Z} --zone=$Z --project=$PROJECT"
  return 0
}

# === run ===
ensure_ssh_rule
echo "=== A100 80GB hunter (STANDARD) starting — machine: $MACHINE_TYPE ==="

while true; do
  ZONES="$(list_candidate_zones | tr '\n' ' ' | sed 's/ *$//')"
  if [[ -z "$ZONES" ]]; then
    echo "No A100 80GB zones discovered. Retrying in ${LOOP_SECONDS}s…"
    sleep "$LOOP_SECONDS"; continue
  fi
  echo "Discovered zones: $ZONES"

  ELIGIBLE_ZONES=""
  REGIONS_OK=""
  REGIONS_BAD=""

  for Z in $ZONES; do
    R="$(region_of "$Z")"
    if in_list "$R" "$REGIONS_OK"; then
      ELIGIBLE_ZONES="$ELIGIBLE_ZONES $Z"
      continue
    fi
    if in_list "$R" "$REGIONS_BAD"; then
      continue
    fi
    if have_quota_for_region "$R"; then
      REGIONS_OK="$REGIONS_OK $R"
      ELIGIBLE_ZONES="$ELIGIBLE_ZONES $Z"
      echo "Region $R: quota OK."
    else
      REGIONS_BAD="$REGIONS_BAD $R"
      echo "Region $R: quota insufficient. Skipping."
    fi
  done

  ELIGIBLE_ZONES="$(echo "$ELIGIBLE_ZONES" | sed 's/^ *//;s/  */ /g')"
  if [[ -z "$ELIGIBLE_ZONES" ]]; then
    echo "No regions with sufficient quota yet. Retrying in ${LOOP_SECONDS}s…"
    sleep "$LOOP_SECONDS"; continue
  fi

  ORDERED_ZONES="$(sort_zones_by_preference "$ELIGIBLE_ZONES" | sed 's/^ *//;s/  */ /g')"
  echo "Eligible zones (by preference): $ORDERED_ZONES"

  for Z in $ORDERED_ZONES; do
    if try_zone "$Z"; then exit 0; fi
  done

  echo "No capacity landed. Retrying in ${LOOP_SECONDS}s…"
  sleep "$LOOP_SECONDS"
done

