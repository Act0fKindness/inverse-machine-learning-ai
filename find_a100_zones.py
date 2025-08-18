#!/usr/bin/env python3
"""
launch_a100_loop.py
Loops until it successfully creates an A100 40GB x4 VM (a2-highgpu-4g).

Defaults:
- project: inverse-machine-learning-ai-v2
- image: inverse-pose-extraction-470000 (stored in 'eu' multi-region)
- instance name: inverse-a100-4g-<timestamp>
- regions searched: europe first; fallbacks optional via --regions

Usage:
  python3 launch_a100_loop.py
  # or customize:
  python3 launch_a100_loop.py --name inverse-a100-4g-worker --regions europe,us,asia --spot

Requires:
- gcloud installed & authenticated
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

PROJECT_DEFAULT = "inverse-machine-learning-ai-v2"
IMAGE_DEFAULT = "inverse-pose-extraction-470000"
MACHINE_TYPE = "a2-highgpu-4g"
ACCELERATOR = "nvidia-tesla-a100"
ACCEL_COUNT = 4

CAPACITY_ERRORS = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
    "RESOURCE_POOL_EXHAUSTED",
    "Insufficient regional quota",
    "does not have enough resources",
    "out of resources",
)
QUOTA_ERRORS = (
    "QUOTA_EXCEEDED",
    "exceeded quota",
    "Insufficient regional quota",
)

def sh_json(cmd):
    try:
        out = subprocess.check_output(cmd + " --format=json", shell=True, stderr=subprocess.STDOUT)
        return json.loads(out.decode() or "[]")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.output.decode())

def sh_text(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return out.decode()
    except subprocess.CalledProcessError as e:
        return e.output.decode()

def get_project(config_project: str | None):
    if config_project:
        return config_project
    return sh_text("gcloud config get-value project").strip() or PROJECT_DEFAULT

def list_up_zones(regions_filter: list[str] | None):
    zones = sh_json("gcloud compute zones list")
    zones = [z for z in zones if z.get("status") == "UP"]
    if regions_filter:
        zones = [z for z in zones if any(z["name"].startswith(r + "-") for r in regions_filter)]
    return zones

def zone_supports_a2(zone: str) -> bool:
    try:
        mts = sh_json(f"gcloud compute machine-types list --zones={zone} --filter='name={MACHINE_TYPE}'")
        return len(mts) > 0
    except Exception:
        return False

def zone_has_a100(zone: str) -> bool:
    try:
        accs = sh_json(f"gcloud compute accelerator-types list --zones={zone}")
    except Exception:
        return False
    for a in accs:
        name = (a.get("name") or "").lower()
        if "a100" in name and "80" not in name:
            return True
    return False

def region_from_zone(zone: str) -> str:
    parts = zone.split("-")
    return "-".join(parts[:3])

def region_quota_remaining(region: str) -> float:
    try:
        info = sh_json(f"gcloud compute regions describe {region}")
    except Exception:
        return 0.0
    qmap = {q.get("metric"): q for q in info.get("quotas", [])}
    for key in ["NVIDIA_A100_GPUS", "A100_GPUS", "NVIDIA_A100_40GB_GPUS", "GPUS"]:
        if key in qmap:
            q = qmap[key]
            return max(0.0, float(q.get("limit", 0)) - float(q.get("usage", 0)))
    # fallback: any GPU metric
    for m, q in qmap.items():
        if "GPU" in m:
            return max(0.0, float(q.get("limit", 0)) - float(q.get("usage", 0)))
    return 0.0

def discover_candidate_zones(prefer_europe_first=True, regions_filter: list[str] | None = None):
    zones = list_up_zones(regions_filter)
    rows = []
    for z in zones:
        zone = z["name"]
        rgn = region_from_zone(zone)
        a2 = zone_supports_a2(zone)
        has = zone_has_a100(zone)
        if a2 and has:
            rem = region_quota_remaining(rgn)
            rows.append((zone, rgn, rem))
    # prioritise europe if requested, then by quota remaining desc
    if prefer_europe_first:
        rows.sort(key=lambda t: (0 if t[0].startswith("europe-") else 1, -t[2], t[0]))
    else:
        rows.sort(key=lambda t: (-t[2], t[0]))
    return rows

def build_create_cmd(project, zone, name, image, spot, boot_size_gb, disk_type, scopes):
    model = "SPOT" if spot else "STANDARD"
    return f"""gcloud compute instances create {name} \
  --project={project} \
  --zone={zone} \
  --provisioning-model={model} \
  --machine-type={MACHINE_TYPE} \
  --maintenance-policy=TERMINATE \
  --accelerator=count={ACCEL_COUNT},type={ACCELERATOR} \
  --image={image} \
  --boot-disk-size={boot_size_gb} \
  --boot-disk-type={disk_type} \
  --metadata="install-nvidia-driver=True" \
  --scopes={scopes}"""

def try_create_once(project, zone, name, image, spot, boot_size_gb, disk_type, scopes) -> tuple[bool, str]:
    cmd = build_create_cmd(project, zone, name, image, spot, boot_size_gb, disk_type, scopes)
    print(f"\n[TRY] {zone}: creating {name} …")
    out = sh_text(cmd)
    print(out.strip()[:2000])  # truncate long output
    lower = out.lower()
    # success heuristic: gcloud prints "Creating instance" then returns 0 if created; but we captured stderr+stdout
    # Detect common failure strings:
    if any(err.lower() in lower for err in [e.lower() for e in CAPACITY_ERRORS]):
        return False, "capacity"
    if any(err.lower() in lower for err in [e.lower() for e in QUOTA_ERRORS]):
        return False, "quota"
    if "not found" in lower and "image" in lower:
        return False, "image"
    if "already exists" in lower:
        return False, "name_taken"
    # quick check: if we can describe it, assume success
    desc = sh_text(f"gcloud compute instances describe {name} --project={project} --zone={zone} --format='get(name)'")
    if desc.strip() == name:
        return True, "ok"
    # fallback: if command returned non-empty without errors, it might still be provisioning
    # One more describe after a short pause:
    time.sleep(5)
    desc2 = sh_text(f"gcloud compute instances describe {name} --project={project} --zone={zone} --format='get(name)'")
    return (desc2.strip() == name, "ok" if desc2.strip() == name else "unknown")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT_DEFAULT, help="GCP project id")
    ap.add_argument("--name", default=f"inverse-a100-4g-{int(time.time())}", help="Instance name")
    ap.add_argument("--image", default=IMAGE_DEFAULT, help="Image name to boot from")
    ap.add_argument("--regions", default="europe", help="Comma list of region prefixes to search (europe,us,asia). Use 'all' for everything.")
    ap.add_argument("--boot-size", default="200GB", help="Boot disk size")
    ap.add_argument("--disk-type", default="pd-ssd", help="Disk type (pd-ssd|pd-balanced|pd-extreme)")
    ap.add_argument("--spot", action="store_true", help="Use SPOT capacity (cheaper, less available). Default off.")
    ap.add_argument("--backoff-min", type=int, default=15, help="Min seconds between attempts")
    ap.add_argument("--backoff-max", type=int, default=120, help="Max seconds between attempts")
    ap.add_argument("--scopes", default="compute-rw,storage-rw", help="OAuth scopes for the instance")
    args = ap.parse_args()

    project = get_project(args.project)
    print(f"# Project: {project}")
    print(f"# Image: {args.image}")
    print(f"# Machine: {MACHINE_TYPE} (A100 x{ACCEL_COUNT})  Spot={args.spot}")
    print(f"# Instance name: {args.name}")

    # region filter
    if args.regions.strip().lower() == "all":
        regions_filter = None
    else:
        regions_filter = [r.strip() for r in args.regions.split(",") if r.strip()]

    # discovery
    print("\n[DISCOVER] Scanning zones for A100 + a2-highgpu-4g …")
    candidates = discover_candidate_zones(prefer_europe_first=True, regions_filter=regions_filter)
    if not candidates:
        print("No candidate zones found (a2-highgpu-4g + A100). Try --regions all.")
        sys.exit(2)

    print("[CANDIDATES]")
    for z, r, q in candidates:
        print(f"  {z:28}  region={r:18}  quota_remaining≈{q}")

    # Loop until success
    attempt = 0
    blacklist_quota_regions: set[str] = set()
    while True:
        attempt += 1
        print(f"\n===== ATTEMPT {attempt} =====")
        # Shuffle within priority bands to avoid hammering the same zone
        random.shuffle(candidates)

        for zone, rgn, qrem in candidates:
            if rgn in blacklist_quota_regions:
                continue
            ok, reason = try_create_once(
                project, zone, args.name, args.image, args.spot, args.boot_size, args.disk_type, args.scopes
            )
            if ok:
                print(f"\n✅ Success: instance '{args.name}' created in zone {zone}")
                print(f"ssh command:\n  gcloud compute ssh {args.name} --project={project} --zone={zone}")
                return
            else:
                if reason == "quota":
                    print(f"[INFO] Quota issue in {rgn}, temporarily skipping region.")
                    blacklist_quota_regions.add(rgn)
                elif reason == "name_taken":
                    # auto-bump name and retry immediately
                    new_name = f"{args.name}-{datetime.utcnow().strftime('%H%M%S')}"
                    print(f"[INFO] Instance name taken; switching to {new_name}")
                    args.name = new_name
                elif reason == "image":
                    print("[ERROR] Image not found. Check --image value and permissions.")
                    sys.exit(3)
                else:
                    print(f"[INFO] Capacity not available in {zone}, will try next.")

        # Backoff before next cycle
        sleep_s = random.randint(args.backoff_min, args.backoff_max)
        print(f"[WAIT] Sleeping {sleep_s}s before next round …")
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()

