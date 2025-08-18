#!/usr/bin/env python3
"""
launch_a100_until_ready.py

Flow:
  1) Try SSH to your CURRENT instance (defaults to inverse-a100-4g-preemptible-europe-west4-b).
     - If instance exists but is STOPPED, start it, wait for RUNNING, then SSH-test again.
     - If SSH works, we’re done.
  2) Otherwise, discover candidate zones with A100 + a2-highgpu-4g and loop until we CREATE one.
     - Try STANDARD then SPOT (flip with --prefer-spot).
     - Verify instance becomes RUNNING and SSH works before exiting.

Defaults:
  project: inverse-machine-learning-ai-v2
  current-name: inverse-a100-4g-preemptible-europe-west4-b
  image: inverse-pose-extraction-470000
  regions search: europe (use --regions all to widen)

Run:
  python3 launch_a100_until_ready.py
"""

import argparse
import json
import random
import string
import subprocess
import sys
import time
from datetime import datetime

# ===== Defaults you asked for =====
PROJECT_DEFAULT = "inverse-machine-learning-ai-v2"
CURRENT_NAME_DEFAULT = "inverse-a100-4g-preemptible-europe-west4-b"
IMAGE_DEFAULT = "inverse-pose-extraction-470000"

# ===== A100 x4 a2 machine =====
MACHINE_TYPE = "a2-highgpu-4g"
ACCELERATOR = "nvidia-tesla-a100"
ACCEL_COUNT = 4

# ===== Error patterns =====
CAPACITY_ERRORS = (
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
    "RESOURCE_POOL_EXHAUSTED",
    "does not have enough resources",
    "out of resources",
    "is unavailable in the zone",
)
QUOTA_ERRORS = (
    "QUOTA_EXCEEDED",
    "exceeded quota",
    "Insufficient regional quota",
)

def run(cmd: str):
    """Run a shell command, return (rc, stdout+stderr)."""
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode(errors="replace")

def jrun(cmd: str):
    rc, out = run(cmd + " --format=json")
    if rc != 0:
        raise RuntimeError(out)
    return json.loads(out or "[]")

def get_project(override: str | None):
    if override:
        return override
    rc, out = run("gcloud config get-value project")
    proj = out.strip()
    return proj if proj and "unset" not in proj.lower() else PROJECT_DEFAULT

# ---------- Current instance helpers ----------
def find_instance(name: str, project: str):
    """Find instance by name across all zones. Return dict with name, zone, status or None."""
    try:
        items = jrun(f"gcloud compute instances list --project={project} --filter='name={name}'")
    except Exception:
        items = []
    if not items:
        return None
    inst = items[0]
    zone_url = inst.get("zone","")
    zone = zone_url.split("/")[-1] if zone_url else ""
    status = inst.get("status","")
    return {"name": inst.get("name"), "zone": zone, "status": status}

def instance_status(name: str, zone: str, project: str):
    rc, out = run(
        f"gcloud compute instances describe {name} --project={project} --zone={zone} --format='get(status)'"
    )
    return out.strip() if rc == 0 else ""

def start_instance(name: str, zone: str, project: str):
    rc, out = run(f"gcloud compute instances start {name} --project={project} --zone={zone} --quiet")
    return rc == 0, out

def wait_until_state(name: str, zone: str, project: str, target="RUNNING", timeout=900, poll=8):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = instance_status(name, zone, project)
        if st == target:
            return True
        time.sleep(poll)
    return False

def ssh_test(name: str, zone: str, project: str, user: str | None = None, timeout=10):
    """
    Attempt a fast, non-interactive ssh check.
    - uses StrictHostKeyChecking=no and short ConnectTimeout.
    """
    user_prefix = f"{user}@" if user else ""
    rc, out = run(
        f"gcloud compute ssh {user_prefix}{name} "
        f"--project={project} --zone={zone} "
        f"--ssh-flag='-o StrictHostKeyChecking=no' "
        f"--ssh-flag='-o ConnectTimeout={timeout}' "
        f"--command='echo ok' --quiet"
    )
    return rc == 0

# ---------- Discovery for new capacity ----------
def list_up_zones(regions_filter: list[str] | None):
    zones = jrun("gcloud compute zones list")
    zones = [z for z in zones if z.get("status") == "UP"]
    if regions_filter:
        zones = [z for z in zones if any(z["name"].startswith(r + "-") for r in regions_filter)]
    return zones

def zone_supports_a2(zone: str) -> bool:
    try:
        mts = jrun(f"gcloud compute machine-types list --zones={zone} --filter='name={MACHINE_TYPE}'")
        return len(mts) > 0
    except Exception:
        return False

def zone_has_a100(zone: str) -> bool:
    try:
        accs = jrun(f"gcloud compute accelerator-types list --zones={zone}")
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
        info = jrun(f"gcloud compute regions describe {region}")
    except Exception:
        return 0.0
    qmap = {q.get("metric"): q for q in info.get("quotas", [])}
    for key in ["NVIDIA_A100_GPUS", "A100_GPUS", "NVIDIA_A100_40GB_GPUS", "GPUS"]:
        if key in qmap:
            q = qmap[key]
            return max(0.0, float(q.get("limit", 0)) - float(q.get("usage", 0)))
    for m, q in qmap.items():
        if "GPU" in m:
            return max(0.0, float(q.get("limit", 0)) - float(q.get("usage", 0)))
    return 0.0

def discover_candidates(prefer_eu=True, regions_filter=None):
    zones = list_up_zones(regions_filter)
    can = []
    for z in zones:
        zone = z["name"]
        if zone_supports_a2(zone) and zone_has_a100(zone):
            rgn = region_from_zone(zone)
            qrem = region_quota_remaining(rgn)
            can.append((zone, rgn, qrem))
    if prefer_eu:
        can.sort(key=lambda t: (0 if t[0].startswith("europe-") else 1, -t[2], t[0]))
    else:
        can.sort(key=lambda t: (-t[2], t[0]))
    return can

# ---------- Create new instance ----------
def build_create_cmd(project, zone, name, image, provisioning_model, boot_size_gb, disk_type, scopes, sa_email, tags, labels, subnet):
    parts = [
        f"gcloud compute instances create {name}",
        f"--project={project}",
        f"--zone={zone}",
        f"--provisioning-model={provisioning_model}",  # STANDARD or SPOT
        f"--machine-type={MACHINE_TYPE}",
        "--maintenance-policy=TERMINATE",
        f"--accelerator=count={ACCEL_COUNT},type={ACCELERATOR}",
        f"--image={image}",
        f"--boot-disk-size={boot_size_gb}",
        f"--boot-disk-type={disk_type}",
        '--metadata="install-nvidia-driver=True"',
        f"--scopes={scopes}",
    ]
    if sa_email:
        parts.append(f"--service-account={sa_email}")
    if tags:
        parts.append(f"--tags={tags}")
    if labels:
        parts.append(f"--labels={labels}")
    if subnet:
        parts.append(f"--subnet={subnet}")
    return " \\\n  ".join(parts)

def instance_exists(project, zone, name):
    rc, out = run(f"gcloud compute instances describe {name} --project={project} --zone={zone} --format='get(name)'")
    return rc == 0 and out.strip() == name

def gcloud_create(project, zone, name, image, provisioning_model, boot_size, disk_type, scopes, sa_email, tags, labels, subnet):
    cmd = build_create_cmd(project, zone, name, image, provisioning_model, boot_size, disk_type, scopes, sa_email, tags, labels, subnet)
    print(f"\n[TRY] {provisioning_model} in {zone}: {name}")
    rc, out = run(cmd)
    print(out.strip()[:2000])
    lower = out.lower()

    if any(s.lower() in lower for s in CAPACITY_ERRORS):
        return False, "capacity"
    if any(s.lower() in lower for s in QUOTA_ERRORS):
        return False, "quota"
    if "already exists" in lower:
        return False, "name_taken"
    if "not found" in lower and "image" in lower:
        return False, "image"

    ok = instance_exists(project, zone, name)
    return ok, "ok" if ok else ("unknown" if rc == 0 else "error")

def wait_until_running(name: str, zone: str, project: str, timeout=900, poll=8):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out = run(f"gcloud compute instances describe {name} --project={project} --zone={zone} --format='get(status)'")
        if rc == 0 and out.strip() == "RUNNING":
            return True
        time.sleep(poll)
    return False

def rand_suffix(k=4):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))

# ---------- Orchestration ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--current-name", default=CURRENT_NAME_DEFAULT, help="Existing instance to try first")
    ap.add_argument("--current-zone", default="", help="Zone of current instance (auto-detect if empty)")
    ap.add_argument("--image", default=IMAGE_DEFAULT)
    ap.add_argument("--regions", default="europe", help="Comma list (europe,us,asia) or 'all'")
    ap.add_argument("--prefer-spot", dest="prefer_spot", action="store_true", help="Try SPOT first, then STANDARD")
    ap.add_argument("--boot-size", default="200GB")
    ap.add_argument("--disk-type", default="pd-ssd", choices=["pd-ssd", "pd-balanced", "pd-extreme"])
    ap.add_argument("--scopes", default="compute-rw,storage-rw")
    ap.add_argument("--service-account", default="", help="Optional service account email")
    ap.add_argument("--tags", default="", help="Optional network tags, comma-separated")
    ap.add_argument("--labels", default="role=pose,env=prod", help="Optional labels k=v,k2=v2")
    ap.add_argument("--subnet", default="", help="Optional VPC subnet name")
    ap.add_argument("--backoff-min", type=int, default=20)
    ap.add_argument("--backoff-max", type=int, default=120)
    ap.add_argument("--refresh-every", type=int, default=5, help="Rediscover zones every N attempts")
    ap.add_argument("--running-timeout", type=int, default=900, help="Seconds to wait for RUNNING")
    ap.add_argument("--ssh-user", default="", help="Optional SSH user (usually leave blank)")
    ap.add_argument("--ssh-timeout", type=int, default=10, help="SSH connect timeout seconds")
    args = ap.parse_args()

    project = get_project(args.project)
    print(f"# Project: {project}")
    print(f"# Current: {args.current-name if hasattr(args, 'current-name') else args.current_name} (zone: {args.current_zone or 'auto'})")
    print(f"# Image:   {args.image}")
    print(f"# Target:  {MACHINE_TYPE} (A100 x{ACCEL_COUNT})")
    print(f"# Policy:  {'SPOT→STANDARD' if args.prefer_spot else 'STANDARD→SPOT'}")
    print("# Regions: " + args.regions)

    # ---- Step 1: Try current instance first ----
    current = find_instance(args.current_name, project)
    if current:
        name = current["name"]
        zone = current["zone"] if not args.current_zone else args.current_zone
        if not zone:
            print("[WARN] Could not determine zone for current instance.")
        else:
            print(f"[CURRENT] Found {name} in {zone} with status {current['status']}")
            status = current["status"]
            if status == "STOPPED" or status == "TERMINATED":
                print("[CURRENT] Instance is stopped; starting it …")
                ok, out = start_instance(name, zone, project)
                if not ok:
                    print("[CURRENT] Failed to start; will proceed to create new. Output:\n" + out[:1000])
                else:
                    if wait_until_state(name, zone, project, target="RUNNING", timeout=args.running_timeout, poll=8):
                        print("[CURRENT] Instance is RUNNING; trying SSH …")
                        if ssh_test(name, zone, project, user=args.ssh_user or None, timeout=args.ssh_timeout):
                            print(f"✅ SSH OK: gcloud compute ssh {name} --project={project} --zone={zone}")
                            return
                        else:
                            print("[CURRENT] SSH failed; proceeding to create new.")
                    else:
                        print("[CURRENT] Did not reach RUNNING in time; proceeding to create new.")
            elif status == "RUNNING":
                print("[CURRENT] Instance already RUNNING; trying SSH …")
                if ssh_test(name, zone, project, user=args.ssh_user or None, timeout=args.ssh_timeout):
                    print(f"✅ SSH OK: gcloud compute ssh {name} --project={project} --zone={zone}")
                    return
                else:
                    print("[CURRENT] SSH failed; proceeding to create new.")
            else:
                print(f"[CURRENT] Status is {status}; proceeding to create new.")
    else:
        print("[CURRENT] No instance found with that name; proceeding to create new.")

    # ---- Step 2: Create new instance (loop until success) ----
    regions_filter = None if args.regions.strip().lower() == "all" else [
        r.strip() for r in args.regions.split(",") if r.strip()
    ]
    prefer_models = ["SPOT", "STANDARD"] if args.prefer_spot else ["STANDARD", "SPOT"]

    def refresh_candidates():
        print("\n[DISCOVER] Refreshing candidate zones …")
        cand = discover_candidates(prefer_eu=True, regions_filter=regions_filter)
        if not cand:
            print("  No candidates right now. Consider --regions all.")
        else:
            for z, r, q in cand:
                print(f"  {z:28} region={r:18} quota≈{q}")
        return cand

    candidates = refresh_candidates() or []
    attempt = 0
    blacklist_quota_regions: set[str] = set()
    new_name = f"inverse-a100-4g-{int(time.time())}"

    while True:
        attempt += 1
        print(f"\n===== ATTEMPT {attempt} =====")
        if attempt == 1 or (attempt % args.refresh_every == 0) or not candidates:
            candidates = refresh_candidates() or []

        cands = [(z, r, q) for (z, r, q) in candidates if r not in blacklist_quota_regions]
        random.shuffle(cands)
        if not cands:
            blacklist_quota_regions.clear()
            cands = candidates[:]
            random.shuffle(cands)

        for zone, rgn, _ in cands:
            for model in prefer_models:
                ok, reason = gcloud_create(
                    project, zone, new_name, args.image, model,
                    args.boot_size, args.disk_type, args.scopes,
                    args.service_account, args.tags, args.labels, args.subnet
                )
                if ok or instance_exists(project, zone, new_name):
                    print(f"\n✅ Created: {new_name} in {zone} ({model}) — waiting RUNNING …")
                    if wait_until_running(new_name, zone, project, timeout=args.running_timeout, poll=8):
                        print("[NEW] Trying SSH …")
                        if ssh_test(new_name, zone, project, user=args.ssh_user or None, timeout=args.ssh_timeout):
                            print(f"✅ SSH OK: gcloud compute ssh {new_name} --project={project} --zone={zone}")
                            return
                        else:
                            print("[NEW] SSH failed (network/keys). You can still try:")
                            print(f"  gcloud compute ssh {new_name} --project={project} --zone={zone}")
                            return
                    else:
                        print("⚠️ Created but not RUNNING within timeout. You can check manually:")
                        print(f"  gcloud compute instances describe {new_name} --project={project} --zone={zone}")
                        print(f"  gcloud compute ssh {new_name} --project={project} --zone={zone}")
                        return
                else:
                    if reason == "quota":
                        print(f"[INFO] Quota blocked in region {rgn}; blacklisting region this round.")
                        blacklist_quota_regions.add(rgn)
                        break
                    elif reason == "name_taken":
                        ts = datetime.utcnow().strftime("%H%M%S")
                        new_name = f"{new_name}-{ts}-{rand_suffix(3)}"
                        print(f"[INFO] Name taken; switching to: {new_name}")
                    elif reason == "image":
                        print("[ERROR] Image not found/accessible. Check --image.")
                        sys.exit(3)
                    else:
                        print(f"[INFO] {model} capacity not available in {zone}. Trying next …")

        # backoff between rounds
        sleep_s = random.randint( max(5, args.backoff_min), max(args.backoff_min, args.backoff_max) )
        print(f"[WAIT] Sleeping {sleep_s}s before next search/attempt …")
        time.sleep(sleep_s)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")

