#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def rename_in_folder(folder: Path, apply: bool):
    if not folder.is_dir():
        return 0, 0, 0
    prefix = folder.name
    changed = skipped = conflicts = 0

    for p in sorted(folder.glob("*.mp4")):
        # Skip if already prefixed correctly
        if p.name.startswith(f"{prefix}_"):
            skipped += 1
            continue

        target = p.with_name(f"{prefix}_{p.name}")

        # Handle conflict: if target exists, don't overwrite
        if target.exists():
            print(f"[conflict] {target} already exists; skipping {p}")
            conflicts += 1
            continue

        print(f"{'[DRY-RUN] ' if not apply else ''}rename: {p} -> {target}")
        if apply:
            os.rename(p, target)
        changed += 1

    return changed, skipped, conflicts

def main():
    ap = argparse.ArgumentParser(
        description="Prefix each .mp4 with its parent folder name (word_001.mp4)."
    )
    ap.add_argument(
        "--root",
        default=str(Path.home() / "projects/dev/inverse/research_videos/BSL_Videos"),
        help="Root directory that contains word folders (default: %(default)s)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform renames (default is dry-run).",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="How many directory levels under root to process (default: 1)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    total_changed = total_skipped = total_conflicts = 0

    # Process immediate subfolders by default; increase depth if needed
    folders = []
    if args.depth == 0:
        folders = [root]
    elif args.depth == 1:
        folders = [d for d in root.iterdir() if d.is_dir()]
    else:
        # Recurse up to depth
        for d in root.rglob("*"):
            if d.is_dir() and len(d.relative_to(root).parts) <= args.depth:
                folders.append(d)

    for folder in sorted(folders):
        c, s, k = rename_in_folder(folder, apply=args.apply)
        total_changed += c
        total_skipped += s
        total_conflicts += k

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] Done. Renamed: {total_changed}, skipped(already ok): {total_skipped}, conflicts: {total_conflicts}")

if __name__ == "__main__":
    main()

