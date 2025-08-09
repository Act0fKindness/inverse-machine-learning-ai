#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Build per-word labels.json and Uni-Sign master CSV.")
    ap.add_argument(
        "--videos-root",
        default=str(Path.home() / "projects/dev/inverse/research_videos/BSL_Videos"),
        help="Root of video folders (each subfolder is a word).",
    )
    ap.add_argument(
        "--labels-root",
        default=str(Path.home() / "projects/dev/inverse/research_videos/BSL_Labels"),
        help="Root to write label folders (mirrors videos-root).",
    )
    ap.add_argument(
        "--pose-root",
        default=str(Path.home() / "projects/dev/inverse/research_videos/BSL_Pose"),
        help="Root folder where pose PKLs will be stored (can be empty for now).",
    )
    ap.add_argument(
        "--exts",
        default=".mp4,.mkv,.mov",
        help="Comma-separated video extensions to include.",
    )
    args = ap.parse_args()

    videos_root = Path(args.videos_root).resolve()
    labels_root = Path(args.labels_root).resolve()
    pose_root   = Path(args.pose_root).resolve()
    exts = tuple(e.strip().lower() for e in args.exts.split(","))

    if not videos_root.exists():
        raise SystemExit(f"Videos root not found: {videos_root}")

    labels_root.mkdir(parents=True, exist_ok=True)

    all_rows = []  # for Uni-Sign CSV
    total_words = total_videos = 0

    for word_dir in sorted([d for d in videos_root.iterdir() if d.is_dir()]):
        word = word_dir.name
        vids = sorted([p for p in word_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in exts])

        items = []
        for v in vids:
            pose_file = pose_root / word / (v.stem + ".pkl")
            items.append({
                "word": word,
                "video": v.name,
                "video_rel_path": str(v.relative_to(videos_root.parent)),
                "label_text": word,
                "source": "BSL_Videos"
            })
            all_rows.append([
                str(v.relative_to(videos_root.parent)),               # video path (relative)
                str(pose_file.relative_to(videos_root.parent)) if pose_file.exists() else "",  # pose path or empty
                word,                                                  # text label
                "train"                                                # split
            ])

        # Write per-word labels.json
        out_dir = labels_root / word
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "labels.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}  ({len(items)} items)")
        total_words += 1
        total_videos += len(items)

    # Write Uni-Sign CSV
    csv_path = labels_root / "BSL_labels_unisign.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "pose", "text", "split"])
        w.writerows(all_rows)

    print(f"\n✅ Wrote Uni-Sign master CSV: {csv_path} ({len(all_rows)} rows)")
    missing_pose = sum(1 for r in all_rows if not r[1])
    if missing_pose:
        print(f"ℹ️  {missing_pose} rows have empty 'pose' (extract poses to fill).")

    print(f"\nDone. Words: {total_words}, Videos indexed: {total_videos}")
    print(f"Labels root: {labels_root}")

if __name__ == "__main__":
    main()

