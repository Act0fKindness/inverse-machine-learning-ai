#!/usr/bin/env python3
import argparse, os, glob, pickle, re
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# fixed import: pull Wholebody & draw_skeleton from the nested package
from rtmlib.rtmlib.tools.solution.wholebody import Wholebody
from rtmlib.rtmlib.visualization.draw import draw_skeleton


def natkey(s: str):
    """Natural sort key: '..._2.mp4' comes before '..._10.mp4'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, (W, H)


def process_video(video_path: str, tgt_dir: str, wholebody, max_workers=32, overwrite=False):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(tgt_dir, f"{stem}.pkl")

    if os.path.exists(output_path) and not overwrite:
        print(f"skip (exists): {output_path}")
        return True, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"fail to open: {video_path}"

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return False, f"no frames: {video_path}"

    data = {"keypoints": [], "scores": []}

    # Per-frame parallelism (CPU threads driving the model). Tune for your box.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_frame, f, wholebody) for f in frames]
        for fut in tqdm(futures, desc=f"Frames {os.path.basename(video_path)}", total=len(frames), leave=False):
            kpts, scores, (W, H) = fut.result()
            # normalize by width/height
            data["keypoints"].append(kpts / np.array([W, H])[None, None])
            data["scores"].append(scores)

    os.makedirs(tgt_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return True, None


def build_file_list(src_dir: str, exts, recursive: bool):
    src = Path(src_dir)
    files = []
    if recursive:
        for e in exts:
            files.extend([str(p) for p in src.rglob(f"*.{e}")])
    else:
        for e in exts:
            files.extend([str(p) for p in src.glob(f"*.{e}")])
    files = sorted(files, key=natkey)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="video dir path")
    ap.add_argument("--tgt_dir", required=True, help="pose dir path")

    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    ap.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    ap.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"])

    ap.add_argument("--video_extensions", nargs="+", default=["mp4"])
    ap.add_argument("--recursive", action="store_true", help="include subfolders (e.g. archive_*)")
    ap.add_argument("--start", type=int, default=0, help="start index in discovered file list (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end index in discovered file list (exclusive)")
    ap.add_argument("--max_workers", type=int, default=16, help="per‑video frame workers")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing .pkl")

    args = ap.parse_args()

    # init model
    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    all_files = build_file_list(args.src_dir, args.video_extensions, args.recursive)
    n_total = len(all_files)
    start = max(0, int(args.start))
    end = n_total if args.end is None else min(n_total, int(args.end))
    files = all_files[start:end]

    print(f"discovered {n_total} videos; processing range [{start}, {end}) → {len(files)} files")

    failures = []
    for vp in tqdm(files, desc="Processing videos"):
        ok, err = process_video(
            video_path=vp,
            tgt_dir=args.tgt_dir,
            wholebody=wholebody,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
        )
        if not ok and err:
            print(err)
            failures.append(err)

    if failures:
        log = os.path.join(args.tgt_dir, "failures.txt")
        with open(log, "a", encoding="utf-8") as f:
            for line in failures:
                f.write(line + "\n")
        print(f"⚠️  {len(failures)} failures logged to {log}")
    else:
        print("✅ done with no failures")


if __name__ == "__main__":
    main()

