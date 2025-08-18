from pathlib import Path

# === Base directory for resolving relative paths ===
BASE_DIR = Path(__file__).resolve().parent

# === MT5 pretrained model path (absolute) ===
mt5_path = str((BASE_DIR / "pretrained_weight" / "mt5-base").resolve())

# === Label paths ===
train_label_paths = {
    'CSL_News': str((BASE_DIR / 'data' / 'CSL_News' / 'CSL_News_Labels.json').resolve()),
    'WLBSL': str((BASE_DIR / 'data' / 'WLBSL' / 'WLBSL_Labels.json').resolve())
}

dev_label_paths = {
    'CSL_News': str((BASE_DIR / 'data' / 'CSL_News' / 'CSL_News_Labels.json').resolve()),
    'WLBSL': str((BASE_DIR / 'data' / 'WLBSL' / 'WLBSL_Labels.json').resolve())
}

test_label_paths = {
    'CSL_News': str((BASE_DIR / 'data' / 'CSL_News' / 'CSL_News_Labels.json').resolve()),
    'WLBSL': str((BASE_DIR / 'data' / 'WLBSL' / 'WLBSL_Labels.json').resolve())
}

# === Video paths ===
rgb_dirs = {
    'CSL_News': str((BASE_DIR / 'data' / 'CSL_News' / 'rgb_format').resolve()),
    'WLBSL': str((BASE_DIR / 'data' / 'WLBSL' / 'rgb_format').resolve())
}

# === Pose paths ===
pose_dirs = {
    'CSL_News': str((BASE_DIR / 'data' / 'CSL_News' / 'pose_format').resolve()),
    'WLBSL': str((BASE_DIR / 'data' / 'WLBSL' / 'pose_format').resolve())
}

