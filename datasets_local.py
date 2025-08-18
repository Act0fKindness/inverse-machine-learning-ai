import os
import csv
import json
import copy
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms

import utils as utils
from config import rgb_dirs, pose_dirs


# -----------------------------
# Helpers for skip list (CSV)
# -----------------------------
def _pick_col(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _derive_pose_basename_from_row(row, pose_col, video_col):
    if pose_col and row.get(pose_col, "").strip():
        name = Path(row[pose_col].strip()).name
        return name if name.endswith(".pkl") else f"{name}.pkl"
    if row.get("expected_pose_path", "").strip():
        return Path(row["expected_pose_path"].strip()).name
    if video_col and row.get(video_col, "").strip():
        v = Path(row[video_col].strip())
        return v.with_suffix(".pkl").name
    return None

def _load_skip_set_for_labels_json_path(labels_json_path: str) -> set:
    """
    Look for skip_pose.csv next to the labels json (or SKIP_POSE_CSV env var).
    Return a set of pose basenames to skip.
    """
    labels_path = Path(labels_json_path)
    csv_path = Path(os.environ.get("SKIP_POSE_CSV", str(labels_path.parent / "skip_pose.csv")))
    if not csv_path.exists():
        return set()
    skip = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        pose_col  = _pick_col(cols, "pose", "pose_name", "pose_path")
        video_col = _pick_col(cols, "video", "video_name", "video_path", "rgb", "rgb_path")
        for row in reader:
            name = _derive_pose_basename_from_row(row, pose_col, video_col)
            if name:
                skip.add(name)
    return skip


# -----------------------------
# Pose/keypoint processing
# -----------------------------
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]
            else:
                raise NotImplementedError

            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        if part == 'body':
            result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
        else:
            assert scale is not None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= thr] = 0

        kps_with_scores[part] = torch.tensor(result)

    return kps_with_scores


def crop_scale(motion, thr):
    """
    motion: [(M), T, 17, 3] -> normalize to [-1, 1]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0]); xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1]); ymax = max(valid_coords[:, 1])
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def bbox_4hands(left_keypoints, right_keypoints, hw):
    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        return (max_x + min_x) / 2, (max_y + min_y) / 2, (max_x - min_x), (max_y - min_y)

    H, W = hw
    if left_keypoints is None:
        left_keypoints = np.zeros([1, 21, 2])
    if right_keypoints is None:
        right_keypoints = np.zeros([1, 21, 2])

    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W * left_mean_x; left_mean_y = H * left_mean_y
    left_diff_x = W * left_diff_x; left_diff_y = H * left_diff_y
    left_diff_x = max(left_diff_x); left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x, left_diff_y)

    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W * right_mean_x; right_mean_y = H * right_mean_y
    right_diff_x = W * right_diff_x; right_diff_y = H * right_diff_y
    right_diff_x = max(right_diff_x); right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x, right_diff_y)

    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw / 2, left_mean_y - box_hw / 2, left_mean_x + box_hw / 2, left_mean_y + box_hw / 2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw / 2, right_mean_y - box_hw / 2, right_mean_x + box_hw / 2, right_mean_y + box_hw / 2]).astype(np.int16)

    return left_new_box.transpose(1, 0), right_new_box.transpose(1, 0), box_hw


def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform):
    support_rgb_dict = {}

    confs = np.array(confs)
    skeletons = np.array(skeletons)

    left_confs_filter = confs[:, 0, 91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:
        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:, 0, [95, 99, 103, 107, 111]].min(-1)
        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)
        left_sample_size = int(np.ceil(0.1 * len(left_confs_filter_indices)))
        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(), size=left_sample_size, replace=False, p=left_probabilities)
        left_sampled_indices = np.sort(left_sampled_indices)
        left_skeletons = skeletons[left_sampled_indices, 0, 91:112]

    right_confs_filter = confs[:, 0, 112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None
    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:, 0, [95 + 21, 99 + 21, 103 + 21, 107 + 21, 111 + 21]].min(-1)
        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)
        right_sample_size = int(np.ceil(0.1 * len(right_confs_filter_indices)))
        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(), size=right_sample_size, replace=False, p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)
        right_skeletons = skeletons[right_sampled_indices, 0, 112:133]

    image_size = 112
    all_indices = []
    if left_sampled_indices is not None:
        all_indices.append(left_sampled_indices)
    if right_sampled_indices is not None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)
        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons, right_skeletons, imgs[0].shape[:2])

    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)
        return support_rgb_dict

    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)
    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)
        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0, 1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0, 1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)
    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)
        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0, 1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0, 1)

    left_idx = 0; right_idx = 0
    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if left_sampled_indices is not None and left_idx < len(left_sampled_indices) and mapping_idx == left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1

        if right_sampled_indices is not None and right_idx < len(right_sampled_indices) and mapping_idx == right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1

    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])
    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)
    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)
    return support_rgb_dict


def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    vr.seek(0)
    buffer = vr.get_batch(tmp).asnumpy()
    del vr
    return buffer


# -----------------------------
# Base dataset
# -----------------------------
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch, src_length_batch, name_batch, pose_tmp, gloss_batch = [], [], [], [], []

        for name_sample, pose_sample, text, gloss, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}
        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            padded_video = [
                torch.cat((vid[key], vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1)), dim=0)
                for vid in pose_tmp
            ]
            img_batch = torch.stack(padded_video, 0)
            src_input[key] = img_batch

            if 'attention_mask' not in src_input:
                src_length_batch = video_length
                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0, batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask
                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch

        if getattr(self, "rgb_support", False):
            support_rgb_dicts = {key: [] for key in batch[0][-1].keys()}
            for _, _, _, _, support_rgb_dict in batch:
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])

            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], 0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], 0)
                img_batch = torch.cat(support_rgb_dicts[rgb_key], 0)

                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        tgt_input = {'gt_sentence': tgt_batch, 'gt_gloss': gloss_batch}
        return src_input, tgt_input


# -----------------------------
# CSL_Daily / WLASL (legacy)
# -----------------------------
class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super().__init__()
        self.args = args
        self.rgb_support = bool(getattr(self.args, "rgb_support", False))
        self.max_length = getattr(args, "max_length", 256)
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]

        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        else:
            raise NotImplementedError

        self.list = list(self.raw_data.keys())

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        gloss = " ".join(sample['gloss']) if "gloss" in sample.keys() else ''
        name_sample = sample['name']

        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))

        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp) + start

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        kps_with_scores = load_part_kp(skeletons_tmp, confs_tmp, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = os.path.join(self.rgb_dir, path)
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons_tmp, confs_tmp, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'


# -----------------------------
# CSL_News (kept)
# -----------------------------
class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super().__init__()
        self.args = args
        self.rgb_support = bool(self.args.rgb_support)
        self.max_length = getattr(args, "max_length", 256)
        self.phase = phase

        p = Path(path)
        with p.open(encoding='utf-8') as f:
            annotation = json.load(f)

        if self.args.dataset != "CSL_News":
            raise NotImplementedError

        self.pose_dir = pose_dirs[args.dataset]
        self.rgb_dir = rgb_dirs[args.dataset]

        # transform
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # Load skip list and filter
        skip_set = _load_skip_set_for_labels_json_path(str(p))

        # Build samples for this phase (avoid slice indexing bugs)
        if isinstance(annotation, dict) and any(k in annotation for k in ("train", "dev", "test")):
            base_list = annotation.get(self.phase, [])
        else:
            # old flat list with 99/1 split
            cut = int(len(annotation) * 0.99)
            base_list = annotation[:cut] if self.phase == 'train' else annotation[cut:]

        if skip_set:
            def _pose_bn(s):
                if s.get('pose'):
                    n = Path(str(s['pose'])).name
                    return n if n.endswith('.pkl') else f"{n}.pkl"
                if s.get('video'):
                    return Path(str(s['video'])).with_suffix('.pkl').name
                return None
            before = len(base_list)
            base_list = [s for s in base_list if (_pose_bn(s) not in skip_set)]
            after = len(base_list)
            if before != after:
                print(f"[datasets_local] CSL_News filtered {before-after} via skip_pose.csv (remain {after}).")

        self.samples = list(base_list)

    def __len__(self):
        return len(self.samples)

    def _resolve_pose_path(self, pose_name, rgb_name):
        names = []
        if pose_name:
            bn = Path(str(pose_name)).name
            names.append(bn if bn.endswith('.pkl') else f"{bn}.pkl")
        if rgb_name:
            names.append(Path(str(rgb_name)).with_suffix('.pkl').name)

        roots = [self.pose_dir]
        extra = os.environ.get("EXTRA_POSE_DIRS", "")
        if extra:
            roots.extend([p for p in extra.split(':') if p])

        for r in roots:
            for n in names:
                cand = os.path.join(r, n)
                if os.path.exists(cand):
                    return cand
        raise FileNotFoundError(f"Pose not found for {pose_name} / {rgb_name} in roots {roots} (tried {names})")

    def __getitem__(self, index):
        if len(self.samples) == 0:
            raise RuntimeError("CSL_News split is empty after filtering/skip list.")

        num_retries = 10
        # normalize index in case a sampler passes numpy types
        index = int(index) % len(self.samples)

        for _ in range(num_retries):
            sample = self.samples[index]

            text = sample.get('text', '')
            gloss = ""  # CSL_News has no gloss field
            name_sample = sample['video']

            # expected pose basename
            if sample.get('pose'):
                pose_basename = Path(str(sample['pose'])).name
                if not pose_basename.endswith('.pkl'):
                    pose_basename = f"{pose_basename}.pkl"
            else:
                pose_basename = Path(str(sample['video'])).with_suffix('.pkl').name

            try:
                pose_path = self._resolve_pose_path(pose_basename, name_sample)
                with open(pose_path, 'rb') as f:
                    pose = pickle.load(f)

                # duration & sampling
                duration = len(pose['scores'])
                if duration > self.max_length:
                    tmp = sorted(random.sample(range(duration), k=self.max_length))
                else:
                    tmp = list(range(duration))
                tmp = np.array(tmp)

                skeletons = pose['keypoints']
                confs = pose['scores']
                sk_tmp, cf_tmp = [], []
                for idx in tmp:
                    sk_tmp.append(skeletons[idx])
                    cf_tmp.append(confs[idx])

                kps_with_scores = load_part_kp(sk_tmp, cf_tmp)

                support_rgb_dict = {}
                if self.rgb_support:
                    full_path = os.path.join(self.rgb_dir, name_sample)
                    support_rgb_dict = load_support_rgb_dict(tmp, sk_tmp, cf_tmp, full_path, self.data_transform)

                return name_sample, kps_with_scores, text, gloss, support_rgb_dict

            except FileNotFoundError:
                index = (index + 1) % len(self.samples)
                continue
            except Exception:
                import traceback
                traceback.print_exc()
                index = (index + 1) % len(self.samples)
                continue

        raise RuntimeError(f"Failed to fetch CSL_News sample after {num_retries} retries.")

    def __str__(self):
        return f'#total {len(self)}'

