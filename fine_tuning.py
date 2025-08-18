#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, time, json, pickle, argparse, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from timm.optim import create_optimizer
from transformers import get_scheduler

# Ensure local imports
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import Uni_Sign, get_requires_grad_dict
import utils as utils
from SLRT_metrics import translation_performance, islr_performance, wer_list
from config import *  # keeps your train_label_paths/dev/test, etc.


# ---------------------------
# Pose loading utilities
# ---------------------------

def _to_TJC(arr: np.ndarray) -> np.ndarray:
    """
    Convert various pose encodings to [T, J, C] where C in {2,3}.
    Accepts shapes like:
      [T, J, C] (already good),
      [T, D]    (flattened J*C),
      [T, J, C, 1] or [1, T, J, C].
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        # [T,J,C] typical
        T, J, C = a.shape
        if C not in (1, 2, 3):
            # Sometimes last dim is multiple subfeatures; flatten them into channels
            a = a.reshape(T, J, -1)
        return a.astype(np.float32)

    if a.ndim == 2:
        # [T, D] → try 3 then 2 channels
        T, D = a.shape
        if D % 3 == 0:
            J, C = D // 3, 3
        elif D % 2 == 0:
            J, C = D // 2, 2
        else:
            # Fallback: treat as 1 channel with J=D
            J, C = D, 1
        return a.reshape(T, J, C).astype(np.float32)

    if a.ndim == 4:
        # Common encodings like [T, J, C, 1] or [1, T, J, C]
        if a.shape[-1] == 1:
            a = a[..., 0]
            return _to_TJC(a)
        if a.shape[0] == 1:
            a = a[0]
            return _to_TJC(a)

    # Last resort: flatten into [T, J, C=1]
    if a.ndim >= 2:
        T = a.shape[0]
        rest = int(np.prod(a.shape[1:]))
        return a.reshape(T, rest, 1).astype(np.float32)

    raise ValueError(f"Unsupported pose array shape: {a.shape}")


def _split_single_pose_to_parts(tjc: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split a single [T, J, C] tensor into parts, assuming **contiguous ordering**:
      (body, left hand, right hand, face)
    Supports:
      - WholeBody (133): 23, 21, 21, 68
      - Compact   ( 69):  9, 21, 21, 18
    If J doesn't match exactly, we try to allocate the remainder to face.
    """
    T, J, C = tjc.shape
    schemes = [
        ("wholebody133", (23, 21, 21, 68)),
        ("compact69",    ( 9, 21, 21, 18)),
    ]

    # Exact matches first
    for _, (b, l, r, f) in schemes:
        if b + l + r + f == J:
            idx = 0
            body  = tjc[:, idx:idx + b, :]; idx += b
            left  = tjc[:, idx:idx + l, :]; idx += l
            right = tjc[:, idx:idx + r, :]; idx += r
            face  = tjc[:, idx:idx + f, :]
            return {"body": body, "left": left, "right": right, "face_all": face}

    # Fallback: common body+hands, remainder to face
    for b_try in (23, 9):
        l_try = r_try = 21
        if J >= b_try + l_try + r_try:
            idx = 0
            body  = tjc[:, idx:idx + b_try, :]; idx += b_try
            left  = tjc[:, idx:idx + l_try, :]; idx += l_try
            right = tjc[:, idx:idx + r_try, :]; idx += r_try
            face  = tjc[:, idx:, :]
            return {"body": body, "left": left, "right": right, "face_all": face}

    # If we get here, we couldn't split J cleanly
    raise RuntimeError(
        f"Cannot split J={J} joints into parts. "
        f"Expected 23+21+21+68 or 9+21+21+18; got {J}."
    )

def _load_pose_parts(pkl_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a pose PKL and return dict with keys: body, left, right, face_all.
    Each value is a float32 torch.Tensor of shape [T, V, C] with C in {2,3}.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # Case 1: already per-part dict
    if isinstance(obj, dict) and all(k in obj for k in ("body", "left", "right", "face_all")):
        out = {}
        for k in ("body", "left", "right", "face_all"):
            tjc = _to_TJC(np.asarray(obj[k]))
            # Pad to C=3 (x,y,score) if needed
            if tjc.shape[-1] == 2:
                pad = np.zeros((tjc.shape[0], tjc.shape[1], 1), dtype=tjc.dtype)
                tjc = np.concatenate([tjc, pad], axis=-1)
            out[k] = torch.tensor(tjc, dtype=torch.float32)
        return out

    # Case 2: single array under common keys
    if isinstance(obj, dict):
        for key in ("keypoints", "pose", "kpts", "skeleton", "data"):
            if key in obj:
                tjc = _to_TJC(np.asarray(obj[key]))
                parts = _split_single_pose_to_parts(tjc)
                out = {}
                for k, a in parts.items():
                    if a.shape[-1] == 2:
                        pad = np.zeros((a.shape[0], a.shape[1], 1), dtype=a.dtype)
                        a = np.concatenate([a, pad], axis=-1)
                    out[k] = torch.tensor(a, dtype=torch.float32)
                return out

        # Otherwise try any 1st array-like value
        for v in obj.values():
            try:
                tjc = _to_TJC(np.asarray(v))
                parts = _split_single_pose_to_parts(tjc)
                out = {}
                for k, a in parts.items():
                    if a.shape[-1] == 2:
                        pad = np.zeros((a.shape[0], a.shape[1], 1), dtype=a.dtype)
                        a = np.concatenate([a, pad], axis=-1)
                    out[k] = torch.tensor(a, dtype=torch.float32)
                return out
            except Exception:
                pass

        raise ValueError(f"Unsupported PKL dict structure in {pkl_path}")

    # Case 3: raw array
    tjc = _to_TJC(np.asarray(obj))
    parts = _split_single_pose_to_parts(tjc)
    out = {}
    for k, a in parts.items():
        if a.shape[-1] == 2:
            pad = np.zeros((a.shape[0], a.shape[1], 1), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=-1)
        out[k] = torch.tensor(a, dtype=torch.float32)
    return out


def _hash_bucket(s: str) -> int:
    return (abs(hash(s)) % 10_000) % 100  # 0..99


# ---------------------------
# WLBSL ISLR dataset
# ---------------------------

class WLBSL_ISLR_Dataset(Dataset):
    """
    Reads a single CSV (video,pose,text) and splits deterministically:
      train 0-89, dev 90-94, test 95-99 by hashing the video path.
    The 'pose' PKL can be either:
      - dict with parts: {body,left,right,face_all} each [T,V,C]
      - single array [T,J,C] or [T,J*C] which will be split.
    """
    def __init__(self, csv_path: str | Path, phase: str):
        self.phase = phase
        csv_path = Path(csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

        import csv
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            assert {"video", "pose", "text"}.issubset(r.fieldnames), \
                f"CSV must have video,pose,text (got {r.fieldnames})"
            for row in r:
                v = (row["video"] or "").strip()
                p = (row["pose"] or "").strip()
                t = (row["text"] or "").strip()
                if not v or not p or not t:
                    continue
                rows.append({"video": v, "pose": p, "text": t})

        items = []
        for row in rows:
            b = _hash_bucket(row["video"])
            if   phase == "train" and b < 90: items.append(row)
            elif phase == "dev"   and 90 <= b < 95: items.append(row)
            elif phase == "test"  and 95 <= b < 100: items.append(row)

        self.items = items
        if utils.is_main_process():
            print(f"[WLBSL_ISLR_Dataset] {phase} items: {len(self.items)} from {csv_path}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        parts = _load_pose_parts(it["pose"])  # dict of [T,V,C] float32
        # Use body length as canonical T if available, else max across parts
        Ts = {k: v.shape[0] for k, v in parts.items()}
        T = Ts.get("body", max(Ts.values()))
        src_input = {
            **parts,  # body,left,right,face_all
            "pose_len": torch.tensor(T).long(),
        }
        tgt_input = {"gt_sentence": it["text"]}
        return src_input, tgt_input

    @staticmethod
    def _pad_4d_list(x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad a list of [T,V,C] to [B, T_max, V, C] (pad T only, assume V,C consistent across batch).
        """
        T_max = max(x.shape[0] for x in x_list)
        V = x_list[0].shape[1]
        C = x_list[0].shape[2]
        B = len(x_list)
        out = x_list[0].new_zeros((B, T_max, V, C))
        for i, x in enumerate(x_list):
            t = x.shape[0]
            out[i, :t] = x
        return out

    @staticmethod
    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)
        poses = [b["pose"] for b in src_list]
        lens  = torch.stack([b["pose_len"] for b in src_list], dim=0)

        # Pad to [B, T, J, C]. Some pose sources may include extra feature
        # dimensions (e.g., confidence or depth channels).  Flatten any
        # additional dims beyond the joint dimension so downstream code
        # always receives a 4-D tensor.
        pad_pose = pad_sequence(poses, batch_first=True, padding_value=0.0)
        B, max_len = pad_pose.shape[:2]
        feat_shape = pad_pose.shape[2:]
        if len(feat_shape) == 1:
            J, C = feat_shape[0], 1
            pad_pose = pad_pose.unsqueeze(-1)
        else:
            J, C = feat_shape[0], int(np.prod(feat_shape[1:]))
            pad_pose = pad_pose.reshape(B, max_len, J, C)

        # Ensure last dimension represents (x, y, score).  Some pose sources
        # only provide a subset of these features or include additional ones.
        # Pad with zeros when fewer than 3 channels are present and truncate
        # extras when more exist so downstream linear layers always receive
        # exactly three inputs per joint.
        if C < 3:
            pad_pose = torch.nn.functional.pad(pad_pose, (0, 3 - C))
            C = 3
        elif C > 3:
            pad_pose = pad_pose[..., :3]
            C = 3

        # Split joints into body/hand/face parts assuming fixed ordering
        idx = 0
        parts = {}
        for name, n in [("body", 9), ("left", 21), ("right", 21), ("face_all", 18)]:
            parts[name] = pad_pose[:, :, idx:idx + n, :]
            idx += n

        # Attention mask for valid timesteps
        mask = torch.arange(max_len).expand(B, max_len) < lens.unsqueeze(1)

        src_input = {
            **parts,
            "attention_mask": mask.long(),
        }
        tgt_input = {"gt_sentence": [b["gt_sentence"] for _, b in batch]}
        return src_input, tgt_input


# ---------------------------
# Train / Eval
# ---------------------------

def build_wlbls_loaders(args):
    train_data = WLBSL_ISLR_Dataset(args.labels, phase='train')
    dev_data   = WLBSL_ISLR_Dataset(args.labels, phase='dev')
    test_data  = WLBSL_ISLR_Dataset(args.labels, phase='test')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        dev_sampler   = torch.utils.data.SequentialSampler(dev_data)
        test_sampler  = torch.utils.data.SequentialSampler(test_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        dev_sampler   = torch.utils.data.SequentialSampler(dev_data)
        test_sampler  = torch.utils.data.SequentialSampler(test_data)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, pin_memory=args.pin_mem,
        collate_fn=WLBSL_ISLR_Dataset.collate_fn, drop_last=True
    )
    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=dev_sampler, pin_memory=args.pin_mem,
        collate_fn=WLBSL_ISLR_Dataset.collate_fn
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=test_sampler, pin_memory=args.pin_mem,
        collate_fn=WLBSL_ISLR_Dataset.collate_fn
    )
    return train_loader, dev_loader, test_loader


def _maybe_cast_and_move(src_input: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]):
    """Move tensors to CUDA and cast only floating tensors to the target dtype."""
    for k in list(src_input.keys()):
        v = src_input[k]
        if isinstance(v, torch.Tensor):
            v = v.cuda(non_blocking=True)
            if dtype is not None and torch.is_floating_point(v):
                v = v.to(dtype)
            src_input[k] = v
    return src_input

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header, print_freq = f'Epoch: [{epoch}/{args.epochs}]', 10
    optimizer.zero_grad()

    # choose input dtype based on args.dtype to match DS model weights
    dtype_flag = str(getattr(args, 'dtype', '')).lower()
    if dtype_flag in ('bf16', 'bfloat16'):
        target_dtype = torch.bfloat16
    elif dtype_flag in ('fp16', 'float16', 'half'):
        target_dtype = torch.float16
    else:
        target_dtype = None  # fp32 inputs

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        src_input = _maybe_cast_and_move(src_input, target_dtype)

        if getattr(args, "probe_shapes", False) and step == 0 and utils.is_main_process():
            print("== Incoming src_input shapes ==")
            for k, v in src_input.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)} {v.dtype} {v.device}", flush=True)
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit(0)

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        if getattr(args, "probe_forward", False) and step == 0:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit(0)

        model.backward(total_loss)
        model.step()

        loss_value = float(total_loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # match input dtype to DS model weights
    dtype_flag = str(getattr(args, 'dtype', '')).lower()
    if dtype_flag in ('bf16', 'bfloat16'):
        target_dtype = torch.bfloat16
    elif dtype_flag in ('fp16', 'float16', 'half'):
        target_dtype = torch.float16
    else:
        target_dtype = None

    with torch.no_grad():
        tgt_pres, tgt_refs = [], []
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, f'Eval[{phase}]:')):
            src_input = _maybe_cast_and_move(src_input, target_dtype)

            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=float(total_loss.item()))

            output = model_without_ddp.generate(stack_out, max_new_tokens=50, num_beams=1)
            if isinstance(output, torch.Tensor):
                output = output.cpu().tolist()
            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
    metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
    metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}

def main(args):
    utils.init_distributed_mode_ds(args)
    print(args)
    utils.set_seed(args.seed)

    # Build WLBSL loaders from a single CSV
    train_loader, dev_loader, test_loader = build_wlbls_loaders(args)
    print(train_loader); print(dev_loader); print(test_loader)

    print("Creating model:")
    model = Uni_Sign(args=args).cuda().train()
    for _, p in model.named_parameters():
        if p.requires_grad: p.data = p.data.to(torch.float32)

    if args.finetune:
        print('***********************************\nLoad Checkpoint...\n***********************************')
        state = torch.load(args.finetune, map_location='cpu')
        state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys:\n', '\n'.join(ret.missing_keys))
        print('Unexpected keys:\n', '\n'.join(ret.unexpected_keys))

    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_loader) / max(1, args.gradient_accumulation_steps)),
        num_training_steps=int(args.epochs * len(train_loader) / max(1, args.gradient_accumulation_steps)),
    )

    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    if args.distributed:
        model_without_ddp = model.module.module
    else:
        model_without_ddp = model.module
    print(optimizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    max_acc = 0.0

    try:
        if args.eval:
            if utils.is_main_process():
                print("📄 dev result");  evaluate(args, dev_loader,  model, model_without_ddp, 'dev')
                print("📄 test result"); evaluate(args, test_loader, model, model_without_ddp, 'test')
            return

        print(f"Start training for {args.epochs} epochs")
        for epoch in range(args.epochs):
            if args.distributed:
                s = getattr(train_loader, "sampler", None)
                if hasattr(s, "set_epoch"): s.set_epoch(epoch)

            train_stats = train_one_epoch(args, model, train_loader, optimizer, epoch)

            # Save checkpoint
            ckpt_path = output_dir / f'checkpoint_{epoch}.pth'
            utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, ckpt_path)

            # Eval
            if utils.is_main_process():
                dev_stats  = evaluate(args, dev_loader,  model, model_without_ddp, 'dev')
                test_stats = evaluate(args, test_loader, model, model_without_ddp, 'test')

                if max_acc < dev_stats.get("top1_acc_pi", 0.0):
                    max_acc = dev_stats["top1_acc_pi"]
                    best_p = output_dir / 'best_checkpoint.pth'
                    utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, best_p)

                print(f"PI accuracy (dev): {dev_stats.get('top1_acc_pi', 0.0):.2f}")
                print(f"Max PI accuracy: {max_acc:.2f}")

                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'dev_{k}': v for k, v in dev_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch, 'n_parameters': n_parameters
                }
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    total = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total))))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    parser.add_argument('--labels', dest='labels', type=str, required=True, help="CSV with video,pose,text")
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--device', default='cuda')

    # Probing aids
    parser.add_argument('--probe-shapes', action='store_true',
                        help='Print first batch src_input shapes then exit cleanly.')
    parser.add_argument('--probe-forward', action='store_true',
                        help='Run exactly one forward pass then exit cleanly.')

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
