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
# Minimal WLBSL ISLR dataset
# ---------------------------

def _load_pose_tensor(pkl_path: str) -> torch.Tensor:
    """Load a pickle pose file into a float32 torch.Tensor of shape [T, J, C]."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        for key in ("keypoints", "pose", "kpts", "skeleton", "data"):
            if key in obj:
                arr = np.asarray(obj[key])
                break
        else:
            arr = None
            for v in obj.values():
                try:
                    arr = np.asarray(v)
                    break
                except Exception:
                    pass
            if arr is None:
                raise ValueError(f"Unsupported PKL structure: {pkl_path}")
    else:
        arr = np.asarray(obj)

    # Normalize to [T, J, C]
    if arr.ndim == 2:  # [T, D] → [T, J, C]
        T, D = arr.shape
        if D % 3 == 0:
            arr = arr.reshape(T, D // 3, 3)
        elif D % 2 == 0:
            arr = arr.reshape(T, D // 2, 2)
        else:
            arr = arr.reshape(T, D, 1)
    elif arr.ndim == 3:
        # Already [T, J, C]
        pass
    elif arr.ndim == 4:
        # Common encodings like [T, J, C, 1] or [1, T, J, C]
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)  # fallback
    else:
        raise ValueError(f"Unsupported pose shape {arr.shape} in {pkl_path}")

    return torch.tensor(arr, dtype=torch.float32)


def _hash_bucket(s: str) -> int:
    return (abs(hash(s)) % 10_000) % 100  # 0..99


class WLBSL_ISLR_Dataset(Dataset):
    """
    Reads a single CSV (video,pose,text) and splits deterministically:
      train 0-89, dev 90-94, test 95-99 by hashing the video path.
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
        pose = _load_pose_tensor(it["pose"])  # [T, J, C]
        T = pose.shape[0]
        src_input = {
            "pose": pose,                       # [T, J, C] float32
            "pose_len": torch.tensor(T).long(), # scalar
        }
        tgt_input = {
            "gt_sentence": it["text"],          # string label
        }
        return src_input, tgt_input

    @staticmethod
    def _split_parts_tensor(pad_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split [B, T, J, C] into parts for (body, left, right, face_all).
        Supports common layouts:
          - WholeBody (133): 23 body, 21 left, 21 right, 68 face
          - Compact (69):     9 body, 21 left, 21 right, 18 face
        """
        B, T, J, C = pad_pose.shape

        schemes = [
            ("wholebody133", (23, 21, 21, 68)),
            ("compact69",    ( 9, 21, 21, 18)),
        ]
        chosen = None
        for _, (b, l, r, f) in schemes:
            if b + l + r + f == J:
                chosen = (b, l, r, f)
                break

        if chosen is None:
            # If not exact, try to map body/hand/hand and put the rest to face
            # using the most common prefix counts
            for b_try in (23, 9):
                l_try, r_try = 21, 21
                if J >= b_try + l_try + r_try:
                    f_try = J - (b_try + l_try + r_try)
                    chosen = (b_try, l_try, r_try, f_try)
                    break

        if chosen is None:
            raise RuntimeError(
                f"Cannot split J={J} joints into parts. "
                f"Expected one of: 23+21+21+68 or 9+21+21+18 (or larger face remainder)."
            )

        bN, lN, rN, fN = chosen
        idx = 0
        parts = {}
        parts["body"]     = pad_pose[:, :, idx:idx + bN, :]; idx += bN
        parts["left"]     = pad_pose[:, :, idx:idx + lN, :]; idx += lN
        parts["right"]    = pad_pose[:, :, idx:idx + rN, :]; idx += rN
        parts["face_all"] = pad_pose[:, :, idx:idx + fN, :]; idx += fN
        return parts

    @staticmethod
    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)
        poses = [b["pose"] for b in src_list]
        lens  = torch.stack([b["pose_len"] for b in src_list], dim=0)

        # Pad to [B, T, J, C]. If extra trailing dims, flatten to channel.
        pad_pose = pad_sequence(poses, batch_first=True, padding_value=0.0)
        B, max_len = pad_pose.shape[:2]
        feat_shape = pad_pose.shape[2:]
        if len(feat_shape) == 1:
            J, C = feat_shape[0], 1
            pad_pose = pad_pose.unsqueeze(-1)
        else:
            J, C = feat_shape[0], int(np.prod(feat_shape[1:]))
            pad_pose = pad_pose.reshape(B, max_len, J, C)

        parts = WLBSL_ISLR_Dataset._split_parts_tensor(pad_pose)

        # Attention mask for valid timesteps
        mask = torch.arange(max_len).expand(B, max_len) < lens.unsqueeze(1)

        src_input = {
            **parts,
            "attention_mask": mask.long(),
        }
        tgt_input = {"gt_sentence": [b["gt_sentence"] for b in tgt_list]}
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
    """Move all tensors to CUDA and cast if dtype is provided."""
    for k in list(src_input.keys()):
        v = src_input[k]
        if isinstance(v, torch.Tensor):
            v = v.cuda(non_blocking=True)
            if dtype is not None:
                v = v.to(dtype)
            src_input[k] = v
    return src_input


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header, print_freq = f'Epoch: [{epoch}/{args.epochs}]', 10
    optimizer.zero_grad()

    # If DeepSpeed bf16/fp16 is enabled the engine will handle casting; keep inputs fp32 unless specified
    target_dtype = None

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        src_input = _maybe_cast_and_move(src_input, target_dtype)

        # Optional probing
        if getattr(args, "probe_shapes", False) and step == 0 and utils.is_main_process():
            print("== Incoming src_input shapes ==")
            for k, v in src_input.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)} {v.dtype} {v.device}", flush=True)
            # Clean shutdown (avoids NCCL warning)
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            raise SystemExit(0)

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']

        # One-pass forward only probe
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

    with torch.no_grad():
        tgt_pres, tgt_refs = [], []
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, f'Eval[{phase}]:')):
            src_input = _maybe_cast_and_move(src_input, None)

            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=float(total_loss.item()))

            output = model_without_ddp.generate(stack_out, max_new_tokens=50, num_beams=1)
            # Convert generated token ids to strings if needed
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
    # Ensure fp32 master params pre-DS init
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if args.finetune:
        print('***********************************\nLoad Checkpoint...\n***********************************')
        state = torch.load(args.finetune, map_location='cpu')
        state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys:\n', '\n'.join(ret.missing_keys))
        print('Unexpected keys:\n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model

    # NOTE: DeepSpeed wrapping happens in utils.init_deepspeed
    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_loader) / max(1, args.gradient_accumulation_steps)),
        num_training_steps=int(args.epochs * len(train_loader) / max(1, args.gradient_accumulation_steps)),
    )

    # DeepSpeed initialize (returns engine-wrapped model, opt, sched)
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)

    # Resolve the underlying module for generation/saving
    try:
        # deepspeed.DeepSpeedEngine has .module (the DDP-wrapped model), which itself has .module (the raw nn.Module)
        model_without_ddp = model.module.module
    except Exception:
        try:
            model_without_ddp = model.module
        except Exception:
            model_without_ddp = model  # fallback

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
                if hasattr(s, "set_epoch"):
                    s.set_epoch(epoch)

            train_stats = train_one_epoch(args, model, train_loader, optimizer, epoch)

            # Save checkpoint (only trainable params)
            ckpt_path = output_dir / f'checkpoint_{epoch}.pth'
            utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, ckpt_path)

            # Eval on dev/test (master only)
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
        # Always clean up distributed to avoid NCCL warnings on exceptions/early exit
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    total = time.time() - start_time
    print('Training time', str(datetime.timedelta(seconds=int(total))))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Base parser from utils (keeps deepspeed/optim args and existing flags)
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])

    # Non-conflicting additions for this stage
    parser.add_argument('--labels', dest='labels', type=str, required=True, help="CSV with video,pose,text")
    parser.add_argument('--stage', type=int, default=2)   # accepted, not used for branching
    parser.add_argument('--device', default='cuda')       # accepted, not used

    # Probing aids
    parser.add_argument('--probe-shapes', action='store_true',
                        help='Print first batch src_input shapes then exit cleanly.')
    parser.add_argument('--probe-forward', action='store_true',
                        help='Run exactly one forward pass then exit cleanly.')

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
