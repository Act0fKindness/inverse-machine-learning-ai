#!/usr/bin/env python3
from pickletools import optimize  # unused but kept if referenced elsewhere
import os
import sys
import csv
import time
import math
import json
import datetime
import argparse
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, Set

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from timm.optim import create_optimizer
from transformers import get_scheduler

from models import Uni_Sign, get_requires_grad_dict
import utils as utils
from datasets_local import S2T_Dataset_news   # <- avoid HF 'datasets' collision
from SLRT_metrics import translation_performance
from config import *

# ---------------------------
# Helpers for skip_pose.csv
# ---------------------------

def _pick_col(cols, *candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _derive_pose_basename_from_row(row: Dict[str, Any], pose_col: Optional[str], video_col: Optional[str]) -> Optional[str]:
    """
    Return the .pkl basename we should check/skip, derived from row fields.
    Priority: explicit 'pose' > 'expected_pose_path' > 'video' (with .mp4→.pkl)
    """
    if pose_col and row.get(pose_col, "").strip():
        name = Path(row[pose_col].strip()).name
        if not name.endswith(".pkl"):
            name = f"{name}.pkl"
        return name

    epp = (row.get("expected_pose_path") or "").strip()
    if epp:
        return Path(epp).name

    if video_col and row.get(video_col, "").strip():
        v = Path(row[video_col].strip())
        return v.with_suffix(".pkl").name

    return None

def load_skip_set_for_labels(label_path: str) -> Set[str]:
    """
    Given a label json path, find skip_pose.csv (env SKIP_POSE_CSV overrides).
    Return set of pose basenames to skip. Empty set if file missing.
    """
    labels_path = Path(label_path)
    default_csv = labels_path.parent / "skip_pose.csv"
    csv_path = Path(os.environ.get("SKIP_POSE_CSV", str(default_csv)))

    if not csv_path.exists():
        return set()

    skip: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        pose_col = _pick_col(fieldnames, "pose", "pose_name", "pose_path")
        video_col = _pick_col(fieldnames, "video", "video_name", "video_path", "rgb", "rgb_path")

        for row in reader:
            name = _derive_pose_basename_from_row(row, pose_col, video_col)
            if name:
                skip.add(name)
    return skip

def _pose_basename_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """
    For a JSON label sample dict, return its pose .pkl basename.
    Uses sample['pose'] if exists, else sample['video'] with .pkl suffix.
    """
    if not isinstance(sample, dict):
        return None
    if sample.get("pose"):
        p = Path(str(sample["pose"]))
        name = p.name
        if not name.endswith(".pkl"):
            name = f"{name}.pkl"
        return name
    if sample.get("video"):
        v = Path(str(sample["video"]))
        return v.with_suffix(".pkl").name
    return None

def filter_labels_json(original_json_path: str, skip_set: Set[str], out_json_path: str) -> Dict[str, int]:
    """
    Load labels JSON and filter out any samples whose pose basename is in skip_set.
    Writes filtered JSON to out_json_path if anything changed.
    Returns counts: {'train_removed': x, 'dev_removed': y, 'test_removed': z, 'total_removed': t}
    """
    with open(original_json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    removed_counts = {"train_removed": 0, "dev_removed": 0, "test_removed": 0, "total_removed": 0}

    def _filter_split(lst):
        keep, removed = [], 0
        for s in lst:
            name = _pose_basename_from_sample(s)
            if name and name in skip_set:
                removed += 1
                continue
            keep.append(s)
        return keep, removed

    changed = False
    for split in ("train", "dev", "test"):
        if split in labels and isinstance(labels[split], list):
            before = len(labels[split])
            labels[split], removed = _filter_split(labels[split])
            after = len(labels[split])
            if removed > 0 or after != before:
                changed = True
            key = f"{split}_removed"
            removed_counts[key] = removed
            removed_counts["total_removed"] += removed

    if changed:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(labels, f)
    return removed_counts

def make_filtered_label_path(original_path: str) -> str:
    """
    Create a deterministic filtered filename next to the original, e.g.:
      CSL_News_Labels.json -> CSL_News_Labels.filtered.skip_pose.json
    """
    p = Path(original_path)
    out_name = f"{p.stem}.filtered.skip_pose.json"
    return str(p.with_name(out_name))

def wait_for_file(path: str, timeout_s: int = 600):
    """
    For non-main processes: wait until 'path' exists and has non-trivial size.
    """
    end = time.time() + timeout_s
    while time.time() < end:
        if os.path.exists(path) and os.path.getsize(path) > 10:
            return
        time.sleep(0.2)
    return

# ---------------------------
# Training / Evaluation
# ---------------------------

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    # Resolve label paths and apply skip_pose filtering (once on main rank)
    train_labels_path = train_label_paths[args.dataset]
    dev_labels_path   = dev_label_paths[args.dataset]

    # Load skip sets (per labels dir)
    skip_train = load_skip_set_for_labels(train_labels_path)
    skip_dev   = load_skip_set_for_labels(dev_labels_path)

    # Prepare filtered paths (no-arg helper to avoid name clashes)
    filtered_train_path = make_filtered_label_path(train_labels_path)
    filtered_dev_path   = make_filtered_label_path(dev_labels_path)

    # Only the main process writes the filtered files
    if utils.is_main_process():
        if skip_train:
            stats = filter_labels_json(train_labels_path, skip_train, filtered_train_path)
            print(f"[skip_pose] train removed: {stats['train_removed']} (total removed: {stats['total_removed']})")
        else:
            if not os.path.exists(filtered_train_path):
                with open(train_labels_path, "r", encoding="utf-8") as f_in, open(filtered_train_path, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            print("[skip_pose] no train skip list found; using original labels")

        if skip_dev:
            stats = filter_labels_json(dev_labels_path, skip_dev, filtered_dev_path)
            print(f"[skip_pose] dev removed: {stats['dev_removed']} (total removed: {stats['total_removed']})")
        else:
            if not os.path.exists(filtered_dev_path):
                with open(dev_labels_path, "r", encoding="utf-8") as f_in, open(filtered_dev_path, "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())
            print("[skip_pose] no dev skip list found; using original labels")

    # Non-main processes wait for the filtered files to appear
    if args.distributed and not utils.is_main_process():
        wait_for_file(filtered_train_path, timeout_s=600)
        wait_for_file(filtered_dev_path, timeout_s=600)

    # Create datasets
    print("Creating dataset:")
    train_data = S2T_Dataset_news(path=filtered_train_path, args=args, phase='train')
    dev_data   = S2T_Dataset_news(path=filtered_dev_path,   args=args, phase='dev')

    if utils.is_main_process():
        try:
            print(f"[dataset sizes] train={len(train_data)}  dev={len(dev_data)}")
        except Exception:
            pass

    # Dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, shuffle=False)
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=args.pin_mem
    )

    # Model
    print("Creating model:")
    model = Uni_Sign(args=args)
    model.cuda()
    model.train()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)

    if args.quick_break <= 0:
        args.quick_break = len(train_dataloader)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(train_dataloader) / args.gradient_accumulation_steps),
    )

    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    print(optimizer)

    # --------------- Auto-resume (model weights only) ---------------
    import re
    start_epoch = 0
    if args.output_dir and Path(args.output_dir).exists():
        ckpts = sorted(
            Path(args.output_dir).glob("checkpoint_*.pth"),
            key=lambda p: int(re.findall(r"\d+", p.stem)[-1]) if re.findall(r"\d+", p.stem) else -1
        )
        if ckpts:
            last = ckpts[-1]
            try:
                state = torch.load(last, map_location="cpu")["model"]
                ret = model_without_ddp.load_state_dict(state, strict=False)
                start_epoch = int(re.findall(r"\d+", last.stem)[-1]) + 1
                if utils.is_main_process():
                    print(f"🔁 Resuming from {last} -> start_epoch={start_epoch}")
                    if hasattr(ret, "missing_keys") and ret.missing_keys:
                        print("Missing keys:", ret.missing_keys)
                    if hasattr(ret, "unexpected_keys") and ret.unexpected_keys:
                        print("Unexpected keys:", ret.unexpected_keys)
            except Exception as e:
                if utils.is_main_process():
                    print(f"⚠️ Failed to resume from {last}: {e}. Starting from epoch 0.")

    output_dir = Path(args.output_dir)
    start_time = time.time()
    max_accuracy = 0

    # Eval-only mode
    if args.eval:
        if utils.is_main_process():
            print("📄 test result")
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        return

    print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch, model_without_ddp=model_without_ddp)

        if args.output_dir:
            ckpt_path = output_dir / f'checkpoint_{epoch}.pth'
            utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, ckpt_path)

        # Skip eval if dev split is empty
        try:
            dev_len = len(dev_data)
        except Exception:
            dev_len = 0

        if dev_len > 0:
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
            bleu4 = float(test_stats.get('bleu4', 0.0))
            print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {bleu4:.2f}")

            if max_accuracy < bleu4:
                max_accuracy = bleu4
                if args.output_dir and utils.is_main_process():
                    best_path = output_dir / 'best_checkpoint.pth'
                    utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, best_path)
        else:
            if utils.is_main_process():
                print("⚠️ Dev set is empty after filtering; skipping evaluation this epoch.")
            test_stats = {"bleu4": 0.0}

        print(f'Max BLEU-4: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model, data_loader, optimizer, epoch, model_without_ddp):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if (step + 1) % args.quick_break == 0:
            if args.output_dir:
                output_dir = Path(args.output_dir)
                utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)},
                                     output_dir / 'checkpoint.pth')

        if target_dtype is not None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        stack_out = model(src_input, tgt_input)

        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp):
    # Early exit if dev loader is empty -> avoids ZeroDivision in utils.log_every
    try:
        dl_len = len(data_loader)
    except TypeError:
        dl_len = 0
    if dl_len == 0:
        if utils.is_main_process():
            print("⚠️ Dev dataloader is empty; skipping evaluation for this epoch.")
        return {"loss": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rouge": 0.0}

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype is not None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())

            output = model_without_ddp.generate(
                stack_out,
                max_new_tokens=100,
                num_beams=4,
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    # Guard: nothing produced
    if len(tgt_pres) == 0:
        return {
            "loss": metric_logger.loss.global_avg if hasattr(metric_logger, "loss") else 0.0,
            "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rouge": 0.0
        }

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    # Ensure first item is long enough before pad_sequence
    if isinstance(tgt_pres[0], torch.Tensor):
        need = max(0, 150 - int(tgt_pres[0].numel()))
        if need > 0:
            pad_tensor = torch.ones(need, device=tgt_pres[0].device) * padding_value
            tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    if args.dataset == 'CSL_News':
        tgt_pres = [' '.join(list(r.replace(" ", '').replace("\n", ''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？", "?").replace(" ", ''))) for r in tgt_refs]

    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    for k, v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)

    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + '/tmp_pres.txt', 'w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i] + '\n')
        with open(args.output_dir + '/tmp_refs.txt', 'w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i] + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

