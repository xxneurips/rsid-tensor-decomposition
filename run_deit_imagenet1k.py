"""
DeiT-Small + RSID-SVD on ImageNet-1k (Action 2 of the NeurIPS 2026 sprint).

Pipeline:
  1. Load DeiT-Small (timm, ImageNet-pretrained 1000-class).
  2. One-shot SVD at rank ratio 0.3 + 5 epochs of KD vs DeiT-S teacher.
  3. RSID-SVD: 3 stages with rank ratios [0.6, 0.42, 0.3], fresh decomposition
     of the original DeiT-S teacher each stage, KD from the most recent student.
  4. 3 seeds (3, 7, 11), recording best val top-1 per seed.

The SVD decomposer is the same as run_deit_rsid.py (reused by import).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import get_data_loaders
from run_deit_rsid import (
    LowRankLinear, svd_decompose_linear, should_decompose,
    apply_svd_to_transformer, KDLoss, count_params,
)

logger = logging.getLogger("deit_imagenet1k")


def setup_log():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def get_deit_small_imagenet():
    """timm DeiT-Small with native 1000-class ImageNet head (no fine-tune needed)."""
    return timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=1000)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_epoch(student, teacher, loader, optimizer, loss_fn, scaler, device, log_every=200):
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            s_logits = student(x)
            with torch.no_grad():
                t_logits = teacher(x)
            loss = loss_fn(s_logits, t_logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (s_logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
        if (step + 1) % log_every == 0:
            logger.info(f"    step {step+1}  running_loss={total_loss/total:.4f}  "
                        f"running_acc={100*correct/total:.2f}%  ({time.time()-t0:.1f}s)")
    return total_loss / total, 100.0 * correct / total


def fine_tune(student, teacher, train_loader, val_loader, epochs, lr, device, tag=""):
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = KDLoss(alpha=0.5, temperature=3.0)
    scaler = GradScaler()
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(student, teacher, train_loader, optimizer, loss_fn, scaler, device)
        val_acc = evaluate(student, val_loader, device)
        scheduler.step()
        dt = time.time() - t0
        logger.info(f"  [{tag}] Epoch {epoch+1}/{epochs}  train_loss={tr_loss:.4f} "
                    f"train_acc={tr_acc:.2f}  val_acc={val_acc:.2f}  ({dt/60:.1f} min)")
        history.append(val_acc)
    return history


def run_oneshot(data_dir, device, epochs, seed, batch_size, workers, rank_ratio):
    set_seed(seed)
    teacher = get_deit_small_imagenet().to(device)
    train_loader, val_loader = get_data_loaders("imagenet", batch_size=batch_size,
                                                 num_workers=workers, data_dir=data_dir)
    teacher_acc = evaluate(teacher, val_loader, device)
    logger.info(f"  Teacher top-1 = {teacher_acc:.2f}%")

    student = copy.deepcopy(teacher)
    apply_svd_to_transformer(student, rank_ratio=rank_ratio)
    n_params = count_params(student)
    logger.info(f"  oneshot SVD@{rank_ratio} student params = {n_params/1e6:.2f}M")
    pre_acc = evaluate(student, val_loader, device, max_batches=50)
    logger.info(f"  pre-FT (sampled) acc = {pre_acc:.2f}%")

    history = fine_tune(student, teacher, train_loader, val_loader,
                        epochs=epochs, lr=1e-3, device=device, tag=f"OS-{rank_ratio}")
    return {
        "seed": seed, "method": "oneshot_svd",
        "teacher_acc": teacher_acc,
        "rank_ratio": rank_ratio,
        "best_val_acc": float(max(history)),
        "final_val_acc": float(history[-1]),
        "params_m": n_params / 1e6, "epochs": epochs,
    }


def run_rsid(data_dir, device, epochs_per_stage, schedule, seed, batch_size, workers):
    set_seed(seed)
    teacher = get_deit_small_imagenet().to(device)
    train_loader, val_loader = get_data_loaders("imagenet", batch_size=batch_size,
                                                 num_workers=workers, data_dir=data_dir)
    teacher_acc = evaluate(teacher, val_loader, device)
    logger.info(f"  Teacher top-1 = {teacher_acc:.2f}%")
    logger.info(f"  RSID schedule: {schedule}")

    current_teacher = teacher
    history_acc = []
    per_stage_val = []
    n_params_final = None
    for si, ri in enumerate(schedule):
        logger.info(f"  Stage {si+1}/{len(schedule)}  ratio={ri:.3f}")
        student = copy.deepcopy(teacher)  # FRESH copy from original
        apply_svd_to_transformer(student, rank_ratio=ri)
        pre_acc = evaluate(student, val_loader, device, max_batches=50)
        logger.info(f"    pre-FT (sampled) acc = {pre_acc:.2f}%")
        h = fine_tune(student, current_teacher, train_loader, val_loader,
                      epochs=epochs_per_stage, lr=1e-3, device=device,
                      tag=f"RSID-s{si+1}-{ri:.2f}")
        history_acc.extend(h)
        per_stage_val.append({
            "stage": si + 1,
            "rank_ratio": float(ri),
            "val_acc_per_epoch": [float(x) for x in h],
            "best_val_acc": float(max(h)) if h else None,
            "final_val_acc": float(h[-1]) if h else None,
            "params": count_params(student),
        })
        current_teacher = student
        n_params_final = count_params(student)
    final_stage = per_stage_val[-1] if per_stage_val else {}
    return {
        "seed": seed, "method": "rsid_svd",
        "teacher_acc": teacher_acc,
        "schedule": list(schedule),
        "best_val_acc": float(max(history_acc)),
        "final_val_acc": float(history_acc[-1]) if history_acc else None,
        "final_stage_best_val_acc": final_stage.get("best_val_acc"),
        "final_stage_final_val_acc": final_stage.get("final_val_acc"),
        "per_stage_val": per_stage_val,
        "params_m": n_params_final / 1e6,
        "epochs_per_stage": epochs_per_stage, "n_stages": len(schedule),
    }


def main():
    setup_log()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", default="/imagenet_data/imagenet")
    parser.add_argument("--output-json", default="./results/deit_small_imagenet1k.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=["oneshot", "rsid"])
    parser.add_argument("--rank-ratio-end", type=float, default=0.3)
    parser.add_argument("--rsid-schedule", type=float, nargs="+", default=[0.6, 0.42, 0.3])
    parser.add_argument("--epochs-oneshot", type=int, default=5)
    parser.add_argument("--epochs-rsid-stage", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)

    for method in args.methods:
        key = f"{method}_deit_small_imagenet"
        if key not in results:
            results[key] = {"per_seed": []}
        existing = {r["seed"] for r in results[key]["per_seed"]}
        for seed in args.seeds:
            if seed in existing:
                logger.info(f"  [{key}] seed {seed} already done  --  skipping")
                continue
            t0 = time.time()
            try:
                if method == "oneshot":
                    r = run_oneshot(args.data_dir, args.device, args.epochs_oneshot,
                                     seed, args.batch_size, args.workers, args.rank_ratio_end)
                elif method == "rsid":
                    r = run_rsid(args.data_dir, args.device, args.epochs_rsid_stage,
                                  args.rsid_schedule, seed, args.batch_size, args.workers)
                else:
                    logger.error(f"unknown method: {method}")
                    continue
            except Exception as e:
                logger.error(f"  [{key}] seed {seed} FAILED: {e}")
                import traceback; traceback.print_exc()
                continue
            results[key]["per_seed"].append(r)
            logger.info(f"  [{key}] seed {seed} done: best={r['best_val_acc']:.2f}%  ({(time.time()-t0)/60:.1f} min)")
            accs = [x["best_val_acc"] for x in results[key]["per_seed"]]
            results[key]["accuracy"] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "values": accs,
                "n_seeds": len(accs),
            }
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DEIT-SMALL / IMAGENET-1K SUMMARY")
    print("=" * 80)
    for k, v in results.items():
        if "accuracy" in v:
            a = v["accuracy"]
            print(f"{k:50s}  {a['mean']:6.2f} +- {a['std']:5.2f}  N={a['n_seeds']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
