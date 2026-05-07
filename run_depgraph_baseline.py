"""
DepGraph (Fang et al. CVPR 2023) structured-pruning baseline on RN-18 / ImageNet-1k.

Uses torch_pruning's MagnitudeImportance + DepGraph dependency-aware pruning at a
matched global prune ratio (default f=0.5, matching the existing structured-pruning
baseline). Fine-tunes for 5 epochs with KD from the FP32 teacher to match the
RSID one-shot baseline schedule.

Output JSON has the same shape as the existing structured-pruning baseline JSON.
"""
import argparse
import json
import logging
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torchvision.models as M
import torch_pruning as tp

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.data import get_data_loaders
from src.distillation.losses import CombinedDistillationLoss

logger = logging.getLogger("depgraph")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def depgraph_prune_resnet18(model, prune_ratio=0.5, device="cuda:0"):
    """Apply DepGraph dependency-aware channel pruning to ResNet-18."""
    model = model.to(device).eval()
    example_inputs = torch.randn(1, 3, 224, 224, device=device)
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs,
        importance=importance,
        global_pruning=False,
        pruning_ratio=prune_ratio,
        ignored_layers=[model.fc],
        round_to=8,
    )
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    logger.info(f"  After DepGraph: {params/1e6:.2f}M params, {macs/1e6:.1f}M MACs")
    return model, params, macs


def fine_tune(student, teacher, train_loader, val_loader, epochs, lr, device):
    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr,
                                 momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    val_history = []
    for epoch in range(epochs):
        student.train()
        teacher.eval()
        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                s = student(x)
                with torch.no_grad():
                    t = teacher(x)
                loss = loss_fn(s, t, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        acc = evaluate(student, val_loader, device)
        val_history.append(acc)
        logger.info(f"  epoch {epoch+1}/{epochs}  val_acc={acc:.2f}%  ({time.time()-t0:.1f}s)")
    return val_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", default="./data/imagenet")
    p.add_argument("--output-json", default="./results/depgraph_rn18_imagenet.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    p.add_argument("--prune-ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    device = args.device
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    train_loader, val_loader = get_data_loaders(
        "imagenet", batch_size=args.batch_size,
        num_workers=args.workers, data_dir=args.data_dir)

    teacher = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    teacher_acc = evaluate(teacher, val_loader, device)
    logger.info(f"Teacher top-1 = {teacher_acc:.3f}%")

    out = {"depgraph_resnet18_imagenet": {"per_seed": []}}
    for seed in args.seeds:
        logger.info(f"\n=== Seed {seed} ===")
        set_seed(seed)
        student = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1).to(device)
        student, params, macs = depgraph_prune_resnet18(
            student, prune_ratio=args.prune_ratio, device=device)

        pre_acc = evaluate(student, val_loader, device)
        logger.info(f"  pre-FT pruned acc = {pre_acc:.2f}%")

        val_hist = fine_tune(student, teacher, train_loader, val_loader,
                             epochs=args.epochs, lr=args.lr, device=device)

        out["depgraph_resnet18_imagenet"]["per_seed"].append({
            "seed": seed,
            "method": "depgraph",
            "teacher_acc": teacher_acc,
            "pre_ft_acc": pre_acc,
            "best_val_acc": float(max(val_hist)),
            "final_val_acc": float(val_hist[-1]),
            "params_m": params / 1e6,
            "macs_m": macs / 1e6,
            "epochs": args.epochs,
            "prune_ratio": args.prune_ratio,
        })
        logger.info(f"  [seed {seed}] best={max(val_hist):.2f}%  final={val_hist[-1]:.2f}%")

        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)

    accs = [r["best_val_acc"] for r in out["depgraph_resnet18_imagenet"]["per_seed"]]
    mean = sum(accs) / len(accs)
    sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1))
    out["depgraph_resnet18_imagenet"]["accuracy"] = {
        "mean": mean, "std": sd, "values": accs, "n_seeds": len(accs)}

    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nFinal: mean={mean:.3f}  std={sd:.3f}  n={len(accs)}")
    logger.info(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
