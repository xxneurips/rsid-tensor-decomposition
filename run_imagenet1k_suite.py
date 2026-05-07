"""
ImageNet-1K compression suite for ResNet-18.

Default settings (tuned for an 11 GB GPU without sm_70+):
  - No torch.compile (requires sm_70+)
  - AMP enabled (FP16 fwd)
  - Batch size 96 to fit in 11 GB after RandomResizedCrop
  - 5 epochs of KD per fine-tune (one-shot, struct) and 3 epochs/stage * 3 stages for RSID
  - Teacher is torchvision IMAGENET1K_V1 RN18 (~69.76% top-1)

For each (method, seed) we record best val top-1, final val top-1, and parameter count.
"""

import argparse
import copy
import json
import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.decomposition import decompose_model
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger

from run_structured_pruning_baseline import (
    StructuredPruneMask, count_effective_params_and_flops, evaluate as struct_eval,
)

logger = setup_logger("imagenet1k_suite")

DEFAULT_SEEDS = [3, 7, 11]
RANK_RATIO = 0.3
EPOCHS_ONESHOT = 5      # one-shot total
EPOCHS_RSID_STAGE = 2   # 3 stages * 2 = 6 total epochs (matches one-shot+1)
RSID_STAGES = 3
EPOCHS_STRUCT = 5
STRUCT_FILTER_RATIO = 0.50


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_teacher(model_name, dataset, device, ckpt_dir="./checkpoints"):
    """Load teacher model. For full ImageNet (1000 classes) the torchvision-
    pretrained model is directly usable. For ImageNet-100 we need a separately
    fine-tuned 100-class head  --  load from ckpt_dir if present, else error."""
    import os
    model = get_model(model_name, dataset=dataset, pretrained=True, device=device)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{dataset}.pt")
    if dataset == "imagenet":
        return model.eval()  # use IMAGENET1K_V1 directly
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded teacher: {ckpt_path}")
    else:
        logger.warning(f"No teacher checkpoint at {ckpt_path}. Using ImageNet-pretrained model with replaced classifier (will perform poorly on {dataset}!).")
    return model.eval()


# Per-architecture rank ratio used by RSID-Tucker.
# ResNet-50 has more bottleneck blocks -> we keep ratio 0.3 to match RN-18 budget reduction.
MODEL_RANK_RATIOS = {"resnet18": 0.3, "resnet50": 0.3}


def get_loaders(dataset, data_dir, batch_size, num_workers):
    return get_data_loaders(dataset, batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)


def make_trainer(teacher, student, train_loader, val_loader, lr, epochs, device):
    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=lr, epochs=epochs, use_amp=True,
        compile_model=False,  # disable torch.compile for compatibility with pre-sm_70 GPUs
    )
    return trainer


@torch.no_grad()
def quick_eval(model, val_loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def run_oneshot(model_name, dataset, data_dir, device, epochs, seed, batch_size, workers):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir="./checkpoints")
    train_loader, val_loader = get_loaders(dataset, data_dir, batch_size, workers)

    teacher_acc = quick_eval(teacher, val_loader, device)
    logger.info(f"  Teacher top-1 = {teacher_acc:.2f}%")

    rank_ratio = MODEL_RANK_RATIOS.get(model_name, RANK_RATIO)
    student = copy.deepcopy(teacher)
    student = decompose_model(student, method="tucker", rank_ratio=rank_ratio, device=device)
    pre_acc = quick_eval(student, val_loader, device)
    logger.info(f"  pre-FT one-shot top-1 = {pre_acc:.2f}%")

    trainer = make_trainer(teacher, student, train_loader, val_loader, lr=0.005, epochs=epochs, device=device)
    history = trainer.train()
    best = max(history["val_acc"])
    final = history["val_acc"][-1]
    n_params = sum(p.numel() for p in trainer.get_student().parameters())
    return {
        "seed": seed, "method": "oneshot_tucker",
        "teacher_acc": teacher_acc,
        "pre_ft_acc": pre_acc,
        "best_val_acc": best, "final_val_acc": final,
        "params_m": n_params / 1e6, "epochs": epochs,
    }


def run_rsid(model_name, dataset, data_dir, device, epochs_per_stage, n_stages, seed, batch_size, workers):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir="./checkpoints")
    train_loader, val_loader = get_loaders(dataset, data_dir, batch_size, workers)

    teacher_acc = quick_eval(teacher, val_loader, device)
    logger.info(f"  Teacher top-1 = {teacher_acc:.2f}%")

    start = 0.8
    end = MODEL_RANK_RATIOS.get(model_name, RANK_RATIO)
    schedule = [start * (end / start) ** ((i + 1) / n_stages) for i in range(n_stages)]
    logger.info(f"  RSID schedule: {[f'{r:.3f}' for r in schedule]}")

    current_teacher = teacher
    history_acc = []
    per_stage_val = []
    n_params_final = None
    for si, ri in enumerate(schedule):
        logger.info(f"  Stage {si+1}/{n_stages}  ratio={ri:.3f}")
        student = copy.deepcopy(teacher)
        student = decompose_model(student, method="tucker", rank_ratio=ri, device=device)

        trainer = make_trainer(current_teacher, student, train_loader, val_loader,
                                lr=0.005, epochs=epochs_per_stage, device=device)
        h = trainer.train()
        history_acc.extend(h["val_acc"])
        per_stage_val.append({
            "stage": si + 1,
            "rank_ratio": float(ri),
            "val_acc_per_epoch": [float(x) for x in h["val_acc"]],
            "best_val_acc": float(max(h["val_acc"])) if h["val_acc"] else None,
            "final_val_acc": float(h["val_acc"][-1]) if h["val_acc"] else None,
            "params": sum(p.numel() for p in trainer.get_student().parameters()),
        })
        current_teacher = trainer.get_student()
        n_params_final = sum(p.numel() for p in current_teacher.parameters())
    final_stage = per_stage_val[-1] if per_stage_val else {}
    return {
        "seed": seed, "method": "rsid_tucker",
        "teacher_acc": teacher_acc,
        "best_val_acc": max(history_acc),
        "final_val_acc": history_acc[-1] if history_acc else None,
        "final_stage_best_val_acc": final_stage.get("best_val_acc"),
        "final_stage_final_val_acc": final_stage.get("final_val_acc"),
        "per_stage_val": per_stage_val,
        "params_m": n_params_final / 1e6,
        "epochs_per_stage": epochs_per_stage, "n_stages": n_stages,
        "schedule": schedule,
    }


def run_struct(model_name, dataset, data_dir, device, prune_ratio, epochs, seed, batch_size, workers):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir="./checkpoints")
    train_loader, val_loader = get_loaders(dataset, data_dir, batch_size, workers)

    teacher_acc = quick_eval(teacher, val_loader, device)
    logger.info(f"  Teacher top-1 = {teacher_acc:.2f}%")

    student = copy.deepcopy(teacher)
    mask_obj = StructuredPruneMask(student, prune_ratio)
    mask_obj.apply()
    pre_acc = struct_eval(student, val_loader, device)
    logger.info(f"  pre-FT struct prune top-1 = {pre_acc:.2f}%")

    trainer = make_trainer(teacher, student, train_loader, val_loader, lr=0.005, epochs=epochs, device=device)
    orig_step = trainer.optimizer.step
    def masked_step(*a, **kw):
        out = orig_step(*a, **kw)
        mask_obj.apply()
        return out
    trainer.optimizer.step = masked_step
    mask_obj.apply()

    history = trainer.train()
    student = trainer.get_student()
    mask_obj.apply()

    info = get_dataset_info(dataset)
    total_params, eff_params = count_effective_params_and_flops(
        student, (1, 3, info["image_size"], info["image_size"]), mask_obj
    )
    return {
        "seed": seed, "method": "structured_prune",
        "teacher_acc": teacher_acc,
        "pre_ft_acc": pre_acc,
        "prune_ratio": prune_ratio,
        "best_val_acc": max(history["val_acc"]),
        "final_val_acc": history["val_acc"][-1],
        "total_params": int(total_params),
        "effective_params": int(eff_params),
        "epochs": epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", default="/imagenet_data/imagenet")
    parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "imagenet100"])
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--output-json", default="./results/imagenet1k_suite.json")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", type=str, nargs="+", default=["oneshot", "rsid", "struct"])
    parser.add_argument("--epochs-oneshot", type=int, default=EPOCHS_ONESHOT)
    parser.add_argument("--epochs-rsid-stage", type=int, default=EPOCHS_RSID_STAGE)
    parser.add_argument("--rsid-stages", type=int, default=RSID_STAGES)
    parser.add_argument("--epochs-struct", type=int, default=EPOCHS_STRUCT)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)

    for method in args.methods:
        key = f"{method}_{args.model}_{args.dataset}"
        if key not in results:
            results[key] = {"per_seed": []}
        existing = {r["seed"] for r in results[key]["per_seed"]}
        for seed in args.seeds:
            if seed in existing:
                logger.info(f"  [{key}] seed {seed} present  --  skipping")
                continue
            t0 = time.time()
            try:
                if method == "oneshot":
                    r = run_oneshot(args.model, args.dataset, args.data_dir, args.device, args.epochs_oneshot,
                                     seed, args.batch_size, args.workers)
                elif method == "rsid":
                    r = run_rsid(args.model, args.dataset, args.data_dir, args.device,
                                  args.epochs_rsid_stage, args.rsid_stages, seed, args.batch_size, args.workers)
                elif method == "struct":
                    r = run_struct(args.model, args.dataset, args.data_dir, args.device, STRUCT_FILTER_RATIO,
                                    args.epochs_struct, seed, args.batch_size, args.workers)
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
    print("IMAGENET-1K SUITE SUMMARY")
    print("=" * 80)
    for k, v in results.items():
        if "accuracy" in v:
            a = v["accuracy"]
            print(f"{k:50s}  {a['mean']:6.2f} +- {a['std']:5.2f}  N={a['n_seeds']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
