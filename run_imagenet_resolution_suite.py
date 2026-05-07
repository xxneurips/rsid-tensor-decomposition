"""
Single-script orchestrator for the ImageNet-resolution experiments
(Imagenette + Imagewoof). For each dataset we run, with 3 seeds:
  (1) one-shot Tucker + KD at 30 epochs
  (2) RSID-Tucker (3 stages) at 10 epochs/stage  (matched 30-epoch budget)
  (3) structured channel pruning + KD at 30 epochs (filter ratio matched
      to the same compression target as RSID-Tucker)

All numbers are written into results/imagenet_resolution_suite.json
incrementally so partial runs are recoverable.
"""

import argparse
import copy
import json
import os
import sys
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.decomposition import decompose_model
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger, profile_model

# Re-use the structured-pruning machinery
from run_structured_pruning_baseline import (
    StructuredPruneMask, count_effective_params_and_flops, evaluate as struct_eval,
)

logger = setup_logger("imagenet_suite")

DEFAULT_SEEDS = [3, 7, 11]
RANK_RATIO = 0.3
EPOCHS_TOTAL = 30           # one-shot + structured pruning use this directly
RSID_STAGES = 3              # RSID schedule = [0.8, sqrt(0.8*0.3)=0.49, 0.3]
RSID_EPOCHS_PER_STAGE = EPOCHS_TOTAL // RSID_STAGES  # 10
STRUCT_FILTER_RATIO = 0.50   # ~ matched dense FLOPs to RSID-Tucker on RN18

DATASETS = ["imagenette", "imagewoof"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_teacher(model_name, dataset, device, ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{dataset}.pt")
    model = get_model(model_name, dataset=dataset, pretrained=True, device=device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded: {ckpt_path}")
    return model.eval()


def run_oneshot(model_name, dataset, rank_ratio, device, data_dir, ckpt_dir, epochs, seed, batch_size):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, num_workers=4, data_dir=data_dir)
    info = get_dataset_info(dataset)

    student = copy.deepcopy(teacher)
    student = decompose_model(student, method="tucker", rank_ratio=rank_ratio, device=device)

    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.005, epochs=epochs, use_amp=True,
        compile_model=False,
    )
    history = trainer.train()
    student = trainer.get_student()
    profile = profile_model(student, input_size=(1, 3, info["image_size"], info["image_size"]), device=device)
    return {
        "seed": seed,
        "method": "oneshot_tucker",
        "best_val_acc": max(history["val_acc"]),
        "final_val_acc": history["val_acc"][-1],
        "params_m": profile.get("total_params_m"),
        "flops_m": profile.get("total_flops_m"),
        "epochs": epochs,
    }


def run_rsid(model_name, dataset, end_ratio, device, data_dir, ckpt_dir, epochs_per_stage, n_stages, seed, batch_size):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, num_workers=4, data_dir=data_dir)
    info = get_dataset_info(dataset)

    # Exponential schedule from 0.8 -> end_ratio over n_stages steps
    start_ratio = 0.8
    schedule = [start_ratio * (end_ratio / start_ratio) ** ((i + 1) / n_stages) for i in range(n_stages)]
    logger.info(f"  RSID schedule: {[f'{r:.3f}' for r in schedule]}")

    current_teacher = teacher
    final_acc = None
    history_acc = []
    for stage_idx, ri in enumerate(schedule):
        logger.info(f"  Stage {stage_idx+1}/{n_stages} ratio={ri:.3f}")
        student = copy.deepcopy(teacher)  # FRESH from original
        student = decompose_model(student, method="tucker", rank_ratio=ri, device=device)

        loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
        trainer = DistillationTrainer(
            teacher=current_teacher, student=student,
            train_loader=train_loader, val_loader=val_loader,
            loss_fn=loss_fn, device=device,
            lr=0.005, epochs=epochs_per_stage, use_amp=True,
            compile_model=False,
        )
        h = trainer.train()
        history_acc.extend(h["val_acc"])
        current_teacher = trainer.get_student()
        final_acc = max(h["val_acc"])

    profile = profile_model(current_teacher, input_size=(1, 3, info["image_size"], info["image_size"]), device=device)
    return {
        "seed": seed,
        "method": "rsid_tucker",
        "best_val_acc": max(history_acc),
        "final_val_acc": final_acc,
        "params_m": profile.get("total_params_m"),
        "flops_m": profile.get("total_flops_m"),
        "epochs_per_stage": epochs_per_stage,
        "n_stages": n_stages,
        "schedule": schedule,
    }


def run_struct(model_name, dataset, prune_ratio, device, data_dir, ckpt_dir, epochs, seed, batch_size):
    set_seed(seed)
    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, batch_size=batch_size, num_workers=4, data_dir=data_dir)
    info = get_dataset_info(dataset)

    student = copy.deepcopy(teacher)
    mask_obj = StructuredPruneMask(student, prune_ratio)
    mask_obj.apply()
    pre_acc = struct_eval(student, val_loader, device)
    logger.info(f"  pre-FT acc after structured prune: {pre_acc:.2f}%")

    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.005, epochs=epochs, use_amp=True,
        compile_model=False,
    )
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

    total_params, eff_params = count_effective_params_and_flops(
        student, (1, 3, info["image_size"], info["image_size"]), mask_obj
    )
    return {
        "seed": seed,
        "method": "structured_prune",
        "prune_ratio": prune_ratio,
        "best_val_acc": max(history["val_acc"]),
        "final_val_acc": history["val_acc"][-1],
        "pre_ft_acc": pre_acc,
        "total_params": int(total_params),
        "effective_params": int(eff_params),
        "epochs": epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output-json", default="./results/imagenet_resolution_suite.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["oneshot", "rsid", "struct"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)

    for dataset in args.datasets:
        for method in args.methods:
            key = f"{method}_resnet18_{dataset}"
            if key not in results:
                results[key] = {"per_seed": []}
            existing = {r["seed"] for r in results[key]["per_seed"]}

            for seed in args.seeds:
                if seed in existing:
                    logger.info(f"  [{key}] seed {seed} already present  --  skipping")
                    continue
                t0 = time.time()
                try:
                    if method == "oneshot":
                        r = run_oneshot("resnet18", dataset, RANK_RATIO,
                                         args.device, args.data_dir, args.checkpoint_dir,
                                         EPOCHS_TOTAL, seed, args.batch_size)
                    elif method == "rsid":
                        r = run_rsid("resnet18", dataset, RANK_RATIO,
                                     args.device, args.data_dir, args.checkpoint_dir,
                                     RSID_EPOCHS_PER_STAGE, RSID_STAGES, seed, args.batch_size)
                    elif method == "struct":
                        r = run_struct("resnet18", dataset, STRUCT_FILTER_RATIO,
                                       args.device, args.data_dir, args.checkpoint_dir,
                                       EPOCHS_TOTAL, seed, args.batch_size)
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
    print("IMAGENET-RESOLUTION SUITE SUMMARY")
    print("=" * 80)
    for k, v in results.items():
        if "accuracy" in v:
            agg = v["accuracy"]
            print(f"{k:50s}  {agg['mean']:6.2f} +- {agg['std']:5.2f}  N={agg['n_seeds']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
