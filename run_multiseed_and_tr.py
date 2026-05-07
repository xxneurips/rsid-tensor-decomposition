"""
Multi-Seed Experiments + Tensor Ring Evaluation
================================================

This script runs two sets of experiments:

1. MULTI-SEED (3 seeds): Key configurations to establish statistical significance
   - ResNet-18/CIFAR-100: Tucker one-shot, RSID-Tucker, RSID-TT (the main RSID claims)
   - ResNet-18/CIFAR-10: Tucker one-shot, TT one-shot, RSID-TT
   Seeds: 42, 123, 7

2. TENSOR RING: Full TR evaluation across all architectures/datasets
   - One-shot TR at rank_ratio=0.3 (same as Tucker/CP/TT)
   - RSID-TR on ResNet-18 (both datasets)

Results saved to results/ as JSON with per-seed breakdowns and mean±std.

Usage:
    # Run everything (multi-seed + TR)
    python run_multiseed_and_tr.py --device cuda:0

    # Run only multi-seed experiments
    python run_multiseed_and_tr.py --mode multiseed --device cuda:0

    # Run only Tensor Ring experiments
    python run_multiseed_and_tr.py --mode tr --device cuda:0

    # Quick test (1 epoch, to verify setup)
    python run_multiseed_and_tr.py --quick-test
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rsid import RSID
from src.models import get_model, get_feature_layers
from src.decomposition import decompose_model
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger, profile_model
from src.utils.rank_schedule import get_rank_schedule

logger = setup_logger("multiseed_tr")

SEEDS = [42, 123, 7]


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_teacher(model_name, dataset, device, checkpoint_dir="./checkpoints"):
    """Load pre-trained teacher model."""
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset}.pt")
    model = get_model(model_name, dataset=dataset, pretrained=True, device=device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path} (acc={ckpt.get('val_acc', '?')})")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}, using pretrained weights")
    return model.eval()


def run_oneshot_seeded(
    model_name, dataset, method, rank_ratio, device, data_dir,
    checkpoint_dir, epochs, seed,
):
    """Run one-shot decomposition with a specific seed."""
    set_seed(seed)
    logger.info(f"  [Seed {seed}] One-shot {method} on {model_name}/{dataset}")

    teacher = load_teacher(model_name, dataset, device, checkpoint_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)
    info = get_dataset_info(dataset)

    student = copy.deepcopy(teacher)
    student = decompose_model(student, method=method, rank_ratio=rank_ratio, device=device)

    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.01, epochs=epochs, use_amp=True,
        compile_model=False,  # disable compile for seed reproducibility
    )
    history = trainer.train()
    student = trainer.get_student()

    input_size = (1, 3, info["image_size"], info["image_size"])
    profile = profile_model(student, input_size=input_size, device=device)

    best_acc = max(history["val_acc"])
    logger.info(f"  [Seed {seed}] Best acc: {best_acc:.2f}%")

    return {
        "seed": seed,
        "best_val_acc": best_acc,
        "final_val_acc": history["val_acc"][-1],
        "profile": profile,
    }


def run_rsid_seeded(
    model_name, dataset, method, device, data_dir,
    checkpoint_dir, epochs_per_stage, seed,
):
    """Run RSID with a specific seed."""
    set_seed(seed)
    logger.info(f"  [Seed {seed}] RSID-{method} on {model_name}/{dataset}")

    teacher = load_teacher(model_name, dataset, device, checkpoint_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)

    rsid = RSID(
        method=method,
        schedule_strategy="exponential",
        start_ratio=0.8,
        end_ratio=0.3,
        num_iterations=3,
        epochs_per_stage=epochs_per_stage,
        device=device,
        use_amp=True,
        compile_model=False,
    )

    feature_layers = get_feature_layers(model_name)
    result = rsid.compress(teacher, train_loader, val_loader, feature_layers)

    final_acc = result["stage_profiles"][-1].get("val_acc", 0)
    logger.info(f"  [Seed {seed}] Final acc: {final_acc:.2f}%")

    return {
        "seed": seed,
        "final_val_acc": final_acc,
        "stage_profiles": result["stage_profiles"],
        "total_time": result["total_time"],
    }


def aggregate_seeds(seed_results, key="best_val_acc"):
    """Compute mean and std from multi-seed results."""
    if key == "best_val_acc":
        # For oneshot, use best_val_acc
        vals = [r.get("best_val_acc", r.get("final_val_acc", 0)) for r in seed_results]
    else:
        vals = [r.get(key, 0) for r in seed_results]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "values": vals,
        "n_seeds": len(vals),
    }


# ======================================================================
# Multi-seed experiment definitions
# ======================================================================

MULTISEED_CONFIGS = [
    # ResNet-18 / CIFAR-100  --  where RSID shows strongest gains
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar100", "method": "tucker"},
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar100", "method": "tt"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar100", "method": "tucker"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar100", "method": "tt"},
    # ResNet-18 / CIFAR-10  --  secondary validation
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar10",  "method": "tucker"},
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar10",  "method": "tt"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar10",  "method": "tt"},
]

# ======================================================================
# Tensor Ring experiment definitions
# ======================================================================

TR_ONESHOT_CONFIGS = [
    {"model": "resnet18",        "dataset": "cifar10"},
    {"model": "resnet18",        "dataset": "cifar100"},
    {"model": "resnet50",        "dataset": "cifar10"},
    {"model": "resnet50",        "dataset": "cifar100"},
    {"model": "mobilenetv2",     "dataset": "cifar10"},
    {"model": "mobilenetv2",     "dataset": "cifar100"},
    {"model": "efficientnet_b0", "dataset": "cifar10"},
    {"model": "efficientnet_b0", "dataset": "cifar100"},
]

TR_RSID_CONFIGS = [
    {"model": "resnet18", "dataset": "cifar10"},
    {"model": "resnet18", "dataset": "cifar100"},
]


def run_multiseed_experiments(args):
    """Run all multi-seed experiments."""
    results = {}

    for cfg in MULTISEED_CONFIGS:
        config_key = f"{cfg['type']}_{cfg['model']}_{cfg['dataset']}_{cfg['method']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-seed: {config_key}")
        logger.info(f"{'='*60}")

        seed_results = []
        for seed in SEEDS:
            if cfg["type"] == "oneshot":
                r = run_oneshot_seeded(
                    cfg["model"], cfg["dataset"], cfg["method"],
                    rank_ratio=0.3, device=args.device,
                    data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
                    epochs=args.epochs, seed=seed,
                )
            else:
                r = run_rsid_seeded(
                    cfg["model"], cfg["dataset"], cfg["method"],
                    device=args.device, data_dir=args.data_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    epochs_per_stage=args.epochs, seed=seed,
                )
            seed_results.append(r)

        acc_key = "best_val_acc" if cfg["type"] == "oneshot" else "final_val_acc"
        agg = aggregate_seeds(seed_results, key=acc_key)
        results[config_key] = {
            "config": cfg,
            "per_seed": seed_results,
            "accuracy": agg,
        }
        logger.info(
            f"  Result: {agg['mean']:.2f} ± {agg['std']:.2f}% "
            f"(seeds: {agg['values']})"
        )

    # Save results
    save_path = os.path.join(args.results_dir, "multiseed_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nMulti-seed results saved to {save_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<50} {'Mean±Std':>15} {'Seeds':>15}")
    print("-" * 80)
    for key, val in results.items():
        agg = val["accuracy"]
        seeds_str = ", ".join(f"{v:.2f}" for v in agg["values"])
        print(f"{key:<50} {agg['mean']:>6.2f}±{agg['std']:<6.2f}  [{seeds_str}]")
    print("=" * 80)

    return results


def run_tr_experiments(args):
    """Run all Tensor Ring experiments."""
    results = {"oneshot": {}, "rsid": {}}

    # --- One-shot TR across all architectures ---
    for cfg in TR_ONESHOT_CONFIGS:
        config_key = f"tr_{cfg['model']}_{cfg['dataset']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"TR One-shot: {cfg['model']}/{cfg['dataset']}")
        logger.info(f"{'='*60}")

        set_seed(42)
        r = run_oneshot_seeded(
            cfg["model"], cfg["dataset"], "tr",
            rank_ratio=0.3, device=args.device,
            data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs, seed=42,
        )
        results["oneshot"][config_key] = r

    # --- RSID-TR on ResNet-18 ---
    for cfg in TR_RSID_CONFIGS:
        config_key = f"tr_rsid_{cfg['model']}_{cfg['dataset']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"TR RSID: {cfg['model']}/{cfg['dataset']}")
        logger.info(f"{'='*60}")

        set_seed(42)
        r = run_rsid_seeded(
            cfg["model"], cfg["dataset"], "tr",
            device=args.device, data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs_per_stage=args.epochs, seed=42,
        )
        results["rsid"][config_key] = r

    # Save results
    save_path = os.path.join(args.results_dir, "tensor_ring_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nTensor Ring results saved to {save_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("TENSOR RING RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<45} {'Acc (%)':>10} {'Params (M)':>12}")
    print("-" * 80)
    for key, val in results["oneshot"].items():
        acc = val.get("best_val_acc", val.get("final_val_acc", 0))
        params = val.get("profile", {}).get("total_params_m", "?")
        print(f"{key:<45} {acc:>10.2f} {params:>12}")
    for key, val in results["rsid"].items():
        acc = val.get("final_val_acc", 0)
        print(f"{key:<45} {acc:>10.2f}     (RSID)")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-seed + Tensor Ring experiments")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "multiseed", "tr"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run 1 epoch with 1 seed to verify setup")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.quick_test:
        args.epochs = 1
        global SEEDS, MULTISEED_CONFIGS, TR_ONESHOT_CONFIGS, TR_RSID_CONFIGS
        SEEDS = [42]
        MULTISEED_CONFIGS = [MULTISEED_CONFIGS[0]]  # just one config
        TR_ONESHOT_CONFIGS = [TR_ONESHOT_CONFIGS[0]]
        TR_RSID_CONFIGS = [TR_RSID_CONFIGS[0]]
        logger.info("QUICK TEST MODE: 1 epoch, 1 seed, 1 config each")

    start_time = time.time()

    if args.mode in ("all", "multiseed"):
        ms_results = run_multiseed_experiments(args)

    if args.mode in ("all", "tr"):
        tr_results = run_tr_experiments(args)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal experiment time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
