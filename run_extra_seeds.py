"""
Extension script: add seeds [3, 11] to existing 3-seed configs.
Loads existing multiseed_results.json, runs only missing seeds, merges back.
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

logger = setup_logger("extra_seeds")

NEW_SEEDS = [3, 11]

# Same configs as the original multiseed run
CONFIGS = [
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar100", "method": "tucker"},
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar100", "method": "tt"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar100", "method": "tucker"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar100", "method": "tt"},
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar10",  "method": "tucker"},
    {"type": "oneshot", "model": "resnet18", "dataset": "cifar10",  "method": "tt"},
    {"type": "rsid",    "model": "resnet18", "dataset": "cifar10",  "method": "tt"},
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_teacher(model_name, dataset, device, checkpoint_dir):
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset}.pt")
    model = get_model(model_name, dataset=dataset, pretrained=True, device=device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    return model.eval()


def run_oneshot_seeded(model_name, dataset, method, device, data_dir, ckpt_dir, epochs, seed):
    set_seed(seed)
    logger.info(f"  [Seed {seed}] One-shot {method} on {model_name}/{dataset}")
    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)
    info = get_dataset_info(dataset)
    student = copy.deepcopy(teacher)
    student = decompose_model(student, method=method, rank_ratio=0.3, device=device)
    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.01, epochs=epochs, use_amp=True,
        compile_model=False,
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


def run_rsid_seeded(model_name, dataset, method, device, data_dir, ckpt_dir, epochs, seed):
    set_seed(seed)
    logger.info(f"  [Seed {seed}] RSID-{method} on {model_name}/{dataset}")
    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)
    rsid = RSID(
        method=method, schedule_strategy="exponential",
        start_ratio=0.8, end_ratio=0.3, num_iterations=3,
        epochs_per_stage=epochs, device=device,
        use_amp=True, compile_model=False,
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


def aggregate(seed_results, key):
    if key == "best_val_acc":
        vals = [r.get("best_val_acc", r.get("final_val_acc", 0)) for r in seed_results]
    else:
        vals = [r.get(key, 0) for r in seed_results]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "values": vals,
        "n_seeds": len(vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--input-json", default="./results/multiseed_results.json")
    parser.add_argument("--output-json", default="./results/multiseed_results_5seed.json")
    args = parser.parse_args()

    # Load existing 3-seed results
    with open(args.input_json) as f:
        existing = json.load(f)
    logger.info(f"Loaded existing results with {len(existing)} configs")

    for cfg in CONFIGS:
        config_key = f"{cfg['type']}_{cfg['model']}_{cfg['dataset']}_{cfg['method']}"
        logger.info(f"\n{'='*60}\nExtending: {config_key}\n{'='*60}")

        if config_key not in existing:
            logger.warning(f"  {config_key} not in existing  --  skipping")
            continue

        existing_seeds = {r["seed"] for r in existing[config_key]["per_seed"]}
        logger.info(f"  Existing seeds: {sorted(existing_seeds)}")

        for seed in NEW_SEEDS:
            if seed in existing_seeds:
                logger.info(f"  Seed {seed} already present  --  skipping")
                continue

            t0 = time.time()
            if cfg["type"] == "oneshot":
                r = run_oneshot_seeded(
                    cfg["model"], cfg["dataset"], cfg["method"],
                    args.device, args.data_dir, args.checkpoint_dir,
                    args.epochs, seed,
                )
            else:
                r = run_rsid_seeded(
                    cfg["model"], cfg["dataset"], cfg["method"],
                    args.device, args.data_dir, args.checkpoint_dir,
                    args.epochs, seed,
                )
            dt = time.time() - t0
            logger.info(f"  Seed {seed} completed in {dt/60:.1f} min")

            existing[config_key]["per_seed"].append(r)

            # Re-aggregate after each seed
            acc_key = "best_val_acc" if cfg["type"] == "oneshot" else "final_val_acc"
            existing[config_key]["accuracy"] = aggregate(existing[config_key]["per_seed"], acc_key)

            # Checkpoint after each seed
            with open(args.output_json, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            logger.info(f"  Saved checkpoint to {args.output_json}")

    # Final summary
    print("\n" + "=" * 80)
    print("5-SEED RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<55} {'Mean ± Std':>15} {'N':>5}")
    print("-" * 80)
    for key, val in existing.items():
        agg = val["accuracy"]
        print(f"{key:<55} {agg['mean']:>6.2f} ± {agg['std']:<5.2f}  {agg['n_seeds']:>5}")
    print("=" * 80)


if __name__ == "__main__":
    main()
