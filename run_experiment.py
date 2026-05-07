"""
Main Experiment Runner
=======================

Runs the full RSID experiment suite: one-shot baselines, RSID compression,
ablation studies, and profiling. Results are saved as JSON for paper figures.

Usage:
    # Run RSID with default settings (ResNet-18, CIFAR-10, Tucker)
    python run_experiment.py

    # Run specific configuration
    python run_experiment.py --model resnet18 --dataset cifar10 --method tucker

    # Run on a secondary GPU
    python run_experiment.py --model mobilenetv2 --dataset cifar10 --device cuda:1

    # Run one-shot baseline comparison
    python run_experiment.py --mode oneshot --model resnet18 --dataset cifar10

    # Run full ablation study
    python run_experiment.py --mode ablation --model resnet18 --dataset cifar100

    # Profile all saved models
    python run_experiment.py --mode profile
"""

import argparse
import copy
import os
import sys
import json
import torch
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rsid import RSID
from src.models import get_model, get_feature_layers
from src.decomposition import decompose_model
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger, profile_model
from src.utils.rank_schedule import get_rank_schedule

logger = setup_logger("experiment")


def load_teacher(model_name: str, dataset: str, device: str, checkpoint_dir: str = "./checkpoints") -> torch.nn.Module:
    """
    Load a pre-trained teacher model from checkpoint.

    If no checkpoint exists, creates the model with ImageNet pre-training
    (won't be as good as fine-tuned, but works for quick testing).
    """
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset}.pt")
    model = get_model(model_name, dataset=dataset, pretrained=True, device=device)

    if os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded (val_acc={ckpt.get('val_acc', 'N/A')})")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}. Using pretrained weights only.")
        logger.warning(f"Run train_baseline.py first for best results.")

    return model.eval()


def run_oneshot(
    model_name: str, dataset: str, method: str,
    rank_ratio: float, device: str, data_dir: str,
    checkpoint_dir: str, epochs: int = 50,
) -> dict:
    """
    Run one-shot decomposition + fine-tuning (baseline comparison).

    One-shot decomposes the model in a single step and then fine-tunes
    with knowledge distillation. This is what RSID improves upon.

    Returns:
        Dict with accuracy, profiling metrics, and training history.
    """
    logger.info(f"\n--- One-Shot {method.upper()} (ratio={rank_ratio:.2f}) ---")

    # Load teacher and data
    teacher = load_teacher(model_name, dataset, device, checkpoint_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)
    info = get_dataset_info(dataset)

    # Decompose in one step
    student = copy.deepcopy(teacher)
    student = decompose_model(student, method=method, rank_ratio=rank_ratio, device=device)

    # Fine-tune with KD
    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.01, epochs=epochs, use_amp=True,
        compile_model=True,
    )
    history = trainer.train()
    student = trainer.get_student()

    # Profile
    input_size = (1, 3, info["image_size"], info["image_size"])
    profile = profile_model(student, input_size=input_size, device=device)

    return {
        "method": method,
        "rank_ratio": rank_ratio,
        "val_acc": history["val_acc"][-1],
        "best_val_acc": max(history["val_acc"]),
        "profile": profile,
        "history": history,
    }


def run_rsid(
    model_name: str, dataset: str, method: str,
    schedule_strategy: str, start_ratio: float, end_ratio: float,
    num_iterations: int, device: str, data_dir: str,
    checkpoint_dir: str, epochs_per_stage: int = 50,
) -> dict:
    """
    Run RSID compression with the specified configuration.

    Returns:
        Dict with final accuracy, per-stage profiles, and training history.
    """
    logger.info(f"\n--- RSID {method.upper()} ({schedule_strategy}, {num_iterations} stages) ---")

    teacher = load_teacher(model_name, dataset, device, checkpoint_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)

    rsid = RSID(
        method=method,
        schedule_strategy=schedule_strategy,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        num_iterations=num_iterations,
        epochs_per_stage=epochs_per_stage,
        device=device,
        use_amp=True,
        compile_model=True,
    )

    feature_layers = get_feature_layers(model_name)
    result = rsid.compress(teacher, train_loader, val_loader, feature_layers)

    return {
        "method": method,
        "schedule": schedule_strategy,
        "num_iterations": num_iterations,
        "start_ratio": start_ratio,
        "end_ratio": end_ratio,
        "final_val_acc": result["stage_profiles"][-1].get("val_acc", 0),
        "stage_profiles": result["stage_profiles"],
        "history": [h for h in result["history"]],
        "total_time": result["total_time"],
    }


def run_ablation(
    model_name: str, dataset: str, device: str,
    data_dir: str, checkpoint_dir: str,
) -> dict:
    """
    Run ablation studies: iterations, schedules, losses.

    Ablation 1: Number of iterations (2, 3, 4, 5)
    Ablation 2: Rank schedules (linear, exponential, adaptive)
    Ablation 3: Distillation losses (KD-only, feature-only, combined)
    Ablation 4: Compression ratios (fine-grained sweep)
    """
    results = {"iterations": [], "schedules": [], "ratios": []}

    # --- Ablation 1: Number of iterations ---
    logger.info("\n=== Ablation: Number of Iterations ===")
    for n_iter in [2, 3, 4, 5]:
        r = run_rsid(
            model_name, dataset, "tucker", "exponential",
            start_ratio=0.8, end_ratio=0.2, num_iterations=n_iter,
            device=device, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
            epochs_per_stage=30,  # Fewer epochs for ablation speed
        )
        results["iterations"].append({"n_iterations": n_iter, **r})
        logger.info(f"  {n_iter} iterations: {r['final_val_acc']:.2f}%")

    # --- Ablation 2: Rank schedules ---
    logger.info("\n=== Ablation: Rank Schedules ===")
    for strategy in ["linear", "exponential", "adaptive"]:
        r = run_rsid(
            model_name, dataset, "tucker", strategy,
            start_ratio=0.8, end_ratio=0.2, num_iterations=3,
            device=device, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
            epochs_per_stage=30,
        )
        results["schedules"].append({"strategy": strategy, **r})
        logger.info(f"  {strategy}: {r['final_val_acc']:.2f}%")

    # --- Ablation 3: Compression ratios ---
    logger.info("\n=== Ablation: Compression Ratios ===")
    for end_ratio in [0.5, 0.35, 0.25, 0.15, 0.1]:
        r = run_rsid(
            model_name, dataset, "tucker", "exponential",
            start_ratio=0.8, end_ratio=end_ratio, num_iterations=3,
            device=device, data_dir=data_dir, checkpoint_dir=checkpoint_dir,
            epochs_per_stage=30,
        )
        results["ratios"].append({"end_ratio": end_ratio, **r})
        logger.info(f"  end_ratio={end_ratio}: {r['final_val_acc']:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run RSID experiments")
    parser.add_argument("--mode", type=str, default="rsid",
                        choices=["rsid", "oneshot", "ablation", "profile", "full"])
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet", "imagenette", "imagewoof"])
    parser.add_argument("--method", type=str, default="tucker",
                        choices=["tucker", "cp", "tt", "tr"])
    parser.add_argument("--schedule", type=str, default="exponential",
                        choices=["linear", "exponential", "adaptive"])
    parser.add_argument("--start-ratio", type=float, default=0.8)
    parser.add_argument("--end-ratio", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == "rsid":
        result = run_rsid(
            args.model, args.dataset, args.method, args.schedule,
            args.start_ratio, args.end_ratio, args.iterations,
            args.device, args.data_dir, args.checkpoint_dir,
            epochs_per_stage=args.epochs,
        )
        save_path = os.path.join(
            args.results_dir,
            f"rsid_{args.model}_{args.dataset}_{args.method}_{args.schedule}.json"
        )

    elif args.mode == "oneshot":
        result = run_oneshot(
            args.model, args.dataset, args.method,
            rank_ratio=args.end_ratio,
            device=args.device, data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
        )
        save_path = os.path.join(
            args.results_dir,
            f"oneshot_{args.model}_{args.dataset}_{args.method}_{args.end_ratio}.json"
        )

    elif args.mode == "ablation":
        result = run_ablation(
            args.model, args.dataset,
            args.device, args.data_dir, args.checkpoint_dir,
        )
        save_path = os.path.join(
            args.results_dir,
            f"ablation_{args.model}_{args.dataset}.json"
        )

    elif args.mode == "profile":
        # Profile all saved checkpoints
        result = {}
        for f in os.listdir(args.checkpoint_dir):
            if f.endswith(".pt") and not f.endswith("_history.json"):
                parts = f.replace(".pt", "").split("_")
                model_name = parts[0]
                dataset_name = "_".join(parts[1:])
                try:
                    model = load_teacher(model_name, dataset_name, args.device, args.checkpoint_dir)
                    info = get_dataset_info(dataset_name)
                    input_size = (1, 3, info["image_size"], info["image_size"])
                    profile = profile_model(model, input_size=input_size, device=args.device)
                    result[f] = profile
                    logger.info(f"{f}: {profile['total_params_m']:.2f}M params")
                except Exception as e:
                    logger.warning(f"Failed to profile {f}: {e}")
        save_path = os.path.join(args.results_dir, "profiles.json")

    elif args.mode == "full":
        # Run full comparison: one-shot baselines + RSID for all methods
        result = {"oneshot": {}, "rsid": {}}

        for method in ["tucker", "cp", "tt"]:
            # One-shot baseline
            r = run_oneshot(
                args.model, args.dataset, method,
                rank_ratio=args.end_ratio,
                device=args.device, data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
            )
            result["oneshot"][method] = r

            # RSID
            r = run_rsid(
                args.model, args.dataset, method, args.schedule,
                args.start_ratio, args.end_ratio, args.iterations,
                args.device, args.data_dir, args.checkpoint_dir,
                epochs_per_stage=args.epochs,
            )
            result["rsid"][method] = r

        save_path = os.path.join(
            args.results_dir,
            f"full_{args.model}_{args.dataset}.json"
        )

    # Save results
    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(save_path, "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    logger.info(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
