"""
Baseline Model Training Script
================================

Trains or loads pre-trained baseline models for all architectures and datasets.
These baselines serve as the teacher models for RSID and as comparison points.

Usage:
    # Train all baselines
    python train_baseline.py --all

    # Train specific model + dataset
    python train_baseline.py --model resnet18 --dataset cifar10

    # Use specific GPU
    python train_baseline.py --model resnet18 --dataset cifar10 --device cuda:1
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import time
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import get_model
from src.utils import get_data_loaders, setup_logger, profile_model

logger = setup_logger("baseline")

# Training configs per dataset
# CIFAR needs more epochs (training from modified pretrained)
# TinyImageNet uses pretrained features more directly
TRAIN_CONFIGS = {
    "cifar10": {"epochs": 100, "lr": 0.01, "batch_size": 128},
    "cifar100": {"epochs": 100, "lr": 0.01, "batch_size": 128},
    "tinyimagenet": {"epochs": 80, "lr": 0.01, "batch_size": 64},
    # Imagenette/Imagewoof: real ImageNet resolution (224x224). ImageNet-pretrained
    # backbone with only the 10-class classifier head swapped, so 30 epochs of fine-tuning
    # at a lower LR is enough.
    "imagenette": {"epochs": 30, "lr": 0.005, "batch_size": 64},
    "imagewoof": {"epochs": 30, "lr": 0.005, "batch_size": 64},
    # ImageNet-100: 100-class subset of ImageNet at 224x224, ~1300 imgs/class.
    # Backbone is ImageNet-pretrained, only the classifier is new  --  15 epochs is plenty.
    "imagenet100": {"epochs": 15, "lr": 0.005, "batch_size": 96},
}


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 0.01,
    device: str = "cuda",
    save_path: str = "checkpoints/model.pt",
    use_amp: bool = True,
) -> dict:
    """
    Train a model from scratch (or fine-tune from pretrained).

    Uses:
    - SGD + Nesterov momentum (standard for image classification)
    - Cosine annealing LR schedule
    - AMP for faster mixed-precision training
    - cuDNN benchmark mode for auto-tuned conv kernels

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        device: CUDA device.
        save_path: Path to save the best checkpoint.
        use_amp: Enable mixed precision training.

    Returns:
        Dict with training history and best accuracy.
    """
    torch.backends.cudnn.benchmark = True

    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device) if use_amp else None

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(epochs):
        # === Training ===
        model.train()
        total_loss = 0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        # === Validation ===
        model.eval()
        correct = total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast(device, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        val_acc = 100.0 * correct / total
        val_loss = val_loss_sum / total

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Best: {best_acc:.2f}%"
            )

    return {"history": history, "best_acc": best_acc}


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"])
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet", "imagenette", "imagewoof", "imagenet100"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--all", action="store_true", help="Train all model+dataset combinations")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    if args.all:
        # Train all combinations
        models = ["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"]
        datasets = ["cifar10", "cifar100", "tinyimagenet"]
        combos = [(m, d) for m in models for d in datasets]
    else:
        combos = [(args.model, args.dataset)]

    for model_name, dataset_name in combos:
        save_path = os.path.join(args.checkpoint_dir, f"{model_name}_{dataset_name}.pt")

        # Skip if checkpoint already exists
        if os.path.exists(save_path):
            logger.info(f"\nSkipping {model_name}/{dataset_name}  --  checkpoint exists: {save_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name} on {dataset_name}")
        logger.info(f"{'='*50}")

        config = TRAIN_CONFIGS[dataset_name]

        # Load data
        train_loader, val_loader = get_data_loaders(
            dataset=dataset_name,
            batch_size=config["batch_size"],
            data_dir=args.data_dir,
        )

        # Load model (with ImageNet pretrained weights)
        model = get_model(model_name, dataset=dataset_name, pretrained=True, device=args.device)

        # Train
        start = time.time()
        results = train_model(
            model, train_loader, val_loader,
            epochs=config["epochs"],
            lr=config["lr"],
            device=args.device,
            save_path=save_path,
            use_amp=not args.no_amp,
        )
        elapsed = time.time() - start

        logger.info(
            f"\nDone: {model_name}/{dataset_name} | "
            f"Best Acc: {results['best_acc']:.2f}% | "
            f"Time: {elapsed/60:.1f} min"
        )

        # Save training history
        history_path = save_path.replace(".pt", "_history.json")
        with open(history_path, "w") as f:
            json.dump(results["history"], f)


if __name__ == "__main__":
    main()
