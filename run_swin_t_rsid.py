"""
Swin-Tiny + RSID on CIFAR-100  --  second transformer architecture
to harden the cross-architecture asymmetry claim in the NeurIPS submission.

Reuses run_deit_rsid.py's machinery (SVD-on-Linear works for any transformer
with nn.Linear QKV/MLP layers  --  Swin-T qualifies). Differences from DeiT-T:
  - timm model: swin_tiny_patch4_window7_224 (~28M params vs DeiT-T's 5.7M)
  - Uses different ckpt path so it doesn't collide with deit-tiny baseline.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import timm

# Reuse all utilities from the DeiT script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_deit_rsid as base


def get_swin_tiny(num_classes=100, pretrained=True):
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )


# Monkey-patch the loader so apply_svd_to_transformer + fine_tune all work
base.get_deit_tiny = get_swin_tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_baseline", "run_compression", "all"], required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10, help="baseline fine-tune epochs")
    parser.add_argument("--kd-epochs", type=int, default=5, help="KD fine-tune epochs per stage")
    parser.add_argument("--lr", type=float, default=0.001)  # Swin tolerates smaller lr
    parser.add_argument("--batch-size", type=int, default=64)  # Swin is heavier than DeiT-T
    parser.add_argument("--rank-ratios", type=float, nargs="+", default=[0.5, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = args.device
    ckpt_path = os.path.join(args.ckpt_dir, "swin_tiny_cifar100.pt")

    if args.mode in ("train_baseline", "all"):
        base.set_seed(42)
        base.logger.info("Loading Swin-Tiny (pretrained ImageNet-1k) ...")
        model = get_swin_tiny(num_classes=100, pretrained=True).to(device)
        base.logger.info(f"  params: {base.count_params(model)/1e6:.2f}M")
        train_loader, val_loader = base.get_cifar100_loaders(args.data_dir, args.batch_size)
        base.logger.info(f"Fine-tuning for {args.epochs} epochs at lr={args.lr} ...")
        best_acc = base.fine_tune(
            model, teacher=None,
            train_loader=train_loader, val_loader=val_loader,
            epochs=args.epochs, lr=args.lr, device=device, tag="BASE",
        )
        torch.save({"model_state_dict": model.state_dict(), "val_acc": best_acc}, ckpt_path)
        base.logger.info(f"Saved baseline to {ckpt_path} (best val acc = {best_acc:.2f}%)")
        del model
        torch.cuda.empty_cache()

    if args.mode in ("run_compression", "all"):
        assert os.path.exists(ckpt_path), f"Missing baseline checkpoint {ckpt_path}."
        results = {"configs": []}

        for seed in args.seeds:
            for rr in args.rank_ratios:
                # --- One-shot SVD + KD
                base.set_seed(seed)
                base.logger.info(f"\n=== One-shot SVD  ratio={rr}  seed={seed} ===")
                teacher = get_swin_tiny(num_classes=100, pretrained=True).to(device)
                teacher.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"]
                )
                teacher.eval()

                student = copy.deepcopy(teacher)
                base.apply_svd_to_transformer(student, rank_ratio=rr)
                train_loader, val_loader = base.get_cifar100_loaders(args.data_dir, args.batch_size)
                acc_before_ft = base.evaluate(student, val_loader, device)
                base.logger.info(f"  pre-FT acc={acc_before_ft:.2f}%")
                acc_os = base.fine_tune(
                    student, teacher, train_loader, val_loader,
                    epochs=args.kd_epochs, lr=args.lr / 3, device=device, tag=f"OS-{rr}",
                )
                os_params = base.count_params(student)

                # --- RSID-SVD (3 stages, fresh decomposition, KD from prev student)
                base.logger.info(f"\n=== RSID-SVD  end_ratio={rr}  seed={seed} ===")
                rs = [0.5, math.sqrt(0.5 * rr), rr]
                current_teacher = teacher
                for stage_idx, ri in enumerate(rs):
                    base.logger.info(f"  stage {stage_idx+1}/{len(rs)}  rank_ratio={ri:.3f}")
                    base.set_seed(seed + stage_idx)
                    s_i = copy.deepcopy(teacher)
                    base.apply_svd_to_transformer(s_i, rank_ratio=ri)
                    acc_pre = base.evaluate(s_i, val_loader, device)
                    base.logger.info(f"    pre-FT acc={acc_pre:.2f}%")
                    acc_post = base.fine_tune(
                        s_i, current_teacher, train_loader, val_loader,
                        epochs=args.kd_epochs, lr=args.lr / 3, device=device,
                        tag=f"RSID-s{stage_idx+1}-{ri:.2f}",
                    )
                    current_teacher = s_i

                rsid_acc = base.evaluate(current_teacher, val_loader, device)
                rsid_params = base.count_params(current_teacher)

                results["configs"].append({
                    "seed": seed,
                    "rank_ratio": rr,
                    "oneshot_acc": acc_os,
                    "oneshot_params": os_params,
                    "rsid_acc": rsid_acc,
                    "rsid_params": rsid_params,
                    "delta_pp": rsid_acc - acc_os,
                })
                base.logger.info(
                    f"  [seed={seed} rr={rr}] OS={acc_os:.2f}%  RSID={rsid_acc:.2f}%  "
                    f"delta={rsid_acc - acc_os:+.2f}pp"
                )

                del teacher, student, current_teacher
                torch.cuda.empty_cache()

        out = os.path.join(args.results_dir, "swin_t_rsid_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        base.logger.info(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
