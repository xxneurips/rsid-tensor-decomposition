"""
Training-budget control: one-shot Tucker + KD at the FULL RSID budget (150
epochs), 5 seeds, RN18/CIFAR-100. This is the missing ablation that isolates
the contribution of staged compression from extended training.

Comparison axis (RN18/CIFAR-100, Tucker, rank ratio 0.3):
  one-shot @  50 epochs  (existing 5-seed result)  :  79.76 +- 0.14 %
  RSID 3 stages @ 50 ep  (existing 5-seed result)  :  80.46 +- 0.17 %
  one-shot @ 150 epochs  (THIS SCRIPT)             :  ?

If one-shot @ 150 catches up to RSID, the +0.70 pp gain is a training-budget
artefact, not a benefit of staged compression. If one-shot @ 150 plateaus
below RSID, the staged design contributes real value.
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

from src.models import get_model
from src.decomposition import decompose_model
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger, profile_model

logger = setup_logger("budget_control")

SEEDS = [3, 7, 11, 42, 123]


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


def run_seed(model_name, dataset, method, rank_ratio, device, data_dir, ckpt_dir, epochs, seed):
    set_seed(seed)
    logger.info(f"  [Seed {seed}] one-shot {method} ratio={rank_ratio} epochs={epochs}")

    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
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
        compile_model=False,  # avoid first-epoch compile cost over many seeds
    )
    history = trainer.train()
    best_acc = max(history["val_acc"])
    final_acc = history["val_acc"][-1]
    logger.info(f"  [Seed {seed}] best={best_acc:.2f}%  final={final_acc:.2f}%")

    return {
        "seed": seed,
        "best_val_acc": best_acc,
        "final_val_acc": final_acc,
        "epochs": epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training-budget control: matches 3x50 RSID budget")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output-json", default="./results/budget_control.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--method", type=str, default="tucker")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--rank-ratio", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)

    key = f"oneshot_{args.method}_{args.model}_{args.dataset}_e{args.epochs}_r{args.rank_ratio}"
    logger.info(f"\n{'='*60}\n{key}\n{'='*60}")
    if key not in results:
        results[key] = {
            "config": {
                "model": args.model, "dataset": args.dataset,
                "method": args.method, "rank_ratio": args.rank_ratio,
                "epochs": args.epochs,
            },
            "per_seed": [],
        }
    existing = {r["seed"] for r in results[key]["per_seed"]}

    for seed in args.seeds:
        if seed in existing:
            logger.info(f"  Seed {seed} already present  --  skipping")
            continue
        t0 = time.time()
        try:
            r = run_seed(args.model, args.dataset, args.method, args.rank_ratio,
                         args.device, args.data_dir, args.checkpoint_dir,
                         args.epochs, seed)
            results[key]["per_seed"].append(r)
            logger.info(f"  Seed {seed} done in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            logger.error(f"  Seed {seed} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue
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
    print("TRAINING-BUDGET CONTROL SUMMARY")
    print("=" * 80)
    for key, val in results.items():
        if "accuracy" in val:
            agg = val["accuracy"]
            print(f"{key}: {agg['mean']:.2f} +- {agg['std']:.2f}  (N={agg['n_seeds']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
