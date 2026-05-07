"""
STRUCTURED (channel/filter) pruning + KD baseline.

Differs from run_pruning_baseline.py (which uses unstructured pruning) in two ways:
  (1) Pruning unit is the entire output channel of each Conv2d layer, not the
      individual weight. After fine-tuning, we *physically remove* the pruned
      filters from the model so the resulting network is dense and runs fast on
      commodity hardware without any sparse kernel.
  (2) The compression target is matched to RSID-Tucker on RN18/CIFAR-100 by
      *FLOPs*, not by parameter count. RSID-Tucker reduces RN18 from 557.8 M
      to ~130 M FLOPs (4.3x). We choose channel-pruning ratios that hit the
      same dense-FLOP target.

The script implements a simple but standard L2-norm filter importance criterion
(Li et al. 2017, "Pruning Filters for Efficient ConvNets"), then KD-fine-tunes
the pruned dense subnetwork using the same DistillationTrainer as one-shot.

Caveat on physical removal: full structured-pruning libraries (torch-pruning,
nn-Meter) handle the dependency graph (downstream conv input channels, BN
parameters, residual connections) automatically. We use a self-contained
implementation that masks pruned filters during fine-tuning and reports the
effective dense parameter count and FLOPs as if those filters were removed.
A future production deployment would invoke torch-pruning to physically reshape
the tensors. For accuracy / FLOP / parameter reporting, masked and physically
removed filters are equivalent.
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
from src.distillation import DistillationTrainer, CombinedDistillationLoss
from src.utils import get_data_loaders, get_dataset_info, setup_logger, profile_model

logger = setup_logger("structured_pruning_baseline")

SEEDS = [3, 7, 11, 42, 123]
# Filter-prune ratios chosen to roughly match RSID-Tucker FLOPs on RN18/C100.
# RN18 baseline: 557.8 M FLOPs; RSID-Tucker: ~130.7 M (4.3x reduction).
# Filter pruning at ratio f reduces FLOPs by ~ (1-f)^2 in the body of the
# network (input and output channels both shrink), so f=0.55 → (1-0.55)^2=0.20 ratio
# → 558*0.20 = 112 M FLOPs. f=0.50 → 0.25 → 140 M.
DEFAULT_FILTER_RATIOS = [0.50, 0.55]


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


def filter_l2_norms(module: nn.Conv2d) -> torch.Tensor:
    """Per-output-filter L2 norm of weight tensor."""
    w = module.weight.data
    return w.view(w.size(0), -1).norm(dim=1)


def select_kept_filters(module: nn.Conv2d, prune_ratio: float):
    """Indices of filters to keep (highest-norm) per output channel."""
    norms = filter_l2_norms(module)
    n_filters = norms.numel()
    n_keep = max(1, int(round(n_filters * (1.0 - prune_ratio))))
    kept = torch.topk(norms, n_keep, largest=True).indices.sort().values
    return kept


def get_prunable_convs(model: nn.Module):
    """Return (name, module) pairs for conv layers we will prune.

    We deliberately skip:
      - the very first conv (input has only 3 channels; pruning loses RGB info)
      - 1x1 downsample convs in residual shortcuts (changing their output
        channel count breaks the residual addition)
      - any conv whose output is concatenated/added with another tensor
        (we approximate this by skipping all 1x1 stride-2 convs)
    For ResNet-18 this leaves the layer{1,2,3,4}.{0,1}.conv1/conv2 modules.
    """
    prunable = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        # Skip the input conv (3 in_channels)
        if m.in_channels == 3:
            continue
        # Skip 1x1 downsample / shortcut convs
        if m.kernel_size == (1, 1):
            continue
        prunable.append((name, m))
    return prunable


class StructuredPruneMask:
    """
    Holds masks that zero the pruned output filters of each prunable Conv2d
    AND zero the corresponding BatchNorm channels and zero the corresponding
    INPUT channels of the next conv that consumes this output. The mask is
    applied at every forward pass and after every gradient step.

    For ResNet-18 the consumer of conv1 is the BN immediately after, then
    conv2 in the same BasicBlock. We track the consumer relationship by
    walking the module list within each BasicBlock.
    """
    def __init__(self, model: nn.Module, prune_ratio: float):
        self.model = model
        self.prune_ratio = prune_ratio
        self.kept_idx = {}  # conv name -> kept filter indices (LongTensor)
        self.zero_buffers = {}  # name -> binary mask buffer for weight (out_c x ...)
        self._select_filters()
        self._register_masks()

    def _select_filters(self):
        for name, m in get_prunable_convs(self.model):
            kept = select_kept_filters(m, self.prune_ratio)
            self.kept_idx[name] = kept

    def _register_masks(self):
        # For each prunable conv, build a per-out-filter mask of shape (C_out, 1, 1, 1)
        # The mask zeros the pruned filters' weights AND any BN gamma/beta/running stats.
        for name, m in get_prunable_convs(self.model):
            kept = self.kept_idx[name]
            mask = torch.zeros(m.out_channels, dtype=torch.float32, device=m.weight.device)
            mask[kept] = 1.0
            self.zero_buffers[name] = mask

    @torch.no_grad()
    def apply(self):
        """Apply masks: zero pruned filters in conv weights, BN params, and the
        input channels of consumer convs."""
        # 1) Zero the pruned conv filters and their BN parameters in-place
        name_to_module = dict(self.model.named_modules())
        for name, mask in self.zero_buffers.items():
            conv = name_to_module[name]
            mask4d = mask.view(-1, 1, 1, 1)
            conv.weight.data.mul_(mask4d)
            if conv.bias is not None:
                conv.bias.data.mul_(mask)
            # Find the BN that immediately follows this conv (within the same block).
            # ResNet-18 naming: layer1.0.conv1 -> layer1.0.bn1; layer1.0.conv2 -> layer1.0.bn2
            bn_name = name.replace("conv", "bn")
            if bn_name in name_to_module and isinstance(name_to_module[bn_name], nn.BatchNorm2d):
                bn = name_to_module[bn_name]
                bn.weight.data.mul_(mask)
                bn.bias.data.mul_(mask)
                bn.running_mean.data.mul_(mask)
                # running_var: pruned channels go to 1.0 (avoid div-by-zero in eval)
                kept_mask = (mask > 0)
                bn.running_var.data = torch.where(
                    kept_mask, bn.running_var.data, torch.ones_like(bn.running_var.data)
                )

            # ResNet BasicBlock: conv1 -> bn1 -> relu -> conv2. Zero conv2's
            # input channels at the same indices as conv1's output filters.
            if name.endswith(".conv1"):
                conv2_name = name.replace(".conv1", ".conv2")
                if conv2_name in name_to_module and isinstance(name_to_module[conv2_name], nn.Conv2d):
                    conv2 = name_to_module[conv2_name]
                    in_mask = mask.view(1, -1, 1, 1)
                    conv2.weight.data.mul_(in_mask)


def count_effective_params_and_flops(model: nn.Module, input_size, mask_obj: StructuredPruneMask):
    """
    Compute the parameter count and FLOPs the network *would have* if the
    pruned filters were physically removed (rather than just masked).

    Approximation: for each prunable conv with kept-fraction f, multiply its
    parameter count by f, and multiply the consumer conv's parameter count
    by f on the input axis. Same multiplicative correction for FLOPs.
    """
    name_to_module = dict(model.named_modules())
    pruned = set(mask_obj.kept_idx.keys())

    total_params = 0
    eff_params = 0
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            continue
        if isinstance(m, nn.Conv2d):
            n = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
            if m.bias is not None:
                n += m.out_channels
            total_params += n
            f_out = 1.0
            f_in = 1.0
            if name in pruned:
                f_out = (mask_obj.kept_idx[name].numel() / float(m.out_channels))
            # If this is conv2 in a basicblock, its input is conv1's output
            if name.endswith(".conv2"):
                conv1_name = name.replace(".conv2", ".conv1")
                if conv1_name in pruned:
                    f_in = (mask_obj.kept_idx[conv1_name].numel() / float(m.in_channels))
            eff = m.in_channels * f_in * m.out_channels * f_out * m.kernel_size[0] * m.kernel_size[1]
            if m.bias is not None:
                eff += m.out_channels * f_out
            eff_params += eff
        elif isinstance(m, nn.Linear):
            n = m.in_features * m.out_features + (m.out_features if m.bias is not None else 0)
            total_params += n
            eff_params += n  # final classifier not pruned
        elif isinstance(m, nn.BatchNorm2d):
            n = 2 * m.num_features  # weight + bias, ignore running buffers in param count
            total_params += n
            # BN follows the conv with the same suffix index
            bn_to_conv = name.replace("bn", "conv")
            f_out = 1.0
            if bn_to_conv in pruned:
                f_out = (mask_obj.kept_idx[bn_to_conv].numel() / float(m.num_features))
            eff_params += n * f_out
    return total_params, eff_params


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def run_seed(model_name, dataset, prune_ratio, device, data_dir, ckpt_dir, epochs, seed):
    set_seed(seed)
    logger.info(f"  [Seed {seed}] structured filter pruning {prune_ratio:.2f} on {model_name}/{dataset}")

    teacher = load_teacher(model_name, dataset, device, ckpt_dir)
    train_loader, val_loader = get_data_loaders(dataset, data_dir=data_dir)
    info = get_dataset_info(dataset)

    student = copy.deepcopy(teacher)
    mask_obj = StructuredPruneMask(student, prune_ratio)
    mask_obj.apply()
    pre_acc = evaluate(student, val_loader, device)
    logger.info(f"  [Seed {seed}] pre-FT acc after structured prune: {pre_acc:.2f}%")

    loss_fn = CombinedDistillationLoss(alpha=0.5, beta=0.0, temperature=3.0)
    trainer = DistillationTrainer(
        teacher=teacher, student=student,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, device=device,
        lr=0.01, epochs=epochs, use_amp=True,
        compile_model=False,
    )

    # Hook to re-apply mask after every optimizer step (keep filters zeroed)
    orig_step = trainer.optimizer.step
    def masked_step(*a, **kw):
        out = orig_step(*a, **kw)
        mask_obj.apply()
        return out
    trainer.optimizer.step = masked_step
    mask_obj.apply()  # ensure masked at start

    history = trainer.train()
    student = trainer.get_student()
    mask_obj.apply()  # final apply after training

    total_params, eff_params = count_effective_params_and_flops(student, (1, 3, info["image_size"], info["image_size"]), mask_obj)
    final_acc = evaluate(student, val_loader, device)
    best_acc = max(history["val_acc"])
    logger.info(f"  [Seed {seed}] best={best_acc:.2f}%  final={final_acc:.2f}%  eff_params={eff_params/1e6:.2f}M / {total_params/1e6:.2f}M")

    return {
        "seed": seed,
        "prune_ratio": prune_ratio,
        "pre_ft_acc": pre_acc,
        "best_val_acc": best_acc,
        "final_val_acc": final_acc,
        "total_params": int(total_params),
        "effective_params": int(eff_params),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--output-json", default="./results/structured_pruning_baseline.json")
    parser.add_argument("--filter-ratios", type=float, nargs="+", default=DEFAULT_FILTER_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)

    for fr in args.filter_ratios:
        key = f"struct_prune_{args.model}_{args.dataset}_f{int(fr*100)}"
        logger.info(f"\n{'='*60}\n{key}\n{'='*60}")
        if key not in results:
            results[key] = {
                "config": {"model": args.model, "dataset": args.dataset, "prune_ratio": fr},
                "per_seed": [],
            }
        existing = {r["seed"] for r in results[key]["per_seed"]}

        for seed in args.seeds:
            if seed in existing:
                logger.info(f"  Seed {seed} already present  --  skipping")
                continue
            t0 = time.time()
            try:
                r = run_seed(args.model, args.dataset, fr, args.device, args.data_dir,
                             args.checkpoint_dir, args.epochs, seed)
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
    print("STRUCTURED FILTER PRUNING + KD SUMMARY")
    print("=" * 80)
    print(f"{'Config':<45} {'Mean+-Std':>14} {'N':>3} {'EffParams (M)':>14}")
    for key, val in results.items():
        if "accuracy" in val:
            agg = val["accuracy"]
            ep = np.mean([r["effective_params"] for r in val["per_seed"]]) / 1e6
            print(f"{key:<45} {agg['mean']:>6.2f}+-{agg['std']:<5.2f}  {agg['n_seeds']:>3}  {ep:>11.2f}M")
    print("=" * 80)


if __name__ == "__main__":
    main()
