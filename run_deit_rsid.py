"""
DeiT-Tiny + RSID on CIFAR-100  --  transformer generalization experiment.

Pipeline:
  1. Fine-tune DeiT-Tiny (timm, ImageNet-pretrained) on CIFAR-100 at 224x224.
  2. Apply SVD-based low-rank decomposition to attention Q/K/V and MLP Linear layers.
     (Tucker/TT reduce to truncated SVD for 2D weights.)
  3. Compare one-shot SVD (rank ratio 0.3) vs RSID-SVD (3 progressive stages with KD).

SVD decomposer for Linear layer:
    W (out x in)  ->  U_r @ S_r @ V_r^T   with rank r = ceil(rho * min(out, in))
    Replaced by two Linear layers: Linear(in, r) -> Linear(r, out)
    Parameters: in*r + r*out  vs original in*out  --  compression when r < in*out/(in+out).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast

import timm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("deit_rsid")


# ==============================================================
# Data
# ==============================================================
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_loaders(data_dir, batch_size=128, num_workers=4, image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
    val = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ==============================================================
# Model
# ==============================================================
def get_deit_tiny(num_classes=100, pretrained=True):
    model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ==============================================================
# SVD decomposer for Linear layers
# ==============================================================
class LowRankLinear(nn.Module):
    """Replacement for nn.Linear: Linear(in, r) -> Linear(r, out)."""
    def __init__(self, in_features, out_features, rank, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.down = nn.Linear(in_features, rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(rank, out_features, bias=bias, **factory_kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

    def forward(self, x):
        return self.up(self.down(x))


def svd_decompose_linear(linear, rank):
    """Return a LowRankLinear approximating `linear` via truncated SVD."""
    W = linear.weight.data  # shape: (out, in)
    out_features, in_features = W.shape
    rank = min(rank, min(out_features, in_features))

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    lr = LowRankLinear(in_features, out_features, rank,
                       bias=linear.bias is not None,
                       device=W.device, dtype=W.dtype)
    # W = (U_r * S_r) @ Vh_r   --  absorb S into U
    with torch.no_grad():
        lr.down.weight.copy_(Vh_r)          # (r, in)
        lr.up.weight.copy_(U_r * S_r.unsqueeze(0))  # (out, r)
        if linear.bias is not None:
            lr.up.bias.copy_(linear.bias.data)
    return lr


def should_decompose(name, module, min_dim=16):
    """Decide whether a Linear layer is a good decomposition candidate."""
    if not isinstance(module, nn.Linear):
        return False
    if module.in_features < min_dim or module.out_features < min_dim:
        return False
    # Skip the classification head (small, kept dense)
    if name.endswith("head") or name.endswith("fc"):
        return False
    return True


def apply_svd_to_transformer(model, rank_ratio, min_dim=16):
    """Replace eligible Linear layers in-place with low-rank versions."""
    replaced = 0
    total_params_before = 0
    total_params_after = 0
    # Collect candidates first to avoid mutating during iteration
    candidates = []
    for name, module in model.named_modules():
        if should_decompose(name, module, min_dim):
            candidates.append((name, module))

    for name, linear in candidates:
        out_f, in_f = linear.out_features, linear.in_features
        orig_params = out_f * in_f + (out_f if linear.bias is not None else 0)
        rank = max(1, math.ceil(rank_ratio * min(out_f, in_f)))
        new_params = in_f * rank + rank * out_f + (out_f if linear.bias is not None else 0)
        if new_params >= orig_params:
            # No compression possible at this rank  --  skip
            continue
        lr = svd_decompose_linear(linear, rank)
        # Replace in parent module
        parent_path = name.rsplit(".", 1)
        if len(parent_path) == 2:
            parent = model.get_submodule(parent_path[0])
            setattr(parent, parent_path[1], lr)
        else:
            setattr(model, parent_path[0], lr)
        replaced += 1
        total_params_before += orig_params
        total_params_after += new_params

    if total_params_before == 0:
        logger.info(f"  No Linear layers compressible at rank_ratio={rank_ratio} "
                    f"(SVD overhead exceeds savings for all candidates)")
    else:
        logger.info(f"  Decomposed {replaced} Linear layers: {total_params_before:,} -> {total_params_after:,} "
                    f"({100*total_params_after/total_params_before:.1f}% kept)")
    return replaced


# ==============================================================
# Training and KD loss
# ==============================================================
class KDLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, student_logits, teacher_logits, targets):
        ce = F.cross_entropy(student_logits, targets)
        soft_s = F.log_softmax(student_logits / self.T, dim=-1)
        soft_t = F.softmax(teacher_logits / self.T, dim=-1)
        kd = F.kl_div(soft_s, soft_t, reduction="batchmean") * (self.T ** 2)
        return self.alpha * kd + (1 - self.alpha) * ce


def train_epoch(model, teacher, loader, optimizer, loss_fn, scaler, device):
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            s_logits = model(x)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = loss_fn(s_logits, t_logits, y)
            else:
                loss = F.cross_entropy(s_logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (s_logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def fine_tune(model, teacher, train_loader, val_loader, epochs, lr, device, tag=""):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = KDLoss(alpha=0.5, temperature=3.0) if teacher is not None else None
    scaler = GradScaler()
    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, teacher, train_loader, optimizer, loss_fn, scaler, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0
        logger.info(f"  [{tag}] Epoch {epoch+1}/{epochs}  train_loss={tr_loss:.4f} "
                    f"train_acc={tr_acc:.2f}  val_acc={val_acc:.2f}  ({dt:.1f}s)")
        best_acc = max(best_acc, val_acc)
    return best_acc


# ==============================================================
# Main experiments
# ==============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_baseline", "run_compression"], required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10, help="baseline fine-tune epochs")
    parser.add_argument("--kd-epochs", type=int, default=5, help="KD fine-tune epochs per stage")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rank-ratios", type=float, nargs="+", default=[0.8, 0.5, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = args.device
    ckpt_path = os.path.join(args.ckpt_dir, "deit_tiny_cifar100.pt")

    if args.mode == "train_baseline":
        # Fine-tune DeiT-Tiny on CIFAR-100 and save checkpoint
        set_seed(42)
        logger.info("Loading DeiT-Tiny (pretrained) ...")
        model = get_deit_tiny(num_classes=100, pretrained=True).to(device)
        logger.info(f"  params: {count_params(model)/1e6:.2f}M")

        train_loader, val_loader = get_cifar100_loaders(args.data_dir, args.batch_size)
        logger.info(f"Fine-tuning for {args.epochs} epochs at lr={args.lr} ...")
        best_acc = fine_tune(model, teacher=None,
                             train_loader=train_loader, val_loader=val_loader,
                             epochs=args.epochs, lr=args.lr, device=device, tag="BASE")
        torch.save({"model_state_dict": model.state_dict(), "val_acc": best_acc}, ckpt_path)
        logger.info(f"Saved baseline to {ckpt_path} (best val acc = {best_acc:.2f}%)")

    elif args.mode == "run_compression":
        assert os.path.exists(ckpt_path), f"Missing baseline checkpoint {ckpt_path}. Run --mode train_baseline first."
        results = {"configs": []}

        for seed in args.seeds:
            for rr in args.rank_ratios:
                # --- One-shot SVD + KD
                set_seed(seed)
                logger.info(f"\n=== One-shot SVD  ratio={rr}  seed={seed} ===")
                teacher = get_deit_tiny(num_classes=100, pretrained=True).to(device)
                teacher.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["model_state_dict"])
                teacher.eval()
                base_acc = 0.0  # lazy eval below

                student = copy.deepcopy(teacher)
                apply_svd_to_transformer(student, rank_ratio=rr)
                train_loader, val_loader = get_cifar100_loaders(args.data_dir, args.batch_size)
                acc_before_ft = evaluate(student, val_loader, device)
                logger.info(f"  pre-FT acc={acc_before_ft:.2f}%")
                acc_os = fine_tune(student, teacher, train_loader, val_loader,
                                   epochs=args.kd_epochs, lr=args.lr / 3, device=device, tag=f"OS-{rr}")

                # Count effective params
                os_params = count_params(student)

                # --- RSID-SVD (3 stages, fresh decomposition each time, KD from previous student)
                logger.info(f"\n=== RSID-SVD  end_ratio={rr}  seed={seed} ===")
                # Schedule starts at 0.5 (square attention Q/K/V not compressible above ~0.5 due to
                # SVD reparametrization overhead; see apply_svd_to_transformer for details).
                # Exponential schedule: r_1 = 0.5, r_2 = sqrt(0.5 * rr), r_3 = rr
                rs = [0.5, math.sqrt(0.5 * rr), rr]
                current_teacher = teacher
                for stage_idx, ri in enumerate(rs):
                    logger.info(f"  stage {stage_idx+1}/{len(rs)}  rank_ratio={ri:.3f}")
                    set_seed(seed + stage_idx)  # differ per stage slightly
                    s_i = copy.deepcopy(teacher)  # FRESH copy from ORIGINAL
                    apply_svd_to_transformer(s_i, rank_ratio=ri)
                    acc_pre = evaluate(s_i, val_loader, device)
                    logger.info(f"    pre-FT acc={acc_pre:.2f}%")
                    acc_post = fine_tune(s_i, current_teacher, train_loader, val_loader,
                                          epochs=args.kd_epochs, lr=args.lr / 3, device=device, tag=f"RSID-s{stage_idx+1}-{ri:.2f}")
                    current_teacher = s_i  # promote to next teacher

                rsid_acc = evaluate(current_teacher, val_loader, device)
                rsid_params = count_params(current_teacher)

                results["configs"].append({
                    "seed": seed,
                    "rank_ratio": rr,
                    "oneshot_acc": acc_os,
                    "oneshot_params": os_params,
                    "rsid_acc": rsid_acc,
                    "rsid_params": rsid_params,
                })

                out_json = os.path.join(args.results_dir, f"deit_rsid_results.json")
                with open(out_json, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Saved results to {out_json}")

        # Final summary
        print("\n" + "=" * 80)
        print(f"{'rank_ratio':<12} {'seed':<6} {'OS acc':<10} {'OS params':<12} {'RSID acc':<10} {'RSID params':<12}")
        for c in results["configs"]:
            print(f"{c['rank_ratio']:<12.2f} {c['seed']:<6} {c['oneshot_acc']:<10.2f} {c['oneshot_params']/1e6:<12.2f} {c['rsid_acc']:<10.2f} {c['rsid_params']/1e6:<12.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
