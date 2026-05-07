"""
Singular-value decay analysis: Conv2d (RN-18) vs Linear (DeiT-Small).

For each decomposable layer, compute the SVD of the unfolded weight matrix and
report (i) the cumulative-energy curve and (ii) the rank ratio needed to capture
{50,75,90,95,99}% of the spectral energy.

Produces:
  - ./results/sv_decay.json
  - ./sv_decay_fig.pdf
  - ./sv_decay_fig.png

The hypothesis under test (Appendix E, Hypothesis 2): if Conv2d weights have
faster spectral decay than Linear weights, then Conv2d is more "low-rank
friendly" and would benefit more from staging.
"""
import os
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_JSON = "./results/sv_decay.json"
OUT_PDF = "./sv_decay_fig.pdf"
OUT_PNG = "./sv_decay_fig.png"


def conv_unfold(weight):
    """Conv2d weight shape (C_out, C_in, k, k). Unfold to (C_out, C_in*k*k)."""
    Co, Ci, kH, kW = weight.shape
    return weight.reshape(Co, Ci * kH * kW)


def normalised_cumsum(s):
    """Cumulative energy from singular values."""
    energy = s ** 2
    cum = np.cumsum(energy)
    return cum / cum[-1]


def rank_for_energy(s, threshold):
    """Smallest rank k such that cumulative energy >= threshold * total."""
    cum = normalised_cumsum(s)
    idx = int(np.searchsorted(cum, threshold))
    return (idx + 1) / len(s)  # rank ratio


def analyse_resnet18():
    """Pretrained RN-18  --  extract Conv2d weights and SVD them."""
    import torchvision.models as M
    model = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.weight.shape[2] > 1:
            # Skip 1x1 convs (no spatial structure to compress)
            W = mod.weight.detach().cpu().float().numpy()
            unf = conv_unfold(torch.from_numpy(W)).numpy()
            U, s, Vt = np.linalg.svd(unf, full_matrices=False)
            layers.append({
                "name": name,
                "shape": list(W.shape),
                "rows": unf.shape[0],
                "cols": unf.shape[1],
                "rank_at_50": rank_for_energy(s, 0.50),
                "rank_at_75": rank_for_energy(s, 0.75),
                "rank_at_90": rank_for_energy(s, 0.90),
                "rank_at_95": rank_for_energy(s, 0.95),
                "rank_at_99": rank_for_energy(s, 0.99),
                "sv_normalized": (s / s[0]).tolist(),
            })
    return layers


def analyse_deit_small():
    """timm DeiT-Small  --  extract Q, K, V, MLP Linear weights and SVD them."""
    import timm
    model = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=1000)
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            # The Q/K/V are folded into one linear in timm DeiT (qkv), separate them
            W = mod.weight.detach().cpu().float().numpy()
            if W.shape[0] < 32 or W.shape[1] < 32:
                continue  # skip tiny linear layers
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            layers.append({
                "name": name,
                "shape": list(W.shape),
                "rows": W.shape[0],
                "cols": W.shape[1],
                "rank_at_50": rank_for_energy(s, 0.50),
                "rank_at_75": rank_for_energy(s, 0.75),
                "rank_at_90": rank_for_energy(s, 0.90),
                "rank_at_95": rank_for_energy(s, 0.95),
                "rank_at_99": rank_for_energy(s, 0.99),
                "sv_normalized": (s / s[0]).tolist(),
            })
    return layers


def aggregate(layers, name):
    """Median rank-for-energy across layers, with min/max."""
    keys = ["rank_at_50", "rank_at_75", "rank_at_90", "rank_at_95", "rank_at_99"]
    out = {"name": name, "n_layers": len(layers)}
    for k in keys:
        vals = [l[k] for l in layers]
        out[k] = {
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def main():
    print("Analyzing ResNet-18 Conv2d weights...")
    rn18_layers = analyse_resnet18()
    print(f"  {len(rn18_layers)} Conv2d layers analysed (excluding 1x1)")

    print("Analyzing DeiT-Small Linear weights...")
    deit_layers = analyse_deit_small()
    print(f"  {len(deit_layers)} Linear layers analysed")

    rn18_agg = aggregate(rn18_layers, "ResNet-18 (Conv2d)")
    deit_agg = aggregate(deit_layers, "DeiT-Small (Linear)")

    out = {
        "resnet18_conv2d": rn18_layers,
        "deit_small_linear": deit_layers,
        "summary": {
            "resnet18_conv2d": rn18_agg,
            "deit_small_linear": deit_agg,
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary: median rank ratio needed for X% spectral energy ===")
    print(f"{'energy':<8} {'RN-18 Conv2d':<18} {'DeiT Linear':<18}")
    for k, label in [("rank_at_50", "50%"), ("rank_at_75", "75%"),
                     ("rank_at_90", "90%"), ("rank_at_95", "95%"),
                     ("rank_at_99", "99%")]:
        print(f"{label:<8} {rn18_agg[k]['median']:<18.3f} {deit_agg[k]['median']:<18.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # Average normalized SV curve per family (after interpolating to common length)
    for layers, color, label in [(rn18_layers, "#1f77b4", "ResNet-18 Conv2d (n=%d)" % len(rn18_layers)),
                                  (deit_layers, "#2ca02c", "DeiT-Small Linear (n=%d)" % len(deit_layers))]:
        # Each layer has different SV count; interpolate to fraction of rank
        x_grid = np.linspace(0, 1, 100)
        curves = []
        for l in layers:
            sv = np.array(l["sv_normalized"])
            x = np.linspace(0, 1, len(sv))
            curves.append(np.interp(x_grid, x, sv))
        curves = np.array(curves)
        median = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25, axis=0)
        q75 = np.percentile(curves, 75, axis=0)
        ax.plot(x_grid, median, color=color, label=label, linewidth=2)
        ax.fill_between(x_grid, q25, q75, color=color, alpha=0.2)

    ax.set_xlabel("Rank ratio (fraction of full rank)")
    ax.set_ylabel("Normalized singular value $\\sigma_i / \\sigma_1$")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-3, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Singular value spectra: Conv2d vs.\\ Linear")
    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=180)

    print(f"\nWrote: {OUT_JSON}\nWrote: {OUT_PDF}\nWrote: {OUT_PNG}")


if __name__ == "__main__":
    main()
