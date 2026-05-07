"""
Action 4: Mechanism experiment.

For each RSID stage rank ratio r_i, compute the per-layer Frobenius
approximation error of decomposing the original ResNet-18 weights at r_i.
Aggregate across layers (weighted by parameter count) and compare to the
observed val-accuracy drop after stage i (from imagenet1k_suite.json).

Outputs:
  results/mechanism_rn18_imagenet.json
  scaling_law_mechanism.pdf

Usage (on pod):
  python run_mechanism_experiment.py --device cuda:0 \
      --output-json results/mechanism_rn18_imagenet.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import get_model

tl.set_backend("pytorch")


@torch.no_grad()
def per_layer_tucker_error(weight, rank_ratio, device):
    """Return relative Frobenius error of Tucker decomposition at the given
    rank ratio. Reconstructs W_hat and computes ||W - W_hat||_F / ||W||_F."""
    weight = weight.to(device)
    c_out, c_in = weight.shape[0], weight.shape[1]
    rank_out = max(1, int(c_out * rank_ratio))
    rank_in = max(1, int(c_in * rank_ratio))
    if rank_out >= c_out and rank_in >= c_in:
        return 0.0
    try:
        result = partial_tucker(weight, modes=[0, 1], rank=[rank_out, rank_in],
                                 init="svd", n_iter_max=0)
        core, factors = result[0][0], result[0][1]
        U_out, U_in = factors[0], factors[1]
        # Reconstruct W_hat by contracting core with factors along modes 0,1.
        # core shape: (r_out, r_in, kh, kw)
        # W_hat[i,j,kh,kw] = sum_p sum_q U_out[i,p] * core[p,q,kh,kw] * U_in[j,q]
        W_hat = torch.einsum("ip,pqhw,jq->ijhw", U_out, core, U_in)
        num = torch.linalg.norm(weight - W_hat).item()
        den = torch.linalg.norm(weight).item() + 1e-12
        return num / den
    except Exception as e:
        print(f"  WARN: tucker error compute failed: {e}", flush=True)
        return None


def collect_conv_weights(model):
    """Yield (name, conv_layer) for all decomposable Conv2d layers in the model.
    We mirror the criteria in src/decomposition/decompose_model.py: skip 1x1
    convs at non-stride positions, skip groups>1 (depthwise), skip very small
    layers."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups > 1:
            continue
        if module.weight.shape[1] < 8 or module.weight.shape[0] < 8:
            continue
        # Skip stem convs (input has 3 channels) -- consistent with decompose_model
        if module.weight.shape[1] == 3:
            continue
        yield name, module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--rank-schedule", type=float, nargs="+",
                    default=[0.5768998281229634, 0.41601676461038084, 0.3])
    ap.add_argument("--output-json",
                    default="./results/mechanism_rn18_imagenet.json")
    ap.add_argument("--imagenet-suite-json",
                    default="./results/imagenet1k_suite.json")
    args = ap.parse_args()

    print(f"=== Loading {args.model} pretrained on ImageNet ===", flush=True)
    model = get_model(args.model, dataset="imagenet", pretrained=True,
                      device=args.device)
    model.eval()

    layers = list(collect_conv_weights(model))
    print(f"  found {len(layers)} decomposable Conv2d layers", flush=True)

    # Per-layer error at each schedule ratio
    print(f"\n=== Computing per-layer Frobenius errors ===", flush=True)
    layer_errors = {}  # {layer_name: {rank: rel_err}}
    layer_params = {}
    for name, conv in layers:
        layer_errors[name] = {}
        c_out, c_in, kh, kw = conv.weight.shape
        layer_params[name] = c_out * c_in * kh * kw
        for r in args.rank_schedule:
            err = per_layer_tucker_error(conv.weight.data, r, args.device)
            layer_errors[name][f"{r:.4f}"] = err
        # Print compactly
        errs = [layer_errors[name][f"{r:.4f}"] for r in args.rank_schedule]
        print(f"  {name:30s} c=({c_out},{c_in}) "
              f"errs@{args.rank_schedule}: {[f'{e:.3f}' if e is not None else 'NA' for e in errs]}", flush=True)

    # Param-count weighted mean error per stage
    print(f"\n=== Aggregating ===", flush=True)
    total_params = sum(layer_params.values())
    stage_summary = []
    for r in args.rank_schedule:
        weighted_err = 0.0
        max_err = 0.0
        for name, err_dict in layer_errors.items():
            err = err_dict[f"{r:.4f}"]
            if err is None:
                continue
            w = layer_params[name] / total_params
            weighted_err += w * err
            max_err = max(max_err, err)
        stage_summary.append({
            "rank_ratio": r,
            "param_weighted_mean_error": weighted_err,
            "max_layer_error": max_err,
        })
        print(f"  rank={r:.3f}  pw_mean_err={weighted_err:.4f}  max_err={max_err:.4f}", flush=True)

    # Compare to observed accuracy chain (read from imagenet1k_suite.json)
    obs = {}
    if os.path.exists(args.imagenet_suite_json):
        with open(args.imagenet_suite_json) as f:
            d = json.load(f)
        # Mean across seeds:
        if "rsid_resnet18_imagenet" in d:
            rsid = d["rsid_resnet18_imagenet"]["accuracy"]["mean"]
            teacher = d["rsid_resnet18_imagenet"]["per_seed"][0]["teacher_acc"]
            obs["rsid_final_acc"] = rsid
            obs["teacher_acc"] = teacher
            obs["rsid_drop_from_teacher_pp"] = teacher - rsid
        if "oneshot_resnet18_imagenet" in d:
            obs["oneshot_final_acc"] = d["oneshot_resnet18_imagenet"]["accuracy"]["mean"]
            obs["oneshot_drop_from_teacher_pp"] = obs.get("teacher_acc", 0) - obs["oneshot_final_acc"]
    print(f"  obs: {obs}", flush=True)

    out = {
        "config": {
            "model": args.model,
            "rank_schedule": args.rank_schedule,
            "n_decomposable_layers": len(layers),
            "total_params": total_params,
        },
        "per_layer_errors": layer_errors,
        "per_stage_summary": stage_summary,
        "observed": obs,
        # Proposition 1 prediction: at the final stage, both fresh-restart
        # and one-shot incur the same single-stage approximation error.
        # Re-decomposition would compound: prod(1+eps_i) - 1.
        "prop1_predictions": {
            "fresh_final_relative_error": stage_summary[-1]["param_weighted_mean_error"],
            "redecompose_compound_bound": float(np.prod(
                [1.0 + s["param_weighted_mean_error"] for s in stage_summary]) - 1.0),
        },
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}", flush=True)
    print(f"Prop 1: fresh ε_final = {out['prop1_predictions']['fresh_final_relative_error']:.4f}")
    print(f"Prop 1: re-decomp ε ≤ {out['prop1_predictions']['redecompose_compound_bound']:.4f}")


if __name__ == "__main__":
    main()
