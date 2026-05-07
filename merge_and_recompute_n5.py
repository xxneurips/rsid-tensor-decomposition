"""
Merge n=2 extra seed results (42, 123) with existing n=3 seed results (3, 7, 11)
into n=5 paired-seed analysis for RN-18 and DeiT-Small on ImageNet-1k.

Run after both .deit_extra_done.flag and .rn18_extra_done.flag are set.
"""
import json
import math
import os

RESULTS = "./results"


def merge(existing, extra):
    """Merge per_seed lists, recompute accuracy stats."""
    merged = list(existing.get("per_seed", []))
    by_seed = {r["seed"]: r for r in merged}
    for r in extra.get("per_seed", []):
        by_seed[r["seed"]] = r
    merged = [by_seed[s] for s in sorted(by_seed)]
    if not merged:
        return existing
    accs = []
    for r in merged:
        acc = r.get("final_stage_best_val_acc")
        if acc is None:
            acc = r.get("best_val_acc", r.get("final_val_acc"))
        if acc is not None:
            accs.append(acc)
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1)
    sd = math.sqrt(var)
    return {
        "per_seed": merged,
        "accuracy": {
            "mean": mean,
            "std": sd,
            "values": accs,
            "n_seeds": len(accs),
        },
    }


def paired_stats(rsid, oneshot):
    rsid_seeds = {r["seed"]: r.get("final_stage_best_val_acc", r.get("best_val_acc")) for r in rsid["per_seed"]}
    one_seeds = {o["seed"]: o.get("best_val_acc") for o in oneshot["per_seed"]}
    common = sorted(set(rsid_seeds) & set(one_seeds))
    diffs = [rsid_seeds[s] - one_seeds[s] for s in common]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(1, n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    t = mean / sem if sem > 0 else float("inf")
    return {
        "seeds": common,
        "diffs": [round(d, 4) for d in diffs],
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "sem": round(sem, 4),
        "t": round(t, 2),
    }


def main():
    print("=== RN-18 / ImageNet-1k merge ===")
    rn18_redo = json.load(open(os.path.join(RESULTS, "imagenet1k_rn18_redo.json")))
    rn18_one = json.load(open(os.path.join(RESULTS, "imagenet1k_suite.json")))
    rn18_extra_path = os.path.join(RESULTS, "imagenet1k_rn18_extra.json")
    if os.path.exists(rn18_extra_path):
        rn18_extra = json.load(open(rn18_extra_path))
        rsid_merged = merge(
            rn18_redo["rsid_resnet18_imagenet"],
            rn18_extra.get("rsid_resnet18_imagenet", {"per_seed": []}),
        )
        one_merged = merge(
            rn18_one["oneshot_resnet18_imagenet"],
            rn18_extra.get("oneshot_resnet18_imagenet", {"per_seed": []}),
        )
        stats = paired_stats(rsid_merged, one_merged)
        print(f"  RSID accuracies: {[r.get('final_stage_best_val_acc', r.get('best_val_acc')) for r in rsid_merged['per_seed']]}")
        print(f"  Oneshot accuracies: {[r.get('best_val_acc') for r in one_merged['per_seed']]}")
        print(f"  RSID mean: {rsid_merged['accuracy']['mean']:.3f} +- {rsid_merged['accuracy']['std']:.3f} (n={rsid_merged['accuracy']['n_seeds']})")
        print(f"  Oneshot mean: {one_merged['accuracy']['mean']:.3f} +- {one_merged['accuracy']['std']:.3f}")
        print(f"  Paired diff: {stats['mean']:+.3f} +- {stats['sd']:.3f} (sem {stats['sem']:.3f}, t={stats['t']}, n={stats['n']}, seeds={stats['seeds']})")
        # Write n=5 merged file
        json.dump({"rsid_resnet18_imagenet": rsid_merged, "oneshot_resnet18_imagenet": one_merged, "paired": stats},
                  open(os.path.join(RESULTS, "imagenet1k_rn18_n5.json"), "w"), indent=2)
    else:
        print(f"  (waiting on {rn18_extra_path})")

    print("\n=== DeiT-Small / ImageNet-1k merge ===")
    deit_main = json.load(open(os.path.join(RESULTS, "deit_small_imagenet1k.json")))
    deit_extra_path = os.path.join(RESULTS, "deit_small_imagenet1k_extra.json")
    if os.path.exists(deit_extra_path):
        deit_extra = json.load(open(deit_extra_path))
        rsid_merged = merge(deit_main["rsid_deit_small_imagenet"],
                            deit_extra.get("rsid_deit_small_imagenet", {"per_seed": []}))
        one_merged = merge(deit_main["oneshot_deit_small_imagenet"],
                           deit_extra.get("oneshot_deit_small_imagenet", {"per_seed": []}))
        stats = paired_stats(rsid_merged, one_merged)
        print(f"  RSID accuracies: {[r.get('final_stage_best_val_acc', r.get('best_val_acc')) for r in rsid_merged['per_seed']]}")
        print(f"  Oneshot accuracies: {[r.get('best_val_acc') for r in one_merged['per_seed']]}")
        print(f"  RSID mean: {rsid_merged['accuracy']['mean']:.3f} +- {rsid_merged['accuracy']['std']:.3f} (n={rsid_merged['accuracy']['n_seeds']})")
        print(f"  Oneshot mean: {one_merged['accuracy']['mean']:.3f} +- {one_merged['accuracy']['std']:.3f}")
        print(f"  Paired diff: {stats['mean']:+.3f} +- {stats['sd']:.3f} (sem {stats['sem']:.3f}, t={stats['t']}, n={stats['n']}, seeds={stats['seeds']})")
        json.dump({"rsid_deit_small_imagenet": rsid_merged, "oneshot_deit_small_imagenet": one_merged, "paired": stats},
                  open(os.path.join(RESULTS, "deit_small_imagenet1k_n5.json"), "w"), indent=2)
    else:
        print(f"  (waiting on {deit_extra_path})")

    print("\n=== Bonferroni at alpha=0.05 across 3 architectures ===")
    print(f"  Alpha-corrected: 0.0167")
    print(f"  Critical t for n=5, df=4 at alpha=0.0167: ~3.747 (from t-tables)")
    print(f"  Critical t for n=3, df=2 at alpha=0.0167: ~5.84")


if __name__ == "__main__":
    main()
