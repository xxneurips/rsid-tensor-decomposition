"""Paired t-tests on 5-seed data."""
import json
from math import sqrt
from statistics import mean, stdev

with open(r"./results\multiseed_results_5seed.json") as f:
    data = json.load(f)

def per_seed(key, acc_field):
    return {r["seed"]: r[acc_field] for r in data[key]["per_seed"]}

from scipy import stats
from scipy.stats import t as t_dist

configs = [
    ("RN18/C100 Tucker", "rsid_resnet18_cifar100_tucker",   "final_val_acc",
                          "oneshot_resnet18_cifar100_tucker", "best_val_acc"),
    ("RN18/C100 TT",     "rsid_resnet18_cifar100_tt",       "final_val_acc",
                          "oneshot_resnet18_cifar100_tt",     "best_val_acc"),
    ("RN18/C10 TT",      "rsid_resnet18_cifar10_tt",        "final_val_acc",
                          "oneshot_resnet18_cifar10_tt",      "best_val_acc"),
]

print("="*90)
print(f"{'Config':<20} {'RSID mean+-sd':<15} {'OneShot mean+-sd':<15} {'Diff':<8} {'t':<8} {'p_two':<10} {'p_one':<10}")
print("="*90)

for label, rkey, rfield, okey, ofield in configs:
    r = per_seed(rkey, rfield)
    o = per_seed(okey, ofield)
    seeds = sorted(set(r) & set(o))
    rv = [r[s] for s in seeds]
    ov = [o[s] for s in seeds]
    diffs = [rv[i]-ov[i] for i in range(len(rv))]
    rm = mean(rv); rs = stdev(rv)
    om = mean(ov); os_ = stdev(ov)
    md = mean(diffs); sd = stdev(diffs); se = sd/sqrt(len(diffs))
    tval = md/se if se>0 else float('inf')
    df = len(diffs)-1
    res = stats.ttest_rel(rv, ov)
    p_two = res.pvalue
    p_one = 1 - t_dist.cdf(tval, df) if tval > 0 else t_dist.cdf(tval, df)
    print(f"{label:<20} {rm:5.2f}+-{rs:4.2f}   {om:5.2f}+-{os_:4.2f}    {md:+.3f}  {tval:+7.3f}  {p_two:.5f}   {p_one:.5f}")
    print(f"  Seeds: {seeds}")
    print(f"  RSID:    {rv}")
    print(f"  OneShot: {ov}")
    print(f"  Diffs:   {[round(d,3) for d in diffs]}")
    print()
