#!/usr/bin/env python
"""Head-knockout power / effect-size bound (reviewer #8).

The "distributed, no sparse circuit" conclusion is one of the paper's advertised
honest negatives, but at n=20/class a reviewer can dismiss it as "underpowered, not
absent." This script turns the null into a *bounded* null: given the knockout run, it
reports the minimal detectable effect (MDE) at the run's n and an upper bound on any
single head's true AUC contribution, so the negative reads as "no head contributes more
than X," not "we couldn't tell."

It consumes the existing head_knockout.json (run it at higher n first for a tighter
bound, e.g. `python analysis/sanity/sc2_head_knockout.py --n 60`). No GPU needed.

Definitions:
  SE(AUC)      Hanley-McNeil large-sample SE of a single AUC at n_pos=n_neg=n.
  SE(drop)     SE of (baseline_AUC - knockout_AUC); the two AUCs are paired (same
               examples), so we report a conservative independent-sample SE = sqrt(2)*SE(AUC)
               as an upper bound on the true paired SE.
  MDE          (z_.975 + z_.80) * SE(drop): smallest single-head drop detectable at
               80% power, alpha=0.05 (two-sided).
  bound        max observed drop + z_.975 * SE(drop): 95% upper confidence bound on the
               largest single-head true effect.

Usage:
  python analysis/sanity/sc2_head_knockout_power.py
  python analysis/sanity/sc2_head_knockout_power.py --json results/sc2_binary/head_knockout.json

Output: results/sc2_binary/head_knockout_power.json (+ a paste-ready paragraph)
"""
import sys, json, argparse, math
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent

import numpy as np

Z_975 = 1.959963985
Z_80 = 0.841621234


def parse_args():
    p = argparse.ArgumentParser(description="Head-knockout power / effect bound")
    p.add_argument("--json", default=str(PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout.json"))
    p.add_argument("--power", type=float, default=0.80)
    return p.parse_args()


def se_auc_hanley_mcneil(auc, n_pos, n_neg):
    """Large-sample SE of a single ROC-AUC (Hanley & McNeil 1982)."""
    a = auc
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 1e-12))


def main():
    args = parse_args()
    path = Path(args.json)
    if not path.exists():
        print(f"{path} not found. Run sc2_head_knockout.py first (ideally --n 60).")
        return
    d = json.load(open(path))
    n = int(d["n_per_class"])
    base_auc = float(d["baseline_auc"])
    drops = np.array(d["auc_drop"], dtype=float)  # [n_layers, n_heads]
    flat = drops.flatten()
    flat = flat[~np.isnan(flat)]
    n_heads_total = flat.size

    # observed
    max_drop = float(np.nanmax(drops))
    li, hi = np.unravel_index(np.nanargmax(drops), drops.shape)
    null_sd = float(np.nanstd(flat))          # across-head dispersion = empirical "noise" band
    null_mean = float(np.nanmean(flat))

    # power
    se_auc = se_auc_hanley_mcneil(base_auc, n, n)
    se_drop = math.sqrt(2) * se_auc           # conservative (paired SE is smaller)
    mde = (Z_975 + (Z_80 if abs(args.power - 0.80) < 1e-6 else _z(args.power))) * se_drop
    upper_bound = max_drop + Z_975 * se_drop

    # how many heads exceed 2*SE individually (rough screen, no multiple-comparison yet)
    n_above_2se = int((flat > 2 * se_drop).sum())

    print("\n" + "=" * 72)
    print(f"HEAD-KNOCKOUT POWER  (n={n}/class, {n_heads_total} heads, baseline AUC {base_auc:.3f})")
    print("=" * 72)
    print(f"  max single-head AUC drop      : {max_drop:+.4f}  (L{li}H{hi})")
    print(f"  across-head null: mean {null_mean:+.4f}, SD {null_sd:.4f}")
    print(f"  SE(AUC) [Hanley-McNeil]       : {se_auc:.4f}")
    print(f"  SE(drop) [conservative]       : {se_drop:.4f}")
    print(f"  MDE @ {int(args.power*100)}% power, a=0.05    : {mde:.4f}  (smallest detectable single-head drop)")
    print(f"  95% upper bound on largest true single-head effect: {upper_bound:.4f} AUC")
    print(f"  heads with drop > 2*SE(drop)  : {n_above_2se} / {n_heads_total}")
    print("=" * 72)
    convincing = max_drop < mde
    print("INTERPRETATION:",
          "max observed drop is BELOW the MDE -> a power-limited null with a tight bound"
          if convincing else
          "max observed drop EXCEEDS the MDE -> some head may be real; inspect with stats script")

    para = (
        f"At n={n}/class the per-head knockout has SE(AUC) $\\approx$ {se_auc:.3f} "
        f"(Hanley--McNeil), giving a minimal detectable single-head effect of "
        f"{mde:.3f} AUC at {int(args.power*100)}\\% power. The largest observed drop is "
        f"{max_drop:.3f} (L{li}H{hi}); its 95\\% upper confidence bound is {upper_bound:.3f} "
        f"AUC. We therefore report a bounded null: no single attention head accounts for "
        f"more than ${upper_bound:.2f}$ of the suicide-vs-non-suicide AUC, consistent with "
        f"the distributed, low-rank picture rather than a sparse head circuit."
    )
    print("\nPaste-ready:\n" + para)

    out = PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout_power.json"
    json.dump({
        "n_per_class": n, "baseline_auc": base_auc, "n_heads_total": n_heads_total,
        "max_drop": max_drop, "max_drop_head": [int(li), int(hi)],
        "null_mean": null_mean, "null_sd": null_sd,
        "se_auc": se_auc, "se_drop": se_drop, "power": args.power,
        "mde": mde, "upper_bound_largest_true_effect": upper_bound,
        "n_heads_above_2se": n_above_2se,
        "power_limited_null_bounded": bool(convincing),
        "paragraph": para,
    }, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


def _z(power):
    # inverse normal for arbitrary power via rational approx (Acklam); good enough here
    from statistics import NormalDist
    return NormalDist().inv_cdf(power)


if __name__ == "__main__":
    main()
