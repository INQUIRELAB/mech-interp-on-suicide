#!/usr/bin/env python
"""Diff the headline rungs between two precision runs and emit the paper table.

The npj draft reports every mechanistic number from a 4-bit model. This pulls
the same quantities out of two results directories and prints a side-by-side
markdown table plus a LaTeX tabular ready to paste into the SI, so the
robustness claim is a table rather than a sentence.

Usage:
    python analysis/sanity/compare_precision.py \
        --a results/sc2_binary --b results/sc2_binary_fp16
    python analysis/sanity/compare_precision.py --a ... --b ... --tex si_precision.tex
"""
import json, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# (row label, file, extractor, "higher is the confirming direction"?)
ROWS = [
    ("Behavioral AUC (gate)",      "gate.json",
     lambda d: d["behav_auc"]),
    ("Probe AUC @ L16",            "gate.json",
     lambda d: d["best_probe_auc"]),
    ("TF-IDF baseline",            "gate.json",
     lambda d: d["tfidf_acc"]),
    ("Probe recall, masked",       "specificity.json",
     lambda d: d["probe_recall_masked"]),
    ("Necessity: baseline AUC",    "faithfulness.json",
     lambda d: d["baseline_auc"]),
    ("Necessity: ablated AUC",     "faithfulness.json",
     lambda d: d["suicidality_ablation_auc"]),
    ("Necessity: random-dir AUC",  "faithfulness.json",
     lambda d: d["random_ablation_auc"]),
    ("Rank-4 ablated AUC",         "rankk.json",
     lambda d: next(s["suic_auc"] for s in d["sweep"] if s["k"] == 4)),
    ("Rank-4 random AUC",          "rankk.json",
     lambda d: next(s["random_auc"] for s in d["sweep"] if s["k"] == 4)),
    ("Steering flip rate @ c=1",   "sufficiency.json",
     lambda d: next(s["frac_over_threshold"] for s in d["sweep"] if s["coef"] == 1.0)),
    ("Off-target LD @ c=1",        "steering_specificity.json",
     lambda d: next(s["off_target_ld"] for s in d["sweep"] if s["coef"] == 1.0)),
    ("Transfer AUC (SDCNL)",       "generalization.json",
     lambda d: d["results"]["SDCNL"]["transfer_auc"]),
    ("Transfer AUC (SWMH)",        "generalization.json",
     lambda d: d["results"]["SWMH"]["transfer_auc"]),
]


def pull(root, fname, fn):
    p = Path(root) / fname
    if not p.exists():
        return None
    try:
        return fn(json.loads(p.read_text()))
    except (KeyError, StopIteration, ValueError):
        return None


def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference run (e.g. the 4-bit results)")
    ap.add_argument("--b", required=True, help="new run (e.g. the fp16 results)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--tex", default=None, help="also write a LaTeX tabular here")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="flag rows moving more than this in absolute terms")
    args = ap.parse_args()

    a_root = (PROJECT_ROOT / args.a) if not Path(args.a).is_absolute() else Path(args.a)
    b_root = (PROJECT_ROOT / args.b) if not Path(args.b).is_absolute() else Path(args.b)
    la = args.label_a or a_root.name.replace("sc2_binary", "").strip("_") or "4bit"
    lb = args.label_b or b_root.name.replace("sc2_binary", "").strip("_") or "run B"

    rows, flagged = [], []
    for label, fname, fn in ROWS:
        va, vb = pull(a_root, fname, fn), pull(b_root, fname, fn)
        delta = None if (va is None or vb is None) else vb - va
        rows.append((label, va, vb, delta))
        if delta is not None and abs(delta) > args.tol:
            flagged.append((label, va, vb, delta))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'Quantity'.ljust(w)}  {la:>8}  {lb:>8}  {'delta':>7}")
    print("-" * (w + 30))
    for label, va, vb, delta in rows:
        mark = "  <-- moved" if delta is not None and abs(delta) > args.tol else ""
        d = "--" if delta is None else f"{delta:+.3f}"
        print(f"{label.ljust(w)}  {fmt(va):>8}  {fmt(vb):>8}  {d:>7}{mark}")

    print()
    if not flagged:
        print(f"No headline quantity moved by more than {args.tol:.02f}. "
              f"The reported conclusions are precision-robust.")
    else:
        print(f"{len(flagged)} quantity(ies) moved by more than {args.tol:.02f}:")
        for label, va, vb, delta in flagged:
            print(f"  {label}: {fmt(va)} -> {fmt(vb)} ({delta:+.3f})")
        print("\nThese need to be re-reported at the new precision before submission.")

    if args.tex:
        out = Path(args.tex)
        lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 rf"Quantity & {la} & {lb} & $\Delta$ \\", r"\midrule"]
        for label, va, vb, delta in rows:
            d = "--" if delta is None else f"{delta:+.3f}"
            lines.append(rf"{label} & {fmt(va)} & {fmt(vb)} & {d} \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
