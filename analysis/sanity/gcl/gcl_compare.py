#!/usr/bin/env python
"""Cross-concept framework comparison (reviewer #5).

The "reusable, concept-agnostic framework" claim is only as strong as the number of
concepts the FULL ladder has been run on. run_gcl.py runs the same gate->locate->
necessary->low-rank->sufficient ladder on any registered concept; this reads the saved
per-concept reports and lays them side by side so the manuscript can show the ladder
holds on >1 concept (suicidality + toxicity), with the matched-negative
suicide-vs-depression run included when present.

This does NOT run the model; it summarizes existing results/gcl/{concept}/{tag}.json.
Run the ladders first, e.g.:
  python analysis/sanity/gcl/run_gcl.py --concept suicide
  python analysis/sanity/gcl/run_gcl.py --concept toxicity
  python analysis/sanity/gcl/run_gcl.py --concept suicide_depression

Usage:
  python analysis/sanity/gcl/gcl_compare.py
  python analysis/sanity/gcl/gcl_compare.py --tag llama_3_1_8b_instruct

Output: results/gcl/framework_comparison.json (+ printed table)
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

GCL = PROJECT_ROOT / "results" / "gcl"
ORDER = ["suicide", "suicide_depression", "toxicity", "sentiment", "sports", "business"]


def parse_args():
    p = argparse.ArgumentParser(description="Summarize GCL ladder across concepts")
    p.add_argument("--tag", default="llama_3_1_8b_instruct",
                   help="model tag (filename stem) to compare across concepts")
    return p.parse_args()


def load_reports(tag):
    reports = {}
    for concept_dir in sorted(GCL.glob("*")):
        if not concept_dir.is_dir():
            continue
        f = concept_dir / f"{tag}.json"
        if f.exists():
            reports[concept_dir.name] = json.load(open(f))
    return reports


def row(r):
    g = r.get("gate", {})
    nec = r.get("necessary")
    lr = r.get("low_rank")
    suf = r.get("sufficient")
    return {
        "concept": r.get("display_name", r.get("concept")),
        "gate": g.get("decision"),
        "behavioral_auc": g.get("behavioral_auc"),
        "probe_best_auc": g.get("probe_best_auc"),
        "probe_best_layer": g.get("probe_best_layer"),
        "lexical_baseline_auc": r.get("lexical_baseline_auc"),
        "necessity_base": nec and nec.get("baseline_auc"),
        "necessity_ablate": nec and nec.get("ablate_auc"),
        "necessity_random": nec and nec.get("random_auc"),
        "k_star": lr and lr.get("k_star"),
        "steering_specific": suf and suf.get("specific"),
        "scorecard": r.get("scorecard"),
    }


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    args = parse_args()
    reports = load_reports(args.tag)
    if not reports:
        print(f"No results/gcl/*/{args.tag}.json found. Run run_gcl.py first.")
        return
    ordered = [c for c in ORDER if c in reports] + [c for c in reports if c not in ORDER]
    rows = [row(reports[c]) for c in ordered]

    print("\n" + "=" * 100)
    print(f"GCL FRAMEWORK COMPARISON  (model tag: {args.tag})")
    print("=" * 100)
    h = (f"{'concept':40} {'gate':>6} {'behav':>7} {'probe':>7} {'base>abl(rand)':>18} "
         f"{'k*':>3} {'steer-spec':>10}")
    print(h)
    print("-" * 100)
    for r in rows:
        nec = (f"{fmt(r['necessity_base'])}>{fmt(r['necessity_ablate'])}"
               f"({fmt(r['necessity_random'])})" if r["necessity_base"] is not None else "--")
        print(f"{r['concept'][:40]:40} {str(r['gate']):>6} {fmt(r['behavioral_auc']):>7} "
              f"{fmt(r['probe_best_auc']):>7} {nec:>18} {str(r['k_star']):>3} "
              f"{str(r['steering_specific']):>10}")
    print("=" * 100)

    # framework claim: >=2 concepts clear the gate AND show necessity (ablate << base, vs random)
    passed = []
    for r in rows:
        if r["gate"] == "GO" and r["necessity_base"] and r["necessity_ablate"] is not None:
            drop = r["necessity_base"] - r["necessity_ablate"]
            rnd_drop = r["necessity_base"] - (r["necessity_random"] or r["necessity_base"])
            if drop > 3 * max(rnd_drop, 0.01):
                passed.append(r["concept"])
    print(f"Concepts where the full ladder confirms a necessary feature: {len(passed)}")
    for c in passed:
        print(f"  - {c}")
    print("Framework claim (ladder transfers across concepts) supported by n =", len(passed))

    out = GCL / "framework_comparison.json"
    json.dump({"tag": args.tag, "concepts": ordered, "rows": rows,
               "ladder_confirmed_concepts": passed,
               "n_concepts_confirmed": len(passed)}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
