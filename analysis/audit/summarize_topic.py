#!/usr/bin/env python
"""Summarize steering_mech.py --followup results (F10): is the steering confound generic affirmation?

    python summarize_topic.py ../../results/steering_topic/llama_3_1_8b_instruct__seed0.json [more.json]

Per model, at the middle kappa: per concept x variant, the shifts of the target, topical
off-target, non-topical always-No and negated questions, plus selectivity. Then:
  generic ratio   non-topical shift / target shift under orig (1 = the push ignores the question)
  removal         share of the orig non-topical shift removed by each projection variant
  cross-concept   Spearman of the orig non-topical shift with cos(concept, shared PC1) and with
                  cos(concept, yes_dir)
Prints markdown.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

VARIANTS = ["orig", "minus_yes", "minus_shared", "minus_shared_loo", "minus_topic", "topic_only", "interchange"]


def row_at(rows, kappa):
    if rows and rows[0]["kappa"] is None:
        return rows[0]
    return min(rows, key=lambda r: abs(r["kappa"] - kappa))


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        kappas = [float(k) for k in d["config"]["kappas"].split(",")]
        kap = kappas[len(kappas) // 2]
        print(f"\n### {d['model']} (window {d['window'][0]}-{d['window'][-1]}, kappa {kap})\n")
        print("| Concept | Variant | Target | Topical off | Non-topical | Negated | Selectivity |")
        print("|---|---|---|---|---|---|---|")
        gen, rem, nt_orig, cs, cy = {}, {v: [] for v in VARIANTS[1:5]}, [], [], []
        for c, res in d["concepts"].items():
            V = res["variants"]
            for v in VARIANTS:
                if v not in V:
                    continue
                r = row_at(V[v], kap)
                nt = r.get("nontopic_shift")
                print(f"| {c} | {v} | {r['on_shift']:+.2f} | {r['off_shift']:+.2f} | "
                      f"{'' if nt is None else f'{nt:+.2f}'} | {r['neg_shift']:+.2f} | {r['selectivity']:+.2f} |")
            o = row_at(V["orig"], kap)
            if "nontopic_shift" in o:
                gen[c] = o["nontopic_shift"] / o["on_shift"] if abs(o["on_shift"]) > 1e-6 else float("nan")
                nt_orig.append(o["nontopic_shift"])
                cs.append(abs(d["geometry"][c]["cos_shared"])); cy.append(abs(d["geometry"][c]["cos_yes"]))
                for v in rem:
                    if v in V and abs(o["nontopic_shift"]) > 1e-6:
                        rem[v].append(1 - row_at(V[v], kap)["nontopic_shift"] / o["nontopic_shift"])
        if gen:
            print("\n**Generic ratio** (non-topical / target shift, orig): "
                  + ", ".join(f"{c} {g:.2f}" for c, g in gen.items())
                  + f"; median {np.nanmedian(list(gen.values())):.2f}")
            print("\n**Share of the orig non-topical shift removed** (median over concepts): "
                  + ", ".join(f"{v} {np.median(x):.2f}" for v, x in rem.items() if x))
            if len(nt_orig) >= 4:
                rs, ps = spearmanr(cs, nt_orig)
                ry, py = spearmanr(cy, nt_orig)
                print(f"\n**Cross-concept** Spearman of the orig non-topical shift with |cos(shared PC1)| "
                      f"{rs:+.2f} (p {ps:.2f}); with |cos(yes_dir)| {ry:+.2f} (p {py:.2f}); n = {len(nt_orig)}")
        g = d["geometry"]
        if any("cos_topic" in v for v in g.values()):
            print("\n**cos(topic_dir, ·):** " + ", ".join(
                f"{c} concept {v['cos_topic']:+.2f} / yes {v['cos_topic_yes']:+.2f} / shared {v['cos_topic_shared']:+.2f}"
                for c, v in g.items()))


if __name__ == "__main__":
    main()
