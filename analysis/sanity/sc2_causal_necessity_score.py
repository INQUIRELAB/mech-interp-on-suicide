"""Aggregate causal-necessity score S for the suicidality feature.

Reviewer 1 (TIST) asked for "an overall confidence / confidence interval for the
causality of a feature" -- the per-rung AUCs never roll up into one number. This
script defines that number and computes it from the already-saved ablation runs.

    S = 1 - (AUC_ablate - 0.5) / (AUC_full - 0.5)   clipped to [0, 1]

Reading:
    S = 0  ablating the feature does nothing (AUC unchanged)      -> not necessary
    S = 1  ablating the feature drives AUC to chance (0.5)        -> fully necessary
S is the fraction of the model's *above-chance* suicide-vs-non-suicide ranking
ability that routes through the feature. A matched random direction / subspace
should sit at S ~ 0.

Point estimates come from results/sc2_binary/*.json. A bootstrap 95% CI needs
per-example scores; re-run sc2_faithfulness_perlayer.py / sc2_rankk.py with
--save-scores and pass --scores here (TODO, tracked as change A03).
"""
import argparse
import json
import os

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "sc2_binary"
)


def score(auc_full, auc_ablate):
    denom = auc_full - 0.5
    if denom <= 0:
        return float("nan")
    return max(0.0, min(1.0, 1.0 - (auc_ablate - 0.5) / denom))


def _load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "causal_necessity_score.json"))
    args = ap.parse_args()

    per = _load("faithfulness_perlayer.json")
    one = _load("faithfulness.json")
    rk = _load("rankk.json")

    rows = []
    rows.append((
        "single direction, all layers", per["baseline"]["auc"],
        per["suicidality_ablation"]["auc"], per["random_ablation"]["auc"],
    ))
    rows.append((
        "single direction, layer 16 only", one["baseline_auc"],
        one["suicidality_ablation_auc"], one["random_ablation_auc"],
    ))
    for s in rk["sweep"]:
        rows.append((
            "rank-%d subspace" % s["k"], rk["baseline_auc"],
            s["suic_auc"], s["random_auc"],
        ))

    out = []
    print("%-34s  %7s  %7s  %7s   %6s  %6s" % (
        "condition", "AUC_full", "AUC_abl", "AUC_rnd", "S", "S_rnd"))
    print("-" * 82)
    for name, full, abl, rnd in rows:
        S, S_rnd = score(full, abl), score(full, rnd)
        out.append({
            "condition": name, "auc_full": full, "auc_ablate": abl,
            "auc_random": rnd, "S": S, "S_random": S_rnd,
        })
        print("%-34s  %7.3f  %7.3f  %7.3f   %6.3f  %6.3f" % (
            name, full, abl, rnd, S, S_rnd))

    payload = {
        "definition": "S = 1 - (AUC_ablate - 0.5) / (AUC_full - 0.5), clipped [0,1]",
        "ci": "point estimates only; bootstrap CI pending per-example scores (A03)",
        "rows": out,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
