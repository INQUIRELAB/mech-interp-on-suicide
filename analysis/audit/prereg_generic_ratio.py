#!/usr/bin/env python
"""Score plan/PREREG_gemma_generic_ratio.md: predicted (gsa_lm.py --mode steer) vs observed
(steering_mech.py --followup, variant orig, kappa = 0.1) generic ratios.

    python prereg_generic_ratio.py <project root>

Observed ratio = nontopic_shift / on_shift. A concept is excluded from a model's comparison if
|observed on_shift| < 1.0 (prereg exclusion rule). Medians are reported over the included concepts
(primary) and over all concepts. Gemma (H1-H3) is scored only when both the prediction and the
observation exist; the prediction's finished_utc must precede the observation file's mtime.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

KAPPA, MIN_ON, CLASS_CUT = 0.1, 1.0, 0.6
CAL = {"llama": ("steer_meta_llama_3_1_8b_instruct", "llama_3_1_8b_instruct"),
       "qwen": ("steer_qwen2_5_7b_instruct", "qwen2_5_7b_instruct"),
       "mistral": ("steer_mistral_7b_instruct_v0_3", "mistral_7b_instruct_v0_3")}
HELD = {"gemma": ("steer_gemma_2_9b_it", "gemma_2_9b_it")}


def load(root, pred_stem, obs_stem):
    pp = root / "results/gsa" / f"{pred_stem}__seed0.json"
    po = root / "results/steering_topic" / f"{obs_stem}__seed0.json"
    if not (pp.exists() and po.exists()):
        return None
    P, O = json.loads(pp.read_text()), json.loads(po.read_text())
    rows = []
    for c, res in O["concepts"].items():
        o = min(res["variants"]["orig"], key=lambda r: abs(r["kappa"] - KAPPA))
        rows.append({"concept": c, "on_shift": o["on_shift"], "nontopic_shift": o["nontopic_shift"],
                     "obs": o["nontopic_shift"] / o["on_shift"],
                     "pred": P["concepts"][c]["pred_generic_ratio"],
                     "included": abs(o["on_shift"]) >= MIN_ON})
    meta = {"pred_file": str(pp), "obs_file": str(po), "pred_finished_utc": P.get("finished_utc"),
            "pred_sha256": P.get("script_sha256"),
            "obs_mtime_utc": datetime.fromtimestamp(os.path.getmtime(po), timezone.utc).isoformat()}
    return rows, meta


def medians(rows, only_included=True):
    r = [x for x in rows if x["included"] or not only_included]
    return float(np.median([x["pred"] for x in r])), float(np.median([x["obs"] for x in r]))


def main():
    root = Path(sys.argv[1])
    out, pts = {"models": {}}, []
    for m, (ps, os_) in CAL.items():
        got = load(root, ps, os_)
        if got is None:
            print(f"{m}: missing files"); continue
        rows, meta = got
        out["models"][m] = {"rows": rows, "meta": meta, "median_included": medians(rows),
                            "median_all": medians(rows, False)}
        pts += [(x["pred"], x["obs"]) for x in rows if x["included"]]
        print(f"\n{m}: " + ", ".join(f"{x['concept']} {x['pred']:+.2f}/{x['obs']:+.2f}"
                                   + ("" if x["included"] else " (excl)") for x in rows))
        print(f"  median pred/obs: included {out['models'][m]['median_included'][0]:.2f}/"
              f"{out['models'][m]['median_included'][1]:.2f}; all {out['models'][m]['median_all'][0]:.2f}/"
              f"{out['models'][m]['median_all'][1]:.2f}")
    if len(out["models"]) == 3:
        rho, p = spearmanr([a for a, _ in pts], [b for _, b in pts])
        med = {m: v["median_included"][0] for m, v in out["models"].items()}
        c2 = med["qwen"] < med["llama"] and med["qwen"] < med["mistral"]
        out["C1"] = {"spearman": float(rho), "p": float(p), "n": len(pts), "pass": bool(rho >= 0.5)}
        out["C2"] = {"pred_medians": med, "pass": bool(c2)}
        print(f"\nC1: Spearman {rho:+.2f} (p {p:.3g}, n {len(pts)}) -> {'PASS' if rho >= 0.5 else 'FAIL'}")
        print(f"C2: predicted medians {med} -> {'PASS' if c2 else 'FAIL'}")
    for m, (ps, os_) in HELD.items():
        got = load(root, ps, os_)
        if got is None:
            print(f"\n{m}: not scored (prediction and/or observation missing)"); continue
        rows, meta = got
        before = meta["pred_finished_utc"] and meta["pred_finished_utc"] < meta["obs_mtime_utc"]
        res = {"rows": rows, "meta": meta, "prediction_precedes_observation": bool(before)}
        for tag, keep in (("all", lambda c: True), ("no_sentiment", lambda c: c != "sentiment")):
            r = [x for x in rows if x["included"] and keep(x["concept"])]
            pm, om = float(np.median([x["pred"] for x in r])), float(np.median([x["obs"] for x in r]))
            rho = spearmanr([x["pred"] for x in r], [x["obs"] for x in r])[0] if len(r) >= 3 else float("nan")
            res[tag] = {"n": len(r), "pred_median": pm, "obs_median": om,
                        "H1": bool((pm >= CLASS_CUT) == (om >= CLASS_CUT)),
                        "H2": bool(abs(pm - om) <= 0.3), "H3_spearman": float(rho), "H3": bool(rho >= 0.4)}
            print(f"\n{m} [{tag}] n {len(r)}: pred median {pm:.2f}, obs median {om:.2f}; "
                  f"H1 {res[tag]['H1']}, H2 {res[tag]['H2']}, H3 rho {rho:+.2f}")
        print(f"prediction finished {meta['pred_finished_utc']}; observation written {meta['obs_mtime_utc']}")
        out["models"][m] = res
    (root / "results/gsa/prereg_generic_ratio_score.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
