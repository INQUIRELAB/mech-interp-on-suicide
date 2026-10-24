"""Does setting the threshold on local validation data fix the default-answer yes-bias?

Register-matched set (SWMH r/SuicideWatch vs r/depression), per-post scores saved by matched_prompts.py
(data/paper2/prompt_invariance/*.npz; 1000 evaluation posts per class). For each model and prompt, draw a
stratified calibration subset of n_cal posts per class, set the threshold at the 90th percentile of the
calibration r/depression scores (target specificity 90%), and apply it to the remaining posts. Repeated R times.
Reports test specificity and sensitivity (mean and 2.5/97.5 percentiles over repeats) next to the default
threshold (logit difference 0; not defined for the rating prompt).

Usage: python threshold_calibration.py <prompt_invariance dir> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
R, TARGET = 500, 0.9
N_CAL = [25, 50, 100, 250, 500]
PROMPTS = ["base", "clinical", "contrast"]
rng = np.random.default_rng(0)
res = {"target_specificity": TARGET, "repeats": R, "n_cal_per_class": N_CAL, "models": {}}


def q(a):
    return [float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


for f in sorted(glob.glob(os.path.join(src, "suicide_depression__*__seed0.npz"))):
    model = os.path.basename(f).split("__")[1]
    z = np.load(f)
    y = z["y"].astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res["models"][model] = {}
    for p in ["tfidf"] + PROMPTS:
        s = z[p]
        row = {}
        if p != "tfidf":
            row["default"] = {"specificity": float((s[neg] <= 0).mean()), "sensitivity": float((s[pos] > 0).mean())}
        for n in N_CAL:
            sp, se = np.empty(R), np.empty(R)
            for r in range(R):
                cp, cn = rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)
                tp, tn = np.setdiff1d(pos, cp), np.setdiff1d(neg, cn)
                thr = np.quantile(s[cn], TARGET)
                sp[r], se[r] = (s[tn] <= thr).mean(), (s[tp] > thr).mean()
            row[f"cal{n}"] = {"specificity": q(sp), "sensitivity": q(se)}
        res["models"][model][p] = row
        d = row.get("default", {})
        print(f"{model[:10]:10s} {p:8s} default spec {d.get('specificity', float('nan')):.3f} | "
              + " ".join(f"n{n}: spec {row[f'cal{n}']['specificity'][0]:.3f} "
                         f"[{row[f'cal{n}']['specificity'][1]:.2f},{row[f'cal{n}']['specificity'][2]:.2f}] "
                         f"sens {row[f'cal{n}']['sensitivity'][0]:.3f}" for n in (50, 100, 500)))

json.dump(res, open(out, "w"), indent=1)
print("saved", out)
