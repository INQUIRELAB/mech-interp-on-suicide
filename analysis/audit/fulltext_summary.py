"""Register-matched set on full posts: summary of the matched_prompts.py --body-tokens 4096 runs.

Both systems see the whole post (TF-IDF is fit and scored on full posts in matched_prompts.py; the LLM body budget
of 4096 tokens cut at most one evaluation post). For every model and prompt: AUC and the paired bootstrap CI of
(LLM - TF-IDF) from the run JSON, sensitivity at 90% specificity with the class-stratified bootstrap used in
tfidf_truncation.py, its paired difference from TF-IDF, specificity at the default answer and the 5-fold stack AUC.
Also the change from the 176-token run of the same model and prompt (prompt_invariance/*.npz).

Usage: python fulltext_summary.py <prompt_invariance_full dir> <prompt_invariance dir> <out.json>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc  # noqa: E402

FULL_DIR, TRUNC_DIR, OUT = sys.argv[1:4]
MODELS = ["llama_3_1_8b_instruct", "qwen2_5_7b_instruct", "mistral_7b_instruct_v0_3", "gemma_2_9b_it"]
VARIANTS = ["base", "neg_flip", "clinical", "contrast", "escalate", "fewshot", "rating"]


def sens_at_spec(yy, s, spec=0.90):
    return float((s[yy == 1] > np.quantile(s[yy == 0], spec)).mean())


res = {"budget_tokens": 4096, "models": {}}
idx = None
for tag in MODELS:
    d = json.load(open(os.path.join(FULL_DIR, f"suicide_depression__{tag}__seed0_full.json")))
    z = np.load(os.path.join(FULL_DIR, f"suicide_depression__{tag}__seed0_full.npz"))
    zt = np.load(os.path.join(TRUNC_DIR, f"suicide_depression__{tag}__seed0.npz"))
    y = z["y"].astype(int)
    assert (y == zt["y"].astype(int)).all()
    r_tf = float(np.corrcoef(z["tfidf"], zt["tfidf"])[0, 1])
    assert r_tf > 0.999, f"{tag}: TF-IDF scores differ from the 176-token run (r={r_tf})"
    if idx is None:   # same resamples as tfidf_truncation.py (B=1000, seed=1)
        rng = np.random.default_rng(1)
        pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = [np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))] for _ in range(1000)]
        tf = z["tfidf"]
        res["tfidf_full"] = {"auc": d["tfidf_auc"], "ci95": d["tfidf_ci95"], "sens90": sens_at_spec(y, tf),
                             "sens90_ci95": np.percentile([sens_at_spec(y[i], tf[i]) for i in idx], [2.5, 97.5]).tolist()}
    m = {"n_eval_truncated": d["n_eval_truncated"], "r_tfidf_vs_176_run": r_tf, "variants": {}}
    for v in VARIANTS:
        s, r = z[v], d["variants"][v]
        st = zt[v]
        row = {"auc": r["auc"], "ci95": r["ci95"], "minus_tfidf_ci95": r["minus_tfidf_ci95"],
               "auc_176": auc(st[y == 1], st[y == 0]), "stack_cv_auc": r["stack_cv_auc"],
               "stack_minus_tfidf_ci95": r["stack_minus_tfidf_ci95"],
               "spec_default": None if r["frac_yes_neg"] is None else 1 - r["frac_yes_neg"],
               "sens90": sens_at_spec(y, s),
               "sens90_ci95": np.percentile([sens_at_spec(y[i], s[i]) for i in idx], [2.5, 97.5]).tolist(),
               "sens90_minus_tfidf_ci95": np.percentile(
                   [sens_at_spec(y[i], s[i]) - sens_at_spec(y[i], z["tfidf"][i]) for i in idx], [2.5, 97.5]).tolist()}
        m["variants"][v] = row
        lo, hi = row["minus_tfidf_ci95"]
        print(f"{tag:26s} {v:9s} AUC {row['auc']:.3f} (176: {row['auc_176']:.3f}) diff [{lo:+.3f},{hi:+.3f}] "
              f"sens90 {row['sens90']:.3f} diff {np.round(row['sens90_minus_tfidf_ci95'], 3).tolist()} "
              f"spec_default {row['spec_default']}", flush=True)
    res["models"][tag] = m

rows = [r for m in res["models"].values() for r in m["variants"].values()]
res["summary"] = {
    "n_below": sum(r["minus_tfidf_ci95"][1] < 0 for r in rows),
    "n_above": sum(r["minus_tfidf_ci95"][0] > 0 for r in rows),
    "n_stack_gain_excludes_0": sum(r["stack_minus_tfidf_ci95"][0] > 0 for r in rows),
    "stack_auc_range": [min(r["stack_cv_auc"] for r in rows), max(r["stack_cv_auc"] for r in rows)],
    "n_auc_up_vs_176": sum(r["auc"] > r["auc_176"] for r in rows)}
print(res["tfidf_full"], res["summary"])
json.dump(res, open(OUT, "w"), indent=1)
