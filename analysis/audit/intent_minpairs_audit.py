#!/usr/bin/env python
"""Intent vs mention on lexically matched minimal pairs, on the audit backend (replaces the 4-bit,
Llama-only analysis/sanity/sc2_intent_minpairs.py, which never produced a saved result).

Each base text is rewritten as genuine first-person intent and as controls that keep the death/suicide
vocabulary but carry no present first-person intent: counterfactual (denial), third_person,
recovery_past, fiction, topical. Keyword models cannot separate these by construction, so this is
where a model that reads meaning should beat one that reads words.

Scores per item:
  behav_base      Yes-No logit difference under the paper's prompt
  behav_clinical  the same under the clinical-definition prompt (matched_prompts.py "clinical")
  proj_site       projection onto the 232K suicidality direction at the site chosen on held-out 232K
  tfidf232        TF-IDF trained on 232K (the benchmark's keyword model), no retraining
  lexicon         explicit suicide terms per 1,000 characters
For genuine vs each control (and pooled): AUC with a stratified bootstrap CI, and a paired statistic,
the share of base texts whose genuine version scores above its own control (sign test). Also the
share of items the model answers "Yes" (LD > 0) per condition, and the projection AUC at every layer.
Output: results/intent_minpairs/<model>__<data stem>.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from backend import Backend
from concepts import PROJECT_ROOT
from matched_prompts import CLINICAL
from suicide_dir import fit_direction, lexicon_rate, project

DATA = PROJECT_ROOT / "data" / "intent_minpairs" / "intent_minpairs_v1.jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-dir", type=int, default=500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "intent_minpairs"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def boot_auc(ps, ns, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ps, ns = np.asarray(ps), np.asarray(ns)
    y = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
    point = float(roc_auc_score(y, np.r_[ps, ns]))
    vals = [roc_auc_score(y, np.r_[ps[rng.integers(0, len(ps), len(ps))], ns[rng.integers(0, len(ns), len(ns))]])
            for _ in range(B)]
    return point, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    a = parse_args()
    if a.smoke:
        a.n_dir, a.n_val = 24, 12
    t0 = time.time()
    tag = a.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    log(f"{len(rows)} items: " + ", ".join(f"{k} {len(v)}" for k, v in by_cond.items()))

    be = Backend(a.model, dtype=a.dtype)
    D = fit_direction(be, n_dir=a.n_dir, n_val=a.n_val, seed=a.seed, batch=a.batch)
    c, site = D["concept"], D["site"]
    log(f"232K direction: site L{site} (held-out 232K AUC {D['val_auc'][site]:.3f})")

    texts = [r["text"] for r in rows]
    ld, H = be.collect([be.polar_prompt(t, c.prefix, c.question) for t in texts], D["yes"], D["no"], a.batch)
    ld_clin = be.logit_diff([be.polar_prompt(t, f"{CLINICAL}\n\nRead the following text.",
                                             "Does the writer express suicidal ideation, as defined above?")
                             for t in texts], D["yes"], D["no"], a.batch)
    proj_all = np.stack([project(H, D, j) for j in range(H.shape[1])], 1)
    S = {"behav_base": ld, "behav_clinical": ld_clin, "proj_site": proj_all[:, site],
         "tfidf232": D["tfidf"].predict_proba(texts)[:, 1], "lexicon": lexicon_rate(texts)}
    idx = {r["id"]: i for i, r in enumerate(rows)}
    gen = by_cond["genuine_intent"]
    controls = [k for k in by_cond if k != "genuine_intent"]

    res = {}
    for cond in controls + ["ALL_controls"]:
        ctrl = [r for k in controls for r in by_cond[k]] if cond == "ALL_controls" else by_cond[cond]
        gi, ci = [idx[r["id"]] for r in gen], [idx[r["id"]] for r in ctrl]
        row = {"n_control": len(ci)}
        for k, s in S.items():
            pt, ci95 = boot_auc(s[gi], s[ci], seed=a.seed)
            # paired: genuine vs its own control from the same base text
            g_by_base = {r["base_id"]: s[idx[r["id"]]] for r in gen}
            d = np.array([g_by_base[r["base_id"]] - s[idx[r["id"]]] for r in ctrl if r["base_id"] in g_by_base])
            nz = d[d != 0]   # ties count 0.5 in the win rate and are dropped from the sign test
            row[k] = {"auc": pt, "auc_ci95": ci95, "n_pairs": len(d),
                      "paired_win_rate": float(np.mean((d > 0) + 0.5 * (d == 0))) if len(d) else None,
                      "sign_test_p": float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else None}
        row["proj_auc_by_layer"] = [float("nan")] + [
            float(roc_auc_score(np.r_[np.ones(len(gi)), np.zeros(len(ci))], np.r_[proj_all[gi, j], proj_all[ci, j]]))
            for j in range(1, proj_all.shape[1])]
        res[cond] = row
        log(f"{cond:15} n {len(ci):3}  " + "  ".join(
            f"{k} {row[k]['auc']:.2f} (win {row[k]['paired_win_rate']:.2f})" for k in S))
    yes_rate = {k: {"base": float((ld[[idx[r['id']] for r in v]] > 0).mean()),
                    "clinical": float((ld_clin[[idx[r['id']] for r in v]] > 0).mean())} for k, v in by_cond.items()}
    log("answers Yes (base / clinical): " + ", ".join(f"{k} {v['base']:.2f}/{v['clinical']:.2f}"
                                                      for k, v in yes_rate.items()))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}__{Path(a.data).stem}{'_smoke' if a.smoke else ''}.json"
    json.dump({"model": a.model, "config": vars(a), "direction": D["meta"], "n_items": len(rows),
               "per_condition": res, "yes_rate": yes_rate,
               "items": [{"id": r["id"], "condition": r["condition"], **{k: float(S[k][i]) for k in S}}
                         for i, r in enumerate(rows)],
               "runtime_min": round((time.time() - t0) / 60, 1),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(path, "w"), indent=1)
    log(f"saved {path} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
