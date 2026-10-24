#!/usr/bin/env python
"""UMD clinician risk vs keyword baselines (fix for the paper's "the axis tracks clinician-assigned risk").

The submitted paper reports that projecting UMD expert-subset users (n = 245, all r/SuicideWatch
posters, so the register is already matched) onto the 232K suicidality direction tracks the
clinicians' a < b < c < d risk levels (Spearman 0.56; high-vs-low AUC 0.81). It never compared this
with a keyword model, and the per-user scores were lost (results/sc2_binary/umd_severity.json was
reconstructed from stdout). This script reruns the analysis on the audit backend (fp/bf16, not 4-bit)
and adds the missing baselines, all scored on the same users:

  model      proj_site   projection onto the 232K direction at the site chosen on held-out 232K
             proj_L16    the same at hidden index 16 (the paper's layer; Llama-8B)
             behav       the model's own Yes-No logit difference
             probe_cv    in-domain logistic probe on the site activations, 5-fold CV over users
  keywords   tfidf232    TF-IDF trained on 232K, applied without retraining (parallel to proj_site)
             tfidf_cv    in-domain TF-IDF, 5-fold CV over users (parallel to probe_cv)
             lexicon     explicit suicide terms per 1,000 characters (no training)
             length      characters; number of posts
Keyword scores are computed on the text the model sees (*_seen, the truncated prompt body) and on
the full per-user text (*_full, up to 4,000 characters).

Reported for each score: Spearman with the 4-level risk, high (c, d) vs low (a, b) AUC with a
bootstrap CI, the paired AUC difference against proj_site, and a partial Spearman given length.
Outputs: results/umd_v2/<model>__seed<s>.json and .npz (per-user scores and IDs; no text is saved).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import Backend
from concepts import PROJECT_ROOT
from stats import auc, boot_auc_ci, boot_auc_diff_ci
from suicide_dir import fit_direction, lexicon_rate, project

sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis_utils import load_umd_with_severity  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-dir", type=int, default=500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "umd_v2"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cv_scores(make, X, y, seed):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        m = make().fit(X[tr] if not isinstance(X, list) else [X[i] for i in tr], y[tr])
        out[te] = m.predict_proba(X[te] if not isinstance(X, list) else [X[i] for i in te])[:, 1]
    return out


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def boot_spearman_diff(sev, a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(sev), len(sev))
        vals.append(spearmanr(sev[i], a[i])[0] - spearmanr(sev[i], b[i])[0])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    a = parse_args()
    if a.smoke:
        a.n_dir, a.n_val = 24, 12
    t0 = time.time()
    tag = a.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    be = Backend(a.model, dtype=a.dtype)
    log(f"umd_keyword_baseline on {a.model}: {be.n_layers} layers")

    D = fit_direction(be, n_dir=a.n_dir, n_val=a.n_val, seed=a.seed, batch=a.batch)
    c, site = D["concept"], D["site"]
    log(f"232K direction: site L{site} (held-out 232K AUC {D['val_auc'][site]:.3f}); "
        f"L16 {D['val_auc'][min(16, be.n_layers)]:.3f}")

    recs = load_umd_with_severity(source="expert")
    if a.smoke:
        recs = recs[::10]
    full = [r["text"] for r in recs]
    seen = [be._truncate(t) for t in full]
    sev = np.array([r["severity"] for r in recs])
    hi = (sev >= 2).astype(int)
    log(f"UMD expert: {len(recs)} users, levels {[int((sev == s).sum()) for s in range(4)]}")

    ld, H = be.collect([be.polar_prompt(t, c.prefix, c.question) for t in full], D["yes"], D["no"], a.batch)
    proj_all = np.stack([project(H, D, j) for j in range(H.shape[1])], 1)   # [N, L+1]

    S = {"proj_site": proj_all[:, site], "proj_L16": proj_all[:, min(16, be.n_layers)], "behav": ld}
    Hs = np.asarray(H[:, site], dtype=np.float32)
    S["probe_cv"] = cv_scores(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
                              Hs, hi, a.seed)
    for kind, texts in (("seen", seen), ("full", full)):
        S[f"tfidf232_{kind}"] = D["tfidf"].predict_proba(texts)[:, 1]
        S[f"tfidf_cv_{kind}"] = cv_scores(lambda: make_pipeline(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
            LogisticRegression(C=1.0, max_iter=2000)), texts, hi, a.seed)
        S[f"lexicon_{kind}"] = lexicon_rate(texts)
    nchar = np.array([len(t) for t in full], dtype=float)
    S["length_chars"] = nchar
    S["n_posts"] = np.array([r["n_posts"] for r in recs], dtype=float)

    ref = S["proj_site"]
    rows = {}
    for k, s in S.items():
        rho, p = spearmanr(sev, s)
        rows[k] = {"spearman": float(rho), "p": float(p),
                   "partial_spearman_given_length": float("nan") if k == "length_chars" else partial_spearman(sev, s, nchar),
                   "auc_hi_lo": auc(s[hi == 1], s[hi == 0]),
                   "auc_hi_lo_ci95": boot_auc_ci(s[hi == 1], s[hi == 0], seed=a.seed),
                   "auc_minus_proj_site_ci95": None if k == "proj_site" else
                   boot_auc_diff_ci(s[hi == 1], s[hi == 0], ref[hi == 1], ref[hi == 0], seed=a.seed),
                   "spearman_minus_proj_site_ci95": None if k == "proj_site" else
                   boot_spearman_diff(sev, s, ref, seed=a.seed)}
        r = rows[k]
        log(f"{k:18} rho {r['spearman']:+.3f}  partial {r['partial_spearman_given_length']:+.3f}  "
            f"AUC {r['auc_hi_lo']:.3f} [{r['auc_hi_lo_ci95'][0]:.3f}, {r['auc_hi_lo_ci95'][1]:.3f}]"
            + ("" if r["auc_minus_proj_site_ci95"] is None else
               f"  vs proj_site {r['auc_minus_proj_site_ci95'][0]:+.3f}..{r['auc_minus_proj_site_ci95'][1]:+.3f}"))
    layer_rho = [float("nan")] + [float(spearmanr(sev, proj_all[:, j])[0]) for j in range(1, proj_all.shape[1])]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{tag}__seed{a.seed}{'_smoke' if a.smoke else ''}"
    json.dump({"model": a.model, "config": vars(a), "direction": D["meta"], "n_users": len(recs),
               "levels": {s: int((sev == s).sum()) for s in range(4)}, "scores": rows,
               "spearman_by_layer": layer_rho, "runtime_min": round((time.time() - t0) / 60, 1),
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(f"{stem}.json", "w"), indent=1)
    np.savez_compressed(f"{stem}.npz", user_id=np.array([r["user_id"] for r in recs]), severity=sev,
                        proj_all=proj_all, **{k: v for k, v in S.items()})
    log(f"saved {stem}.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
