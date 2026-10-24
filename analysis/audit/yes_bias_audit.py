#!/usr/bin/env python
"""Does removing the generic "yes" tendency fix LLM screening? Plus UMD on the full user text.

Steering showed that the suicide direction pushes every yes/no answer toward "yes" (non-topical/target
ratio near 1 in Llama and Mistral). This script asks whether that tendency also contaminates the screening
score itself, and whether it can be removed per post, without labels, from questions the model is
already asked in the steering audit. For each text we score
  target     the screening question (base prompt; and the contrast prompt on the matched set)
  nt_*       three non-topical questions whose true answer is always No (French? all capitals? over
             2,000 words?), same text, same template (NONTOPIC_QUESTIONS in steering_mech.py)
and form
  raw        target logit difference (Yes - No), as in the paper
  minus_nt   target - mean(nt): beta = 1, no fitting, no labels. Threshold 0 = "more yes than for French?"
  resid_ols  target - beta_ols * mean(nt), beta_ols from an unlabeled OLS fit on the fit split
  beta_cv    beta chosen by AUC on the fit split (uses labels, like TF-IDF does)
Sets
  matched    SWMH r/SuicideWatch vs r/depression, same split as matched_prompts.py (fit 2000 / eval 1000)
  minpairs   the 64 synthetic texts: "yes" rate per condition and genuine-vs-denial AUC
  umd_seen   245 UMD users, text truncated to the paper's 176 body tokens
  umd_full   the same users with the full per-user text (up to 4,000 characters; --umd-budget tokens)
For UMD, in-domain TF-IDF (5-fold CV over users) is refit on exactly the text each readout saw.

Usage: python yes_bias_audit.py --model unsloth/Meta-Llama-3.1-8B-Instruct
Outputs: results/yes_bias/<model>__seed<s>.json and .npz (per-text scores; no text is saved).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from backend import Backend
from concepts import PROJECT_ROOT, get_concept
from stats import auc, boot_auc_ci, boot_auc_diff_ci

sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis_utils import load_umd_with_severity  # noqa: E402

NONTOPIC_QUESTIONS = [   # copied from steering_mech.py so this script does not import the steering stack
    ("french", "Read the following text and determine its language.", "Is this text written in French?"),
    ("caps", "Read the following text and look at its formatting.",
     "Is this text written entirely in capital letters?"),
    ("long", "Read the following text and estimate its length.", "Is this text longer than 2,000 words?"),
]
CONTRAST = ("The writer of the following text is struggling with their mental health.",
            "Is the writer expressing suicidal thoughts, as opposed to depression without suicidal thoughts?")
MINPAIRS = PROJECT_ROOT / "data" / "intent_minpairs" / "intent_minpairs_v1.jsonl"
BETAS = np.round(np.arange(0, 2.01, 0.05), 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--umd-batch", type=int, default=2)
    p.add_argument("--umd-budget", type=int, default=1200, help="body tokens for umd_full (4,000 chars fit)")
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-beta", type=int, default=500, help="fit posts per class scored to estimate beta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sets", default="matched,minpairs,umd_seen,umd_full")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "yes_bias"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_fit, a.n_eval, a.n_beta = 200, 40, 40
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sens_at_spec(s, y, spec=0.9):
    return float((s[y == 1] > np.quantile(s[y == 0], spec)).mean())


def ols_beta(s, nt):
    nt_c = nt - nt.mean()
    return float((nt_c @ (s - s.mean())) / (nt_c @ nt_c + 1e-9))


def auc_beta(s, nt, y):
    vals = [auc((s - b * nt)[y == 1], (s - b * nt)[y == 0]) for b in BETAS]
    return float(BETAS[int(np.argmax(vals))])


def summarize(scores, y, ref=None, seed=0):
    """AUC, CI, difference vs ref (same items), sens at 90% spec, and specificity at threshold 0."""
    out = {}
    for k, s in scores.items():
        sp, sn = s[y == 1], s[y == 0]
        r = {"auc": auc(sp, sn), "ci95": boot_auc_ci(sp, sn, seed=seed),
             "sens_at_90spec": sens_at_spec(s, y),
             "spec_at_0": float((sn <= 0).mean()), "sens_at_0": float((sp > 0).mean())}
        if ref is not None:
            r["minus_ref_ci95"] = boot_auc_diff_ci(sp, sn, ref[y == 1], ref[y == 0], seed=seed)
        out[k] = r
        log(f"    {k:22s} AUC {r['auc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            + (f"  vs ref {r['minus_ref_ci95'][0]:+.3f}..{r['minus_ref_ci95'][1]:+.3f}" if ref is not None else "")
            + f"  sens@90 {r['sens_at_90spec']:.3f}  spec@0 {r['spec_at_0']:.3f}")
    return out


def main():
    a = parse_args()
    t0 = time.time()
    sets = [s for s in a.sets.split(",") if s]
    tag = a.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    be = Backend(a.model, dtype=a.dtype)
    c = get_concept("suicide_depression")
    cs = get_concept("suicide")   # minpairs and UMD use the 232K prompt, as in the paper
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    log(f"yes_bias_audit on {a.model}: {be.n_layers} layers; sets {sets}")

    def score(texts, budget, batch, targets):
        be.truncate_side, be.body_budget = c.truncate, budget
        out = {}
        for name, (pre, q) in targets.items():
            out[name] = be.logit_diff([be.polar_prompt(t, pre, q) for t in texts], yes, no, batch)
        for name, pre, q in NONTOPIC_QUESTIONS:
            out[f"nt_{name}"] = be.logit_diff([be.polar_prompt(t, pre, q) for t in texts], yes, no, batch)
        out["nt_mean"] = np.mean([out[f"nt_{n}"] for n, _, _ in NONTOPIC_QUESTIONS], 0)
        return out

    results = {"model": a.model, "config": vars(a), "sets": {}}
    npz = {}

    # ---------------------------------------------------------------- matched register
    if "matched" in sets:
        pos, neg = c.loader(a.n_fit + a.n_eval, a.seed)
        fit_p, fit_n = pos[:a.n_fit], neg[:a.n_fit]
        ev = pos[a.n_fit:a.n_fit + a.n_eval] + neg[a.n_fit:a.n_fit + a.n_eval]
        y = np.r_[np.ones(a.n_eval), np.zeros(a.n_eval)]
        bfit = fit_p[:a.n_beta] + fit_n[:a.n_beta]
        yb = np.r_[np.ones(a.n_beta), np.zeros(a.n_beta)]
        tf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                           LogisticRegression(C=1.0, max_iter=2000))
        tf.fit(fit_p + fit_n, [1] * a.n_fit + [0] * a.n_fit)
        tf_s = tf.predict_proba(ev)[:, 1]
        targets = {"base": (c.prefix, c.question), "contrast": CONTRAST}
        log("matched: scoring eval and beta-fit posts")
        E, F = score(ev, c.body_tokens, a.batch, targets), score(bfit, c.body_tokens, a.batch, targets)
        S, betas = {"tfidf": tf_s}, {}
        for t in targets:
            bo, ba = ols_beta(F[t], F["nt_mean"]), auc_beta(F[t], F["nt_mean"], yb)
            betas[t] = {"ols": bo, "auc_fit": ba}
            S[f"{t}_raw"] = E[t]
            S[f"{t}_minus_nt"] = E[t] - E["nt_mean"]
            S[f"{t}_resid_ols"] = E[t] - bo * E["nt_mean"]
            S[f"{t}_beta_cv"] = E[t] - ba * E["nt_mean"]
        S["nt_mean_alone"] = E["nt_mean"]   # how much do the always-No questions themselves separate the classes?
        log(f"matched betas {betas}")
        folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(ev, y))

        def stack(x):
            X, o = np.c_[tf_s, (x - x.mean()) / (x.std() + 1e-9)], np.zeros(len(y))
            for tr, te in folds:
                o[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            return o

        rows = summarize(S, y, ref=tf_s, seed=a.seed)
        for k in [k for k in S if k.endswith(("_raw", "_minus_nt"))]:
            st = stack(S[k])
            rows[k]["stack_auc"] = auc(st[y == 1], st[y == 0])
        results["sets"]["matched"] = {"n_eval_per_class": a.n_eval, "betas": betas, "rows": rows,
                                      "nt_frac_yes": {k: float((E[k] > 0).mean()) for k in E if k.startswith("nt_")}}
        npz.update({f"matched_{k}": v for k, v in {**S, **{f"E_{k}": v for k, v in E.items()}}.items()})
        npz["matched_y"] = y

    # ---------------------------------------------------------------- minimal pairs
    if "minpairs" in sets:
        rows_mp = [json.loads(l) for l in MINPAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
        texts = [r["text"] for r in rows_mp]
        cond = np.array([r["condition"] for r in rows_mp])
        E = score(texts, c.body_tokens, a.batch, {"base": (cs.prefix, cs.question)})
        S = {"raw": E["base"], "minus_nt": E["base"] - E["nt_mean"]}
        mp = {"yes_rate": defaultdict(dict), "genuine_vs": defaultdict(dict)}
        for k, s in S.items():
            for cd in sorted(set(cond)):
                mp["yes_rate"][cd][k] = float((s[cond == cd] > 0).mean())
                if cd != "genuine_intent":
                    mp["genuine_vs"][cd][k] = auc(s[cond == "genuine_intent"], s[cond == cd])
        log(f"minpairs yes rate {json.dumps(mp['yes_rate'])}")
        results["sets"]["minpairs"] = mp
        npz.update({f"minpairs_{k}": v for k, v in E.items()})
        npz["minpairs_condition"] = cond

    # ---------------------------------------------------------------- UMD, seen and full text
    umd_sets = [s for s in sets if s.startswith("umd_")]
    if umd_sets:
        recs = load_umd_with_severity(source="expert")
        if a.smoke:
            recs = recs[::10]
        full = [r["text"] for r in recs]
        hi = (np.array([r["severity"] for r in recs]) >= 2).astype(int)
        npz["umd_severity"] = np.array([r["severity"] for r in recs])
        for us in umd_sets:
            budget = c.body_tokens if us == "umd_seen" else a.umd_budget
            be.truncate_side, be.body_budget = c.truncate, budget
            seen = [be._truncate(t) for t in full]
            n_cut = int(sum(len(be.tok(t, add_special_tokens=False)["input_ids"]) > budget for t in full))
            log(f"{us}: {len(recs)} users, budget {budget} tokens, {n_cut} users truncated")
            E = score(full, budget, a.umd_batch if us == "umd_full" else a.batch, {"base": (cs.prefix, cs.question)})
            tfcv = np.zeros(len(hi))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=a.seed).split(np.zeros(len(hi)), hi):
                m = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2),
                                  LogisticRegression(C=1.0, max_iter=2000)).fit([seen[i] for i in tr], hi[tr])
                tfcv[te] = m.predict_proba([seen[i] for i in te])[:, 1]
            S = {"tfidf_cv": tfcv, "raw": E["base"], "minus_nt": E["base"] - E["nt_mean"]}
            results["sets"][us] = {"budget_tokens": budget, "n_users": len(recs), "n_truncated": n_cut,
                                   "rows": summarize(S, hi.astype(float), ref=tfcv, seed=a.seed)}
            npz.update({f"{us}_{k}": v for k, v in {**S, **{f"E_{k}": v for k, v in E.items()}}.items()})

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{tag}__seed{a.seed}{'_smoke' if a.smoke else ''}"
    results["runtime_min"] = round((time.time() - t0) / 60, 1)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(results, open(f"{stem}.json", "w"), indent=1)
    np.savez_compressed(f"{stem}.npz", **npz)
    log(f"saved {stem}.json ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
