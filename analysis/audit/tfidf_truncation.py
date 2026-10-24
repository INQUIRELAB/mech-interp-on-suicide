"""Does the register-matched TF-IDF advantage survive when TF-IDF sees the same truncated text as the LLMs?

matched_prompts.py fits and scores TF-IDF on full posts, while each LLM sees only the first 176 tokens of a
post (concepts.py body_tokens; backend._truncate: encode without special tokens, keep the first 176 ids,
decode). This script rebuilds the SWMH r/SuicideWatch vs r/depression split (same filters and seeds as
matched_prompts.py / swmh_label_noise_probe.py), truncates every post with each model's own tokenizer
exactly as the backend does, and refits TF-IDF:
  trunc   fit on truncated fitting posts, scored on truncated evaluation posts (matched input)
  fulltr  fit on full fitting posts, scored on truncated evaluation posts
The saved per-post LLM scores (prompt_invariance/*.npz) are then compared with the matched TF-IDF with the
same statistics as the paper: AUC with bootstrap CI, paired bootstrap CI of the difference, 5-fold CV stack,
sensitivity at 90% specificity (class-stratified bootstrap) and threshold calibration with 100 posts per class.

Usage: python tfidf_truncation.py <SWMH train.jsonl> <prompt_invariance dir> <tokenizer root> <out.json>
  tokenizer root holds one folder per model tag (llama_3_1_8b_instruct, qwen2_5_7b_instruct,
  mistral_7b_instruct_v0_3, gemma_2_9b_it) with the tokenizer files of the model used in the run.
"""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc, boot_auc_ci, boot_auc_diff_ci  # noqa: E402

TRAIN, PI_DIR, TOK_ROOT, OUT = sys.argv[1:5]
BUDGET, SEED = 176, 0
VARIANTS = ["base", "neg_flip", "clinical", "contrast", "escalate", "fewshot", "rating"]
MODELS = ["llama_3_1_8b_instruct", "qwen2_5_7b_instruct", "mistral_7b_instruct_v0_3", "gemma_2_9b_it"]


def clean(t, n, seed):
    rng = np.random.default_rng(seed)
    out = [x for x in t if x and len(x) > 80]
    rng.shuffle(out)
    return out[:n]


sui, dep = [], []
for line in open(TRAIN, encoding="utf-8"):
    r = json.loads(line)
    lab, txt = r.get("label", ""), (r.get("text") or "").strip()
    if lab == "self.SuicideWatch":
        sui.append(txt)
    elif lab == "self.depression":
        dep.append(txt)
pos, neg = clean(sui, 3000, SEED), clean(dep, 3000, SEED + 1)
fit, ev = pos[:2000] + neg[:2000], pos[2000:3000] + neg[2000:3000]
y_fit = np.r_[np.ones(2000), np.zeros(2000)].astype(int)
y = np.r_[np.ones(1000), np.zeros(1000)].astype(int)
P, N = y == 1, y == 0


def tfidf(fit_texts, eval_texts):
    m = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                      LogisticRegression(C=1.0, max_iter=2000))
    m.fit(fit_texts, y_fit)
    return m.predict_proba(eval_texts)[:, 1]


folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ev, y))


def cv_pred(X):
    out = np.zeros(len(y))
    for tr, te in folds:
        out[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def sens_at_spec(yy, s, spec=0.90):
    return float((s[yy == 1] > np.quantile(s[yy == 0], spec)).mean())


def boot_idx(B=1000, seed=1):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(P)[0], np.where(N)[0]
    return [np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))] for _ in range(B)]


IDX = boot_idx()


def sens_ci(s):
    return np.percentile([sens_at_spec(y[i], s[i]) for i in IDX], [2.5, 97.5]).tolist()


def sens_diff_ci(a, b):
    return np.percentile([sens_at_spec(y[i], a[i]) - sens_at_spec(y[i], b[i]) for i in IDX], [2.5, 97.5]).tolist()


def calibrated_sens(s, n=100, R=500, seed=0):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(P)[0], np.where(N)[0]
    sp, se = np.empty(R), np.empty(R)
    for r in range(R):
        cp, cn = rng.choice(pi, n, replace=False), rng.choice(ni, n, replace=False)
        tp, tn = np.setdiff1d(pi, cp), np.setdiff1d(ni, cn)
        thr = np.quantile(s[cn], 0.9)
        sp[r], se[r] = (s[tn] <= thr).mean(), (s[tp] > thr).mean()
    return {"specificity": float(sp.mean()), "sensitivity": float(se.mean())}


full = tfidf(fit, ev)
per_post = {"y": y, "tfidf_full": full}
res = {"budget_tokens": BUDGET, "n_eval_per_class": 1000,
       "tfidf_full": {"auc": auc(full[P], full[N]), "ci95": boot_auc_ci(full[P], full[N], seed=SEED),
                      "sens90": sens_at_spec(y, full), "sens90_ci95": sens_ci(full)},
       "models": {}}
print(f"full-text TF-IDF AUC {res['tfidf_full']['auc']:.4f} sens90 {res['tfidf_full']['sens90']:.3f}", flush=True)

for tag in MODELS:
    z = np.load(os.path.join(PI_DIR, f"suicide_depression__{tag}__seed0.npz"))
    assert (z["y"].astype(int) == y).all()
    r_align = float(np.corrcoef(full, z["tfidf"])[0, 1])
    assert r_align > 0.999, f"{tag}: rebuilt split does not match the saved scores (r={r_align})"
    tok = AutoTokenizer.from_pretrained(os.path.join(TOK_ROOT, tag))
    n_tok = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in ev])

    def cut(t):
        return tok.decode(tok(t, add_special_tokens=False)["input_ids"][:BUDGET])

    fit_c, ev_c = [cut(t) for t in fit], [cut(t) for t in ev]
    tr = tfidf(fit_c, ev_c)          # matched input
    ft = tfidf(fit, ev_c)            # full-text fit, truncated evaluation
    tr_cv = cv_pred(tr[:, None])
    per_post[f"tfidf_trunc__{tag}"] = tr
    per_post[f"ntok__{tag}"] = n_tok
    row = {"tokenizer": tok.name_or_path, "align_r_saved_tfidf": r_align,
           "frac_truncated": {"SuicideWatch": float((n_tok[P] > BUDGET).mean()),
                              "depression": float((n_tok[N] > BUDGET).mean()),
                              "all": float((n_tok > BUDGET).mean())},
           "median_tokens": {"SuicideWatch": float(np.median(n_tok[P])), "depression": float(np.median(n_tok[N]))},
           "tfidf_trunc": {"auc": auc(tr[P], tr[N]), "ci95": boot_auc_ci(tr[P], tr[N], seed=SEED),
                           "minus_full_ci95": boot_auc_diff_ci(tr[P], tr[N], full[P], full[N], seed=SEED),
                           "sens90": sens_at_spec(y, tr), "sens90_ci95": sens_ci(tr),
                           "calibrated_n100": calibrated_sens(tr), "cv_auc": auc(tr_cv[P], tr_cv[N])},
           "tfidf_fullfit_trunceval": {"auc": auc(ft[P], ft[N]), "ci95": boot_auc_ci(ft[P], ft[N], seed=SEED)},
           "variants": {}}
    print(f"\n{tag}: truncated {row['frac_truncated']['all']:.3f} "
          f"(SW {row['frac_truncated']['SuicideWatch']:.3f}, dep {row['frac_truncated']['depression']:.3f}); "
          f"TF-IDF trunc AUC {row['tfidf_trunc']['auc']:.4f} {np.round(row['tfidf_trunc']['ci95'], 3).tolist()} "
          f"sens90 {row['tfidf_trunc']['sens90']:.3f}; full-fit/trunc-eval {row['tfidf_fullfit_trunceval']['auc']:.4f}",
          flush=True)
    for v in VARIANTS:
        s = z[v].astype(float)
        st = cv_pred(np.c_[tr, (s - s.mean()) / (s.std() + 1e-9)])
        vr = {"auc": auc(s[P], s[N]),
              "minus_tfidf_trunc_ci95": boot_auc_diff_ci(s[P], s[N], tr[P], tr[N], seed=SEED),
              "minus_tfidf_full_ci95": boot_auc_diff_ci(s[P], s[N], full[P], full[N], seed=SEED),
              "stack_trunc_cv_auc": auc(st[P], st[N]),
              "stack_minus_tfidf_trunc_ci95": boot_auc_diff_ci(st[P], st[N], tr_cv[P], tr_cv[N], seed=SEED)}
        if v in ("base", "clinical", "contrast"):
            vr["sens90"] = sens_at_spec(y, s)
            vr["sens90_minus_tfidf_trunc_ci95"] = sens_diff_ci(s, tr)
            vr["calibrated_n100"] = calibrated_sens(s)
            vr["stack_trunc_sens90"] = sens_at_spec(y, st)
        row["variants"][v] = vr
        print(f"  {v:9s} AUC {vr['auc']:.3f} - trunc TF-IDF {np.round(vr['minus_tfidf_trunc_ci95'], 3).tolist()}"
              f" | stack {vr['stack_trunc_cv_auc']:.3f}"
              + (f" | sens90 {vr['sens90']:.3f} diff {np.round(vr['sens90_minus_tfidf_trunc_ci95'], 3).tolist()}"
                 if "sens90" in vr else ""), flush=True)
    res["models"][tag] = row

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
np.savez_compressed(os.path.splitext(OUT)[0] + ".npz", **per_post)   # per-post scores for figures and tables
print("saved", OUT)
