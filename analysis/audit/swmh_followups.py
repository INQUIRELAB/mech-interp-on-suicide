#!/usr/bin/env python
"""Follow-up checks on the register-matched set (SWMH r/SuicideWatch vs r/depression), CPU only.

Rebuilds the split of matched_prompts.py (same filters and seeds; asserts that the rebuilt full-post TF-IDF
scores match the saved ones) and reads the saved full-post LLM scores (prompt_invariance_full/*.npz).

  learning_curve   TF-IDF fit on n posts per class (random subsets of the fitting split), scored on the
                   1000 + 1000 evaluation posts: how many labeled posts the zero-shot LLM answer is worth
  duplicates       exact and near-duplicate posts (token containment >= 0.9, at least 15 word types) between
                   the two classes and between the fitting and evaluation splits; AUCs recomputed without the
                   flagged evaluation posts
  stack_control    does stacking TF-IDF with a second, non-LLM score (character n-gram TF-IDF; an explicit-term
                   lexicon) improve on TF-IDF as much as stacking it with the LLM score?
  calibration      the default answer read as a probability, P(yes) = sigmoid(Yes - No logit): mean P(yes) per
                   class, calibration slope and intercept against the forum labels, Brier score

Usage: python swmh_followups.py <SWMH train.jsonl> <prompt_invariance_full dir> <out.json>
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc, boot_auc_diff_ci  # noqa: E402

TRAIN, PI_DIR, OUT = sys.argv[1:4]
SEED = 0
MODELS = ["llama_3_1_8b_instruct", "qwen2_5_7b_instruct", "mistral_7b_instruct_v0_3", "gemma_2_9b_it"]
VARIANTS = ["base", "clinical", "contrast"]
SUICIDE_TERMS = re.compile(
    r"\b(suicid\w*|kill(?:ing)? myself|end(?:ing)? (?:my life|it all)|end it|take my (?:own )?life|"
    r"want(?:ed)? to die|wanna die|better off dead|overdos\w*|hang(?:ing)? myself|not wake up)\b", re.I)


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


def tfidf_model():
    return make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                         LogisticRegression(C=1.0, max_iter=2000))


full_model = tfidf_model().fit(fit, y_fit)
full = full_model.predict_proba(ev)[:, 1]
Z = {m: np.load(os.path.join(PI_DIR, f"suicide_depression__{m}__seed0_full.npz")) for m in MODELS}
for m, z in Z.items():
    assert (z["y"].astype(int) == y).all()
    r_align = float(np.corrcoef(full, z["tfidf"])[0, 1])
    assert r_align > 0.999, f"{m}: rebuilt split does not match the saved scores (r={r_align})"
res = {"n_eval_per_class": 1000, "tfidf_full_auc": auc(full[P], full[N])}
print(f"TF-IDF full AUC {res['tfidf_full_auc']:.4f}", flush=True)


def sens_at_spec(yy, s, spec=0.90):
    return float((s[yy == 1] > np.quantile(s[yy == 0], spec)).mean())


# ------------------------------------------------------------------ learning curve
rng = np.random.default_rng(SEED)
fit_pos_idx, fit_neg_idx = np.arange(2000), np.arange(2000, 4000)
lc = {}
for n, reps in ((25, 50), (50, 50), (100, 30), (250, 20), (500, 10), (1000, 10), (2000, 1)):
    aucs, sens = [], []
    for _ in range(reps):
        ip = rng.choice(fit_pos_idx, n, replace=False) if n < 2000 else fit_pos_idx
        ineg = rng.choice(fit_neg_idx, n, replace=False) if n < 2000 else fit_neg_idx
        idx = np.r_[ip, ineg]
        m = tfidf_model().fit([fit[i] for i in idx], y_fit[idx])
        s = m.predict_proba(ev)[:, 1]
        aucs.append(auc(s[P], s[N]))
        sens.append(sens_at_spec(y, s))
    lc[n] = {"reps": reps, "auc_mean": float(np.mean(aucs)), "auc_p2_5": float(np.percentile(aucs, 2.5)),
             "auc_p97_5": float(np.percentile(aucs, 97.5)), "sens90_mean": float(np.mean(sens))}
    print(f"learning curve n={n}: AUC {lc[n]['auc_mean']:.3f} [{lc[n]['auc_p2_5']:.3f}, {lc[n]['auc_p97_5']:.3f}]",
          flush=True)
llm_auc = {m: {v: auc(Z[m][v][P], Z[m][v][N]) for v in VARIANTS} for m in MODELS}
ns = sorted(lc)


def labels_equivalent(a):
    """Smallest n per class at which mean TF-IDF AUC reaches a (linear interpolation on log n)."""
    xs, ys = np.log(ns), np.array([lc[n]["auc_mean"] for n in ns])
    if a <= ys[0]:
        return f"<= {ns[0]}"
    if a > ys[-1]:
        return f"> {ns[-1]}"
    k = int(np.searchsorted(ys, a))
    t = (a - ys[k - 1]) / (ys[k] - ys[k - 1])
    return float(np.exp(xs[k - 1] + t * (xs[k] - xs[k - 1])))


res["learning_curve"] = {"tfidf": lc, "llm_auc_full_posts": llm_auc,
                         "labels_per_class_equivalent": {m: {v: labels_equivalent(a) for v, a in d.items()}
                                                         for m, d in llm_auc.items()}}
print("labels/class equivalent:", res["learning_curve"]["labels_per_class_equivalent"], flush=True)


# ------------------------------------------------------------------ duplicates
def norm_text(t):
    return " ".join(re.findall(r"[a-z0-9']+", t.lower()))


def types(t):
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def h(t):
    return hashlib.md5(norm_text(t).encode()).hexdigest()


all_sui = [x for x in sui if x and len(x) > 80]
all_dep = [x for x in dep if x and len(x) > 80]
hs, hd = defaultdict(int), defaultdict(int)
for t in all_sui:
    hs[h(t)] += 1
for t in all_dep:
    hd[h(t)] += 1
cross_exact = sum(1 for k in hs if k in hd)
dup = {"swmh_train_posts": {"SuicideWatch": len(all_sui), "depression": len(all_dep)},
       "exact_text_in_both_forums_full_train": cross_exact}

ev_types = [types(t) for t in ev]
fit_types = [types(t) for t in fit]


def contained(a, b, thr=0.9):
    if len(a) < 15:
        return False
    return len(a & b) / len(a) >= thr


inv = defaultdict(list)          # inverted index on word types -> fit post ids (rarest types only)
df = defaultdict(int)
for s in fit_types:
    for w in s:
        df[w] += 1
for i, s in enumerate(fit_types):
    for w in sorted(s, key=lambda w: df[w])[:8]:
        inv[w].append(i)


def near_dup_in_fit(e):
    cands = set()
    for w in sorted(e, key=lambda w: df.get(w, 0))[:8]:
        cands.update(inv.get(w, ()))
    return [i for i in cands if contained(e, fit_types[i]) or contained(fit_types[i], e)]


flag_fit = np.zeros(len(ev), bool)
flag_fit_other_class = np.zeros(len(ev), bool)
for k, e in enumerate(ev_types):
    hits = near_dup_in_fit(e)
    if hits:
        flag_fit[k] = True
        flag_fit_other_class[k] = any(y_fit[i] != y[k] for i in hits)
flag_cross_eval = np.zeros(len(ev), bool)
neg_idx = np.where(N)[0]
for i in np.where(P)[0]:
    for j in neg_idx:
        if contained(ev_types[i], ev_types[j]) or contained(ev_types[j], ev_types[i]):
            flag_cross_eval[i] = flag_cross_eval[j] = True
flag = flag_fit | flag_cross_eval
dup.update({"eval_near_dup_of_fit_post": int(flag_fit.sum()),
            "eval_near_dup_of_fit_post_other_class": int(flag_fit_other_class.sum()),
            "eval_near_dup_across_classes_within_eval": int(flag_cross_eval.sum()),
            "eval_flagged_any": int(flag.sum()),
            "eval_flagged_by_class": {"SuicideWatch": int(flag[P].sum()), "depression": int(flag[N].sum())}})
keep = ~flag
Pk, Nk = keep & P, keep & N
tfidf_kept = auc(full[Pk], full[Nk])
dup["auc_without_flagged"] = {"tfidf_full": tfidf_kept,
                              "models": {m: {v: {"auc": auc(Z[m][v][Pk], Z[m][v][Nk]),
                                                 "minus_tfidf_ci95": boot_auc_diff_ci(Z[m][v][Pk], Z[m][v][Nk],
                                                                                      full[Pk], full[Nk], seed=SEED)}
                                             for v in VARIANTS} for m in MODELS}}
res["duplicates"] = dup
print("duplicates:", {k: v for k, v in dup.items() if k != "auc_without_flagged"}, flush=True)
print(f"  TF-IDF without flagged {tfidf_kept:.4f}", flush=True)

# ------------------------------------------------------------------ stacking control
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(ev, y))


def cv_pred(X):
    out = np.zeros(len(y))
    for tr, te in folds:
        out[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def z(s):
    return (s - s.mean()) / (s.std() + 1e-9)


char = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000),
                     LogisticRegression(C=1.0, max_iter=2000)).fit(fit, y_fit).predict_proba(ev)[:, 1]
lex = np.array([1000.0 * len(SUICIDE_TERMS.findall(t)) / max(len(t), 1) for t in ev])
tf_cv = cv_pred(full[:, None])
sc = {"tfidf_cv_auc": auc(tf_cv[P], tf_cv[N]), "char_tfidf_auc": auc(char[P], char[N]),
      "lexicon_auc": auc(lex[P], lex[N]), "stacks": {}}
for name, s in [("char_tfidf", char), ("lexicon", lex)] + \
               [(f"{m}|{v}", Z[m][v].astype(float)) for m in MODELS for v in ("base", "contrast")]:
    st = cv_pred(np.c_[full, z(s)])
    sc["stacks"][name] = {"stack_auc": auc(st[P], st[N]),
                          "gain_ci95": boot_auc_diff_ci(st[P], st[N], tf_cv[P], tf_cv[N], seed=SEED)}
st3 = cv_pred(np.c_[full, z(char), z(lex)])
sc["stacks"]["char_tfidf+lexicon"] = {"stack_auc": auc(st3[P], st3[N]),
                                      "gain_ci95": boot_auc_diff_ci(st3[P], st3[N], tf_cv[P], tf_cv[N], seed=SEED)}
res["stack_control"] = sc
print("stack control:", {k: round(v["stack_auc"], 4) for k, v in sc["stacks"].items()}, flush=True)


# ------------------------------------------------------------------ calibration of the default answer
def calib(ld):
    p = 1 / (1 + np.exp(-ld))
    m = LogisticRegression(C=1e6, max_iter=5000).fit(ld[:, None], y)
    return {"mean_p_yes_SuicideWatch": float(p[P].mean()), "mean_p_yes_depression": float(p[N].mean()),
            "calibration_slope": float(m.coef_[0, 0]), "calibration_intercept": float(m.intercept_[0]),
            "brier": float(np.mean((p - y) ** 2))}


res["calibration_full_posts"] = {m: {v: calib(Z[m][v].astype(float)) for v in VARIANTS} for m in MODELS}
print("calibration:", {m: {v: round(d["calibration_slope"], 3) for v, d in r.items()}
                       for m, r in res["calibration_full_posts"].items()}, flush=True)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, default=float)
print("saved", OUT)
