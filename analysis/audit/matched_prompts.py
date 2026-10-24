#!/usr/bin/env python
"""Paper 2: is the matched-register failure (F6) a prompt artifact?

Same data and split as the p2_matched gate (suicide vs depression, n_fit 2000, n_eval 1000 per
class, seed 0). The model's judgment on the eval set is scored under several prompt formats:
  base       the gate prompt (reproduces F6)
  neg_flip   the negated question; score = -(Yes - No), a polarity / Yes-bias check
  clinical   C-SSRS-style definition of suicidal ideation, then the question
  contrast   asks explicitly for suicidal thoughts as opposed to depression without them
  escalate   triage framing: should a crisis counselor be alerted?
  fewshot    four labeled examples from the fit split (2 per class), then the question
  rating     1-5 suicide-risk rating; score = expected rating over the digit tokens
Each variant: AUC with bootstrap CI, and CI of (variant - TF-IDF). Then "stack": 5-fold CV
logistic regression on [TF-IDF probability, variant score] over the eval set, asking whether the
LLM adds anything beyond lexical cues (stacked AUC vs TF-IDF alone, same folds).

Usage: python matched_prompts.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from backend import Backend
from concepts import PROJECT_ROOT, get_concept
from stats import auc, boot_auc_ci, boot_auc_diff_ci

CLINICAL = ("Suicidal ideation means thoughts of wanting to die or to end one's own life. It includes "
            "passive wishes to be dead or not wake up, active thoughts of killing oneself, and any "
            "method, plan or intent. Sadness, hopelessness or other symptoms of depression without "
            "such thoughts do not count.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--concept", default="suicide_depression")
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--variants", default="base,neg_flip,clinical,contrast,escalate,fewshot,rating")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "paper2" / "prompt_invariance"))
    p.add_argument("--smoke", action="store_true")
    # full-text rerun: body-token budget for the scored post (0 = the concept's 176); few-shot examples keep 176
    p.add_argument("--body-tokens", type=int, default=0)
    p.add_argument("--max-batch-tokens", type=int, default=0, help="length-sorted batches capped at this many tokens")
    p.add_argument("--out-suffix", default="")
    a = p.parse_args()
    if a.smoke:
        a.n_fit, a.n_eval = 200, 40
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    a = parse_args()
    t0 = time.time()
    c = get_concept(a.concept)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    pos, neg = c.loader(a.n_fit + a.n_eval, a.seed)
    fit_p, fit_n = pos[:a.n_fit], neg[:a.n_fit]
    ev_p, ev_n = pos[a.n_fit:a.n_fit + a.n_eval], neg[a.n_fit:a.n_fit + a.n_eval]
    texts = ev_p + ev_n
    y = np.r_[np.ones(len(ev_p)), np.zeros(len(ev_n))]
    log(f"{a.concept}: fit {len(fit_p)}/{len(fit_n)}, eval {len(ev_p)}/{len(ev_n)}")

    tf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                       LogisticRegression(C=1.0, max_iter=2000))
    tf.fit(fit_p + fit_n, [1] * len(fit_p) + [0] * len(fit_n))
    tf_s = tf.predict_proba(texts)[:, 1]
    tf_auc = auc(tf_s[y == 1], tf_s[y == 0])
    log(f"TF-IDF AUC {tf_auc:.3f}")

    be = Backend(a.model, dtype=a.dtype)
    be.truncate_side, be.body_budget = c.truncate, c.body_tokens
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    # each digit as "1" and " 1": the last token of each encoding (SentencePiece may split off "▁")
    digits = [sorted({be.tok.encode(f, add_special_tokens=False)[-1] for f in (str(k), f" {k}")})
              for k in range(1, 6)]

    def polar(prefix, question):
        return [be.polar_prompt(t, prefix, question) for t in texts]

    rng = np.random.default_rng(a.seed + 11)
    shots = [(fit_p[i], "Yes") for i in rng.choice(len(fit_p), 2, replace=False)] + \
            [(fit_n[i], "No") for i in rng.choice(len(fit_n), 2, replace=False)]
    shots = [shots[i] for i in rng.permutation(4)]
    shot_block = "\n\n".join(f"Text: {be._truncate(t)}\nShows signs of suicidal ideation? {ans}" for t, ans in shots)
    n_long = int(sum(len(be.tok(t, add_special_tokens=False)["input_ids"]) > c.body_tokens for t in texts))
    n_cut = n_long
    if a.body_tokens:
        be.body_budget = a.body_tokens
        n_cut = int(sum(len(be.tok(t, add_special_tokens=False)["input_ids"]) > a.body_tokens for t in texts))
    log(f"body budget {be.body_budget} tokens: {n_cut} of {len(texts)} eval posts truncated "
        f"({n_long} exceed {c.body_tokens})")

    def score_logits(prompts):
        if not a.max_batch_tokens:
            return be.last_logits(prompts, a.batch)
        lens = [len(be.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = np.argsort(lens)[::-1]   # longest first, so an OOM shows up at once
        out, i = [None] * len(prompts), 0
        while i < len(order):
            n = max(1, min(a.batch, a.max_batch_tokens // lens[order[i]]))
            idx = order[i:i + n]
            lg = be.last_logits([prompts[j] for j in idx], n)
            for k, j in enumerate(idx):
                out[j] = lg[k]
            i += n
        return torch.stack(out)

    builders = {
        "base": lambda: (polar(c.prefix, c.question), "ld"),
        "neg_flip": lambda: (polar(c.prefix, c.neg_question), "neg_ld"),
        "clinical": lambda: (polar(f"{CLINICAL}\n\nRead the following text.",
                                   "Does the writer express suicidal ideation, as defined above?"), "ld"),
        "contrast": lambda: (polar("The writer of the following text is struggling with their mental health.",
                                   "Is the writer expressing suicidal thoughts, as opposed to depression "
                                   "without suicidal thoughts?"), "ld"),
        "escalate": lambda: (polar("You are helping a crisis-support team triage posts.",
                                   "Should this post be escalated to a crisis counselor because the writer "
                                   "may be at risk of suicide?"), "ld"),
        "fewshot": lambda: ([be._wrap_chat(
            f"Decide whether each text shows signs of suicidal ideation.\n\n{shot_block}\n\n"
            f"Text: {be._truncate(t)}\nShows signs of suicidal ideation? Answer with Yes or No.\nAnswer:")
            for t in texts], "ld"),
        "rating": lambda: ([be._wrap_chat(
            f"Read the following text.\n\nText: {be._truncate(t)}\n\nOn a scale from 1 (no suicide risk) "
            f"to 5 (high suicide risk), how much suicide risk does the writer show? "
            f"Answer with a single digit.\nAnswer:") for t in texts], "rating"),
    }

    results = {"model": a.model, "model_tag": tag, "concept": a.concept, "config": vars(a),
               "n_eval_per_class": len(ev_p), "tfidf_auc": tf_auc,
               "body_budget": be.body_budget, "n_eval_truncated": n_cut,
               "tfidf_ci95": boot_auc_ci(tf_s[y == 1], tf_s[y == 0], seed=a.seed), "variants": {}}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(texts, y))

    def cv_pred(X):
        out = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
        return out

    tf_pred = cv_pred(tf_s[:, None])
    tf_cv = auc(tf_pred[y == 1], tf_pred[y == 0])
    results["tfidf_cv_auc"] = tf_cv
    scores = {"y": y, "tfidf": tf_s}
    for v in [x for x in a.variants.split(",") if x]:
        prompts, kind = builders[v]()
        lg = score_logits(prompts)
        if kind == "rating":
            dl = torch.stack([torch.logsumexp(lg[:, ids].float(), -1) for ids in digits], -1)
            p = torch.softmax(dl, -1).numpy()
            s = p @ np.arange(1, 6)
        else:
            s = (lg[:, yes] - lg[:, no]).float().numpy()
            s = -s if kind == "neg_ld" else s
        sp, sn = s[y == 1], s[y == 0]
        X = np.c_[tf_s, (s - s.mean()) / (s.std() + 1e-9)]
        st = cv_pred(X)
        scores[v] = s
        row = {"auc": auc(sp, sn), "ci95": boot_auc_ci(sp, sn, seed=a.seed),
               "minus_tfidf_ci95": boot_auc_diff_ci(sp, sn, tf_s[y == 1], tf_s[y == 0], seed=a.seed),
               "stack_cv_auc": auc(st[y == 1], st[y == 0]), "tfidf_cv_auc": tf_cv,
               # paired bootstrap over eval texts: does adding the LLM score to TF-IDF help?
               "stack_minus_tfidf_ci95": boot_auc_diff_ci(st[y == 1], st[y == 0], tf_pred[y == 1],
                                                          tf_pred[y == 0], seed=a.seed),
               "score_mean_pos": float(sp.mean()), "score_mean_neg": float(sn.mean()),
               "frac_yes_pos": float((sp > 0).mean()) if kind != "rating" else None,
               "frac_yes_neg": float((sn > 0).mean()) if kind != "rating" else None}
        results["variants"][v] = row
        log(f"{v:9s} AUC {row['auc']:.3f} {np.round(row['ci95'], 3).tolist()} "
            f"vs TF-IDF diff {np.round(row['minus_tfidf_ci95'], 3).tolist()}; "
            f"stack {row['stack_cv_auc']:.3f} vs TF-IDF {tf_cv:.3f} "
            f"(gain CI {np.round(row['stack_minus_tfidf_ci95'], 3).tolist()})")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{a.concept}__{tag}__seed{a.seed}{a.out_suffix}{'_smoke' if a.smoke else ''}.json"
    path.write_text(json.dumps(results, indent=1))
    np.savez_compressed(path.with_suffix(".npz"), **scores)   # per-text scores for later analyses
    log(f"saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
