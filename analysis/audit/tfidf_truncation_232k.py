"""Suicide-Watch-232K: TF-IDF on the same 176-token truncation the Llama answer saw.

Rebuilds the run_audit.py split (concept "suicide": 2000 posts per class longer than 80 characters, seeds 0/1;
first 1000 per class fit, next 1000 evaluate), checks that full-text TF-IDF reproduces the saved AUC, then refits
TF-IDF on truncated text. The run_audit.py output keeps only summary AUCs, not per-post model scores, so the
model-minus-TF-IDF difference cannot be paired here.

Usage: python tfidf_truncation_232k.py <232K hf_cache dir> <llama tokenizer dir> <audit json> <out.json>
"""
import json
import os
import sys

import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import auc, boot_auc_ci, boot_auc_diff_ci  # noqa: E402

HF, TOK, AUDIT, OUT = sys.argv[1:5]
BUDGET = 176


def clean(t, n, seed):
    rng = np.random.default_rng(seed)
    out = [x for x in t if x and len(x) > 80]
    rng.shuffle(out)
    return out[:n]


ds = load_from_disk(HF)["train"]
texts, labels = ds["text"], ds["label"]
pos = clean([t for t, l in zip(texts, labels) if l == "suicide"], 2000, 0)
neg = clean([t for t, l in zip(texts, labels) if l == "non-suicide"], 2000, 1)
fit, ev = pos[:1000] + neg[:1000], pos[1000:2000] + neg[1000:2000]
y_fit = np.r_[np.ones(1000), np.zeros(1000)].astype(int)
P = np.r_[np.ones(1000), np.zeros(1000)].astype(bool)


def tfidf(a, b):
    m = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), LogisticRegression(C=1.0, max_iter=2000))
    return m.fit(a, y_fit).predict_proba(b)[:, 1]


saved = json.load(open(AUDIT))["gate"]
full = tfidf(fit, ev)
tok = AutoTokenizer.from_pretrained(TOK)
n_tok = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in ev])
cut = lambda t: tok.decode(tok(t, add_special_tokens=False)["input_ids"][:BUDGET])
tr = tfidf([cut(t) for t in fit], [cut(t) for t in ev])
res = {"saved_tfidf_auc": saved["tfidf_auc"], "saved_behavioral_auc": saved["behavioral_auc"],
       "saved_behavioral_ci95": saved["behavioral_ci95"],
       "tfidf_full_auc": auc(full[P], full[~P]),
       "frac_truncated": {"suicide": float((n_tok[P] > BUDGET).mean()), "nonsuicide": float((n_tok[~P] > BUDGET).mean())},
       "tfidf_trunc_auc": auc(tr[P], tr[~P]), "tfidf_trunc_ci95": boot_auc_ci(tr[P], tr[~P]),
       "trunc_minus_full_ci95": boot_auc_diff_ci(tr[P], tr[~P], full[P], full[~P])}
print(json.dumps(res, indent=1))
assert abs(res["tfidf_full_auc"] - saved["tfidf_auc"]) < 0.002, "split does not reproduce the saved TF-IDF AUC"
json.dump(res, open(OUT, "w"), indent=1)
