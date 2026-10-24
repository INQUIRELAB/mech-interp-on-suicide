"""Shared helper: the paper's Suicide-Watch-232K suicidality direction, refit on the audit backend.

Difference of means of final-token hidden states at every hidden index, using the "suicide" concept
(same loader, prompt and truncation as run_audit.py). The site is the layer whose projection best
separates a held-out 232K split; it is chosen without looking at the evaluation data of the caller.
Also returns a TF-IDF model trained on the same benchmark (the keyword baseline that "transfers
without retraining", like the direction does).

Used by umd_keyword_baseline.py and intent_minpairs_audit.py.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from concepts import get_concept
from stats import auc

# Explicit suicide vocabulary (no training): a lexicon baseline.
SUICIDE_TERMS = re.compile(
    r"\b(suicid\w*|kill(?:ing)? myself|end(?:ing)? (?:my life|it all)|end it|take my (?:own )?life|"
    r"want(?:ed)? to die|wanna die|better off dead|overdos\w*|hang(?:ing)? myself|not wake up)\b", re.I)


def lexicon_rate(texts):
    """Explicit suicide terms per 1,000 characters."""
    return np.array([1000.0 * len(SUICIDE_TERMS.findall(t)) / max(len(t), 1) for t in texts])


def fit_direction(be, n_dir=500, n_val=300, n_tfidf=2000, seed=0, batch=8):
    c = get_concept("suicide")
    be.truncate_side, be.body_budget = c.truncate, c.body_tokens
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    pos, neg = c.loader(max(n_tfidf, n_dir + n_val), seed)

    def H(texts):
        return be.collect([be.polar_prompt(t, c.prefix, c.question) for t in texts], yes, no, batch)[1]

    Hp, Hn = H(pos[:n_dir]).astype(np.float32), H(neg[:n_dir]).astype(np.float32)
    Vp, Vn = H(pos[n_dir:n_dir + n_val]), H(neg[n_dir:n_dir + n_val])
    mp, mn = Hp.mean(0), Hn.mean(0)                       # [L+1, d]
    diff = mp - mn
    D = {"concept": c, "yes": yes, "no": no,
         "dirs": diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8), "mids": 0.5 * (mp + mn)}
    D["val_auc"] = [float("nan")] + [auc(project(Vp, D, j), project(Vn, D, j)) for j in range(1, len(mp))]
    D["site"] = int(np.nanargmax(D["val_auc"]))
    tf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                       LogisticRegression(C=1.0, max_iter=2000))
    tf.fit(pos[:n_tfidf] + neg[:n_tfidf], [1] * len(pos[:n_tfidf]) + [0] * len(neg[:n_tfidf]))
    D["tfidf"] = tf
    D["meta"] = {"n_dir_per_class": n_dir, "n_val_per_class": n_val, "n_tfidf_per_class": n_tfidf,
                 "seed": seed, "site": D["site"], "val_auc_by_layer": D["val_auc"]}
    return D


def project(H, D, j):
    """Projection of hidden states H [N, L+1, d] at hidden index j, centred at the class midpoint."""
    return (np.asarray(H[:, j], dtype=np.float32) - D["mids"][j]) @ D["dirs"][j]
