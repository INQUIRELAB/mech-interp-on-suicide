"""Statistics for the audit: bootstrap CIs, tests against null families, and the
intersection-union (IUT) gate.

A rung passes only if the target's statistic beats EVERY null family on EVERY statistic
the rung declares (mean and tail), after Holm correction across all of those tests. This
is SteerCheck's gate, applied to each rung of the ladder rather than to steering alone.
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def auc(pos_scores, neg_scores):
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(roc_auc_score(y, np.r_[pos_scores, neg_scores]))


def boot_auc_ci(pos_scores, neg_scores, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    sp, sn = np.asarray(pos_scores), np.asarray(neg_scores)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = auc(sp[rng.integers(0, len(sp), len(sp))], sn[rng.integers(0, len(sn), len(sn))])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_auc_diff_ci(a_pos, a_neg, b_pos, b_neg, B=2000, seed=0):
    """Paired bootstrap CI of AUC(a) - AUC(b) when a and b score the same examples."""
    rng = np.random.default_rng(seed)
    a_pos, a_neg, b_pos, b_neg = map(np.asarray, (a_pos, a_neg, b_pos, b_neg))
    vals = np.empty(B)
    for i in range(B):
        ip = rng.integers(0, len(a_pos), len(a_pos))
        ineg = rng.integers(0, len(a_neg), len(a_neg))
        vals[i] = auc(a_pos[ip], a_neg[ineg]) - auc(b_pos[ip], b_neg[ineg])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def test_vs_null(observed, null_values, larger_is_effect=True):
    """z-test of the target statistic against one null family's draws.

    With few draws (10-20) the empirical p-value is floored at 1/(M+1), so the gate uses
    the parametric z; both are reported.
    """
    null = np.asarray(null_values, dtype=float)
    mu, sd = float(null.mean()), float(null.std(ddof=1)) if len(null) > 1 else 0.0
    sd = max(sd, 1e-9)
    z = (observed - mu) / sd if larger_is_effect else (mu - observed) / sd
    exceed = (null >= observed) if larger_is_effect else (null <= observed)
    return {"observed": float(observed), "null_mean": mu, "null_sd": sd, "z": float(z),
            "p": float(norm.sf(z)), "p_empirical": float((1 + exceed.sum()) / (len(null) + 1)),
            "n_draws": int(len(null))}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def iut_gate(tests, alpha=0.01):
    """tests: {(statistic, family): test_vs_null dict}. Adds p_holm; returns verdict dict."""
    keys = list(tests)
    adj = holm([tests[k]["p"] for k in keys])
    for k, a in zip(keys, adj):
        tests[k]["p_holm"] = float(a)
    passed = bool(all(a < alpha for a in adj)) if keys else False
    weakest = keys[int(np.argmax(adj))] if keys else None
    return {"pass": passed, "alpha": alpha, "n_tests": len(keys),
            "max_p_holm": float(adj.max()) if keys else None,
            "weakest": list(weakest) if weakest else None,
            "tests": {f"{s}|{f}": v for (s, f), v in tests.items()}}
