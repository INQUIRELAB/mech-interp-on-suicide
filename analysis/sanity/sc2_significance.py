#!/usr/bin/env python
"""Reviewer #7: significance / equivalence tests for the load-bearing comparisons.

The scorecard rungs carried 2000-sample bootstrap CIs but no inferential test
that the feature beats its matched control. This adds, with a short GPU re-run to
recover the per-example scores the paired tests need (the cache stored only
summary AUCs + bootstrap CIs):

  PART A (GPU, 232K n=150/class, same seed-aligned examples as gate.json):
    behavioral logit-diff, layer-16 probe (5-fold OOF proba), and a MATCHED
    TF-IDF lexical baseline (OOF proba on the SAME 300 texts) -> per-case scores
    on a common set of cases, so we can run:
      * DeLong's test for two correlated ROC AUCs (gate vs lexical, probe vs lexical)
      * a paired (case-resampled) bootstrap of the AUC difference
    The lexical reference is the matched-n TF-IDF *AUC* (not the n=800 accuracy
    0.894 the older draft quoted against an AUC).

  PART B (cached): the remaining scorecard rungs as one-/two-sample tests from
    cached AUCs + n (Hanley-McNeil analytic SE): suicide-vs-depression specificity,
    cross-dataset transfer vs chance, the necessity ablation (paired-bootstrap p
    already in downstream_table.json), and the UMD severity correlation.

  PART C: a TOST equivalence test for the severity readouts (proj rho 0.543 vs
    behavioral rho 0.550) and a Benjamini-Hochberg FDR correction across the
    family of positive scorecard rungs.

Output: results/sc2_binary/significance.json   (no figure)
Run:    python analysis/sanity/sc2_significance.py
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import numpy as np
from scipy import stats
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

N_MODEL = 150
PROBE_LAYER = 16
MAXLEN = 256
device = "cuda"
SEED = get_seed()


# ----------------------------------------------------------------------------
# DeLong's test for two correlated ROC curves (fast algorithm, Sun & Xu 2014).
# ----------------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based average rank over ties
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first. Returns (aucs, cov)."""
    n_total = preds_sorted.shape[1]
    n = n_total - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, n_total])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_paired(y, score_a, score_b):
    """DeLong z and two-sided p for AUC(a) - AUC(b) on the SAME cases (y in {0,1})."""
    y = np.asarray(y); order = np.argsort(-y)            # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(score_a)[order], np.asarray(score_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else np.inf * np.sign(diff)
    p = 2 * stats.norm.sf(abs(z))
    se_a, se_b = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "se_a": float(se_a), "se_b": float(se_b), "z": float(z), "p": float(p),
            "ci95_a": [float(aucs[0] - 1.96 * se_a), float(aucs[0] + 1.96 * se_a)]}


def paired_bootstrap_diff(y, score_a, score_b, B=10000, seed=SEED + 7):
    """Case-resampled (stratified) bootstrap of AUC(a)-AUC(b); two-sided p + 95% CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(score_a); b = np.asarray(score_b)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    diffs = np.empty(B)
    for i in range(B):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        yy = y[idx]
        diffs[i] = roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: 2 x the smaller tail mass at 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)],
            "p": float(min(p, 1.0))}


# ----------------------------------------------------------------------------
# Hanley-McNeil analytic SE for a single AUC, and a two-sample (unpaired) z.
# ----------------------------------------------------------------------------
def auc_se(a, n_pos, n_neg):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def auc_se_chance(n_pos, n_neg):
    return float(np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)))


def z_two_sample(a1, se1, a2, se2):
    se = np.sqrt(se1 ** 2 + se2 ** 2)
    z = (a1 - a2) / se
    return float(z), float(2 * stats.norm.sf(abs(z))), float(se)


def bh_fdr(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        prev = min(prev, p[idx] * m / i)
        adj[idx] = prev
    return adj.tolist()


def fisher_z(r):
    return np.arctanh(r)


def tost_correlations(r1, r2, n, margin_r=0.15, alpha=0.05):
    """Two one-sided tests that |z(r1)-z(r2)| < margin, on the SAME n cases.

    Per-user readout pairs were not cached, so we cannot use the dependent
    (Steiger / Meng) covariance; we use the INDEPENDENT-samples SE, which
    OVER-states the variance and is therefore conservative for an equivalence
    test (positive coupling between the two readouts would only shrink it)."""
    z1, z2 = fisher_z(r1), fisher_z(r2)
    diff = z1 - z2
    se = np.sqrt(2.0 / (n - 3))                       # conservative (independent)
    # equivalence bound in z-space at r ~ mean(r1,r2)
    rbar = 0.5 * (r1 + r2)
    bound = margin_r / (1 - rbar ** 2)                # delta-method: dz/dr = 1/(1-r^2)
    t_lower = (diff - (-bound)) / se                  # H0: diff <= -bound
    t_upper = (diff - bound) / se                     # H0: diff >= +bound
    p_lower = stats.norm.sf(t_lower)                  # want diff > -bound
    p_upper = stats.norm.cdf(t_upper)                 # want diff < +bound
    p_tost = max(p_lower, p_upper)
    # difference test (are they distinguishable at all?)
    z_diff = diff / se
    p_diff = 2 * stats.norm.sf(abs(z_diff))
    ci = [float(np.tanh(diff - 1.96 * se)), float(np.tanh(diff + 1.96 * se))]
    return {"rho_a": r1, "rho_b": r2, "n": n, "diff_r": float(r1 - r2),
            "diff_ci95_r": ci, "margin_r": margin_r, "bound_z": float(bound),
            "p_tost": float(p_tost), "equivalent": bool(p_tost < alpha),
            "p_difference": float(p_diff), "distinguishable": bool(p_diff < alpha)}


# ============================================================================
# PART A -- GPU: recover per-case scores on 232K and run the paired tests.
# ============================================================================
def part_a():
    import torch
    from src.models import ModelLoader
    from src.analysis_utils import PROMPT_TEMPLATES

    rng = np.random.default_rng(SEED)
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[:N_MODEL], non[:N_MODEL]
    all_texts = sui + non
    y = np.concatenate([np.ones(N_MODEL), np.zeros(N_MODEL)]).astype(int)

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def feat(text):
        out = model(chat_ids(text), output_hidden_states=True)
        lg = out.logits[0, -1, :]
        ld = (lg[yes_id] - lg[no_id]).item()
        h = out.hidden_states[PROBE_LAYER][0, -1, :].float().cpu().numpy()
        return ld, h

    print(f"PART A: scoring {len(all_texts)} examples (232K, n={N_MODEL}/class)...")
    behav, hidden = [], []
    for i, t in enumerate(all_texts):
        ld, h = feat(t); behav.append(ld); hidden.append(h)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(all_texts)}")
    behav = np.array(behav); hidden = np.stack(hidden)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probe = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                              hidden, y, cv=cv, method="predict_proba")[:, 1]
    lex_pipe = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                             LogisticRegression(C=1.0, max_iter=2000))
    lexical = cross_val_predict(lex_pipe, all_texts, y, cv=cv, method="predict_proba")[:, 1]

    auc_behav = roc_auc_score(y, behav)
    auc_probe = roc_auc_score(y, probe)
    auc_lex = roc_auc_score(y, lexical)
    print(f"  behavioral AUC={auc_behav:.4f}  probe@L{PROBE_LAYER} AUC={auc_probe:.4f}  "
          f"lexical(matched n) AUC={auc_lex:.4f}")

    gate = {"delong": delong_paired(y, behav, lexical),
            "paired_bootstrap": paired_bootstrap_diff(y, behav, lexical)}
    deco = {"delong": delong_paired(y, probe, lexical),
            "paired_bootstrap": paired_bootstrap_diff(y, probe, lexical)}
    print(f"  GATE  (behav vs lexical): DeLong p={gate['delong']['p']:.2e}  "
          f"boot p={gate['paired_bootstrap']['p']:.4f}")
    print(f"  DECOD (probe vs lexical): DeLong p={deco['delong']['p']:.2e}  "
          f"boot p={deco['paired_bootstrap']['p']:.4f}")
    return {"n_per_class": N_MODEL, "probe_layer": PROBE_LAYER,
            "auc_behavioral": float(auc_behav), "auc_probe": float(auc_probe),
            "auc_lexical_matched": float(auc_lex),
            "gate_vs_lexical": gate, "decodability_vs_lexical": deco}


# ============================================================================
# PART B/C -- cached rungs, severity TOST, and the BH correction.
# ============================================================================
def part_bc(part_a_out):
    R = PROJECT_ROOT / "results"
    out = {}

    # --- specificity: suicide-vs-depression (register-matched), n=150/class ---
    sd = json.load(open(R / "gcl" / "suicide_depression" / "llama_3_1_8b_instruct.json"))
    n = sd["n_per_class"]
    lex = sd["lexical_baseline_auc"]
    probe = sd["gate"]["probe_best_auc"]; behav = sd["gate"]["behavioral_auc"]
    z_p, p_p, se = z_two_sample(probe, auc_se(probe, n, n), lex, auc_se(lex, n, n))
    z_b, p_b, _ = z_two_sample(behav, auc_se(behav, n, n), lex, auc_se(lex, n, n))
    out["specificity_suicide_vs_depression"] = {
        "n_per_class": n, "lexical_auc": lex,
        "probe_auc": probe, "probe_p_vs_lexical": p_p, "probe_z": z_p,
        "behavioral_auc": behav, "behavioral_p_vs_lexical": p_b, "behavioral_z": z_b,
        "note": "unpaired Hanley-McNeil z (per-case scores not cached); conservative"}
    print(f"\nPART B: specificity (suicide-vs-depression): probe {probe:.3f} vs lex {lex:.3f} "
          f"p={p_p:.3f}; behav {behav:.3f} p={p_b:.3f}")

    # --- cross-dataset transfer vs chance (0.5) ---
    bc = json.load(open(R / "sc2_binary" / "bootstrap_ci.json"))
    trans = {}
    for tag, n_ext in (("SDCNL", 120), ("SWMH", 120)):
        a = bc[f"transfer_auc_{tag}"][0]
        z, p, _ = z_two_sample(a, auc_se(a, n_ext, n_ext), 0.5, auc_se_chance(n_ext, n_ext))
        trans[tag] = {"auc": a, "ci95": bc[f"transfer_auc_{tag}"][1:], "p_vs_chance": p, "z": z}
        print(f"  transfer {tag}: AUC {a:.3f} vs chance p={p:.2e}")
    out["transfer_vs_chance"] = trans

    # --- necessity: paired-bootstrap p already computed (downstream_table) ---
    dt = json.load(open(R / "sc2_binary" / "downstream_table.json"))
    sui_ab = dt["conditions"]["Suicidality-direction ablation"]
    B_dt = 2000  # downstream table bootstrap resamples
    p_nec = sui_ab["p_drop_auc"] if sui_ab["p_drop_auc"] > 0 else 1.0 / B_dt
    out["necessity_ablation"] = {
        "full_auc": dt["conditions"]["Full model"]["auc"],
        "ablated_auc": sui_ab["auc"], "random_dir_auc": dt["conditions"]["Random-direction ablation"]["auc"],
        "p_drop_auc": p_nec, "reported_as": ("<%.4f" % (1.0 / B_dt)) if sui_ab["p_drop_auc"] == 0 else str(p_nec),
        "note": "paired bootstrap of F1/AUC drop vs full model (downstream_table.json)"}
    print(f"  necessity ablation: {out['necessity_ablation']['full_auc']:.3f}->{sui_ab['auc']:.3f} "
          f"p_drop={out['necessity_ablation']['reported_as']}")

    # --- severity: Spearman significance + TOST equivalence of the two readouts ---
    sev = json.load(open(R / "sc2_binary" / "umd_severity.json"))
    rho_proj = sev["spearman_proj_vs_severity"]["rho"]
    rho_behav = sev["spearman_behav_vs_severity"]["rho"]
    n_users = sev["n_users"]
    tost = tost_correlations(rho_proj, rho_behav, n_users, margin_r=0.15)
    out["severity_spearman"] = {
        "rho_projection": rho_proj, "p_projection": sev["spearman_proj_vs_severity"]["p"],
        "rho_behavioral": rho_behav, "n_users": n_users}
    out["severity_tost"] = tost
    print(f"\nPART C: severity TOST proj rho={rho_proj:.3f} vs behav rho={rho_behav:.3f} "
          f"(n={n_users}): equivalent={tost['equivalent']} (p_tost={tost['p_tost']:.3f}), "
          f"distinguishable={tost['distinguishable']} (p={tost['p_difference']:.3f})")

    # --- Benjamini-Hochberg across the positive scorecard rungs ---
    family = [
        ("decodability_232K (probe vs lexical)", part_a_out["decodability_vs_lexical"]["delong"]["p"]),
        ("gate_232K (behavioral vs lexical)", part_a_out["gate_vs_lexical"]["delong"]["p"]),
        ("specificity (probe, suicide-vs-depression)", p_p),
        ("necessity (ablation AUC drop)", p_nec),
        ("transfer (SDCNL vs chance)", trans["SDCNL"]["p_vs_chance"]),
        ("transfer (SWMH vs chance)", trans["SWMH"]["p_vs_chance"]),
        ("severity (projection rho vs 0)", sev["spearman_proj_vs_severity"]["p"]),
    ]
    names = [f[0] for f in family]; praw = [f[1] for f in family]
    q = bh_fdr(praw)
    bh = [{"rung": nm, "p": float(pp), "q_bh": float(qq), "sig_q05": bool(qq < 0.05)}
          for nm, pp, qq in zip(names, praw, q)]
    out["benjamini_hochberg"] = {"family_size": len(family), "alpha": 0.05, "rungs": bh,
                                 "n_significant": int(sum(r["sig_q05"] for r in bh))}
    print("\n  Benjamini-Hochberg across %d scorecard rungs:" % len(family))
    for r in bh:
        print(f"    {'PASS' if r['sig_q05'] else 'n.s.'}  q={r['q_bh']:.2e}  p={r['p']:.2e}  {r['rung']}")
    return out


def main():
    a = part_a()
    bc = part_bc(a)
    result = {"description": "Reviewer #7 significance / equivalence tests",
              "seed": SEED, **a, **bc}
    outpath = PROJECT_ROOT / "results" / "sc2_binary" / "significance.json"
    json.dump(result, open(outpath, "w"), indent=2)
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()
