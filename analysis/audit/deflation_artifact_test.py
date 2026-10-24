#!/usr/bin/env python
"""Does the paper's rank-k degradation come from ablating massive-activation coordinates?

At every layer (as in sc2_rank_sweep.py), ablate on held-out data:
  paper      the paper's deflation subspace (direction 1 = DoM; 2..k = float32 residue)
  dom1       the DoM direction alone (k = 1)
  dom1+top   DoM direction + the (k-1) largest-magnitude coordinates (orthonormalized)
  dom1+rand  DoM direction + (k-1) random isotropic directions (the paper's control, plus dir 1)
  top        the k largest-magnitude coordinates alone (no concept information)
If paper ~= dom1+top and both are far below dom1 ~= dom1+rand, the rank-k effect is an
artifact of hitting massive activations rather than evidence of a concept subspace.

Usage: python deflation_artifact_test.py --model Qwen/Qwen2.5-1.5B-Instruct --n-fit 150 --n-eval 100
"""
import argparse
import json

import numpy as np

from backend import Backend
from concepts import get_concept
from deflation_diagnostic import paper_deflate
from interventions import erase_fn
from stats import auc, boot_auc_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concept", default="suicide")
    ap.add_argument("--n-fit", type=int, default=150)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="deflation_artifact.json")
    a = ap.parse_args()
    c = get_concept(a.concept)
    pos, neg = c.loader(a.n_fit + a.n_eval, 0)
    fp, fn_, ep, en = pos[:a.n_fit], neg[:a.n_fit], pos[a.n_fit:], neg[a.n_fit:]
    be = Backend(a.model, dtype=a.dtype)
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    P = lambda ts: [be.polar_prompt(t, c.prefix, c.question) for t in ts]
    _, Hp = be.collect(P(fp), yes, no, verbose=False)
    _, Hn = be.collect(P(fn_), yes, no, verbose=False)
    Pep, Pen = P(ep), P(en)
    nL, d = be.n_layers, Hp.shape[-1]
    rng = np.random.default_rng(0)
    ks = [int(k) for k in a.ks.split(",")]
    kmax = max(ks)

    per_layer = {}
    for j in range(1, nL + 1):
        A, B = Hp[:, j].astype(np.float32), Hn[:, j].astype(np.float32)
        Qp, _ = paper_deflate(A, B, kmax)
        top = np.argsort(np.abs(np.vstack([A, B])).mean(0))[::-1][:kmax]
        per_layer[j] = {"paper": Qp, "dom1": Qp[:, :1], "top": top,
                        "rand": rng.standard_normal((d, kmax))}

    def subspace(kind, j, k):
        L = per_layer[j]
        if kind == "paper":
            return L["paper"][:, :k]
        if kind == "dom1":
            return L["dom1"]
        if kind == "top":
            E = np.zeros((d, k), np.float32); E[L["top"][:k], np.arange(k)] = 1.0
            return E
        extra = (np.eye(d, dtype=np.float32)[:, L["top"][:k - 1]] if kind == "dom1+top"
                 else L["rand"][:, :k - 1])
        return np.linalg.qr(np.hstack([L["dom1"], extra]))[0].astype(np.float32)

    def score(kind, k):
        hooks = {}
        for j in range(1, nL + 1):
            R = be.to_dev(subspace(kind, j, k))
            hooks[j] = erase_fn(R, R)
        with be.hooked(hooks):
            lp, ln = be.logit_diff(Pep, yes, no), be.logit_diff(Pen, yes, no)
        return {"auc": auc(lp, ln), "ci95": boot_auc_ci(lp, ln), "ld_pos": float(lp.mean()),
                "ld_neg": float(ln.mean()), "ld_pos_sd": float(lp.std()), "ld_neg_sd": float(ln.std())}

    lp, ln = be.logit_diff(Pep, yes, no), be.logit_diff(Pen, yes, no)
    res = {"model": a.model, "concept": a.concept, "n_fit": a.n_fit, "n_eval": a.n_eval,
           "baseline": {"auc": auc(lp, ln), "ci95": boot_auc_ci(lp, ln),
                        "ld_pos": float(lp.mean()), "ld_neg": float(ln.mean())}, "rows": []}
    print(f"baseline AUC {res['baseline']['auc']:.3f}")
    print(f"{'k':>3s} {'paper':>8s} {'dom1':>8s} {'dom1+top':>9s} {'dom1+rand':>10s} {'top':>8s}")
    for k in ks:
        row = {"k": k}
        for kind in ("paper", "dom1", "dom1+top", "dom1+rand", "top"):
            if kind in ("dom1+top", "dom1+rand") and k == 1:
                row[kind] = row["dom1"]; continue
            row[kind] = score(kind, k)
        res["rows"].append(row)
        print(f"{k:3d} " + " ".join(f"{row[x]['auc']:8.3f}" for x in
                                     ("paper", "dom1", "dom1+top", "dom1+rand", "top")))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
