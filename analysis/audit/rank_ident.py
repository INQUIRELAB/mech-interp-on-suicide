#!/usr/bin/env python
"""Rank identifiability on a real model (Phase 3, contribution (a)).

For one (concept, model, seed): estimate rank-k subspaces with several estimators and
sample sizes, ablate them, and classify what the ablation did.

Estimators (estimators.py):
  deflation    the paper's method. Directions 2..k are floating-point residue (see
               deflation_diagnostic.py), reported to reproduce the paper
  inlp         iterative nullspace projection (ridge)
  cluster_dom  subtype difference of means (k-means on the positive class)
  leace        rank-1 oblique LEACE eraser
  das          distributed alignment search, trained on the model's own Yes/No behavior
  random       random orthonormal subspace (control)
  top_coords   the k largest-magnitude coordinates (massive-activation control)

Scopes:
  all_layers   per-layer subspaces ablated at every layer and position (the paper's setting;
               closed-form estimators only)
  site         one subspace at the site layer, every position (all estimators, including
               DAS, compared on equal footing; mean-projection)
  site_last    as site, last token only (optional; in smoke tests it has no effect for any
               estimator, since later layers re-read the concept from the text positions)

For every ablation we record the behavioral AUC with a bootstrap CI and the mean logit
difference per class (sign-aware). For selected ranks we also record a FRESH linear probe
retrained on downstream activations, then classify the outcome:
  erased        behavior at chance and the downstream probe near chance
  inverted      behavior below chance (CI upper < 0.5): the decision axis rotated
  info_unused   behavior at chance but the downstream probe still decodes the concept
  no_effect     behavior within 0.05 of baseline
  partial       anything else
Estimated ranks: k_hat (first k at chance and not inverted) and k_sat (first k within
0.02 of the sweep minimum), per estimator, n and seed.

Usage:
  python rank_ident.py --concept suicide --model meta-llama/Llama-3.1-8B-Instruct
  python rank_ident.py --concept suicide --model Qwen/Qwen2.5-1.5B-Instruct --smoke
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import Backend
from concepts import PROJECT_ROOT, get_concept
from estimators import estimate
from interventions import erase_fn
from stats import auc, boot_auc_ci

CHANCE_HI = 0.55
SAT_TOL = 0.02


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--n-fit", type=int, default=2000, help="max per class; subsampled by --n-sweep")
    p.add_argument("--n-sweep", default="200,1000,2000")
    p.add_argument("--n-eval", type=int, default=400, help="per class, held out")
    p.add_argument("--ks", default="1,2,3,4,6,8,12,16")
    p.add_argument("--probe-ks", default="1,4,16", help="ranks that also get a downstream probe")
    p.add_argument("--estimators", default="deflation,inlp,cluster_dom,leace,das,random,top_coords")
    p.add_argument("--scopes", default="all_layers,site")
    p.add_argument("--site", type=int, default=None, help="hidden index; default ~45%% depth")
    p.add_argument("--das-steps", type=int, default=300)
    p.add_argument("--das-batch", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "rank_ident"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_fit, a.n_sweep, a.n_eval, a.ks, a.probe_ks = 60, "30,60", 30, "1,2,4", "1,4"
        a.das_steps = 20
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def outcome(base_auc, r, probe_auc):
    lo, hi = r["ci95"]
    if hi < 0.5:
        return "inverted"
    if r["auc"] >= base_auc - 0.05:
        return "no_effect"
    if lo <= 0.5 <= hi or r["auc"] <= CHANCE_HI:
        if probe_auc is None:
            return "chance"
        return "erased" if probe_auc <= 0.6 else "info_unused"
    return "partial"


def ranks(ks, rows):
    ok = [(k, r) for k, r in zip(ks, rows) if r["auc"] is not None]   # skip non-finite rows
    if not ok:
        return None, None
    aucs = [r["auc"] for _, r in ok]
    hit = [k for k, r in ok if r["auc"] <= CHANCE_HI and r["ci95"][1] >= 0.5]
    sat = [k for (k, _), v in zip(ok, aucs) if v <= min(aucs) + SAT_TOL]
    return (hit[0] if hit else None), sat[0]


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    c = get_concept(a.concept)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    ks = [int(k) for k in a.ks.split(",")]
    probe_ks = {int(k) for k in a.probe_ks.split(",")}
    n_sweep = [int(n) for n in a.n_sweep.split(",")]
    ests = [e for e in a.estimators.split(",") if e]
    scopes = [s for s in a.scopes.split(",") if s]

    pos, neg = c.loader(a.n_fit + a.n_eval, a.seed)
    n = min(len(pos), len(neg))
    n_fit = min(a.n_fit, n - a.n_eval)
    n_sweep = sorted({min(x, n_fit) for x in n_sweep})
    fp, fn_ = pos[:n_fit], neg[:n_fit]
    ep, en = pos[n_fit:n_fit + a.n_eval], neg[n_fit:n_fit + a.n_eval]

    be = Backend(a.model, dtype=a.dtype)
    be.truncate_side, be.body_budget = c.truncate, c.body_tokens
    nL = be.n_layers
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    P = lambda ts: [be.polar_prompt(t, c.prefix, c.question) for t in ts]
    site = a.site if a.site is not None else max(1, round(0.45 * nL))
    probe_j = min(nL, site + 4)
    log(f"rank_ident {c.name} on {a.model}: {nL} layers, site L{site}, probe L{probe_j}, n_fit {n_fit}")

    _, Hfp = be.collect(P(fp), yes, no, a.batch)
    _, Hfn = be.collect(P(fn_), yes, no, a.batch)
    Pep, Pen = P(ep), P(en)
    lp0, ln0 = be.logit_diff(Pep, yes, no, a.batch), be.logit_diff(Pen, yes, no, a.batch)
    base = {"auc": auc(lp0, ln0), "ci95": boot_auc_ci(lp0, ln0, seed=a.seed),
            "ld_pos": float(lp0.mean()), "ld_neg": float(ln0.mean())}
    log(f"baseline behavioral AUC {base['auc']:.3f}")
    scale = np.abs(np.vstack([Hfp[:, site], Hfn[:, site]]).astype(np.float32)).mean(0)
    top8 = np.argsort(scale)[::-1][:8]

    def probe_after(hooks):
        """Fresh probe on downstream activations under the ablation (fit/eval split of eval)."""
        half_p, half_n = len(ep) // 2, len(en) // 2
        with be.hooked(hooks):
            _, Hp = be.collect(Pep, yes, no, a.batch, verbose=False)
            _, Hn = be.collect(Pen, yes, no, a.batch, verbose=False)
        Xp, Xn = Hp[:, probe_j].astype(np.float32), Hn[:, probe_j].astype(np.float32)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        clf.fit(np.r_[Xp[:half_p], Xn[:half_n]], np.r_[np.ones(half_p), np.zeros(half_n)])
        return auc(clf.decision_function(Xp[half_p:]), clf.decision_function(Xn[half_n:]))

    def evaluate(hooks, k):
        with be.hooked(hooks):
            lp, ln = be.logit_diff(Pep, yes, no, a.batch), be.logit_diff(Pen, yes, no, a.batch)
        bad = int((~np.isfinite(lp)).sum() + (~np.isfinite(ln)).sum())
        if bad:  # record and move on instead of losing the whole run
            log(f"    k={k}: {bad} non-finite logit differences; recorded as failed")
            return {"k": k, "auc": None, "ci95": None, "nonfinite": bad, "probe_auc": None,
                    "outcome": "nonfinite"}
        r = {"k": k, "auc": auc(lp, ln), "ci95": boot_auc_ci(lp, ln, B=1000, seed=a.seed),
             "ld_pos": float(lp.mean()), "ld_neg": float(ln.mean())}
        r["probe_auc"] = probe_after(hooks) if k in probe_ks else None
        r["outcome"] = outcome(base["auc"], r, r["probe_auc"])
        return r

    def last_pos(fn):
        def g(h):
            out = h.clone()
            out[:, -1, :] = fn(h[:, -1, :])
            return out
        return g

    # ---------------------------------------------------------------- DAS on the model
    def train_das(k, Hp_site, Hn_site, fit_p, fit_n):
        d = Hp_site.shape[1]
        lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d, k, bias=False)).to(be.device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        src_all = {1: torch.tensor(Hp_site, device=be.device, dtype=torch.float32),
                   0: torch.tensor(Hn_site, device=be.device, dtype=torch.float32)}
        prompts = {1: P(fit_p), 0: P(fit_n)}
        for p in be.model.parameters():
            p.requires_grad_(False)
        g = np.random.default_rng(a.seed + k)
        state = {}

        def hook_fn(h):
            R = lin.weight.T                                   # [d, k]
            hl = h[:, -1, :].float()
            new = hl + ((state["src"] - hl) @ R) @ R.T
            out = h.clone()
            out[:, -1, :] = new.to(h.dtype)
            return out

        for step in range(a.das_steps):
            src_lab = int(g.integers(0, 2))                    # source class; base is the other
            bi = g.integers(0, len(prompts[1 - src_lab]), a.das_batch)
            si = g.integers(0, len(src_all[src_lab]), a.das_batch)
            state["src"] = src_all[src_lab][si]
            ids, mask, posn = be._encode([prompts[1 - src_lab][i] for i in bi])
            with be.hooked({site: hook_fn}):
                lg = be.model(input_ids=ids, attention_mask=mask, position_ids=posn).logits[:, -1, :]
            two = torch.stack([lg[:, no], lg[:, yes]], 1).float()
            loss = torch.nn.functional.cross_entropy(two, torch.full((len(bi),), src_lab, device=be.device))
            opt.zero_grad(); loss.backward(); opt.step()
        R = lin.weight.T.detach().float().cpu().numpy()
        return R, float(loss.item())

    results = {"concept": c.name, "model": a.model, "model_tag": tag, "seed": a.seed,
               "site": site, "probe_hidden_index": probe_j, "baseline": base, "ks": ks,
               "n_sweep": n_sweep, "n_eval": len(ep), "config": vars(a),
               "massive_coords_top8_scale_share": float(scale[top8].sum() / scale.sum()),
               "runs": []}
    mu_all = {j: (0.5 * (Hfp[:, j].astype(np.float64).mean(0) + Hfn[:, j].astype(np.float64).mean(0)))
              .astype(np.float32) for j in range(1, nL + 1)}

    for n_use in n_sweep:
        sel_p = rng.choice(n_fit, n_use, replace=False)
        sel_n = rng.choice(n_fit, n_use, replace=False)
        for scope in scopes:
            for est in ests:
                if scope == "all_layers" and est == "das":
                    continue
                kk = [1] if est == "leace" else ks
                rows, das_loss = [], {}
                nested = {}                     # nested estimators: fit once at max k, slice
                for k in kk:
                    hooks = {}
                    layers = range(1, nL + 1) if scope == "all_layers" else [site]
                    local = scope != "all_layers"
                    for j in layers:
                        Xp = Hfp[sel_p, j].astype(np.float32)
                        Xn = Hfn[sel_n, j].astype(np.float32)
                        if est == "das":
                            R, das_loss[k] = train_das(k, Xp, Xn, [fp[i] for i in sel_p],
                                                       [fn_[i] for i in sel_n])
                            U = V = R; mu = mu_all[j]
                        elif est == "leace":
                            U, V, mu = estimate("leace", Xp, Xn, 1)
                        elif est in ("deflation", "inlp", "random", "top_coords"):
                            if j not in nested:
                                nested[j] = estimate(est, Xp, Xn, max(kk), rng)[0]
                            U = V = np.ascontiguousarray(nested[j][:, :k])
                            mu = mu_all[j] if local else None   # all_layers: paper's zero-projection
                        else:
                            U, V = estimate(est, Xp, Xn, k, rng)
                            mu = mu_all[j] if local else None
                        fn = erase_fn(be.to_dev(U), be.to_dev(V), None if mu is None else be.to_dev(mu))
                        hooks[j] = last_pos(fn) if scope == "site_last" else fn
                        if j == site:
                            mass = float(((U[top8] ** 2).sum()) / (U ** 2).sum())
                    r = evaluate(hooks, k)
                    r["mass_on_top8_coords"] = mass
                    if est == "das":
                        r["das_final_loss"] = das_loss[k]
                    rows.append(r)
                    if r["auc"] is None:
                        log(f"n={n_use} {scope:10s} {est:11s} k={k:2d} non-finite outputs")
                    else:
                        log(f"n={n_use} {scope:10s} {est:11s} k={k:2d} AUC {r['auc']:.3f} "
                            f"[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] {r['outcome']}"
                            + (f" probe {r['probe_auc']:.3f}" if r["probe_auc"] is not None else ""))
                k_hat, k_sat = ranks(kk, rows)
                results["runs"].append({"n_fit": n_use, "scope": scope, "estimator": est,
                                        "rows": rows, "k_hat": k_hat, "k_sat": k_sat})
                part = Path(a.out_dir) / c.name                     # checkpoint after every estimator
                part.mkdir(parents=True, exist_ok=True)
                (part / f"{tag}__seed{a.seed}.partial.json").write_text(json.dumps(results, indent=1))
    out = Path(a.out_dir) / c.name
    out.mkdir(parents=True, exist_ok=True)
    results["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{tag}__seed{a.seed}.json"
    path.write_text(json.dumps(results, indent=1))
    log(f"saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
