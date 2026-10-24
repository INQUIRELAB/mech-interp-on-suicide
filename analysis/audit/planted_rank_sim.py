#!/usr/bin/env python
"""Planted-rank ground truth for the rank-identifiability study (Phase 3).

Synthetic "activations" with a KNOWN behavioral rank k*. The positive class is a mixture of
k* subtypes, each displaced along its own orthonormal direction Q_c; the negative class has
no displacement. The "behavior" detects any subtype, r(x) = max_c Q_c.x (a nonlinear readout,
as downstream layers are). Under orthogonal projection the behavior survives until all k*
subtype directions are removed, so the behavioral rank is exactly k*. Noise has a power-law
spectrum like real residual streams, plus an isotropic floor and readout noise.

Note why a nonlinear readout is needed: with a LINEAR readout, projecting out the exact
difference-of-means direction equalizes the class means along every remaining direction, so
any linear behavior is erased at k = 1 whatever the planted structure.

Estimators (estimators.py): oracle (true span), deflation (the paper's; degenerate for
k >= 2), inlp, cluster_dom (subtype DoM), leace (rank 1, oblique), random.
k_hat = smallest k with held-out AUC <= CHANCE_HI; k_sat = smallest k within SAT_TOL of the
sweep minimum (defined even when chance is never reached).

  H3a  rank estimates: unstable at small n, stable at large n?
  H3b  deflation cannot recover k* > 1 at any n (directions 2..k are numerical residue)
  H3c  LEACE erases all LINEAR information at k = 1 yet leaves a nonlinear behavior intact:
       the "rank" is a property of the intervention family, not of the concept alone
  H3d  (Phase 4c) gsa: eigendecomposition of the first-order interchange-effect matrix
       A = E[g_x (x_src - x)^T] (attribution patching over subspaces; g = readout gradient).
       Its spectral rank k_spec (Horn parallel analysis vs same-class pairs, which carry no
       class shift) estimates k* WITHOUT an ablation sweep or optimization.

--nuisance g adds g * sum_j tanh(N_j.x / sd_j) over the 32 highest-variance noise directions
to the readout, so the readout gradient is no longer confined to span(Q) (default 0 = the
original world, where the gradient trivially lies in span(Q)). DAS is supervised by the
same behavior.

Usage: python planted_rank_sim.py [--d 1024] [--seeds 20] [--out rank_sim.json]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from estimators import cluster_dom, deflation, erase_np, inlp, leace, random_subspace

CHANCE_HI = 0.55
SAT_TOL = 0.02
KS = [1, 2, 3, 4, 6, 8, 12, 16]
DAS_STEPS = 400
EST = ("oracle", "deflation", "inlp", "cluster_dom", "das", "leace", "random", "gsa")
NUISANCE = 0.0
N_NUIS = 32
GSA_NULL_DRAWS = 10


def make_world(d, kstar, rng, sep=3.0):
    Q = np.linalg.qr(rng.standard_normal((d, kstar)))[0]
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
    spec = 1.0 / np.arange(1, d + 1) ** 0.8
    sd = np.sqrt(spec / spec.mean())
    # nuisance readout directions: the highest-variance noise axes (style-like, class-free)
    return {"Q": Q, "sep": sep, "basis": basis, "sd": sd, "N": basis[:, :N_NUIS], "nsd": sd[:N_NUIS],
            "gam": NUISANCE}


def beh_np(w, X):
    """Noise-free behavior: max_c Q_c.x + gam * sum_j tanh(N_j.x / sd_j)."""
    r = (X @ w["Q"]).max(1)
    if w["gam"]:
        r = r + w["gam"] * np.tanh((X @ w["N"]) / w["nsd"]).sum(1)
    return r


def grad_np(w, X):
    """Exact gradient of beh_np (a.e.): Q_argmax + gam * sum_j (1 - tanh^2) N_j / sd_j."""
    G = w["Q"][:, (X @ w["Q"]).argmax(1)].T
    if w["gam"]:
        t = np.tanh((X @ w["N"]) / w["nsd"])
        G = G + w["gam"] * ((1 - t ** 2) / w["nsd"]) @ w["N"].T
    return G


def gsa_spectrum(w, Xp, Xn, rng, null_draws=None):
    """Interchange-effect matrix (symmetrized) and its parallel-analysis spectral rank.
    For base x and source x' of the OTHER class, patching subspace S moves the readout by
    ~ g_x . P_S (x' - x); signed so that a move toward the source class is positive."""
    null_draws = null_draws or GSA_NULL_DRAWS

    def A_of(B, S, sign):
        G = grad_np(w, B)
        A = (G * sign[:, None]).T @ (S - B) / len(B)
        return (A + A.T) / 2
    m = min(len(Xp), len(Xn))
    P, N = Xp[rng.permutation(len(Xp))[:m]], Xn[rng.permutation(len(Xn))[:m]]
    B, S = np.r_[N, P], np.r_[P, N]                      # base / source, both directions
    sign = np.r_[np.ones(m), -np.ones(m)]                # source positive -> readout should rise
    evals, evecs = np.linalg.eigh(A_of(B, S, sign))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    nulls = []                                           # same-class pairs: no class shift
    for _ in range(null_draws):
        Pp, Nn = P[rng.permutation(m)], N[rng.permutation(m)]
        s0 = rng.choice([-1.0, 1.0], 2 * m)
        nulls.append(np.sort(np.linalg.eigvalsh(A_of(np.r_[N, P], np.r_[Nn, Pp], s0)))[::-1])
    thr = np.max(nulls, 0)
    # Added after the 2026-10-07 grid: at nuisance 0, sym(A) has rank <= 2k*, and the count ran
    # through numerical zeros (k_spec ~ 300-400). Eigenvalues must also clear numpy's matrix_rank
    # tolerance. Results at nuisance > 0 are unchanged (rank_sim_tol_recount.py).
    tol = np.abs(evals).max() * len(evals) * np.finfo(float).eps
    k_spec = 0
    while k_spec < len(evals) and evals[k_spec] > max(thr[k_spec], tol):
        k_spec += 1
    return evals, evecs, k_spec, thr


def sample(w, n, positive, rng, iso=0.3):
    d, kstar = w["Q"].shape
    X = (rng.standard_normal((n, d)) * w["sd"]) @ w["basis"].T + iso * rng.standard_normal((n, d))
    if positive:
        c = rng.integers(0, kstar, n)
        X += w["sep"] * w["Q"][:, c].T
    return X


def readout(w, X, rng, noise=0.3):
    return beh_np(w, X) + noise * rng.standard_normal(len(X))


def das_sim(w, Xp, Xn, k, seed, steps=400, lr=0.02, batch=256):
    """DAS analogue: learn an orthonormal R [d, k] such that swapping R-coordinates between a
    negative base and a positive source makes the readout fire (and vice versa). Supervised by
    the behavior, like DAS on the real model (rank_ident.py)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    d = Xp.shape[1]
    Q = torch.tensor(w["Q"], dtype=torch.float32)
    lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d, k, bias=False))
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    P, N = torch.tensor(Xp, dtype=torch.float32), torch.tensor(Xn, dtype=torch.float32)
    Nn = torch.tensor(w["N"], dtype=torch.float32)
    nsd = torch.tensor(w["nsd"], dtype=torch.float32)
    beh = lambda X: (X @ Q).max(1).values + w["gam"] * torch.tanh((X @ Nn) / nsd).sum(1)
    thr = 0.5 * (beh(P).median() + beh(N).median()).detach()
    for _ in range(steps):
        ip = torch.randint(0, len(P), (batch,), generator=g)
        ineg = torch.randint(0, len(N), (batch,), generator=g)
        R = lin.weight.T                                   # [d, k] orthonormal columns
        to_pos = N[ineg] + (P[ip] - N[ineg]) @ R @ R.T     # negative base, positive source
        to_neg = P[ip] + (N[ineg] - P[ip]) @ R @ R.T       # positive base, negative source
        logits = torch.cat([beh(to_pos) - thr, beh(to_neg) - thr])
        target = torch.cat([torch.ones(batch), torch.zeros(batch)])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    R = lin.weight.T.detach().numpy().astype(np.float32)
    return R, R


def subspace(est, w, Xp, Xn, k, rng, seed):
    d, kstar = w["Q"].shape
    if est == "das":
        return (*das_sim(w, Xp, Xn, k, seed, steps=DAS_STEPS), None)
    if est == "gsa":
        if "_gsa" not in w:   # one spectrum per (world, sample); cached across k
            w["_gsa"] = gsa_spectrum(w, Xp, Xn, np.random.default_rng(seed + 77))
        R = w["_gsa"][1][:, :k].astype(np.float32)
        return R, R, None
    if est == "oracle":
        R = w["Q"][:, :min(k, kstar)]
        if k > kstar:
            R = np.linalg.qr(np.hstack([R, rng.standard_normal((d, k - kstar))]))[0]
        return R, R, None
    if est == "leace":
        return leace(Xp, Xn)
    if est == "random":
        return (*random_subspace(d, k, rng), None)
    if est == "cluster_dom":
        return (*cluster_dom(Xp, Xn, k, seed), None)
    return (*(deflation if est == "deflation" else inlp)(Xp, Xn, k), None)


def run(d, kstars, ns, seeds, n_test=2000):
    rows = []
    for kstar in kstars:
        for seed in range(seeds):
            rng = np.random.default_rng(1000 * kstar + seed)
            w = make_world(d, kstar, rng)
            Tp, Tn = sample(w, n_test, True, rng), sample(w, n_test, False, rng)
            y = np.r_[np.ones(n_test), np.zeros(n_test)]
            rp_noise, rn_noise = rng.standard_normal(n_test), rng.standard_normal(n_test)
            score = lambda A, B: float(roc_auc_score(
                y, np.r_[beh_np(w, A) + 0.3 * rp_noise, beh_np(w, B) + 0.3 * rn_noise]))
            base = score(Tp, Tn)
            for n in ns:
                Xp, Xn = sample(w, n, True, rng), sample(w, n, False, rng)
                w.pop("_gsa", None)
                for est in EST:
                    ks = [1] if est == "leace" else KS
                    aucs = []
                    for k in ks:
                        U, V, m = subspace(est, w, Xp, Xn, k, rng, seed)
                        aucs.append(score(erase_np(Tp, U, V, m), erase_np(Tn, U, V, m)))
                    hit = [k for k, v in zip(ks, aucs) if v <= CHANCE_HI]
                    sat = [k for k, v in zip(ks, aucs) if v <= min(aucs) + SAT_TOL]
                    row = {"kstar": kstar, "seed": seed, "n": n, "estimator": est,
                           "base_auc": base, "ks": ks, "aucs": aucs,
                           "k_hat": hit[0] if hit else None, "k_sat": sat[0],
                           "min_auc": float(min(aucs)), "inverted": bool(min(aucs) < 0.45)}
                    if est == "gsa":
                        ev, _, k_spec, thr = w["_gsa"]
                        row.update({"k_spec": int(k_spec), "evals_top": ev[:20].tolist(),
                                    "null_top": thr[:20].tolist()})
                    if est == "leace":           # LEACE guarantee: linear probe at chance
                        Ep, En = erase_np(Tp, U, V, m), erase_np(Tn, U, V, m)
                        from sklearn.linear_model import LogisticRegression
                        half = n_test // 2
                        clf = LogisticRegression(max_iter=2000).fit(
                            np.r_[Ep[:half], En[:half]], np.r_[np.ones(half), np.zeros(half)])
                        row["linear_probe_auc_after"] = float(roc_auc_score(
                            np.r_[np.ones(half), np.zeros(half)],
                            clf.decision_function(np.r_[Ep[half:], En[half:]])))
                    rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["estimator"], r["kstar"], r["n"]), []).append(r)
    table = []
    for (est, kstar, n), rs in sorted(groups.items()):
        kh = [r["k_hat"] for r in rs]
        found = [k for k in kh if k is not None]
        ks_ = [r["k_sat"] for r in rs]
        t = {"estimator": est, "kstar": kstar, "n": n, "seeds": len(rs),
             "base_auc": float(np.mean([r["base_auc"] for r in rs])),
             "k_hat_median": float(np.median(found)) if found else None,
             "frac_k_hat_correct": float(np.mean([k == kstar for k in kh])),
             "frac_never_chance": float(np.mean([k is None for k in kh])),
             "k_sat_median": float(np.median(ks_)),
             "k_sat_iqr": [float(np.percentile(ks_, 25)), float(np.percentile(ks_, 75))],
             "frac_k_sat_correct": float(np.mean([k == kstar for k in ks_])),
             "frac_inverted": float(np.mean([r["inverted"] for r in rs])),
             "auc_k1": float(np.mean([r["aucs"][0] for r in rs]))}
        if est == "gsa":
            sp = [r["k_spec"] for r in rs]
            t.update({"k_spec_median": float(np.median(sp)),
                      "k_spec_iqr": [float(np.percentile(sp, 25)), float(np.percentile(sp, 75))],
                      "frac_k_spec_correct": float(np.mean([k == kstar for k in sp]))})
        if est != "leace":
            t["auc_at_kstar"] = float(np.mean([r["aucs"][r["ks"].index(kstar)] for r in rs
                                               if kstar in r["ks"]]))
        else:
            t["linear_probe_auc_after"] = float(np.mean([r["linear_probe_auc_after"] for r in rs]))
        table.append(t)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--kstars", default="1,2,4,8")
    ap.add_argument("--ns", default="50,100,200,500,1000,5000")
    ap.add_argument("--out", default="rank_sim.json")
    ap.add_argument("--ks", default=None, help="ablation grid, e.g. 1-16 or 1,2,4 (default: KS)")
    ap.add_argument("--estimators", default=None, help="subset of EST, comma-separated")
    ap.add_argument("--das-steps", type=int, default=None, help="DAS optimizer steps (default 400)")
    ap.add_argument("--nuisance", type=float, default=0.0, help="nuisance readout weight (H3d)")
    a = ap.parse_args()
    global KS, EST, DAS_STEPS, NUISANCE
    NUISANCE = a.nuisance
    if a.ks:
        KS = (list(range(int(a.ks.split("-")[0]), int(a.ks.split("-")[1]) + 1)) if "-" in a.ks
              else [int(x) for x in a.ks.split(",")])
    if a.estimators:
        EST = [e for e in a.estimators.split(",") if e in EST]
    if a.das_steps:
        DAS_STEPS = a.das_steps
    t0 = time.time()
    rows = run(a.d, [int(x) for x in a.kstars.split(",")], [int(x) for x in a.ns.split(",")], a.seeds)
    table = summarize(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": vars(a), "chance_hi": CHANCE_HI, "sat_tol": SAT_TOL,
                                       "ks": KS, "summary": table, "rows": rows}, indent=1))
    print(f"{'est':12s} {'k*':>3s} {'n':>5s} {'base':>5s} {'AUC@1':>6s} {'AUC@k*':>7s} "
          f"{'k_hat':>6s} {'ok':>5s} {'never':>6s} {'k_sat':>6s} {'IQR':>7s} {'ok':>5s} {'inv':>5s} {'probe':>6s}")
    for t in table:
        kh = f"{t['k_hat_median']:.0f}" if t["k_hat_median"] is not None else "-"
        ak = f"{t['auc_at_kstar']:.3f}" if "auc_at_kstar" in t else "-"
        pr = f"{t['linear_probe_auc_after']:.3f}" if "linear_probe_auc_after" in t else ""
        print(f"{t['estimator']:12s} {t['kstar']:3d} {t['n']:5d} {t['base_auc']:5.3f} {t['auc_k1']:6.3f} "
              f"{ak:>7s} {kh:>6s} {t['frac_k_hat_correct']:5.2f} {t['frac_never_chance']:6.2f} "
              f"{t['k_sat_median']:6.0f} {t['k_sat_iqr'][0]:3.0f}-{t['k_sat_iqr'][1]:<3.0f} "
              f"{t['frac_k_sat_correct']:5.2f} {t['frac_inverted']:5.2f} {pr:>6s}"
              + (f"  k_spec {t['k_spec_median']:.0f} ({t['k_spec_iqr'][0]:.0f}-{t['k_spec_iqr'][1]:.0f}) "
                 f"ok {t['frac_k_spec_correct']:.2f}" if "k_spec_median" in t else ""))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
