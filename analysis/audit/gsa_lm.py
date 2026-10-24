#!/usr/bin/env python
"""Phase 4c: decodability vs sensitivity vs use, measured on real models.

Every quantity is built from the readout gradient g_x = dLD/dh (LD = logit Yes - logit No)
and the activations h at a site, so no optimization is involved.

--mode rank   (test 2) one concept, last-token residual at hidden index --site (default: the
    earliest layer where swapping the FULL last-token state moves >= --site-frac of the class gap;
    at mid-depth the last token often carries almost none of the decision).
    A = sym E[ s * g_x (h_src - h_x)^T ] over cross-class (base, source) pairs, s = +1 if the
    source is positive. g_x . P_S (h_src - h_x) is the first-order effect of interchanging
    subspace S (attribution patching over subspaces), so the top eigenvectors of A are the
    directions the readout USES to separate the classes.
    k_spec  = Horn parallel analysis: leading eigenvalues above the max over same-class-pair
              nulls (which carry no class shift).
    proj vs white: a projection swap only sees sym(A), whose top eigenvectors are generically
            (g_hat +- d_hat)/sqrt2 mixtures of a sensitive-but-dormant direction and a varying-but-
            unused one: the subspace-patching illusion (Makelov et al. 2024), reported as the
            illusion index (proj_top1 effect / full-swap effect). The whitened estimator swaps PCA-
            whitened coordinates (oblique in raw space), where every direction has unit natural
            variance, so dormant directions carry no weight (the independence GSA requires).
    Checks  on held-out pairs: measured last-token interchange of the top-k eigenspace, the DoM
            direction, random directions and the full state, vs. the first-order prediction.
    Contrast for each direction v: sensitivity E[g.v] (what steering measures) vs use (the
            measured interchange effect along v).
--mode steer  (test 3; pre-registered in plan/PREREG_gemma_generic_ratio.md)
    Rebuilds steering_mech.py's concept vectors (same seed, window, gap scaling) and computes
    the first-order shift d LD_q / dt of every readout q when t * gap_j * v_j is added at every
    window layer, at t = 0. Predicted generic ratio = mean nontopic derivative / target
    derivative: a parameter-free prediction of steering_mech's observed ratio.

Usage:
  python gsa_lm.py --mode rank --concept suicide --model unsloth/Meta-Llama-3.1-8B-Instruct --site 16
  python gsa_lm.py --mode steer --model Qwen/Qwen2.5-7B-Instruct
  python gsa_lm.py --mode rank --model Qwen/Qwen2.5-1.5B-Instruct --smoke
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from backend import Backend
from concepts import OFFTARGET_QUESTIONS, PROJECT_ROOT, get_concept
from interventions import unit
from steering_mech import DEFAULT_CONCEPTS, NONTOPIC_QUESTIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rank", "steer"], required=True)
    p.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--concept", default="suicide")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    p.add_argument("--site", type=int, default=None, help="hidden index (default ~45%% depth)")
    p.add_argument("--n-fit", type=int, default=300, help="rank: per class, for A")
    p.add_argument("--n-val", type=int, default=150, help="rank: per class, held-out checks")
    p.add_argument("--n-null", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-scan", type=int, default=32, help="rank: per class, for the site scan")
    p.add_argument("--no-scan", action="store_true")
    p.add_argument("--site-frac", type=float, default=0.8,
                   help="rank: earliest layer whose full last-token swap moves >= this share of the gap")
    p.add_argument("--pca-m", type=int, default=256, help="rank: whitened coordinates kept")
    # steer mode: same defaults as steering_mech.py so the vectors are identical
    p.add_argument("--n-steer-fit", type=int, default=500)
    p.add_argument("--n-steer", type=int, default=200)
    p.add_argument("--n-items", type=int, default=64, help="steer: items differentiated per readout")
    p.add_argument("--half-window", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "gsa"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.n_fit, a.n_val, a.n_null, a.n_boot, a.k_max, a.n_scan = 32, 16, 3, 3, 4, 8
        a.concepts, a.n_steer_fit, a.n_steer, a.n_items = "suicide,sports,refusal", 40, 16, 8
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tag_of(model):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model.split("/")[-1]).lower()


# ------------------------------------------------------------------ gradients
def last_token_grads(be, prompts, j, yes, no, batch):
    """Last-token residual at hidden j, dLD/d(that residual), and LD. float32 numpy."""
    H, G, LD = [], [], []
    for i in range(0, len(prompts), batch):
        ids, att, pos = be._encode(prompts[i:i + batch])
        st = {}

        def fn(h):
            d = torch.zeros(h.shape[0], h.shape[-1], device=h.device, dtype=torch.float32, requires_grad=True)
            st["d"], st["h"] = d, h[:, -1, :].detach().float().cpu()
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + d.to(h.dtype)
            return h
        with be.hooked({j: fn}):
            lg = be.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
        ld = lg[:, yes] - lg[:, no]
        ld.sum().backward()
        G.append(st["d"].grad.float().cpu().numpy()); H.append(st["h"].numpy()); LD.append(ld.detach().cpu().numpy())
    return np.concatenate(H), np.concatenate(G), np.concatenate(LD)


@torch.no_grad()
def interchange(be, prompts, src_h, j, R, yes, no, batch):
    """LD after interchanging the last-token state at hidden j with the source's.
    R = None: the full state. R = [d, k] orthonormal: projection swap h + ((s - h) R) R^T.
    R = (Uo, Vi): oblique swap h + ((s - h) Vi) Uo^T (a swap of whitened coordinates)."""
    out = []
    if R is not None and not isinstance(R, tuple):
        R = (R, R)
    for i in range(0, len(prompts), batch):
        ids, att, pos = be._encode(prompts[i:i + batch])
        S = torch.as_tensor(src_h[i:i + batch], dtype=torch.float32)

        def fn(h, S=S):
            h = h.clone()
            x = h[:, -1, :].float()
            s = S.to(h.device)
            if R is None:
                new = s
            else:
                Uo = torch.as_tensor(R[0], dtype=torch.float32, device=h.device)
                Vi = torch.as_tensor(R[1], dtype=torch.float32, device=h.device)
                new = x + ((s - x) @ Vi) @ Uo.T
            h[:, -1, :] = new.to(h.dtype)
            return h
        with be.hooked({j: fn}):
            lg = be.model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
        out.append((lg[:, yes] - lg[:, no]).cpu().numpy())
    return np.concatenate(out)


def sym_lowrank_eig(Gs, D):
    """Eigendecomposition of sym(Gs^T D)/n without forming the d x d matrix. Gs, D: [n, d]."""
    n = len(Gs)
    Q, Rm = np.linalg.qr(np.hstack([Gs.T, D.T]).astype(np.float64))   # d x 2n
    Gc, Dc = Rm[:, :n], Rm[:, n:]
    M = (Gc @ Dc.T + Dc @ Gc.T) / (2 * n)
    ev, V = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    return ev[order], (Q @ V[:, order]).astype(np.float32)


def sym_eig(Gs, D):
    """Eigendecomposition of sym(Gs^T D)/n for small dimension (whitened coordinates)."""
    A = Gs.astype(np.float64).T @ D.astype(np.float64) / len(Gs)
    ev, V = np.linalg.eigh((A + A.T) / 2)
    order = np.argsort(ev)[::-1]
    return ev[order], V[:, order].astype(np.float32)


def _count_above(ev, thr):
    k = 0
    while k < len(thr) and ev[k] > thr[k]:
        k += 1
    return k


def parallel_rank(eig_fn, Gs, D, null_sets, n_boot, rng):
    """Horn parallel analysis: leading eigenvalues above the max over null spectra; pair bootstrap."""
    ev, U = eig_fn(Gs, D)
    thr = np.max([eig_fn(g0, d0)[0][:50] for g0, d0 in null_sets], 0)
    k = _count_above(ev, thr)
    boot = [_count_above(eig_fn(Gs[idx], D[idx])[0], thr)
            for idx in (rng.integers(0, len(Gs), len(Gs)) for _ in range(n_boot))]
    return ev, U, thr, k, boot


def scan_sites(be, P, pos, neg, layers, yes, no, batch):
    """Share of the class gap moved by swapping the FULL last-token state at each hidden index."""
    pp, nn = P(pos), P(neg)
    Lp, Hp = be.collect(pp, yes, no, batch, verbose=False)
    Ln, Hn = be.collect(nn, yes, no, batch, verbose=False)
    n = min(len(pp), len(nn))
    gap = float(Lp[:n].mean() - Ln[:n].mean())
    sign = np.r_[np.ones(n), -np.ones(n)]
    out = {}
    for j in layers:
        src = np.r_[Hp[:n, j], Hn[:n, j]].astype(np.float32)
        ld = interchange(be, nn[:n] + pp[:n], src, j, None, yes, no, batch)
        out[j] = float(((ld - np.r_[Ln[:n], Lp[:n]]) * sign).mean() / gap)
    return out, gap


# ------------------------------------------------------------------ test 2
def mode_rank(a, be, yes, no):
    c = get_concept(a.concept)
    be.truncate_side, be.body_budget = c.truncate, c.body_tokens
    pos, neg = c.loader(a.n_fit + a.n_val + a.n_scan, a.seed)
    m = min(len(pos), len(neg), a.n_fit + a.n_val + a.n_scan)
    nf = min(a.n_fit, m - a.n_val - a.n_scan)
    P = lambda ts: [be.polar_prompt(t, c.prefix, c.question) for t in ts]
    rng = np.random.default_rng(a.seed)
    nL = be.n_layers

    # ---- site: earliest layer where the last-token state carries >= site_frac of the decision
    scan = None
    if not a.no_scan:
        layers = sorted({max(1, round(f * nL)) for f in np.arange(0.3, 1.0001, 0.1)} | {nL})
        scan, scan_gap = scan_sites(be, P, pos[m - a.n_scan:m], neg[m - a.n_scan:m], layers, yes, no, a.batch)
        log("full last-token swap, share of class gap by layer: "
            + ", ".join(f"L{j} {v:.2f}" for j, v in scan.items()))
    if a.site is not None:
        j = a.site
    elif scan:
        ok = [l for l, v in scan.items() if v >= a.site_frac]
        j = min(ok) if ok else max(scan, key=scan.get)
    else:
        j = max(1, round(0.45 * nL))
    log(f"rank mode: {a.concept} on {a.model}, site {j}, n_fit {nf}, n_val {a.n_val} per class")

    Hp, Gp, Lp = last_token_grads(be, P(pos[:nf]), j, yes, no, a.batch)
    Hn, Gn, Ln = last_token_grads(be, P(neg[:nf]), j, yes, no, a.batch)
    dom = unit(Hp.mean(0) - Hn.mean(0))
    gbar = unit(np.r_[Gp, Gn].mean(0))

    # cross-class pairs (both directions; s = +1 when the source is positive) and same-class nulls
    Hb, Gb = np.r_[Hn, Hp], np.r_[Gn, Gp]
    Hs = np.r_[Hp[rng.permutation(nf)], Hn[rng.permutation(nf)]]
    s = np.r_[np.ones(nf), -np.ones(nf)]
    Gs, D = Gb * s[:, None], Hs - Hb
    null_pairs = []
    for _ in range(a.n_null):
        s0 = rng.choice([-1.0, 1.0], 2 * nf)
        null_pairs.append((Gb * s0[:, None], np.r_[Hn[rng.permutation(nf)], Hp[rng.permutation(nf)]] - Hb))

    # ---- (i) projection estimator in raw coordinates (illusion-prone: mixes dormant and disconnected)
    ev, U, thr, k_proj, boot_proj = parallel_rank(sym_lowrank_eig, Gs, D, null_pairs, a.n_boot, rng)
    tr = float((Gs * D).sum(1).mean())
    log(f"[proj ] top {np.round(ev[:5], 3).tolist()} null {np.round(thr[:5], 3).tolist()} k {k_proj} "
        f"(boot {np.percentile(boot_proj, 5):.0f}-{np.percentile(boot_proj, 95):.0f}); trace {tr:.3f}")

    # ---- (ii) whitened estimator: swaps in PCA-whitened coordinates (every direction has unit natural variance)
    X = np.r_[Hp, Hn].astype(np.float64)
    mu = X.mean(0)
    _, S_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lam = S_ ** 2 / (len(X) - 1)
    mm = int(min(a.pca_m, (lam > lam[0] * 1e-6).sum()))
    Um, sq = Vt[:mm].T.astype(np.float32), np.sqrt(lam[:mm]).astype(np.float32)
    to_z = lambda Gx, Dx: ((Gx @ Um) * sq, (Dx @ Um) / sq)
    Gz, Dz = to_z(Gs, D)
    nz = [to_z(g0, d0) for g0, d0 in null_pairs]
    evz, W, thrz, k_white, boot_white = parallel_rank(sym_eig, Gz, Dz, nz, a.n_boot, rng)
    trz = float((Gz * Dz).sum(1).mean())
    pos_mass = np.cumsum(np.clip(evz, 0, None)) / max(np.clip(evz, 0, None).sum(), 1e-12)
    k90 = int(np.searchsorted(pos_mass, 0.9) + 1)
    log(f"[white] m {mm}, top {np.round(evz[:5], 3).tolist()} null {np.round(thrz[:5], 3).tolist()} k {k_white} "
        f"(boot {np.percentile(boot_white, 5):.0f}-{np.percentile(boot_white, 95):.0f}); k90 {k90}; "
        f"trace (PC-span swap) {trz:.3f}")
    white = lambda Wk: ((Um * sq) @ Wk, (Um / sq) @ Wk)    # (Uo, Vi) in raw coordinates

    # ---- held-out checks: measured vs first-order effects of each swap
    vp, vn = P(pos[nf:nf + a.n_val]), P(neg[nf:nf + a.n_val])
    Hvp, Gvp, Lvp = last_token_grads(be, vp, j, yes, no, a.batch)
    Hvn, Gvn, Lvn = last_token_grads(be, vn, j, yes, no, a.batch)
    nv = min(len(vp), len(vn))
    base_prompts, base_ld = vn[:nv] + vp[:nv], np.r_[Lvn[:nv], Lvp[:nv]]
    src = np.r_[Hvp[rng.permutation(len(Hvp))[:nv]], Hvn[rng.permutation(len(Hvn))[:nv]]]
    Hbv, Gbv = np.r_[Hvn[:nv], Hvp[:nv]], np.r_[Gvn[:nv], Gvp[:nv]]
    sv = np.r_[np.ones(nv), -np.ones(nv)]
    full = float(((interchange(be, base_prompts, src, j, None, yes, no, a.batch) - base_ld) * sv).mean())
    gap = float(Lvp.mean() - Lvn.mean())
    rnd = lambda k: np.linalg.qr(rng.standard_normal((Hbv.shape[1], k)))[0].astype(np.float32)
    rndz = lambda k: np.linalg.qr(rng.standard_normal((mm, k)))[0].astype(np.float32)
    ks = sorted({1, 2, 3, 4, a.k_max, max(1, k_white)} | ({k_white + 1} if k_white < mm else set()))
    cands = {**{f"proj_top{k}": U[:, :k] for k in (1, 2, a.k_max)},
             **{f"white_top{k}": white(W[:, :k]) for k in ks},
             "pc_span": Um, "dom": dom[:, None].astype(np.float32), "grad_mean": gbar[:, None].astype(np.float32),
             "random1": rnd(1), "white_random1": white(rndz(1)), f"white_random{a.k_max}": white(rndz(a.k_max))}
    checks = {}
    for name, R in cands.items():
        Uo, Vi = R if isinstance(R, tuple) else (R, R)
        meas = float(((interchange(be, base_prompts, src, j, R, yes, no, a.batch) - base_ld) * sv).mean())
        pred = float((((Gbv * sv[:, None]) @ Uo) * ((src - Hbv) @ Vi)).sum(1).mean())
        checks[name] = {"k": int(Uo.shape[1]), "measured": meas, "first_order": pred,
                        "share_of_full": meas / full if full else None}
        log(f"  {name:16s} measured {meas:+.3f} ({meas / full if full else float('nan'):+.2f} of full) "
            f"first-order {pred:+.3f}")
    w1 = unit(white(W[:, :1])[0][:, 0])
    cos = {"proj1_vs_dom": float(abs(U[:, 0] @ dom)), "proj1_vs_grad_mean": float(abs(U[:, 0] @ gbar)),
           "white1_vs_dom": float(abs(w1 @ dom)), "white1_vs_grad_mean": float(abs(w1 @ gbar)),
           "dom_vs_grad_mean": float(abs(dom @ gbar))}
    illusion = checks["proj_top1"]["measured"] / full if full else None
    log(f"full swap moves LD {full:+.3f} (gap {gap:+.3f}); illusion index (proj_top1 / full) "
        f"{illusion if illusion is None else round(illusion, 2)}; cos {({k: round(v, 2) for k, v in cos.items()})}")
    return {"concept": a.concept, "site": j, "scan_full_swap_share": scan, "n_fit": nf, "n_val": nv,
            "proj": {"spectrum_top": ev[:50].tolist(), "null_max_top": thr.tolist(), "k_spec": k_proj,
                     "k_spec_boot": boot_proj, "trace": tr},
            "white": {"m": mm, "spectrum_top": evz[:50].tolist(), "null_max_top": thrz.tolist(), "k_spec": k_white,
                      "k_spec_boot": boot_white, "k90_positive_mass": k90, "trace_pc_span": trz},
            "measured_full": full, "class_gap_val": gap, "illusion_index": illusion,
            "checks": checks, "cosines": cos}


# ------------------------------------------------------------------ test 3
def mode_steer(a, be, yes, no):
    """First-order readout shifts for steering_mech.py's concept vectors (orig variant)."""
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    nL = be.n_layers
    site = a.site if a.site is not None else max(1, round(0.45 * nL))
    window = [j for j in range(site - a.half_window, site + a.half_window + 1) if 1 <= j <= nL]
    log(f"steer mode on {a.model}: window {window}")
    out = {"window": window, "concepts": {}}
    for name in a.concepts.split(","):
        c = get_concept(name)
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        pos, neg = c.loader(a.n_steer_fit + a.n_steer, a.seed)
        n = min(len(pos), len(neg))
        nf = min(a.n_steer_fit, n - a.n_steer)
        P = lambda ts, q=c.question, pre=c.prefix: [be.polar_prompt(t, pre, q) for t in ts]
        _, Hp = be.collect(P(pos[:nf]), yes, no, a.batch, verbose=False)
        _, Hn = be.collect(P(neg[:nf]), yes, no, a.batch, verbose=False)
        raw = {j: Hp[:, j].astype(np.float64).mean(0) - Hn[:, j].astype(np.float64).mean(0) for j in window}
        vec = {j: torch.as_tensor(np.linalg.norm(raw[j]) * unit(raw[j]), dtype=torch.float32) for j in window}
        items = neg[nf:nf + a.n_steer][: a.n_items]
        offq = [(nm, pr, q) for nm, pr, q in OFFTARGET_QUESTIONS if nm not in name]
        readouts = {"on": P(items), "neg": P(items, c.neg_question),
                    **{f"off_{nm}": P(items, q, pr) for nm, pr, q in offq},
                    **{f"nt_{nm}": P(items, q, pr) for nm, pr, q in NONTOPIC_QUESTIONS}}
        der = {}
        for k, prompts in readouts.items():
            tot = 0.0
            for i in range(0, len(prompts), a.batch):
                t = torch.zeros((), device=be.device, dtype=torch.float32, requires_grad=True)
                hooks = {j: (lambda h, v=vec[j]: h + (t.to(h.device) * v.to(h.device)).to(h.dtype)) for j in window}
                ids, att, pos_ = be._encode(prompts[i:i + a.batch])
                with be.hooked(hooks):
                    lg = be.model(input_ids=ids, attention_mask=att, position_ids=pos_, logits_to_keep=1).logits[:, -1, :].float()
                (lg[:, yes] - lg[:, no]).sum().backward()
                tot += float(t.grad)
            der[k] = tot / len(prompts)
        nt = np.mean([der[k] for k in der if k.startswith("nt_")])
        off = np.mean([der[k] for k in der if k.startswith("off_")])
        res = {"derivatives": der, "pred_generic_ratio": float(nt / der["on"]) if der["on"] else None,
               "pred_off_ratio": float(off / der["on"]) if der["on"] else None,
               "pred_neg_ratio": float(der["neg"] / der["on"]) if der["on"] else None}
        out["concepts"][name] = res
        log(f"{name:10s} dLD/dt: on {der['on']:+.3f} off {off:+.3f} nt {nt:+.3f} neg {der['neg']:+.3f} -> "
            f"pred generic ratio {res['pred_generic_ratio']:+.2f}")
    ratios = [v["pred_generic_ratio"] for v in out["concepts"].values() if v["pred_generic_ratio"] is not None]
    out["pred_median_generic_ratio"] = float(np.median(ratios))
    log(f"predicted median generic ratio {out['pred_median_generic_ratio']:+.2f}")
    return out


def main():
    a = parse_args()
    t0 = time.time()
    be = Backend(a.model, dtype=a.dtype)
    for prm in be.model.parameters():
        prm.requires_grad_(False)
    yes, no = be.token_id(" Yes"), be.token_id(" No")
    res = mode_rank(a, be, yes, no) if a.mode == "rank" else mode_steer(a, be, yes, no)
    import hashlib
    res.update({"model": a.model, "mode": a.mode, "config": vars(a), "runtime_min": round((time.time() - t0) / 60, 2),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = f"{a.mode}_{a.concept + '_' if a.mode == 'rank' else ''}{tag_of(a.model)}__seed{a.seed}{'_smoke' if a.smoke else ''}.json"
    (out / name).write_text(json.dumps(res, indent=1))
    log(f"saved {out / name} ({res['runtime_min']} min)")


if __name__ == "__main__":
    main()
