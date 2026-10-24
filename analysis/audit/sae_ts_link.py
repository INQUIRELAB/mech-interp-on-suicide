#!/usr/bin/env python
"""Phase 4b step 2: is the shared concept-steering component SAE-TS's generic "bias" effect?

SAE-TS (Chalnev, Siu, Conmy 2024) fits effect(v) ~ vM + b, where effect is the change in SAE
features caused by steering with v, and b is what *any* steering vector does. We measure the
same object without fitting the 50k-vector map, using antithetic pairs at a fixed norm r:
    E(v)    = mean over prompts of [SAE code at h_read (steered by +v at h_steer) - unsteered]
    even(v) = (E(v) + E(-v)) / 2      sign-independent response (b plus higher even orders)
    odd(v)  = (E(v) - E(-v)) / 2      linear response (vM)
    b_hat   = mean of even(u) over random unit u   (the generic response of any perturbation)

Questions, all at the same edit norm r (set so a random direction gives reference KL ~ kappa):
  A  cos(E(c_i), b_hat) for each concept vector c_i, and the share of ||E(c_i)||^2 along b_hat
  B  PC1 of the concept effects {E(c_i)}: cos with b_hat
  C  decoder view: cos(b_hat @ W_dec, shared PC1 of the concept DoMs at h_read)
  D  the same three numbers for the odd (linear) part, to see whether the shared part is even
     (a generic perturbation response) or odd (a shared linear direction)

Gemma: steer at hidden 10 (after Gemma layer 9), read with Gemma Scope 9b-it layer 20 (hidden 21).

Usage: python sae_ts_link.py --model google/gemma-2-9b-it --dtype bfloat16 [--smoke]
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from backend import Backend
from concepts import PROJECT_ROOT, get_concept
from interventions import add_fn, calibrate_alpha, mean_kl, unit
from sae_shared import SAE, SAES
from steering_mech import DEFAULT_CONCEPTS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    p.add_argument("--h-steer", type=int, default=10)
    p.add_argument("--n-fit", type=int, default=200, help="per class, for concept DoMs")
    p.add_argument("--n-prompts", type=int, default=3, help="readout prompts per concept")
    p.add_argument("--n-random", type=int, default=128, help="antithetic random pairs")
    p.add_argument("--n-ref", type=int, default=32)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "sae_shared"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.concepts, a.n_fit, a.n_random, a.n_ref = "suicide,sports,refusal", 24, 8, 8
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cos(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    h_read, key = SAES[a.model]
    hs = a.h_steer
    sae = SAE(key)
    be = Backend(a.model, dtype=a.dtype)
    yes, no = be.token_id(" Yes"), be.token_id(" No")
    d_model = sae.D.shape[1]

    # ---------------------------------------------------------- concept DoMs + readout prompts
    dom_s, dom_r, prompts, ref_texts = {}, {}, [], []
    for name in a.concepts.split(","):
        c = get_concept(name)
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        pos, neg = c.loader(a.n_fit + a.n_prompts + a.n_ref, a.seed)
        nf = min(a.n_fit, min(len(pos), len(neg)) - a.n_prompts - 4)
        P = lambda ts, q=c.question, pre=c.prefix: [be.polar_prompt(t, pre, q) for t in ts]
        _, Hp = be.collect(P(pos[:nf]), yes, no, a.batch, verbose=False)
        _, Hn = be.collect(P(neg[:nf]), yes, no, a.batch, verbose=False)
        dom_s[name] = unit(Hp[:, hs].astype(np.float64).mean(0) - Hn[:, hs].astype(np.float64).mean(0))
        dom_r[name] = unit(Hp[:, h_read].astype(np.float64).mean(0) - Hn[:, h_read].astype(np.float64).mean(0))
        prompts += P(neg[nf:nf + a.n_prompts])      # readout prompts: each concept's own question
        ref_texts += neg[nf + a.n_prompts:]
        log(f"concept {name}: n_fit {nf}")
    names = list(dom_s)
    rng.shuffle(ref_texts)
    ref_prompts = [be._wrap_chat(f"Continue the following text.\n\n{be._truncate(t)}") for t in ref_texts[:a.n_ref]]
    base_lp = torch.log_softmax(be.last_logits(ref_prompts, a.batch), -1)

    def codes(hooks):
        with be.hooked(hooks):
            _, H = be.collect(prompts, yes, no, a.batch, verbose=False)
        return sae.encode(H[:, h_read].astype(np.float32)).mean(0)

    base_codes = codes({})

    # ---------------------------------------------------------- edit norm r from a random direction
    u0 = unit(rng.standard_normal(d_model)).astype(np.float32)
    make = lambda al: {hs: add_fn(be.to_dev(al * u0))}
    r, kl0 = calibrate_alpha(be, ref_prompts, make, base_lp, a.kappa, alpha0=10.0, batch=a.batch)
    log(f"edit norm r = {r:.2f} (random direction, reference KL {kl0:.3f})")

    def effect(v):
        v = unit(np.asarray(v, np.float64)).astype(np.float32)
        ep = codes({hs: add_fn(be.to_dev(r * v))}) - base_codes
        en = codes({hs: add_fn(be.to_dev(-r * v))}) - base_codes
        return ep, (ep + en) / 2, (ep - en) / 2

    # ---------------------------------------------------------- generic response b_hat
    evens, odds = [], []
    for i in range(a.n_random):
        _, ev, od = effect(rng.standard_normal(d_model))
        evens.append(ev); odds.append(od)
        if (i + 1) % 16 == 0:
            log(f"random pairs {i + 1}/{a.n_random}")
    evens, odds = np.array(evens), np.array(odds)
    b_hat = evens.mean(0)
    half = len(evens) // 2
    b_split = cos(evens[:half].mean(0), evens[half:].mean(0))   # reliability of b_hat

    # ---------------------------------------------------------- concept effects
    E, Ev, Od, kl_c = {}, {}, {}, {}
    for name in names:
        E[name], Ev[name], Od[name] = effect(dom_s[name])
        kl_c[name] = mean_kl(be, ref_prompts, {hs: add_fn(be.to_dev(r * dom_s[name].astype(np.float32)))}, base_lp, a.batch)
    def share_along(x, y):
        y_ = y / (np.linalg.norm(y) + 1e-12)
        return float((x @ y_) ** 2 / (x @ x + 1e-12))
    def pc1(vs):
        M = np.stack(vs)   # uncentered on purpose: PC1 then captures the part all effects share
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        v = Vt[0] * np.sign(Vt[0] @ M.mean(0))
        return v, float(S[0] ** 2 / (S ** 2).sum())

    pcE, varE = pc1([E[n] for n in names])
    pcEv, varEv = pc1([Ev[n] for n in names])
    pcOd, varOd = pc1([Od[n] for n in names])
    Mr = np.stack([dom_r[n] for n in names])
    _, _, Vt = np.linalg.svd(Mr, full_matrices=False)
    shared_r = Vt[0] * np.sign(Vt[0] @ Mr.mean(0))
    b_resid = b_hat @ sae.D          # residual-stream direction of the generic response at h_read
    random_effect_norm = float(np.mean(np.linalg.norm(evens + odds, axis=1)))

    res = {
        "model": a.model, "sae": sae.name, "h_steer": hs, "h_read": h_read, "config": vars(a),
        "edit_norm": r, "random_dir_kl": kl0, "b_hat_split_half_cos": b_split,
        "b_hat_norm": float(np.linalg.norm(b_hat)), "random_effect_norm_mean": random_effect_norm,
        "b_hat_top_features": np.argsort(-np.abs(b_hat))[:20].tolist(),
        "b_hat_top_values": b_hat[np.argsort(-np.abs(b_hat))[:20]].round(4).tolist(),
        "A_concepts": {n: {"kl_at_r": kl_c[n], "effect_norm": float(np.linalg.norm(E[n])),
                           "cos_effect_bhat": cos(E[n], b_hat), "share_effect_along_bhat": share_along(E[n], b_hat),
                           "even_share_of_effect": float(Ev[n] @ Ev[n] / (E[n] @ E[n] + 1e-12)),
                           "cos_even_bhat": cos(Ev[n], b_hat), "cos_odd_bhat": cos(Od[n], b_hat)}
                       for n in names},
        "B_pc1_concept_effects": {"var": varE, "cos_with_bhat": cos(pcE, b_hat),
                                  "even_pc1_var": varEv, "even_pc1_cos_bhat": cos(pcEv, b_hat),
                                  "odd_pc1_var": varOd, "odd_pc1_cos_bhat": cos(pcOd, b_hat)},
        "C_decoder": {"cos_bhat_decoded_vs_shared_pc1_at_read": cos(b_resid, shared_r)},
        "concept_effect_pairwise_cos": {f"{x}|{y}": cos(E[x], E[y]) for i, x in enumerate(names) for y in names[i + 1:]},
    }
    log(f"b_hat split-half cos {b_split:.2f}; |b_hat| {np.linalg.norm(b_hat):.2f} vs mean |E(random)| {random_effect_norm:.2f}")
    for n in names:
        q = res["A_concepts"][n]
        log(f"{n:10s} KL {q['kl_at_r']:.3f} |E| {q['effect_norm']:.2f} cos(E,b) {q['cos_effect_bhat']:+.2f} "
            f"even share {q['even_share_of_effect']:.2f} cos(even,b) {q['cos_even_bhat']:+.2f} cos(odd,b) {q['cos_odd_bhat']:+.2f}")
    log(f"PC1 of concept effects: var {varE:.2f}, cos with b_hat {cos(pcE, b_hat):+.2f} "
        f"(even PC1 {cos(pcEv, b_hat):+.2f}, odd PC1 {cos(pcOd, b_hat):+.2f}); "
        f"decoded b_hat vs shared PC1 at read {res['C_decoder']['cos_bhat_decoded_vs_shared_pc1_at_read']:+.2f}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{tag}__sae_ts_link__seed{a.seed}{'_smoke' if a.smoke else ''}.json"
    path.write_text(json.dumps(res, indent=1))
    log(f"saved {path} ({res['runtime_min']} min)")


if __name__ == "__main__":
    main()
