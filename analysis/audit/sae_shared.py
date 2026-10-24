#!/usr/bin/env python
"""Phase 4b: is the shared steering component made of SAE features, and do SAE features steer selectively?

One model, one SAE at one residual site (hidden index h = SAE layer + 1, see backend.py):
  gemma   google/gemma-2-9b-it   + Gemma Scope 9b-it-res layer 20  (h = 21)
  llama   Llama-3.1-8B-Instruct  + Llama Scope Base L15R-8x        (h = 16; base-model SAE, check FVU)

  1. Fit each concept's difference of means (DoM) at h, the shared PC1 of the unit concept
     vectors, and yes_dir (same texts, "English?" vs "French?").
  2. SAE sanity on the instruct model's own activations: FVU and L0.
  3. Decomposition
       decoder view   cosine of each vector with every decoder direction; top features, and
                      the share of the vector's norm captured by its top-10 decoder features
       feature view   feature-space DoM dF = mean SAE code(pos) - mean SAE code(neg);
                      "shared features" = in the top-20 of at least half of the concepts;
                      per concept: share of positive dF mass on shared features, pairwise Jaccard
  4. Steering at h only, every variant matched on reference-text KL (several kappas):
       orig, minus_shared                    DoM, and DoM with shared PC1 projected out
       sae_top{k}                            sum of the top-k dF features' decoder directions
       sae_spec{k}                           same, after dropping the shared features
       minus_sharedfeat                      DoM with the span of the shared features' decoders removed
       null_sign_random (x n_null)           SteerCheck's sign-randomized DoM
     Readouts as in steering_mech.py --followup: target, negated, topical off-target,
     non-topical always-No, 4-way MC.

Usage: python sae_shared.py --model google/gemma-2-9b-it --dtype bfloat16
       python sae_shared.py --model meta-llama/Llama-3.1-8B-Instruct
       python sae_shared.py --model google/gemma-2-9b-it --dtype bfloat16 --smoke
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from backend import Backend
from concepts import OFFTARGET_MC, OFFTARGET_QUESTIONS, PROJECT_ROOT, get_concept
from interventions import NullSampler, add_fn, calibrate_alpha, unit
from steering_mech import DEFAULT_CONCEPTS, NONTOPIC_QUESTIONS

SAES = {  # model -> (hidden index, loader key)
    "google/gemma-2-9b-it": (21, "gemma_l20"),
    "meta-llama/Llama-3.1-8B-Instruct": (16, "llama_l15"),
}
SAES.update({  # public mirrors of the same weights (for hosts without an HF token)
    "unsloth/gemma-2-9b-it": SAES["google/gemma-2-9b-it"],
    "unsloth/Meta-Llama-3.1-8B-Instruct": SAES["meta-llama/Llama-3.1-8B-Instruct"],
})


class SAE:
    """JumpReLU SAE in numpy (float32). encode(x [N, d]) -> codes [N, n]; unit decoder rows D [n, d]."""

    def __init__(self, key, width="16k", l0="91"):
        self.key = key
        if key == "gemma_l20":
            self.name = f"gemma-scope-9b-it-res/layer_20/width_{width}/average_l0_{l0}"
            z = np.load(hf_hub_download("google/gemma-scope-9b-it-res",
                                        f"layer_20/width_{width}/average_l0_{l0}/params.npz"))
            self.W_enc, self.b_enc, self.thr = z["W_enc"], z["b_enc"], z["threshold"]
            self.W_dec, self.b_dec = z["W_dec"], z["b_dec"]
            self.scale = 1.0
            self.thr_uses_dec_norm = False
        elif key == "llama_l15":
            from safetensors.torch import load_file   # bfloat16 weights; numpy cannot read them
            repo, sub = "fnlp/Llama3_1-8B-Base-LXR-8x", "Llama3_1-8B-Base-L15R-8x"
            self.name = f"{repo}/{sub}"
            hp = json.load(open(hf_hub_download(repo, f"{sub}/hyperparams.json")))
            w = {k: v.float().numpy() for k, v in
                 load_file(hf_hub_download(repo, f"{sub}/checkpoints/final.safetensors")).items()}
            self.W_enc = w["encoder.weight"].T                               # [d, n]
            self.b_enc = w["encoder.bias"]
            self.W_dec = w["decoder.weight"].T                               # [n, d]
            self.b_dec = w["decoder.bias"]
            self.thr = np.full(self.W_enc.shape[1], hp["jump_relu_threshold"], np.float32)
            # dataset-wise normalization: inputs scaled so the average norm is sqrt(d_model)
            self.scale = float(np.sqrt(hp["d_model"]) / hp["dataset_average_activation_norm"]["in"])
            self.thr_uses_dec_norm = bool(hp.get("sparsity_include_decoder_norm", False))
        else:
            raise ValueError(key)
        self.W_enc, self.W_dec = self.W_enc.astype(np.float32), self.W_dec.astype(np.float32)
        self.b_enc, self.b_dec = self.b_enc.astype(np.float32), self.b_dec.astype(np.float32)
        self.thr = self.thr.astype(np.float32)
        self.dec_norm = np.linalg.norm(self.W_dec, axis=1)
        self.D = self.W_dec / np.maximum(self.dec_norm[:, None], 1e-8)

    def encode(self, x, thr_dec_norm=None):
        use = self.thr_uses_dec_norm if thr_dec_norm is None else thr_dec_norm
        pre = (np.asarray(x, np.float32) * self.scale) @ self.W_enc + self.b_enc
        gate = pre * self.dec_norm if use else pre
        return np.where(gate > self.thr, pre, 0.0).astype(np.float32)

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) / self.scale

    def fvu(self, x, **kw):
        """(FVU, L0, relative error |x - x_hat|^2 / |x|^2, mean cos(x, x_hat)).
        FVU is inflated when the sample is all one token position (small variance across rows)."""
        x = np.asarray(x, np.float32)
        f = self.encode(x, **kw)
        r = self.decode(f)
        cos = (x * r).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(r, axis=1) + 1e-8)
        return (float(((x - r) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()), float((f > 0).sum(1).mean()),
                float(((x - r) ** 2).sum() / (x ** 2).sum()), float(cos.mean()))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    p.add_argument("--width", default="16k", help="Gemma Scope width")
    p.add_argument("--l0", default="91", help="Gemma Scope average_l0 variant")
    p.add_argument("--n-fit", type=int, default=500)
    p.add_argument("--n-steer", type=int, default=200)
    p.add_argument("--n-yes", type=int, default=300)
    p.add_argument("--n-ref", type=int, default=48)
    p.add_argument("--n-null", type=int, default=5)
    p.add_argument("--kappas", default="0.03,0.1,0.3")
    p.add_argument("--ks", default="1,5,20")
    p.add_argument("--top-shared", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "sae_shared"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.concepts, a.n_fit, a.n_steer, a.n_yes, a.n_ref, a.n_null, a.kappas = \
            "suicide,sports,refusal", 40, 16, 32, 8, 1, "0.1"
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    h, key = SAES[a.model]
    sae = SAE(key, a.width, a.l0)
    log(f"SAE {sae.name}: {sae.D.shape[0]} features, input scale {sae.scale:.3f}")
    be = Backend(a.model, dtype=a.dtype)
    yes, no = be.token_id(" Yes"), be.token_id(" No")
    letter_ids = [be.token_id(" " + ch) for ch in "ABCD"]
    kappas = [float(k) for k in a.kappas.split(",")]
    ks = [int(k) for k in a.ks.split(",")]

    # ---------------------------------------------------------- concept vectors and SAE codes
    C, sanity = {}, {}
    for name in a.concepts.split(","):
        c = get_concept(name)
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        pos, neg = c.loader(a.n_fit + a.n_steer, a.seed)
        n = min(len(pos), len(neg))
        nf = min(a.n_fit, n - a.n_steer)
        P = lambda ts, q=c.question, pre=c.prefix: [be.polar_prompt(t, pre, q) for t in ts]
        _, Hp = be.collect(P(pos[:nf]), yes, no, a.batch, verbose=False)
        _, Hn = be.collect(P(neg[:nf]), yes, no, a.batch, verbose=False)
        Xp, Xn = Hp[:, h].astype(np.float32), Hn[:, h].astype(np.float32)
        raw = Xp.astype(np.float64).mean(0) - Xn.astype(np.float64).mean(0)
        Fp, Fn = sae.encode(Xp), sae.encode(Xn)
        fv, l0, rel, cs = sae.fvu(np.vstack([Xp, Xn]))
        sanity[name] = {"fvu": fv, "l0": l0, "rel_err": rel, "cos": cs}
        if key == "llama_l15":  # record the other threshold convention too
            sanity[name]["alt"] = sae.fvu(np.vstack([Xp, Xn]), thr_dec_norm=not sae.thr_uses_dec_norm)
        C[name] = {"cfg": c, "items": neg[nf:nf + a.n_steer], "dir": unit(raw).astype(np.float32),
                   "gap": float(np.linalg.norm(raw)), "dF": Fp.mean(0) - Fn.mean(0),
                   "nulls": NullSampler(Hp[:, h:h + 1], Hn[:, h:h + 1], [0])}
        log(f"concept {name}: n_fit {nf}; SAE rel err {rel:.3f}, cos {cs:.3f}, L0 {l0:.0f}, FVU {fv:.2f}")
        del Hp, Hn

    names = list(C)
    pool = [t for v in C.values() for t in v["items"]]
    rng.shuffle(pool)
    pool = pool[: a.n_yes]
    be.truncate_side, be.body_budget = "right", 176
    pre = "Read the following text."
    _, Hy = be.collect([be.polar_prompt(t, pre, "Is this text written in English?") for t in pool], yes, no, a.batch, verbose=False)
    _, Hno = be.collect([be.polar_prompt(t, pre, "Is this text written in French?") for t in pool], yes, no, a.batch, verbose=False)
    yes_dir = unit(Hy[:, h].astype(np.float64).mean(0) - Hno[:, h].astype(np.float64).mean(0)).astype(np.float32)
    del Hy, Hno
    M = np.stack([C[c]["dir"] for c in names]).astype(np.float64)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    shared = (Vt[0] * np.sign(Vt[0] @ M.mean(0))).astype(np.float32)
    pc1_var = float(S[0] ** 2 / (S ** 2).sum())
    log(f"shared PC1 explains {pc1_var:.2f}; cos(shared, yes_dir) {shared @ yes_dir:+.2f}")

    # ---------------------------------------------------------- decomposition
    def decoder_view(v, top=10):
        cs = sae.D @ v
        idx = np.argsort(-np.abs(cs))[:top]
        coef, *_ = np.linalg.lstsq(sae.D[idx].T, v, rcond=None)   # share of v spanned by top decoders
        resid = v - sae.D[idx].T @ coef
        return {"top_features": idx.tolist(), "top_cos": cs[idx].round(4).tolist(),
                "top10_span_r2": float(1 - resid @ resid / (v @ v))}

    top_sets = {c: set(np.argsort(-C[c]["dF"])[:a.top_shared].tolist()) for c in names}
    counts = {}
    for s_ in top_sets.values():
        for f in s_:
            counts[f] = counts.get(f, 0) + 1
    shared_feats = sorted(f for f, k in counts.items() if k >= int(np.ceil(len(names) / 2)))
    log(f"{len(shared_feats)} shared features (top-{a.top_shared} of >= half the concepts): {shared_feats}")
    feat_view = {}
    for c in names:
        dF = C[c]["dF"]
        pos_mass = np.clip(dF, 0, None)
        feat_view[c] = {"top_features": np.argsort(-dF)[:a.top_shared].tolist(),
                        "top_dF": np.sort(dF)[::-1][:a.top_shared].round(4).tolist(),
                        "shared_mass_frac": float(pos_mass[shared_feats].sum() / max(pos_mass.sum(), 1e-8)),
                        "cos_dir_shared": float(C[c]["dir"] @ shared), "cos_dir_yes": float(C[c]["dir"] @ yes_dir)}
    jacc = {f"{x}|{y}": len(top_sets[x] & top_sets[y]) / len(top_sets[x] | top_sets[y])
            for i, x in enumerate(names) for y in names[i + 1:]}
    if shared_feats:
        wts = np.mean([C[c]["dF"][shared_feats] for c in names], 0)
        sf_dir = unit(np.clip(wts, 0, None) @ sae.D[shared_feats]).astype(np.float32)
        Q, _ = np.linalg.qr(sae.D[shared_feats].T.astype(np.float64))   # [d, s] orthonormal span
    else:
        sf_dir, Q = np.zeros_like(shared), None
    decomposition = {
        "shared_pc1": decoder_view(shared), "yes_dir": decoder_view(yes_dir),
        "concepts_decoder": {c: decoder_view(C[c]["dir"]) for c in names},
        "concepts_feature": feat_view, "top_feature_jaccard": jacc, "shared_features": shared_feats,
        "shared_feature_counts": {int(f): k for f, k in counts.items() if k >= 2},
        "cos_sharedfeat_dir_vs_shared_pc1": float(sf_dir @ shared),
        "cos_sharedfeat_dir_vs_yes_dir": float(sf_dir @ yes_dir)}
    log(f"cos(shared-feature direction, shared PC1) {sf_dir @ shared:+.2f}; "
        f"shared PC1 top-10 decoder span R2 {decomposition['shared_pc1']['top10_span_r2']:.2f}")

    # ---------------------------------------------------------- steering at h, matched KL
    ref_prompts = [be._wrap_chat(f"Continue the following text.\n\n{be._truncate(t)}") for t in pool[-a.n_ref:]]
    base_lp = torch.log_softmax(be.last_logits(ref_prompts, a.batch), -1)

    def feat_dir(dF, k, exclude=()):
        ex = set(exclude)
        order = [f for f in np.argsort(-dF) if f not in ex][:k]
        return unit(np.clip(dF[order], 0, None) @ sae.D[order]).astype(np.float32)

    results = {"model": a.model, "model_tag": tag, "hidden_index": h, "sae": sae.name, "config": vars(a),
               "sae_sanity": sanity, "shared_pc1_explained_var": pc1_var,
               "cos_shared_yes": float(shared @ yes_dir), "decomposition": decomposition, "concepts": {}}
    for name, cc in C.items():
        c = cc["cfg"]
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        items = cc["items"]
        P = lambda ts, q=c.question, pre=c.prefix: [be.polar_prompt(t, pre, q) for t in ts]
        offq = [(nm, pr, q) for nm, pr, q in OFFTARGET_QUESTIONS if nm not in name]
        mc_r = np.random.default_rng(a.seed + 7)
        mc_opts, mc_tgt = [], []
        for _ in items:
            order = mc_r.permutation(4)
            labels = [c.mc_label] + OFFTARGET_MC
            mc_opts.append([labels[k] for k in order]); mc_tgt.append(int(np.where(order == 0)[0][0]))
        mc_tgt = np.array(mc_tgt)
        prompts = {"on": P(items), "neg": P(items, c.neg_question),
                   **{f"off_{nm}": P(items, q, pr) for nm, pr, q in offq},
                   **{f"nt_{nm}": P(items, q, pr) for nm, pr, q in NONTOPIC_QUESTIONS},
                   "mc": [be.mc_prompt(t, o) for t, o in zip(items, mc_opts)]}

        def readouts(hooks):
            out = {}
            with be.hooked(hooks):
                for k, pr in prompts.items():
                    if k == "mc":
                        lp = be.letter_logprobs(pr, letter_ids, a.batch)
                        tgt = lp[np.arange(len(lp)), mc_tgt]
                        out["mc_t"], out["mc_o"] = tgt, (lp.sum(1) - tgt) / 3
                    else:
                        out[k] = be.logit_diff(pr, yes, no, a.batch)
            return out

        base = readouts({})
        offk = [k for k in base if k.startswith("off_")]
        ntk = [k for k in base if k.startswith("nt_")]

        def stats(r):
            don = r["on"] - base["on"]; doff = np.mean([r[k] - base[k] for k in offk], 0)
            dnt = np.mean([r[k] - base[k] for k in ntk], 0); dneg = r["neg"] - base["neg"]
            return {"on_shift": float(don.mean()), "off_shift": float(doff.mean()),
                    "nontopic_shift": float(dnt.mean()), "neg_shift": float(dneg.mean()),
                    "selectivity": float((don - doff).mean()),
                    "selectivity_tail_p05": float(np.quantile(don - doff, 0.05)),
                    "polarity": float((don - dneg).mean()),
                    "mc_selectivity": float(((r["mc_t"] - base["mc_t"]) - (r["mc_o"] - base["mc_o"])).mean()),
                    "flip_rate_vs_zero": float((r["on"] > 0).mean())}

        d = cc["dir"]
        variants = {"orig": d, "minus_shared": unit(d - (d @ shared) * shared).astype(np.float32)}
        if Q is not None:
            variants["minus_sharedfeat"] = unit(d - Q @ (Q.T @ d)).astype(np.float32)
        for k in ks:
            variants[f"sae_top{k}"] = feat_dir(cc["dF"], k)
            if shared_feats:
                variants[f"sae_spec{k}"] = feat_dir(cc["dF"], k, exclude=shared_feats)
        for i in range(a.n_null):
            variants[f"null_sign_random_{i}"] = cc["nulls"].sample("sign_random", rng)[0]
        res = {"baseline": {k: float(v.mean()) for k, v in base.items()},
               "variant_cos_with_dir": {v: float(vec @ d) for v, vec in variants.items()},
               "variant_cos_with_shared": {v: float(vec @ shared) for v, vec in variants.items()},
               "variants": {}}
        alpha0 = 1.0
        for vname, vec in variants.items():
            res["variants"][vname] = []
            for kap in kappas:
                make = lambda al, vec=vec: {h: add_fn(be.to_dev(al * cc["gap"] * vec))}
                alpha, kl = calibrate_alpha(be, ref_prompts, make, base_lp, kap, alpha0, batch=a.batch)
                if not vname.startswith("null"):
                    alpha0 = alpha
                row = {"kappa": kap, "alpha": alpha, "kl": kl, **stats(readouts(make(alpha)))}
                res["variants"][vname].append(row)
                log(f"{name:10s} {vname:20s} KL {kl:.3f}: on {row['on_shift']:+.2f} off {row['off_shift']:+.2f} "
                    f"nt {row['nontopic_shift']:+.2f} neg {row['neg_shift']:+.2f} sel {row['selectivity']:+.2f} "
                    f"mc {row['mc_selectivity']:+.2f}")
        results["concepts"][name] = res

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{tag}__{key}__seed{a.seed}{'_smoke' if a.smoke else ''}.json"
    path.write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
