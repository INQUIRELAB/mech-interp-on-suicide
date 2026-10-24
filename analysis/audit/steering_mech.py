#!/usr/bin/env python
"""Why does steering with a concept vector raise every Yes/No readout? (Phase 4, contribution (b))

For one model and a set of concepts:
  1. Fit each concept's difference-of-means vector over the steering window (around ~45% depth).
  2. Estimate two candidate generic components per layer:
       yes_dir    answer polarity with content held fixed: the SAME texts asked a question whose
                  answer is Yes ("Is this text written in English?") vs one whose answer is No
                  ("... in French?"); difference of means of the final-token states
       shared_pc  first principal direction of the stacked unit concept vectors
  3. Report cosines of every concept vector with both, and PC1's explained variance.
  4. Steer each concept's negative-class texts with four variants, each at a matched off-target KL
     budget (several kappas):
       orig           the concept vector
       minus_yes      concept vector with yes_dir projected out, renormalized
       minus_shared   concept vector with shared_pc projected out, renormalized
       interchange    no addition: set the projection on the unit concept vector to the
                      positive-class mean (in-distribution sufficiency; no KL tuning)
     Readouts: target question, negated question, three off-target questions, 4-way MC.
     Statistics: on/off/neg shifts, selectivity (on - mean off), polarity (on - neg),
     MC selectivity, flip rate.
  5. Across concepts, regress off-target shift on |cos(v, yes_dir)| and |cos(v, shared_pc)|.

Prediction if the confound is a shared answer-polarity component: orig has polarity ~ 0 and
large off-target shifts; minus_yes / minus_shared restore selectivity and polarity; off-target
shift rises with the cosine.

--followup (topic-presence test): adds
  readouts   three non-topical questions whose answer is always No (French? all capitals? over
             2,000 words?). A generic Yes push raises them; a "topic is present" push does not.
  topic_dir  per layer, leave-one-concept-out: on each OTHER concept's positive texts, its own
             question (honest Yes) minus the off-target topic questions (honest No), averaged
             over concepts
  variants   minus_topic (topic_dir projected out) and topic_only (steer with topic_dir itself);
             minus_shared_loo (PC1 of the OTHER concepts' vectors projected out, so the steered
             concept never shapes the component removed from it)
--only-variants a,b   run just these variants (e.g. a cheap minus_shared_loo top-up)

Usage: python steering_mech.py --model meta-llama/Llama-3.1-8B-Instruct
       python steering_mech.py --model Qwen/Qwen2.5-1.5B-Instruct --smoke
       python steering_mech.py --followup --out-dir ../../results/steering_topic
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from backend import Backend
from concepts import OFFTARGET_MC, OFFTARGET_QUESTIONS, PROJECT_ROOT, get_concept
from interventions import _follow, add_fn, calibrate_alpha, unit

DEFAULT_CONCEPTS = "suicide,toxicity,sentiment,sports,business,refusal,truth,anger"
NONTOPIC_QUESTIONS = [
    ("french", "Read the following text and determine its language.", "Is this text written in French?"),
    ("caps", "Read the following text and look at its formatting.",
     "Is this text written entirely in capital letters?"),
    ("long", "Read the following text and estimate its length.", "Is this text longer than 2,000 words?"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    p.add_argument("--n-fit", type=int, default=500, help="per class, for concept vectors")
    p.add_argument("--n-steer", type=int, default=200)
    p.add_argument("--n-yes", type=int, default=300, help="texts for the yes direction")
    p.add_argument("--n-ref", type=int, default=48)
    p.add_argument("--kappas", default="0.03,0.1,0.3")
    p.add_argument("--half-window", type=int, default=4)
    p.add_argument("--site", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "steering_mech"))
    p.add_argument("--followup", action="store_true", help="topic-presence test (see docstring)")
    p.add_argument("--n-topic", type=int, default=100, help="positive texts per concept for topic_dir")
    p.add_argument("--only-variants", default="", help="comma-separated subset of variants to run")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.concepts, a.n_fit, a.n_steer, a.n_yes, a.n_ref, a.kappas = "suicide,sports,refusal", 40, 16, 32, 8, "0.1"
        a.n_topic = 16
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    be = Backend(a.model, dtype=a.dtype)
    nL = be.n_layers
    site = a.site if a.site is not None else max(1, round(0.45 * nL))
    window = [j for j in range(site - a.half_window, site + a.half_window + 1) if 1 <= j <= nL]
    yes, no = be.token_id(" Yes"), be.token_id(" No")
    letter_ids = [be.token_id(" " + ch) for ch in "ABCD"]
    kappas = [float(k) for k in a.kappas.split(",")]
    log(f"steering_mech on {a.model}: site L{site}, window {window}")

    # ---------------------------------------------------------- concept vectors
    C = {}
    for name in a.concepts.split(","):
        c = get_concept(name)
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        pos, neg = c.loader(a.n_fit + a.n_steer, a.seed)
        n = min(len(pos), len(neg))
        nf = min(a.n_fit, n - a.n_steer)
        P = lambda ts, q=c.question, pre=c.prefix: [be.polar_prompt(t, pre, q) for t in ts]
        _, Hp = be.collect(P(pos[:nf]), yes, no, a.batch, verbose=False)
        _, Hn = be.collect(P(neg[:nf]), yes, no, a.batch, verbose=False)
        raw = {j: Hp[:, j].astype(np.float64).mean(0) - Hn[:, j].astype(np.float64).mean(0) for j in window}
        C[name] = {"cfg": c, "items": neg[nf:nf + a.n_steer],
                   "dir": {j: unit(raw[j]).astype(np.float32) for j in window},
                   "gap": {j: float(np.linalg.norm(raw[j])) for j in window},
                   "pos_proj": {j: float(Hp[:, j].astype(np.float32).mean(0) @ unit(raw[j])) for j in window}}
        if a.followup:  # own question (Yes) minus off-target topic questions (No), same texts
            tp = pos[:min(a.n_topic, nf)]
            Hown = Hp[:len(tp)]
            Hoff = [be.collect(P(tp, q, pr), yes, no, a.batch, verbose=False)[1]
                    for nm, pr, q in OFFTARGET_QUESTIONS if nm not in name]
            C[name]["topic_raw"] = {j: Hown[:, j].astype(np.float64).mean(0)
                                    - np.mean([H[:, j].astype(np.float64).mean(0) for H in Hoff], 0)
                                    for j in window}
        log(f"concept {name}: n_fit {nf}, steer items {len(C[name]['items'])}")

    # ---------------------------------------------------------- generic components
    pool = [t for v in C.values() for t in v["items"]]
    rng.shuffle(pool)
    pool = pool[: a.n_yes]
    be.truncate_side, be.body_budget = "right", 176
    pre = "Read the following text."
    _, Hy = be.collect([be.polar_prompt(t, pre, "Is this text written in English?") for t in pool], yes, no, a.batch, verbose=False)
    ld_n, Hno = be.collect([be.polar_prompt(t, pre, "Is this text written in French?") for t in pool], yes, no, a.batch, verbose=False)
    yes_dir = {j: unit(Hy[:, j].astype(np.float64).mean(0) - Hno[:, j].astype(np.float64).mean(0)).astype(np.float32)
               for j in window}
    shared, pc1_var = {}, {}
    for j in window:
        M = np.stack([C[c]["dir"][j] for c in C]).astype(np.float64)
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
        v = Vt[0] * np.sign(Vt[0] @ M.mean(0))
        shared[j], pc1_var[j] = v.astype(np.float32), float(S[0] ** 2 / (S ** 2).sum())
    geometry = {c: {"cos_yes": float(np.mean([C[c]["dir"][j] @ yes_dir[j] for j in window])),
                    "cos_shared": float(np.mean([C[c]["dir"][j] @ shared[j] for j in window]))} for c in C}
    shared_loo = {}
    if a.followup:  # leave-one-concept-out shared PC1
        for c in C:
            others = [o for o in C if o != c]
            shared_loo[c] = {}
            for j in window:
                M = np.stack([C[o]["dir"][j] for o in others]).astype(np.float64)
                _, _, Vt = np.linalg.svd(M, full_matrices=False)
                shared_loo[c][j] = (Vt[0] * np.sign(Vt[0] @ M.mean(0))).astype(np.float32)
    topic = {}
    if a.followup:  # leave-one-concept-out, so a concept's own direction never enters its topic_dir
        for c in C:
            topic[c] = {j: unit(np.mean([unit(C[o]["topic_raw"][j]) for o in C if o != c], 0)).astype(np.float32)
                        for j in window}
            geometry[c]["cos_topic"] = float(np.mean([C[c]["dir"][j] @ topic[c][j] for j in window]))
            geometry[c]["cos_topic_yes"] = float(np.mean([topic[c][j] @ yes_dir[j] for j in window]))
            geometry[c]["cos_topic_shared"] = float(np.mean([topic[c][j] @ shared[j] for j in window]))
        log("cos with topic_dir: " + ", ".join(f"{c} {g['cos_topic']:+.2f}" for c, g in geometry.items()))
    names = list(C)
    pair = {f"{x}|{y}": float(np.mean([C[x]["dir"][j] @ C[y]["dir"][j] for j in window]))
            for i, x in enumerate(names) for y in names[i + 1:]}
    log("cos with yes_dir: " + ", ".join(f"{c} {g['cos_yes']:+.2f}" for c, g in geometry.items()))
    log(f"shared PC1 explains {np.mean(list(pc1_var.values())):.2f} of concept-vector variance")

    # ---------------------------------------------------------- steering variants
    ref_texts = pool[-a.n_ref:]
    ref_prompts = [be._wrap_chat(f"Continue the following text.\n\n{be._truncate(t)}") for t in ref_texts]
    base_lp = torch.log_softmax(be.last_logits(ref_prompts, a.batch), -1)

    def project_out(vecs, comp):
        return {j: unit(vecs[j] - (vecs[j] @ comp[j]) * comp[j]).astype(np.float32) for j in vecs}

    def interchange_hooks(cc):
        hooks = {}
        for j in window:
            v = be.to_dev(cc["dir"][j]); target = cc["pos_proj"][j]
            def mk(v, target):
                get = _follow(v)
                def f(h):
                    v_ = get(h.device)[0]
                    return h + (target - (h @ v_)).unsqueeze(-1) * v_
                return f
            hooks[j] = mk(v, target)
        return hooks

    results = {"model": a.model, "model_tag": tag, "site": site, "window": window,
               "config": vars(a), "geometry": geometry, "pairwise_concept_cos": pair,
               "shared_pc1_explained_var": pc1_var, "concepts": {}}
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
                   **({f"nt_{nm}": P(items, q, pr) for nm, pr, q in NONTOPIC_QUESTIONS} if a.followup else {}),
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
            dneg = r["neg"] - base["neg"]
            return {"on_shift": float(don.mean()), "off_shift": float(doff.mean()),
                    "neg_shift": float(dneg.mean()), "selectivity": float((don - doff).mean()),
                    "polarity": float((don - dneg).mean()),
                    "mc_selectivity": float(((r["mc_t"] - base["mc_t"]) - (r["mc_o"] - base["mc_o"])).mean()),
                    "flip_rate_vs_zero": float((r["on"] > 0).mean()),
                    **({"nontopic_shift": float(np.mean([r[k] - base[k] for k in ntk], 0).mean()),
                        "nontopic_flip": float(np.mean([(r[k] > 0).mean() for k in ntk]))} if ntk else {})}

        variants = {"orig": cc["dir"], "minus_yes": project_out(cc["dir"], yes_dir),
                    "minus_shared": project_out(cc["dir"], shared)}
        if a.followup:
            variants["minus_topic"] = project_out(cc["dir"], topic[name])
            variants["topic_only"] = topic[name]
            variants["minus_shared_loo"] = project_out(cc["dir"], shared_loo[name])
        if a.only_variants:
            keep = a.only_variants.split(",")
            variants = {k: v for k, v in variants.items() if k in keep}
        res = {"baseline": {k: float(v.mean()) for k, v in base.items()}, "variants": {}}
        alpha0 = 1.0
        for vname, vecs in variants.items():
            res["variants"][vname] = []
            for kap in kappas:
                make = lambda al, vecs=vecs: {j: add_fn(be.to_dev(al * cc["gap"][j] * vecs[j])) for j in window}
                alpha, kl = calibrate_alpha(be, ref_prompts, make, base_lp, kap, alpha0, batch=a.batch)
                alpha0 = alpha
                row = {"kappa": kap, "alpha": alpha, "kl": kl, **stats(readouts(make(alpha)))}
                res["variants"][vname].append(row)
                log(f"{name:10s} {vname:12s} KL {kl:.3f}: on {row['on_shift']:+.2f} off {row['off_shift']:+.2f} "
                    f"neg {row['neg_shift']:+.2f} sel {row['selectivity']:+.2f} pol {row['polarity']:+.2f} "
                    f"mc {row['mc_selectivity']:+.2f}"
                    + (f" nt {row['nontopic_shift']:+.2f}" if "nontopic_shift" in row else ""))
        if not a.only_variants or "interchange" in a.only_variants.split(","):
            ih = interchange_hooks(cc)
            res["variants"]["interchange"] = [{"kappa": None, **stats(readouts(ih))}]
            r = res["variants"]["interchange"][0]
            log(f"{name:10s} {'interchange':12s}          on {r['on_shift']:+.2f} off {r['off_shift']:+.2f} "
                f"neg {r['neg_shift']:+.2f} sel {r['selectivity']:+.2f} pol {r['polarity']:+.2f}")
        results["concepts"][name] = res

    # ---------------------------------------------------------- cross-concept regression
    mid = kappas[len(kappas) // 2]
    xs_y, xs_s, ys = [], [], []
    for name, res in results["concepts"].items():
        if "orig" not in res["variants"]:
            continue
        row = next(r for r in res["variants"]["orig"] if r["kappa"] == mid)
        xs_y.append(abs(geometry[name]["cos_yes"])); xs_s.append(abs(geometry[name]["cos_shared"]))
        ys.append(row["off_shift"])
    if len(ys) >= 3:
        from scipy.stats import spearmanr
        results["regression_at_kappa"] = {
            "kappa": mid,
            "spearman_offshift_vs_cos_yes": list(map(float, spearmanr(xs_y, ys))),
            "spearman_offshift_vs_cos_shared": list(map(float, spearmanr(xs_s, ys)))}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{tag}__seed{a.seed}.json"
    path.write_text(json.dumps(results, indent=1))
    log(f"saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
