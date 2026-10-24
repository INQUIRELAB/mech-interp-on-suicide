#!/usr/bin/env python
"""Off-target necessity: does removing one concept's direction also remove the answers to other questions?

run_audit.py tests whether projecting a concept's difference-of-means direction out of every layer removes that
concept's own class gap, against null directions that do the same damage. Those nulls carry no class information,
so the test cannot tell a concept-specific direction from a generic yes/no readout. This script adds the missing
comparison. For one model:

  1. For each concept, fit the difference of means of final-token states at every hidden index (fit split, the
     concept's own yes/no prompt; same procedure as run_audit.py). Also fit an answer-polarity direction with
     content held fixed: the same texts asked "Is this text written in English?" (Yes) vs "... in French?" (No).
  2. For each direction set, zero-project it out of the residual stream at every layer and position ("all", the
     ablation used in the paper) and, separately, only in the upper half of the network ("upper").
  3. Under each edit, measure every concept's task on held-out texts: the Yes-minus-No logit-difference gap between
     classes (share of baseline, bootstrap CI) and AUC, plus the mean KL divergence on neutral reference prompts.

A direction specific to its concept removes its own gap and leaves the others; a generic yes/no readout removes
all of them. Off-diagonal directions also serve as class-informative nulls for the suicide tasks.

Usage: python offtarget_necessity.py --model meta-llama/Llama-3.1-8B-Instruct
       python offtarget_necessity.py --model Qwen/Qwen2.5-7B-Instruct --smoke
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
from interventions import ablation_hooks, diff_of_means, mean_kl
from stats import auc

DEFAULT_CONCEPTS = "suicide,suicide_depression,sentiment,toxicity,sports,truth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--concepts", default=DEFAULT_CONCEPTS)
    p.add_argument("--n-fit", type=int, default=500, help="per class, for directions")
    p.add_argument("--n-eval", type=int, default=200, help="per class, held out")
    p.add_argument("--n-yes", type=int, default=300, help="texts for the answer-polarity direction")
    p.add_argument("--n-ref", type=int, default=48)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "offtarget_necessity"))
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.concepts, a.n_fit, a.n_eval, a.n_yes, a.n_ref, a.n_boot = "suicide,sentiment", 24, 16, 16, 8, 50
    return a


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gap_share_ci(lp, ln, base_gap, rng, B):
    """Bootstrap CI of (mean lp - mean ln) / base_gap, resampling items within each class."""
    lp, ln = np.asarray(lp), np.asarray(ln)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = (lp[rng.integers(0, len(lp), len(lp))].mean() - ln[rng.integers(0, len(ln), len(ln))].mean()) / base_gap
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    be = Backend(a.model, dtype=a.dtype)
    nL = be.n_layers
    yes, no = be.token_id(" Yes"), be.token_id(" No")
    all_j = list(range(1, nL + 1))
    upper_j = [j for j in all_j if j > nL // 2]
    log(f"offtarget_necessity on {a.model}: {nL} layers; upper half = L{upper_j[0]}-L{upper_j[-1]}")

    # ---------------------------------------------------------- per-concept data, prompts, directions
    T = {}
    for name in a.concepts.split(","):
        c = get_concept(name)
        be.truncate_side, be.body_budget = c.truncate, c.body_tokens
        pos, neg = c.loader(a.n_fit + a.n_eval, a.seed)
        n = min(len(pos), len(neg))
        nf = min(a.n_fit, n - a.n_eval)
        P = [be.polar_prompt(t, c.prefix, c.question) for t in pos[:nf]]
        N = [be.polar_prompt(t, c.prefix, c.question) for t in neg[:nf]]
        _, Hp = be.collect(P, yes, no, a.batch, verbose=False)
        _, Hn = be.collect(N, yes, no, a.batch, verbose=False)
        dirs, _ = diff_of_means(Hp, Hn)
        T[name] = {"dirs": dirs,
                   "eval_pos": [be.polar_prompt(t, c.prefix, c.question) for t in pos[nf:nf + a.n_eval]],
                   "eval_neg": [be.polar_prompt(t, c.prefix, c.question) for t in neg[nf:nf + a.n_eval]],
                   "ref_src": [be._truncate(t) for t in neg[nf:nf + a.n_eval]],
                   "n_fit": nf}
        del Hp, Hn
        log(f"concept {name}: n_fit {nf}, n_eval {len(T[name]['eval_pos'])}")

    # answer-polarity direction: same texts, a question whose answer is Yes vs one whose answer is No
    be.truncate_side, be.body_budget = "right", 176
    pool = [t for v in T.values() for t in v["ref_src"]]
    rng.shuffle(pool)
    pool = pool[: a.n_yes]
    pre = "Read the following text."
    _, Hy = be.collect([be.polar_prompt(t, pre, "Is this text written in English?") for t in pool], yes, no, a.batch, verbose=False)
    _, Hno = be.collect([be.polar_prompt(t, pre, "Is this text written in French?") for t in pool], yes, no, a.batch, verbose=False)
    answer_dirs, _ = diff_of_means(Hy, Hno)
    del Hy, Hno

    per = max(1, a.n_ref // len(T))
    ref_texts = [t for v in T.values() for t in v["ref_src"][-per:]][: a.n_ref]
    ref_prompts = [be._wrap_chat(f"Continue the following text.\n\n{t}") for t in ref_texts]
    base_lp = torch.log_softmax(be.last_logits(ref_prompts, a.batch), -1)

    def evaluate(hooks):
        out = {}
        with be.hooked(hooks):
            for name, v in T.items():
                out[name] = (be.logit_diff(v["eval_pos"], yes, no, a.batch), be.logit_diff(v["eval_neg"], yes, no, a.batch))
        return out

    base = evaluate({})
    base_stats = {k: {"auc": auc(*v), "gap": float(v[0].mean() - v[1].mean())} for k, v in base.items()}
    log("baseline: " + ", ".join(f"{k} AUC {s['auc']:.3f} gap {s['gap']:.2f}" for k, s in base_stats.items()))

    # cosines between direction sets (mean over layers, and in the upper half)
    sets = {**{k: v["dirs"] for k, v in T.items()}, "answer_polarity": answer_dirs}
    names = list(sets)
    cos = {f"{x}|{y}": {"all": float(np.mean([sets[x][j] @ sets[y][j] for j in all_j])),
                        "upper": float(np.mean([sets[x][j] @ sets[y][j] for j in upper_j]))}
           for i, x in enumerate(names) for y in names[i + 1:]}

    results = {"model": a.model, "model_tag": tag, "n_layers": nL, "upper_layers": upper_j, "config": vars(a),
               "n_fit": {k: v["n_fit"] for k, v in T.items()}, "baseline": base_stats, "cosines": cos, "ablations": {}}
    for dname, dset in sets.items():
        for scope, js in (("all", all_j), ("upper", upper_j)):
            hooks = ablation_hooks(be, {j: dset[j] for j in js}, "zero")
            kl = mean_kl(be, ref_prompts, hooks, base_lp, a.batch)
            ev = evaluate(hooks)
            row = {"kl": kl, "tasks": {}}
            for tname, (lp, ln) in ev.items():
                g0 = base_stats[tname]["gap"]
                row["tasks"][tname] = {"auc": auc(lp, ln), "gap": float(lp.mean() - ln.mean()),
                                       "gap_share": float((lp.mean() - ln.mean()) / g0),
                                       "gap_share_ci95": gap_share_ci(lp, ln, g0, rng, a.n_boot),
                                       "yes_rate_pos": float((lp > 0).mean()), "yes_rate_neg": float((ln > 0).mean())}
            results["ablations"][f"{dname}|{scope}"] = row
            log(f"ablate {dname:18s} [{scope:5s}] KL {kl:.3f}: " +
                ", ".join(f"{t} {r['gap_share']:+.2f}" for t, r in row["tasks"].items()))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = out / f"{tag}__seed{a.seed}.json"
    path.write_text(json.dumps(results, indent=1))
    log(f"saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
