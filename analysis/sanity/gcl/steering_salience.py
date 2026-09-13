#!/usr/bin/env python
"""The steering-salience illusion: a concept-agnostic audit of steering "sufficiency".

A widespread inference in mechanistic interpretability is: if ADDING a concept
direction d to the residual stream raises the model's readout for that concept,
then d is causally *sufficient* for the concept (ActAdd; refusal-direction work;
most concept-steering papers). When the readout is a polar yes/no logit, this
inference is confounded: a diff-of-means direction can carry a generic
salience / affirmative component that raises UNRELATED yes/no questions just as
much. The fix is an off-target control -- read an unrelated yes/no question under
the same steer -- and a corrected, differential sufficiency metric.

This script runs that audit across N registered concepts at once. For each concept
we fit the per-layer diff-of-means direction, steer neutral substrate posts in a
fixed mid-network window, and read EVERY concept's yes/no question plus an
off-target "food?" question. The result is a (direction x readout-question) matrix
of mean logit-difference shifts.

Metrics (at the headline coefficient c=1):
  on-target           shift[d][d's own question]
  off-target mean     mean over the other questions (incl. food)
  selectivity index   on-target - off-target mean   (corrected sufficiency)
  salience ratio      off-target mean / on-target
A direction is concept-SELECTIVE only if its own question is the argmax shift AND
the selectivity index clears a margin. We expect affective concepts (suicide,
toxicity, sentiment) to fail (salience dominates) and test whether neutral-topic
directions (sports, business) behave differently -- characterizing WHEN steering
sufficiency is trustworthy.

Output: results/gcl/steering_salience.json + steering_salience.png
Run:    python analysis/sanity/gcl/steering_salience.py
        python analysis/sanity/gcl/steering_salience.py --concepts suicide toxicity sentiment sports business
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.repro import get_seed, set_seed
set_seed()
from src.models import ModelLoader
from concept_config import get_concept

MAXLEN = 256
WIN = list(range(12, 21))          # fixed mid-network steering window
N_DIR_TRAIN = 100                  # examples/class to fit each direction
N_SUBSTRATE = 40                   # neutral posts to steer
device = "cuda"

OFFTARGET = ("food", "Read the following text and determine if it is primarily "
             "about food or cooking.", "Is this text primarily about food or cooking?")


def parse_args():
    p = argparse.ArgumentParser(description="Cross-concept steering-salience audit")
    p.add_argument("--concepts", nargs="+",
                   default=["suicide", "toxicity", "sentiment", "sports", "business"])
    p.add_argument("--coefs", nargs="+", type=float, default=[0.5, 1.0])
    p.add_argument("--n-train", type=int, default=N_DIR_TRAIN)
    p.add_argument("--n-substrate", type=int, default=N_SUBSTRATE)
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers

    # readout questions: every concept's own question + the off-target one
    questions = {}
    for c in args.concepts:
        cc = get_concept(c)
        questions[c] = (cc.prompt_prefix, cc.question)
    questions[OFFTARGET[0]] = (OFFTARGET[1], OFFTARGET[2])
    qkeys = list(questions.keys())

    def build_ids(text, prefix, question):
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = f"{prefix}\n\nText: {tok.decode(body)}\n\n{question} Answer with Yes or No.\nAnswer:"
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def hidden_lastpos(text, prefix, question):
        out = model(build_ids(text, prefix, question), output_hidden_states=True)
        return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])

    @torch.no_grad()
    def logit_diff(text, prefix, question):
        lg = model(build_ids(text, prefix, question)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    # ---- fit per-layer diff-of-means direction + projection gap per concept ----
    def fit_direction(concept_name):
        cc = get_concept(concept_name)
        pos, neg = cc.loader(args.n_train, seed=args.seed)
        pos, neg = pos[: args.n_train], neg[: args.n_train]
        Hp = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in pos])
        Hn = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in neg])
        raw = Hp.mean(0) - Hn.mean(0)
        unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
        delta = ((Hp.mean(0) - Hn.mean(0)) * unit).sum(1)
        print(f"  fitted {concept_name}")
        return unit, delta

    print("Fitting concept directions...")
    dirs = {c: fit_direction(c) for c in args.concepts}
    ref_unit, ref_delta = dirs[args.concepts[0]]
    rand_unit = rng.standard_normal(ref_unit.shape)
    rand_unit = rand_unit / np.linalg.norm(rand_unit, axis=1, keepdims=True)
    dirs["random"] = (rand_unit, ref_delta)        # comparable magnitude
    dkeys = args.concepts + ["random"]

    # ---- substrate: neutral non-suicide posts (fixed across all conditions) ----
    _, sub_neg = get_concept("suicide").loader(args.n_substrate + 60, seed=args.seed + 7)
    substrate = sub_neg[: args.n_substrate]

    def add_hooks(unit, delta, coef):
        amap = {i: torch.tensor(coef * delta[i + 1] * unit[i + 1], device=device) for i in WIN}
        def mk(i):
            add = amap[i]
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h2 = h + add.to(h.dtype)
                return (h2,) + out[1:] if isinstance(out, tuple) else h2
            return hook
        return [layers[i].register_forward_hook(mk(i)) for i in WIN]

    def mean_ld(qk):
        prefix, q = questions[qk]
        return float(np.mean([logit_diff(t, prefix, q) for t in substrate]))

    print("Baselines (unsteered)...")
    base = {qk: mean_ld(qk) for qk in qkeys}
    for qk, v in base.items():
        print(f"  {qk:10s} {v:+.2f}")

    # ---- matrix: shift[direction][coef][question] ----
    matrix = {}
    for dname in dkeys:
        unit, delta = dirs[dname]
        matrix[dname] = {}
        for c in args.coefs:
            h_ = add_hooks(unit, delta, c)
            try:
                shifts = {qk: mean_ld(qk) - base[qk] for qk in qkeys}
            finally:
                [x.remove() for x in h_]
            matrix[dname][f"{c}"] = shifts
            print(f"  steer={dname:9s} c={c}: " +
                  "  ".join(f"{qk[:4]}{shifts[qk]:+.1f}" for qk in qkeys))

    # ---- corrected sufficiency metrics at headline coef ----
    head = f"{args.coefs[-1]}"
    metrics = {}
    for c in args.concepts:
        s = matrix[c][head]
        on = s[c]
        offs = [s[q] for q in qkeys if q != c]
        off_mean = float(np.mean(offs))
        argmax_q = max(qkeys, key=lambda q: s[q])
        ordered = sorted((s[q] for q in qkeys), reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        metrics[c] = {
            "on_target": on, "off_target_mean": off_mean,
            "selectivity_index": float(on - off_mean),         # corrected sufficiency
            "salience_ratio": float(off_mean / on) if abs(on) > 1e-6 else float("nan"),
            "argmax_question": argmax_q,
            "is_selective": bool(argmax_q == c and (on - second) > 0.5),
            "food_shift": s[OFFTARGET[0]],
        }

    n_selective = sum(m["is_selective"] for m in metrics.values())
    verdict = (f"SALIENCE CONFOUND CONFIRMED: {len(args.concepts)-n_selective}/"
               f"{len(args.concepts)} concept directions are NOT concept-selective under "
               f"the off-target control (steering raises unrelated readouts as much); "
               f"polar-readout steering 'sufficiency' is confounded by salience."
               if n_selective < len(args.concepts) else
               "All directions concept-selective: steering sufficiency holds here.")

    print("\n" + "=" * 72)
    print("STEERING-SALIENCE AUDIT  (headline coef c=%s, %d concepts)" % (head, len(args.concepts)))
    print(f"{'concept':10s} {'on':>7s} {'off_mean':>9s} {'select':>7s} {'food':>7s}  argmax  selective?")
    for c in args.concepts:
        m = metrics[c]
        print(f"{c:10s} {m['on_target']:+7.2f} {m['off_target_mean']:+9.2f} "
              f"{m['selectivity_index']:+7.2f} {m['food_shift']:+7.2f}  {m['argmax_question'][:8]:8s} {m['is_selective']}")
    print(f"\n  random dir food-shift @c={head}: {matrix['random'][head][OFFTARGET[0]]:+.2f} (magnitude control)")
    print(f"  VERDICT: {verdict.split(':')[0]}")
    print("=" * 72)

    out_dir = PROJECT_ROOT / "results" / "gcl"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"concepts": args.concepts, "questions": list(qkeys), "window": WIN,
               "coefs": args.coefs, "n_train": args.n_train, "n_substrate": args.n_substrate,
               "baselines": base, "matrix": matrix, "metrics": metrics,
               "n_selective": n_selective, "verdict": verdict},
              open(out_dir / "steering_salience.json", "w"), indent=2)

    # ---- figure: shift heatmap (dir x question) + selectivity bars ----
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    M = np.array([[matrix[d][head][q] for q in qkeys] for d in dkeys])
    vmax = np.nanmax(np.abs(M))
    im = axh.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axh.set_xticks(range(len(qkeys))); axh.set_xticklabels([q[:8] for q in qkeys], rotation=40, ha="right", fontsize=9)
    axh.set_yticks(range(len(dkeys))); axh.set_yticklabels(dkeys, fontsize=9)
    axh.set_xlabel("readout question"); axh.set_ylabel("steering direction")
    axh.set_title(f"Logit-difference shift under steering (c={head})", fontsize=12, fontweight="bold")
    for i in range(len(dkeys)):
        for j in range(len(qkeys)):
            axh.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(M[i, j]) > vmax * 0.55 else "#222")
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.02)

    sel = [metrics[c]["selectivity_index"] for c in args.concepts]
    cols = ["#27AE60" if metrics[c]["is_selective"] else "#E23744" for c in args.concepts]
    axb.barh(range(len(args.concepts)), sel, color=cols, edgecolor="white", zorder=3)
    axb.axvline(0, c="#444", lw=1)
    axb.set_yticks(range(len(args.concepts))); axb.set_yticklabels(args.concepts, fontsize=9)
    axb.invert_yaxis()
    axb.set_xlabel("corrected sufficiency\n(on-target $-$ off-target mean)")
    axb.set_title("Selectivity index", fontsize=12, fontweight="bold")
    axb.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_dir / "steering_salience.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/gcl/steering_salience.json + steering_salience.png")


if __name__ == "__main__":
    main()
