#!/usr/bin/env python
"""Cross-concept steering specificity matrix.

The single-question off-target test (run_gcl SUFFICIENT rung) suggested the
diff-of-means "concept direction" may be entangled with a generic "answer Yes"
direction: steering raised an off-target ("food?") question as much as the
on-target one, while random did nothing. This script runs the decisive test
using TWO real concept directions (suicide, toxicity):

  substrate posts (neutral non-suicide), steered at a FIXED mid-network window
  with direction D in {suicide, toxicity, random}, read out as 3 yes/no
  questions Q in {suicide?, toxicity?, food?}, over a coefficient sweep.

  Entry[D][Q] = mean shift in logit(Yes)-logit(No) vs unsteered baseline.

Interpretation:
  - SPECIFIC  : diagonal dominates -- suicide-dir raises suicide-Q most,
                toxicity-dir raises toxicity-Q most.
  - GENERIC-YES: every meaningful direction raises every question similarly
                (and more than random) -> the directions are dominated by a
                shared affirmative component; the sufficiency claim is confounded.

Output: results/gcl/specificity_matrix.json
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch, numpy as np
from src.repro import get_seed, set_seed
set_seed()
from src.models import ModelLoader
from concept_config import get_concept

MAXLEN = 256
WIN = list(range(12, 21))          # fixed mid-network window (paper's choice)
N_DIR_TRAIN = 100                  # examples/class to fit each direction
N_SUBSTRATE = 40                   # neutral posts to steer
COEFS = [0.25, 0.5, 1.0]
device = "cuda"
rng = np.random.default_rng(get_seed())

CONCEPTS = ["suicide", "toxicity"]
QUESTIONS = {
    "suicide": ("Read the following text and determine if the person shows signs of suicidal ideation.",
                "Does this text show signs of suicidal ideation?"),
    "toxicity": ("Read the following comment and determine if it is toxic (rude, disrespectful, or hateful).",
                 "Is this comment toxic?"),
    "food": ("Read the following text and determine if it is primarily about food or cooking.",
             "Is this text primarily about food or cooking?"),
}

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]
layers = model.model.layers
nL = len(layers)


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


# ---- fit per-layer diff-of-means direction + delta for each concept ----
def fit_direction(concept_name):
    cc = get_concept(concept_name)
    pos, neg = cc.loader(N_DIR_TRAIN, seed=get_seed())
    pos, neg = pos[:N_DIR_TRAIN], neg[:N_DIR_TRAIN]
    Hp = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in pos])
    Hn = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in neg])
    raw = Hp.mean(0) - Hn.mean(0)
    unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    delta = ((Hp.mean(0) - Hn.mean(0)) * unit).sum(1)
    print(f"  fitted {concept_name} direction")
    return unit, delta


print("Fitting concept directions...")
dirs = {}
for c in CONCEPTS:
    dirs[c] = fit_direction(c)
rand_unit = rng.standard_normal(dirs[CONCEPTS[0]][0].shape)
rand_unit = rand_unit / np.linalg.norm(rand_unit, axis=1, keepdims=True)
rand_delta = dirs[CONCEPTS[0]][1]   # reuse a comparable magnitude
dirs["random"] = (rand_unit, rand_delta)

# ---- substrate: neutral non-suicide posts ----
sub_pos, sub_neg = get_concept("suicide").loader(N_SUBSTRATE + 50, seed=get_seed() + 7)
substrate = sub_neg[:N_SUBSTRATE]


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


def mean_ld(question_key, hooks_on=None):
    prefix, q = QUESTIONS[question_key]
    vals = [logit_diff(t, prefix, q) for t in substrate]
    return float(np.mean(vals))


# ---- baselines (unsteered) per question ----
print("Baselines...")
base = {qk: mean_ld(qk) for qk in QUESTIONS}
for qk, v in base.items():
    print(f"  baseline {qk}: {v:+.2f}")

# ---- matrix: shift[direction][coef][question] ----
matrix = {}
for dname in ["suicide", "toxicity", "random"]:
    unit, delta = dirs[dname]
    matrix[dname] = {}
    for c in COEFS:
        h_ = add_hooks(unit, delta, c)
        try:
            shifts = {qk: mean_ld(qk) - base[qk] for qk in QUESTIONS}
        finally:
            [x.remove() for x in h_]
        matrix[dname][str(c)] = shifts
        print(f"  steer={dname:8s} c={c}: " +
              "  ".join(f"{qk}{shifts[qk]:+.2f}" for qk in QUESTIONS))

# ---- verdict: at each coef, does each concept-dir raise its OWN question most? ----
verdict = {}
for c in COEFS:
    s_sui = matrix["suicide"][str(c)]
    s_tox = matrix["toxicity"][str(c)]
    sui_specific = s_sui["suicide"] > s_sui["food"] and s_sui["suicide"] >= s_sui["toxicity"]
    tox_specific = s_tox["toxicity"] > s_tox["food"] and s_tox["toxicity"] >= s_tox["suicide"]
    verdict[str(c)] = {"suicide_dir_specific": bool(sui_specific),
                       "toxicity_dir_specific": bool(tox_specific)}

any_specific = any(v["suicide_dir_specific"] and v["toxicity_dir_specific"] for v in verdict.values())
print("\n" + "=" * 64)
print("SPECIFICITY MATRIX VERDICT")
for c in COEFS:
    v = verdict[str(c)]
    print(f"  c={c}: suicide-dir specific={v['suicide_dir_specific']}  toxicity-dir specific={v['toxicity_dir_specific']}")
print(f"  => {'SPECIFIC at some dose' if any_specific else 'GENERIC-YES: directions raise all questions; sufficiency is confounded'}")

out = {"window": WIN, "coefs": COEFS, "n_substrate": N_SUBSTRATE,
       "baselines": base, "matrix": matrix, "verdict": verdict,
       "any_dose_specific": bool(any_specific)}
(PROJECT_ROOT / "results" / "gcl").mkdir(parents=True, exist_ok=True)
json.dump(out, open(PROJECT_ROOT / "results" / "gcl" / "specificity_matrix.json", "w"), indent=2)
print("Saved results/gcl/specificity_matrix.json")
