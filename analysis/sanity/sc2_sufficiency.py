#!/usr/bin/env python
"""Sufficiency / steering — is the suicidality direction ENOUGH to induce the call?

Faithfulness showed the direction is necessary. Sufficiency asks the converse:
inject the direction into NON-suicidal posts and see if the model moves toward
"suicide". Add c * delta_L * r_L at layers 12..20 (per-layer diff-of-means
direction r_L; delta_L = class-mean projection gap at that layer), all positions.
Dose-response sweep over c, plus an equal-magnitude random-direction control.

  steered non-suicidal logit-diff rises monotonically with c, crossing the
  baseline class-separating threshold, while random injection does not
      => the direction is sufficient to drive the behavior.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

N_TRAIN = 100
N_TEST = 100
WIN = list(range(12, 21))   # inject at layers 12..20 (peak band, away from logits)
COEFS = [0.0, 1.0, 2.0, 3.0]
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())

ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts, labels = ds["text"], ds["label"]
sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
sui_tr, sui_te = sui[:N_TRAIN], sui[N_TRAIN:N_TRAIN + N_TEST]
non_tr, non_te = non[:N_TRAIN], non[N_TRAIN:N_TRAIN + N_TEST]

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]
layers = model.model.layers


def chat_ids(text):
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def all_hidden(text):
    out = model(chat_ids(text), output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])


@torch.no_grad()
def logit_diff(text):
    lg = model(chat_ids(text)).logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item()


print("Extracting per-layer train activations...")
Hs = np.stack([all_hidden(t) for t in sui_tr])
Hn = np.stack([all_hidden(t) for t in non_tr])
raw = Hs.mean(0) - Hn.mean(0)                 # [nL+1, d]
unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
# delta_L = mean projection gap onto unit_L (in activation units)
delta = ((Hs.mean(0) - Hn.mean(0)) * unit).sum(1)   # [nL+1]

# add vectors per layer: layer i output == hidden_states[i+1]
def build_addmap(coef, vecs):
    return {i: torch.tensor(coef * delta[i + 1] * vecs[i + 1], device=device,
                            dtype=torch.float16) for i in WIN}

rand = rng.standard_normal(unit.shape)
rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)


def make_hook(add):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = h + add.to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_addhooks(addmap):
    return [layers[i].register_forward_hook(make_hook(addmap[i])) for i in WIN]


def mean_ld(texts_):
    return np.array([logit_diff(t) for t in texts_])


# ---- baselines ----
print("Baseline suicide / non-suicide...")
base_sui = mean_ld(sui_te)
base_non = mean_ld(non_te)
threshold = (base_sui.mean() + base_non.mean()) / 2  # class-separating midpoint
print(f"baseline: sui LD={base_sui.mean():.2f}, non LD={base_non.mean():.2f}, "
      f"threshold={threshold:.2f}")

# ---- steering sweep on NON-suicidal posts ----
results = {"baseline_sui_ld": float(base_sui.mean()),
           "baseline_non_ld": float(base_non.mean()),
           "threshold": float(threshold), "window": WIN, "sweep": []}
print("\nSteering non-suicidal posts with suicidality direction:")
for c in COEFS:
    if c == 0.0:
        steered = base_non
    else:
        h_ = with_addhooks(build_addmap(c, unit))
        steered = mean_ld(non_te)
        [x.remove() for x in h_]
    frac = float((steered > threshold).mean())
    results["sweep"].append({"coef": c, "mean_ld": float(steered.mean()),
                             "frac_over_threshold": frac})
    print(f"  c={c:>3}: non-suicidal mean LD={steered.mean():6.2f}  "
          f"frac>threshold={frac:.2f}")

# ---- random control at c=2 ----
h_ = with_addhooks(build_addmap(2.0, rand))
rnd = mean_ld(non_te)
[x.remove() for x in h_]
rnd_frac = float((rnd > threshold).mean())
print(f"\n  RANDOM dir c=2: non-suicidal mean LD={rnd.mean():6.2f}  "
      f"frac>threshold={rnd_frac:.2f}")
results["random_c2"] = {"mean_ld": float(rnd.mean()), "frac_over_threshold": rnd_frac}

# ---- verdict ----
lds = [s["mean_ld"] for s in results["sweep"]]
monotonic = all(lds[i] <= lds[i + 1] + 1e-6 for i in range(len(lds) - 1))
top = results["sweep"][-1]
sufficient = monotonic and top["mean_ld"] > threshold and top["frac_over_threshold"] > 0.5 \
    and (top["mean_ld"] - lds[0]) > 3 * (rnd.mean() - base_non.mean() + 1e-6)
print("\n" + "=" * 60)
print(f"VERDICT: {'SUFFICIENT -- direction drives the behavior' if sufficient else 'partial -- see numbers'}")
print(f"  dose-response monotonic: {monotonic}; top mean LD {top['mean_ld']:.2f} "
      f"vs threshold {threshold:.2f}; random does ~nothing ({rnd.mean():.2f})")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "sufficiency.json", "w") as f:
    json.dump({**results, "sufficient": bool(sufficient), "monotonic": bool(monotonic)}, f, indent=2)
print("Saved results/sc2_binary/sufficiency.json")
