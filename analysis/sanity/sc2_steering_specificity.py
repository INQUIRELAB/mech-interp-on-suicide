#!/usr/bin/env python
"""Steering SPECIFICITY: does the suicidality direction induce SUICIDALITY, or
just push the model to say "Yes" to anything?

§4.5 shows that adding the direction flips 100% of non-suicidal posts past the
"suicidal?" threshold. But the model is affirmatively biased (it answers "Yes"
to most distressed text), so a reviewer can object: maybe the direction just
inflates the Yes-logit on ANY yes/no question. The clean control is an
OFF-TARGET question read out from the SAME steered forward pass.

We steer non-suicidal posts with c * delta_L * r_L (layers 12-20, exactly as
sc2_sufficiency.py) and, on each steered pass, read TWO yes/no questions:
  on-target : "Does this text show signs of suicidal ideation?"
  off-target: "Is this text primarily about food or cooking?"
both as logit(" Yes") - logit(" No").

  steering moves the on-target LD a lot but the off-target LD ~nothing
      => the direction is suicidality-SPECIFIC, not generic Yes-pushing.

Output: results/sc2_binary/steering_specificity.json
Run:    python analysis/sanity/sc2_steering_specificity.py
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
WIN = list(range(12, 21))
COEFS = [0.0, 1.0, 2.0]
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())

# off-target yes/no prompt with the same structure as the standard template
OFFTARGET_TEMPLATE = (
    "Read the following text and determine if it is primarily about food or cooking.\n\n"
    "Text: {text}\n\n"
    "Is this text primarily about food or cooking? Answer with Yes or No.\n"
    "Answer:"
)

ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts, labels = ds["text"], ds["label"]
sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
sui_tr = sui[:N_TRAIN]
non_tr = non[:N_TRAIN]
non_te = non[N_TRAIN:N_TRAIN + N_TEST]

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]
layers = model.model.layers


def chat_ids(text, template):
    prefix, suffix = template.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def all_hidden(text):
    out = model(chat_ids(text, PROMPT_TEMPLATES["standard"]), output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])


@torch.no_grad()
def logit_diff(text, template):
    lg = model(chat_ids(text, template)).logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item()


# ---- suicidality direction (from on-target activations) ----
print("Extracting per-layer train activations...")
Hs = np.stack([all_hidden(t) for t in sui_tr])
Hn = np.stack([all_hidden(t) for t in non_tr])
raw = Hs.mean(0) - Hn.mean(0)
unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
delta = ((Hs.mean(0) - Hn.mean(0)) * unit).sum(1)


def build_addmap(coef):
    return {i: torch.tensor(coef * delta[i + 1] * unit[i + 1], device=device,
                            dtype=torch.float16) for i in WIN}


def make_hook(add):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = h + add.to(h.dtype)
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_addhooks(addmap):
    return [layers[i].register_forward_hook(make_hook(addmap[i])) for i in WIN]


def mean_ld(texts_, template):
    return np.array([logit_diff(t, template) for t in texts_])


# ---- sweep: read BOTH questions on every steered pass ----
results = {"window": WIN, "sweep": []}
print("\nSteering non-suicidal posts; reading on-target vs off-target question:")
for c in COEFS:
    if c == 0.0:
        on = mean_ld(non_te, PROMPT_TEMPLATES["standard"])
        off = mean_ld(non_te, OFFTARGET_TEMPLATE)
    else:
        addmap = build_addmap(c)
        h_ = with_addhooks(addmap)
        on = mean_ld(non_te, PROMPT_TEMPLATES["standard"])
        [x.remove() for x in h_]
        h_ = with_addhooks(addmap)
        off = mean_ld(non_te, OFFTARGET_TEMPLATE)
        [x.remove() for x in h_]
    results["sweep"].append({"coef": c, "on_target_ld": float(on.mean()),
                             "off_target_ld": float(off.mean())})
    print(f"  c={c:>3}: on-target(suicidal?) LD={on.mean():6.2f}   "
          f"off-target(food?) LD={off.mean():6.2f}")

base = results["sweep"][0]
top = results["sweep"][-1]
on_shift = top["on_target_ld"] - base["on_target_ld"]
off_shift = top["off_target_ld"] - base["off_target_ld"]
specific = on_shift > 3 * abs(off_shift) and on_shift > 2.0
print("\n" + "=" * 60)
print(f"  on-target shift  = {on_shift:+.2f}")
print(f"  off-target shift = {off_shift:+.2f}")
print(f"VERDICT: {'SPECIFIC -- direction induces suicidality, not generic Yes' if specific else 'NOT specific -- off-target moves too (generic Yes-pushing)'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "steering_specificity.json", "w") as f:
    json.dump({**results, "on_target_shift": float(on_shift),
               "off_target_shift": float(off_shift), "specific": bool(specific),
               "off_target_question": "Is this text primarily about food or cooking?"},
              f, indent=2)
print("Saved results/sc2_binary/steering_specificity.json")
