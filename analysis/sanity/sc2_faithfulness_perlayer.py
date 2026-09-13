#!/usr/bin/env python
"""Faithfulness (per-layer) — ablate each layer's OWN suicidality direction.

The single-direction test (sc2_faithfulness.py) collapsed the decision signal
91% (logit-diff gap 5.33->0.48) but left AUC at 0.71 because earlier layers
re-derive the direction. This ablates the difference-of-means direction at EVERY
layer simultaneously (Arditi-style), which is the rigorous necessity test.
Control: per-layer random directions of equal norm.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

N_TRAIN = 100
N_TEST = 100
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
nL = len(layers)


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


def auc_over(fn):
    s = [fn(t) for t in sui_te]; n = [fn(t) for t in non_te]
    y = np.array([1] * len(s) + [0] * len(n)); sc = np.array(s + n)
    return roc_auc_score(y, sc), float(np.mean(s)), float(np.mean(n))


print("Extracting per-layer train activations...")
Hs = np.stack([all_hidden(t) for t in sui_tr])  # [n, nL+1, d]
Hn = np.stack([all_hidden(t) for t in non_tr])
dirs = Hs.mean(0) - Hn.mean(0)                   # [nL+1, d]
dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
# per-layer random controls
rand = rng.standard_normal(dirs.shape)
rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)

# layer i output == hidden_states[i+1] -> ablate dirs[i+1]
R = {i: torch.tensor(dirs[i + 1], device=device, dtype=torch.float16) for i in range(nL)}
Rr = {i: torch.tensor(rand[i + 1], device=device, dtype=torch.float16) for i in range(nL)}


def make_hook(vec):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(h.dtype)
        h2 = h - (h @ v).unsqueeze(-1) * v
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_hooks(vecmap):
    return [layers[i].register_forward_hook(make_hook(vecmap[i])) for i in range(nL)]


print("Baseline..."); base = auc_over(logit_diff)
print("Per-layer suicidality ablation...")
h_ = with_hooks(R); sui_a = auc_over(logit_diff); [x.remove() for x in h_]
print("Per-layer random ablation (control)...")
h_ = with_hooks(Rr); rnd_a = auc_over(logit_diff); [x.remove() for x in h_]

print("\n" + "=" * 64)
print(f"PER-LAYER FAITHFULNESS (all {nL} layers, test n={N_TEST}/class)")
print("=" * 64)
print(f"{'condition':30} {'AUC':>7} {'LD sui':>8} {'LD non':>8} {'gap':>7}")
print("-" * 64)
for name, (a, s, n) in [("baseline", base), ("ablate suicidality (all L)", sui_a),
                        ("ablate random (all L)", rnd_a)]:
    print(f"{name:30} {a:7.3f} {s:8.2f} {n:8.2f} {s-n:7.2f}")
print("=" * 64)
drop, cdrop = base[0] - sui_a[0], base[0] - rnd_a[0]
faithful = sui_a[0] < 0.60 and drop > 3 * max(cdrop, 0.01)
print(f"AUC drop: suicidality {drop:+.3f} vs random {cdrop:+.3f}")
print(f"VERDICT: {'FAITHFUL -- direction is causally necessary' if faithful else 'partial -- see numbers'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "faithfulness_perlayer.json", "w") as f:
    json.dump({"n_train": N_TRAIN, "n_test": N_TEST, "n_layers": nL,
               "baseline": {"auc": base[0], "ld_sui": base[1], "ld_non": base[2]},
               "suicidality_ablation": {"auc": sui_a[0], "ld_sui": sui_a[1], "ld_non": sui_a[2]},
               "random_ablation": {"auc": rnd_a[0], "ld_sui": rnd_a[1], "ld_non": rnd_a[2]},
               "auc_drop_suicidality": drop, "auc_drop_random": cdrop,
               "faithful": bool(faithful)}, f, indent=2)
print("Saved results/sc2_binary/faithfulness_perlayer.json")
