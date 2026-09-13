#!/usr/bin/env python
"""Rank-k subspace ablation — how many directions encode suicidality?

Per-layer rank-1 (diff-of-means) ablation left residual behavioral AUC ~0.66:
the feature is causally dominant but not perfectly rank-1. This ablates a
k-dimensional suicidality subspace per layer (built by iterative deflation:
diff-of-means -> remove -> recompute) and sweeps k to see how fast the signal
dies. Random rank-k subspace is the control.

  AUC -> chance at small k => the circuit is low-rank (a few directions).
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
N_TEST = 80
KS = [1, 2, 4, 8]
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
sui = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "suicide"], N_TRAIN + N_TEST)
non = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "non-suicide"], N_TRAIN + N_TEST)

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
    return (model(chat_ids(text)).logits[0, -1, yes_id]
            - model(chat_ids(text)).logits[0, -1, no_id]).item()


def collect(texts, tag):
    hs = []
    for i, t in enumerate(texts):
        hs.append(all_hidden(t))
        if (i + 1) % 50 == 0:
            print(f"    {tag} {i+1}/{len(texts)}")
    return np.stack(hs)


print("Extracting train activations...")
Hs = collect(sui[:N_TRAIN], "sui"); Hn = collect(non[:N_TRAIN], "non")


def deflate_subspace(A, B, k):
    """k diff-of-means directions via deflation; returns orthonormal [d,k]."""
    A, B = A.copy(), B.copy()
    V = []
    for _ in range(k):
        d = A.mean(0) - B.mean(0)
        d = d / (np.linalg.norm(d) + 1e-8)
        V.append(d)
        A = A - np.outer(A @ d, d)
        B = B - np.outer(B @ d, d)
    V = np.stack(V, axis=1)  # [d,k]
    Q, _ = np.linalg.qr(V)   # orthonormalize
    return Q[:, :k]


# Precompute per-layer subspaces for the max k
maxk = max(KS)
subspaces = {i: deflate_subspace(Hs[:, i + 1, :], Hn[:, i + 1, :], maxk) for i in range(nL)}
rand_sub = {i: np.linalg.qr(rng.standard_normal((Hs.shape[2], maxk)))[0] for i in range(nL)}


def make_hook(Vt):  # Vt: torch [d,k] orthonormal
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        V = Vt.to(h.dtype)
        coeff = h @ V            # [..., k]
        h2 = h - coeff @ V.T     # remove subspace
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_hooks(submap, k):
    handles = []
    for i in range(nL):
        Vt = torch.tensor(submap[i][:, :k], device=device, dtype=torch.float16)
        handles.append(layers[i].register_forward_hook(make_hook(Vt)))
    return handles


sui_te, non_te = sui[N_TRAIN:N_TRAIN + N_TEST], non[N_TRAIN:N_TRAIN + N_TEST]
yte = np.array([1] * N_TEST + [0] * N_TEST)


def behav_auc():
    s = [logit_diff(t) for t in sui_te]; n = [logit_diff(t) for t in non_te]
    return roc_auc_score(yte, np.array(s + n)), float(np.mean(s) - np.mean(n))


print("Baseline...")
base = behav_auc()
rows = []
for k in KS:
    h_ = with_hooks(subspaces, k); a = behav_auc(); [x.remove() for x in h_]
    h_ = with_hooks(rand_sub, k); ar = behav_auc(); [x.remove() for x in h_]
    rows.append((k, a[0], a[1], ar[0]))

print("\n" + "=" * 60)
print("RANK-k SUBSPACE ABLATION (per layer, test n=%d/class)" % N_TEST)
print("=" * 60)
print(f"  baseline AUC={base[0]:.3f} gap={base[1]:.2f}")
print(f"{'k':>3} {'suic AUC':>9} {'gap':>7} {'random AUC':>11}")
for k, a, g, ar in rows:
    print(f"{k:>3} {a:9.3f} {g:7.2f} {ar:11.3f}")
print("=" * 60)
best = min(rows, key=lambda r: r[1])
print(f"min suicidality AUC: {best[1]:.3f} at k={best[0]} "
      f"(random at same k: {best[3]:.3f})")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "rankk.json", "w") as f:
    json.dump({"n_train": N_TRAIN, "n_test": N_TEST, "baseline_auc": base[0],
               "sweep": [{"k": k, "suic_auc": a, "gap": g, "random_auc": ar}
                         for k, a, g, ar in rows]}, f, indent=2)
print("Saved results/sc2_binary/rankk.json")
