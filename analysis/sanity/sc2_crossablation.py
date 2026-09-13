#!/usr/bin/env python
"""Cross-dataset causal ablation — does the 07 direction control OTHER datasets?

Generalization showed the 07 suicidality direction DECODES SDCNL/SWMH. This is
the causal version: ablate the 07 per-layer directions out of the residual stream
and check whether the model's behavioral decision signal collapses on SDCNL and
SWMH too. Control: equal-norm random per-layer directions.

  07-direction ablation collapses behavioral AUC / logit-diff gap on external
  datasets while random does nothing => causal transfer (one general circuit).
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
from src.analysis_utils import (
    PROMPT_TEMPLATES, ensure_dir, load_dataset_by_category, get_text,
)

N_DIR = 100
N_EXT = 80
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


ds07 = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
s07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "suicide"], N_DIR)
n07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "non-suicide"], N_DIR)
dssd = load_from_disk("data/datasets/08_SDCNL/hf_cache")["train"]
sd_s = clean([t for t, l in zip(dssd["text"], dssd["class"]) if l == "suicide"], N_EXT)
sd_n = clean([t for t, l in zip(dssd["text"], dssd["class"]) if l == "non-suicide"], N_EXT)
swmh = load_dataset_by_category(dataset_name="swmh")
sw_s = clean([get_text(x) for x in swmh.get("Death-Me", [])], N_EXT)
sw_n = clean([get_text(x) for x in swmh.get("Life-Not-Me", [])], N_EXT)

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


print("Building 07 per-layer directions...")
Hs = np.stack([all_hidden(t) for t in s07])
Hn = np.stack([all_hidden(t) for t in n07])
dirs = Hs.mean(0) - Hn.mean(0)
dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
rand = rng.standard_normal(dirs.shape); rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
R = {i: torch.tensor(dirs[i + 1], device=device, dtype=torch.float16) for i in range(nL)}
Rr = {i: torch.tensor(rand[i + 1], device=device, dtype=torch.float16) for i in range(nL)}


def make_hook(vec):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(h.dtype)
        h2 = h - (h @ v).unsqueeze(-1) * v
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_hooks(vmap):
    return [layers[i].register_forward_hook(make_hook(vmap[i])) for i in range(nL)]


def evaluate(ss, nn):
    def auc_gap(fn):
        s = np.array([fn(t) for t in ss]); n = np.array([fn(t) for t in nn])
        y = np.array([1] * len(s) + [0] * len(n))
        return roc_auc_score(y, np.concatenate([s, n])), float(s.mean() - n.mean())
    base = auc_gap(logit_diff)
    h_ = with_hooks(R); abl = auc_gap(logit_diff); [x.remove() for x in h_]
    h_ = with_hooks(Rr); rnd = auc_gap(logit_diff); [x.remove() for x in h_]
    return {"baseline": base, "ablate07": abl, "random": rnd}


results = {}
for name, ss, nn in [("SDCNL", sd_s, sd_n), ("SWMH", sw_s, sw_n)]:
    print(f"\nCross-ablation on {name} (n={len(ss)}/class)...")
    results[name] = evaluate(ss, nn)

print("\n" + "=" * 72)
print("CROSS-DATASET CAUSAL ABLATION (ablate 07 direction on external data)")
print("=" * 72)
print(f"{'dataset':8} {'cond':12} {'AUC':>7} {'LD gap':>8}")
print("-" * 72)
ok = True
for name, r in results.items():
    for cond in ["baseline", "ablate07", "random"]:
        a, g = r[cond]
        print(f"{name:8} {cond:12} {a:7.3f} {g:8.2f}")
    drop = r["baseline"][0] - r["ablate07"][0]
    cdrop = r["baseline"][0] - r["random"][0]
    ok = ok and drop > 3 * max(cdrop, 0.01)
    print(f"{'':8} -> AUC drop: 07-dir {drop:+.3f} vs random {cdrop:+.3f}")
    print("-" * 72)
print(f"VERDICT: {'CAUSAL TRANSFER -- 07 direction controls external datasets' if ok else 'partial -- see numbers'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "crossablation.json", "w") as f:
    json.dump({"n_dir": N_DIR, "n_ext": N_EXT,
               "results": {k: {c: {"auc": v[c][0], "ld_gap": v[c][1]} for c in v}
                           for k, v in results.items()},
               "causal_transfer": bool(ok)}, f, indent=2)
print("Saved results/sc2_binary/crossablation.json")
