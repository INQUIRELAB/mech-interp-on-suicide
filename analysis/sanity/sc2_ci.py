#!/usr/bin/env python
"""Bootstrap 95% CIs on the headline decoding numbers (Llama, layer 16).

Adds rigor to the point estimates: probe AUC (07), behavioral AUC (07), and
cross-dataset transfer AUC (SDCNL, SWMH) — each with a 2000-sample bootstrap CI.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.models import ModelLoader
from src.analysis_utils import (
    PROMPT_TEMPLATES, ensure_dir, load_dataset_by_category, get_text,
)

N07 = 150
NEXT = 120
LAYER = 16
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
s07 = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "suicide"], N07)
n07 = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "non-suicide"], N07)
sd = load_from_disk("data/datasets/08_SDCNL/hf_cache")["train"]
sd_s = clean([t for t, l in zip(sd["text"], sd["class"]) if l == "suicide"], NEXT)
sd_n = clean([t for t, l in zip(sd["text"], sd["class"]) if l == "non-suicide"], NEXT)
sw = load_dataset_by_category(dataset_name="swmh")
sw_s = clean([get_text(x) for x in sw["Death-Me"]], NEXT)
sw_n = clean([get_text(x) for x in sw["Life-Not-Me"]], NEXT)

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]


def chat_ids(text):
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def feat(text):
    out = model(chat_ids(text), output_hidden_states=True)
    lg = out.logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item(), out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()


def collect(texts, tag):
    lds, hs = [], []
    for i, t in enumerate(texts):
        ld, h = feat(t); lds.append(ld); hs.append(h)
        if (i + 1) % 60 == 0:
            print(f"    {tag} {i+1}/{len(texts)}")
    return np.array(lds), np.stack(hs)


def boot_auc(y, s, B=2000):
    idx = np.arange(len(y)); aucs = []
    for _ in range(B):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) == 2:
            aucs.append(roc_auc_score(y[bi], s[bi]))
    return [float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


print("Collecting...")
ld7s, h7s = collect(s07, "07-sui"); ld7n, h7n = collect(n07, "07-non")
ldds, hds = collect(sd_s, "sd-sui"); lddn, hdn = collect(sd_n, "sd-non")
ldws, hws = collect(sw_s, "sw-sui"); ldwn, hwn = collect(sw_n, "sw-non")

dir07 = h7s.mean(0) - h7n.mean(0); dir07 = dir07 / np.linalg.norm(dir07)

# probe AUC (07) with oof proba
y7 = np.array([1] * N07 + [0] * N07)
X7 = np.concatenate([h7s, h7n])
proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                          X7, y7, cv=cv, method="predict_proba")[:, 1]

out = {
    "probe_auc_07": boot_auc(y7, proba),
    "behavioral_auc_07": boot_auc(y7, np.concatenate([ld7s, ld7n])),
    "transfer_auc_SDCNL": boot_auc(np.array([1] * NEXT + [0] * NEXT),
                                   np.concatenate([hds, hdn]) @ dir07),
    "transfer_auc_SWMH": boot_auc(np.array([1] * NEXT + [0] * NEXT),
                                  np.concatenate([hws, hwn]) @ dir07),
    "behavioral_auc_SDCNL": boot_auc(np.array([1] * NEXT + [0] * NEXT),
                                     np.concatenate([ldds, lddn])),
    "behavioral_auc_SWMH": boot_auc(np.array([1] * NEXT + [0] * NEXT),
                                    np.concatenate([ldws, ldwn])),
}

print("\n" + "=" * 60)
print("BOOTSTRAP 95% CIs (Llama, layer 16, 2000 resamples)")
print("=" * 60)
for k, (m, lo, hi) in out.items():
    print(f"  {k:24} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
print("=" * 60)

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "bootstrap_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/sc2_binary/bootstrap_ci.json")
