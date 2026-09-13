#!/usr/bin/env python
"""Generalization — does the suicidality circuit transfer across datasets?

A framework (not a one-off) requires the same pipeline + the same direction to
work on other distributions. For each external dataset (SWMH, SDCNL):

  1. behavioral AUC      — does the model do the task there? (logit-diff)
  2. within-probe AUC    — is suicidality linearly decodable there? (5-fold CV)
  3. TRANSFER AUC        — does the direction learned on 07_Suicide_Watch_232K
                           separate THIS dataset's classes with no retraining?
  4. direction cosine    — angle between the 07 direction and this dataset's own
                           difference-of-means direction (same feature?)

High transfer AUC + high cosine across datasets => a general suicidality feature.
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import (
    PROMPT_TEMPLATES, ensure_dir, load_dataset_by_category, get_text,
)

N_DIR = 100    # per class to fit the 07 reference direction
N_EXT = 120    # per class per external dataset
LAYER = 16
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


# ---- assemble dataset text pools ----
ds07 = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
s07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "suicide"], N_DIR)
n07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "non-suicide"], N_DIR)

dssd = load_from_disk("data/datasets/08_SDCNL/hf_cache")["train"]
sd_s = clean([t for t, l in zip(dssd["text"], dssd["class"]) if l == "suicide"], N_EXT)
sd_n = clean([t for t, l in zip(dssd["text"], dssd["class"]) if l == "non-suicide"], N_EXT)

swmh = load_dataset_by_category(dataset_name="swmh")
sw_s = clean([get_text(x) for x in swmh.get("Death-Me", [])], N_EXT)        # SuicideWatch
sw_n = clean([get_text(x) for x in swmh.get("Life-Not-Me", [])], N_EXT)     # other subreddits
print(f"pools: 07({len(s07)}/{len(n07)})  SDCNL({len(sd_s)}/{len(sd_n)})  SWMH({len(sw_s)}/{len(sw_n)})")

# ---- model ----
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
    ld = (lg[yes_id] - lg[no_id]).item()
    h = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
    return ld, h


def collect(texts, tag):
    lds, hs = [], []
    for i, t in enumerate(texts):
        ld, h = feat(t); lds.append(ld); hs.append(h)
        if (i + 1) % 60 == 0:
            print(f"    {tag} {i+1}/{len(texts)}")
    return np.array(lds), np.stack(hs)


print("Extracting 07 reference direction...")
_, h07s = collect(s07, "07-sui"); _, h07n = collect(n07, "07-non")
dir07 = h07s.mean(0) - h07n.mean(0)
dir07 = dir07 / np.linalg.norm(dir07)

results = {}
for name, ss, nn in [("SDCNL", sd_s, sd_n), ("SWMH", sw_s, sw_n)]:
    print(f"\nProcessing {name}...")
    ld_s, h_s = collect(ss, f"{name}-sui")
    ld_n, h_n = collect(nn, f"{name}-non")
    y = np.array([1] * len(ld_s) + [0] * len(ld_n))
    # 1. behavioral
    behav_auc = roc_auc_score(y, np.concatenate([ld_s, ld_n]))
    # 2. within probe
    X = np.concatenate([h_s, h_n])
    proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                              X, y, cv=cv, method="predict_proba")[:, 1]
    within_auc = roc_auc_score(y, proba)
    within_acc = accuracy_score(y, (proba > 0.5).astype(int))
    # 3. transfer: project onto 07 direction
    transfer_auc = roc_auc_score(y, X @ dir07)
    # 4. cosine with this dataset's own direction
    own = h_s.mean(0) - h_n.mean(0); own = own / np.linalg.norm(own)
    cosine = float(own @ dir07)
    results[name] = {"behavioral_auc": float(behav_auc), "within_probe_auc": float(within_auc),
                     "within_probe_acc": float(within_acc), "transfer_auc": float(transfer_auc),
                     "direction_cosine_vs_07": cosine, "n_per_class": len(ld_s)}

print("\n" + "=" * 70)
print("GENERALIZATION RESULTS")
print("=" * 70)
print(f"{'dataset':10} {'behav AUC':>10} {'within AUC':>11} {'TRANSFER AUC':>13} {'cos(dir,07)':>12}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:10} {r['behavioral_auc']:10.3f} {r['within_probe_auc']:11.3f} "
          f"{r['transfer_auc']:13.3f} {r['direction_cosine_vs_07']:12.3f}")
print("=" * 70)
ok = all(r["transfer_auc"] > 0.75 and r["within_probe_auc"] > 0.75 for r in results.values())
print(f"VERDICT: {'GENERALIZES -- same direction works across datasets' if ok else 'partial -- see numbers'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "generalization.json", "w") as f:
    json.dump({"layer": LAYER, "reference_dataset": "07_Suicide_Watch_232K",
               "n_dir": N_DIR, "results": results, "generalizes": bool(ok)}, f, indent=2)
print("Saved results/sc2_binary/generalization.json")
