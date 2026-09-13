#!/usr/bin/env python
"""SAE autointerp: WHAT does each top suicide-selective L16 feature detect?

§6 (sc2_sae.py) shows a handful of SAE atoms are sharply suicide-selective and
that injecting them flips non-suicidal posts. But the paper never says what those
features MEAN -- so "semantic" rests on the indirect keyword-masking test, not on
a positive characterization of the feature. This closes that gap: for each top
feature we surface (a) its max-activating posts and (b) the single token in each
post that most activates it. That tells us whether the feature tracks
"suicidality" specifically or just "intense negative self-referential affect".

Loads the SAE trained by sc2_sae.py (results/sc2_binary/sae_weights_L16.pt).
Run sc2_sae.py first.

ETHICS: exemplars are truncated to 160 chars for inspection only. Paraphrase /
do not quote identifiable posts before putting anything in the paper (see §Ethics,
safe-messaging). This script does not generate text; it ranks existing posts.

Output: results/sc2_binary/sae_autointerp_L16.json
Run:    python analysis/sanity/sc2_sae_autointerp.py
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, torch.nn as nn, numpy as np
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

LAYER = 16
MAXLEN = 256
N_POOL = 300        # posts/class to search for exemplars
TOP_FEATURES = 5    # characterize the top-5 suicide-selective atoms
TOP_EXEMPLARS = 6   # max-activating posts per feature
EXCERPT_CHARS = 160
device = "cuda"
rng = np.random.default_rng(0)

WEIGHTS = PROJECT_ROOT / "results" / "sc2_binary" / "sae_weights_L16.pt"
if not WEIGHTS.exists():
    raise FileNotFoundError(f"{WEIGHTS} not found -- run analysis/sanity/sc2_sae.py first")


class SparseAutoencoder(nn.Module):
    def __init__(self, d_input, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_sae, d_input))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_input, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)


ckpt = torch.load(WEIGHTS, map_location="cpu")
sae = SparseAutoencoder(ckpt["d_input"], ckpt["d_sae"])
sae.load_state_dict(ckpt["state_dict"])
sae.eval()
a_mean, a_std = ckpt["act_mean"], ckpt["act_std"]
top_idx = ckpt["top_features"][:TOP_FEATURES]
print(f"Loaded SAE d_sae={ckpt['d_sae']}; characterizing features {top_idx}")

ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts, labels = ds["text"], ds["label"]
sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
pool = [(t, "suicide") for t in sui[:N_POOL]] + [(t, "non-suicide") for t in non[:N_POOL]]

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()


def chat_ids(text):
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    body_ids = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body_ids) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    enc = tok(s, add_special_tokens=False, return_tensors="pt")
    return enc["input_ids"].to(device)


@torch.no_grad()
def feature_acts(text):
    """Return (last_token_z[top], per_position_z[top], token_ids) for this post."""
    ids = chat_ids(text)
    out = model(ids, output_hidden_states=True)
    H = out.hidden_states[LAYER][0].float().cpu()             # [seq, d]
    norm = (H - a_mean) / a_std
    z = sae.encode(norm)                                     # [seq, d_sae]
    z_top = z[:, top_idx]                                    # [seq, k]
    return z_top[-1].numpy(), z_top.numpy(), ids[0].cpu().numpy()


# collect last-token activation of each top feature on every pooled post
print(f"Scoring {len(pool)} posts...")
last_acts = np.zeros((len(pool), len(top_idx)))
records = []
for i, (text, lab) in enumerate(pool):
    last_z, perpos_z, ids = feature_acts(text)
    last_acts[i] = last_z
    records.append((text, lab, perpos_z, ids))
    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{len(pool)}")

# per feature: top exemplars by last-token activation + max-activating token
out_features = []
for fi, feat in enumerate(top_idx):
    order = np.argsort(last_acts[:, fi])[::-1][:TOP_EXEMPLARS]
    exemplars = []
    for j in order:
        text, lab, perpos_z, ids = records[j]
        # token that most activates this feature within the post
        pos = int(np.argmax(perpos_z[:, fi]))
        tok_str = tok.decode([int(ids[pos])]).strip()
        exemplars.append({
            "label": lab,
            "last_token_activation": float(last_acts[j, fi]),
            "max_token": tok_str,
            "max_token_activation": float(perpos_z[pos, fi]),
            "excerpt": text[:EXCERPT_CHARS].replace("\n", " "),
        })
    frac_sui = np.mean([1 for e in exemplars if e["label"] == "suicide"]) if exemplars else 0
    out_features.append({"feature": int(feat), "exemplars": exemplars,
                         "frac_exemplars_suicide": float(frac_sui)})
    print(f"\nF{feat}: top exemplars {sum(e['label']=='suicide' for e in exemplars)}/{len(exemplars)} suicide; "
          f"max-tokens = {[e['max_token'] for e in exemplars]}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "sae_autointerp_L16.json", "w") as f:
    json.dump({"layer": LAYER, "n_pool_per_class": N_POOL, "top_features": top_idx,
               "features": out_features,
               "note": "excerpts truncated to 160 chars; paraphrase before paper use"},
              f, indent=2)
print("\nSaved results/sc2_binary/sae_autointerp_L16.json")
print("NOTE: paraphrase/truncate exemplars before quoting in the paper (Ethics: safe-messaging).")
