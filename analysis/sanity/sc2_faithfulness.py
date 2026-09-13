#!/usr/bin/env python
"""Faithfulness test — is the layer-16 suicidality direction causally USED?

Specificity showed the direction is present and semantic. Faithfulness asks
whether the model actually relies on it: project the direction out of the
residual stream (layers 15..end, all positions) and measure whether behavioral
suicide-detection collapses. Control: ablate a RANDOM direction of equal norm
(rules out "any ablation degrades the model").

Direction = difference-of-means at layer 16, fit on a TRAIN split, evaluated on a
disjoint TEST split.

  suicidality-ablation AUC -> ~0.5 while random-ablation AUC stays high
      => the direction is causally necessary (faithful circuit).
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

N_TRAIN = 100   # per class, to fit the direction
N_TEST = 100    # per class, held-out evaluation
LAYER = 16      # hidden_states[16] == output of decoder layer 15
ABLATE_FROM = 15  # hook decoder layers 15..last
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
print(f"model has {len(layers)} decoder layers; ablating {ABLATE_FROM}..{len(layers)-1}")


def chat_ids(text):
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def hidden16(text):
    out = model(chat_ids(text), output_hidden_states=True)
    return out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()


@torch.no_grad()
def logit_diff(text):
    lg = model(chat_ids(text)).logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item()


def auc_over(suite_sui, suite_non, fn):
    s = [fn(t) for t in suite_sui]
    n = [fn(t) for t in suite_non]
    y = np.array([1] * len(s) + [0] * len(n))
    sc = np.array(s + n)
    return roc_auc_score(y, sc), float(np.mean(s)), float(np.mean(n))


# ---- fit direction on TRAIN ----
print("Extracting train activations for the direction...")
h_sui = np.stack([hidden16(t) for t in sui_tr])
h_non = np.stack([hidden16(t) for t in non_tr])
direction = h_sui.mean(0) - h_non.mean(0)
direction = direction / np.linalg.norm(direction)
r = torch.tensor(direction, device=device, dtype=torch.float16)

# random control of equal (unit) norm
g = rng.standard_normal(r.shape[0]); g = g / np.linalg.norm(g)
r_rand = torch.tensor(g, device=device, dtype=torch.float16)


def make_hook(vec):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(h.dtype)
        proj = (h @ v).unsqueeze(-1) * v  # remove component along vec, all positions
        h2 = h - proj
        return (h2,) + out[1:] if isinstance(out, tuple) else h2
    return hook


def with_hooks(vec):
    handles = [layers[i].register_forward_hook(make_hook(vec))
               for i in range(ABLATE_FROM, len(layers))]
    return handles


# ---- baseline ----
print("Baseline (no ablation)...")
base_auc, base_s, base_n = auc_over(sui_te, non_te, logit_diff)

# ---- ablate suicidality direction ----
print("Ablating suicidality direction...")
h_ = with_hooks(r)
sui_auc, sui_s, sui_n = auc_over(sui_te, non_te, logit_diff)
for x in h_: x.remove()

# ---- ablate random direction (control) ----
print("Ablating random direction (control)...")
h_ = with_hooks(r_rand)
rnd_auc, rnd_s, rnd_n = auc_over(sui_te, non_te, logit_diff)
for x in h_: x.remove()

print("\n" + "=" * 64)
print("FAITHFULNESS RESULTS (test set, n=%d/class)" % N_TEST)
print("=" * 64)
print(f"{'condition':28} {'AUC':>7} {'mean LD sui':>12} {'mean LD non':>12}")
print("-" * 64)
print(f"{'baseline':28} {base_auc:7.3f} {base_s:12.2f} {base_n:12.2f}")
print(f"{'ablate suicidality dir':28} {sui_auc:7.3f} {sui_s:12.2f} {sui_n:12.2f}")
print(f"{'ablate random dir (ctrl)':28} {rnd_auc:7.3f} {rnd_s:12.2f} {rnd_n:12.2f}")
print("=" * 64)
drop = base_auc - sui_auc
ctrl_drop = base_auc - rnd_auc
faithful = sui_auc < 0.65 and drop > 2 * max(ctrl_drop, 0.01)
print(f"AUC drop: suicidality {drop:+.3f}  vs  random {ctrl_drop:+.3f}")
print(f"VERDICT: {'FAITHFUL -- direction is causally necessary' if faithful else 'WEAK -- ablation effect not specific'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "faithfulness.json", "w") as f:
    json.dump({"layer": LAYER, "ablate_from": ABLATE_FROM, "n_train": N_TRAIN, "n_test": N_TEST,
               "baseline_auc": base_auc, "suicidality_ablation_auc": sui_auc,
               "random_ablation_auc": rnd_auc, "auc_drop_suicidality": drop,
               "auc_drop_random": ctrl_drop, "faithful": bool(faithful),
               "mean_ld": {"baseline": [base_s, base_n], "suicidality": [sui_s, sui_n],
                           "random": [rnd_s, rnd_n]}}, f, indent=2)
print("Saved results/sc2_binary/faithfulness.json")
