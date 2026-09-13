#!/usr/bin/env python
"""Second-model replication — does the suicidality circuit exist in other models?

Replicates the core results on a second model (Qwen2.5-7B / Mistral-7B, both base)
to make the framework cross-model:
  - probe AUC per layer (is suicidality linearly represented?)
  - behavioral AUC (logit-diff)
  - faithfulness: per-layer diff-of-means ablation vs random control

Uses raw prompting (base models have no chat template). Representation-level
results (probe, faithfulness) are the primary cross-model claim.

Usage: python sc2_secondmodel.py [qwen|mistral]
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.models import ModelLoader
from src.analysis_utils import build_prompt_tokens, ensure_dir

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen"
N = 150          # per class total (probe via CV)
N_TRAIN = 100    # per class for ablation direction
N_TEST = 50      # per class held-out for ablation
MAXLEN = 256
BOOT = 2000      # bootstrap resamples for 95% CIs
device = "cuda"
rng = np.random.default_rng(0)
brng = np.random.default_rng(12345)   # bootstrap RNG
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def boot_ci(score_pos, score_neg, B=BOOT):
    """Stratified bootstrap 95% CI for ROC-AUC (resample within each class)."""
    sp, sn = np.asarray(score_pos), np.asarray(score_neg)
    np_, nn = len(sp), len(sn)
    y = np.r_[np.ones(np_), np.zeros(nn)]
    aucs = np.empty(B)
    for b in range(B):
        bs = sp[brng.integers(0, np_, np_)]; bn = sn[brng.integers(0, nn, nn)]
        aucs[b] = roc_auc_score(y, np.r_[bs, bn])
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
sui = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "suicide"], N)
non = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "non-suicide"], N)

loader = ModelLoader(device=device)
load_fn = {"qwen": loader.load_qwen, "mistral": loader.load_mistral}[MODEL]
model, tok = load_fn(load_in_4bit=True, auto_quantize=False)
model.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]
layers = model.model.layers
nL = len(layers)
print(f"{MODEL}: {nL} layers; yes_id={yes_id} no_id={no_id}")


@torch.no_grad()
def feat(text):
    ids = build_prompt_tokens(text, tok, style="standard", max_total_tokens=MAXLEN, device=device)
    out = model(ids, output_hidden_states=True)
    lg = out.logits[0, -1, :]
    ld = (lg[yes_id] - lg[no_id]).item()
    hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
    return ld, hs


@torch.no_grad()
def logit_diff(text):
    ids = build_prompt_tokens(text, tok, style="standard", max_total_tokens=MAXLEN, device=device)
    lg = model(ids).logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item()


def collect(texts, tag):
    lds, hs = [], []
    for i, t in enumerate(texts):
        ld, h = feat(t); lds.append(ld); hs.append(h)
        if (i + 1) % 50 == 0:
            print(f"    {tag} {i+1}/{len(texts)}")
    return np.array(lds), np.stack(hs)


print("Extracting activations...")
ld_s, H_s = collect(sui, "sui"); ld_n, H_n = collect(non, "non")

# ---- probe per layer ----
y = np.array([1] * N + [0] * N)
best = (0, 0.0)
best_proba = None
per_layer = []
for L in range(H_s.shape[1]):
    X = np.concatenate([H_s[:, L, :], H_n[:, L, :]])
    proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                              X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    per_layer.append((L, auc))
    if auc > best[1]:
        best = (L, auc); best_proba = proba.copy()
behav_auc = roc_auc_score(y, np.concatenate([ld_s, ld_n]))

# ---- bootstrap 95% CIs on the headline AUCs ----
probe_ci = boot_ci(best_proba[:N], best_proba[N:])
behav_ci = boot_ci(ld_s, ld_n)

# ---- faithfulness: per-layer ablation, train/test split ----
trs, tes = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
dirs = H_s[trs].mean(0) - H_n[trs].mean(0)
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


sui_te, non_te = sui[N_TRAIN:N_TRAIN + N_TEST], non[N_TRAIN:N_TRAIN + N_TEST]
yte = np.array([1] * N_TEST + [0] * N_TEST)


def behav_auc_te():
    s = [logit_diff(t) for t in sui_te]; n = [logit_diff(t) for t in non_te]
    return roc_auc_score(yte, np.array(s + n)), float(np.mean(s) - np.mean(n))


print("Faithfulness baseline / ablate / random...")
base_f = behav_auc_te()
h_ = with_hooks(R); abl_f = behav_auc_te(); [x.remove() for x in h_]
h_ = with_hooks(Rr); rnd_f = behav_auc_te(); [x.remove() for x in h_]

print("\n" + "=" * 64)
print(f"SECOND-MODEL REPLICATION: {MODEL} (4-bit, n={N}/class)")
print("=" * 64)
print(f"  best probe layer: {best[0]}  AUC={best[1]:.3f}  CI95=[{probe_ci[0]:.3f},{probe_ci[1]:.3f}]")
print(f"  behavioral AUC:   {behav_auc:.3f}  CI95=[{behav_ci[0]:.3f},{behav_ci[1]:.3f}]")
print(f"  faithfulness (test n={N_TEST}/class):")
print(f"    baseline       AUC={base_f[0]:.3f} gap={base_f[1]:.2f}")
print(f"    ablate suic.   AUC={abl_f[0]:.3f} gap={abl_f[1]:.2f}")
print(f"    ablate random  AUC={rnd_f[0]:.3f} gap={rnd_f[1]:.2f}")
drop, cdrop = base_f[0] - abl_f[0], base_f[0] - rnd_f[0]
ok = best[1] > 0.75 and drop > 3 * max(cdrop, 0.01)
print(f"  -> AUC drop suic {drop:+.3f} vs random {cdrop:+.3f}")
print(f"VERDICT: {'REPLICATES -- suicidality circuit exists in '+MODEL if ok else 'partial -- see numbers'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / f"secondmodel_{MODEL}.json", "w") as f:
    json.dump({"model": MODEL, "n": N, "best_probe_layer": best[0], "best_probe_auc": best[1],
               "best_probe_auc_ci95": probe_ci, "behavioral_auc_ci95": behav_ci,
               "bootstrap_resamples": BOOT,
               "behavioral_auc": behav_auc,
               "faithfulness": {"baseline_auc": base_f[0], "baseline_gap": base_f[1],
                                "ablate_auc": abl_f[0], "ablate_gap": abl_f[1],
                                "random_auc": rnd_f[0], "random_gap": rnd_f[1]},
               "per_layer_auc": [{"layer": L, "auc": a} for L, a in per_layer],
               "replicates": bool(ok)}, f, indent=2)
print(f"Saved results/sc2_binary/secondmodel_{MODEL}.json")
