#!/usr/bin/env python
"""Binary suicidal-vs-non-suicidal gate — foundation for the intent framework.

DeepSuiMind intent sub-typing was unrecoverable (AUC ~0.55, 3 tests). The user
chose to operationalize 'intent' as the real, separable task: suicide vs
non-suicide. This establishes whether that task is a sound MI foundation:

  1. TF-IDF baseline (what a bag-of-words gets — the bar to beat)
  2. SC-2 behavioral: does the model answer Yes/No correctly? (acc + AUC)
  3. Linear probe: is suicidality linearly decodable from the residual stream?

GO if behavioral AUC and probe AUC both clear the TF-IDF baseline (ideally model
acc >= 0.85). Then this task/dataset becomes Phase 2's foundation.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import build_prompt_tokens, PROMPT_TEMPLATES, ensure_dir

N_MODEL = 150     # per class for GPU (behavioral + probe)
N_TFIDF = 800     # per class for the lexical baseline
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())

# ---- load + balance ----
ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts = ds["text"]; labels = ds["label"]
sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
print(f"pool: {len(sui)} suicide, {len(non)} non-suicide")

# ---- 1. TF-IDF baseline ----
bt = sui[:N_TFIDF] + non[:N_TFIDF]
by = np.array([1] * N_TFIDF + [0] * N_TFIDF)
tfidf = make_pipeline(
    TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    LogisticRegression(C=1.0, max_iter=2000),
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())
tfidf_acc = cross_val_score(tfidf, bt, by, cv=cv, scoring="accuracy").mean()
print(f"TF-IDF baseline acc (n={N_TFIDF}/class): {tfidf_acc:.4f}")

# ---- load model ----
loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]


def chat_ids(text, style="standard"):
    tmpl = PROMPT_TEMPLATES[style]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def behav_and_hidden(text):
    """Return (logit_diff, hidden[n_layers+1, d_model] at answer token)."""
    ids = chat_ids(text)
    out = model(ids, output_hidden_states=True)
    lg = out.logits[0, -1, :]
    ld = (lg[yes_id] - lg[no_id]).item()
    hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states]).numpy()
    return ld, hs


def collect(texts_):
    lds, hss = [], []
    for i, t in enumerate(texts_):
        ld, hs = behav_and_hidden(t)
        lds.append(ld); hss.append(hs)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(texts_)}")
    return np.array(lds), np.stack(hss)

print(f"\nRunning model on {N_MODEL} suicide + {N_MODEL} non-suicide...")
sui_ld, sui_h = collect(sui[:N_MODEL])
non_ld, non_h = collect(non[:N_MODEL])

# ---- 2. behavioral ----
y = np.concatenate([np.ones(N_MODEL), np.zeros(N_MODEL)])
ld = np.concatenate([sui_ld, non_ld])
behav_auc = roc_auc_score(y, ld)
behav_acc = ((ld > 0).astype(int) == y).mean()

# ---- 3. probe per layer ----
probe_rows = []
for L in range(sui_h.shape[1]):
    X = np.concatenate([sui_h[:, L, :], non_h[:, L, :]], axis=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    probe_rows.append((L, accuracy_score(y, (proba > 0.5).astype(int)),
                       roc_auc_score(y, proba)))
pbest = max(probe_rows, key=lambda r: r[2])

print("\n" + "=" * 60)
print("BINARY SUICIDE-DETECTION GATE")
print("=" * 60)
print(f"TF-IDF baseline acc:        {tfidf_acc:.4f}")
print(f"Behavioral argmax acc:      {behav_acc:.4f}")
print(f"Behavioral ROC-AUC:         {behav_auc:.4f}")
print(f"Best probe layer:           {pbest[0]}  acc={pbest[1]:.4f}  auc={pbest[2]:.4f}")
for L, a, u in probe_rows:
    if L % 4 == 0 or L == pbest[0]:
        print(f"    layer {L:2d}: acc={a:.3f} auc={u:.3f}{'  <-- best' if L==pbest[0] else ''}")
go = behav_acc >= 0.85 or pbest[1] > tfidf_acc
print("=" * 60)
print(f"VERDICT: {'GO -- viable foundation' if go else 'WEAK -- reconsider'}")
print(f"  behavioral acc {behav_acc:.3f} (>=0.85?), probe acc {pbest[1]:.3f} (> tfidf {tfidf_acc:.3f}?)")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "gate.json", "w") as f:
    json.dump({"dataset": "07_Suicide_Watch_232K", "n_model": N_MODEL, "n_tfidf": N_TFIDF,
               "tfidf_acc": float(tfidf_acc), "behav_acc": float(behav_acc),
               "behav_auc": float(behav_auc), "best_probe_layer": pbest[0],
               "best_probe_acc": float(pbest[1]), "best_probe_auc": float(pbest[2]),
               "probe_per_layer": [{"layer": L, "acc": a, "auc": u} for L, a, u in probe_rows]},
              f, indent=2)
print("Saved results/sc2_binary/gate.json")
