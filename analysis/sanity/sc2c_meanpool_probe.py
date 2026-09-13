#!/usr/bin/env python
"""SC-2c: Mean-pooled linear probe — last chance to find intent signal.

Last-token decoder states are summarization-biased. Mean-pooling hidden
states over the input often surfaces semantic content the final token drops.
If Death-Me vs Life-Not-Me is still ~chance under mean-pooling, the intent
information is genuinely not in the model on this dataset.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import (
    load_dataset_by_category, get_text, build_prompt_tokens, ensure_dir,
)

N = 150
MAXLEN = 256
BASELINE = 0.5988
device = "cuda"

cats = load_dataset_by_category(dataset_name="deepsuimind")
death_me = cats["Death-Me"][:N]
life_not_me = cats["Life-Not-Me"][:N]

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()


@torch.no_grad()
def meanpool_hidden(text):
    ids = build_prompt_tokens(text, tok, style="standard",
                              max_total_tokens=MAXLEN, device=device)
    out = model(ids, output_hidden_states=True)
    # mean over sequence positions, per layer -> [n_layers+1, d_model]
    hs = torch.stack([h[0].float().mean(dim=0).cpu() for h in out.hidden_states])
    return hs.numpy()


def collect(examples):
    mats = []
    for i, ex in enumerate(examples):
        mats.append(meanpool_hidden(get_text(ex)))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(examples)}")
    return np.stack(mats)


print(f"Extracting mean-pooled activations: Death-Me (n={N})")
X_dme = collect(death_me)
print(f"Extracting mean-pooled activations: Life-Not-Me (n={N})")
X_lnm = collect(life_not_me)

y = np.concatenate([np.ones(N), np.zeros(N)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())
rows = []
for L in range(X_dme.shape[1]):
    X = np.concatenate([X_dme[:, L, :], X_lnm[:, L, :]], axis=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    rows.append((L, accuracy_score(y, (proba > 0.5).astype(int)),
                 roc_auc_score(y, proba)))

best = max(rows, key=lambda r: r[2])
print("\n" + "=" * 60)
print("MEAN-POOLED PROBE: Death-Me vs Life-Not-Me")
print("=" * 60)
for L, a, u in rows:
    if L % 2 == 0 or L == best[0]:
        mark = "  <-- best" if L == best[0] else ""
        print(f"  layer {L:2d}: acc={a:.3f}  auc={u:.3f}{mark}")
print(f"  BEST: layer {best[0]}  acc={best[1]:.3f}  auc={best[2]:.3f}")
print(f"  vs baseline {BASELINE}: "
      f"{'VIABLE' if best[1] > BASELINE else 'NOT viable (<= baseline)'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2b")
with open(PROJECT_ROOT / "results" / "sc2b" / "meanpool_probe.json", "w") as f:
    json.dump({"baseline": BASELINE, "n_per_class": N, "pooling": "mean",
               "contrast": "Death-Me_vs_Life-Not-Me",
               "per_layer": [{"layer": L, "acc": a, "auc": u} for L, a, u in rows],
               "best_layer": best[0], "best_acc": best[1], "best_auc": best[2]},
              f, indent=2)
print("Saved results/sc2b/meanpool_probe.json")
