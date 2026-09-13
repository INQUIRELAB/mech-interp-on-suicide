#!/usr/bin/env python
"""SC-2b: Representational probing gate for a suicidal-INTENT framework.

SC-2 showed the model won't *report* Death-Me vs Life-Not-Me via Yes/No
(AUC 0.557). The framework question is different: does the model *represent*
the distinction internally? A linear probe on the residual stream answers it.

For each layer, extract the hidden state at the answer-token position and run
5-fold CV logistic regression. Compare best-layer accuracy/AUC to the 0.5988
TF-IDF baseline.

  probe >> 0.5988  -> intent is linearly decodable -> framework viable; this
      layer/direction is the Phase-2 foundation.
  probe ~ baseline -> info not in the model -> intent framework not supportable.

Also runs the easier Death-vs-Life contrast (target collapsed) as a second gate.
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

N = 150            # per class
MAXLEN = 256
BASELINE = 0.5988  # TF-IDF logistic-regression baseline (SC-1)
device = "cuda"

cats = load_dataset_by_category(dataset_name="deepsuimind")
death_me = cats["Death-Me"][:N]
life_not_me = cats["Life-Not-Me"][:N]
death_not_me = cats["Death-Not-Me"][:N]   # for Death-vs-Life contrast

print(f"Loading Llama 4-bit...")
loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()


@torch.no_grad()
def hidden_at_answer(text):
    """Return [n_layers+1, d_model] hidden states at the last (answer) token."""
    ids = build_prompt_tokens(text, tok, style="standard",
                              max_total_tokens=MAXLEN, device=device)
    out = model(ids, output_hidden_states=True)
    # hidden_states: tuple(len = n_layers+1) of [1, L, d_model]
    hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])
    return hs.numpy()


def collect(examples):
    mats = []
    for i, ex in enumerate(examples):
        mats.append(hidden_at_answer(get_text(ex)))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(examples)}")
    return np.stack(mats)  # [n, n_layers+1, d_model]


def probe_all_layers(Xa, Xb, name):
    """5-fold CV linear probe per layer. Xa=class1, Xb=class0."""
    n_layers = Xa.shape[1]
    y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())
    rows = []
    for L in range(n_layers):
        X = np.concatenate([Xa[:, L, :], Xb[:, L, :]], axis=0)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000))
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba > 0.5).astype(int)
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, proba)
        rows.append((L, acc, auc))
    return rows


print(f"\nExtracting activations: Death-Me (n={len(death_me)})")
X_dme = collect(death_me)
print(f"Extracting activations: Life-Not-Me (n={len(life_not_me)})")
X_lnm = collect(life_not_me)
print(f"Extracting activations: Death-Not-Me (n={len(death_not_me)})")
X_dnm = collect(death_not_me)

results = {"baseline": BASELINE, "n_per_class": N, "contrasts": {}}

for name, Xa, Xb in [
    ("Death-Me_vs_Life-Not-Me", X_dme, X_lnm),     # the project's core claim
    ("Death_vs_Life", np.concatenate([X_dme, X_dnm]), X_lnm),  # collapse target-ish
]:
    rows = probe_all_layers(Xa, Xb, name)
    best = max(rows, key=lambda r: r[2])
    results["contrasts"][name] = {
        "per_layer": [{"layer": L, "acc": a, "auc": u} for L, a, u in rows],
        "best_layer": best[0], "best_acc": best[1], "best_auc": best[2],
    }
    print("\n" + "=" * 60)
    print(f"PROBE: {name}")
    print("=" * 60)
    for L, a, u in rows:
        mark = "  <-- best" if L == best[0] else ""
        if L % 2 == 0 or L == best[0]:
            print(f"  layer {L:2d}: acc={a:.3f}  auc={u:.3f}{mark}")
    print(f"  BEST: layer {best[0]}  acc={best[1]:.3f}  auc={best[2]:.3f}")
    verdict = "VIABLE (beats baseline)" if best[1] > BASELINE else "NOT viable (<= baseline)"
    print(f"  vs baseline {BASELINE}: {verdict}")

ensure_dir(PROJECT_ROOT / "results" / "sc2b")
with open(PROJECT_ROOT / "results" / "sc2b" / "probe_gate.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/sc2b/probe_gate.json")
