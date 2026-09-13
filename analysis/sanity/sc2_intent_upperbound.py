#!/usr/bin/env python
"""Intent NO-GO upper bound: is the failure the MODEL or the DATASET/linearity?

The paper's negative result (§4.1) currently rests on a *linear* logistic probe
(<=0.57) below a TF-IDF baseline (0.599). Two reviewer objections remain open:

  (1) "Did you try a NONLINEAR probe?" -- maybe Death-Me vs Life-Not-Me is
      represented but not LINEARLY at the answer token. We add a 2-layer MLP
      probe per layer. If it is still ~chance, the negative result survives
      nonlinearity ("not even nonlinearly decodable at the answer token").

  (2) "Are the LABELS even separable by anything?" -- maybe the synthetic
      Death-Me/Life-Not-Me split is barely learnable, so the model 'failing' is
      a dataset property, not a fixable model limitation. We add a strong
      SUPERVISED upper bound on the raw text (word 1-2grams + char 2-5grams,
      tuned LR, 5-fold). If even that is ~0.6, the task is intrinsically hard.

Positive control: the SAME MLP probe on the validated binary task (232K) should
recover ~0.97 at L16, confirming the probe family and pipeline are sound -- so a
chance result on intent is about the signal, not the method.

Output: results/sc2b/intent_upperbound.json
Run:    python analysis/sanity/sc2_intent_upperbound.py
Needs the 4-bit Llama on a >=12 GB GPU.
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
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import (
    load_dataset_by_category, get_text, build_prompt_tokens,
    PROMPT_TEMPLATES, ensure_dir,
)

N = 150            # per class for the intent task
N_POS = 150        # per class for the binary positive control
LAYER_POS = 16     # positive-control readout layer
MAXLEN = 256
TFIDF_BASELINE = 0.5988
device = "cuda"
rng = np.random.default_rng(get_seed())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())


def linear_probe(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), accuracy_score(y, (p > 0.5).astype(int))


def mlp_probe(X, y):
    # small 2-layer MLP; the question is only whether ANY nonlinear boundary helps
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                      early_stopping=True, random_state=get_seed()),
    )
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), accuracy_score(y, (p > 0.5).astype(int))


# --------------------------------------------------------------------------
# 1. Supervised text upper bound on the intent labels (no model needed)
# --------------------------------------------------------------------------
cats = load_dataset_by_category(dataset_name="deepsuimind")
dme = [get_text(e) for e in cats["Death-Me"][:N]]
lnm = [get_text(e) for e in cats["Life-Not-Me"][:N]]
texts = dme + lnm
y = np.array([1] * N + [0] * N)

word = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
char = TfidfVectorizer(max_features=8000, ngram_range=(2, 5), analyzer="char_wb")
from scipy.sparse import hstack
Xtxt = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
sup_p = cross_val_predict(LogisticRegression(C=2.0, max_iter=4000), Xtxt, y, cv=cv,
                          method="predict_proba")[:, 1]
sup_auc = roc_auc_score(y, sup_p)
print(f"[supervised upper bound] word+char TF-IDF LR AUC on intent = {sup_auc:.3f} "
      f"(plain TF-IDF baseline {TFIDF_BASELINE})")

# --------------------------------------------------------------------------
# 2. Model activations -> linear + MLP probe per layer (intent)
# --------------------------------------------------------------------------
loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()


@torch.no_grad()
def last_token_hidden(text):
    ids = build_prompt_tokens(text, tok, style="standard",
                              max_total_tokens=MAXLEN, device=device)
    out = model(ids, output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])


def collect(texts_):
    mats = []
    for i, t in enumerate(texts_):
        mats.append(last_token_hidden(t))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(texts_)}")
    return np.stack(mats)


print(f"\nExtracting intent activations: Death-Me / Life-Not-Me (n={N}/class)")
X_dme = collect(dme)
X_lnm = collect(lnm)

per_layer = []
best_lin = (0, 0.0)
best_mlp = (0, 0.0)
for L in range(X_dme.shape[1]):
    X = np.concatenate([X_dme[:, L, :], X_lnm[:, L, :]], axis=0)
    lin_auc, lin_acc = linear_probe(X, y)
    mlp_auc, mlp_acc = mlp_probe(X, y)
    per_layer.append({"layer": L, "linear_auc": lin_auc, "linear_acc": lin_acc,
                      "mlp_auc": mlp_auc, "mlp_acc": mlp_acc})
    if lin_auc > best_lin[1]:
        best_lin = (L, lin_auc)
    if mlp_auc > best_mlp[1]:
        best_mlp = (L, mlp_auc)
    if L % 4 == 0:
        print(f"  L{L:2d}: linear AUC={lin_auc:.3f}  MLP AUC={mlp_auc:.3f}")

# --------------------------------------------------------------------------
# 3. Positive control: MLP probe on the validated binary task at L16
# --------------------------------------------------------------------------
print(f"\nPositive control: MLP probe on binary 232K at L{LAYER_POS} (n={N_POS}/class)")
ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts_b, labels_b = ds["text"], ds["label"]
sui = [t for t, l in zip(texts_b, labels_b) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts_b, labels_b) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
sui, non = sui[:N_POS], non[:N_POS]


@torch.no_grad()
def hidden_at(text, layer):
    ids = build_prompt_tokens(text, tok, style="standard",
                              max_total_tokens=MAXLEN, device=device)
    out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float().cpu().numpy()


Xp = np.stack([hidden_at(t, LAYER_POS) for t in sui] +
              [hidden_at(t, LAYER_POS) for t in non])
yp = np.array([1] * N_POS + [0] * N_POS)
pos_lin_auc, _ = linear_probe(Xp, yp)
pos_mlp_auc, _ = mlp_probe(Xp, yp)
print(f"  binary L{LAYER_POS}: linear AUC={pos_lin_auc:.3f}  MLP AUC={pos_mlp_auc:.3f}")

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
intent_ceiling = max(best_lin[1], best_mlp[1], sup_auc)
robust_nogo = intent_ceiling <= TFIDF_BASELINE + 0.03 and pos_mlp_auc > 0.85
print("\n" + "=" * 68)
print("INTENT NO-GO UPPER BOUND")
print("=" * 68)
print(f"  supervised text ceiling (word+char): AUC {sup_auc:.3f}")
print(f"  best LINEAR probe (intent):  L{best_lin[0]}  AUC {best_lin[1]:.3f}")
print(f"  best MLP    probe (intent):  L{best_mlp[0]}  AUC {best_mlp[1]:.3f}")
print(f"  positive control (binary L16) MLP AUC {pos_mlp_auc:.3f}  (pipeline {'OK' if pos_mlp_auc>0.85 else 'CHECK'})")
print(f"VERDICT: {'NO-GO ROBUST -- not even nonlinearly decodable, and labels barely separable by any supervised method' if robust_nogo else 'reconsider -- something exceeds baseline'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2b")
with open(PROJECT_ROOT / "results" / "sc2b" / "intent_upperbound.json", "w") as f:
    json.dump({
        "n_per_class_intent": N, "n_per_class_binary": N_POS,
        "tfidf_baseline": TFIDF_BASELINE,
        "supervised_text_auc": float(sup_auc),
        "intent_best_linear": {"layer": best_lin[0], "auc": float(best_lin[1])},
        "intent_best_mlp": {"layer": best_mlp[0], "auc": float(best_mlp[1])},
        "intent_per_layer": per_layer,
        "positive_control_binary_L16": {"linear_auc": float(pos_lin_auc),
                                        "mlp_auc": float(pos_mlp_auc)},
        "nogo_robust": bool(robust_nogo),
    }, f, indent=2)
print("Saved results/sc2b/intent_upperbound.json")
