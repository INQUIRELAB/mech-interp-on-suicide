#!/usr/bin/env python
"""SC-2 AUC test: does the Yes-No logit-diff separate the classes at all?

argmax(Yes,No) gives always-Yes due to safety bias. The fairer question:
is death_logit_diff reliably > life_logit_diff? ROC-AUC answers it
independent of the decision threshold.

  AUC ~0.5  -> model has NO signal for Death-Me vs Life-Not-Me -> NO-GO
  AUC high  -> signal exists, just miscalibrated -> proceed w/ recalibration
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, numpy as np
from sklearn.metrics import roc_auc_score
from src.analysis_utils import (
    load_model_wrapped, load_dataset_by_category, get_text, PROMPT_TEMPLATES, ensure_dir,
)

N = 100
MAXLEN = 256
device = "cuda"
STYLE = "standard"

cats = load_dataset_by_category(dataset_name="deepsuimind")
death = cats["Death-Me"][:N]
life = cats["Life-Not-Me"][:N]
model, tok = load_model_wrapped("llama", device=device)

yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]


def chat_ids(text, style=STYLE):
    tmpl = PROMPT_TEMPLATES[style]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


def diffs(data):
    out = []
    for ex in data:
        ids = chat_ids(get_text(ex))
        with torch.no_grad():
            lg = model(ids)[0, -1, :]
        out.append((lg[yes_id] - lg[no_id]).item())
    return np.array(out)

d_ld = diffs(death)
l_ld = diffs(life)

y_true = np.concatenate([np.ones(len(d_ld)), np.zeros(len(l_ld))])
y_score = np.concatenate([d_ld, l_ld])
auc = roc_auc_score(y_true, y_score)

# Best-threshold accuracy (Youden-style sweep over observed scores).
thresholds = np.unique(y_score)
best_acc, best_t = 0.0, 0.0
for t in thresholds:
    pred = (y_score > t).astype(int)
    acc = (pred == y_true).mean()
    if acc > best_acc:
        best_acc, best_t = acc, float(t)

print("=" * 60)
print(f"SC-2 AUC test (chat template, ' Yes'/' No', {STYLE} prompt, n={N}/class)")
print("=" * 60)
print(f"Death-Me  logit-diff: {d_ld.mean():7.3f} +/- {d_ld.std():.3f}")
print(f"Life-Not  logit-diff: {l_ld.mean():7.3f} +/- {l_ld.std():.3f}")
print(f"ROC-AUC:              {auc:.4f}")
print(f"Best-threshold acc:   {best_acc:.4f}  (threshold logit-diff > {best_t:.3f})")
print(f"argmax(Yes,No) acc:   {((y_score>0).astype(int)==y_true).mean():.4f}")
print("=" * 60)

out = {
    "n_per_class": N, "style": STYLE, "method": "chat_template_space_tokens",
    "death_ld_mean": float(d_ld.mean()), "death_ld_std": float(d_ld.std()),
    "life_ld_mean": float(l_ld.mean()), "life_ld_std": float(l_ld.std()),
    "roc_auc": float(auc), "best_threshold_acc": float(best_acc),
    "best_threshold": best_t,
    "argmax_acc": float(((y_score > 0).astype(int) == y_true).mean()),
}
ensure_dir(PROJECT_ROOT / "results" / "sc2")
with open(PROJECT_ROOT / "results" / "sc2" / "auc.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved results/sc2/auc.json")
