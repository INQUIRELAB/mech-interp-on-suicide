#!/usr/bin/env python
"""Specificity test — does the suicidality circuit capture MORE than keywords?

TF-IDF gets 0.894 on the binary task, so much of it is lexical. The framework's
scientific value depends on the layer-16 representation encoding semantics, not
just keyword presence. Two complementary tests:

  Test A (no mutation): on examples TF-IDF MISCLASSIFIES (lexically hard),
      does the model / probe still get them right? If yes -> non-lexical signal.
  Test B (keyword ablation): strip the top TF-IDF suicide keywords out of
      suicidal posts. Does the model still rank them suicidal (behavioral AUC,
      probe recall) while TF-IDF collapses? If yes -> semantic, not keyword.
"""
import sys, json, re
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import build_prompt_tokens, PROMPT_TEMPLATES, ensure_dir

N = 200
LAYER = 16
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())

ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
texts, labels = ds["text"], ds["label"]
sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
rng.shuffle(sui); rng.shuffle(non)
sui, non = sui[:N], non[:N]
all_texts = sui + non
y = np.array([1] * N + [0] * N)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())

# ---- TF-IDF: oof predictions + top suicide keywords ----
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
Xtf = vec.fit_transform(all_texts)
tfidf_clf = LogisticRegression(C=1.0, max_iter=2000)
tfidf_oof = cross_val_predict(tfidf_clf, Xtf, y, cv=cv)
tfidf_acc = accuracy_score(y, tfidf_oof)

tfidf_clf.fit(Xtf, y)
vocab = np.array(vec.get_feature_names_out())
coefs = tfidf_clf.coef_[0]
top_idx = np.argsort(coefs)[::-1][:40]
keywords = set()
for ng in vocab[top_idx]:
    for w in ng.split():
        if len(w) > 2:
            keywords.add(w.lower())
print(f"TF-IDF acc={tfidf_acc:.4f}; top suicide keywords ({len(keywords)}):")
print("  " + ", ".join(sorted(keywords)))


def mask_keywords(text):
    def repl(m):
        return " "
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
                     flags=re.IGNORECASE)
    return pat.sub(repl, text)

sui_masked = [mask_keywords(t) for t in sui]

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
def run(text):
    out = model(chat_ids(text), output_hidden_states=True)
    ld = (out.logits[0, -1, yes_id] - out.logits[0, -1, no_id]).item()
    h = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
    return ld, h


def collect(texts_):
    lds, hs = [], []
    for i, t in enumerate(texts_):
        ld, h = run(t); lds.append(ld); hs.append(h)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(texts_)}")
    return np.array(lds), np.stack(hs)

print(f"\nOriginal suicide (n={N})"); sui_ld, sui_h = collect(sui)
print(f"Original non-suicide (n={N})"); non_ld, non_h = collect(non)
print(f"Keyword-masked suicide (n={N})"); msk_ld, msk_h = collect(sui_masked)

# ---- probe: oof preds on original, fit on original for masked test ----
H = np.concatenate([sui_h, non_h])
probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
probe_oof = cross_val_predict(probe, H, y, cv=cv, method="predict_proba")[:, 1]
probe_acc = accuracy_score(y, (probe_oof > 0.5).astype(int))
probe.fit(H, y)

# ===================== Test A: lexically-hard subset =====================
wrong = tfidf_oof != y
nA = int(wrong.sum())
behav_oof = np.concatenate([sui_ld, non_ld])
# probe acc on the TF-IDF-wrong subset
probeA = accuracy_score(y[wrong], (probe_oof[wrong] > 0.5).astype(int)) if nA else float("nan")
behavA = accuracy_score(y[wrong], (behav_oof[wrong] > np.median(behav_oof)).astype(int)) if nA else float("nan")

# ===================== Test B: keyword ablation =====================
# behavioral: masked-suicide vs original non-suicide
yB = np.array([1] * N + [0] * N)
ldB = np.concatenate([msk_ld, non_ld])
behavB_auc = roc_auc_score(yB, ldB)
# probe recall on suicide before vs after masking
probe_recall_orig = (probe.predict(sui_h) == 1).mean()
probe_recall_mask = (probe.predict(msk_h) == 1).mean()
# TF-IDF recall on masked suicide
tfidf_recall_mask = (tfidf_clf.predict(vec.transform(sui_masked)) == 1).mean()
tfidf_recall_orig = (tfidf_clf.predict(vec.transform(sui)) == 1).mean()

print("\n" + "=" * 64)
print("SPECIFICITY RESULTS")
print("=" * 64)
print(f"TF-IDF oof acc: {tfidf_acc:.3f}   Probe oof acc: {probe_acc:.3f}")
print(f"\n[Test A] lexically-hard subset (TF-IDF wrong), n={nA}")
print(f"  probe acc on these: {probeA:.3f}   (chance=0.5)")
print(f"  -> model recovers lexically-hard cases" if probeA > 0.65 else
      "  -> probe no better than chance on hard cases")
print(f"\n[Test B] keyword ablation (top {len(keywords)} suicide words removed)")
print(f"  TF-IDF suicide recall:  {tfidf_recall_orig:.3f} -> {tfidf_recall_mask:.3f}  "
      f"(drop {tfidf_recall_orig-tfidf_recall_mask:+.3f})")
print(f"  Probe  suicide recall:  {probe_recall_orig:.3f} -> {probe_recall_mask:.3f}  "
      f"(drop {probe_recall_orig-probe_recall_mask:+.3f})")
print(f"  Behavioral AUC (masked-suicide vs non): {behavB_auc:.3f}")
robust = probe_recall_mask > 0.65 and (probe_recall_orig - probe_recall_mask) < (tfidf_recall_orig - tfidf_recall_mask)
print("=" * 64)
print(f"VERDICT: {'SEMANTIC -- circuit survives keyword removal (more than keywords)' if robust else 'LEXICAL -- circuit collapses with keywords (reframe needed)'}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "specificity.json", "w") as f:
    json.dump({
        "n_per_class": N, "layer": LAYER, "n_keywords": len(keywords),
        "keywords": sorted(keywords),
        "tfidf_acc": float(tfidf_acc), "probe_acc": float(probe_acc),
        "testA_hard_n": nA, "testA_probe_acc": float(probeA),
        "tfidf_recall_orig": float(tfidf_recall_orig), "tfidf_recall_masked": float(tfidf_recall_mask),
        "probe_recall_orig": float(probe_recall_orig), "probe_recall_masked": float(probe_recall_mask),
        "behav_auc_masked_vs_non": float(behavB_auc), "verdict_semantic": bool(robust),
    }, f, indent=2)
print("Saved results/sc2_binary/specificity.json")
