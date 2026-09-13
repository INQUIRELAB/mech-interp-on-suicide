#!/usr/bin/env python
"""Intent vs. mention — does the suicidality feature track INTENT or just TOPIC?

The binary task (sc2_*) is ~89% lexical: it separates suicide-topic text from
non-suicide text, and much of that is keyword presence. The open question the
paper never answers: when the *words* are held constant and only the INTENT
changes, does the model's layer-16 feature still fire correctly?

This script uses a lexically-controlled minimal-pairs set
(`data/intent_minpairs/intent_minpairs_v1.jsonl`): every item carries the same
death/suicide vocabulary, but only `genuine_intent` expresses a first-person,
self-directed wish to die. The controls share the vocabulary with NO intent:
  - counterfactual : explicit denial / negation ("I would never end my life")
  - third_person   : someone else's risk ("my brother said he wants to die")
  - topical        : informational / statistical ("suicide rates rose")
  - fiction        : a character in a book/film/game
  - recovery_past  : past, resolved ("I used to want to die, but I'm okay now")

Three readouts, each genuine-intent vs. one control condition:
  1. TF-IDF AUC      -- lexical baseline. Near chance == words can't separate intent.
  2. Behavioral AUC  -- Yes-No logit difference under the standard prompt.
  3. Projection AUC  -- projection of the layer-16 residual onto the EXISTING
                        suicidality direction (difference-of-means fit on the 232K
                        binary task, the paper's central object).

KEY DIAGNOSTIC. If projection AUC for genuine-vs-topical is ~0.5, the paper's
"suicidality feature" is really a death-TOPIC detector. If it is high, the same
direction encodes intent, which strengthens the paper enormously. The
genuine-vs-counterfactual ("I would never") column is the hardest and most
safety-relevant case.

Output: results/sc2_binary/intent_minpairs.json + intent_minpairs.png
Run:    python analysis/sanity/sc2_intent_minpairs.py --n_dir 100
"""
import sys, json, argparse
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
LAYER = 16
device = "cuda"
DEFAULT_DATA = PROJECT_ROOT / "data" / "intent_minpairs" / "intent_minpairs_v1.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Intent-vs-mention minimal pairs probe")
    p.add_argument("--n_dir", type=int, default=100,
                   help="examples/class from 232K to fit the suicidality direction")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                   help="path to the intent jsonl (synthetic minpairs or real set)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def load_minpairs(data_path):
    rows = [json.loads(l) for l in Path(data_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    print(f"Loaded {len(rows)} minimal-pair items across conditions:")
    for c, items in by_cond.items():
        print(f"  {c}: {len(items)}")
    return rows, by_cond


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows, by_cond = load_minpairs(args.data)

    genuine = by_cond["genuine_intent"]
    controls = [c for c in by_cond if c != "genuine_intent"]

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

    def collect(texts, tag):
        lds, hs = [], []
        for i, t in enumerate(texts):
            ld, h = run(t); lds.append(ld); hs.append(h)
            if (i + 1) % 25 == 0:
                print(f"    {tag} {i+1}/{len(texts)}")
        return np.array(lds), np.stack(hs)

    # ---- fit the EXISTING suicidality direction on the 232K binary task ----
    print(f"\nFitting layer-{LAYER} suicidality direction on 232K (n={args.n_dir}/class)...")
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    _, sui_h = collect(sui[: args.n_dir], "dir-sui")
    _, non_h = collect(non[: args.n_dir], "dir-non")
    direction = sui_h.mean(0) - non_h.mean(0)          # difference-of-means at L16
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    # center projections at the midpoint of the two class means
    mid = 0.5 * (sui_h.mean(0) + non_h.mean(0))

    def project(H):
        return (H - mid) @ direction

    # ---- run the model on every minimal-pair item ----
    print("\nRunning model on minimal-pair items...")
    item_ld, item_proj = {}, {}
    for cond, items in by_cond.items():
        ld, H = collect([r["text"] for r in items], cond)
        pr = project(H)
        for r, l, p in zip(items, ld, pr):
            item_ld[r["id"]] = float(l); item_proj[r["id"]] = float(p)

    # ---- TF-IDF lexical baseline (per-condition CV AUC) ----
    def tfidf_auc(pos_items, neg_items):
        txt = [r["text"] for r in pos_items] + [r["text"] for r in neg_items]
        y = np.array([1] * len(pos_items) + [0] * len(neg_items))
        if len(set(y)) < 2 or min((y == 1).sum(), (y == 0).sum()) < 3:
            return float("nan")
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = vec.fit_transform(txt)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        k = min(5, (y == 1).sum(), (y == 0).sum())
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=get_seed())
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))

    def auc(pos_ids, neg_ids, table):
        y = np.array([1] * len(pos_ids) + [0] * len(neg_ids))
        s = np.array([table[i] for i in pos_ids] + [table[i] for i in neg_ids])
        return float(roc_auc_score(y, s))

    def boot_auc(pos_ids, neg_ids, table, n_boot=2000, seed=0):
        """Stratified bootstrap 95% CI for AUC (resample within each class).

        With n_genuine=16 and some controls at n=4, point AUCs are fragile; the CI
        is what should be reported. Returns (auc, lo, hi)."""
        br = np.random.default_rng(seed)
        ps = np.array([table[i] for i in pos_ids]); ns = np.array([table[i] for i in neg_ids])
        np_, nn = len(ps), len(ns)
        point = auc(pos_ids, neg_ids, table)
        if np_ < 2 or nn < 2:
            return point, float("nan"), float("nan")
        ys = np.r_[np.ones(np_), np.zeros(nn)]
        aucs = []
        for _ in range(n_boot):
            idx = np.r_[br.integers(0, np_, np_), np_ + br.integers(0, nn, nn)]
            s = np.r_[ps, ns][idx]; y = ys[idx]
            if len(set(y.tolist())) < 2:
                continue
            aucs.append(roc_auc_score(y, s))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        return point, float(lo), float(hi)

    def make_row(c_items, c_ids):
        pa, plo, phi = boot_auc(g_ids, c_ids, item_proj)
        ba, blo, bhi = boot_auc(g_ids, c_ids, item_ld)
        return {
            "n_control": len(c_ids),
            "tfidf_auc": tfidf_auc(genuine, c_items),
            "behavioral_auc": ba, "behavioral_auc_ci": [blo, bhi],
            "projection_auc": pa, "projection_auc_ci": [plo, phi],
        }

    def show(name, row):
        ci = row["projection_auc_ci"]
        print(f"{name:<16}{row['n_control']:>4}{row['tfidf_auc']:>9.3f}"
              f"{row['behavioral_auc']:>9.3f}{row['projection_auc']:>9.3f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>22}")

    g_ids = [r["id"] for r in genuine]
    results = {}
    for cond in controls:
        c_items = by_cond[cond]
        results[cond] = make_row(c_items, [r["id"] for r in c_items])

    all_ctrl_ids = [r["id"] for c in controls for r in by_cond[c]]
    all_ctrl_items = [r for c in controls for r in by_cond[c]]
    results["ALL_controls"] = make_row(all_ctrl_items, all_ctrl_ids)
    pooled = results["ALL_controls"]

    print("\n" + "=" * 84)
    print(f"INTENT vs MENTION  (genuine intent vs each control; n_genuine={len(g_ids)})")
    print("=" * 84)
    print(f"{'condition':<16}{'n':>4}{'TF-IDF':>9}{'behav':>9}{'proj':>9}{'proj 95% CI':>22}")
    for cond in controls:
        show(cond, results[cond])
    print("-" * 84)
    show("ALL_controls", pooled)
    print("=" * 84)

    # ---- verdict heuristic ----
    proj_topical = results.get("topical", {}).get("projection_auc", float("nan"))
    proj_cf = results.get("counterfactual", {}).get("projection_auc", float("nan"))
    if pooled["projection_auc"] >= 0.70 and proj_cf >= 0.65:
        verdict = ("INTENT-SENSITIVE: the layer-16 feature separates genuine intent "
                   "from lexically-matched mention, including negation. The feature "
                   "encodes intent, not just topic.")
    elif pooled["projection_auc"] < 0.60:
        verdict = ("TOPIC-DETECTOR: the layer-16 feature fires on death vocabulary "
                   "regardless of intent (proj AUC ~chance). The paper's 'suicidality "
                   "feature' is a topic detector; this is a major safety finding.")
    else:
        verdict = ("PARTIAL: the feature carries some intent signal but is confounded "
                   "by topic. Counterfactual/negation is likely the failure mode.")
    print("VERDICT:", verdict)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    # tag by full file stem so paired runs (e.g. v2_raw vs v2_lemma) don't clobber;
    # fall back to the legacy real/synthetic tags for the original two files.
    stem = Path(args.data).stem
    if stem == "intent_minpairs_v1":
        tag = "synthetic"
    elif stem == "intent_real_v1":
        tag = "real"
    else:
        tag = stem.replace("intent_", "")
    json.dump({
        "dataset": Path(args.data).name, "layer": LAYER, "n_dir_per_class": args.n_dir,
        "n_genuine": len(g_ids), "per_condition": results, "verdict": verdict,
        "item_behavioral_ld": item_ld, "item_projection": item_proj,
    }, open(out_dir / f"intent_minpairs_{tag}.json", "w"), indent=2)

    # ---- plot ----
    conds = controls + ["ALL_controls"]
    x = np.arange(len(conds)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(conds) * 1.3), 4.4))
    ax.bar(x - w, [results[c]["tfidf_auc"] for c in conds], w, label="TF-IDF (lexical)", color="#bbb")
    ax.bar(x, [results[c]["behavioral_auc"] for c in conds], w, label="Behavioral (Yes-No)", color="#3498db")
    ax.bar(x + w, [results[c]["projection_auc"] for c in conds], w, label="Layer-16 projection", color="#e74c3c")
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("AUC: genuine intent vs control"); ax.set_ylim(0, 1.05)
    ax.set_title("Does the suicidality feature track INTENT or TOPIC?\n"
                 "(vocabulary held constant; only intent differs)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"intent_minpairs_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results/sc2_binary/intent_minpairs_{tag}.json + .png")


if __name__ == "__main__":
    main()
