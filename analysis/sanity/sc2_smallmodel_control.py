#!/usr/bin/env python
"""Small-model yes/no positive control (reviewer #4).

The cross-scale claim is "the suicidality REPRESENTATION is present from 0.5B but the
behavioral readout only emerges by ~3B" (encoding without use). The objection: sub-2B
instruct models are simply bad at constrained yes/no formatting, so "fails suicidality
yes/no" might be a readout artifact, not a fact about suicidality. This control
distinguishes the two: run the SAME small models on OTHER yes/no tasks (sentiment,
topic) with the same template and readout. If they handle sentiment/sports yes/no
competently while failing suicidality, the dissociation is real and concept-specific;
if they fail every yes/no task, the cross-scale story is a readout artifact and must be
softened.

For each (model, concept) we report behavioral AUC (logit-diff ranking) and probe-best
AUC (is the representation there?). The diagnostic is the behavioral row: small models
should be high on sentiment/sports, low on suicidality, while probe AUC stays high
everywhere.

Usage:
  python analysis/sanity/sc2_smallmodel_control.py
  python analysis/sanity/sc2_smallmodel_control.py --models Qwen/Qwen2.5-0.5B-Instruct meta-llama/Llama-3.2-1B-Instruct
  python analysis/sanity/sc2_smallmodel_control.py --concepts suicide sentiment sports --n 120

Output: results/sc2_binary/smallmodel_control.json
Each model is 0.5-1.5B in 4-bit -> fits comfortably on a 12 GB GPU.
"""
import sys, json, argparse, gc
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "sanity" / "gcl"))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import ensure_dir
from concept_config import get_concept

MAXLEN = 256
device = "cuda"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
DEFAULT_CONCEPTS = ["suicide", "sentiment", "sports"]


def parse_args():
    p = argparse.ArgumentParser(description="Small-model yes/no positive control")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    p.add_argument("--n", type=int, default=120, help="examples/class")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def eval_concept(model, tok, concept, pos, neg, cv):
    """Return behavioral AUC (logit-diff) and best-layer probe AUC for one concept."""
    yes_id = tok.encode(concept.pos_answer, add_special_tokens=False)[0]
    no_id = tok.encode(concept.neg_answer, add_special_tokens=False)[0]
    use_chat = tok.chat_template is not None

    def make_ids(text):
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = (f"{concept.prompt_prefix}\n\nText: {tok.decode(body)}\n\n"
                   f"{concept.question} Answer with Yes or No.\nAnswer:")
        if use_chat:
            s = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return tok(content, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def feat(text):
        out = model(make_ids(text), output_hidden_states=True)
        lg = out.logits[0, -1, :]
        ld = (lg[yes_id] - lg[no_id]).item()
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
        return ld, hs

    lds, Hs = [], []
    for t in pos + neg:
        ld, h = feat(t); lds.append(ld); Hs.append(h)
    Hs = np.stack(Hs)
    y = np.array([1] * len(pos) + [0] * len(neg))
    behav_auc = float(roc_auc_score(y, np.array(lds)))

    best = 0.0
    for L in range(Hs.shape[1]):
        proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                                  Hs[:, L, :], y, cv=cv, method="predict_proba")[:, 1]
        best = max(best, roc_auc_score(y, proba))
    return behav_auc, float(best)


def main():
    args = parse_args()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())

    # Pre-load concept text once (same texts across models for comparability).
    concept_data = {}
    for cname in args.concepts:
        c = get_concept(cname)
        pos, neg = c.loader(args.n, args.seed)
        n = min(len(pos), len(neg))
        concept_data[cname] = (c, pos[:n], neg[:n])
        print(f"concept {cname}: {n}/class")

    results = {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        loader = ModelLoader(device=device)
        try:
            model, tok = loader.load_model(model_name, load_in_4bit=True, auto_quantize=False)
        except Exception as e:
            print(f"  skip {model_name}: {e}")
            continue
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        results[model_name] = {}
        for cname, (concept, pos, neg) in concept_data.items():
            behav, probe = eval_concept(model, tok, concept, pos, neg, cv)
            results[model_name][cname] = {"behavioral_auc": behav, "probe_best_auc": probe}
            print(f"  {cname:10}: behavioral {behav:.3f}   probe {probe:.3f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- verdict ----
    print("\n" + "=" * 70)
    print("SMALL-MODEL DISSOCIATION (behavioral AUC; probe AUC in parens)")
    print("=" * 70)
    header = f"{'model':32}" + "".join(f"{c:>14}" for c in args.concepts)
    print(header)
    for m, row in results.items():
        line = f"{m.split('/')[-1]:32}"
        for c in args.concepts:
            if c in row:
                line += f"  {row[c]['behavioral_auc']:.2f}({row[c]['probe_best_auc']:.2f})"
        print(line)
    print("=" * 70)
    # dissociation holds if small models do a NON-suicide concept well behaviorally
    # while failing suicidality behaviorally (probe high in both)
    other = [c for c in args.concepts if c != "suicide"]
    verdict = {}
    for m, row in results.items():
        if "suicide" not in row or not other:
            continue
        sui_b = row["suicide"]["behavioral_auc"]
        best_other_b = max(row[c]["behavioral_auc"] for c in other if c in row)
        verdict[m] = bool(best_other_b > 0.70 and sui_b < 0.65)
    print("dissociation (other-concept yes/no works, suicidality does not):")
    for m, v in verdict.items():
        print(f"  {m.split('/')[-1]:32} {'YES -> encoding-without-use is real' if v else 'no -> readout-artifact risk'}")

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    out = PROJECT_ROOT / "results" / "sc2_binary" / "smallmodel_control.json"
    json.dump({"models": args.models, "concepts": args.concepts, "n_per_class": args.n,
               "results": results, "dissociation": verdict}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
