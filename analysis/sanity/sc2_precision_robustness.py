#!/usr/bin/env python
"""Precision robustness (reviewer #2) — do the headline rungs survive higher precision?

The whole project runs Llama-3.1-8B in 4-bit NF4 (12 GB GPU). A reviewer rightly
asks whether the geometry we report (probe AUC, necessity ablation, rank-4 collapse)
is a property of the MODEL or of its 4-bit approximation. This script recomputes the
three load-bearing rungs at a chosen precision and writes a precision-tagged JSON so
4-bit vs 8-bit (vs fp16/bf16 on a bigger GPU) can be diffed directly.

Rungs reproduced (same data/prompt/readout as sc2_faithfulness_perlayer.py):
  R1 decodability : linear-probe AUC at layer 16 (5-fold)
  R2 necessity    : per-layer diff-of-means ablation AUC (vs random control)
  R3 low-rank     : rank-4 subspace ablation AUC (vs random rank-4)

Usage:
  python analysis/sanity/sc2_precision_robustness.py --precision 4bit   # baseline
  python analysis/sanity/sc2_precision_robustness.py --precision 8bit   # fits 12 GB
  python analysis/sanity/sc2_precision_robustness.py --precision fp16   # needs ~16 GB
After running 4bit + another precision, re-run with --compare to print the deltas.

Output: results/sc2_binary/precision_{precision}.json
Note: fp16/bf16 of an 8B model needs ~16 GB VRAM and will OOM on a 12 GB card; run
those on a larger GPU (or use --model for a smaller model that fits at fp16).
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
LAYER = 16
RANK = 4
device = "cuda"
OUTDIR = PROJECT_ROOT / "results" / "sc2_binary"


def parse_args():
    p = argparse.ArgumentParser(description="Precision robustness of headline rungs")
    p.add_argument("--precision", choices=["4bit", "8bit", "fp16", "bf16"], default="4bit")
    p.add_argument("--model", default="llama-3.1-8b-instruct")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--compare", action="store_true",
                   help="don't run the model; just diff existing precision_*.json files")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def load_model_at(model_name, precision):
    loader = ModelLoader(device=device)
    kw = dict(auto_quantize=False)
    if precision == "4bit":
        kw.update(load_in_4bit=True)
    elif precision == "8bit":
        kw.update(load_in_8bit=True)
    elif precision == "fp16":
        pass  # default cuda dtype is fp16
    elif precision == "bf16":
        kw.update(torch_dtype=torch.bfloat16)
    return loader.load_model(model_name, **kw)


def deflate_subspace(Hs, Hn, k):
    """k diff-of-means directions via iterative deflation, orthonormalized -> [k, d]."""
    rs, rn = Hs.copy(), Hn.copy()
    dirs = []
    for _ in range(k):
        d = rs.mean(0) - rn.mean(0)
        u = d / (np.linalg.norm(d) + 1e-8)
        dirs.append(u)
        rs = rs - np.outer(rs @ u, u)
        rn = rn - np.outer(rn @ u, u)
    Q, _ = np.linalg.qr(np.stack(dirs, 1))
    return Q.T


def compare():
    files = sorted(OUTDIR.glob("precision_*.json"))
    if not files:
        print("No precision_*.json files yet. Run with --precision 4bit then 8bit.")
        return
    rows = [json.load(open(f)) for f in files]
    base = next((r for r in rows if r["precision"] == "4bit"), rows[0])
    print("\n" + "=" * 74)
    print(f"PRECISION ROBUSTNESS (baseline = {base['precision']})")
    print("=" * 74)
    print(f"{'precision':10} {'probe@16':>10} {'ablate AUC':>11} {'rand AUC':>9} {'rank4 AUC':>10}")
    print("-" * 74)
    for r in rows:
        print(f"{r['precision']:10} {r['probe_auc_L16']:>10.3f} {r['necessity']['ablate_auc']:>11.3f} "
              f"{r['necessity']['random_auc']:>9.3f} {r['low_rank']['rank4_auc']:>10.3f}")
    print("-" * 74)
    for r in rows:
        if r["precision"] == base["precision"]:
            continue
        dp = r["probe_auc_L16"] - base["probe_auc_L16"]
        da = r["necessity"]["ablate_auc"] - base["necessity"]["ablate_auc"]
        dr = r["low_rank"]["rank4_auc"] - base["low_rank"]["rank4_auc"]
        print(f"delta {r['precision']} vs {base['precision']}: probe {dp:+.3f}  ablate {da:+.3f}  rank4 {dr:+.3f}")
    print("=" * 74)


def main():
    args = parse_args()
    if args.compare:
        compare()
        return

    rng = np.random.default_rng(args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui_tr, sui_te = sui[:args.n_train], sui[args.n_train:args.n_train + args.n_test]
    non_tr, non_te = non[:args.n_train], non[args.n_train:args.n_train + args.n_test]

    print(f"=== precision robustness :: {args.precision} :: {args.model} ===")
    try:
        model, tok = load_model_at(args.model, args.precision)
    except torch.cuda.OutOfMemoryError:
        print(f"OOM loading {args.model} at {args.precision}. An 8B model at fp16/bf16 needs "
              f"~16 GB VRAM; run this precision on a larger GPU or pass a smaller --model.")
        return
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers
    nL = len(layers)

    def chat_ids(text):
        prefix, suffix = PROMPT_TEMPLATES["standard"].split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        if tok.chat_template is not None:
            s = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return tok(content, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def all_hidden(text):
        out = model(chat_ids(text), output_hidden_states=True)
        return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])

    @torch.no_grad()
    def logit_diff(text):
        lg = model(chat_ids(text)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def auc_over(fn):
        s = [fn(t) for t in sui_te]; n = [fn(t) for t in non_te]
        y = np.array([1] * len(s) + [0] * len(n))
        return roc_auc_score(y, np.array(s + n))

    print("collecting train activations...")
    Hs = np.stack([all_hidden(t) for t in sui_tr])   # [n, nL+1, d]
    Hn = np.stack([all_hidden(t) for t in non_tr])

    # ---- R1 decodability: probe AUC at layer 16 (5-fold on train pool) ----
    X = np.concatenate([Hs[:, LAYER, :], Hn[:, LAYER, :]])
    y = np.array([1] * len(Hs) + [0] * len(Hn))
    proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                              X, y, cv=cv, method="predict_proba")[:, 1]
    probe_auc = float(roc_auc_score(y, proba))
    print(f"R1 probe AUC @L{LAYER}: {probe_auc:.3f}")

    # ---- directions for ablation ----
    dirs = Hs.mean(0) - Hn.mean(0)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    rand = rng.standard_normal(dirs.shape)
    rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
    Rdir = {i: torch.tensor(dirs[i + 1], device=device) for i in range(nL)}
    Rrnd = {i: torch.tensor(rand[i + 1], device=device) for i in range(nL)}

    def proj_hook(vec):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = vec.to(h.dtype)
            h2 = h - (h @ v).unsqueeze(-1) * v
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def subspace_hook(R):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            Rt = R.to(h.dtype)
            h2 = h - (h @ Rt) @ Rt.T
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def add_hooks(factory, arg):
        return [layers[i].register_forward_hook(factory(arg[i])) for i in range(nL)]

    # ---- R2 necessity ----
    base_auc = auc_over(logit_diff)
    h_ = add_hooks(proj_hook, Rdir); abl_auc = auc_over(logit_diff); [x.remove() for x in h_]
    h_ = add_hooks(proj_hook, Rrnd); rnd_auc = auc_over(logit_diff); [x.remove() for x in h_]
    print(f"R2 necessity: base {base_auc:.3f} -> ablate {abl_auc:.3f} (random {rnd_auc:.3f})")

    # ---- R3 low-rank (rank-4) ----
    subs = np.stack([deflate_subspace(Hs[:, L, :], Hn[:, L, :], RANK) for L in range(nL + 1)])  # [nL+1,k,d]
    Rk = {i: torch.tensor(subs[i + 1].T, device=device) for i in range(nL)}                     # [d,k]
    rsub = np.stack([np.linalg.qr(rng.standard_normal((dirs.shape[1], RANK)))[0] for _ in range(nL)])
    Rkr = {i: torch.tensor(rsub[i], device=device) for i in range(nL)}
    h_ = add_hooks(subspace_hook, Rk); rank4_auc = auc_over(logit_diff); [x.remove() for x in h_]
    h_ = add_hooks(subspace_hook, Rkr); rank4_rnd = auc_over(logit_diff); [x.remove() for x in h_]
    print(f"R3 low-rank (k={RANK}): AUC {rank4_auc:.3f} (random {rank4_rnd:.3f})")

    report = {
        "precision": args.precision, "model": args.model,
        "n_train": args.n_train, "n_test": args.n_test, "layer": LAYER, "rank": RANK,
        "probe_auc_L16": probe_auc,
        "necessity": {"baseline_auc": float(base_auc), "ablate_auc": float(abl_auc),
                      "random_auc": float(rnd_auc)},
        "low_rank": {"rank4_auc": float(rank4_auc), "rank4_random_auc": float(rank4_rnd)},
    }
    ensure_dir(OUTDIR)
    out = OUTDIR / f"precision_{args.precision}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"Saved {out}")
    print("Run another --precision and then --compare to print deltas.")


if __name__ == "__main__":
    main()
