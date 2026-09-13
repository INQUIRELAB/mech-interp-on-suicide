#!/usr/bin/env python
"""Q1-grade statistics for the per-head knockout (binary 232K).

The full 32x32 head sweep (sc2_head_knockout.py -> head_knockout.json) gives a
point AUC drop per head but no uncertainty and no control -- below Q1 standard.
This adds, without recomputing the full sweep:

  1. An empirical NULL distribution: the AUC drops of all 1024 heads (most heads
     do nothing, so this is the "knock out an arbitrary head" control).
  2. BOOTSTRAP 95% CIs on the AUC drop of the top-K heads (resample the held-out
     examples B times; same n and seed as the sweep so examples match).
  3. A PERMUTATION / empirical p-value for each top head = fraction of the null
     >= its observed drop, then BENJAMINI-HOCHBERG FDR correction over K tests.

Output: results/sc2_binary/head_knockout_stats.json
        results/sc2_binary/head_knockout_forest.png   (forest plot, CIs vs null band)

Run:  python analysis/sanity/sc2_head_knockout_stats.py --top-k 15
Needs the same model/data as sc2_head_knockout.py; recomputes only the top-K heads.
"""
import sys, json, argparse
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
from sklearn.metrics import roc_auc_score
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap CIs + FDR for head knockout")
    p.add_argument("--n", type=int, default=20, help="examples/class (match the sweep; default 20)")
    p.add_argument("--top-k", type=int, default=15, help="top heads to characterize (default 15)")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=get_seed(), help="MUST match the sweep seed for example alignment")
    return p.parse_args()


def boot_ci(scores_s, scores_n, base_auc, B, rng):
    """Bootstrap 95% CI of (base_auc - knockout_auc) by resampling examples."""
    s = np.asarray(scores_s); n = np.asarray(scores_n)
    ns, nn = len(s), len(n)
    drops = np.empty(B)
    for b in range(B):
        bs = s[rng.integers(0, ns, ns)]; bn = n[rng.integers(0, nn, nn)]
        y = np.r_[np.ones(ns), np.zeros(nn)]
        drops[b] = base_auc - roc_auc_score(y, np.r_[bs, bn])
    return float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    adj = np.empty(m); prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        prev = min(prev, p[idx] * m / i)
        adj[idx] = prev
    return adj.tolist()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)             # for example sampling (match sweep)
    brng = np.random.default_rng(args.seed + 1000)     # for bootstrap

    sweep_path = PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout.json"
    if not sweep_path.exists():
        sys.exit("Run sc2_head_knockout.py first (need head_knockout.json).")
    sweep = json.load(open(sweep_path))
    all_drops = np.array(sweep["auc_drop"]).flatten()
    null_drops = all_drops.copy()                       # empirical null over all heads
    top_heads = sweep["top_heads"][: args.top_k]

    # ---- rebuild the SAME example set as the sweep (seed-aligned) ----
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[: args.n], non[: args.n]

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    ids_sui = [chat_ids(t) for t in sui]
    ids_non = [chat_ids(t) for t in non]

    @torch.no_grad()
    def ld(ids):
        lg = model(ids).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def scores(prehook=None, layer=None):
        h = None
        if prehook is not None:
            h = layers[layer].self_attn.o_proj.register_forward_pre_hook(prehook)
        try:
            s = [ld(i) for i in ids_sui]; n = [ld(i) for i in ids_non]
        finally:
            if h: h.remove()
        return s, n

    def make_prehook(head):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        def prehook(module, inp):
            x = inp[0].clone(); x[..., sl] = 0
            return (x,) + inp[1:]
        return prehook

    # ---- baseline (per-example) ----
    base_s, base_n = scores()
    y = np.r_[np.ones(len(base_s)), np.zeros(len(base_n))]
    base_auc = roc_auc_score(y, np.r_[base_s, base_n])
    print(f"baseline AUC={base_auc:.3f} (n={args.n}/class)")

    # ---- per top head: observed drop, bootstrap CI, empirical p ----
    results = []
    pvals = []
    for t in top_heads:
        L, hh = t["layer"], t["head"]
        s, n = scores(make_prehook(hh), L)
        auc = roc_auc_score(y, np.r_[s, n])
        drop = base_auc - auc
        lo, hi = boot_ci(s, n, base_auc, args.boot, brng)
        p = float((null_drops >= drop).mean())         # empirical, vs all-head null
        pvals.append(p)
        results.append({"layer": L, "head": hh, "auc_drop": float(drop),
                        "ci95": [lo, hi], "p_emp": p})
        print(f"  L{L:2d}H{hh:2d}: drop={drop:+.3f}  CI95=[{lo:+.3f},{hi:+.3f}]  p={p:.3f}")

    qvals = bh_fdr(pvals)
    for r, q in zip(results, qvals):
        r["q_fdr"] = float(q)
        r["sig_fdr_0.05"] = bool(q < 0.05 and r["ci95"][0] > 0)

    n_sig = sum(r["sig_fdr_0.05"] for r in results)
    null_mean, null_sd = float(null_drops.mean()), float(null_drops.std())
    print(f"\nnull (all 1024 heads): mean={null_mean:+.4f} sd={null_sd:.4f}")
    print(f"top-{args.top_k} heads significant after FDR & CI>0: {n_sig}")

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    out = {
        "dataset": "07_Suicide_Watch_232K", "n_per_class": args.n,
        "baseline_auc": float(base_auc), "bootstrap_resamples": args.boot,
        "null_all_heads": {"mean": null_mean, "sd": null_sd,
                           "p95": float(np.percentile(null_drops, 95)),
                           "p99": float(np.percentile(null_drops, 99))},
        "top_heads": results, "n_significant_fdr": n_sig,
        "method": ("Per-head o_proj-slice knockout; bootstrap 95pct CI over examples "
                   "(B={}); empirical p vs all-head null; Benjamini-Hochberg FDR.".format(args.boot)),
    }
    json.dump(out, open(PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout_stats.json", "w"), indent=2)

    # ---- forest plot ----
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(results))))
    ys = np.arange(len(results))[::-1]
    labels_ = [f"L{r['layer']}H{r['head']}" + ("*" if r["sig_fdr_0.05"] else "") for r in results]
    drops = [r["auc_drop"] for r in results]
    errs = [[r["auc_drop"] - r["ci95"][0] for r in results],
            [r["ci95"][1] - r["auc_drop"] for r in results]]
    ax.errorbar(drops, ys, xerr=errs, fmt="o", color="C3", capsize=3)
    ax.axvspan(0, np.percentile(null_drops, 95), color="grey", alpha=0.15,
               label="null 0–95th pct (any head)")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels(labels_, fontsize=8)
    ax.set_xlabel("AUC drop (baseline − knockout), 95% CI")
    ax.set_title("Top heads vs all-head null (* = FDR<0.05 & CI>0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout_forest.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved head_knockout_stats.json + head_knockout_forest.png")


if __name__ == "__main__":
    main()
