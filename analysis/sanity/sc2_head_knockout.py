#!/usr/bin/env python
"""Per-head knockout on the VALIDATED binary task (07_Suicide_Watch_232K).

The original Phase 5 head-knockout ran on DeepSuiMind (Death-Me vs Life-Not-Me) --
a task the model performs at chance, so head "importance" there attributes a
behavior that does not exist. This re-runs the same idea on the binary
suicide-vs-non-suicide task, where the behavior is real (AUC 0.95), using the
exact model/prompt/readout of sc2_faithfulness.py so the result is comparable to
the direction-level causal claims.

Method: for each attention head (layer L, head h), zero that head's contribution
by masking its slice of the o_proj INPUT (the concatenated per-head outputs, dim
n_heads*head_dim) at all positions, then measure how much the behavioral signal
drops. Importance is reported two ways:
  - auc_drop      : baseline ROC-AUC minus knockout ROC-AUC (suicide vs non)
  - ld_gap_drop   : baseline (mean LD_sui - mean LD_non) minus knockout gap

A large, localized auc_drop at a few heads => the circuit routes through specific
heads, not a smear. (Slice masking at o_proj input is GQA-safe: by o_proj the 8
KV heads are already expanded to 32 query heads.)

Output: results/sc2_binary/head_knockout.json  +  head_knockout_heatmap.png

Run:  python analysis/sanity/sc2_head_knockout.py --n 20
      python analysis/sanity/sc2_head_knockout.py --n 25 --layers 8-20   # faster
Cost: ~ n_layers*n_heads*2*n forward passes. n=20 => ~41k fwd (~45-90 min, 12 GB).
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
    p = argparse.ArgumentParser(description="Per-head knockout on binary 232K")
    p.add_argument("--n", type=int, default=20, help="examples per class (default 20)")
    p.add_argument("--layers", type=str, default=None,
                   help="layer range to test, e.g. '8-20' (default: all layers)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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
    n_layers = len(layers)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"{n_layers} layers, {n_heads} heads, head_dim={head_dim}")

    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layer_ids = list(range(lo, hi + 1))
    else:
        layer_ids = list(range(n_layers))

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # precompute token ids once
    ids_sui = [chat_ids(t) for t in sui]
    ids_non = [chat_ids(t) for t in non]

    @torch.no_grad()
    def logit_diff(ids):
        lg = model(ids).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def auc_gap():
        s = [logit_diff(i) for i in ids_sui]
        n = [logit_diff(i) for i in ids_non]
        y = np.array([1] * len(s) + [0] * len(n))
        auc = roc_auc_score(y, np.array(s + n))
        gap = float(np.mean(s) - np.mean(n))
        return auc, gap

    # ---- pre-hook factory: zero one head's slice at o_proj input ----
    def make_prehook(h):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        def prehook(module, inp):
            x = inp[0].clone()
            x[..., sl] = 0
            return (x,) + inp[1:]
        return prehook

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    partial_path = PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout.partial.json"

    # ---- resume from checkpoint if present ----
    auc_drop = np.full((n_layers, n_heads), np.nan)
    gap_drop = np.full((n_layers, n_heads), np.nan)
    done_layers = set()
    base_auc = base_gap = None
    if partial_path.exists():
        ckpt = json.load(open(partial_path))
        base_auc, base_gap = ckpt["baseline_auc"], ckpt["baseline_ld_gap"]
        auc_drop = np.array(ckpt["auc_drop"]); gap_drop = np.array(ckpt["gap_drop"])
        done_layers = set(ckpt["done_layers"])
        print(f"Resuming: {len(done_layers)} layers already done -> {sorted(done_layers)}")

    if base_auc is None:
        print("Baseline (no knockout)...")
        base_auc, base_gap = auc_gap()
    print(f"  baseline AUC={base_auc:.3f}  LD gap={base_gap:.2f}")

    def save_partial():
        with open(partial_path, "w") as f:
            json.dump({"baseline_auc": base_auc, "baseline_ld_gap": base_gap,
                       "auc_drop": np.nan_to_num(auc_drop).tolist(),
                       "gap_drop": np.nan_to_num(gap_drop).tolist(),
                       "done_layers": sorted(done_layers)}, f)

    for L in layer_ids:
        if L in done_layers:
            continue
        oproj = layers[L].self_attn.o_proj
        for h in range(n_heads):
            handle = oproj.register_forward_pre_hook(make_prehook(h))
            try:
                a, g = auc_gap()
            finally:
                handle.remove()
            auc_drop[L, h] = base_auc - a
            gap_drop[L, h] = base_gap - g
        done_layers.add(L)
        save_partial()  # checkpoint after every layer -> power cut costs <=1 layer
        top_h = int(np.nanargmax(auc_drop[L]))
        print(f"  L{L:2d}: max head H{top_h} auc_drop={auc_drop[L, top_h]:+.3f}  [checkpointed]")

    # ---- rank heads ----
    flat = [(L, h, auc_drop[L, h], gap_drop[L, h])
            for L in layer_ids for h in range(n_heads)]
    flat.sort(key=lambda r: r[2], reverse=True)
    top = flat[:20]
    print("\n" + "=" * 60)
    print("TOP 20 HEADS BY AUC DROP (binary 232K)")
    print("=" * 60)
    for L, h, ad, gd in top:
        print(f"  L{L:2d}H{h:2d}  auc_drop={ad:+.3f}  ld_gap_drop={gd:+.2f}")

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    with open(PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout.json", "w") as f:
        json.dump({
            "dataset": "07_Suicide_Watch_232K", "n_per_class": args.n,
            "n_layers": n_layers, "n_heads": n_heads,
            "layers_tested": layer_ids,
            "baseline_auc": base_auc, "baseline_ld_gap": base_gap,
            "auc_drop": np.nan_to_num(auc_drop).tolist(),
            "ld_gap_drop": np.nan_to_num(gap_drop).tolist(),
            "top_heads": [{"layer": L, "head": h, "auc_drop": ad, "ld_gap_drop": gd}
                          for L, h, ad, gd in top],
        }, f, indent=2)

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.35), max(6, len(layer_ids) * 0.3)))
    data = auc_drop[layer_ids, :]
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layer_ids)))
    ax.set_yticklabels([f"L{l}" for l in layer_ids], fontsize=7)
    ax.set_title("Per-head knockout importance (AUC drop)\nbinary suicide detection, 232K")
    fig.colorbar(im, ax=ax, label="baseline AUC - knockout AUC")
    fig.tight_layout()
    fig.savefig(str(PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    if partial_path.exists():
        partial_path.unlink()  # final result written; drop the checkpoint
    print("\nSaved results/sc2_binary/head_knockout.json + head_knockout_heatmap.png")


if __name__ == "__main__":
    main()
