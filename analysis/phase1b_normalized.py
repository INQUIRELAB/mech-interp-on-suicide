#!/usr/bin/env python
"""Phase 1b: Normalized IE (Liu et al. 2025 -comparable).

Phase 1 reports raw IE values with a near-zero denominator threshold
(1e-8), which lets degenerate pairs where the clean/corrupt LDs nearly
coincide produce IE magnitudes of 10-50+. Liu et al. (ACM TOIS 2025,
"How Do LLMs Understand Relevance?") use the same formula

    IE = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)

but operate in a regime where |LD_clean - LD_corrupt| is large by
construction (positive vs. negative documents), so the ratio naturally
stays in [-1, 1]. To make our suicide-detection numbers directly
comparable, this phase:

  1. Filters out pairs whose contrast |LD_clean - LD_corrupt| is below
     MIN_CONTRAST (default 1.0 logit, roughly the scale at which a
     patched component can meaningfully "restore" clean behavior).
  2. Clips per-case IE to [-1.5, 1.5] so a single near-zero-denom pair
     that slips through cannot dominate the mean.
  3. Reports both raw and clipped means, plus the fraction of pairs
     kept, so you can see how sensitive the aggregate is.
  4. Produces Liu-comparable per-layer curves (bounded to [-1, 1]-ish)
     and a side-by-side plot of raw vs normalized IE.

Outputs:
    results/phase1b/{dataset}/{model}_attn_ie_norm.pt    # [n_layers]
    results/phase1b/{dataset}/{model}_mlp_ie_norm.pt     # [n_layers]
    results/phase1b/{dataset}/{model}_per_case.pt        # [n_kept, n_layers]
    results/phase1b/{dataset}/phase1b_normalized_{model}.png
    results/phase1b/{dataset}/phase1b_raw_vs_norm_{model}.png
"""

import sys
import gc
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair,
    run_activation_patching, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 100
MAX_TOKENS = 256
MIN_CONTRAST = 1.0        # |LD_clean - LD_corrupt| floor (logit units)
CLIP_RANGE = 1.5          # clip per-case IE to [-CLIP_RANGE, +CLIP_RANGE]
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1b"
MODELS = ["llama"]


def analyze_model(model_name, death_me, life_not_me, device, out_dir, dataset_label):
    print(f"\n[{model_name}] Loading model...")
    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    raw_attn = torch.zeros(n_pairs, n_layers)
    raw_mlp = torch.zeros(n_pairs, n_layers)
    contrasts = torch.zeros(n_pairs)
    kept_mask = torch.zeros(n_pairs, dtype=torch.bool)

    print(f"[{model_name}] Running {n_pairs} cases; "
          f"min contrast = {MIN_CONTRAST}, clip = +/- {CLIP_RANGE}")
    for i in range(n_pairs):
        clean_text = get_text(death_me[i])
        corrupt_text = get_text(life_not_me[i])
        clean_prompt = build_prompt(clean_text, style="standard")
        corrupt_prompt = build_prompt(corrupt_text, style="standard")
        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer, clean_prompt, corrupt_prompt,
            max_length=MAX_TOKENS, device=device,
        )
        attn_ie, mlp_ie, ld_clean, ld_corrupt = run_activation_patching(
            model, clean_tokens, corrupt_tokens, yes_id, no_id,
        )
        contrast = abs(float(ld_clean) - float(ld_corrupt))
        contrasts[i] = contrast
        raw_attn[i] = attn_ie
        raw_mlp[i] = mlp_ie
        keep = contrast >= MIN_CONTRAST
        kept_mask[i] = keep
        flag = "KEEP" if keep else "DROP"
        print(f"  case {i+1:2d}: LDc={float(ld_clean):+.2f} LDx={float(ld_corrupt):+.2f} "
              f"contrast={contrast:.2f} [{flag}]  "
              f"peakAttn={attn_ie.abs().max():.2f}  peakMLP={mlp_ie.abs().max():.2f}")

    n_kept = int(kept_mask.sum())
    if n_kept == 0:
        print(f"[{model_name}] WARNING: no pair passed contrast threshold {MIN_CONTRAST}. "
              f"Lower MIN_CONTRAST or pick a different category pairing.")
        unload_model(model)
        return

    # Clip per-case IE and average over kept cases
    kept_attn = raw_attn[kept_mask].clamp(-CLIP_RANGE, CLIP_RANGE)
    kept_mlp = raw_mlp[kept_mask].clamp(-CLIP_RANGE, CLIP_RANGE)
    norm_attn = kept_attn.mean(dim=0)
    norm_mlp = kept_mlp.mean(dim=0)

    # Also compute the raw (unfiltered, unclipped) means for side-by-side
    raw_attn_mean = raw_attn.mean(dim=0)
    raw_mlp_mean = raw_mlp.mean(dim=0)

    torch.save(norm_attn, out_dir / f"{model_name}_attn_ie_norm.pt")
    torch.save(norm_mlp, out_dir / f"{model_name}_mlp_ie_norm.pt")
    torch.save({
        "raw_attn": raw_attn, "raw_mlp": raw_mlp,
        "contrasts": contrasts, "kept_mask": kept_mask,
        "min_contrast": MIN_CONTRAST, "clip_range": CLIP_RANGE,
    }, out_dir / f"{model_name}_per_case.pt")

    # --- Summary ---
    def stage(x, lo, hi):
        return float(x[lo:hi].mean())
    print(f"\n[{model_name}] NORMALIZED SUMMARY ({n_kept}/{n_pairs} pairs kept)")
    print(f"  Three-stage (Attention):  early(L0-9)={stage(norm_attn,0,10):+.3f}  "
          f"mid(L10-19)={stage(norm_attn,10,20):+.3f}  late(L20+)={stage(norm_attn,20,n_layers):+.3f}")
    print(f"  Three-stage (MLP):        early(L0-9)={stage(norm_mlp,0,10):+.3f}  "
          f"mid(L10-19)={stage(norm_mlp,10,20):+.3f}  late(L20+)={stage(norm_mlp,20,n_layers):+.3f}")
    top_a = norm_attn.topk(5)
    top_m = norm_mlp.topk(5)
    print(f"  Top 5 Attn: " + ", ".join(
        f"L{i.item()}={v.item():+.3f}" for v, i in zip(top_a.values, top_a.indices)))
    print(f"  Top 5 MLP : " + ", ".join(
        f"L{i.item()}={v.item():+.3f}" for v, i in zip(top_m.values, top_m.indices)))

    # --- Plot: normalized curves ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = range(n_layers)
    ax.plot(layers, norm_attn.numpy(), "o-", color="#e74c3c",
            label=f"Attention (n={n_kept})", markersize=4, lw=1.8)
    ax.plot(layers, norm_mlp.numpy(), "s-", color="#2980b9",
            label=f"MLP (n={n_kept})", markersize=4, lw=1.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized IE  (clipped per case, then averaged)")
    ax.set_title(f"Phase 1b: Normalized IE ({model_name.capitalize()}, {dataset_label})\n"
                 f"contrast >= {MIN_CONTRAST}, per-case clip [±{CLIP_RANGE}]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = out_dir / f"phase1b_normalized_{model_name}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Plot: raw vs normalized ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(layers, raw_attn_mean.numpy(), "o-", color="#e74c3c",
                 label="Attention", markersize=3)
    axes[0].plot(layers, raw_mlp_mean.numpy(), "s-", color="#2980b9",
                 label="MLP", markersize=3)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"Raw IE  (n={n_pairs}, no filter, no clip)\n"
                      f"— affected by degenerate pairs")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Raw IE")
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(layers, norm_attn.numpy(), "o-", color="#e74c3c",
                 label="Attention", markersize=3)
    axes[1].plot(layers, norm_mlp.numpy(), "s-", color="#2980b9",
                 label="MLP", markersize=3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title(f"Normalized IE  (n={n_kept}, contrast>={MIN_CONTRAST}, "
                      f"clip ±{CLIP_RANGE})\n— Liu et al. 2025 -comparable")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Normalized IE")
    axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].set_ylim(-1.2, 1.2)
    fig.suptitle(f"Phase 1b: Raw vs Normalized IE ({model_name.capitalize()}, {dataset_label})",
                 fontsize=12)
    fig.tight_layout()
    out2 = out_dir / f"phase1b_raw_vs_norm_{model_name}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    unload_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    global MIN_CONTRAST, CLIP_RANGE
    parser = argparse.ArgumentParser(description="Phase 1b: Normalized IE")
    add_dataset_arg(parser)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--min-contrast", type=float, default=MIN_CONTRAST,
                        help="Skip pairs with |LD_clean - LD_corrupt| below this.")
    parser.add_argument("--clip", type=float, default=CLIP_RANGE,
                        help="Clip per-case IE to [-clip, +clip] before averaging.")
    args = parser.parse_args()

    # Allow CLI override
    MIN_CONTRAST = args.min_contrast
    CLIP_RANGE = args.clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = load_dataset_by_category(dataset_name=args.dataset)
    death_me = cats.get("Death-Me", [])
    life_not_me = cats.get("Life-Not-Me", [])
    if len(death_me) < 2 or len(life_not_me) < 2:
        raise RuntimeError(
            f"Not enough examples: DM={len(death_me)}, LNM={len(life_not_me)}")

    dataset_label = DATASETS[args.dataset]["display_name"]
    out_dir = get_output_dir(BASE_OUTPUT_DIR, args.dataset)
    ensure_dir(out_dir)

    for m in args.models:
        analyze_model(m, death_me, life_not_me, device, out_dir, dataset_label)


if __name__ == "__main__":
    main()
