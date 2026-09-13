#!/usr/bin/env python
"""Phase 10: Attention patterns + Logit Lens.

Highly visual, easy-to-read phase aimed at showcasing mechanism to
non-MI readers.

Part A — Attention patterns of critical heads
  For the top-K heads identified by Phase 5 (or EAP in Phase 9), plot
  the attention pattern the head places on input tokens from the final
  (answer) position. Tokens are highlighted by attention weight.

Part B — Logit lens
  Project the residual stream at every layer through the final LM head
  (after RMSNorm) and compute P(yes) - P(no) at every token position.
  Produces a heatmap [n_layers × n_tokens] showing where the decision
  crystallizes inside the network.

Output:
    results/phase10/{dataset}/phase10_attention_{model}.png
    results/phase10/{dataset}/phase10_logitlens_{model}.png
    results/phase10/{dataset}/phase10_logitlens_curve_{model}.png
"""

import sys
import gc
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_HEADS_TO_SHOW = 4
N_POSTS_PER_CATEGORY = 2
MAX_TOKENS = 192
MAX_TOKENS_TO_SHOW = 60  # display only the last N tokens for readability
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase10"
MODELS = ["llama"]


# ---------------------------------------------------------------------------
# Top-head selection
# ---------------------------------------------------------------------------

def load_top_heads(model_name, dataset_name, k=N_HEADS_TO_SHOW):
    """Load top-K heads from Phase 5 results. Returns list of (layer, head, score)."""
    candidates = [
        PROJECT_ROOT / "results" / "phase5" / dataset_name / f"{model_name}_head_importance.pt",
        PROJECT_ROOT / "results" / "phase5" / f"{model_name}_head_importance.pt",
        PROJECT_ROOT / "results" / "phase9" / dataset_name / f"{model_name}_eap_attn.pt",
    ]
    for path in candidates:
        if path.exists():
            data = torch.load(path, weights_only=True)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            flat = data.abs().flatten()
            topk = flat.topk(k)
            n_heads = data.shape[1]
            heads = [(idx.item() // n_heads, idx.item() % n_heads,
                      float(data.flatten()[idx]))
                     for idx in topk.indices]
            print(f"  Top-{k} heads from {path.name}: {heads}")
            return heads
    print(f"  WARNING: no Phase 5/9 results found for {model_name}/{dataset_name}. "
          f"Falling back to mid-layer heads.")
    return None


# ---------------------------------------------------------------------------
# Attention pattern extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_attention_patterns(model, tokens):
    """Return per-layer attention patterns via TL's hook_pattern cache key.

    Returns:
        attentions: list of [n_heads, seq, seq] tensors (one per layer)
    """
    tl = model.model  # HookedTransformer
    _, cache = tl.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    return [cache[f"blocks.{L}.attn.hook_pattern"][0].detach().float().cpu()
            for L in range(n_layers)]


def render_attention_over_tokens(token_strs, attn_from_last, ax, title, max_show):
    """Render tokens with background color = attention weight from last position.

    token_strs: list of decoded token strings.
    attn_from_last: 1D tensor over tokens, sums to ~1.
    """
    N = len(token_strs)
    start = max(0, N - max_show)
    toks = token_strs[start:]
    w = attn_from_last[start:].numpy()
    w_norm = w / (w.max() + 1e-8)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(1, (len(toks) + 7) // 8))
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")

    x, y = 0.0, 0.2
    for tok, wt in zip(toks, w_norm):
        display = tok.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")
        if len(display) > 14:
            display = display[:14]
        color_alpha = float(wt)
        ax.text(x, y, display,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=(0.85, 0.15, 0.15, color_alpha),
                          edgecolor="none"),
                family="monospace")
        x += 0.55 + 0.05 * len(display)
        if x > 9.3:
            x = 0.0
            y += 1.0


def plot_attention_panel(model, tokenizer, top_heads, posts, out_path, title):
    """Grid: rows = posts, cols = heads. Token highlights = attention from last pos."""
    n_rows = len(posts)
    n_cols = len(top_heads)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3 * n_rows),
                             squeeze=False)
    for r, (label, text) in enumerate(posts):
        tokens = tokenizer(build_prompt(text), return_tensors="pt",
                           max_length=MAX_TOKENS, truncation=True,
                           padding="max_length")["input_ids"].to(model.device)
        # Decode to list of strings
        token_ids = tokens[0].tolist()
        token_strs = [tokenizer.decode([tid], skip_special_tokens=False)
                      for tid in token_ids]
        atts = get_attention_patterns(model, tokens)
        for c, (L, H, score) in enumerate(top_heads):
            if L >= len(atts):
                continue
            # attention from last token position: [n_heads, seq, seq] -> [seq]
            attn_from_last = atts[L][H, -1, :]
            render_attention_over_tokens(
                token_strs, attn_from_last, axes[r, c],
                title=f"{label}  ·  L{L}H{H}  (|score|={abs(score):.3f})",
                max_show=MAX_TOKENS_TO_SHOW,
            )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_logit_lens(model, tokens, yes_id, no_id):
    """Apply TL's final LayerNorm + unembed to every layer's post-block residual.

    Returns:
        lens: [n_layers, seq_len] logit difference yes-no at every position
    """
    tl = model.model  # HookedTransformer
    _, cache = tl.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    lens = []
    for L in range(n_layers):
        h = cache[f"blocks.{L}.hook_resid_post"]  # [1, seq, d_model]
        h = tl.ln_final(h)
        logits = tl.unembed(h)  # [1, seq, vocab]
        diff = (logits[0, :, yes_id] - logits[0, :, no_id]).float().cpu()
        lens.append(diff)
    return torch.stack(lens)  # [n_layers, seq]


def plot_logit_lens_heatmap(lens, token_strs, out_path, title, max_show=40):
    """Heatmap [n_layers, last-N tokens] of yes-no logit diff."""
    n_layers, seq = lens.shape
    start = max(0, seq - max_show)
    data = lens[:, start:].numpy()
    toks = [t.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")[:10]
            for t in token_strs[start:]]

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(toks)), max(5, 0.28 * n_layers)))
    vmax = float(np.abs(data).max()) or 1.0
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=7, family="monospace")
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.set_ylabel("Layer")
    ax.set_xlabel("Token position (last {} of {} shown)".format(len(toks), seq))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="logit(Yes) − logit(No)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_logit_lens_curves(curves, out_path, title):
    """Line chart: P(yes) at last token vs layer, one curve per example."""
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Death-Me": "#c0392b", "Death-Not-Me": "#f39c12",
               "Life-Not-Me": "#2980b9"}
    for label, ys in curves:
        color = palette.get(label.split(" ")[0], "gray")
        ax.plot(range(len(ys)), ys, "-o", label=label,
                color=color, alpha=0.8, markersize=4, lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("logit(Yes) − logit(No)  at answer position")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_model(model_name, cats, device, out_dir, dataset_label, dataset_name):
    print(f"\n[{model_name}] Loading model...")
    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get top heads
    top_heads = load_top_heads(model_name, dataset_name, k=N_HEADS_TO_SHOW)
    if top_heads is None:
        # Fallback: take mid-layer heads evenly spaced
        top_heads = [(n_layers // 2, h, 0.0) for h in range(0, n_heads,
                                                            max(1, n_heads // N_HEADS_TO_SHOW))][:N_HEADS_TO_SHOW]

    # Pick display posts
    posts = []
    for cat in ("Death-Me", "Death-Not-Me", "Life-Not-Me"):
        if cat not in cats or not cats[cat]:
            continue
        for i in range(min(N_POSTS_PER_CATEGORY, len(cats[cat]))):
            ex = cats[cat][i]
            sev = ex.get("umd_label", cat)
            posts.append((f"{cat} ({sev})", get_text(ex)))

    # --- Part A: Attention ---
    plot_attention_panel(
        model, tokenizer, top_heads,
        posts[:min(len(posts), 4)],
        out_path=out_dir / f"phase10_attention_{model_name}.png",
        title=f"Phase 10A: Attention of critical heads ({model_name.capitalize()}, {dataset_label})",
    )

    # --- Part B: Logit lens ---
    # heatmap for one representative Death-Me post
    rep = cats["Death-Me"][0] if cats.get("Death-Me") else posts[0][1]
    rep_text = get_text(rep) if isinstance(rep, dict) else rep
    tokens = tokenizer(build_prompt(rep_text), return_tensors="pt",
                       max_length=MAX_TOKENS, truncation=True,
                       padding="max_length")["input_ids"].to(device)
    token_strs = [tokenizer.decode([tid]) for tid in tokens[0].tolist()]
    lens = run_logit_lens(model, tokens, yes_id, no_id)
    plot_logit_lens_heatmap(
        lens, token_strs,
        out_path=out_dir / f"phase10_logitlens_{model_name}.png",
        title=f"Phase 10B: Logit Lens ({model_name.capitalize()}, {dataset_label}) — "
              f"where the Yes/No decision forms",
    )

    # curves for multiple examples
    curves = []
    for label, text in posts[:6]:
        t = tokenizer(build_prompt(text), return_tensors="pt",
                      max_length=MAX_TOKENS, truncation=True,
                      padding="max_length")["input_ids"].to(device)
        lns = run_logit_lens(model, t, yes_id, no_id)
        curves.append((label, lns[:, -1].numpy()))
    plot_logit_lens_curves(
        curves,
        out_path=out_dir / f"phase10_logitlens_curve_{model_name}.png",
        title=f"Phase 10B: Decision formation by layer ({model_name.capitalize()}, {dataset_label})",
    )

    unload_model(model)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = load_dataset_by_category(dataset_name=args.dataset)
    dataset_label = DATASETS[args.dataset]["display_name"]
    out_dir = get_output_dir(BASE_OUTPUT_DIR, args.dataset)
    ensure_dir(out_dir)

    for m in args.models:
        analyze_model(m, cats, device, out_dir, dataset_label, args.dataset)


if __name__ == "__main__":
    main()
