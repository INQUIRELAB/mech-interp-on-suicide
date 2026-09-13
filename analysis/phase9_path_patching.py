#!/usr/bin/env python
"""Phase 9: Path Patching / Edge Attribution.

Goes beyond Phase 1 (residual-stream patching) by attributing causal influence
to individual nodes (attn heads, MLPs) via gradient-based edge attribution
patching (EAP). Produces:

  (a) Node attribution heatmap (layer x head) and MLP bars — the *signed*
      contribution of each component to the logit difference.
  (b) A circuit diagram showing the top-K nodes ordered by layer, connected
      in residual-stream flow. This is the "clean circuit figure" reviewers
      expect in mechanistic interpretability papers.
  (c) Sanity check: cumulative IE vs. number of nodes retained.

EAP formula (one forward + one backward pass):
    attribution(c) = grad[logit_diff, act_c]  ·  (act_c_clean - act_c_corrupt)

Fast: O(1) passes, not O(n_layers * n_heads).

Output:
    results/phase9/{dataset}/{model}_eap_attn.pt    - [n_layers, n_heads]
    results/phase9/{dataset}/{model}_eap_mlp.pt     - [n_layers]
    results/phase9/{dataset}/phase9_attribution_{model}.png
    results/phase9/{dataset}/phase9_circuit_{model}.png
    results/phase9/{dataset}/phase9_cumulative_{model}.png
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 100
MAX_TOKENS = 256
TOP_K_CIRCUIT = 12
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase9"
MODELS = ["llama"]  # extend to ["llama", "qwen", "mistral"] on A100


# ---------------------------------------------------------------------------
# Edge attribution patching via gradients
# ---------------------------------------------------------------------------

def run_eap(model, tokenizer, clean_tokens, corrupt_tokens, yes_id, no_id):
    """Compute node-level EAP scores for every attention head and MLP.

    TL-backed implementation: uses HookedTransformer hook points
    `blocks.{L}.hook_attn_out` and `blocks.{L}.hook_mlp_out` to capture
    activations with retained gradients, then backprops through the logit
    difference to get per-node attribution.

    Returns:
        attn_attr: [n_layers, n_heads]  signed contribution to logit_diff
        mlp_attr:  [n_layers]
        ld_clean, ld_corrupt: scalars
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    tl_model = model.model  # HookedTransformer
    tl_model.eval()

    # --- 1) Cache clean activations (no grad) ---
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        ld_clean = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).float().item()

    # --- 2) Corrupt forward WITH gradients on attn_out / mlp_out hook points ---
    corrupt_attn_outs = {}  # layer -> tensor [B,T,D]
    corrupt_mlp_outs = {}

    def make_save(store, L):
        def hook(act, hook):
            act.requires_grad_(True)
            act.retain_grad()
            store[L] = act
            return act
        return hook

    # Freeze params (we only need activation gradients, not weight gradients)
    for p in tl_model.parameters():
        p.requires_grad_(False)

    fwd_hooks = []
    for L in range(n_layers):
        fwd_hooks.append((f"blocks.{L}.hook_attn_out", make_save(corrupt_attn_outs, L)))
        fwd_hooks.append((f"blocks.{L}.hook_mlp_out",  make_save(corrupt_mlp_outs,  L)))

    tl_model.reset_hooks()
    with torch.enable_grad():
        corrupt_logits = tl_model.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks)
        ld_corrupt = (corrupt_logits[0, -1, yes_id] - corrupt_logits[0, -1, no_id])
        ld_corrupt.backward()

    ld_corrupt_val = ld_corrupt.detach().float().item()
    tl_model.reset_hooks()

    # --- 3) Compute attribution = (clean - corrupt) . grad ---
    attn_attr = torch.zeros(n_layers, n_heads)
    mlp_attr = torch.zeros(n_layers)

    for L in range(n_layers):
        if L not in corrupt_attn_outs:
            continue
        c_attn = corrupt_attn_outs[L].detach()       # [1, T, D]
        g_attn = corrupt_attn_outs[L].grad           # [1, T, D]
        if g_attn is None:
            continue
        clean_attn = clean_cache[f"blocks.{L}.hook_attn_out"].to(c_attn.device)
        diff = (clean_attn - c_attn)  # [1, T, D]
        # Per-head split along last dim (assumes heads are concatenated)
        # Attribution at last token position (the answer position)
        diff_last = diff[0, -1, :]
        grad_last = g_attn[0, -1, :]
        prod = (diff_last * grad_last).reshape(n_heads, d_head).sum(dim=-1)
        attn_attr[L] = prod.detach().float().cpu()

        if L in corrupt_mlp_outs and corrupt_mlp_outs[L].grad is not None:
            c_mlp = corrupt_mlp_outs[L].detach()
            g_mlp = corrupt_mlp_outs[L].grad
            clean_mlp = clean_cache[f"blocks.{L}.hook_mlp_out"].to(c_mlp.device)
            mdiff = (clean_mlp - c_mlp)[0, -1, :]
            mgrad = g_mlp[0, -1, :]
            mlp_attr[L] = (mdiff * mgrad).sum().detach().float().cpu()

    return attn_attr, mlp_attr, ld_clean, ld_corrupt_val


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_node_attribution(attn_attr, mlp_attr, title, save_path):
    n_layers, n_heads = attn_attr.shape
    fig = plt.figure(figsize=(max(10, n_heads * 0.55), max(6, n_layers * 0.28) + 2))
    gs = fig.add_gridspec(1, 2, width_ratios=[n_heads, 1.5], wspace=0.25)

    # Heads heatmap (diverging)
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = float(attn_attr.abs().max()) or 1.0
    im = ax1.imshow(attn_attr.numpy(), cmap="RdBu_r", aspect="auto",
                    vmin=-vmax, vmax=vmax)
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_title("Attention head attribution")
    ax1.set_xticks(range(n_heads))
    ax1.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    fig.colorbar(im, ax=ax1, label="signed contribution to Δlogit")

    # MLP bars
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in mlp_attr]
    ax2.barh(range(n_layers), mlp_attr.numpy(), color=colors)
    ax2.set_xlabel("MLP attribution")
    ax2.set_ylabel("Layer")
    ax2.invert_yaxis()
    ax2.set_title("MLP attribution")
    ax2.axvline(0, color="k", lw=0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_circuit_diagram(attn_attr, mlp_attr, top_k, title, save_path):
    """Render the top-K components as a layered circuit graph.

    Layout: nodes sorted by layer (y-axis top=early → bottom=late).
    Color = sign (red=boosts YES, blue=suppresses).
    Size = |attribution|.
    Arrows = residual flow (every upstream node connects to every downstream one
    with opacity proportional to the product of their magnitudes).
    """
    n_layers, n_heads = attn_attr.shape
    # Collect all nodes with attribution scores
    nodes = []
    for L in range(n_layers):
        for H in range(n_heads):
            nodes.append(("A", L, H, float(attn_attr[L, H])))
        nodes.append(("M", L, None, float(mlp_attr[L])))
    # Rank by absolute attribution
    nodes.sort(key=lambda t: -abs(t[3]))
    top = nodes[:top_k]
    # Sort top by layer ascending for display (and head for tie)
    top.sort(key=lambda t: (t[1], t[2] if t[2] is not None else -1))

    fig, ax = plt.subplots(figsize=(9, max(6, 0.55 * len(top) + 2)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(top) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    max_abs = max(abs(t[3]) for t in top) or 1.0

    # Draw nodes
    positions = []
    for i, (kind, L, H, v) in enumerate(top):
        x = 3.0 + (1.0 if kind == "M" else 0.0)
        y = i
        positions.append((x, y, kind, L, H, v))
        color = "#c0392b" if v > 0 else "#2980b9"
        alpha = 0.4 + 0.6 * (abs(v) / max_abs)
        label = f"L{L} MLP" if kind == "M" else f"L{L} H{H}"
        box = FancyBboxPatch((x - 1.2, y - 0.35), 2.4, 0.7,
                             boxstyle="round,pad=0.05",
                             linewidth=1.5,
                             edgecolor="black",
                             facecolor=color, alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=10, weight="bold")
        # Attribution score next to box
        ax.text(x + 1.6, y, f"{v:+.3f}", ha="left", va="center",
                fontsize=9, color="#333333")

    # Draw residual-flow arrows (every node feeds downstream)
    for i, (xi, yi, *_ ) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            xj, yj, *_ = positions[j]
            # Attribution product for weighting
            w = abs(positions[i][5]) * abs(positions[j][5]) / (max_abs * max_abs)
            if w < 0.05:
                continue
            arr = FancyArrowPatch((xi, yi + 0.35), (xj, yj - 0.35),
                                  arrowstyle="->",
                                  connectionstyle="arc3,rad=0.15",
                                  color="gray", alpha=min(0.6, 0.2 + w),
                                  linewidth=0.8 + w * 2)
            ax.add_patch(arr)

    ax.set_title(title, fontsize=12, pad=10)
    # Legend
    ax.text(7.5, -0.4, "red = boosts YES", color="#c0392b", fontsize=10)
    ax.text(7.5, -0.1, "blue = suppresses YES", color="#2980b9", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_cumulative(attn_attr, mlp_attr, title, save_path):
    """Cumulative |attribution| vs number of top nodes — shows circuit compactness."""
    flat = []
    n_layers, n_heads = attn_attr.shape
    for L in range(n_layers):
        for H in range(n_heads):
            flat.append(abs(float(attn_attr[L, H])))
        flat.append(abs(float(mlp_attr[L])))
    flat.sort(reverse=True)
    arr = np.array(flat)
    cum = np.cumsum(arr) / arr.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cum) + 1), cum, color="#8e44ad", lw=2)
    for frac in (0.5, 0.8, 0.9):
        k = int(np.searchsorted(cum, frac)) + 1
        ax.axhline(frac, color="gray", ls="--", lw=0.6)
        ax.axvline(k, color="gray", ls="--", lw=0.6)
        ax.text(k + 1, frac - 0.02, f"{int(frac*100)}% → {k} nodes", fontsize=9)
    ax.set_xlabel("Number of top components retained")
    ax.set_ylabel("Cumulative fraction of |attribution|")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_model(model_name, death_me, life_not_me, device, out_dir, dataset_label):
    print(f"\n[{model_name}] Loading model (bf16)...")
    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))
    print(f"[{model_name}] {n_layers}L x {n_heads}H; running EAP on {n_pairs} pairs")

    attn_sum = torch.zeros(n_layers, n_heads)
    mlp_sum = torch.zeros(n_layers)

    for i in range(n_pairs):
        clean_text = get_text(death_me[i])
        corrupt_text = get_text(life_not_me[i])
        ct, xt = tokenize_pair(
            tokenizer, build_prompt(clean_text), build_prompt(corrupt_text),
            max_length=MAX_TOKENS, device=device,
        )
        attn_attr, mlp_attr, ldc, ldx = run_eap(model, tokenizer, ct, xt, yes_id, no_id)
        attn_sum += attn_attr
        mlp_sum += mlp_attr
        top_l, top_h = np.unravel_index(attn_attr.abs().argmax().item(), attn_attr.shape)
        print(f"  case {i+1}/{n_pairs}: LDc={ldc:+.2f} LDx={ldx:+.2f}  "
              f"top head L{top_l}H{top_h}={attn_attr[top_l, top_h]:+.3f}")

    attn_mean = attn_sum / n_pairs
    mlp_mean = mlp_sum / n_pairs

    torch.save(attn_mean, out_dir / f"{model_name}_eap_attn.pt")
    torch.save(mlp_mean, out_dir / f"{model_name}_eap_mlp.pt")

    plot_node_attribution(
        attn_mean, mlp_mean,
        title=f"Phase 9: Node Attribution ({model_name.capitalize()}, {dataset_label})\n"
              f"EAP signed contribution to Yes-No logit diff (avg over {n_pairs} pairs)",
        save_path=out_dir / f"phase9_attribution_{model_name}.png",
    )
    plot_circuit_diagram(
        attn_mean, mlp_mean, TOP_K_CIRCUIT,
        title=f"Phase 9: Top-{TOP_K_CIRCUIT} Circuit ({model_name.capitalize()}, {dataset_label})",
        save_path=out_dir / f"phase9_circuit_{model_name}.png",
    )
    plot_cumulative(
        attn_mean, mlp_mean,
        title=f"Phase 9: Circuit compactness ({model_name.capitalize()}, {dataset_label})",
        save_path=out_dir / f"phase9_cumulative_{model_name}.png",
    )

    unload_model(model)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = load_dataset_by_category(dataset_name=args.dataset)
    death_me = cats.get("Death-Me", [])
    life_not_me = cats.get("Life-Not-Me", [])
    if len(death_me) < 2 or len(life_not_me) < 2:
        raise RuntimeError(f"Not enough examples: DM={len(death_me)}, LNM={len(life_not_me)}")

    dataset_label = DATASETS[args.dataset]["display_name"]
    out_dir = get_output_dir(BASE_OUTPUT_DIR, args.dataset)
    ensure_dir(out_dir)

    for m in args.models:
        analyze_model(m, death_me, life_not_me, device, out_dir, dataset_label)


if __name__ == "__main__":
    main()
