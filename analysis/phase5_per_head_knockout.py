#!/usr/bin/env python
"""Phase 5: Per-Head Knockout Analysis.

Zero-ablates each attention head individually across all layers
to identify which specific heads are critical for suicide detection.

Runs on all three models (Llama, Qwen, Mistral) sequentially.

Output (per model):
    results/phase5/{model}_head_importance.pt        - [n_layers, n_heads] importance
    results/phase5/phase5_heads_{model}.png          - Head importance heatmap
    results/phase5/phase5_top_heads_{model}.png      - Bar chart of top heads
    results/phase5/phase5_comparison.png             - All models overlaid
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
    run_head_knockout, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 15  # Increased from 8 to 15 since per-head is expensive (n_layers * n_heads per case)
MAX_TOKENS = 256
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase5"
MODELS = ["llama", "qwen", "mistral"]


def analyze_single_model(model_name, death_me, life_not_me, device, output_dir, dataset_label):
    """Run per-head knockout for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    print(f"  [{model_name}] Model: {n_layers} layers, {n_heads} heads")
    print(f"  [{model_name}] Total knockouts per case: {n_layers * n_heads}")
    print(f"  [{model_name}] Cases: {n_pairs}")

    all_importance = np.zeros((n_pairs, n_layers, n_heads))

    for i in range(n_pairs):
        clean_text = get_text(death_me[i])
        corrupt_text = get_text(life_not_me[i])

        clean_prompt = build_prompt(clean_text)
        corrupt_prompt = build_prompt(corrupt_text)

        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer, clean_prompt, corrupt_prompt,
            max_length=MAX_TOKENS, device=device,
        )

        importance = run_head_knockout(
            model, corrupt_tokens, clean_tokens, yes_id, no_id,
        )

        all_importance[i] = importance
        top_idx = np.unravel_index(importance.argmax(), importance.shape)
        print(f"  [{model_name}] Case {i+1}/{n_pairs}: most important head = "
              f"L{top_idx[0]}H{top_idx[1]} (importance={importance.max():.4f})")

    mean_importance = all_importance.mean(axis=0)

    # Save per-model results
    torch.save(torch.from_numpy(mean_importance),
               output_dir / f"{model_name}_head_importance.pt")

    # --- Per-model Figure 1: Heatmap ---
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.8), max(8, n_layers * 0.3)))
    im = ax.imshow(mean_importance, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Head Importance ({model_name.capitalize()}, {dataset_label}, avg over {n_pairs} cases)\n"
                 f"|LD_baseline - LD_ablated| for each head")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
    ax.set_yticks(range(0, n_layers, 2))
    ax.set_yticklabels([f"L{l}" for l in range(0, n_layers, 2)], fontsize=8)
    fig.colorbar(im, ax=ax, label="Importance (LD drop)")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase5_heads_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 2: Top heads bar chart ---
    flat = mean_importance.flatten()
    top_k = min(20, len(flat))
    top_indices = np.argsort(flat)[::-1][:top_k]

    labels = []
    values = []
    for idx in top_indices:
        layer = idx // n_heads
        head = idx % n_heads
        labels.append(f"L{layer}H{head}")
        values.append(flat[idx])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_k))
    ax.barh(labels[::-1], values[::-1], color=colors_bar[::-1])
    ax.set_xlabel("Importance (LD drop)")
    ax.set_title(f"Top {top_k} Most Critical Heads ({model_name.capitalize()}, {dataset_label})")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase5_top_heads_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print(f"\n  [{model_name}] Top 10 most critical heads:")
    for idx in top_indices[:10]:
        layer = idx // n_heads
        head = idx % n_heads
        print(f"    L{layer}H{head}: importance = {flat[idx]:.4f}")

    layer_means = mean_importance.mean(axis=1)
    top_layers = np.argsort(layer_means)[::-1][:5]
    print(f"  [{model_name}] Top 5 layers by mean importance:")
    for l in top_layers:
        print(f"    Layer {l}: mean = {layer_means[l]:.4f}")

    result = {
        "mean_importance": mean_importance,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }

    # Free VRAM
    unload_model(model)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def plot_comparison(all_results, output_dir, dataset_label):
    """Generate combined comparison plot showing all models overlaid."""
    model_colors = {"llama": "#e74c3c", "qwen": "#3498db", "mistral": "#2ecc71"}
    model_names = list(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Layer-wise mean importance overlaid
    ax = axes[0]
    for m in model_names:
        r = all_results[m]
        layer_means = r["mean_importance"].mean(axis=1)
        layers = list(range(r["n_layers"]))
        ax.plot(layers, layer_means, "o-", label=m.capitalize(),
                color=model_colors.get(m, "#333"), markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Head Importance")
    ax.set_title("Layer-wise Head Importance by Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Top-K head importance comparison
    ax = axes[1]
    top_k = 10
    x = np.arange(top_k)
    width = 0.25
    for i, m in enumerate(model_names):
        r = all_results[m]
        flat = r["mean_importance"].flatten()
        top_vals = np.sort(flat)[::-1][:top_k]
        ax.bar(x + i * width, top_vals, width, label=m.capitalize(),
               color=model_colors.get(m, "#333"), alpha=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Importance (LD drop)")
    ax.set_title(f"Top {top_k} Head Importances by Model")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"#{j+1}" for j in range(top_k)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 5: Cross-Model Head Importance Comparison ({dataset_label})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase5_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase5_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    path = output_dir / f"{model_name}_head_importance.pt"
    if path.exists():
        data = torch.load(path, weights_only=True)
        if isinstance(data, torch.Tensor):
            importance = data.numpy()
        else:
            importance = data
        return {
            "mean_importance": importance,
            "n_layers": importance.shape[0],
            "n_heads": importance.shape[1],
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Per-Head Knockout Analysis")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 5: Per-Head Knockout Analysis ({', '.join(m.capitalize() for m in models_to_run)})")
    print(f"Dataset: {dataset_label}")
    print("=" * 70)

    ensure_dir(output_dir)

    categories = load_dataset_by_category(dataset_name=dataset_name)
    death_me = categories.get("Death-Me", [])
    life_not_me = categories.get("Life-Not-Me", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for model_name in models_to_run:
        try:
            all_results[model_name] = analyze_single_model(
                model_name, death_me, life_not_me, device, output_dir, dataset_label)
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            print(f"Skipping {model_name}")
            continue

    # Load previously saved results for models we didn't run
    for model_name in MODELS:
        if model_name not in all_results:
            saved = load_saved_results(model_name, output_dir)
            if saved is not None:
                all_results[model_name] = saved
                print(f"Loaded saved results for {model_name}")

    # Combined comparison plot
    if len(all_results) >= 2:
        plot_comparison(all_results, output_dir, dataset_label)

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    for model_name, r in all_results.items():
        flat = r["mean_importance"].flatten()
        top_idx = np.argsort(flat)[::-1][0]
        layer = top_idx // r["n_heads"]
        head = top_idx % r["n_heads"]
        print(f"\n{model_name.capitalize()}:")
        print(f"  Layers: {r['n_layers']}, Heads: {r['n_heads']}")
        print(f"  Most important head: L{layer}H{head} ({flat[top_idx]:.4f})")
        print(f"  Mean importance: {flat.mean():.4f}")

    print(f"\nPhase 5 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
