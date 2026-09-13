#!/usr/bin/env python
"""Phase 4: Cross-Model Comparison.

Compares safety circuit structure across three models:
- Llama-3.1-8B-Instruct
- Qwen-2.5-7B
- Mistral-7B

Models are loaded ONE AT A TIME to conserve VRAM.

Output (per model):
    results/phase4/{model}_circuit.pt            - Model IE data
    results/phase4/phase4_cross_model_{model}.png - Per-model IE plot
    results/phase4/phase4_cross_model.png        - Side-by-side IE comparison
    results/phase4/phase4_overlay.png            - Overlay comparison
    results/phase4/phase4_comparison.png         - Alias for overlay
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
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase4"
MODELS = ["llama", "qwen", "mistral"]


def analyze_model(model_name, death_me, life_not_me, device, output_dir, dataset_label):
    """Run activation patching for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    all_attn_ie = torch.zeros(n_pairs, n_layers)
    all_mlp_ie = torch.zeros(n_pairs, n_layers)
    logit_diffs = []

    for i in range(n_pairs):
        clean_text = get_text(death_me[i])
        corrupt_text = get_text(life_not_me[i])

        clean_prompt = build_prompt(clean_text)
        corrupt_prompt = build_prompt(corrupt_text)

        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer, clean_prompt, corrupt_prompt,
            max_length=MAX_TOKENS, device=device,
        )

        attn_ie, mlp_ie, ld_clean, ld_corrupt = run_activation_patching(
            model, clean_tokens, corrupt_tokens, yes_id, no_id,
        )

        all_attn_ie[i] = attn_ie
        all_mlp_ie[i] = mlp_ie
        logit_diffs.append(ld_clean.item())

        print(f"  [{model_name}] Case {i+1}/{n_pairs}: LD={ld_clean:.2f}, "
              f"max_attn_IE={attn_ie.max():.3f}")

    result = {
        "attn_ie": all_attn_ie.mean(dim=0),
        "mlp_ie": all_mlp_ie.mean(dim=0),
        "logit_diffs": logit_diffs,
        "mean_ld": np.mean(logit_diffs),
        "n_layers": n_layers,
    }

    # Per-model plot
    layers = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(layers, result["attn_ie"].numpy(), "o-", color="#e74c3c", markersize=3)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Attention IE")
    axes[0].set_title(f"Attention IE ({model_name.capitalize()})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(layers, result["mlp_ie"].numpy(), "s-", color="#3498db", markersize=3)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("MLP IE")
    axes[1].set_title(f"MLP IE ({model_name.capitalize()})")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Phase 4: Safety Circuit ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase4_cross_model_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Free VRAM before loading next model
    unload_model(model)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    path = output_dir / f"{model_name}_circuit.pt"
    if path.exists():
        data = torch.load(path, weights_only=False)
        return data
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Cross-Model Comparison")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 4: Cross-Model Safety Circuit Comparison ({', '.join(m.capitalize() for m in models_to_run)}) [{dataset_label}]")
    print("=" * 70)

    ensure_dir(output_dir)

    categories = load_dataset_by_category(dataset_name=dataset_name)
    death_me = categories.get("Death-Me", [])
    life_not_me = categories.get("Life-Not-Me", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for model_name in models_to_run:
        try:
            results[model_name] = analyze_model(
                model_name, death_me, life_not_me, device, output_dir, dataset_label)
            torch.save(results[model_name], output_dir / f"{model_name}_circuit.pt")
        except Exception as e:
            print(f"ERROR loading {model_name}: {e}")
            print(f"Skipping {model_name}")
            continue

    # Load previously saved results for models we didn't run
    for model_name in MODELS:
        if model_name not in results:
            saved = load_saved_results(model_name, output_dir)
            if saved is not None:
                results[model_name] = saved
                print(f"Loaded saved results for {model_name}")

    if len(results) < 2:
        print("WARNING: Need at least 2 models for comparison plot (have %d)" % len(results))
        if len(results) == 0:
            return

    # --- Figure: Cross-model side-by-side ---
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    colors = {"llama": "#e74c3c", "qwen": "#3498db", "mistral": "#2ecc71"}

    for col, (model_name, r) in enumerate(results.items()):
        n_layers = r["n_layers"]
        layers = list(range(n_layers))
        color = colors.get(model_name, "#333333")

        ax = axes[0, col]
        ax.plot(layers, r["attn_ie"].numpy(), "o-", color=color, markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention IE")
        ax.set_title(f"{model_name.capitalize()} Attention")
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(layers, r["mlp_ie"].numpy(), "s-", color=color, markersize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("MLP IE")
        ax.set_title(f"{model_name.capitalize()} MLP")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 4: Cross-Model Safety Circuit Comparison [{dataset_label}]\n"
                 "(Death-Me vs Life-Not-Me, denoising IE)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase4_cross_model.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Overlay figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, r in results.items():
        layers = list(range(r["n_layers"]))
        color = colors.get(model_name, "#333333")
        axes[0].plot(layers, r["attn_ie"].numpy(), "o-", label=model_name.capitalize(),
                     color=color, markersize=3)
        axes[1].plot(layers, r["mlp_ie"].numpy(), "s-", label=model_name.capitalize(),
                     color=color, markersize=3)

    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Attention IE")
    axes[0].set_title("Attention IE"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("MLP IE")
    axes[1].set_title("MLP IE"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 4: Cross-Model IE Overlay [{dataset_label}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase4_overlay.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(output_dir / "phase4_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 RESULTS")
    print("=" * 70)
    for model_name, r in results.items():
        print(f"\n{model_name.capitalize()}:")
        print(f"  Layers: {r['n_layers']}")
        print(f"  Mean LD: {r['mean_ld']:.3f}")
        print(f"  Mean Attn IE: {r['attn_ie'].mean():.4f}")
        print(f"  Peak Attn IE: {r['attn_ie'].max():.4f} (L{r['attn_ie'].argmax()})")
        print(f"  Mean MLP IE:  {r['mlp_ie'].mean():.4f}")

    print(f"\nPhase 4 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
