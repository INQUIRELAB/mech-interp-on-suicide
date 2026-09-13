#!/usr/bin/env python
"""Phase 3: Distress-Aware Prompting Analysis.

Tests how different prompting strategies affect the safety circuit:
- Standard: neutral prompt
- Empathetic: compassionate mental health framing
- Clinical: clinical psychologist risk assessment framing

Runs on all three models (Llama, Qwen, Mistral) sequentially.

Output (per model):
    results/phase3/{model}_standard_ie.pt        - [2, n_layers] standard IE
    results/phase3/{model}_empathetic_ie.pt      - [2, n_layers] empathetic IE
    results/phase3/{model}_clinical_ie.pt        - [2, n_layers] clinical IE
    results/phase3/phase3_prompting_{model}.png  - IE curves for 3 prompt styles
    results/phase3/phase3_logit_diffs_{model}.png - Bar chart of logit diffs
    results/phase3/phase3_comparison.png         - All models overlaid
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
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3"
PROMPT_STYLES = ["standard", "empathetic", "clinical"]
MODELS = ["llama", "qwen", "mistral"]


def analyze_single_model(model_name, death_me, life_not_me, device, output_dir, dataset_label):
    """Run Phase 3 analysis for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    results = {}

    for style in PROMPT_STYLES:
        print(f"\n  [{model_name}] --- Prompt style: {style} ---")
        all_attn_ie = torch.zeros(n_pairs, n_layers)
        all_mlp_ie = torch.zeros(n_pairs, n_layers)
        logit_diffs = []

        for i in range(n_pairs):
            clean_text = get_text(death_me[i])
            corrupt_text = get_text(life_not_me[i])

            clean_prompt = build_prompt(clean_text, style=style)
            corrupt_prompt = build_prompt(corrupt_text, style=style)

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

            print(f"    [{model_name}:{style}] Case {i+1}/{n_pairs}: LD={ld_clean:.2f}, "
                  f"max_attn_IE={attn_ie.max():.3f}")

        mean_attn = all_attn_ie.mean(dim=0)
        mean_mlp = all_mlp_ie.mean(dim=0)

        results[style] = {
            "attn_ie": mean_attn,
            "mlp_ie": mean_mlp,
            "logit_diffs": logit_diffs,
            "mean_ld": np.mean(logit_diffs),
        }

        torch.save(
            torch.stack([mean_attn, mean_mlp]),
            output_dir / f"{model_name}_{style}_ie.pt",
        )

    # --- Per-model Figure 1: Attention IE curves by prompting style ---
    layers = list(range(n_layers))
    style_colors = {"standard": "#95a5a6", "empathetic": "#2ecc71", "clinical": "#9b59b6"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for style in PROMPT_STYLES:
        ax.plot(layers, results[style]["attn_ie"].numpy(), "o-",
                label=style.capitalize(), color=style_colors[style], markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Indirect Effect")
    ax.set_title(f"Attention IE ({model_name.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for style in PROMPT_STYLES:
        ax.plot(layers, results[style]["mlp_ie"].numpy(), "s-",
                label=style.capitalize(), color=style_colors[style], markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("MLP Indirect Effect")
    ax.set_title(f"MLP IE ({model_name.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 3: Prompting Effects ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase3_prompting_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 2: Logit diff bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    style_labels = [s.capitalize() for s in PROMPT_STYLES]
    means = [results[s]["mean_ld"] for s in PROMPT_STYLES]
    stds = [np.std(results[s]["logit_diffs"]) for s in PROMPT_STYLES]
    bar_colors = [style_colors[s] for s in PROMPT_STYLES]

    bars = ax.bar(style_labels, means, yerr=stds, color=bar_colors, capsize=5, alpha=0.8)
    ax.set_ylabel("Logit Difference (Yes - No)")
    ax.set_title(f"Safety Signal by Prompting Strategy ({model_name.capitalize()})\n"
                 f"(Death-Me cases)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase3_logit_diffs_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\n  [{model_name}] RESULTS:")
    for style in PROMPT_STYLES:
        r = results[style]
        print(f"  [{model_name}] {style.capitalize()}: "
              f"Mean LD={r['mean_ld']:.3f}, "
              f"Mean Attn IE={r['attn_ie'].mean():.4f}, "
              f"Peak Attn IE={r['attn_ie'].max():.4f} (L{r['attn_ie'].argmax()})")

    result = {
        "style_results": results,
        "n_layers": n_layers,
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = list(all_results.keys())

    # Panel 1: Mean LD by model and prompt style (grouped bars)
    ax = axes[0]
    x = np.arange(len(PROMPT_STYLES))
    width = 0.25
    for i, m in enumerate(model_names):
        means = [all_results[m]["style_results"][s]["mean_ld"] for s in PROMPT_STYLES]
        ax.bar(x + i * width, means, width, label=m.capitalize(),
               color=model_colors.get(m, "#333"), alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in PROMPT_STYLES])
    ax.set_ylabel("Mean Logit Diff")
    ax.set_title("Safety Signal by Model & Prompt")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Clinical prompt attention IE overlaid across models
    ax = axes[1]
    for m in model_names:
        r = all_results[m]
        layers = list(range(r["n_layers"]))
        ax.plot(layers, r["style_results"]["clinical"]["attn_ie"].numpy(), "o-",
                label=m.capitalize(), color=model_colors.get(m, "#333"), markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention IE")
    ax.set_title("Clinical Prompt Attn IE by Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Prompt improvement (clinical - standard) by model
    ax = axes[2]
    improvements = []
    for m in model_names:
        sr = all_results[m]["style_results"]
        improvements.append(sr["clinical"]["mean_ld"] - sr["standard"]["mean_ld"])
    bar_colors = [model_colors.get(m, "#333") for m in model_names]
    bars = ax.bar([m.capitalize() for m in model_names], improvements,
                  color=bar_colors, alpha=0.8)
    ax.set_ylabel("LD Improvement (Clinical - Standard)")
    ax.set_title("Clinical Prompt Benefit by Model")
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{imp:+.2f}", ha="center", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 3: Cross-Model Prompting Strategy Comparison [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase3_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase3_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    style_results = {}
    for style in PROMPT_STYLES:
        path = output_dir / f"{model_name}_{style}_ie.pt"
        if not path.exists():
            return None
        ie_data = torch.load(path, weights_only=True)  # [2, n_layers]
        style_results[style] = {
            "attn_ie": ie_data[0], "mlp_ie": ie_data[1],
            "logit_diffs": [],  # Not available from saved data
            "mean_ld": 0.0,  # Not available from saved data
        }
    n_layers = style_results["standard"]["attn_ie"].shape[0]
    return {"style_results": style_results, "n_layers": n_layers}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Prompting Strategy Analysis")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 3: Prompting Strategy Analysis ({', '.join(m.capitalize() for m in models_to_run)})")
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
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    for model_name, r in all_results.items():
        print(f"\n{model_name.capitalize()}:")
        for style in PROMPT_STYLES:
            sr = r["style_results"][style]
            print(f"  {style.capitalize()}: Mean LD={sr['mean_ld']:.3f}, "
                  f"Mean Attn IE={sr['attn_ie'].mean():.4f}")

    print(f"\nPhase 3 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
