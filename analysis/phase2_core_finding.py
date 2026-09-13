#!/usr/bin/env python
"""Phase 2: Why Do Implicit Cases Fail? (Core Finding)

Compares the model's safety circuit response between:
- Death-Me cases (implicit suicidal ideation - hardest to detect)
- Death-Not-Me cases (explicit death references - easier)
- Life-Not-Me cases (control/baseline)

Runs on all three models (Llama, Qwen, Mistral) sequentially.

Output (per model):
    results/phase2/{model}_death_me_ie.pt        - [2, n_layers] attn+mlp IE for Death-Me
    results/phase2/{model}_death_not_me_ie.pt    - [2, n_layers] for Death-Not-Me
    results/phase2/{model}_logit_diffs.pt        - Per-category logit differences
    results/phase2/phase2_core_finding_{model}.png  - Comparison bar chart + IE curves
    results/phase2/phase2_ie_comparison_{model}.png - IE curves overlaid
    results/phase2/phase2_comparison.png         - All models overlaid
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
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"
MODELS = ["llama", "qwen", "mistral"]


def run_category_analysis(model, tokenizer, cases, corrupt_cases, yes_id, no_id,
                          category_name, model_name, device):
    """Run activation patching for one category of cases."""
    n_pairs = min(N_CASES, len(cases), len(corrupt_cases))
    n_layers = model.cfg.n_layers

    all_attn_ie = torch.zeros(n_pairs, n_layers)
    all_mlp_ie = torch.zeros(n_pairs, n_layers)
    logit_diffs = []

    for i in range(n_pairs):
        clean_text = get_text(cases[i])
        corrupt_text = get_text(corrupt_cases[i])

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

        print(f"  [{model_name}:{category_name}] Case {i+1}/{n_pairs}: LD={ld_clean:.2f}, "
              f"max_attn_IE={attn_ie.max():.3f}")

    return {
        "attn_ie": all_attn_ie.mean(dim=0),
        "mlp_ie": all_mlp_ie.mean(dim=0),
        "logit_diffs": logit_diffs,
        "mean_ld": np.mean(logit_diffs),
    }


def analyze_single_model(model_name, death_me, death_not_me, life_not_me, device,
                         output_dir, dataset_label):
    """Run Phase 2 analysis for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)
    n_layers = model.cfg.n_layers

    # Run analysis for each category
    print(f"\n  [{model_name}] --- Death-Me (implicit, hardest) ---")
    dm_results = run_category_analysis(
        model, tokenizer, death_me, life_not_me, yes_id, no_id,
        "Death-Me", model_name, device)

    print(f"\n  [{model_name}] --- Death-Not-Me (explicit death references) ---")
    dnm_results = run_category_analysis(
        model, tokenizer, death_not_me, life_not_me, yes_id, no_id,
        "Death-Not-Me", model_name, device)

    # Save per-model data
    torch.save(
        torch.stack([dm_results["attn_ie"], dm_results["mlp_ie"]]),
        output_dir / f"{model_name}_death_me_ie.pt",
    )
    torch.save(
        torch.stack([dnm_results["attn_ie"], dnm_results["mlp_ie"]]),
        output_dir / f"{model_name}_death_not_me_ie.pt",
    )
    torch.save({
        "Death-Me": dm_results["logit_diffs"],
        "Death-Not-Me": dnm_results["logit_diffs"],
    }, output_dir / f"{model_name}_logit_diffs.pt")

    # --- Per-model Figure 1: Logit Difference Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cats = ["Death-Me\n(implicit)", "Death-Not-Me\n(explicit)"]
    means = [dm_results["mean_ld"], dnm_results["mean_ld"]]
    stds = [np.std(dm_results["logit_diffs"]), np.std(dnm_results["logit_diffs"])]
    colors = ["#e74c3c", "#3498db"]
    bars = ax.bar(cats, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.set_ylabel("Logit Difference (Yes - No)")
    ax.set_title(f"Safety Signal Strength ({model_name.capitalize()})")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=11)

    ax = axes[1]
    layers = list(range(n_layers))
    ax.plot(layers, dm_results["attn_ie"].numpy(), "o-",
            label="Death-Me (implicit)", color="#e74c3c", markersize=3, alpha=0.8)
    ax.plot(layers, dnm_results["attn_ie"].numpy(), "s-",
            label="Death-Not-Me (explicit)", color="#3498db", markersize=3, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Indirect Effect")
    ax.set_title(f"Safety Circuit: Implicit vs Explicit ({model_name.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 2: Why Implicit Cases Fail ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase2_core_finding_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 2: Full IE comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(layers, dm_results["attn_ie"].numpy(), "o-", label="Death-Me",
            color="#e74c3c", markersize=3)
    ax.plot(layers, dnm_results["attn_ie"].numpy(), "s-", label="Death-Not-Me",
            color="#3498db", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Indirect Effect")
    ax.set_title(f"Attention IE ({model_name.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, dm_results["mlp_ie"].numpy(), "o-", label="Death-Me",
            color="#e74c3c", markersize=3)
    ax.plot(layers, dnm_results["mlp_ie"].numpy(), "s-", label="Death-Not-Me",
            color="#3498db", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Indirect Effect")
    ax.set_title(f"MLP IE ({model_name.capitalize()})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 2: IE Comparison ({model_name.capitalize()}) [{dataset_label}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase2_ie_comparison_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\n  [{model_name}] RESULTS:")
    print(f"  [{model_name}] Mean LD - Death-Me (implicit): {dm_results['mean_ld']:.3f}")
    print(f"  [{model_name}] Mean LD - Death-Not-Me (explicit): {dnm_results['mean_ld']:.3f}")
    gap = dnm_results['mean_ld'] - dm_results['mean_ld']
    print(f"  [{model_name}] Gap (explicit - implicit): {gap:.3f}")
    print(f"  [{model_name}] Mean Attn IE - Death-Me: {dm_results['attn_ie'].mean():.4f}")
    print(f"  [{model_name}] Mean Attn IE - Death-Not-Me: {dnm_results['attn_ie'].mean():.4f}")

    result = {
        "dm_results": dm_results,
        "dnm_results": dnm_results,
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
    colors = {"llama": "#e74c3c", "qwen": "#3498db", "mistral": "#2ecc71"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Mean LD bar chart across models
    ax = axes[0]
    model_names = list(all_results.keys())
    dm_means = [all_results[m]["dm_results"]["mean_ld"] for m in model_names]
    dnm_means = [all_results[m]["dnm_results"]["mean_ld"] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, dm_means, width, label="Death-Me (implicit)",
           color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, dnm_means, width, label="Death-Not-Me (explicit)",
           color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in model_names])
    ax.set_ylabel("Mean Logit Diff (Yes - No)")
    ax.set_title("Safety Signal Strength by Model")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Death-Me attention IE overlaid
    ax = axes[1]
    for m in model_names:
        r = all_results[m]
        layers = list(range(r["n_layers"]))
        ax.plot(layers, r["dm_results"]["attn_ie"].numpy(), "o-",
                label=m.capitalize(), color=colors.get(m, "#333"), markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention IE")
    ax.set_title("Death-Me Attention IE by Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Implicit-explicit gap by model
    ax = axes[2]
    gaps = [all_results[m]["dnm_results"]["mean_ld"] - all_results[m]["dm_results"]["mean_ld"]
            for m in model_names]
    bar_colors = [colors.get(m, "#333") for m in model_names]
    bars = ax.bar([m.capitalize() for m in model_names], gaps, color=bar_colors, alpha=0.8)
    ax.set_ylabel("LD Gap (Explicit - Implicit)")
    ax.set_title("Implicit Detection Gap by Model")
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{gap:.2f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 2: Cross-Model Implicit Failure Comparison [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase2_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase2_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    dm_path = output_dir / f"{model_name}_death_me_ie.pt"
    dnm_path = output_dir / f"{model_name}_death_not_me_ie.pt"
    ld_path = output_dir / f"{model_name}_logit_diffs.pt"
    if dm_path.exists() and dnm_path.exists() and ld_path.exists():
        dm_ie = torch.load(dm_path, weights_only=True)  # [2, n_layers]
        dnm_ie = torch.load(dnm_path, weights_only=True)
        ld_data = torch.load(ld_path, weights_only=False)
        n_layers = dm_ie.shape[1]
        return {
            "dm_results": {
                "attn_ie": dm_ie[0], "mlp_ie": dm_ie[1],
                "logit_diffs": ld_data["Death-Me"],
                "mean_ld": np.mean(ld_data["Death-Me"]),
            },
            "dnm_results": {
                "attn_ie": dnm_ie[0], "mlp_ie": dnm_ie[1],
                "logit_diffs": ld_data["Death-Not-Me"],
                "mean_ld": np.mean(ld_data["Death-Not-Me"]),
            },
            "n_layers": n_layers,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Implicit vs Explicit")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 2: Implicit vs Explicit ({', '.join(m.capitalize() for m in models_to_run)})")
    print(f"Dataset: {dataset_label}")
    print("=" * 70)

    ensure_dir(output_dir)

    categories = load_dataset_by_category(dataset_name=dataset_name)
    death_me = categories.get("Death-Me", [])
    death_not_me = categories.get("Death-Not-Me", [])
    life_not_me = categories.get("Life-Not-Me", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for model_name in models_to_run:
        try:
            all_results[model_name] = analyze_single_model(
                model_name, death_me, death_not_me, life_not_me, device,
                output_dir=output_dir, dataset_label=dataset_label)
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            print(f"Skipping {model_name}")
            continue

    # Load previously saved results for models we didn't run
    for model_name in MODELS:
        if model_name not in all_results:
            saved = load_saved_results(model_name, output_dir=output_dir)
            if saved is not None:
                all_results[model_name] = saved
                print(f"Loaded saved results for {model_name}")

    # Combined comparison plot
    if len(all_results) >= 2:
        plot_comparison(all_results, output_dir=output_dir, dataset_label=dataset_label)

    # Final summary
    print("\n" + "=" * 70)
    print(f"PHASE 2 COMPLETE [{dataset_label}]")
    print("=" * 70)
    for model_name, r in all_results.items():
        dm = r["dm_results"]
        dnm = r["dnm_results"]
        gap = dnm["mean_ld"] - dm["mean_ld"]
        print(f"\n{model_name.capitalize()}:")
        print(f"  Mean LD - Death-Me (implicit):     {dm['mean_ld']:.3f}")
        print(f"  Mean LD - Death-Not-Me (explicit): {dnm['mean_ld']:.3f}")
        print(f"  Gap (explicit - implicit): {gap:.3f}")
        print(f"  Mean Attn IE - Death-Me: {dm['attn_ie'].mean():.4f}")
        print(f"  Mean Attn IE - Death-Not-Me: {dnm['attn_ie'].mean():.4f}")

    print(f"\nPhase 2 complete. Dataset: {dataset_label}")


if __name__ == "__main__":
    main()
