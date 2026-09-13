#!/usr/bin/env python
"""Phase 1: Safety Circuit Mapping via Activation Patching.

Maps the three-stage safety circuit by running real activation patching
(denoising) on suicide detection prompts across all three models.

Clean: Death-Me cases (implicit suicidal ideation)
Corrupt: Life-Not-Me cases (no death association)

Output (per model):
    results/phase1/{model}_attn_ie.pt          - [n_layers] attention IE
    results/phase1/{model}_mlp_ie.pt           - [n_layers] MLP IE
    results/phase1/phase1_circuit_{model}.png  - IE curves (attn + MLP by layer)
    results/phase1/phase1_heatmap_{model}.png  - IE heatmap
    results/phase1/phase1_comparison.png       - All models overlaid
"""

import sys
import gc
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair,
    run_activation_patching, plot_ie_curves, plot_ie_heatmap, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 100
MAX_TOKENS = 256
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1"
MODELS = ["llama", "qwen", "mistral"]


def analyze_single_model(model_name, death_me, life_not_me, device, output_dir, dataset_label):
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

        all_attn_ie[i] = attn_ie
        all_mlp_ie[i] = mlp_ie

        print(f"  [{model_name}] Case {i+1}/{n_pairs}: LD_clean={ld_clean:.2f}, "
              f"LD_corrupt={ld_corrupt:.2f}, "
              f"max_attn_IE={attn_ie.max():.3f} (L{attn_ie.argmax()}), "
              f"max_mlp_IE={mlp_ie.max():.3f} (L{mlp_ie.argmax()})")

    mean_attn_ie = all_attn_ie.mean(dim=0)
    mean_mlp_ie = all_mlp_ie.mean(dim=0)

    # Save per-model results
    torch.save(mean_attn_ie, output_dir / f"{model_name}_attn_ie.pt")
    torch.save(mean_mlp_ie, output_dir / f"{model_name}_mlp_ie.pt")
    print(f"  [{model_name}] Saved IE data to {output_dir}")

    # Per-model IE curves plot
    plot_ie_curves(
        mean_attn_ie, mean_mlp_ie,
        title=f"Phase 1: Safety Circuit ({model_name.capitalize()}, {dataset_label}, avg over {n_pairs} cases)\n"
              f"Denoising IE: Death-Me (clean) vs Life-Not-Me (corrupt)",
        save_path=str(output_dir / f"phase1_circuit_{model_name}.png"),
    )

    # Per-model heatmap
    plot_ie_heatmap(
        mean_attn_ie.numpy(), mean_mlp_ie.numpy(),
        title=f"Phase 1: Layer-wise IE Heatmap ({model_name.capitalize()}, {dataset_label})",
        save_path=str(output_dir / f"phase1_heatmap_{model_name}.png"),
    )

    # Summary for this model
    print(f"\n  [{model_name}] RESULTS SUMMARY")
    top_attn = mean_attn_ie.topk(5)
    top_mlp = mean_mlp_ie.topk(5)

    print(f"  [{model_name}] Top 5 Attention layers by IE:")
    for val, idx in zip(top_attn.values, top_attn.indices):
        print(f"    Layer {idx.item():2d}: IE = {val.item():.4f}")

    print(f"  [{model_name}] Top 5 MLP layers by IE:")
    for val, idx in zip(top_mlp.values, top_mlp.indices):
        print(f"    Layer {idx.item():2d}: IE = {val.item():.4f}")

    result = {
        "attn_ie": mean_attn_ie,
        "mlp_ie": mean_mlp_ie,
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"llama": "#e74c3c", "qwen": "#3498db", "mistral": "#2ecc71"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, r in all_results.items():
        layers = list(range(r["n_layers"]))
        color = colors.get(model_name, "#333333")
        axes[0].plot(layers, r["attn_ie"].numpy(), "o-",
                     label=model_name.capitalize(), color=color, markersize=3)
        axes[1].plot(layers, r["mlp_ie"].numpy(), "s-",
                     label=model_name.capitalize(), color=color, markersize=3)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Attention IE")
    axes[0].set_title("Attention IE by Model")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("MLP IE")
    axes[1].set_title("MLP IE by Model")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Phase 1: Cross-Model Safety Circuit Comparison ({dataset_label})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase1_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    attn_path = output_dir / f"{model_name}_attn_ie.pt"
    mlp_path = output_dir / f"{model_name}_mlp_ie.pt"
    if attn_path.exists() and mlp_path.exists():
        attn_ie = torch.load(attn_path, weights_only=True)
        mlp_ie = torch.load(mlp_path, weights_only=True)
        return {"attn_ie": attn_ie, "mlp_ie": mlp_ie, "n_layers": len(attn_ie)}
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Safety Circuit Mapping")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)
    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 1: Safety Circuit Mapping")
    print(f"  Dataset: {dataset_label}")
    print(f"  Models:  {', '.join(m.capitalize() for m in models_to_run)}")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    ensure_dir(output_dir)

    categories = load_dataset_by_category(dataset_name=dataset_name)
    death_me = categories.get("Death-Me", [])
    life_not_me = categories.get("Life-Not-Me", [])

    if not death_me or not life_not_me:
        print("ERROR: Missing required categories in dataset")
        return

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
    print(f"PHASE 1 COMPLETE ({dataset_label})")
    print("=" * 70)
    for model_name, r in all_results.items():
        print(f"\n{model_name.capitalize()}:")
        print(f"  Layers: {r['n_layers']}")
        print(f"  Mean Attn IE: {r['attn_ie'].mean():.4f}")
        print(f"  Peak Attn IE: {r['attn_ie'].max():.4f} (L{r['attn_ie'].argmax()})")
        print(f"  Mean MLP IE:  {r['mlp_ie'].mean():.4f}")
        print(f"  Peak MLP IE:  {r['mlp_ie'].max():.4f} (L{r['mlp_ie'].argmax()})")

        for label, ie in [("Attention", r["attn_ie"]), ("MLP", r["mlp_ie"])]:
            n = r["n_layers"]
            third = n // 3
            early = ie[:third].mean().item()
            mid = ie[third:2*third].mean().item()
            late = ie[2*third:].mean().item()
            print(f"  Three-stage pattern ({label}):")
            print(f"    Early (L0-{third-1}):    mean IE = {early:.4f}")
            print(f"    Middle (L{third}-{2*third-1}): mean IE = {mid:.4f}")
            print(f"    Late (L{2*third}+):     mean IE = {late:.4f}")

    print(f"\nPhase 1 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
