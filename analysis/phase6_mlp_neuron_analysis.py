#!/usr/bin/env python
"""Phase 6: MLP Neuron Analysis.

Identifies MLP neurons that differentially respond to implicit suicidal
content (Death-Me) vs non-suicidal content (Life-Not-Me), with proper
normalization to avoid the last-layer artifact.

Runs on all three models (Llama, Qwen, Mistral) sequentially.

Three analyses per model:
1. Normalized differential activation
2. Selectivity analysis (implicit vs explicit)
3. Causal validation via neuron patching

Output (per model):
    results/phase6/{model}_neuron_activations.pt       - Per-layer neuron data
    results/phase6/phase6_neurons_{model}.png          - Normalized layer analysis
    results/phase6/phase6_selectivity_{model}.png      - Selectivity analysis
    results/phase6/phase6_causal_validation_{model}.png - Causal validation
    results/phase6/phase6_comparison.png               - All models overlaid
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
    build_prompt, get_yes_no_ids, tokenize_pair, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 100
N_CAUSAL_CASES = 8
TOP_K_NEURONS = 20
MAX_TOKENS = 256
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase6"
MODELS = ["llama", "qwen", "mistral"]


@torch.no_grad()
def get_mlp_activations(model, tokens):
    """Get MLP output activations for all layers."""
    _, cache = model.run_with_cache(tokens)
    activations = {}
    for layer in range(model.cfg.n_layers):
        key = f"blocks.{layer}.hook_mlp_out"
        if key in cache:
            activations[layer] = cache[key]
    return activations


@torch.no_grad()
def collect_differential_activations(model, tokenizer, category_a, category_b,
                                     n_cases, device):
    """Collect mean differential MLP activations between two categories."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_pairs = min(n_cases, len(category_a), len(category_b))

    diff_by_layer = torch.zeros(n_layers, d_model)
    norm_by_layer = torch.zeros(n_layers)

    for i in range(n_pairs):
        text_a = get_text(category_a[i])
        text_b = get_text(category_b[i])

        tokens_a, tokens_b = tokenize_pair(
            tokenizer,
            build_prompt(text_a), build_prompt(text_b),
            max_length=MAX_TOKENS, device=device,
        )

        acts_a = get_mlp_activations(model, tokens_a)
        acts_b = get_mlp_activations(model, tokens_b)

        for layer in range(n_layers):
            if layer in acts_a and layer in acts_b:
                a = acts_a[layer][0, -1, :].float().cpu()
                b = acts_b[layer][0, -1, :].float().cpu()
                diff_by_layer[layer] += (a - b).abs()
                norm_by_layer[layer] += (a.norm() + b.norm()) / 2.0

    diff_by_layer /= n_pairs
    norm_by_layer /= n_pairs

    return diff_by_layer, norm_by_layer


@torch.no_grad()
def causal_validate_neurons(model, tokenizer, death_me, life_not_me,
                            top_neurons, yes_id, no_id, device):
    """Patch top neurons from clean into corrupt and measure LD change."""
    n_cases = min(N_CAUSAL_CASES, len(death_me), len(life_not_me))
    baseline_lds = []
    patched_lds = []

    neurons_by_layer = {}
    for layer, neuron in top_neurons:
        neurons_by_layer.setdefault(layer, []).append(neuron)

    for i in range(n_cases):
        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer,
            build_prompt(get_text(death_me[i])),
            build_prompt(get_text(life_not_me[i])),
            max_length=MAX_TOKENS, device=device,
        )

        _, clean_cache = model.run_with_cache(clean_tokens)

        corrupt_logits = model(corrupt_tokens)
        ld_corrupt = (corrupt_logits[0, -1, yes_id] - corrupt_logits[0, -1, no_id]).float().item()
        baseline_lds.append(ld_corrupt)

        for layer, neuron_list in neurons_by_layer.items():
            hook_name = f"blocks.{layer}.hook_mlp_out"
            clean_act = clean_cache.get(hook_name)
            if clean_act is None:
                continue

            def make_neuron_patch_hook(cached_act, neurons):
                def hook_fn(output, hook):
                    modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                    for n in neurons:
                        modified[:, :, n] = cached_act[:, :, n].to(modified.device)
                    return modified
                return hook_fn

            model.add_hook(hook_name, make_neuron_patch_hook(clean_act, neuron_list))

        patched_logits = model(corrupt_tokens)
        model.remove_hooks()

        ld_patched = (patched_logits[0, -1, yes_id] - patched_logits[0, -1, no_id]).float().item()
        patched_lds.append(ld_patched)

    return baseline_lds, patched_lds


def analyze_single_model(model_name, death_me, death_not_me, life_not_me, device,
                         output_dir, dataset_label):
    """Run Phase 6 analysis for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"  [{model_name}] Model: {n_layers} layers, d_model={d_model}")

    # Analysis 1: Normalized differential (Death-Me vs Life-Not-Me)
    print(f"\n  [{model_name}] --- Analysis 1: Normalized differential ---")
    diff_dm, norm_dm = collect_differential_activations(
        model, tokenizer, death_me, life_not_me, N_CASES, device)
    norm_diff_dm = diff_dm.mean(dim=1) / (norm_dm + 1e-8)

    for i in range(min(N_CASES, len(death_me), len(life_not_me))):
        print(f"    [{model_name}] Case {i+1} processed")

    # Analysis 2: Selectivity (Death-Not-Me vs Life-Not-Me)
    print(f"\n  [{model_name}] --- Analysis 2: Selectivity ---")
    diff_dnm, norm_dnm = collect_differential_activations(
        model, tokenizer, death_not_me, life_not_me, N_CASES, device)
    norm_diff_dnm = diff_dnm.mean(dim=1) / (norm_dnm + 1e-8)

    for i in range(min(N_CASES, len(death_not_me), len(life_not_me))):
        print(f"    [{model_name}] Case {i+1} processed")

    selectivity = diff_dm / (diff_dnm + 1e-8)

    # Find top neurons
    norm_diff_per_neuron = diff_dm / (norm_dm.unsqueeze(1) + 1e-8)
    flat_norm_diff = norm_diff_per_neuron.flatten()
    top_global = flat_norm_diff.topk(TOP_K_NEURONS * 3)
    top_neurons_with_scores = [
        (idx.item() // d_model, idx.item() % d_model, val.item())
        for idx, val in zip(top_global.indices, top_global.values)
    ]

    top_neurons_filtered = []
    for l, n, v in top_neurons_with_scores:
        sel = selectivity[l, n].item()
        top_neurons_filtered.append((l, n, v, sel))
    top_neurons_filtered.sort(key=lambda x: x[2], reverse=True)
    top_neurons_final = top_neurons_filtered[:TOP_K_NEURONS]

    # Analysis 3: Causal validation
    print(f"\n  [{model_name}] --- Analysis 3: Causal validation ---")
    top_neuron_coords = [(l, n) for l, n, v, s in top_neurons_final]
    baseline_lds, patched_lds = causal_validate_neurons(
        model, tokenizer, death_me, life_not_me,
        top_neuron_coords, yes_id, no_id, device,
    )

    for i, (bl, pl) in enumerate(zip(baseline_lds, patched_lds)):
        print(f"    [{model_name}] Case {i+1}: "
              f"baseline LD={bl:.3f}, patched LD={pl:.3f}, delta={pl-bl:+.3f}")

    # Save per-model data
    torch.save({
        "diff_dm": diff_dm,
        "diff_dnm": diff_dnm,
        "norm_dm": norm_dm,
        "norm_dnm": norm_dnm,
        "selectivity": selectivity,
        "top_neurons": top_neurons_final,
        "causal_baseline_lds": baseline_lds,
        "causal_patched_lds": patched_lds,
    }, output_dir / f"{model_name}_neuron_activations.pt")

    # --- Per-model Figure 1: Normalized layer-wise analysis ---
    raw_diff_dm = diff_dm.mean(dim=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(range(n_layers), raw_diff_dm.numpy(), "o-",
            color="#95a5a6", markersize=3, alpha=0.6, label="Raw |diff|")
    ax2.plot(range(n_layers), norm_diff_dm.numpy(), "s-",
             color="#e74c3c", markersize=4, label="Normalized |diff|/norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Raw |Clean - Corrupt|", color="#95a5a6")
    ax2.set_ylabel("Normalized Differential", color="#e74c3c")
    ax.set_title(f"MLP Differential ({model_name.capitalize()})")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(n_layers), norm_diff_dm.numpy(), "o-",
            color="#e74c3c", markersize=4, label="Death-Me (implicit)")
    ax.plot(range(len(norm_diff_dnm)), norm_diff_dnm.numpy(), "s-",
            color="#3498db", markersize=4, label="Death-Not-Me (explicit)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Differential")
    ax.set_title(f"Implicit vs Explicit ({model_name.capitalize()})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels_neurons = [f"L{l}N{n}" for l, n, v, s in top_neurons_final[:15]]
    values_neurons = [v for l, n, v, s in top_neurons_final[:15]]
    sels = [s for l, n, v, s in top_neurons_final[:15]]
    sel_arr = np.array(sels)
    sel_norm = np.clip((sel_arr - 0.5) / 2.0, 0, 1)
    colors_n = plt.cm.RdYlBu_r(sel_norm)
    ax.barh(labels_neurons[::-1], values_neurons[::-1], color=colors_n[::-1])
    ax.set_xlabel("Normalized Differential")
    ax.set_title(f"Top 15 Neurons ({model_name.capitalize()})")
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r",
                                norm=plt.Normalize(vmin=min(sels), vmax=max(sels)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Selectivity", fraction=0.05)

    fig.suptitle(f"Phase 6: MLP Neuron Analysis ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase6_neurons_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 2: Selectivity ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    layer_sel = selectivity.mean(dim=1).numpy()
    colors_layer = ["#e74c3c" if s > 1.0 else "#3498db" for s in layer_sel]
    ax.bar(range(n_layers), layer_sel, color=colors_layer, alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Equal response")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Selectivity (DM/DNM)")
    ax.set_title(f"Layer-wise Selectivity ({model_name.capitalize()})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    top_layer_idxs = norm_diff_dm.topk(5).indices.tolist()
    for li, layer in enumerate(top_layer_idxs):
        dm_vals = diff_dm[layer].numpy()
        dnm_vals = diff_dnm[layer].numpy()
        mask = dm_vals > np.percentile(dm_vals, 95)
        ax.scatter(dnm_vals[mask], dm_vals[mask], alpha=0.5, s=15,
                   label=f"L{layer}", zorder=2)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]],
            "k--", alpha=0.3, label="Equal line")
    ax.set_xlabel("Death-Not-Me |diff| (explicit)")
    ax.set_ylabel("Death-Me |diff| (implicit)")
    ax.set_title(f"Neuron Selectivity ({model_name.capitalize()})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase 6: Selectivity ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase6_selectivity_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 3: Causal validation ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(baseline_lds))
    width = 0.35
    ax.bar(x - width/2, baseline_lds, width, label="Corrupt baseline",
           color="#95a5a6", alpha=0.8)
    ax.bar(x + width/2, patched_lds, width, label="Neuron-patched",
           color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Case Index")
    ax.set_ylabel("Logit Diff (Yes - No)")
    ax.set_title(f"Causal Patching ({model_name.capitalize()})")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    deltas = [p - b for p, b in zip(patched_lds, baseline_lds)]
    mean_delta = np.mean(deltas)
    flips = sum(1 for b, p in zip(baseline_lds, patched_lds) if b < 0 and p > 0)

    layer_counts = {}
    for l, n, v, s in top_neurons_final:
        layer_counts[l] = layer_counts.get(l, 0) + 1
    layers_sorted = sorted(layer_counts.keys())
    counts_sorted = [layer_counts[l] for l in layers_sorted]

    ax.bar([f"L{l}" for l in layers_sorted], counts_sorted,
           color="#9b59b6", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Count in Top 20")
    ax.set_title(f"Top Neuron Distribution ({model_name.capitalize()})\n"
                 f"Mean LD shift: {mean_delta:+.3f}, Flips: {flips}/{len(baseline_lds)}")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 6: Causal Validation ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase6_causal_validation_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print(f"\n  [{model_name}] RESULTS:")
    top5_norm = norm_diff_dm.topk(5)
    for val, idx in zip(top5_norm.values, top5_norm.indices):
        print(f"    [{model_name}] Layer {idx.item()}: normalized = {val.item():.4f}")
    print(f"  [{model_name}] Causal mean LD shift: {mean_delta:+.3f}")
    print(f"  [{model_name}] Flips (No -> Yes): {flips}/{len(baseline_lds)}")

    result = {
        "norm_diff_dm": norm_diff_dm,
        "norm_diff_dnm": norm_diff_dnm,
        "top_neurons": top_neurons_final,
        "causal_mean_delta": mean_delta,
        "causal_flips": flips,
        "causal_total": len(baseline_lds),
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
    model_names = list(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Normalized differential overlaid
    ax = axes[0]
    for m in model_names:
        r = all_results[m]
        layers = list(range(r["n_layers"]))
        ax.plot(layers, r["norm_diff_dm"].numpy(), "o-",
                label=m.capitalize(), color=model_colors.get(m, "#333"), markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Differential")
    ax.set_title("Death-Me Normalized MLP Differential")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Causal validation summary
    ax = axes[1]
    x = np.arange(len(model_names))
    deltas = [all_results[m]["causal_mean_delta"] for m in model_names]
    bar_colors = [model_colors.get(m, "#333") for m in model_names]
    bars = ax.bar([m.capitalize() for m in model_names], deltas,
                  color=bar_colors, alpha=0.8)
    ax.set_ylabel("Mean LD Shift (patched - baseline)")
    ax.set_title("Causal Validation: Neuron Patching Effect")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{d:+.3f}", ha="center", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Flip rates
    ax = axes[2]
    flips = [all_results[m]["causal_flips"] for m in model_names]
    totals = [all_results[m]["causal_total"] for m in model_names]
    rates = [f/t if t > 0 else 0 for f, t in zip(flips, totals)]
    bars = ax.bar([m.capitalize() for m in model_names], rates,
                  color=bar_colors, alpha=0.8)
    ax.set_ylabel("Flip Rate (No -> Yes)")
    ax.set_title("Detection Flip Rate from Neuron Patching")
    for bar, f, t in zip(bars, flips, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{f}/{t}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 6: Cross-Model MLP Neuron Comparison [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase6_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase6_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    path = output_dir / f"{model_name}_neuron_activations.pt"
    if path.exists():
        data = torch.load(path, weights_only=False)
        diff_dm = data["diff_dm"]
        diff_dnm = data["diff_dnm"]
        norm_dm = data["norm_dm"]
        norm_dnm = data["norm_dnm"]
        n_layers = diff_dm.shape[0]
        norm_diff_dm = diff_dm.mean(dim=1) / (norm_dm + 1e-8)
        norm_diff_dnm = diff_dnm.mean(dim=1) / (norm_dnm + 1e-8)
        baseline_lds = data.get("causal_baseline_lds", [])
        patched_lds = data.get("causal_patched_lds", [])
        deltas = [p - b for p, b in zip(patched_lds, baseline_lds)]
        return {
            "norm_diff_dm": norm_diff_dm,
            "norm_diff_dnm": norm_diff_dnm,
            "top_neurons": data.get("top_neurons", []),
            "causal_mean_delta": np.mean(deltas) if deltas else 0.0,
            "causal_flips": sum(1 for b, p in zip(baseline_lds, patched_lds) if b < 0 and p > 0),
            "causal_total": len(baseline_lds),
            "n_layers": n_layers,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 6: MLP Neuron Analysis")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 6: MLP Neuron Analysis ({', '.join(m.capitalize() for m in models_to_run)})")
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
                output_dir, dataset_label)
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
    print("PHASE 6 COMPLETE")
    print("=" * 70)
    for model_name, r in all_results.items():
        print(f"\n{model_name.capitalize()}:")
        print(f"  Causal mean LD shift: {r['causal_mean_delta']:+.3f}")
        print(f"  Causal flips: {r['causal_flips']}/{r['causal_total']}")
        print(f"  Top neurons: {[(l, n) for l, n, v, s in r['top_neurons'][:5]]}...")

    print(f"\nPhase 6 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
