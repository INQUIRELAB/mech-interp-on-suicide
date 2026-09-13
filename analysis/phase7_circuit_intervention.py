#!/usr/bin/env python
"""Phase 7: Circuit-Informed Intervention.

Tests multiple intervention strategies to improve implicit suicide detection,
informed by findings from earlier phases:
  - Phase 1: Critical attention layers (IE analysis) - per model
  - Phase 5: Critical attention heads (knockout analysis) - per model

Runs on all three models (Llama, Qwen, Mistral) sequentially.

Three strategies per model:
  A. Targeted head patching: patch top heads (from Phase 5) with clean
     activations, targeting specific heads rather than entire layers.
  B. Activation steering: compute a "safety direction" from
     mean(clean - corrupt) at critical layers and add it during inference.
  C. Combined attention + MLP: amplify both attention (top heads) and
     MLP (top layers from Phase 1) simultaneously.

Output (per model):
    results/phase7/{model}_intervention_results.pt  - Detailed results
    results/phase7/phase7_intervention_{model}.png  - Strategy comparison
    results/phase7/phase7_per_case_{model}.png      - Per-case breakdown
    results/phase7/phase7_comparison.png            - All models overlaid
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
MAX_TOKENS = 256
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase7"
STEERING_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]
MODELS = ["llama", "qwen", "mistral"]
# Track dataset name for Phase 5 path resolution
CURRENT_DATASET = None


def load_phase5_heads(model_name, n_top=10, dataset_name=None):
    """Load top heads from Phase 5 knockout analysis for a specific model."""
    # Try per-model, per-dataset file first
    if dataset_name:
        path = PROJECT_ROOT / "results" / "phase5" / dataset_name / f"{model_name}_head_importance.pt"
        if not path.exists():
            print(f"  [{model_name}] Phase 5 results not found at {path}")
            # Try legacy path without dataset (only for llama)
            if model_name == "llama":
                path = PROJECT_ROOT / "results" / "phase5" / "head_importance.pt"
            else:
                return None
    else:
        path = PROJECT_ROOT / "results" / "phase5" / f"{model_name}_head_importance.pt"

    data = torch.load(path, weights_only=True)
    if isinstance(data, dict):
        importance = data.get("head_importance", data.get("importance"))
        if importance is None:
            importance = next(iter(data.values()))
    else:
        importance = data

    if isinstance(importance, np.ndarray):
        importance = torch.from_numpy(importance)

    flat = importance.flatten()
    top_k = flat.topk(n_top)
    n_heads = importance.shape[1]
    heads = [(idx.item() // n_heads, idx.item() % n_heads, val.item())
             for idx, val in zip(top_k.indices, top_k.values)]
    return heads, importance


def load_phase1_layers(model_name, n_top=5):
    """Load top attention and MLP layers from Phase 1 IE analysis for a specific model."""
    attn_path = PROJECT_ROOT / "results" / "phase1" / f"{model_name}_attn_ie.pt"
    mlp_path = PROJECT_ROOT / "results" / "phase1" / f"{model_name}_mlp_ie.pt"

    attn_layers = None
    mlp_layers = None

    if attn_path.exists():
        attn_ie = torch.load(attn_path, weights_only=True)
        attn_layers = attn_ie.topk(n_top).indices.tolist()

    if mlp_path.exists():
        mlp_ie = torch.load(mlp_path, weights_only=True)
        mlp_layers = mlp_ie.topk(n_top).indices.tolist()

    return attn_layers, mlp_layers


@torch.no_grad()
def compute_steering_vectors(model, tokenizer, death_me, life_not_me,
                             target_layers, n_cases, device):
    """Compute mean(clean - corrupt) direction at target layers."""
    n_pairs = min(n_cases, len(death_me), len(life_not_me))

    vectors = {layer: torch.zeros(model.cfg.d_model) for layer in target_layers}
    count = 0

    for i in range(n_pairs):
        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer,
            build_prompt(get_text(death_me[i])),
            build_prompt(get_text(life_not_me[i])),
            max_length=MAX_TOKENS, device=device,
        )

        _, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupt_cache = model.run_with_cache(corrupt_tokens)

        for layer in target_layers:
            for component in ["hook_attn_out", "hook_mlp_out"]:
                key = f"blocks.{layer}.{component}"
                if key in clean_cache and key in corrupt_cache:
                    c = clean_cache[key][0, -1, :].float().cpu()
                    x = corrupt_cache[key][0, -1, :].float().cpu()
                    vectors[layer] += (c - x)

        count += 1

    for layer in target_layers:
        vectors[layer] /= (count * 2)

    return vectors


@torch.no_grad()
def strategy_baseline(model, tokens, yes_id, no_id):
    """No intervention - just run the model."""
    logits = model(tokens)
    return (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()


@torch.no_grad()
def strategy_head_patching(model, clean_tokens, corrupt_tokens,
                           top_heads, yes_id, no_id):
    """Patch specific head outputs from clean run into corrupt run."""
    d_head = model.cfg.d_head

    _, clean_cache = model.run_with_cache(clean_tokens)

    heads_by_layer = {}
    for layer, head, importance in top_heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, head_list in heads_by_layer.items():
        hook_name = f"blocks.{layer}.hook_attn_out"
        clean_act = clean_cache.get(hook_name)
        if clean_act is None:
            continue

        def make_head_patch_hook(cached_act, heads, dh):
            def hook_fn(output, hook):
                modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                for h in heads:
                    start = h * dh
                    end = start + dh
                    modified[:, :, start:end] = cached_act[:, :, start:end].to(modified.device)
                return modified
            return hook_fn

        model.add_hook(hook_name, make_head_patch_hook(clean_act, head_list, d_head))

    logits = model(corrupt_tokens)
    model.remove_hooks()

    return (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()


@torch.no_grad()
def strategy_steering(model, tokens, steering_vectors, scale,
                      yes_id, no_id):
    """Add steering vector (safety direction) during forward pass."""
    for layer, vec in steering_vectors.items():
        hook_name = f"blocks.{layer}.hook_mlp_out"

        def make_steer_hook(v, s):
            def hook_fn(output, hook):
                modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                modified[:, -1, :] += (v * s).to(modified.device)
                return modified
            return hook_fn

        model.add_hook(hook_name, make_steer_hook(vec, scale))

    logits = model(tokens)
    model.remove_hooks()

    return (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()


@torch.no_grad()
def strategy_combined(model, clean_tokens, corrupt_tokens,
                      top_heads, mlp_layers, steering_vectors, scale,
                      yes_id, no_id):
    """Combined: patch top heads + steer MLP at critical layers."""
    d_head = model.cfg.d_head

    _, clean_cache = model.run_with_cache(clean_tokens)

    heads_by_layer = {}
    for layer, head, importance in top_heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, head_list in heads_by_layer.items():
        hook_name = f"blocks.{layer}.hook_attn_out"
        clean_act = clean_cache.get(hook_name)
        if clean_act is None:
            continue

        def make_head_patch_hook(cached_act, heads, dh):
            def hook_fn(output, hook):
                modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                for h in heads:
                    start = h * dh
                    end = start + dh
                    modified[:, :, start:end] = cached_act[:, :, start:end].to(modified.device)
                return modified
            return hook_fn

        model.add_hook(hook_name, make_head_patch_hook(clean_act, head_list, d_head))

    for layer, vec in steering_vectors.items():
        hook_name = f"blocks.{layer}.hook_mlp_out"

        def make_steer_hook(v, s):
            def hook_fn(output, hook):
                modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                modified[:, -1, :] += (v * s).to(modified.device)
                return modified
            return hook_fn

        model.add_hook(hook_name, make_steer_hook(vec, scale))

    logits = model(corrupt_tokens)
    model.remove_hooks()

    return (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()


def analyze_single_model(model_name, death_me, life_not_me, device, output_dir, dataset_label, dataset_name):
    """Run Phase 7 analysis for one model."""
    print(f"\n{'='*50}")
    print(f"[{model_name}] Loading model...")
    print(f"{'='*50}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    # Load prior phase results for THIS model
    print(f"\n  [{model_name}] --- Loading prior phase results ---")
    top_heads_result = load_phase5_heads(model_name, n_top=10, dataset_name=dataset_name)
    attn_layers, mlp_layers = load_phase1_layers(model_name, n_top=5)

    if top_heads_result is not None:
        top_heads, head_importance = top_heads_result
        print(f"  [{model_name}] Phase 5: Top 10 heads: "
              f"{[(l,h) for l,h,v in top_heads[:5]]}...")
    else:
        print(f"  [{model_name}] Phase 5 not available. Using fallback heads.")
        n_layers = model.cfg.n_layers
        top_heads = [(n_layers-1, h, 0.1) for h in [0, 1, 2, 3, 4]] + \
                    [(n_layers-2, h, 0.05) for h in [0, 1, 2, 3, 4]]
        head_importance = None

    if attn_layers:
        print(f"  [{model_name}] Phase 1 attention layers: {attn_layers}")
    if mlp_layers:
        print(f"  [{model_name}] Phase 1 MLP layers: {mlp_layers}")

    # Determine steering layers
    n_layers = model.cfg.n_layers
    default_attn = [n_layers // 3, n_layers * 2 // 3, n_layers - 4]
    default_mlp = [0, n_layers // 2, n_layers * 3 // 4]
    steering_layers = sorted(set(
        (attn_layers or default_attn) + (mlp_layers or default_mlp)
    ))[:8]
    print(f"  [{model_name}] Steering layers: {steering_layers}")

    # Compute steering vectors
    print(f"\n  [{model_name}] --- Computing steering vectors ---")
    n_steer_cases = min(5, n_pairs)
    steering_vectors = compute_steering_vectors(
        model, tokenizer, death_me, life_not_me,
        steering_layers, n_steer_cases, device,
    )
    for layer, vec in steering_vectors.items():
        print(f"    [{model_name}] Layer {layer}: steering vector norm = {vec.norm():.3f}")

    # Run all strategies
    print(f"\n  [{model_name}] --- Running interventions on {n_pairs} cases ---")

    results = {
        "baseline": [],
        "head_patch": [],
    }
    for s in STEERING_SCALES:
        results[f"steer_{s}x"] = []
        results[f"combined_{s}x"] = []

    for i in range(n_pairs):
        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer,
            build_prompt(get_text(death_me[i])),
            build_prompt(get_text(life_not_me[i])),
            max_length=MAX_TOKENS, device=device,
        )

        bl = strategy_baseline(model, clean_tokens, yes_id, no_id)
        results["baseline"].append(bl)

        hp = strategy_head_patching(
            model, clean_tokens, corrupt_tokens, top_heads, yes_id, no_id)
        results["head_patch"].append(hp)

        for s in STEERING_SCALES:
            sv = strategy_steering(
                model, clean_tokens, steering_vectors, s, yes_id, no_id)
            results[f"steer_{s}x"].append(sv)

            cb = strategy_combined(
                model, clean_tokens, corrupt_tokens,
                top_heads, mlp_layers or [], steering_vectors, s,
                yes_id, no_id)
            results[f"combined_{s}x"].append(cb)

        print(f"    [{model_name}] Case {i+1}/{n_pairs}: baseline={bl:.2f}, "
              f"head_patch={hp:.2f}, "
              f"steer_2x={results['steer_2.0x'][-1]:.2f}, "
              f"combined_2x={results['combined_2.0x'][-1]:.2f}")

    # Save per-model results
    save_data = {
        "results": results,
        "top_heads": top_heads,
        "steering_layers": steering_layers,
        "steering_scales": STEERING_SCALES,
    }
    torch.save(save_data, output_dir / f"{model_name}_intervention_results.pt")

    # Find best strategies
    best_steer_key = max([f"steer_{s}x" for s in STEERING_SCALES],
                         key=lambda k: np.mean(results[k]))
    best_combined_key = max([f"combined_{s}x" for s in STEERING_SCALES],
                            key=lambda k: np.mean(results[k]))

    # --- Per-model Figure 1: Strategy comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    strategy_names = ["baseline", "head_patch", best_steer_key, best_combined_key]
    strategy_labels = ["Baseline", "Head Patch",
                       f"Steer ({best_steer_key.split('_')[1]})",
                       f"Combined ({best_combined_key.split('_')[1]})"]
    colors_strat = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]

    means = [np.mean(results[k]) for k in strategy_names]
    stds = [np.std(results[k]) for k in strategy_names]

    bars = ax.bar(strategy_labels, means, yerr=stds, capsize=5,
                  color=colors_strat, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=means[0], color="gray", linestyle="--", alpha=0.5,
               label="Baseline level")
    ax.set_ylabel("Mean Logit Diff (Yes - No)")
    ax.set_title(f"Strategy Comparison ({model_name.capitalize()})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for j, (m, s_name) in enumerate(zip(means, strategy_names)):
        if j > 0:
            delta = m - means[0]
            ax.annotate(f"{delta:+.2f}", xy=(j, m + stds[j] + 0.05),
                        ha="center", fontsize=8,
                        color="green" if delta > 0 else "red")

    ax = axes[1]
    steer_means = [np.mean(results[f"steer_{s}x"]) for s in STEERING_SCALES]
    combined_means = [np.mean(results[f"combined_{s}x"]) for s in STEERING_SCALES]
    steer_stds = [np.std(results[f"steer_{s}x"]) for s in STEERING_SCALES]
    combined_stds = [np.std(results[f"combined_{s}x"]) for s in STEERING_SCALES]

    ax.errorbar(STEERING_SCALES, steer_means, yerr=steer_stds,
                fmt="o-", color="#3498db", capsize=4, label="Steering only")
    ax.errorbar(STEERING_SCALES, combined_means, yerr=combined_stds,
                fmt="s-", color="#2ecc71", capsize=4, label="Combined")
    ax.axhline(y=np.mean(results["baseline"]), color="gray", linestyle="--",
               alpha=0.5, label="Baseline")
    ax.axhline(y=np.mean(results["head_patch"]), color="#e74c3c", linestyle=":",
               alpha=0.5, label="Head patch")
    ax.set_xlabel("Steering Scale")
    ax.set_ylabel("Mean Logit Diff (Yes - No)")
    ax.set_title(f"Steering Scale Sweep ({model_name.capitalize()})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    flip_data = {}
    for key in strategy_names:
        flip_data[key] = sum(1 for ld in results[key] if ld > 0)
    flip_vals = list(flip_data.values())

    bars = ax.bar(strategy_labels, flip_vals, color=colors_strat, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Cases with LD > 0 (Yes wins)")
    ax.set_title(f"Detection Rate ({model_name.capitalize()}, out of {n_pairs})")
    ax.set_ylim(0, n_pairs + 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, flip_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(f"Phase 7: Circuit Intervention ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase7_intervention_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-model Figure 2: Per-case breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(n_pairs)
    width = 0.2
    ax.bar(x - 1.5*width, results["baseline"], width,
           label="Baseline", color="#95a5a6", alpha=0.8)
    ax.bar(x - 0.5*width, results["head_patch"], width,
           label="Head Patch", color="#e74c3c", alpha=0.8)
    ax.bar(x + 0.5*width, results[best_steer_key], width,
           label="Steer", color="#3498db", alpha=0.8)
    ax.bar(x + 1.5*width, results[best_combined_key], width,
           label="Combined", color="#2ecc71", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Case Index")
    ax.set_ylabel("Logit Diff (Yes - No)")
    ax.set_title(f"Per-Case Comparison ({model_name.capitalize()})")
    ax.legend(fontsize=7)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    deltas_hp = [h - b for h, b in zip(results["head_patch"], results["baseline"])]
    deltas_cb = [c - b for c, b in zip(results[best_combined_key], results["baseline"])]
    ax.bar(x - width/2, deltas_hp, width,
           label="Head Patch delta", color="#e74c3c", alpha=0.7)
    ax.bar(x + width/2, deltas_cb, width,
           label="Combined delta", color="#2ecc71", alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Case Index")
    ax.set_ylabel("LD Improvement vs Baseline")
    ax.set_title(f"Per-Case Improvement ({model_name.capitalize()})")
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 7: Per-Case Effects ({model_name.capitalize()}) [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"phase7_per_case_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\n  [{model_name}] RESULTS:")
    for key, label in [("baseline", "Baseline"),
                       ("head_patch", "Head Patch"),
                       (best_steer_key, f"Best Steering ({best_steer_key})"),
                       (best_combined_key, f"Best Combined ({best_combined_key})")]:
        m_val = np.mean(results[key])
        s_val = np.std(results[key])
        flips = sum(1 for ld in results[key] if ld > 0)
        delta = m_val - np.mean(results["baseline"])
        print(f"    [{model_name}] {label}: Mean LD={m_val:.3f} +/- {s_val:.3f} "
              f"({delta:+.3f}), Detection={flips}/{n_pairs}")

    result = {
        "results": results,
        "best_steer_key": best_steer_key,
        "best_combined_key": best_combined_key,
        "baseline_mean": np.mean(results["baseline"]),
        "head_patch_mean": np.mean(results["head_patch"]),
        "best_steer_mean": np.mean(results[best_steer_key]),
        "best_combined_mean": np.mean(results[best_combined_key]),
        "baseline_flips": sum(1 for ld in results["baseline"] if ld > 0),
        "combined_flips": sum(1 for ld in results[best_combined_key] if ld > 0),
        "n_pairs": n_pairs,
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

    # Panel 1: Baseline vs best combined by model
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.3
    baseline_means = [all_results[m]["baseline_mean"] for m in model_names]
    combined_means = [all_results[m]["best_combined_mean"] for m in model_names]
    ax.bar(x - width/2, baseline_means, width, label="Baseline",
           color="#95a5a6", alpha=0.8)
    ax.bar(x + width/2, combined_means, width, label="Best Combined",
           color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in model_names])
    ax.set_ylabel("Mean Logit Diff (Yes - No)")
    ax.set_title("Baseline vs Best Intervention")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Improvement (combined - baseline) by model
    ax = axes[1]
    improvements = [all_results[m]["best_combined_mean"] - all_results[m]["baseline_mean"]
                    for m in model_names]
    bar_colors = [model_colors.get(m, "#333") for m in model_names]
    bars = ax.bar([m.capitalize() for m in model_names], improvements,
                  color=bar_colors, alpha=0.8)
    ax.set_ylabel("LD Improvement (Combined - Baseline)")
    ax.set_title("Intervention Improvement by Model")
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{imp:+.2f}", ha="center", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Detection rate comparison
    ax = axes[2]
    baseline_flips = [all_results[m]["baseline_flips"] for m in model_names]
    combined_flips = [all_results[m]["combined_flips"] for m in model_names]
    ax.bar(x - width/2, baseline_flips, width, label="Baseline",
           color="#95a5a6", alpha=0.8)
    ax.bar(x + width/2, combined_flips, width, label="Best Combined",
           color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in model_names])
    ax.set_ylabel("Cases with LD > 0")
    ax.set_title("Detection Rate by Model")
    ax.legend(fontsize=8)
    n_pairs = max(all_results[m]["n_pairs"] for m in model_names)
    ax.set_ylim(0, n_pairs + 0.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Phase 7: Cross-Model Intervention Comparison [{dataset_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_dir / "phase7_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'phase7_comparison.png'}")


def load_saved_results(model_name, output_dir):
    """Try to load previously saved results for a model."""
    path = output_dir / f"{model_name}_intervention_results.pt"
    if path.exists():
        data = torch.load(path, weights_only=False)
        results = data["results"]
        steering_scales = data.get("steering_scales", STEERING_SCALES)
        best_steer_key = max([f"steer_{s}x" for s in steering_scales],
                             key=lambda k: np.mean(results.get(k, [0])))
        best_combined_key = max([f"combined_{s}x" for s in steering_scales],
                                key=lambda k: np.mean(results.get(k, [0])))
        return {
            "results": results,
            "best_steer_key": best_steer_key,
            "best_combined_key": best_combined_key,
            "baseline_mean": np.mean(results["baseline"]),
            "head_patch_mean": np.mean(results["head_patch"]),
            "best_steer_mean": np.mean(results[best_steer_key]),
            "best_combined_mean": np.mean(results[best_combined_key]),
            "baseline_flips": sum(1 for ld in results["baseline"] if ld > 0),
            "combined_flips": sum(1 for ld in results[best_combined_key] if ld > 0),
            "n_pairs": len(results["baseline"]),
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 7: Circuit-Informed Intervention")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="all", help="Which model to run (default: all)")
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 7: Circuit-Informed Intervention ({', '.join(m.capitalize() for m in models_to_run)}) [{dataset_label}]")
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
                model_name, death_me, life_not_me, device, output_dir, dataset_label, dataset_name)
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
    print("PHASE 7 COMPLETE")
    print("=" * 70)
    for model_name, r in all_results.items():
        print(f"\n{model_name.capitalize()}:")
        print(f"  Baseline mean LD: {r['baseline_mean']:.3f}")
        print(f"  Head Patch mean LD: {r['head_patch_mean']:.3f}")
        print(f"  Best Steer mean LD: {r['best_steer_mean']:.3f} ({r['best_steer_key']})")
        print(f"  Best Combined mean LD: {r['best_combined_mean']:.3f} ({r['best_combined_key']})")
        print(f"  Detection rate: baseline={r['baseline_flips']}/{r['n_pairs']}, "
              f"combined={r['combined_flips']}/{r['n_pairs']}")

    print(f"\nPhase 7 complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
