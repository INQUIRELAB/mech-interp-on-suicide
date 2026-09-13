#!/usr/bin/env python
"""Phase 8: Sparse Autoencoder Analysis.

Investigates whether implicit suicide detection features exist in
SUPERPOSITION within the model's MLP representations. Phase 6 found that
the safety signal is distributed across thousands of neurons with no
"smoking gun." SAEs can decompose these distributed representations into
interpretable monosemantic features.

Pipeline:
1. Collect MLP activations from Death-Me, Death-Not-Me, Life-Not-Me examples
2. Train a sparse autoencoder (overcomplete dictionary) on mixed activations
3. Identify SAE features that differentially activate on Death-Me vs controls
4. Causally validate top features by clamping them during inference
5. Analyze feature geometry (clustering, cosine similarity to safety direction)

Runs on Llama-3.1-8B-Instruct (primary model) at target layers identified
by Phase 6 as having the strongest differential signal.

Output:
    results/phase8/{model}_sae_weights_L{layer}.pt       - Trained SAE weights
    results/phase8/{model}_sae_features_L{layer}.pt      - Feature analysis data
    results/phase8/phase8_feature_discovery_{model}.png   - Feature activation patterns
    results/phase8/phase8_feature_geometry_{model}.png    - Feature space geometry
    results/phase8/phase8_causal_validation_{model}.png   - Causal validation results
    results/phase8/phase8_superposition_{model}.png       - Superposition evidence
"""

import sys
import gc
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_COLLECT = 40          # Examples per category for activation collection
N_CAUSAL = 8            # Examples for causal validation
MAX_TOKENS = 256
MODELS = ["llama"]      # Primary model only (extend to ["llama", "qwen", "mistral"])
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase8"

# SAE hyperparameters
SAE_EXPANSION = 8       # d_sae = SAE_EXPANSION * d_model (overcomplete factor)
SAE_L1_COEFF = 5e-3     # Sparsity penalty weight
SAE_LR = 1e-3
SAE_EPOCHS = 200
SAE_BATCH_SIZE = 32
STEERING_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]  # Multiple scales for steering sweep

# Layers to analyze (will auto-detect from Phase 6 results if available)
TARGET_LAYERS = None     # Set to e.g. [5, 10, 15] to override auto-detection
N_TARGET_LAYERS = 3      # How many top layers to pick from Phase 6


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """Overcomplete sparse autoencoder for decomposing MLP activations.

    Architecture:
        Encoder: x -> W_enc @ (x - b_dec) + b_enc -> ReLU -> z  (sparse code)
        Decoder: z -> W_dec @ z + b_dec -> x_hat  (reconstruction)

    The decoder columns (W_dec[:, i]) are the learned feature directions.
    The hidden activations z are the feature coefficients.
    L1 penalty on z encourages sparsity (monosemantic features).
    """

    def __init__(self, d_input, d_sae):
        super().__init__()
        self.d_input = d_input
        self.d_sae = d_sae

        # Encoder
        self.W_enc = nn.Parameter(torch.randn(d_sae, d_input) * (1.0 / d_input ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder (tied init with encoder transpose, but untied for training)
        self.W_dec = nn.Parameter(torch.randn(d_input, d_sae) * (1.0 / d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True) + 1e-8

    def encode(self, x):
        """Encode input to sparse feature activations."""
        z = torch.relu(self.W_enc @ (x - self.b_dec).unsqueeze(-1) + self.b_enc.unsqueeze(-1))
        return z.squeeze(-1)

    def decode(self, z):
        """Decode sparse features back to input space."""
        return self.W_dec @ z.unsqueeze(-1) + self.b_dec.unsqueeze(-1)

    def forward(self, x):
        """Full forward pass. x: [batch, d_input]"""
        # Encode
        pre_act = (x - self.b_dec) @ self.W_enc.T + self.b_enc  # [batch, d_sae]
        z = torch.relu(pre_act)  # sparse activations

        # Decode
        x_hat = z @ self.W_dec.T + self.b_dec  # [batch, d_input]

        return x_hat, z

    def loss(self, x, x_hat, z):
        """Compute reconstruction + sparsity loss."""
        recon_loss = ((x - x_hat) ** 2).mean()
        l1_loss = z.abs().mean()
        return recon_loss + SAE_L1_COEFF * l1_loss, recon_loss, l1_loss


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_activations_by_category(model, tokenizer, categories, target_layers,
                                     n_cases, device):
    """Collect MLP activations for each category at target layers.

    Returns:
        activations: {layer: {category_name: [n_cases, d_model] tensor}}
    """
    activations = {layer: defaultdict(list) for layer in target_layers}

    for cat_name, examples in categories.items():
        n = min(n_cases, len(examples))
        print(f"  Collecting {n} activations for '{cat_name}'...")

        for i in range(n):
            text = get_text(examples[i])
            prompt = build_prompt(text)
            tokens = tokenizer(
                prompt, return_tensors="pt", max_length=MAX_TOKENS,
                truncation=True, padding="max_length",
            )["input_ids"].to(device)

            _, cache = model.run_with_cache(tokens)

            for layer in target_layers:
                key = f"blocks.{layer}.hook_mlp_out"
                if key in cache:
                    # Last token position, float32
                    act = cache[key][0, -1, :].float().cpu()
                    activations[layer][cat_name].append(act)

            if (i + 1) % 10 == 0:
                print(f"    {cat_name}: {i+1}/{n} done")

    # Stack into tensors
    for layer in target_layers:
        for cat_name in activations[layer]:
            activations[layer][cat_name] = torch.stack(activations[layer][cat_name])

    return activations


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------

def train_sae(activations_dict, d_model, layer, device="cpu"):
    """Train sparse autoencoder on mixed activations from all categories.

    Args:
        activations_dict: {category_name: [n, d_model] tensor}
        d_model: model hidden dimension
        layer: layer index (for logging)
        device: training device

    Returns:
        Trained SparseAutoencoder, training history
    """
    # Combine all categories for training
    all_acts = torch.cat(list(activations_dict.values()), dim=0)
    n_total = all_acts.shape[0]

    # Normalize (zero mean, unit variance per feature)
    act_mean = all_acts.mean(dim=0)
    act_std = all_acts.std(dim=0).clamp(min=1e-8)
    all_acts_norm = (all_acts - act_mean) / act_std

    d_sae = SAE_EXPANSION * d_model
    sae = SparseAutoencoder(d_model, d_sae).to(device)

    optimizer = optim.Adam(sae.parameters(), lr=SAE_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SAE_EPOCHS)

    history = {"total": [], "recon": [], "l1": [], "l0": []}

    print(f"\n  Training SAE for Layer {layer}: d_input={d_model}, d_sae={d_sae}, "
          f"n_samples={n_total}")

    all_acts_norm = all_acts_norm.to(device)

    for epoch in range(SAE_EPOCHS):
        # Shuffle
        perm = torch.randperm(n_total)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for start in range(0, n_total, SAE_BATCH_SIZE):
            batch_idx = perm[start:start + SAE_BATCH_SIZE]
            batch = all_acts_norm[batch_idx]

            x_hat, z = sae(batch)
            loss, recon, l1 = sae.loss(batch, x_hat, z)

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

            optimizer.step()

            # Normalize decoder columns to unit norm (prevents feature collapse)
            with torch.no_grad():
                sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True) + 1e-8

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_l1 += l1.item()
            epoch_l0 += (z > 0).float().sum(dim=1).mean().item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_l0 = epoch_l0 / n_batches

        history["total"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["l1"].append(avg_l1)
        history["l0"].append(avg_l0)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{SAE_EPOCHS}: loss={avg_loss:.5f} "
                  f"recon={avg_recon:.5f} L1={avg_l1:.5f} L0={avg_l0:.1f}")

    sae.cpu()
    return sae, history, act_mean, act_std


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

def analyze_features(sae, activations_dict, act_mean, act_std, layer, device="cpu"):
    """Analyze which SAE features are selective for each category.

    Returns:
        feature_stats: dict with per-feature activation statistics
    """
    sae.eval().to(device)

    # Compute feature activations per category
    feature_acts = {}
    for cat_name, acts in activations_dict.items():
        acts_norm = ((acts - act_mean) / act_std).to(device)
        with torch.no_grad():
            _, z = sae(acts_norm)  # [n, d_sae]
        feature_acts[cat_name] = z.cpu()

    d_sae = sae.d_sae

    # Per-feature statistics
    mean_act = {}
    frac_active = {}
    for cat_name, z in feature_acts.items():
        mean_act[cat_name] = z.mean(dim=0)            # [d_sae]
        frac_active[cat_name] = (z > 0).float().mean(dim=0)  # [d_sae]

    # Differential: Death-Me vs Life-Not-Me
    dm_acts = mean_act.get("Death-Me", torch.zeros(d_sae))
    lnm_acts = mean_act.get("Life-Not-Me", torch.zeros(d_sae))
    dnm_acts = mean_act.get("Death-Not-Me", torch.zeros(d_sae))

    # Features that activate MORE on Death-Me than Life-Not-Me
    diff_dm_lnm = dm_acts - lnm_acts                    # positive = suicide-selective
    diff_dm_dnm = dm_acts - dnm_acts                    # positive = implicit-selective

    # Selectivity ratio: how much more does feature fire on implicit vs explicit?
    # Filter to avoid divide-by-zero when LNM_active is near 0
    lnm_frac_active = frac_active.get("Life-Not-Me", torch.zeros(d_sae))
    dnm_frac_active = frac_active.get("Death-Not-Me", torch.zeros(d_sae))
    dm_frac_active = frac_active.get("Death-Me", torch.zeros(d_sae))

    # Only compute selectivity for features with meaningful activation (>=5% on at least one category)
    min_active_threshold = 0.05
    valid_mask = (torch.maximum(dm_frac_active, torch.maximum(lnm_frac_active, dnm_frac_active)) >= min_active_threshold)

    selectivity = torch.full_like(dm_acts, float('nan'))
    selectivity[valid_mask] = dm_acts[valid_mask] / (dnm_acts[valid_mask] + 1e-8)

    # Find top features by different criteria
    # 1. Most suicide-selective (Death-Me > Life-Not-Me)
    top_suicide = diff_dm_lnm.topk(20)
    # 2. Most implicit-selective (Death-Me > Death-Not-Me)
    top_implicit = diff_dm_dnm.topk(20)
    # 3. Most active on Death-Me overall
    top_active_dm = dm_acts.topk(20)

    # Feature sparsity: what fraction of inputs activate each feature?
    overall_sparsity = torch.cat([z for z in feature_acts.values()], dim=0)
    frac_active_overall = (overall_sparsity > 0).float().mean(dim=0)

    # Dead features (never activate)
    n_dead = (frac_active_overall == 0).sum().item()
    # Ultra-sparse features (activate on <5% of inputs)
    n_sparse = ((frac_active_overall > 0) & (frac_active_overall < 0.05)).sum().item()

    print(f"\n  Layer {layer} Feature Statistics:")
    print(f"    Total features: {d_sae}")
    print(f"    Dead features (never active): {n_dead} ({100*n_dead/d_sae:.1f}%)")
    print(f"    Ultra-sparse (<5% active): {n_sparse} ({100*n_sparse/d_sae:.1f}%)")
    print(f"    Mean L0 (features active per input): {(overall_sparsity > 0).float().sum(dim=1).mean():.1f}")

    print(f"\n    Top 10 Suicide-Selective Features (Death-Me - Life-Not-Me):")
    for i in range(10):
        idx = top_suicide.indices[i].item()
        val = top_suicide.values[i].item()
        sel = selectivity[idx].item()
        dm_frac = frac_active.get("Death-Me", torch.zeros(d_sae))[idx].item()
        lnm_frac = frac_active.get("Life-Not-Me", torch.zeros(d_sae))[idx].item()
        sel_str = f"{sel:.2f}" if not torch.isnan(torch.tensor(sel)) else "N/A"
        print(f"      F{idx:5d}: diff={val:+.4f}  sel={sel_str}  "
              f"DM_active={dm_frac:.0%}  LNM_active={lnm_frac:.0%}")

    print(f"\n    Top 10 Implicit-Selective Features (Death-Me - Death-Not-Me):")
    for i in range(10):
        idx = top_implicit.indices[i].item()
        val = top_implicit.values[i].item()
        dm_frac = frac_active.get("Death-Me", torch.zeros(d_sae))[idx].item()
        dnm_frac = frac_active.get("Death-Not-Me", torch.zeros(d_sae))[idx].item()
        print(f"      F{idx:5d}: diff={val:+.4f}  "
              f"DM_active={dm_frac:.0%}  DNM_active={dnm_frac:.0%}")

    sae.cpu()
    return {
        "feature_acts": feature_acts,
        "mean_act": mean_act,
        "frac_active": frac_active,
        "diff_dm_lnm": diff_dm_lnm,
        "diff_dm_dnm": diff_dm_dnm,
        "selectivity": selectivity,
        "top_suicide": top_suicide,
        "top_implicit": top_implicit,
        "top_active_dm": top_active_dm,
        "frac_active_overall": frac_active_overall,
        "n_dead": n_dead,
        "n_sparse": n_sparse,
    }


# ---------------------------------------------------------------------------
# Causal validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def causal_validate_sae_features(model, tokenizer, sae, act_mean, act_std,
                                  top_feature_indices, target_layer,
                                  death_me, life_not_me, yes_id, no_id, device,
                                  steering_scales=STEERING_SCALES):
    """Validate top SAE features by steering MLP activations across multiple scales.

    For each case and scale:
    1. Run corrupt (Life-Not-Me) input
    2. Encode MLP output at target layer through SAE
    3. Boost top suicide-selective features by their mean Death-Me activation * scale
    4. Decode back and patch into the model
    5. Measure change in logit difference

    Returns:
        baseline_lds, steered_lds_by_scale: {scale: [ld_list]} for each scale
    """
    n_cases = min(N_CAUSAL, len(death_me), len(life_not_me))
    baseline_lds = []
    steered_lds_by_scale = {s: [] for s in steering_scales}

    sae_device = device
    sae.to(sae_device).eval()

    for i in range(n_cases):
        clean_tokens, corrupt_tokens = tokenize_pair(
            tokenizer,
            build_prompt(get_text(death_me[i])),
            build_prompt(get_text(life_not_me[i])),
            max_length=MAX_TOKENS, device=device,
        )

        # Get clean activations to compute target feature values
        _, clean_cache = model.run_with_cache(clean_tokens)
        clean_mlp = clean_cache[f"blocks.{target_layer}.hook_mlp_out"][0, -1, :].float()
        clean_norm = ((clean_mlp.cpu() - act_mean) / act_std).to(sae_device)
        _, clean_z = sae(clean_norm.unsqueeze(0))
        clean_z = clean_z.squeeze(0)

        # Corrupt baseline
        corrupt_logits = model(corrupt_tokens)
        ld_corrupt = (corrupt_logits[0, -1, yes_id] - corrupt_logits[0, -1, no_id]).float().item()
        baseline_lds.append(ld_corrupt)

        # Steering: try multiple scales
        for scale in steering_scales:
            def make_sae_steer_hook(sae_model, mean, std, feat_indices, clean_feat_vals, sc, dev):
                def hook_fn(output, hook):
                    out = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                    # Get last-position activation
                    last_act = out[0, -1, :].float()
                    # Normalize
                    last_norm = ((last_act.cpu() - mean) / std).to(dev)
                    # Encode
                    _, z = sae_model(last_norm.unsqueeze(0))
                    z = z.squeeze(0)
                    # Boost target features to their clean (Death-Me) values * scale
                    for fi in feat_indices:
                        z[fi] = clean_feat_vals[fi] * sc
                    # Decode back
                    x_hat = (z @ sae_model.W_dec.T + sae_model.b_dec)
                    # Denormalize
                    reconstructed = (x_hat.cpu() * std + mean).to(out.device)
                    out[0, -1, :] = reconstructed.to(out.dtype)
                    return out
                return hook_fn

            hook_name = f"blocks.{target_layer}.hook_mlp_out"
            model.add_hook(
                hook_name,
                make_sae_steer_hook(sae, act_mean, act_std, top_feature_indices,
                                    clean_z, scale, sae_device),
            )
            steered_logits = model(corrupt_tokens)
            model.remove_hooks()

            ld_steered = (steered_logits[0, -1, yes_id] - steered_logits[0, -1, no_id]).float().item()
            steered_lds_by_scale[scale].append(ld_steered)

        deltas_str = ", ".join([f"s{scale}={steered_lds_by_scale[scale][-1] - ld_corrupt:+.3f}"
                                for scale in steering_scales])
        print(f"    Case {i+1}: baseline LD={ld_corrupt:.3f}, {deltas_str}")

    sae.cpu()
    return baseline_lds, steered_lds_by_scale


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_feature_discovery(feature_stats, layer, model_name, history, output_dir,
                           dataset_label=""):
    """Plot feature activation patterns and training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Training curves
    ax = axes[0, 0]
    epochs = range(1, len(history["total"]) + 1)
    ax.plot(epochs, history["recon"], label="Reconstruction", color="#e74c3c")
    ax.plot(epochs, history["l1"], label="L1 sparsity", color="#3498db")
    ax.plot(epochs, history["total"], label="Total", color="#2c3e50", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SAE Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.plot(epochs, history["l0"], "--", color="#27ae60", alpha=0.7, label="L0")
    ax2.set_ylabel("L0 (active features)", color="#27ae60")
    ax2.legend(fontsize=8, loc="center right")

    # Panel 2: Feature activation by category
    ax = axes[0, 1]
    cats = list(feature_stats["mean_act"].keys())
    cat_colors = {"Death-Me": "#e74c3c", "Death-Not-Me": "#f39c12", "Life-Not-Me": "#3498db"}

    # Show top 20 suicide-selective features
    top_idx = feature_stats["top_suicide"].indices[:20].numpy()
    x_pos = np.arange(len(top_idx))
    width = 0.25
    for ci, cat in enumerate(cats):
        vals = feature_stats["mean_act"][cat][top_idx].numpy()
        ax.bar(x_pos + ci * width, vals, width,
               label=cat, color=cat_colors.get(cat, "#999"), alpha=0.8)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Top 20 Suicide-Selective Features")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f"F{i}" for i in top_idx], rotation=45, fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Differential histogram
    ax = axes[1, 0]
    diff = feature_stats["diff_dm_lnm"].numpy()
    alive_mask = feature_stats["frac_active_overall"].numpy() > 0
    diff_alive = diff[alive_mask]

    ax.hist(diff_alive, bins=80, color="#95a5a6", alpha=0.7, edgecolor="none")
    # Mark top features
    top_vals = diff[feature_stats["top_suicide"].indices[:10].numpy()]
    for v in top_vals:
        ax.axvline(v, color="#e74c3c", alpha=0.5, linewidth=1)
    ax.set_xlabel("Mean Activation Difference (Death-Me - Life-Not-Me)")
    ax.set_ylabel("Count")
    ax.set_title("Feature Differential Distribution")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 4: Sparsity distribution
    ax = axes[1, 1]
    frac = feature_stats["frac_active_overall"].numpy()
    frac_nonzero = frac[frac > 0]
    ax.hist(frac_nonzero, bins=50, color="#9b59b6", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Fraction of Inputs Activating Feature")
    ax.set_ylabel("Count")
    ax.set_title(f"Feature Sparsity (dead={feature_stats['n_dead']}, "
                 f"ultra-sparse={feature_stats['n_sparse']})")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.05, color="#e74c3c", linestyle="--", alpha=0.5, label="5% threshold")
    ax.legend(fontsize=8)

    title = f"Phase 8: SAE Feature Discovery — {model_name.capitalize()} Layer {layer}"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = output_dir / f"phase8_feature_discovery_{model_name}_L{layer}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_geometry(feature_stats, sae, layer, model_name, output_dir,
                          dataset_label=""):
    """Plot feature space geometry: cosine similarity, clustering."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Cosine similarity between top suicide and implicit features
    ax = axes[0]
    W_dec = sae.W_dec.data  # [d_input, d_sae]
    top_suicide_idx = feature_stats["top_suicide"].indices[:15].numpy()
    top_implicit_idx = feature_stats["top_implicit"].indices[:15].numpy()

    # Decoder direction cosine similarity matrix
    all_idx = np.unique(np.concatenate([top_suicide_idx, top_implicit_idx]))
    dirs = W_dec[:, all_idx].T  # [n_features, d_input]
    dirs_norm = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (dirs_norm @ dirs_norm.T).numpy()

    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_idx)))
    ax.set_xticklabels([f"F{i}" for i in all_idx], rotation=90, fontsize=6)
    ax.set_yticks(range(len(all_idx)))
    ax.set_yticklabels([f"F{i}" for i in all_idx], fontsize=6)
    ax.set_title("Feature Direction Cosine Similarity")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Panel 2: Feature activation scatter (Death-Me vs Life-Not-Me)
    ax = axes[1]
    dm_mean = feature_stats["mean_act"].get("Death-Me", torch.zeros(sae.d_sae)).numpy()
    lnm_mean = feature_stats["mean_act"].get("Life-Not-Me", torch.zeros(sae.d_sae)).numpy()
    alive = feature_stats["frac_active_overall"].numpy() > 0

    ax.scatter(lnm_mean[alive], dm_mean[alive], alpha=0.15, s=8, c="#95a5a6", label="All")
    ax.scatter(lnm_mean[top_suicide_idx], dm_mean[top_suicide_idx],
               alpha=0.9, s=40, c="#e74c3c", marker="^", zorder=5,
               label="Top suicide-selective")
    ax.scatter(lnm_mean[top_implicit_idx], dm_mean[top_implicit_idx],
               alpha=0.9, s=40, c="#f39c12", marker="s", zorder=5,
               label="Top implicit-selective")

    lim = max(dm_mean[alive].max(), lnm_mean[alive].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("Mean Activation on Life-Not-Me")
    ax.set_ylabel("Mean Activation on Death-Me")
    ax.set_title("Feature Activation: Death-Me vs Life-Not-Me")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Implicit selectivity vs suicide selectivity
    ax = axes[2]
    diff_suicide = feature_stats["diff_dm_lnm"].numpy()
    diff_implicit = feature_stats["diff_dm_dnm"].numpy()

    ax.scatter(diff_suicide[alive], diff_implicit[alive],
               alpha=0.15, s=8, c="#95a5a6")
    ax.scatter(diff_suicide[top_suicide_idx], diff_implicit[top_suicide_idx],
               alpha=0.9, s=40, c="#e74c3c", marker="^", zorder=5,
               label="Top suicide-selective")
    ax.scatter(diff_suicide[top_implicit_idx], diff_implicit[top_implicit_idx],
               alpha=0.9, s=40, c="#f39c12", marker="s", zorder=5,
               label="Top implicit-selective")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Suicide Selectivity (DM - LNM)")
    ax.set_ylabel("Implicit Selectivity (DM - DNM)")
    ax.set_title("Feature Selectivity Space")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    title = f"Phase 8: Feature Geometry — {model_name.capitalize()} Layer {layer}"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = output_dir / f"phase8_feature_geometry_{model_name}_L{layer}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_causal_validation(baseline_lds, steered_lds_by_scale, layer, model_name, output_dir,
                           dataset_label=""):
    """Plot causal validation results with steering scale sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n = len(baseline_lds)
    steering_scales = sorted(steered_lds_by_scale.keys())

    # Panel 1: Side-by-side bars (baseline vs best scale)
    ax = axes[0]
    # Find best scale
    best_scale = max(steering_scales, key=lambda s: np.mean(steered_lds_by_scale[s]))
    steered_lds = steered_lds_by_scale[best_scale]

    x = np.arange(n)
    width = 0.35
    ax.bar(x - width/2, baseline_lds, width, label="Corrupt baseline",
           color="#95a5a6", alpha=0.8)
    ax.bar(x + width/2, steered_lds, width, label=f"SAE-steered ({best_scale}x)",
           color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Case Index")
    ax.set_ylabel("Logit Diff (Yes - No)")
    ax.set_title(f"SAE Feature Steering Effect (best scale={best_scale}x)")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Steering scale sweep
    ax = axes[1]
    scale_means = [np.mean(steered_lds_by_scale[s]) for s in steering_scales]
    scale_stds = [np.std(steered_lds_by_scale[s]) for s in steering_scales]
    ax.errorbar(steering_scales, scale_means, yerr=scale_stds,
                fmt="o-", color="#2ecc71", capsize=4, label="SAE steering")
    ax.axhline(y=np.mean(baseline_lds), color="gray", linestyle="--",
               alpha=0.5, label="Baseline")
    ax.set_xlabel("Steering Scale")
    ax.set_ylabel("Mean Logit Diff (Yes - No)")
    ax.set_title("Steering Scale Sweep")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Summary statistics
    ax = axes[2]
    deltas_by_scale = {s: [bld - b for bld, b in zip(steered_lds_by_scale[s], baseline_lds)]
                        for s in steering_scales}
    mean_deltas = [np.mean(deltas_by_scale[s]) for s in steering_scales]
    flips_by_scale = {s: sum(1 for b, st in zip(baseline_lds, steered_lds_by_scale[s]) if b < 0 and st > 0)
                      for s in steering_scales}

    # Compare with Phase 6 neuron patching (if available)
    labels = [f"SAE {s}x" for s in steering_scales]
    values = mean_deltas
    colors = ["#2ecc71"] * len(steering_scales)

    phase6_path = PROJECT_ROOT / "results" / "phase6" / f"{model_name}_neuron_activations.pt"
    if phase6_path.exists():
        p6 = torch.load(phase6_path, weights_only=False)
        bl6 = p6.get("causal_baseline_lds", [])
        pl6 = p6.get("causal_patched_lds", [])
        if bl6 and pl6:
            p6_delta = np.mean([p - b for p, b in zip(pl6, bl6)])
            labels.append("Phase 6\nNeuron Patch")
            values.append(p6_delta)
            colors.append("#9b59b6")

    bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean LD Shift")
    ax.set_title("SAE vs Neuron Patching")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add flip counts
    for i, (s, lbl) in enumerate(zip(steering_scales, labels)):
        flips = flips_by_scale[s]
        ax.text(i, min(values) - 0.05, f"{flips}/{n}", ha="center", fontsize=8)

    title = f"Phase 8: Causal Validation — {model_name.capitalize()} Layer {layer}"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = output_dir / f"phase8_causal_validation_{model_name}_L{layer}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_superposition_evidence(feature_stats, sae, layer, model_name, output_dir,
                                dataset_label=""):
    """Plot evidence for/against superposition of safety features."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    W_dec = sae.W_dec.data  # [d_input, d_sae]
    d_sae = sae.d_sae
    alive_mask = feature_stats["frac_active_overall"] > 0

    # Panel 1: Number of features encoding safety info vs neuron-level
    ax = axes[0]
    # How many features have positive suicide selectivity?
    diff = feature_stats["diff_dm_lnm"]
    n_positive = (diff > 0).sum().item()
    n_strong = (diff > diff.std()).sum().item()
    n_very_strong = (diff > 2 * diff.std()).sum().item()

    categories = ["Any\npositive", "> 1 std", "> 2 std"]
    counts = [n_positive, n_strong, n_very_strong]
    bars = ax.bar(categories, counts, color=["#3498db", "#e74c3c", "#c0392b"], alpha=0.8)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(c), ha="center", fontsize=10)
    ax.set_ylabel("Number of SAE Features")
    ax.set_title("Suicide-Selective Features Found")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Feature direction alignment with raw safety direction
    ax = axes[1]
    # Compute raw safety direction (mean DM act - mean LNM act in original space)
    dm_acts = feature_stats["feature_acts"].get("Death-Me")
    lnm_acts = feature_stats["feature_acts"].get("Life-Not-Me")
    if dm_acts is not None and lnm_acts is not None:
        # Project back through decoder to get original-space directions
        top_feat_idx = feature_stats["top_suicide"].indices[:20]
        feat_dirs = W_dec[:, top_feat_idx].T  # [20, d_input]
        feat_dirs_norm = feat_dirs / (feat_dirs.norm(dim=1, keepdim=True) + 1e-8)

        # Pairwise cosines between top features
        pairwise_cos = (feat_dirs_norm @ feat_dirs_norm.T).numpy()
        off_diag = pairwise_cos[np.triu_indices(len(top_feat_idx), k=1)]

        ax.hist(off_diag, bins=30, color="#e74c3c", alpha=0.7, edgecolor="none")
        ax.axvline(np.mean(off_diag), color="black", linewidth=2,
                   label=f"Mean={np.mean(off_diag):.3f}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Pairwise Cos-Sim of Top 20 Safety Features")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)

    # Panel 3: Superposition index
    # If features are in superposition, more SAE features than neurons should encode safety
    ax = axes[2]
    # Count neurons with positive differential (from raw activations)
    # vs SAE features with positive differential
    n_neurons_positive = 0
    n_features_positive = (diff > diff.std()).sum().item()

    # Estimate neurons from Phase 6 if available
    phase6_path = PROJECT_ROOT / "results" / "phase6" / f"{model_name}_neuron_activations.pt"
    if phase6_path.exists():
        p6 = torch.load(phase6_path, weights_only=False)
        diff_dm_p6 = p6.get("diff_dm")
        if diff_dm_p6 is not None:
            norm_p6 = p6.get("norm_dm", torch.ones(diff_dm_p6.shape[0]))
            norm_diff_p6 = diff_dm_p6 / (norm_p6.unsqueeze(1) + 1e-8)
            flat_p6 = norm_diff_p6.flatten()
            n_neurons_positive = (flat_p6 > flat_p6.std()).sum().item()

    labels = ["Neurons\n(Phase 6)", "SAE Features\n(Phase 8)"]
    vals = [n_neurons_positive, n_features_positive]
    ax.bar(labels, vals, color=["#9b59b6", "#e74c3c"], alpha=0.8, width=0.5)
    for i, (l, v) in enumerate(zip(labels, vals)):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (> 1 std above mean)")
    ax.set_title("Superposition Evidence:\nSAE Features vs Raw Neurons")
    ax.grid(True, alpha=0.3, axis="y")

    title = f"Phase 8: Superposition Analysis — {model_name.capitalize()} Layer {layer}"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = output_dir / f"phase8_superposition_{model_name}_L{layer}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Auto-detect target layers from Phase 6
# ---------------------------------------------------------------------------

def detect_target_layers(model_name, n_layers_model):
    """Pick target layers from Phase 6 results or use heuristic."""
    phase6_path = PROJECT_ROOT / "results" / "phase6" / f"{model_name}_neuron_activations.pt"

    if phase6_path.exists():
        print(f"  Found Phase 6 results, auto-detecting target layers...")
        data = torch.load(phase6_path, weights_only=False)
        diff_dm = data["diff_dm"]
        norm_dm = data["norm_dm"]
        norm_diff = diff_dm.mean(dim=1) / (norm_dm + 1e-8)

        # Pick layers with highest normalized differential
        top_layers = norm_diff.topk(N_TARGET_LAYERS).indices.tolist()
        top_layers.sort()
        print(f"  Target layers (from Phase 6): {top_layers}")
        return top_layers
    else:
        # Heuristic: early, middle, late
        early = n_layers_model // 4
        mid = n_layers_model // 2
        late = int(n_layers_model * 0.85)
        layers = [early, mid, late]
        print(f"  No Phase 6 results found. Using heuristic layers: {layers}")
        return layers


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_single_model(model_name, categories, device, output_dir,
                         dataset_label=""):
    """Run Phase 8 SAE analysis for one model."""
    print(f"\n{'='*60}")
    print(f"[{model_name}] Phase 8: Sparse Autoencoder Analysis")
    print(f"{'='*60}")

    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Determine target layers
    target_layers = TARGET_LAYERS or detect_target_layers(model_name, n_layers)

    # Step 1: Collect activations
    print(f"\n--- Step 1: Collecting MLP activations at layers {target_layers} ---")
    activations = collect_activations_by_category(
        model, tokenizer, categories, target_layers, N_COLLECT, device)

    all_results = {}

    for layer in target_layers:
        print(f"\n{'~'*50}")
        print(f"[{model_name}] Analyzing Layer {layer}")
        print(f"{'~'*50}")

        acts_dict = activations[layer]

        # Step 2: Train SAE
        print(f"\n--- Step 2: Training Sparse Autoencoder ---")
        sae_device = "cuda" if torch.cuda.is_available() else "cpu"
        sae, history, act_mean, act_std = train_sae(
            acts_dict, d_model, layer, device=sae_device)

        # Save SAE weights
        torch.save({
            "state_dict": sae.state_dict(),
            "d_input": d_model,
            "d_sae": sae.d_sae,
            "act_mean": act_mean,
            "act_std": act_std,
            "history": history,
        }, output_dir / f"{model_name}_sae_weights_L{layer}.pt")

        # Step 3: Analyze features
        print(f"\n--- Step 3: Analyzing SAE features ---")
        feature_stats = analyze_features(
            sae, acts_dict, act_mean, act_std, layer, device=sae_device)

        # Step 4: Causal validation
        print(f"\n--- Step 4: Causal validation ---")
        top_feature_idx = feature_stats["top_suicide"].indices[:10].tolist()
        baseline_lds, steered_lds_by_scale = causal_validate_sae_features(
            model, tokenizer, sae, act_mean, act_std,
            top_feature_idx, layer,
            categories.get("Death-Me", []),
            categories.get("Life-Not-Me", []),
            yes_id, no_id, device,
        )

        # Save feature analysis data
        torch.save({
            "feature_stats": {
                k: v if isinstance(v, (int, float, str)) else
                   {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
                    for kk, vv in v.items()} if isinstance(v, dict) else
                   (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in feature_stats.items()
            },
            "baseline_lds": baseline_lds,
            "steered_lds_by_scale": steered_lds_by_scale,
            "target_layer": layer,
            "top_feature_indices": top_feature_idx,
        }, output_dir / f"{model_name}_sae_features_L{layer}.pt")

        # Step 5: Visualize
        print(f"\n--- Step 5: Generating visualizations ---")
        plot_feature_discovery(feature_stats, layer, model_name, history,
                               output_dir, dataset_label)
        plot_feature_geometry(feature_stats, sae, layer, model_name,
                              output_dir, dataset_label)
        plot_causal_validation(baseline_lds, steered_lds_by_scale, layer, model_name,
                               output_dir, dataset_label)
        plot_superposition_evidence(feature_stats, sae, layer, model_name,
                                    output_dir, dataset_label)

        # Find best scale for summary
        best_scale = max(steered_lds_by_scale.keys(),
                        key=lambda s: np.mean(steered_lds_by_scale[s]))
        best_steered = steered_lds_by_scale[best_scale]

        all_results[layer] = {
            "n_dead": feature_stats["n_dead"],
            "n_sparse": feature_stats["n_sparse"],
            "top_suicide_features": top_feature_idx,
            "causal_mean_delta": np.mean([s - b for s, b in zip(best_steered, baseline_lds)]),
            "causal_flips": sum(1 for b, s in zip(baseline_lds, best_steered) if b < 0 and s > 0),
            "causal_best_scale": best_scale,
            "causal_total": len(baseline_lds),
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"[{model_name}] Phase 8 RESULTS SUMMARY")
    print(f"{'='*60}")
    for layer, r in all_results.items():
        print(f"\n  Layer {layer}:")
        print(f"    Dead features: {r['n_dead']}")
        print(f"    Ultra-sparse features: {r['n_sparse']}")
        print(f"    Top suicide-selective features: {r['top_suicide_features'][:5]}...")
        print(f"    Causal mean LD shift: {r['causal_mean_delta']:+.3f}")
        print(f"    Causal flips (No->Yes): {r['causal_flips']}/{r['causal_total']}")

    # Free VRAM
    unload_model(model)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


def main():
    global TARGET_LAYERS, N_COLLECT, SAE_EXPANSION, SAE_EPOCHS, SAE_L1_COEFF

    parser = argparse.ArgumentParser(description="Phase 8: Sparse Autoencoder Analysis")
    parser.add_argument("--model", type=str, choices=["llama", "qwen", "mistral", "all"],
                        default="llama", help="Which model to run (default: llama)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (e.g., '5,15,28')")
    parser.add_argument("--n-collect", type=int, default=N_COLLECT,
                        help=f"Examples per category to collect (default: {N_COLLECT})")
    parser.add_argument("--sae-expansion", type=int, default=SAE_EXPANSION,
                        help=f"SAE overcomplete factor (default: {SAE_EXPANSION})")
    parser.add_argument("--sae-epochs", type=int, default=SAE_EPOCHS,
                        help=f"SAE training epochs (default: {SAE_EPOCHS})")
    parser.add_argument("--l1-coeff", type=float, default=SAE_L1_COEFF,
                        help=f"SAE L1 sparsity coefficient (default: {SAE_L1_COEFF})")
    add_dataset_arg(parser)
    args = parser.parse_args()

    # Apply overrides
    if args.layers:
        TARGET_LAYERS = [int(x) for x in args.layers.split(",")]
    N_COLLECT = args.n_collect
    SAE_EXPANSION = args.sae_expansion
    SAE_EPOCHS = args.sae_epochs
    SAE_L1_COEFF = args.l1_coeff

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)

    models_to_run = MODELS if args.model == "all" else [args.model]

    print("=" * 70)
    print(f"PHASE 8: Sparse Autoencoder Analysis")
    print(f"  Dataset: {dataset_label}")
    print(f"  Models: {', '.join(m.capitalize() for m in models_to_run)}")
    print(f"  SAE expansion: {SAE_EXPANSION}x | L1: {SAE_L1_COEFF} | Epochs: {SAE_EPOCHS}")
    print(f"  Examples per category: {N_COLLECT}")
    print("=" * 70)

    ensure_dir(output_dir)

    categories = load_dataset_by_category(dataset_name=dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in models_to_run:
        try:
            analyze_single_model(model_name, categories, device, output_dir,
                                 dataset_label)
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nPhase 8 complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
