#!/usr/bin/env python
"""Phase 12: SAE Feature Cards + Circuit Faithfulness.

Two publication-quality deliverables:

Part A — Feature Cards
  For each top suicide-selective SAE feature discovered in Phase 8,
  generate a one-page card showing:
    * mean activation per severity level (a/b/c/d) — ordinal signature
    * top-K activating examples with token-level activation highlights
    * cosine similarity to the "safety direction"
    * causal effect magnitude (from phase 8 causal validation)
  These cards follow the layout of public SAE feature browsers and make learned
  features legible to non-technical reviewers.

Part B — Circuit Faithfulness / Necessity / Minimality
  Using the top-K EAP circuit from Phase 9:
    faithfulness(k) = 1 − |LD_full − LD_only_circuit(k)| / |LD_full|
    necessity(k)    = |LD_full − LD_ablate_circuit(k)| / |LD_full|
    minimality      = smallest k such that faithfulness ≥ 0.9
  LD = logit(Yes) − logit(No). Tests whether our discovered circuit is
  both sufficient (restore it → behaviour returns) and necessary
  (ablate it → behaviour collapses).

Outputs:
    results/phase12/{dataset}/phase12_feature_card_L{L}_F{idx}.png  (one per feature)
    results/phase12/{dataset}/phase12_feature_severity_{model}.png  (all-features overview)
    results/phase12/{dataset}/phase12_faithfulness_{model}.png
    results/phase12/{dataset}/phase12_metrics_{model}.json
"""

import sys
import gc
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis_utils import (
    load_model_wrapped, unload_model, load_dataset_by_category, get_text,
    build_prompt, get_yes_no_ids, tokenize_pair, ensure_dir,
    add_dataset_arg, get_output_dir, DATASETS,
    load_umd_with_severity, UMD_LABEL_NAMES,
)

# Import the SAE class from phase 8
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
from phase8_sparse_autoencoder import SparseAutoencoder  # noqa: E402

MAX_TOKENS = 192
N_TOP_EXAMPLES = 6
N_TOP_FEATURES = 6
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase12"
MODELS = ["llama"]


# ---------------------------------------------------------------------------
# Feature cards
# ---------------------------------------------------------------------------

def load_sae(sae_path):
    """Load a trained SAE checkpoint saved by phase 8."""
    ckpt = torch.load(sae_path, weights_only=False)
    sae = SparseAutoencoder(ckpt["d_input"], ckpt["d_sae"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt["act_mean"], ckpt["act_std"]


@torch.no_grad()
def feature_activations_per_token(model, tokenizer, sae, act_mean, act_std,
                                  feature_idx, text, layer, device):
    """Return list of (token_string, activation_value) for a given feature on text."""
    tokens = tokenizer(build_prompt(text), return_tensors="pt",
                       max_length=MAX_TOKENS, truncation=True,
                       padding="max_length")["input_ids"].to(device)
    _, cache = model.run_with_cache(tokens)
    mlp = cache[f"blocks.{layer}.hook_mlp_out"][0].float().cpu()  # [seq, d]
    mlp_norm = (mlp - act_mean) / act_std
    sae_dev = next(sae.parameters()).device
    _, z = sae(mlp_norm.to(sae_dev))  # [seq, d_sae]
    z = z.detach().cpu()
    acts = z[:, feature_idx].numpy()
    token_strs = [tokenizer.decode([tid]) for tid in tokens[0].tolist()]
    return token_strs, acts


def render_feature_card(model, tokenizer, sae, act_mean, act_std,
                        feature_idx, layer, records_per_sev,
                        severity_means, device, out_path, model_name,
                        safety_direction=None, causal_delta=None):
    """Render one feature's card."""
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.2], hspace=0.45, wspace=0.25)

    # --- Title ---
    fig.suptitle(f"SAE Feature Card — {model_name.capitalize()} · L{layer} · F{feature_idx}",
                 fontsize=14, y=0.98)

    # --- A) Severity bars ---
    ax1 = fig.add_subplot(gs[0, 0])
    levels = ["a", "b", "c", "d"]
    vals = [severity_means.get(lv, 0.0) for lv in levels]
    colors = ["#2980b9", "#16a085", "#e67e22", "#c0392b"]
    ax1.bar([UMD_LABEL_NAMES[lv] for lv in levels], vals, color=colors,
            edgecolor="black", linewidth=0.6)
    ax1.set_title("Mean activation by severity", fontsize=11)
    ax1.tick_params(axis="x", labelsize=8, rotation=20)
    ax1.grid(axis="y", alpha=0.3)

    # --- B) Summary stats ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    lines = []
    if causal_delta is not None:
        lines.append(f"Causal effect (Δ logit diff): {causal_delta:+.3f}")
    if safety_direction is not None:
        lines.append(f"|cos θ| to safety direction: {safety_direction:.3f}")
    levels_activ = [(UMD_LABEL_NAMES[lv], severity_means.get(lv, 0.0)) for lv in levels]
    max_lbl, max_val = max(levels_activ, key=lambda t: t[1])
    lines.append(f"Most active on: {max_lbl}  (μ={max_val:.3f})")
    if vals[3] > 0 and vals[0] > 0:
        lines.append(f"Selectivity d/a: {vals[3] / (vals[0] + 1e-8):.1f}×")
    ax2.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
             fontsize=11, family="monospace")
    ax2.set_title("Summary", fontsize=11)

    # --- C) Top-K activating excerpts per severity ---
    # Collect candidate records across severities, compute max activation,
    # pick top-K and render with token highlights
    candidates = []
    for sev_label, recs in records_per_sev.items():
        for rec in recs[:10]:  # cap per level
            candidates.append((sev_label, rec))

    scored = []
    for sev_label, rec in candidates:
        ts, acts = feature_activations_per_token(
            model, tokenizer, sae, act_mean, act_std,
            feature_idx, rec["text"], layer, device,
        )
        scored.append((max(acts), sev_label, rec, ts, acts))
    scored.sort(key=lambda t: -t[0])
    top = scored[:N_TOP_EXAMPLES]

    ax3 = fig.add_subplot(gs[1:, :])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, N_TOP_EXAMPLES * 2.2)
    ax3.invert_yaxis()
    ax3.axis("off")
    ax3.set_title(f"Top-{N_TOP_EXAMPLES} activating excerpts (color intensity = activation)",
                  fontsize=11, loc="left")

    for row, (mx, sev_label, rec, ts, acts) in enumerate(top):
        y = row * 2.0 + 0.3
        ax3.text(0.0, y - 0.25,
                 f"[{sev_label}/{UMD_LABEL_NAMES.get(sev_label, sev_label)}]  "
                 f"user u{rec.get('user_id','?')}  |  max act = {mx:.2f}",
                 fontsize=9, weight="bold", color="#333333")
        # Highlight token-by-token, up to display width
        a = acts / (acts.max() + 1e-8)
        x = 0.0
        for tok, wt in zip(ts[-80:], a[-80:]):
            display = tok.replace("Ġ", " ").replace("▁", " ")\
                        .replace("\n", "\\n")[:12]
            ax3.text(x, y + 0.4, display,
                     fontsize=8, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.1",
                               facecolor=(0.8, 0.1, 0.1, float(wt)),
                               edgecolor="none"))
            x += 0.4 + 0.04 * len(display)
            if x > 9.5:
                break

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Faithfulness / necessity / minimality
# ---------------------------------------------------------------------------

def rank_circuit_nodes(attn_attr, mlp_attr):
    """Return flat list of ("A"/M, layer, head_or_none, |attr|) sorted desc."""
    nodes = []
    n_layers, n_heads = attn_attr.shape
    for L in range(n_layers):
        for H in range(n_heads):
            nodes.append(("A", L, H, float(abs(attn_attr[L, H]))))
        nodes.append(("M", L, None, float(abs(mlp_attr[L]))))
    nodes.sort(key=lambda t: -t[3])
    return nodes


@torch.no_grad()
def measure_circuit_behavior(model, tokenizer, clean_tokens, corrupt_tokens,
                             yes_id, no_id, circuit_nodes, mode,
                             clean_cache, corrupt_cache):
    """Re-run the corrupt input while either:
        mode='only'   → restore (patch in clean) ONLY the circuit nodes
        mode='ablate' → zero-ablate the circuit nodes; everything else runs normally
    Returns the logit diff (yes-no) at the answer position on corrupt input.
    """
    tl = model.model  # HookedTransformer
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    circuit_attn = {(L, H) for (k, L, H, _) in circuit_nodes if k == "A"}
    circuit_mlp = {L for (k, L, _, _) in circuit_nodes if k == "M"}

    def make_attn_hook(L):
        def hook(act, hook):
            # act: [batch, seq, d_model] where heads are concatenated along last dim
            modified = act.clone()
            for H in range(n_heads):
                start = H * d_head
                end = start + d_head
                if (L, H) in circuit_attn:
                    if mode == "only":
                        clean_act = clean_cache[f"blocks.{L}.hook_attn_out"].to(act.device)
                        modified[:, :, start:end] = clean_act[:, :, start:end].to(act.dtype)
                    else:  # ablate
                        modified[:, :, start:end] = 0.0
            return modified
        return hook

    def make_mlp_hook(L):
        def hook(act, hook):
            if L in circuit_mlp:
                if mode == "only":
                    clean_act = clean_cache[f"blocks.{L}.hook_mlp_out"].to(act.device)
                    return clean_act.to(act.dtype)
                else:
                    return torch.zeros_like(act)
            return act
        return hook

    fwd_hooks = []
    for L in range(n_layers):
        fwd_hooks.append((f"blocks.{L}.hook_attn_out", make_attn_hook(L)))
        fwd_hooks.append((f"blocks.{L}.hook_mlp_out",  make_mlp_hook(L)))

    tl.reset_hooks()
    try:
        with torch.no_grad():
            logits = tl.run_with_hooks(corrupt_tokens, fwd_hooks=fwd_hooks)
        ld = (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()
    finally:
        tl.reset_hooks()
    return ld


def compute_faithfulness_curve(model, tokenizer, attn_attr, mlp_attr,
                                death_me, life_not_me, device,
                                yes_id, no_id, k_grid):
    """Run faithfulness test at multiple k values."""
    ranked = rank_circuit_nodes(attn_attr, mlp_attr)

    # Use a small sample of pairs
    results = {"k": [], "faithfulness": [], "necessity": [], "ld_full": [],
               "ld_only": [], "ld_ablate": []}

    # Pre-cache clean / corrupt activations for a single representative pair
    # (averaging over multiple is expensive; we average over pairs here)
    n_pairs = min(3, len(death_me), len(life_not_me))
    for k in k_grid:
        circuit = ranked[:k]
        lds_full, lds_only, lds_ab = [], [], []
        for i in range(n_pairs):
            clean_text = get_text(death_me[i])
            corrupt_text = get_text(life_not_me[i])
            ct, xt = tokenize_pair(tokenizer, build_prompt(clean_text),
                                   build_prompt(corrupt_text),
                                   max_length=MAX_TOKENS, device=device)
            # Full-model reference LD on clean input (TL returns logits directly)
            with torch.no_grad():
                full_logits = model.model(ct)
                ld_full = (full_logits[0, -1, yes_id] - full_logits[0, -1, no_id]).float().item()

            # Caches needed for restoration
            _, clean_cache = model.run_with_cache(ct)
            _, corrupt_cache = model.run_with_cache(xt)

            ld_only = measure_circuit_behavior(
                model, tokenizer, ct, xt, yes_id, no_id,
                circuit, mode="only",
                clean_cache=clean_cache, corrupt_cache=corrupt_cache,
            )
            ld_ab = measure_circuit_behavior(
                model, tokenizer, ct, xt, yes_id, no_id,
                circuit, mode="ablate",
                clean_cache=clean_cache, corrupt_cache=corrupt_cache,
            )
            lds_full.append(ld_full)
            lds_only.append(ld_only)
            lds_ab.append(ld_ab)

        ld_full_m = float(np.mean(lds_full))
        ld_only_m = float(np.mean(lds_only))
        ld_ab_m = float(np.mean(lds_ab))
        denom = abs(ld_full_m) + 1e-6
        faith = 1.0 - abs(ld_full_m - ld_only_m) / denom
        necc = abs(ld_full_m - ld_ab_m) / denom
        results["k"].append(k)
        results["ld_full"].append(ld_full_m)
        results["ld_only"].append(ld_only_m)
        results["ld_ablate"].append(ld_ab_m)
        results["faithfulness"].append(faith)
        results["necessity"].append(necc)
        print(f"  k={k:3d}: LD_full={ld_full_m:+.2f}  LD_only={ld_only_m:+.2f}  "
              f"LD_ablate={ld_ab_m:+.2f}  faith={faith:.2f}  necc={necc:.2f}")

    return results


def plot_faithfulness(results, out_path, title):
    ks = results["k"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, results["faithfulness"], "-o", color="#8e44ad",
            lw=2, label="Faithfulness  (restore only circuit)")
    ax.plot(ks, results["necessity"], "-s", color="#c0392b",
            lw=2, label="Necessity  (ablate circuit)")
    ax.axhline(0.9, color="gray", ls="--", lw=0.6)
    ax.axhline(1.0, color="gray", ls=":", lw=0.6)
    # Find minimality k: smallest k with faithfulness >= 0.9
    faith = np.array(results["faithfulness"])
    valid = np.where(faith >= 0.9)[0]
    if len(valid):
        k_min = ks[valid[0]]
        ax.axvline(k_min, color="red", ls=":", lw=1)
        ax.text(k_min, 0.5, f"min k @ 0.9 = {k_min}", rotation=90,
                color="red", fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Number of circuit nodes (top-k EAP)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    add_dataset_arg(parser)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--skip-cards", action="store_true",
                        help="Skip feature cards; faithfulness only.")
    parser.add_argument("--skip-faith", action="store_true",
                        help="Skip faithfulness; cards only.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = get_output_dir(BASE_OUTPUT_DIR, args.dataset)
    ensure_dir(out_dir)
    dataset_label = DATASETS[args.dataset]["display_name"]

    # Severity-aware records (UMD only)
    records_per_sev = defaultdict(list)
    if args.dataset == "umd":
        flat = load_umd_with_severity(source="expert")
        for r in flat:
            records_per_sev[r["umd_label"]].append(r)
    else:
        # Fallback: use standard categories as sev buckets
        cats = load_dataset_by_category(dataset_name=args.dataset)
        for cat, exs in cats.items():
            records_per_sev[cat[:1].lower()].extend(exs)

    cats = load_dataset_by_category(dataset_name=args.dataset)

    for model_name in args.models:
        print(f"\n=== [{model_name}] ===")
        model, tokenizer = load_model_wrapped(model_name, device=device)
        yes_id, no_id = get_yes_no_ids(tokenizer)

        # --- Part A: feature cards ---
        if not args.skip_cards:
            # Find phase-8 SAE checkpoints
            sae_dir = PROJECT_ROOT / "results" / "phase8" / args.dataset
            if not sae_dir.exists():
                sae_dir = PROJECT_ROOT / "results" / "phase8"
            sae_files = sorted(sae_dir.glob(f"{model_name}_sae_weights_L*.pt"))
            if not sae_files:
                print(f"  No SAE checkpoints found in {sae_dir}; skipping feature cards")
            for sae_path in sae_files:
                layer = int(sae_path.stem.split("_L")[-1])
                feat_path = sae_dir / f"{model_name}_sae_features_L{layer}.pt"
                if not feat_path.exists():
                    print(f"  No feature stats for L{layer}, skipping")
                    continue
                print(f"\n  Feature cards from L{layer}")
                sae, act_mean, act_std = load_sae(sae_path)
                sae.to(device)
                feat = torch.load(feat_path, weights_only=False)

                top_idx = feat.get("top_feature_indices", [])[:N_TOP_FEATURES]
                # causal deltas per feature (per phase 8)
                b, s = feat.get("baseline_lds"), feat.get("steered_lds")
                causal_delta = None
                if b is not None and s is not None and len(b) == len(s):
                    causal_delta = float(np.mean([si - bi for si, bi in zip(s, b)]))

                # Per-feature severity means
                # Aggregate activations by severity for each top feature
                severity_means_by_feature = defaultdict(dict)
                sample_per_sev = {sv: records_per_sev.get(sv, [])[:15]
                                  for sv in ("a", "b", "c", "d")}
                for sv, recs in sample_per_sev.items():
                    acts_acc = []
                    for rec in recs:
                        tokens = tokenizer(build_prompt(rec["text"]),
                                           return_tensors="pt", max_length=MAX_TOKENS,
                                           truncation=True, padding="max_length")["input_ids"].to(device)
                        with torch.no_grad():
                            _, cache = model.run_with_cache(tokens)
                            mlp = cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].float().cpu()
                            mlp_norm = (mlp - act_mean) / act_std
                            _, z = sae(mlp_norm.to(device).unsqueeze(0))
                            acts_acc.append(z[0].detach().cpu().numpy())
                    if not acts_acc:
                        continue
                    mean_z = np.mean(acts_acc, axis=0)
                    for fidx in top_idx:
                        severity_means_by_feature[fidx][sv] = float(mean_z[fidx])

                for fidx in top_idx:
                    out_path = out_dir / f"phase12_feature_card_{model_name}_L{layer}_F{fidx}.png"
                    render_feature_card(
                        model, tokenizer, sae, act_mean, act_std,
                        fidx, layer, sample_per_sev,
                        severity_means_by_feature[fidx],
                        device, out_path, model_name,
                        safety_direction=None,
                        causal_delta=causal_delta,
                    )

        # --- Part B: faithfulness ---
        if not args.skip_faith:
            eap_attn_path = (PROJECT_ROOT / "results" / "phase9" / args.dataset /
                             f"{model_name}_eap_attn.pt")
            eap_mlp_path = (PROJECT_ROOT / "results" / "phase9" / args.dataset /
                            f"{model_name}_eap_mlp.pt")
            if not eap_attn_path.exists():
                print(f"  Phase 9 EAP not found at {eap_attn_path}; skipping faithfulness")
            else:
                attn_attr = torch.load(eap_attn_path, weights_only=True)
                mlp_attr = torch.load(eap_mlp_path, weights_only=True)
                death_me = cats.get("Death-Me", [])
                life_not_me = cats.get("Life-Not-Me", [])
                if len(death_me) >= 2 and len(life_not_me) >= 2:
                    # k-grid: exponentially spaced
                    total_nodes = attn_attr.numel() + mlp_attr.numel()
                    k_grid = sorted(set([1, 2, 4, 8, 16, 32, 64, 128,
                                         min(256, total_nodes // 2),
                                         total_nodes]))
                    print(f"  Faithfulness sweep over k = {k_grid}")
                    faith_res = compute_faithfulness_curve(
                        model, tokenizer, attn_attr, mlp_attr,
                        death_me, life_not_me, device, yes_id, no_id, k_grid,
                    )
                    plot_faithfulness(
                        faith_res,
                        out_path=out_dir / f"phase12_faithfulness_{model_name}.png",
                        title=f"Phase 12B: Circuit faithfulness & necessity "
                              f"({model_name.capitalize()}, {dataset_label})",
                    )
                    # Compute minimality
                    faiths = np.array(faith_res["faithfulness"])
                    valid = np.where(faiths >= 0.9)[0]
                    min_k = int(faith_res["k"][valid[0]]) if len(valid) else -1
                    metrics = {
                        "k_grid": faith_res["k"],
                        "faithfulness": faith_res["faithfulness"],
                        "necessity": faith_res["necessity"],
                        "min_k_at_faith_0.9": min_k,
                        "full_circuit_size": total_nodes,
                    }
                    with open(out_dir / f"phase12_metrics_{model_name}.json", "w") as f:
                        json.dump(metrics, f, indent=2)
                    print(f"  Minimality: {min_k} nodes preserve ≥ 0.9 faithfulness "
                          f"out of {total_nodes} total")

        unload_model(model)


if __name__ == "__main__":
    main()
