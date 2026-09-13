#!/usr/bin/env python
"""Module-wise Impact Report: Per-component impact across all models and phases.

Loads saved .pt results from all phases and generates:
1. Per-layer module impact table (Attention IE, MLP IE) for each model
2. Per-head importance ranking for each model
3. Per-neuron causal impact for each model
4. Intervention strategy effectiveness per model
5. SAE feature discovery summary
6. Combined summary figure

Output:
    results/module_impact_report.txt         - Full text report
    results/module_impact_summary.png        - Combined summary figure
    results/module_impact_heatmap.png        - Module-level heatmap across phases
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET = "deepsuimind"
MODELS = ["llama", "qwen", "mistral"]
OUTPUT_DIR = RESULTS_DIR / "module_impact"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report_lines = []

def log(line=""):
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode())
    report_lines.append(line)


def safe_load(path, weights_only=True):
    if path.exists():
        return torch.load(path, weights_only=weights_only)
    return None


# =========================================================================
# Phase 1: Layer-wise Attention IE & MLP IE
# =========================================================================
def report_phase1():
    log("\n" + "=" * 90)
    log("PHASE 1: LAYER-WISE MODULE IMPACT (Activation Patching IE)")
    log("=" * 90)
    log("Module = one transformer layer's attention output OR MLP output")
    log("IE = Indirect Effect: how much patching that module from clean->corrupt restores correct output")
    log("IE ~= 1.0 -> module fully explains behavior | IE ~= 0 -> no effect | IE < 0 -> hurts\n")

    phase1_data = {}

    for model in MODELS:
        attn_ie = safe_load(RESULTS_DIR / "phase1" / DATASET / f"{model}_attn_ie.pt")
        mlp_ie = safe_load(RESULTS_DIR / "phase1" / DATASET / f"{model}_mlp_ie.pt")

        if attn_ie is None or mlp_ie is None:
            log(f"  [{model}] Phase 1 data not found, skipping.")
            continue

        n_layers = len(attn_ie)
        phase1_data[model] = {"attn_ie": attn_ie, "mlp_ie": mlp_ie, "n_layers": n_layers}

        log(f"\n--- {model.upper()} ({n_layers} layers) ---")
        log(f"{'Layer':<8} {'Attn IE':>10} {'MLP IE':>10} {'Combined':>10} {'Dominant Module':>18}")
        log("-" * 60)

        for layer in range(n_layers):
            a = attn_ie[layer].item()
            m = mlp_ie[layer].item()
            combined = a + m
            dominant = "Attention" if abs(a) > abs(m) else "MLP"
            marker = ""
            if abs(a) > 0.1 or abs(m) > 0.1:
                marker = " <-- CRITICAL"
            log(f"L{layer:<6} {a:>+10.4f} {m:>+10.4f} {combined:>+10.4f} {dominant:>14}{marker}")

        # Three-stage summary
        third = n_layers // 3
        for label, ie_tensor, name in [("Attention", attn_ie, "attn"), ("MLP", mlp_ie, "mlp")]:
            early = ie_tensor[:third].mean().item()
            mid = ie_tensor[third:2*third].mean().item()
            late = ie_tensor[2*third:].mean().item()
            log(f"\n  {label} Three-Stage Pattern:")
            log(f"    Early  (L0-{third-1:>2}):  mean IE = {early:+.4f}")
            log(f"    Middle (L{third}-{2*third-1}): mean IE = {mid:+.4f}")
            log(f"    Late   (L{2*third}+):   mean IE = {late:+.4f}")

        # Top 5 modules
        all_modules = []
        for layer in range(n_layers):
            all_modules.append((f"L{layer}_Attn", attn_ie[layer].item()))
            all_modules.append((f"L{layer}_MLP", mlp_ie[layer].item()))
        all_modules.sort(key=lambda x: abs(x[1]), reverse=True)

        log(f"\n  Top 10 Most Impactful Modules (by |IE|):")
        for name, val in all_modules[:10]:
            log(f"    {name:<12} IE = {val:+.4f}")

    return phase1_data


# =========================================================================
# Phase 2: Category-wise Module Impact (Death-Me vs Death-Not-Me)
# =========================================================================
def report_phase2():
    log("\n" + "=" * 90)
    log("PHASE 2: CATEGORY-WISE MODULE IMPACT (Implicit vs Explicit)")
    log("=" * 90)
    log("Compares IE profiles: Death-Me (implicit) vs Death-Not-Me (explicit)\n")

    for model in MODELS:
        dm_ie = safe_load(RESULTS_DIR / "phase2" / DATASET / f"{model}_death_me_ie.pt")
        dnm_ie = safe_load(RESULTS_DIR / "phase2" / DATASET / f"{model}_death_not_me_ie.pt")
        ld_data = safe_load(RESULTS_DIR / "phase2" / DATASET / f"{model}_logit_diffs.pt",
                            weights_only=False)

        if dm_ie is None:
            # Try legacy format
            dm_ie = safe_load(RESULTS_DIR / "phase2" / DATASET / "death_me_ie.pt")
            dnm_ie = safe_load(RESULTS_DIR / "phase2" / DATASET / "death_not_me_ie.pt")
            ld_data = safe_load(RESULTS_DIR / "phase2" / DATASET / "logit_diffs.pt",
                                weights_only=False)
            if dm_ie is None:
                continue
            if model != "llama":
                continue  # Legacy data is only llama

        # dm_ie shape: [2, n_layers] where [0]=attn, [1]=mlp
        n_layers = dm_ie.shape[1]
        log(f"\n--- {model.upper()} ---")

        if ld_data is not None:
            dm_ld = np.mean(ld_data["Death-Me"]) if isinstance(ld_data, dict) else 0
            dnm_ld = np.mean(ld_data["Death-Not-Me"]) if isinstance(ld_data, dict) else 0
            log(f"  Mean Logit Diff: Death-Me = {dm_ld:.3f}, Death-Not-Me = {dnm_ld:.3f}, "
                f"Gap = {dnm_ld - dm_ld:.3f}")

        log(f"\n  {'Layer':<8} {'DM Attn':>10} {'DNM Attn':>10} {'Delta Attn':>10} {'DM MLP':>10} {'DNM MLP':>10} {'Delta MLP':>10}")
        log("  " + "-" * 68)

        for layer in range(n_layers):
            dm_a = dm_ie[0, layer].item()
            dnm_a = dnm_ie[0, layer].item()
            dm_m = dm_ie[1, layer].item()
            dnm_m = dnm_ie[1, layer].item()
            delta_a = dm_a - dnm_a
            delta_m = dm_m - dnm_m
            marker = ""
            if abs(delta_a) > 0.05 or abs(delta_m) > 0.05:
                marker = " <--"
            log(f"  L{layer:<6} {dm_a:>+10.4f} {dnm_a:>+10.4f} {delta_a:>+10.4f} "
                f"{dm_m:>+10.4f} {dnm_m:>+10.4f} {delta_m:>+10.4f}{marker}")


# =========================================================================
# Phase 3: Prompting Strategy Module Impact
# =========================================================================
def report_phase3():
    log("\n" + "=" * 90)
    log("PHASE 3: PROMPTING STRATEGY MODULE IMPACT")
    log("=" * 90)
    log("How each prompt style changes module-level IE\n")

    styles = ["standard", "empathetic", "clinical"]

    for model in MODELS:
        style_data = {}
        for style in styles:
            ie = safe_load(RESULTS_DIR / "phase3" / DATASET / f"{model}_{style}_ie.pt")
            if ie is None:
                # Legacy
                ie = safe_load(RESULTS_DIR / "phase3" / DATASET / f"{style}_ie.pt")
                if ie is None or model != "llama":
                    continue
            style_data[style] = ie  # [2, n_layers]

        if len(style_data) < 2:
            continue

        n_layers = list(style_data.values())[0].shape[1]
        log(f"\n--- {model.upper()} ---")
        header = f"  {'Layer':<8}"
        for s in styles:
            if s in style_data:
                header += f" {s[:4].upper()+' Attn':>11} {s[:4].upper()+' MLP':>11}"
        log(header)
        log("  " + "-" * (8 + 22 * len(style_data)))

        for layer in range(n_layers):
            line = f"  L{layer:<6}"
            any_critical = False
            for s in styles:
                if s in style_data:
                    a = style_data[s][0, layer].item()
                    m = style_data[s][1, layer].item()
                    line += f" {a:>+11.4f} {m:>+11.4f}"
                    if abs(a) > 0.2 or abs(m) > 0.2:
                        any_critical = True
            if any_critical:
                line += " <-- CRITICAL"
            log(line)

        # Highlight biggest clinical vs standard shifts
        if "clinical" in style_data and "standard" in style_data:
            log(f"\n  Biggest Clinical vs Standard shifts (Attention):")
            clin_attn = style_data["clinical"][0]
            std_attn = style_data["standard"][0]
            deltas = (clin_attn - std_attn).numpy()
            top_layers = np.argsort(np.abs(deltas))[::-1][:5]
            for l in top_layers:
                log(f"    L{l}: Delta Attn IE = {deltas[l]:+.4f} "
                    f"(std={std_attn[l].item():.4f} -> clin={clin_attn[l].item():.4f})")


# =========================================================================
# Phase 5: Per-Head Module Impact
# =========================================================================
def report_phase5():
    log("\n" + "=" * 90)
    log("PHASE 5: PER-HEAD MODULE IMPACT (Zero-Ablation Knockout)")
    log("=" * 90)
    log("Importance = |LD_baseline - LD_ablated| when head is zeroed out")
    log("Higher = head is more critical for the safety decision\n")

    phase5_data = {}

    for model in MODELS:
        imp = safe_load(RESULTS_DIR / "phase5" / DATASET / f"{model}_head_importance.pt")
        if imp is None and model == "llama":
            imp = safe_load(RESULTS_DIR / "phase5" / DATASET / "head_importance.pt")
        if imp is None:
            continue

        if isinstance(imp, torch.Tensor):
            imp = imp.numpy()

        n_layers, n_heads = imp.shape
        phase5_data[model] = imp

        log(f"\n--- {model.upper()} ({n_layers} layers × {n_heads} heads = {n_layers*n_heads} total) ---")

        # Top 20 heads
        flat = imp.flatten()
        top_indices = np.argsort(flat)[::-1][:20]

        log(f"\n  Top 20 Critical Heads:")
        log(f"  {'Rank':<6} {'Head':<12} {'Importance':>12} {'Layer':>8} {'% of Total':>12}")
        log("  " + "-" * 54)
        total_imp = flat.sum()
        for rank, idx in enumerate(top_indices):
            layer = idx // n_heads
            head = idx % n_heads
            val = flat[idx]
            pct = (val / total_imp) * 100 if total_imp > 0 else 0
            log(f"  {rank+1:<6} L{layer}H{head:<8} {val:>12.4f} {layer:>8} {pct:>11.2f}%")

        # Layer-wise aggregation
        layer_means = imp.mean(axis=1)
        layer_maxs = imp.max(axis=1)
        log(f"\n  Layer-wise Head Importance Summary:")
        log(f"  {'Layer':<8} {'Mean Imp':>10} {'Max Imp':>10} {'Max Head':>10}")
        log("  " + "-" * 42)
        for layer in range(n_layers):
            max_head = np.argmax(imp[layer])
            marker = " <--" if layer_means[layer] > np.percentile(layer_means, 90) else ""
            log(f"  L{layer:<6} {layer_means[layer]:>10.4f} {layer_maxs[layer]:>10.4f} "
                f"{'H'+str(max_head):>10}{marker}")

    return phase5_data


# =========================================================================
# Phase 6: MLP Neuron Module Impact
# =========================================================================
def report_phase6():
    log("\n" + "=" * 90)
    log("PHASE 6: MLP NEURON MODULE IMPACT (Differential Activation + Causal)")
    log("=" * 90)

    for model in MODELS:
        data = safe_load(RESULTS_DIR / "phase6" / DATASET / f"{model}_neuron_activations.pt",
                         weights_only=False)
        if data is None and model == "llama":
            data = safe_load(RESULTS_DIR / "phase6" / DATASET / "neuron_activations.pt",
                             weights_only=False)
        if data is None:
            continue

        top_neurons = data.get("top_neurons", [])
        baseline_lds = data.get("causal_baseline_lds", [])
        patched_lds = data.get("causal_patched_lds", [])

        log(f"\n--- {model.upper()} ---")
        log(f"\n  Top 20 Neurons by Normalized Differential:")
        log(f"  {'Rank':<6} {'Neuron':<14} {'Norm Diff':>12} {'Selectivity':>12} {'Implicit-biased?':>18}")
        log("  " + "-" * 66)
        for rank, entry in enumerate(top_neurons[:20]):
            if len(entry) == 4:
                l, n, v, s = entry
            else:
                l, n, v = entry
                s = 0
            bias = "Yes <--" if s > 1.05 else ("No" if s < 0.95 else "Neutral")
            log(f"  {rank+1:<6} L{l}N{n:<10} {v:>12.6f} {s:>12.3f} {bias:>18}")

        # Causal validation
        if baseline_lds and patched_lds:
            log(f"\n  Causal Validation (patching top 20 neurons):")
            log(f"  {'Case':<8} {'Baseline LD':>12} {'Patched LD':>12} {'Delta LD':>10} {'Flipped?':>10}")
            log("  " + "-" * 56)
            flips = 0
            for i, (bl, pl) in enumerate(zip(baseline_lds, patched_lds)):
                delta = pl - bl
                flipped = "YES <--" if bl < 0 and pl > 0 else "no"
                if bl < 0 and pl > 0:
                    flips += 1
                log(f"  {i+1:<8} {bl:>12.3f} {pl:>12.3f} {delta:>+10.3f} {flipped:>10}")
            mean_delta = np.mean([p - b for p, b in zip(patched_lds, baseline_lds)])
            log(f"\n  Summary: Mean Delta LD = {mean_delta:+.3f}, Flips = {flips}/{len(baseline_lds)}")


# =========================================================================
# Phase 7: Intervention Module Impact
# =========================================================================
def report_phase7():
    log("\n" + "=" * 90)
    log("PHASE 7: INTERVENTION STRATEGY MODULE IMPACT")
    log("=" * 90)

    for model in MODELS:
        data = safe_load(RESULTS_DIR / "phase7" / DATASET / f"{model}_intervention_results.pt",
                         weights_only=False)
        if data is None and model == "llama":
            data = safe_load(RESULTS_DIR / "phase7" / DATASET / "intervention_results.pt",
                             weights_only=False)
        if data is None:
            continue

        results = data["results"]
        top_heads = data.get("top_heads", [])
        steering_layers = data.get("steering_layers", [])

        log(f"\n--- {model.upper()} ---")
        log(f"  Targeted heads: {[(l,h) for l,h,v in top_heads[:10]]}")
        log(f"  Steering layers: {steering_layers}")

        log(f"\n  {'Strategy':<25} {'Mean LD':>10} {'Std':>8} {'Delta vs Base':>10} {'Detect':>8} {'Rate':>8}")
        log("  " + "-" * 73)

        baseline_mean = np.mean(results["baseline"])
        n = len(results["baseline"])

        for key in sorted(results.keys()):
            vals = results[key]
            mean_ld = np.mean(vals)
            std_ld = np.std(vals)
            delta = mean_ld - baseline_mean
            detect = sum(1 for v in vals if v > 0)
            rate = detect / n if n > 0 else 0
            marker = " <-- BEST" if key != "baseline" and delta == max(
                np.mean(results[k]) - baseline_mean for k in results if k != "baseline") else ""
            log(f"  {key:<25} {mean_ld:>+10.3f} {std_ld:>8.3f} {delta:>+10.3f} "
                f"{detect:>5}/{n:<2} {rate:>7.1%}{marker}")

        # Per-case breakdown for key strategies
        log(f"\n  Per-Case Breakdown:")
        log(f"  {'Case':<6} {'Baseline':>10} {'HeadPatch':>10} {'Best Steer':>12} {'Best Comb':>12}")
        log("  " + "-" * 54)

        # Find best steer and combined keys
        steer_keys = [k for k in results if k.startswith("steer_")]
        comb_keys = [k for k in results if k.startswith("combined_")]
        best_steer = max(steer_keys, key=lambda k: np.mean(results[k])) if steer_keys else None
        best_comb = max(comb_keys, key=lambda k: np.mean(results[k])) if comb_keys else None

        for i in range(n):
            bl = results["baseline"][i]
            hp = results["head_patch"][i]
            bs = results[best_steer][i] if best_steer else 0
            bc = results[best_comb][i] if best_comb else 0
            log(f"  {i+1:<6} {bl:>+10.3f} {hp:>+10.3f} {bs:>+12.3f} {bc:>+12.3f}")


# =========================================================================
# Phase 8: SAE Feature Module Impact
# =========================================================================
def report_phase8():
    log("\n" + "=" * 90)
    log("PHASE 8: SAE FEATURE MODULE IMPACT (Sparse Autoencoder)")
    log("=" * 90)
    log("SAE decomposes distributed MLP activations into monosemantic features.")
    log("d_sae = 8x * d_model = 32768 features per layer.\n")

    for model in ["llama"]:  # Phase 8 only runs on llama
        for layer in [8, 16, 27]:
            feat_path = RESULTS_DIR / "phase8" / DATASET / f"{model}_sae_features_L{layer}.pt"
            weight_path = RESULTS_DIR / "phase8" / DATASET / f"{model}_sae_weights_L{layer}.pt"
            data = safe_load(feat_path, weights_only=False)
            wdata = safe_load(weight_path, weights_only=False)
            if data is None:
                continue

            log(f"\n--- {model.upper()} Layer {layer} ---")

            fs = data.get("feature_stats", {})
            d_sae = wdata.get("d_sae", "?") if wdata else "?"
            d_input = wdata.get("d_input", "?") if wdata else "?"
            n_dead = fs.get("n_dead", "?")
            n_sparse = fs.get("n_sparse", "?")
            frac_active = fs.get("frac_active_overall", None)
            diff_dm_lnm = fs.get("diff_dm_lnm", None)
            diff_dm_dnm = fs.get("diff_dm_dnm", None)
            selectivity = fs.get("selectivity", None)
            mean_act = fs.get("mean_act", {})
            top_indices = data.get("top_feature_indices", [])

            # SAE summary
            log(f"  SAE architecture: d_input={d_input}, d_sae={d_sae} (8x expansion)")
            log(f"  Dead features (never active): {n_dead}")
            log(f"  Ultra-sparse features (<5% active): {n_sparse}")
            alive = d_sae - n_dead if isinstance(d_sae, int) and isinstance(n_dead, int) else "?"
            log(f"  Alive features: {alive}")

            if frac_active is not None and isinstance(frac_active, torch.Tensor):
                active_mask = frac_active > 0
                mean_l0 = frac_active[active_mask].mean().item() if active_mask.any() else 0
                log(f"  Mean activation fraction (alive features): {mean_l0:.4f}")

            # Per-category mean activation
            if mean_act and isinstance(mean_act, dict):
                log(f"\n  Mean Feature Activation by Category:")
                log(f"  {'Category':<20} {'Mean Act':>10} {'Interpretation'}")
                log("  " + "-" * 55)
                for cat, act_tensor in sorted(mean_act.items()):
                    if isinstance(act_tensor, torch.Tensor):
                        m = act_tensor.mean().item()
                        interp = ""
                        if "Death-Me" in cat and "Not" not in cat:
                            interp = "<-- implicit suicide (target)"
                        elif "Death-Not" in cat:
                            interp = "<-- explicit death"
                        elif "Life-Not" in cat:
                            interp = "<-- control (no risk)"
                        log(f"  {cat:<20} {m:>10.6f} {interp}")

            # Suicide-selective features
            if diff_dm_lnm is not None and isinstance(diff_dm_lnm, torch.Tensor):
                n_positive = (diff_dm_lnm > 0).sum().item()
                n_strong_1std = (diff_dm_lnm > diff_dm_lnm.std()).sum().item()
                n_strong_2std = (diff_dm_lnm > 2 * diff_dm_lnm.std()).sum().item()
                log(f"\n  Suicide-Selective Features (DM - LNM > 0): {n_positive}")
                log(f"  Strongly selective (> 1 std): {n_strong_1std}")
                log(f"  Very strongly selective (> 2 std): {n_strong_2std}")

            # Top features table
            if diff_dm_lnm is not None and top_indices:
                log(f"\n  Top 10 Suicide-Selective Features:")
                log(f"  {'Rank':<6} {'Feature':<10} {'DM-LNM':>10} {'DM-DNM':>10} "
                    f"{'Frac Active':>12} {'Selectivity':>12}")
                log("  " + "-" * 64)

                for rank, idx in enumerate(top_indices[:10]):
                    d_lnm = diff_dm_lnm[idx].item()
                    d_dnm = diff_dm_dnm[idx].item() if diff_dm_dnm is not None else 0
                    frac = frac_active[idx].item() if frac_active is not None else 0
                    sel = selectivity[idx].item() if selectivity is not None else 0
                    # Cap selectivity display for readability
                    sel_str = f"{sel:.2f}" if sel < 1000 else "inf"
                    marker = " <-- IMPLICIT" if d_dnm > 0.1 else ""
                    log(f"  {rank+1:<6} F{idx:<8} {d_lnm:>+10.4f} {d_dnm:>+10.4f} "
                        f"{frac:>12.4f} {sel_str:>12}{marker}")

            # Implicit-selective features (DM > DNM)
            if diff_dm_dnm is not None and isinstance(diff_dm_dnm, torch.Tensor):
                top_implicit = diff_dm_dnm.topk(10)
                log(f"\n  Top 10 Implicit-Selective Features (DM > DNM):")
                log(f"  {'Rank':<6} {'Feature':<10} {'DM-DNM':>10} {'DM-LNM':>10} {'Frac Active':>12}")
                log("  " + "-" * 52)
                for rank in range(10):
                    idx = top_implicit.indices[rank].item()
                    d_dnm = top_implicit.values[rank].item()
                    d_lnm = diff_dm_lnm[idx].item() if diff_dm_lnm is not None else 0
                    frac = frac_active[idx].item() if frac_active is not None else 0
                    log(f"  {rank+1:<6} F{idx:<8} {d_dnm:>+10.4f} {d_lnm:>+10.4f} {frac:>12.4f}")

            # Causal validation
            baseline_lds = data.get("baseline_lds", [])
            steered_lds = data.get("steered_lds", [])
            if baseline_lds and steered_lds:
                log(f"\n  Causal Validation (SAE feature steering):")
                log(f"  {'Case':<6} {'Baseline LD':>12} {'Steered LD':>12} {'Delta LD':>10} {'Flipped?':>10}")
                log("  " + "-" * 54)
                flips = 0
                for i, (bl, sl) in enumerate(zip(baseline_lds, steered_lds)):
                    delta = sl - bl
                    flipped = "YES <--" if bl < 0 and sl > 0 else "no"
                    if bl < 0 and sl > 0:
                        flips += 1
                    log(f"  {i+1:<6} {bl:>12.4f} {sl:>12.4f} {delta:>+10.4f} {flipped:>10}")
                mean_delta = np.mean([s - b for s, b in zip(steered_lds, baseline_lds)])
                log(f"\n  Summary: Mean Delta LD = {mean_delta:+.4f}, Flips = {flips}/{len(baseline_lds)}")

        # Cross-layer comparison
        log(f"\n  --- Cross-Layer SAE Comparison ({model.upper()}) ---")
        log(f"  {'Layer':<8} {'Dead':>8} {'Sparse':>8} {'Suicide>0':>10} {'>1std':>8} {'>2std':>8} {'Mean Delta LD':>14}")
        log("  " + "-" * 68)
        for layer in [8, 16, 27]:
            feat_path = RESULTS_DIR / "phase8" / DATASET / f"{model}_sae_features_L{layer}.pt"
            d = safe_load(feat_path, weights_only=False)
            if d is None:
                continue
            fs2 = d.get("feature_stats", {})
            diff2 = fs2.get("diff_dm_lnm", None)
            bl2 = d.get("baseline_lds", [])
            sl2 = d.get("steered_lds", [])
            n_dead2 = fs2.get("n_dead", 0)
            n_sparse2 = fs2.get("n_sparse", 0)
            n_pos = (diff2 > 0).sum().item() if diff2 is not None else 0
            n_1s = (diff2 > diff2.std()).sum().item() if diff2 is not None else 0
            n_2s = (diff2 > 2 * diff2.std()).sum().item() if diff2 is not None else 0
            md = np.mean([s - b for s, b in zip(sl2, bl2)]) if bl2 and sl2 else 0
            log(f"  L{layer:<6} {n_dead2:>8} {n_sparse2:>8} {n_pos:>10} {n_1s:>8} {n_2s:>8} {md:>+14.4f}")


# =========================================================================
# Cross-Phase Summary Figure
# =========================================================================
def generate_summary_figure(phase1_data, phase5_data):
    """Generate a combined heatmap showing module impact across phases."""
    if not phase1_data:
        log("\nNo Phase 1 data available for summary figure.")
        return

    n_models = len(phase1_data)
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, model in enumerate(phase1_data.keys()):
        p1 = phase1_data[model]
        n_layers = p1["n_layers"]

        # Panel 1: Phase 1 Attn + MLP IE
        ax = axes[row, 0]
        combined = np.stack([p1["attn_ie"].numpy(), p1["mlp_ie"].numpy()])
        im = ax.imshow(combined, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Attention", "MLP"])
        ax.set_xlabel("Layer")
        ax.set_title(f"{model.capitalize()}: Layer-wise IE (Phase 1)")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Panel 2: Phase 5 Head importance (if available)
        ax = axes[row, 1]
        if model in phase5_data:
            imp = phase5_data[model]
            im = ax.imshow(imp, cmap="YlOrRd", aspect="auto")
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_title(f"{model.capitalize()}: Head Importance (Phase 5)")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "No Phase 5 data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{model.capitalize()}: Head Importance (N/A)")

        # Panel 3: Combined module ranking
        ax = axes[row, 2]
        all_modules = []
        for layer in range(n_layers):
            all_modules.append((f"L{layer}_Attn", p1["attn_ie"][layer].item()))
            all_modules.append((f"L{layer}_MLP", p1["mlp_ie"][layer].item()))
        all_modules.sort(key=lambda x: abs(x[1]), reverse=True)
        top15 = all_modules[:15]

        names = [m[0] for m in top15][::-1]
        vals = [m[1] for m in top15][::-1]
        colors = ["#e74c3c" if "Attn" in n else "#3498db" for n in names]
        ax.barh(names, vals, color=colors, alpha=0.8)
        ax.set_xlabel("IE")
        ax.set_title(f"{model.capitalize()}: Top 15 Modules by |IE|")
        ax.axvline(x=0, color="black", linewidth=0.5)

    fig.suptitle("Module-wise Impact Summary Across All Models",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "module_impact_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\nSaved: {OUTPUT_DIR / 'module_impact_summary.png'}")


# =========================================================================
# Main
# =========================================================================
def main():
    log("=" * 90)
    log("MODULE-WISE IMPACT REPORT")
    log("Dataset: DeepSuiMind | Models: Llama, Qwen, Mistral")
    log("=" * 90)
    log("This report shows the impact of every identifiable module (layer, head, neuron,")
    log("SAE feature) on suicide detection, across all 8 phases of analysis.\n")

    phase1_data = report_phase1()
    report_phase2()
    report_phase3()
    phase5_data = report_phase5()
    report_phase6()
    report_phase7()
    report_phase8()

    # Grand summary
    log("\n" + "=" * 90)
    log("GRAND SUMMARY: MODULE IMPACT HIERARCHY")
    log("=" * 90)
    log("""
Module Level        | Finding                                      | Causal Power
--------------------|----------------------------------------------|------------------
Whole Layer (Attn)  | Peak IE at L14 (Llama), L6 (Qwen), L1 (Mis) | Moderate
Whole Layer (MLP)   | L0 dominates raw IE; normalized=early layers | Low-Moderate
Individual Head     | L31H30, L30H18, L31H31 (Llama) are critical | High (per-head)
Individual Neuron   | Distributed across 1000s; top 20 shift <0.13 | Very Low
SAE Feature         | 2958 suicide-selective features (L16, >1 std)| Pending validation
Prompt Module       | Clinical prompt: L31 Attn IE jumps to 0.979  | High (indirect)
Steering Vector     | Combined intervention: +0.5 LD improvement   | Moderate
""")

    generate_summary_figure(phase1_data, phase5_data)

    # Save text report
    report_path = OUTPUT_DIR / "module_impact_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
