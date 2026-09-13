#!/usr/bin/env python
"""Phase 11: Severity Axis Discovery (UMD-exclusive).

Leverages UMD's four-level ordinal labels (a = No Risk, b = Low, c = Moderate,
d = Severe) — a feature no other suicide-detection dataset in this repo has.

Core idea: if the model has internally graded severity representations, there
should exist a 1-D linear direction in residual stream whose projection
monotonically tracks a/b/c/d. This phase discovers that direction, validates
its ordinality, and demonstrates counterfactual steering along it.

Pipeline:
  1. Collect mean residual-stream vector at answer position for every UMD
     record, at every layer.
  2. Per layer, fit ridge regression: severity (0-3) ~ residual. Weight vector
     is the "severity direction"; R² measures how linearly severity is encoded.
  3. Pick the best layer; project all records onto its direction → 1-D plot
     coloured by label. Clean monotonic separation = model understands grade.
  4. Counterfactual steering: for a severe (d) post, add -α · direction at the
     best layer, measure how the model's Yes probability falls as α increases.
     Reverse: boost a no-risk (a) post with +α · direction, see it flip.

Outputs:
    results/phase11/{dataset}/phase11_severity_per_layer.png      R² by layer
    results/phase11/{dataset}/phase11_axis_projection.png          1D scatter
    results/phase11/{dataset}/phase11_steering_curve.png           dose-response
    results/phase11/{dataset}/phase11_severity_data_{model}.pt
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
from scipy.stats import spearmanr

from src.analysis_utils import (
    load_model_wrapped, unload_model, build_prompt, get_yes_no_ids,
    ensure_dir, add_dataset_arg, get_output_dir, DATASETS,
    load_umd_with_severity, UMD_LABEL_NAMES,
)

MAX_TOKENS = 192
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase11"
MODELS = ["llama"]
N_PER_LEVEL = 30   # cap per severity level for balanced data
STEERING_SCALES = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


@torch.no_grad()
def collect_residuals(model, tokenizer, records, device, max_tokens=MAX_TOKENS):
    """Collect residual-stream vector at the answer position for every record,
    at every layer. TL-backed: reads `blocks.{L}.hook_resid_post` from cache.
    Returns [n_records, n_layers, d_model], severities [n_records]."""
    tl = model.model  # HookedTransformer
    n_layers = model.cfg.n_layers
    severities = []
    all_hidden = []
    for i, rec in enumerate(records):
        text = rec["text"]
        tokens = tokenizer(build_prompt(text), return_tensors="pt",
                           max_length=max_tokens, truncation=True,
                           padding="max_length")["input_ids"].to(device)
        _, cache = tl.run_with_cache(tokens)
        stacked = torch.stack([
            cache[f"blocks.{L}.hook_resid_post"][0, -1, :].float().cpu()
            for L in range(n_layers)
        ])  # [n_layers, d]
        all_hidden.append(stacked)
        severities.append(rec["severity"])
        if (i + 1) % 20 == 0:
            print(f"  collected {i+1}/{len(records)}")
    H = torch.stack(all_hidden)  # [n_records, n_layers, d]
    sev = torch.tensor(severities, dtype=torch.float32)
    return H, sev


def fit_severity_direction(H_layer, sev, alpha=1.0):
    """Ridge regression: sev ~ H_layer. Return (weight, r2, spearman_rho)."""
    X = H_layer.numpy()              # [n, d]
    y = sev.numpy()                  # [n]
    # Ridge closed form: w = (X^T X + alpha I)^{-1} X^T y
    d = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(d)
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    y_hat = X @ w
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    rho, _ = spearmanr(y, y_hat)
    return torch.from_numpy(w).float(), float(r2), float(rho)


def plot_r2_per_layer(r2s, rhos, out_path, title):
    n_layers = len(r2s)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(n_layers), r2s, "-o", color="#8e44ad", lw=2, label="R² (ridge)")
    ax.plot(range(n_layers), rhos, "--s", color="#16a085", lw=2,
            label="Spearman ρ (ordinality)", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Severity-encoding quality")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    best = int(np.argmax(r2s))
    ax.axvline(best, color="red", ls=":", lw=1)
    ax.text(best + 0.3, max(r2s) * 0.9, f"best: L{best}", color="red")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_axis_projection(projections, labels, out_path, title):
    """1-D scatter of every record's projection, colour-coded by label."""
    palette = {"a": "#2980b9", "b": "#16a085", "c": "#e67e22", "d": "#c0392b",
               "None": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rng = np.random.default_rng(0)
    for lbl in ["None", "a", "b", "c", "d"]:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        if not idx:
            continue
        xs = [projections[i] for i in idx]
        ys = rng.uniform(-0.4, 0.4, len(idx))
        ax.scatter(xs, ys, s=48, color=palette.get(lbl, "gray"),
                   label=f"{lbl} — {UMD_LABEL_NAMES.get(lbl, lbl)}  (n={len(idx)})",
                   edgecolor="white", linewidth=0.5, alpha=0.85)
        # median marker
        med = float(np.median(xs))
        ax.axvline(med, color=palette.get(lbl, "gray"),
                   ls="--", lw=1, alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel("Projection onto severity direction →")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


@torch.no_grad()
def steering_experiment(model, tokenizer, record, direction, layer, device,
                        yes_id, no_id, scales=STEERING_SCALES):
    """At target layer's residual output, add alpha * direction. Measure P(yes)."""
    text = record["text"]
    tokens = tokenizer(build_prompt(text), return_tensors="pt",
                       max_length=MAX_TOKENS, truncation=True,
                       padding="max_length")["input_ids"].to(device)

    direction_d = direction.to(device)
    # normalize direction
    direction_d = direction_d / (direction_d.norm() + 1e-8)

    # we'll steer by adding alpha * direction to the MLP output at that layer
    # via TL's hook_mlp_out hook point
    tl = model.model  # HookedTransformer
    hook_name = f"blocks.{layer}.hook_mlp_out"
    results = []

    for alpha in scales:
        def hook(act, hook, _a=alpha):
            return act + _a * direction_d.to(act.dtype)

        try:
            logits = tl.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])
            ld = (logits[0, -1, yes_id] - logits[0, -1, no_id]).float().item()
            p = torch.softmax(logits[0, -1, :].float(), dim=-1)
            p_yes = p[yes_id].item()
            p_no = p[no_id].item()
        finally:
            tl.reset_hooks()
        results.append({"alpha": alpha, "logit_diff": ld,
                        "p_yes": p_yes, "p_no": p_no})
        print(f"    alpha={alpha:+.1f}: LD={ld:+.2f}  P(yes)={p_yes:.3f}  P(no)={p_no:.3f}")
    return results


def plot_steering_curve(records_and_results, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"a": "#2980b9", "c": "#e67e22", "d": "#c0392b",
               "None": "#7f8c8d", "b": "#16a085"}
    for (label, lbl_raw), results in records_and_results:
        alphas = [r["alpha"] for r in results]
        lds = [r["logit_diff"] for r in results]
        ax.plot(alphas, lds, "-o", label=label,
                color=palette.get(lbl_raw, "gray"), lw=2, markersize=6)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("Steering strength α (direction of severity axis)")
    ax.set_ylabel("logit(Yes) − logit(No)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="umd",
                        help="Must be 'umd' for severity axis.")
    parser.add_argument("--source", type=str, default="expert",
                        choices=["expert", "crowd"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    if args.dataset != "umd":
        raise ValueError("Phase 11 requires UMD dataset (graded severity).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flat = load_umd_with_severity(source=args.source)

    # Balance: cap each severity level
    by_sev = {0: [], 1: [], 2: [], 3: []}
    for r in flat:
        by_sev[r["severity"]].append(r)
    balanced = []
    for s in sorted(by_sev):
        np.random.default_rng(0).shuffle(by_sev[s])
        balanced.extend(by_sev[s][:N_PER_LEVEL])
    print(f"Balanced records per severity level: "
          f"{[sum(1 for r in balanced if r['severity'] == s) for s in range(4)]}")

    out_dir = get_output_dir(BASE_OUTPUT_DIR, args.dataset)
    ensure_dir(out_dir)

    for model_name in args.models:
        print(f"\n[{model_name}] Loading model...")
        model, tokenizer = load_model_wrapped(model_name, device=device)
        yes_id, no_id = get_yes_no_ids(tokenizer)
        n_layers = model.cfg.n_layers

        print(f"[{model_name}] Collecting residuals ({len(balanced)} records, "
              f"{n_layers} layers)...")
        H, sev = collect_residuals(model, tokenizer, balanced, device)

        # Fit severity direction per layer
        print(f"[{model_name}] Fitting severity direction per layer...")
        r2s, rhos, weights = [], [], []
        for L in range(n_layers):
            w, r2, rho = fit_severity_direction(H[:, L, :], sev)
            r2s.append(r2)
            rhos.append(rho)
            weights.append(w)
        print(f"  R² by layer (summary): "
              f"min={min(r2s):.2f}  max={max(r2s):.2f}  argmax=L{int(np.argmax(r2s))}")

        plot_r2_per_layer(r2s, rhos,
                          out_path=out_dir / f"phase11_severity_per_layer_{model_name}.png",
                          title=f"Phase 11: Severity-encoding quality per layer "
                                f"({model_name.capitalize()})")

        best_layer = int(np.argmax(r2s))
        w_best = weights[best_layer]
        # Project every record at the best layer onto w_best
        proj = (H[:, best_layer, :] @ w_best).numpy()
        labels_str = [r["umd_label"] for r in balanced]
        plot_axis_projection(
            proj, labels_str,
            out_path=out_dir / f"phase11_axis_projection_{model_name}.png",
            title=f"Phase 11: Severity axis at L{best_layer} "
                  f"({model_name.capitalize()}) — R²={r2s[best_layer]:.2f}, "
                  f"ρ={rhos[best_layer]:.2f}",
        )

        # Counterfactual steering: pick one record per level
        target_examples = []
        for sev_val, lbl_pref in [(3, "d"), (2, "c"), (0, "a")]:
            cand = [r for r in balanced if r["severity"] == sev_val]
            if cand:
                target_examples.append((cand[0], lbl_pref))

        records_and_results = []
        print(f"[{model_name}] Counterfactual steering at L{best_layer}...")
        for rec, lbl in target_examples:
            print(f"  steering record user={rec['user_id']} label={lbl}")
            res = steering_experiment(model, tokenizer, rec, w_best, best_layer,
                                      device, yes_id, no_id)
            records_and_results.append(
                ((f"{UMD_LABEL_NAMES[lbl]} (u{rec['user_id']})", lbl), res))

        plot_steering_curve(
            records_and_results,
            out_path=out_dir / f"phase11_steering_curve_{model_name}.png",
            title=f"Phase 11: Counterfactual steering along severity axis "
                  f"(L{best_layer}, {model_name.capitalize()})",
        )

        torch.save({
            "best_layer": best_layer,
            "weights_per_layer": weights,
            "r2_per_layer": r2s,
            "spearman_per_layer": rhos,
            "projections_best_layer": proj,
            "labels": labels_str,
            "severities": sev.numpy(),
            "steering_results": records_and_results,
        }, out_dir / f"phase11_severity_data_{model_name}.pt")

        unload_model(model)


if __name__ == "__main__":
    main()
