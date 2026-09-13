#!/usr/bin/env python
"""Phase 1c: Token-Position-Grouped Activation Patching.

Decomposes the per-layer Indirect Effect (IE) by token-position group,
matching Liu et al. 2026 (TOIS, "How Do LLMs Understand Relevance? A
Mechanistic Interpretability Perspective", Figs. 4/8/10).

For each (group, layer, component) we measure the IE of patching only the
clean-cache activations at positions belonging to that group.

Token groups (mapped from the suicide-detection prompt):
    instruction : "Read the following text..." prefix + "Answer:" suffix
    document    : the post body
    query       : "Does this text show signs of suicidal ideation?..."
    last_token  : the final non-pad position (where logits emerge)

Outputs (per model, per dataset):
    {model}_position_attn_ie.pt   - dict: group -> [n_pairs, n_layers]
    {model}_position_mlp_ie.pt    - dict: group -> [n_pairs, n_layers]
    phase1c_heatmap_{model}.png   - 4-row x layers heatmap (attn + mlp)
    phase1c_curves_{model}.png    - line plots, mean ± 95% bootstrap CI
    phase1c_comparison.png        - side-by-side across all loaded models
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
    get_yes_no_ids, ensure_dir, add_dataset_arg, get_output_dir, DATASETS,
)

N_CASES = 100
MAX_TOKENS = 320
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "phase1c"
MODELS = ["llama", "qwen", "mistral"]
GROUPS = ["document", "query", "instruction", "last_token"]
GROUP_COLORS = {
    "document": "#1f77b4",
    "query": "#ff7f0e",
    "instruction": "#2ca02c",
    "last_token": "#d62728",
}

# Hard-coded prompt segments (must stay aligned with the standard template
# in src/analysis_utils.PROMPT_TEMPLATES["standard"]).
INSTR_PREFIX = (
    "Read the following text and determine if the person shows signs of "
    "suicidal ideation.\n\nText: "
)
INSTR_BETWEEN = "\n\n"
QUERY_TEXT = "Does this text show signs of suicidal ideation? Answer with Yes or No."
INSTR_SUFFIX = "\nAnswer:"


def build_prompt_with_spans(text: str):
    """Build the standard prompt and return (full_prompt, group_char_spans)."""
    s_instr_a_start = 0
    s_instr_a_end = len(INSTR_PREFIX)
    s_doc_start = s_instr_a_end
    s_doc_end = s_doc_start + len(text)
    s_instr_b_start = s_doc_end
    s_instr_b_end = s_instr_b_start + len(INSTR_BETWEEN)
    s_query_start = s_instr_b_end
    s_query_end = s_query_start + len(QUERY_TEXT)
    s_instr_c_start = s_query_end
    s_instr_c_end = s_instr_c_start + len(INSTR_SUFFIX)

    full = INSTR_PREFIX + text + INSTR_BETWEEN + QUERY_TEXT + INSTR_SUFFIX
    spans = {
        "instruction": [
            (s_instr_a_start, s_instr_a_end),
            (s_instr_b_start, s_instr_b_end),
            (s_instr_c_start, s_instr_c_end),
        ],
        "document": [(s_doc_start, s_doc_end)],
        "query": [(s_query_start, s_query_end)],
    }
    return full, spans


def tokenize_with_groups(tokenizer, text, max_length, device):
    """Tokenize the prompt for `text` and return (input_ids, group_to_positions)."""
    full, spans = build_prompt_with_spans(text)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(
        full,
        return_offsets_mapping=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    offsets = enc["offset_mapping"][0].tolist()
    attn_mask = enc["attention_mask"][0]

    def collect(char_spans):
        idx = []
        for tok_i, (a, b) in enumerate(offsets):
            if attn_mask[tok_i].item() == 0:
                continue
            if b == a:  # special tokens (BOS, etc.) have offset (0,0)
                continue
            for (cs, ce) in char_spans:
                if a >= cs and b <= ce:
                    idx.append(tok_i)
                    break
        return idx

    last_real = int(attn_mask.sum().item()) - 1
    groups = {
        "instruction": collect(spans["instruction"]),
        "document": collect(spans["document"]),
        "query": collect(spans["query"]),
        "last_token": [last_real],
    }

    input_ids = enc["input_ids"].to(device)
    return input_ids, groups


@torch.no_grad()
def run_position_patching(model, clean_tokens, corrupt_tokens, corrupt_groups,
                          yes_id, no_id):
    """Position-grouped denoising patching.

    For each (group, layer, component): patch only the positions in `group`
    on the corrupt forward pass, using the clean cache.

    Returns:
        out: dict[group_name] -> {"attn_ie": [n_layers], "mlp_ie": [n_layers]}
        ld_clean, ld_corrupt: scalars
    """
    n_layers = model.cfg.n_layers

    clean_logits = model(clean_tokens)
    ld_clean = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).float()

    corrupt_logits = model(corrupt_tokens)
    ld_corrupt = (corrupt_logits[0, -1, yes_id] - corrupt_logits[0, -1, no_id]).float()

    _, clean_cache = model.run_with_cache(clean_tokens)

    denom = (ld_clean - ld_corrupt).abs()
    zeros = lambda: torch.zeros(n_layers)
    out = {g: {"attn_ie": zeros(), "mlp_ie": zeros()} for g in corrupt_groups}

    if denom < 1e-8:
        return out, ld_clean.item(), ld_corrupt.item()

    for layer in range(n_layers):
        for component, key in [("hook_attn_out", "attn_ie"),
                               ("hook_mlp_out", "mlp_ie")]:
            hook_name = f"blocks.{layer}.{component}"
            if hook_name not in clean_cache:
                continue
            clean_act = clean_cache[hook_name]

            for group_name, positions in corrupt_groups.items():
                if not positions:
                    continue
                pos = torch.tensor(positions, dtype=torch.long,
                                   device=clean_act.device)

                def make_hook(cached, pos_idx):
                    def hook_fn(output, hook):
                        is_tensor = isinstance(output, torch.Tensor)
                        out_t = output if is_tensor else output[0]
                        cached_dev = cached.to(out_t.device).to(out_t.dtype)
                        out_t = out_t.clone()
                        out_t[:, pos_idx, :] = cached_dev[:, pos_idx, :]
                        return out_t if is_tensor else (out_t,) + tuple(output[1:])
                    return hook_fn

                model.add_hook(hook_name, make_hook(clean_act, pos))
                patched_logits = model(corrupt_tokens)
                model.remove_hooks()

                ld_patched = (patched_logits[0, -1, yes_id]
                              - patched_logits[0, -1, no_id]).float()
                out[group_name][key][layer] = ((ld_patched - ld_corrupt) / denom).item()

    return out, ld_clean.item(), ld_corrupt.item()


def bootstrap_ci(arr, n_boot=1000, ci=95, seed=0):
    """Bootstrap mean +/- CI band over the first axis. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n == 0:
        return None, None, None
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    return arr.mean(axis=0), lo, hi


def plot_heatmap(group_attn, group_mlp, model_name, dataset_label, save_path):
    """Two stacked heatmaps: attention IE and MLP IE, rows=groups, cols=layers."""
    n_layers = next(iter(group_attn.values())).shape[1]
    rows = GROUPS  # in fixed order

    attn_mat = np.stack([group_attn[g].mean(axis=0).numpy() for g in rows], axis=0)
    mlp_mat = np.stack([group_mlp[g].mean(axis=0).numpy() for g in rows], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_layers * 0.25), 5),
                             sharex=True)
    vmax = max(np.abs(attn_mat).max(), np.abs(mlp_mat).max(), 1e-6)

    for ax, mat, title in [
        (axes[0], attn_mat, f"IE: Attention ({model_name.capitalize()}, {dataset_label})"),
        (axes[1], mlp_mat, f"IE: MLP ({model_name.capitalize()}, {dataset_label})"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.replace("_", " ").title() for r in rows])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    axes[1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_curves(group_attn, group_mlp, model_name, dataset_label, save_path):
    """Per-group line plots with bootstrap 95% CIs (matches paper Fig. 4 right)."""
    n_layers = next(iter(group_attn.values())).shape[1]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey="row")
    for col, g in enumerate(GROUPS):
        for row, (group_dict, label) in enumerate([(group_attn, "Attention IE"),
                                                   (group_mlp, "MLP IE")]):
            mean, lo, hi = bootstrap_ci(group_dict[g].numpy())
            ax = axes[row, col]
            color = GROUP_COLORS[g]
            ax.plot(layers, mean, color=color, linewidth=1.6)
            ax.fill_between(layers, lo, hi, color=color, alpha=0.25)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            if row == 0:
                ax.set_title(g.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Phase 1c: Position-Grouped IE — {model_name.capitalize()} "
        f"({dataset_label}, mean +/- 95% bootstrap CI)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cross_model(all_results, dataset_label, save_path):
    """One figure: rows = models, cols = {Attn, MLP}; each cell is a heatmap."""
    models = list(all_results.keys())
    if not models:
        return

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.6 * n_rows + 1),
                             squeeze=False)

    # Find global vmax for consistent color scale
    vmax = 1e-6
    mats_by_model = {}
    for m in models:
        r = all_results[m]
        a_mat = np.stack([r["attn"][g].mean(axis=0).numpy() for g in GROUPS], 0)
        m_mat = np.stack([r["mlp"][g].mean(axis=0).numpy() for g in GROUPS], 0)
        mats_by_model[m] = (a_mat, m_mat)
        vmax = max(vmax, np.abs(a_mat).max(), np.abs(m_mat).max())

    for i, m in enumerate(models):
        a_mat, m_mat = mats_by_model[m]
        for j, (mat, title) in enumerate([(a_mat, "Attention"), (m_mat, "MLP")]):
            ax = axes[i, j]
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_yticks(range(len(GROUPS)))
            ax.set_yticklabels([g.replace("_", " ").title() for g in GROUPS],
                               fontsize=8)
            ax.set_title(f"{m.capitalize()} — {title}", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel("Layer")
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.suptitle(
        f"Phase 1c: Cross-Model Position-Grouped IE ({dataset_label})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")


def analyze_single_model(model_name, death_me, life_not_me, device,
                         output_dir, dataset_label):
    print(f"\n{'='*60}\n[{model_name}] Loading model...\n{'='*60}")
    model, tokenizer = load_model_wrapped(model_name, device=device)
    yes_id, no_id = get_yes_no_ids(tokenizer)

    n_layers = model.cfg.n_layers
    n_pairs = min(N_CASES, len(death_me), len(life_not_me))

    group_attn = {g: torch.zeros(n_pairs, n_layers) for g in GROUPS}
    group_mlp = {g: torch.zeros(n_pairs, n_layers) for g in GROUPS}

    for i in range(n_pairs):
        clean_text = get_text(death_me[i])
        corrupt_text = get_text(life_not_me[i])

        clean_tokens, _ = tokenize_with_groups(tokenizer, clean_text,
                                               MAX_TOKENS, device)
        corrupt_tokens, corrupt_groups = tokenize_with_groups(
            tokenizer, corrupt_text, MAX_TOKENS, device,
        )

        result, ld_c, ld_x = run_position_patching(
            model, clean_tokens, corrupt_tokens, corrupt_groups, yes_id, no_id,
        )

        for g in GROUPS:
            group_attn[g][i] = result[g]["attn_ie"]
            group_mlp[g][i] = result[g]["mlp_ie"]

        peak_g = max(GROUPS,
                     key=lambda g: float(result[g]["attn_ie"].abs().max()))
        peak_l = int(result[peak_g]["attn_ie"].abs().argmax())
        peak_v = float(result[peak_g]["attn_ie"][peak_l])
        print(f"  [{model_name}] Case {i+1}/{n_pairs}: LD_c={ld_c:.2f} "
              f"LD_x={ld_x:.2f} | peak attn-IE={peak_v:+.3f} ({peak_g} L{peak_l})")

    torch.save({g: group_attn[g] for g in GROUPS},
               output_dir / f"{model_name}_position_attn_ie.pt")
    torch.save({g: group_mlp[g] for g in GROUPS},
               output_dir / f"{model_name}_position_mlp_ie.pt")
    print(f"  [{model_name}] Saved position IE tensors to {output_dir}")

    plot_heatmap(group_attn, group_mlp, model_name, dataset_label,
                 str(output_dir / f"phase1c_heatmap_{model_name}.png"))
    plot_curves(group_attn, group_mlp, model_name, dataset_label,
                str(output_dir / f"phase1c_curves_{model_name}.png"))

    print(f"\n  [{model_name}] PER-GROUP PEAK SUMMARY")
    for g in GROUPS:
        a = group_attn[g].mean(dim=0)
        m = group_mlp[g].mean(dim=0)
        print(f"    {g:11s}: attn peak={a.abs().max():.3f} (L{int(a.abs().argmax())}) "
              f"| mlp peak={m.abs().max():.3f} (L{int(m.abs().argmax())})")

    out = {
        "attn": {g: group_attn[g] for g in GROUPS},
        "mlp": {g: group_mlp[g] for g in GROUPS},
        "n_layers": n_layers,
    }

    unload_model(model)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def load_saved_results(model_name, output_dir):
    a = output_dir / f"{model_name}_position_attn_ie.pt"
    m = output_dir / f"{model_name}_position_mlp_ie.pt"
    if a.exists() and m.exists():
        a_d = torch.load(a, weights_only=True)
        m_d = torch.load(m, weights_only=True)
        n_layers = next(iter(a_d.values())).shape[1]
        return {"attn": a_d, "mlp": m_d, "n_layers": n_layers}
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1c: Token-Position-Grouped Activation Patching",
    )
    parser.add_argument("--models", nargs="+",
                        choices=MODELS + ["all"], default=["all"])
    add_dataset_arg(parser)
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label = DATASETS[dataset_name]["display_name"]
    output_dir = get_output_dir(BASE_OUTPUT_DIR, dataset_name)
    models_to_run = MODELS if "all" in args.models else args.models

    print("=" * 70)
    print("PHASE 1c: Position-Grouped Activation Patching")
    print(f"  Dataset: {dataset_label}")
    print(f"  Models:  {', '.join(m.capitalize() for m in models_to_run)}")
    print(f"  Groups:  {', '.join(GROUPS)}")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    ensure_dir(output_dir)

    cats = load_dataset_by_category(dataset_name=dataset_name)
    death_me = cats.get("Death-Me", [])
    life_not_me = cats.get("Life-Not-Me", [])
    if not death_me or not life_not_me:
        print("ERROR: dataset missing required categories")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}
    for m in models_to_run:
        try:
            all_results[m] = analyze_single_model(
                m, death_me, life_not_me, device, output_dir, dataset_label,
            )
        except Exception as e:
            print(f"ERROR with {m}: {e}")
            import traceback; traceback.print_exc()

    # Pull in any previously saved models for the comparison plot
    for m in MODELS:
        if m not in all_results:
            saved = load_saved_results(m, output_dir)
            if saved is not None:
                all_results[m] = saved
                print(f"Loaded saved results for {m}")

    if len(all_results) >= 1:
        plot_cross_model(all_results, dataset_label,
                         str(output_dir / "phase1c_comparison.png"))

    print("\n" + "=" * 70)
    print(f"PHASE 1c COMPLETE ({dataset_label})")
    print("=" * 70)


if __name__ == "__main__":
    main()
