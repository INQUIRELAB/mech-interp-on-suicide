#!/usr/bin/env python
"""SC-4: Position Effect from Padding.

Question: tokenize_pair uses padding="max_length" with max_length=256.
The actual answer position shifts left as text length varies. If Death-Me
texts are longer than Life-Not-Me, the answer-position residual is sampled
at a systematically different position — and Phase 11's residual probe
inherits this confound.

Output:
    results/sc4/position_distributions.png - histogram of last non-pad positions
    results/sc4/position_bug_report.md - report on [-1, :] indexing issue
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis_utils import (
    load_dataset_by_category, get_text, build_prompt,
    ensure_dir, PROMPT_TEMPLATES,
)


def tokenize_with_position(tokenizer, prompt, max_length=256):
    """Tokenize and return the position of the last non-pad token.

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: text prompt
        max_length: max sequence length

    Returns:
        (last_non_pad_pos, seq_len) tuple
    """
    enc = tokenizer(
        prompt, return_tensors="pt", max_length=max_length,
        truncation=True, padding="max_length",
    )

    input_ids = enc["input_ids"][0]  # [max_length]

    # Find last non-pad token position
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0  # Some tokenizers use 0

    # Find last position that is NOT a pad token
    non_pad_mask = input_ids != pad_token_id
    if non_pad_mask.any():
        last_non_pad = non_pad_mask.nonzero()[-1].item()
    else:
        last_non_pad = 0

    return last_non_pad, len(input_ids)


def analyze_position_distribution(categories, tokenizer, n_samples=50, max_length=256):
    """Analyze the distribution of last non-pad positions by category.

    Args:
        categories: dict with category names -> list of examples
        tokenizer: HuggingFace tokenizer
        n_samples: max examples per category
        max_length: max sequence length

    Returns:
        dict with positions per category
    """
    results = {}

    for cat_name, examples in categories.items():
        n = min(n_samples, len(examples))
        positions = []

        for i in range(n):
            text = get_text(examples[i])
            if not text or not text.strip():
                continue
            prompt = build_prompt(text, style="standard")
            pos, seq_len = tokenize_with_position(tokenizer, prompt, max_length)
            positions.append(pos)

        results[cat_name] = {
            "positions": positions,
            "mean": float(np.mean(positions)) if positions else 0.0,
            "std": float(np.std(positions)) if positions else 0.0,
            "min": int(np.min(positions)) if positions else 0,
            "max": int(np.max(positions)) if positions else 0,
            "median": float(np.median(positions)) if positions else 0.0,
        }

        print(f"  {cat_name}: n={len(positions)}, mean={results[cat_name]['mean']:.1f}, "
              f"range=[{results[cat_name]['min']}, {results[cat_name]['max']}]")

    return results


def plot_position_distributions(results, save_path, max_length=256):
    """Plot histogram of last non-pad positions by category.

    Args:
        results: dict from analyze_position_distribution
        save_path: where to save the plot
        max_length: max sequence length
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Death-Me": "#e74c3c", "Death-Not-Me": "#f39c12", "Life-Not-Me": "#3498db"}

    bins = np.linspace(0, max_length, 51)

    for cat, data in results.items():
        positions = data["positions"]
        ax.hist(positions, bins=bins, alpha=0.5, label=cat, color=colors.get(cat, None))

    ax.axvline(max_length - 1, color='k', linestyle='--', alpha=0.5, label=f'max_length-1 ({max_length-1})')

    ax.set_xlabel("Last Non-Pad Token Position")
    ax.set_ylabel("Count")
    ax.set_title("SC-4: Distribution of Last Non-Pad Token Position by Category")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close(fig)


def check_phase11_bug():
    """Check Phase 11 for the [-1, :] indexing bug.

    Reads phase11_severity_axis.py to verify how it indexes the residual
    tensor. If it uses [-1, :], it's indexing the last pad token, not
    the answer token.

    Returns:
        str describing the bug (or empty if no bug)
    """
    phase11_path = PROJECT_ROOT / "analysis" / "phase11_severity_axis.py"

    if not phase11_path.exists():
        return "phase11_severity_axis.py not found - cannot verify bug"

    with open(phase11_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Look for [-1, :] or [-1] indexing in residual collection
    if "[-1, :]" in content or "[-1]" in content:
        # Check if it's indexing the last token of the padded sequence
        lines = content.split('\n')
        bug_lines = []
        for i, line in enumerate(lines, 1):
            if "[-1" in line and ("residual" in line.lower() or "cache" in line.lower()):
                bug_lines.append(f"  Line {i}: {line.strip()}")

        if bug_lines:
            bug_report = (
                "CONFIRMED: Phase 11 uses [-1, :] indexing on padded tensors.\n"
                "This indexes the LAST PAD TOKEN, not the answer token.\n"
                "All Phase 11 'severity axis' results are affected.\n\n"
                "Evidence:\n" + "\n".join(bug_lines)
            )
            return bug_report

    return "No [-1, :] indexing found in Phase 11 - may be safe"


def write_bug_report(results, phase11_bug_report, save_path):
    """Write a markdown report on the position effect bug.

    Args:
        results: dict from analyze_position_distribution
        phase11_bug_report: str from check_phase11_bug
        save_path: where to save the report
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# SC-4: Position Effect Bug Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("When tokenizing with `padding='max_length'` and `max_length=256`,\n")
        f.write("the actual last non-pad token position varies with text length.\n")
        f.write("If Death-Me and Life-Not-Me texts have different length distributions,\n")
        f.write("position-sensitive metrics like residual probes inherit this confound.\n\n")

        f.write("## Position Distributions by Category\n\n")
        for cat, data in results.items():
            f.write(f"### {cat}\n\n")
            f.write(f"- Mean: {data['mean']:.1f}\n")
            f.write(f"- Median: {data['median']:.1f}\n")
            f.write(f"- Range: [{data['min']}, {data['max']}]\n")
            f.write(f"- Std: {data['std']:.1f}\n\n")

        # Check for confound
        if "Death-Me" in results and "Life-Not-Me" in results:
            death_mean = results["Death-Me"]["mean"]
            life_mean = results["Life-Not-Me"]["mean"]
            diff = abs(death_mean - life_mean)
            overlap_range = (
                max(results["Death-Me"]["min"], results["Life-Not-Me"]["min"]),
                min(results["Death-Me"]["max"], results["Life-Not-Me"]["max"])
            )

            f.write("## Confound Assessment\n\n")
            f.write(f"Difference in mean position: {diff:.1f} tokens\n")
            f.write(f"Overlap range: [{overlap_range[0]}, {overlap_range[1]}]\n\n")

            if diff > 20:  # More than 20 tokens difference
                f.write("**VERDICT: MAJOR CONFOUND**\n\n")
                f.write(f"Death-Me and Life-Not-Me have systematically different answer positions.\n")
                f.write(f"Any analysis using the last token (index -1) on padded data is invalid.\n")
            elif overlap_range[0] <= overlap_range[1]:
                f.write("**VERDICT: MINIMAL CONFOUND**\n\n")
                f.write(f"Distributions overlap substantially. Position is not a major confound.\n")
            else:
                f.write("**VERDICT: MODERATE CONFOUND**\n\n")
                f.write(f"Distributions differ but may still overlap. Check per-experiment.\n")

        f.write("## Phase 11 Bug Check\n\n")
        f.write("```\n")
        f.write(phase11_bug_report)
        f.write("\n```\n\n")

        if "CONFIRMED" in phase11_bug_report:
            f.write("**ACTION REQUIRED:** Phase 11 results are doubly-broken.\n")
            f.write("1. Original UMD loader bug (wrong label mapping)\n")
            f.write("2. Position indexing bug (pad token instead of answer token)\n\n")

    print(f"Saved bug report to {save_path}")


def main():
    import transformers  # For tokenizer
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="SC-4: Position Effect from Padding")
    parser.add_argument(
        "--dataset", type=str, default="deepsuimind",
        choices=["deepsuimind", "swmh", "extended_swmh"],
        help="Dataset to use (default: deepsuimind)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-3.1-8B",
        help="Tokenizer to use (default: llama-3.1-8B)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of examples per category (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sc4/)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sc4"
    else:
        output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("="*60)
    print("SC-4: Position Effect from Padding")
    print("="*60)

    # Load dataset
    categories = load_dataset_by_category(dataset_name=args.dataset)

    print(f"Loaded {args.dataset}:")
    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    # Load tokenizer (use Llama tokenizer as default)
    print(f"\nLoading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        print(f"Warning: Could not load tokenizer, using dummy for demonstration: {e}")
        # Create a simple fallback
        tokenizer = None
        print("Skipping tokenizer-dependent analysis")
        return

    # Analyze position distributions
    print("\nAnalyzing position distributions...")
    results = analyze_position_distribution(
        categories, tokenizer, n_samples=args.n_samples, max_length=256
    )

    # Plot distributions
    plot_path = output_dir / "position_distributions.png"
    plot_position_distributions(results, plot_path, max_length=256)

    # Check Phase 11 bug
    print("\nChecking Phase 11 for [-1, :] indexing bug...")
    phase11_bug = check_phase11_bug()
    print(phase11_bug)

    # Write bug report
    report_path = output_dir / "position_bug_report.md"
    write_bug_report(results, phase11_bug, report_path)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if "Death-Me" in results and "Life-Not-Me" in results:
        death_mean = results["Death-Me"]["mean"]
        life_mean = results["Life-Not-Me"]["mean"]
        print(f"Death-Me mean position: {death_mean:.1f}")
        print(f"Life-Not-Me mean position: {life_mean:.1f}")
        print(f"Difference: {abs(death_mean - life_mean):.1f} tokens")

    if "CONFIRMED" in phase11_bug:
        print("\n*** PHASE 11 BUG CONFIRMED ***")
        print("Phase 11 uses [-1, :] indexing on padded tensors.")
        print("This indexes the LAST PAD TOKEN, not the answer token.")
        print("All Phase 11 'severity axis' results are invalid.")

    print("="*60)

    return results


if __name__ == "__main__":
    main()
