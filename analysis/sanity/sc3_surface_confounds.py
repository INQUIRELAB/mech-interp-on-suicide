#!/usr/bin/env python
"""SC-3: Surface Confounds - Length, First-Person, Emotional Density.

Question: Are Death-Me texts systematically longer / more emotional /
more first-person than Life-Not-Me? If so, the "circuit" might be
detecting those surface features.

Output:
    results/sc3/confounds.json - statistics per category
    results/sc3/distributions.png - histograms overlaid
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis_utils import load_dataset_by_category, get_text, ensure_dir


# Emotional lexicon (inline, per plan)
EMOTIONAL_LEXICON = {
    # Sadness/hopelessness
    "sad", "hopeless", "hopelessness", "despair", "desperate", "desperation",
    "empty", "emptiness", "numb", "numbness", "hollow", "void",

    # Pain/suffering
    "hurt", "pain", "painful", "suffering", "suffer", "agony", "ache",
    "wounded", "bleeding", "broken", "shattered", "torn",

    # Loneliness/isolation
    "alone", "lonely", "loneliness", "isolated", "isolation", "abandoned",
    "forsaken", "nobody", "nowhere", "invisible", "unseen",

    # Death-related (but not explicit suicide methods)
    "death", "die", "dying", "dead", "end", "ending", "gone", "loss", "lost",
    "fade", "fading", "darkness", "dark", "black", "abyss",

    # Burden/guilt
    "burden", "worthless", "useless", "failure", "fail", "failed", "guilt",
    "ashamed", "shame", "regret", "mistake", "unworthy",

    # Escape/release language
    "escape", "release", "relief", "give up", "let go", "done", "finished",
    "can't go on", "no way out", "trapped", "stuck",
}

# First-person pronouns (lowercased)
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}


def compute_text_statistics(text):
    """Compute surface statistics for a single text.

    Returns:
        dict with char_length, token_count (approximate), first_person_ratio,
        emotional_density
    """
    text = text.strip()
    char_length = len(text)

    # Approximate token count by splitting (good enough for confound check)
    words = text.lower().split()
    token_count = len(words)

    if token_count == 0:
        return {
            "char_length": char_length,
            "token_count": 0,
            "first_person_ratio": 0.0,
            "emotional_density": 0.0,
        }

    # First-person pronoun ratio
    fp_count = sum(1 for w in words if w in FIRST_PERSON)
    first_person_ratio = fp_count / token_count

    # Emotional word density
    emo_count = sum(1 for w in words if w in EMOTIONAL_LEXICON)
    emotional_density = emo_count / token_count

    return {
        "char_length": char_length,
        "token_count": token_count,
        "first_person_ratio": first_person_ratio,
        "emotional_density": emotional_density,
    }


def analyze_confounds(categories):
    """Compute confound statistics per category.

    Args:
        categories: dict with category names -> list of examples

    Returns:
        results dict with statistics per category
    """
    all_stats = {}
    for cat_name, examples in categories.items():
        if not examples:
            continue

        stats_list = []
        for ex in examples:
            text = get_text(ex)
            if text and text.strip():
                stats = compute_text_statistics(text)
                stats_list.append(stats)

        if not stats_list:
            continue

        # Compute summary statistics
        arrays = {
            "char_length": np.array([s["char_length"] for s in stats_list]),
            "token_count": np.array([s["token_count"] for s in stats_list]),
            "first_person_ratio": np.array([s["first_person_ratio"] for s in stats_list]),
            "emotional_density": np.array([s["emotional_density"] for s in stats_list]),
        }

        cat_stats = {}
        for metric, values in arrays.items():
            cat_stats[f"{metric}_mean"] = float(values.mean())
            cat_stats[f"{metric}_std"] = float(values.std())
            cat_stats[f"{metric}_median"] = float(np.median(values))
            cat_stats[f"{metric}_q25"] = float(np.percentile(values, 25))
            cat_stats[f"{metric}_q75"] = float(np.percentile(values, 75))
            cat_stats[f"{metric}_values"] = values.tolist()  # For plotting

        cat_stats["n_examples"] = len(stats_list)
        all_stats[cat_name] = cat_stats

        print(f"  {cat_name}: {len(stats_list)} examples analyzed")

    return all_stats


def plot_distributions(all_stats, save_path):
    """Plot overlaid histograms for each metric by category.

    Args:
        all_stats: dict from analyze_confounds
        save_path: where to save the plot
    """
    categories = list(all_stats.keys())
    metrics = ["char_length", "token_count", "first_person_ratio", "emotional_density"]
    metric_labels = ["Character Length", "Token Count", "First-Person Ratio", "Emotional Density"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {"Death-Me": "#e74c3c", "Death-Not-Me": "#f39c12", "Life-Not-Me": "#3498db"}

    for ax, metric, label in zip(axes, metrics, metric_labels):
        for cat in categories:
            if cat not in all_stats:
                continue
            values = all_stats[cat][f"{metric}_values"]
            ax.hist(values, bins=30, alpha=0.5, label=cat, color=colors.get(cat, None), density=True)

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} by Category")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("SC-3: Surface Confound Distributions", fontsize=14)
    fig.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close(fig)


def print_confound_report(all_stats):
    """Print summary and decision guidance."""
    print("\n" + "="*60)
    print("CONFOUND ANALYSIS")
    print("="*60)

    cats = list(all_stats.keys())

    # Check for length confound
    if "Death-Me" in all_stats and "Life-Not-Me" in all_stats:
        death_len = all_stats["Death-Me"]["char_length_mean"]
        life_len = all_stats["Life-Not-Me"]["char_length_mean"]
        ratio = death_len / life_len if life_len > 0 else float('inf')

        print(f"\nMean character length:")
        for cat in cats:
            mean = all_stats[cat]["char_length_mean"]
            std = all_stats[cat]["char_length_std"]
            print(f"  {cat}: {mean:.1f} +/- {std:.1f}")

        print(f"\n  Ratio Death-Me / Life-Not-Me: {ratio:.2f}x")

        if ratio > 1.5 or ratio < 1/1.5:
            print("  -> MAJOR CONFOUND: Length differs by >1.5x")
            print("     Length-sensitive metrics (residual norm, attention entropy) are contaminated")
        else:
            print("  -> No major length confound")

        # Check first-person ratio
        print(f"\nMean first-person ratio:")
        for cat in cats:
            mean = all_stats[cat]["first_person_ratio_mean"]
            std = all_stats[cat]["first_person_ratio_std"]
            print(f"  {cat}: {mean:.4f} +/- {std:.4f}")

        death_fp = all_stats["Death-Me"]["first_person_ratio_mean"]
        life_fp = all_stats["Life-Not-Me"]["first_person_ratio_mean"]
        fp_ratio = death_fp / life_fp if life_fp > 0 else float('inf')

        print(f"\n  Ratio Death-Me / Life-Not-Me: {fp_ratio:.2f}x")

        if fp_ratio > 2.0:
            print("  -> First-person differs by >2x: semantically expected but flag")
        else:
            print("  -> First-person usage comparable")

        # Check emotional density
        print(f"\nMean emotional density:")
        for cat in cats:
            mean = all_stats[cat]["emotional_density_mean"]
            std = all_stats[cat]["emotional_density_std"]
            print(f"  {cat}: {mean:.4f} +/- {std:.4f}")

        death_emo = all_stats["Death-Me"]["emotional_density_mean"]
        life_emo = all_stats["Life-Not-Me"]["emotional_density_mean"]
        emo_ratio = death_emo / life_emo if life_emo > 0 else float('inf')

        print(f"\n  Ratio Death-Me / Life-Not-Me: {emo_ratio:.2f}x")

        if emo_ratio > 2.0:
            print("  -> Emotional density differs by >2x: expected for task")
        else:
            print("  -> Emotional density comparable")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="SC-3: Surface Confounds")
    parser.add_argument(
        "--dataset", type=str, default="deepsuimind",
        choices=["deepsuimind", "swmh", "extended_swmh"],
        help="Dataset to use (default: deepsuimind)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sc3/)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sc3"
    else:
        output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("="*60)
    print("SC-3: Surface Confounds (Length, First-Person, Emotion)")
    print("="*60)

    # Load dataset
    categories = load_dataset_by_category(dataset_name=args.dataset)

    print(f"Loaded {args.dataset}:")
    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    # Analyze confounds
    all_stats = analyze_confounds(categories)

    # Save JSON (without raw value arrays for readability)
    json_path = output_dir / "confounds.json"
    save_stats = {}
    for cat, stats in all_stats.items():
        save_stats[cat] = {k: v for k, v in stats.items() if not k.endswith("_values")}
    with open(json_path, "w") as f:
        json.dump(save_stats, f, indent=2)
    print(f"\nSaved statistics to {json_path}")

    # Plot distributions
    plot_path = output_dir / "distributions.png"
    plot_distributions(all_stats, plot_path)

    # Print report
    print_confound_report(all_stats)

    return all_stats


if __name__ == "__main__":
    main()
