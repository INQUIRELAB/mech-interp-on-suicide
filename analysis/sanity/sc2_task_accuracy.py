#!/usr/bin/env python
"""SC-2: Model Task Accuracy - Does Llama answer correctly?

Question: Does Llama actually answer Yes correctly on Death-Me and
No correctly on Life-Not-Me? Logit-difference metrics from Phases 1/2/4/5
are uninterpretable if the model can't do the task.

Output:
    results/sc2/accuracy.json - accuracy metrics, confusion matrix, logit diffs
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.analysis_utils import (
    load_model_wrapped, load_dataset_by_category, get_text,
    build_prompt, build_prompt_tokens, get_yes_no_ids, ensure_dir,
)


def compute_task_accuracy(model, tokenizer, death_me, life_not_me,
                         n_samples=100, max_length=256, device="cuda", prompt_style="standard"):
    """Compute model task accuracy on Death-Me and Life-Not-Me.

    Args:
        model: HookedHFModel
        tokenizer: HuggingFace tokenizer
        death_me: list of Death-Me examples
        life_not_me: list of Life-Not-Me examples
        n_samples: number of examples to test per class
        max_length: max token length
        device: device to run on

    Returns:
        results dict with accuracies, confusion matrix, logit diffs
    """
    yes_id, no_id = get_yes_no_ids(tokenizer)

    # Sample examples
    n_death = min(n_samples, len(death_me))
    n_life = min(n_samples, len(life_not_me))

    results = {
        "death_me": {"correct": 0, "total": n_death, "logit_diffs": [], "predictions": []},
        "life_not_me": {"correct": 0, "total": n_life, "logit_diffs": [], "predictions": []},
    }

    # Confusion matrix: rows=true, cols=pred
    # order: [Death-Me, Life-Not-Me]
    confusion = np.zeros((2, 2), dtype=int)

    print(f"Testing on {n_death} Death-Me and {n_life} Life-Not-Me examples...")

    # Test Death-Me (should predict Yes)
    for i in range(n_death):
        text = get_text(death_me[i])
        input_ids = build_prompt_tokens(
            text, tokenizer, style=prompt_style,
            max_total_tokens=max_length, device=device,
        )

        with torch.no_grad():
            logits = model(input_ids)
            last_logits = logits[0, -1, :]  # [vocab_size]

        yes_logit = last_logits[yes_id].item()
        no_logit = last_logits[no_id].item()
        logit_diff = yes_logit - no_logit

        predicted_yes = yes_logit > no_logit

        results["death_me"]["logit_diffs"].append(logit_diff)
        results["death_me"]["predictions"].append(1 if predicted_yes else 0)

        if predicted_yes:
            results["death_me"]["correct"] += 1
            confusion[0, 0] += 1  # True positive
        else:
            confusion[0, 1] += 1  # False negative

        if (i + 1) % 20 == 0:
            print(f"  Death-Me: {i+1}/{n_death} processed")

    # Test Life-Not-Me (should predict No)
    for i in range(n_life):
        text = get_text(life_not_me[i])
        input_ids = build_prompt_tokens(
            text, tokenizer, style=prompt_style,
            max_total_tokens=max_length, device=device,
        )

        with torch.no_grad():
            logits = model(input_ids)
            last_logits = logits[0, -1, :]

        yes_logit = last_logits[yes_id].item()
        no_logit = last_logits[no_id].item()
        logit_diff = yes_logit - no_logit

        predicted_yes = yes_logit > no_logit

        results["life_not_me"]["logit_diffs"].append(logit_diff)
        results["life_not_me"]["predictions"].append(1 if predicted_yes else 0)

        if not predicted_yes:  # Predicted No correctly
            results["life_not_me"]["correct"] += 1
            confusion[1, 1] += 1  # True negative
        else:
            confusion[1, 0] += 1  # False positive

        if (i + 1) % 20 == 0:
            print(f"  Life-Not-Me: {i+1}/{n_life} processed")

    # Compute statistics
    death_acc = results["death_me"]["correct"] / results["death_me"]["total"]
    life_acc = results["life_not_me"]["correct"] / results["life_not_me"]["total"]
    overall_acc = (results["death_me"]["correct"] + results["life_not_me"]["correct"]) / (results["death_me"]["total"] + results["life_not_me"]["total"])

    death_ld_mean = np.mean(results["death_me"]["logit_diffs"])
    death_ld_std = np.std(results["death_me"]["logit_diffs"])
    life_ld_mean = np.mean(results["life_not_me"]["logit_diffs"])
    life_ld_std = np.std(results["life_not_me"]["logit_diffs"])

    summary = {
        "death_me_accuracy": float(death_acc),
        "life_not_me_accuracy": float(life_acc),
        "overall_accuracy": float(overall_acc),
        "confusion_matrix": confusion.tolist(),
        "death_me_logit_diff_mean": float(death_ld_mean),
        "death_me_logit_diff_std": float(death_ld_std),
        "life_not_me_logit_diff_mean": float(life_ld_mean),
        "life_not_me_logit_diff_std": float(life_ld_std),
        "n_death_me_tested": n_death,
        "n_life_not_me_tested": n_life,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="SC-2: Model Task Accuracy")
    parser.add_argument(
        "--dataset", type=str, default="deepsuimind",
        choices=["deepsuimind", "swmh", "extended_swmh"],
        help="Dataset to use (default: deepsuimind)"
    )
    parser.add_argument(
        "--model", type=str, default="llama",
        choices=["llama", "qwen", "mistral"],
        help="Model to test (default: llama)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of examples to test per class (default: 100)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sc2/)"
    )
    parser.add_argument(
        "--style", type=str, default="standard",
        choices=["standard", "empathetic", "clinical"],
        help="Prompt style (default: standard)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sc2"
    else:
        output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("="*60)
    print(f"SC-2: Model Task Accuracy ({args.model})")
    print("="*60)

    # Load dataset
    categories = load_dataset_by_category(dataset_name=args.dataset)

    death_me = categories.get("Death-Me", [])
    life_not_me = categories.get("Life-Not-Me", [])

    if not death_me or not life_not_me:
        raise ValueError(f"Need both Death-Me and Life-Not-Me examples. "
                        f"Got Death-Me: {len(death_me)}, Life-Not-Me: {len(life_not_me)}")

    print(f"Loaded {args.dataset}:")
    print(f"  Death-Me: {len(death_me)} examples")
    print(f"  Life-Not-Me: {len(life_not_me)} examples")

    # Load model
    print(f"\nLoading {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_wrapped(args.model, device=device)

    # Compute accuracy
    results = compute_task_accuracy(
        model, tokenizer, death_me, life_not_me,
        n_samples=args.n_samples, device=device, prompt_style=args.style,
    )

    # Save results
    json_path = output_dir / "accuracy.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Death-Me accuracy (P(Yes|Death-Me)): {results['death_me_accuracy']:.4f}")
    print(f"Life-Not-Me accuracy (P(No|Life-Not-Me)): {results['life_not_me_accuracy']:.4f}")
    print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred [Yes, No]):")
    print(f"  Death-Me:    [{results['confusion_matrix'][0][0]:3d}  {results['confusion_matrix'][0][1]:3d}]")
    print(f"  Life-Not-Me: [{results['confusion_matrix'][1][0]:3d}  {results['confusion_matrix'][1][1]:3d}]")
    print(f"\nLogit difference (Yes - No):")
    print(f"  Death-Me:    {results['death_me_logit_diff_mean']:>8.4f} +/- {results['death_me_logit_diff_std']:.4f}")
    print(f"  Life-Not-Me: {results['life_not_me_logit_diff_mean']:>8.4f} +/- {results['life_not_me_logit_diff_std']:.4f}")

    # Decision guidance
    death_acc = results["death_me_accuracy"]
    life_acc = results["life_not_me_accuracy"]

    print("\nDecision tree per plan/00_sanity_checks.md:")
    min_acc = min(death_acc, life_acc)
    if min_acc >= 0.85:
        print("  -> Both accuracies >= 0.85: Metrics are reliable; proceed")
    elif min_acc >= 0.60:
        print(f"  -> Accuracy 0.60-0.85: Metrics are noisy but interpretable")
        print("     -> Flag baseline accuracy={min_acc:.3f} in all downstream plots")
    else:
        print(f"  -> Accuracy < 0.60: Model can't do the task reliably")
        print("     -> Either change prompt (try 'clinical' or 'empathetic') or change model")
        print("     -> DO NOT proceed with current setup")

    print("="*60)

    return results


if __name__ == "__main__":
    main()
