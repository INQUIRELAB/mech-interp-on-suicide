#!/usr/bin/env python
"""SC-1: Lexical Baseline - TF-IDF + Logistic Regression.

Question: Can a TF-IDF + logistic-regression classifier distinguish
Death-Me from Life-Not-Me? If yes (≥90% accuracy), the model isn't
doing anything that requires MI to explain.

Output:
    results/sc1/baseline_results.json - accuracy, top ngrams per class
    results/sc1/top_ngrams.txt - most predictive n-grams
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

from src.analysis_utils import load_dataset_by_category, ensure_dir


def extract_text(example):
    """Extract text from a dataset example."""
    for key in ["text", "content", "message"]:
        if key in example and example[key]:
            return example[key]
    return str(example)


def run_tfidf_baseline(death_me, life_not_me, n_folds=5, max_features=5000):
    """Run TF-IDF + LR baseline with k-fold cross-validation.

    Args:
        death_me: list of Death-Me examples
        life_not_me: list of Life-Not-Me examples
        n_folds: number of cross-validation folds
        max_features: max features for TF-IDF

    Returns:
        results dict with accuracy, std, top ngrams
    """
    # Build (text, label) pairs: Death-Me -> 1, Life-Not-Me -> 0
    texts = []
    labels = []

    for ex in death_me:
        text = extract_text(ex).strip()
        if text:
            texts.append(text)
            labels.append(1)

    for ex in life_not_me:
        text = extract_text(ex).strip()
        if text:
            texts.append(text)
            labels.append(0)

    print(f"Built dataset: {len(texts)} examples (Death-Me: {sum(labels)}, Life-Not-Me: {len(labels) - sum(labels)})")

    if len(texts) < 50:
        raise ValueError(f"Too few examples: {len(texts)}. Need at least 50 for meaningful CV.")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english',
    )

    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    print(f"TF-IDF features: {X.shape[1]}")

    # Logistic regression with cross-validation
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"5-fold CV accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    # Fit on full data to get feature importances
    clf.fit(X, y)

    # Get top predictive n-grams per class
    feature_names = vectorizer.get_feature_names_out()
    coefficients = clf.coef_[0]

    # Positive coef -> predicts Death-Me (label 1)
    # Negative coef -> predicts Life-Not-Me (label 0)
    death_me_indices = np.argsort(coefficients)[-20:][::-1]  # Top 20 positive
    life_not_me_indices = np.argsort(coefficients)[:20]  # Top 20 negative (most negative)

    top_death_me = [(feature_names[i], coefficients[i]) for i in death_me_indices]
    top_life_not_me = [(feature_names[i], coefficients[i]) for i in life_not_me_indices]

    results = {
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "n_examples": len(texts),
        "n_death_me": int(sum(labels)),
        "n_life_not_me": int(len(labels) - sum(labels)),
        "top_death_me_ngrams": [(word, float(coef)) for word, coef in top_death_me],
        "top_life_not_me_ngrams": [(word, float(coef)) for word, coef in top_life_not_me],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="SC-1: Lexical Baseline (TF-IDF + LR)")
    parser.add_argument(
        "--dataset", type=str, default="deepsuimind",
        choices=["deepsuimind", "swmh", "extended_swmh"],
        help="Dataset to use (default: deepsuimind)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sc1/)"
    )
    parser.add_argument(
        "--max-features", type=int, default=5000,
        help="Max TF-IDF features (default: 5000)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sc1"
    else:
        output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("="*60)
    print("SC-1: Lexical Baseline (TF-IDF + Logistic Regression)")
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

    # Run baseline
    results = run_tfidf_baseline(death_me, life_not_me, max_features=args.max_features)

    # Save JSON results
    json_path = output_dir / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Save top n-grams as readable text
    txt_path = output_dir / "top_ngrams.txt"
    with open(txt_path, "w") as f:
        f.write("SC-1: Top Predictive N-grams\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})\n\n")

        f.write("Top 20 n-grams predicting Death-Me (implicit suicidal ideation):\n")
        f.write("-"*60 + "\n")
        for word, coef in results["top_death_me_ngrams"]:
            f.write(f"  {coef:>8.4f}  {word}\n")

        f.write("\nTop 20 n-grams predicting Life-Not-Me (no death association):\n")
        f.write("-"*60 + "\n")
        for word, coef in results["top_life_not_me_ngrams"]:
            f.write(f"  {coef:>8.4f}  {word}\n")

    print(f"Saved top n-grams to {txt_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TF-IDF + LR accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")

    # Decision guidance
    acc = results["mean_accuracy"]
    print("\nDecision tree per plan/00_sanity_checks.md:")
    if acc >= 0.90:
        print("  -> ACCURACY >= 0.90: Project framing must shift.")
        print("     The 'circuit' is at most keyword detection.")
        print("     -> PIVOT to Tier 3 (refusal-circuit reframing)")
    elif acc >= 0.75:
        print("  -> ACCURACY 0.75-0.90: Baseline non-trivial but model presumably does more.")
        print("     -> GO to Tier 1, but specificity test becomes critical")
    else:
        print("  -> ACCURACY < 0.75: Task genuinely requires inference.")
        print("     -> GO to Tier 1 with full confidence")

    print("="*60)

    return results


if __name__ == "__main__":
    main()
