#!/usr/bin/env python
"""
Download and cache all datasets needed for the research project.

This script downloads datasets from HuggingFace and caches them locally
so they're available for all phases without needing internet during analysis.

Usage:
    python download_datasets.py

This will download:
1. babytreecc/Implicit-suicide-detection (train, validation, test splits)
"""

import os
from pathlib import Path

def download_implicit_suicide_dataset():
    """Download the Implicit-suicide-detection dataset from HuggingFace."""
    from datasets import load_dataset

    cache_dir = "data/huggingface_cache"
    dataset_name = "babytreecc/Implicit-suicide-detection"

    os.makedirs(cache_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading: {dataset_name}")
    print(f"{'='*70}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Try to determine available splits
    splits_to_try = ['train', 'validation', 'test']
    datasets = {}

    for split in splits_to_try:
        print(f"Downloading {split} split...")
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir
            )
            datasets[split] = dataset
            print(f"  [OK] {split}: {len(dataset)} examples")
        except Exception as e:
            error_msg = str(e)
            if "Unknown split" in error_msg:
                print(f"  [SKIP] {split}: Not available")
            else:
                print(f"  [FAIL] {split}: {error_msg[:100]}")

    if not datasets:
        print("[FAIL] No splits could be downloaded")
        return False

    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")

    for split, dataset in datasets.items():
        print(f"\n{split.upper()} SPLIT ({len(dataset)} examples)")
        print(f"Columns: {dataset.column_names}")

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample:")
            for key, value in list(sample.items())[:3]:
                val_str = str(value)[:80]
                print(f"  {key}: {val_str}")

        # Label distribution if available
        if 'Intention Category' in dataset.column_names:
            categories = dataset['Intention Category']
            unique_cats = set(categories)
            print(f"Categories: {unique_cats}")

    return True

def verify_cached_datasets():
    """Verify that all datasets are properly cached."""
    from datasets import load_dataset

    cache_dir = "data/huggingface_cache"
    dataset_name = "babytreecc/Implicit-suicide-detection"

    print(f"\n{'='*70}")
    print("Verifying Cached Datasets")
    print(f"{'='*70}")

    splits_to_verify = ['train', 'validation', 'test']
    verified = False

    for split in splits_to_verify:
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir
            )
            print(f"  [OK] {split}: {len(dataset)} examples (cached)")
            verified = True
        except Exception as e:
            error_msg = str(e)
            if "Unknown split" not in error_msg:
                print(f"  [WARN] {split}: {error_msg[:80]}")

    return verified

def main():
    """Main function."""
    print("\n" + "="*70)
    print("DATASET DOWNLOAD & CACHING")
    print("="*70)

    # Download datasets
    print("\nStep 1: Downloading datasets from HuggingFace...")
    if not download_implicit_suicide_dataset():
        print("\n[FAIL] Failed to download datasets")
        return 1

    # Verify cached datasets
    print("\nStep 2: Verifying cached datasets...")
    if not verify_cached_datasets():
        print("\n[FAIL] Failed to verify datasets")
        return 1

    # Success
    print("\n" + "="*70)
    print("SUCCESS: All datasets downloaded and cached!")
    print("="*70)
    print("\nDatasets are now available at: data/huggingface_cache/")
    print("\nYou can now run Phase 1 without needing internet:")
    print("  python analysis/phase1_llama_circuit.py")
    print("\n" + "="*70)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
