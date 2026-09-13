#!/usr/bin/env python
"""
Export cached datasets to JSONL format for faster offline access.

This creates local JSONL copies of the datasets so analysis scripts
can load them without querying HuggingFace cache.

Usage:
    python cache_datasets.py

This will create:
1. data/implicit_suicide_train.jsonl
2. data/implicit_suicide_validation.jsonl (if available)
3. data/implicit_suicide_test.jsonl (if available)
"""

import os
import json
from pathlib import Path

def export_to_jsonl():
    """Export cached datasets to JSONL format."""
    from datasets import load_dataset

    cache_dir = "data/huggingface_cache"
    dataset_name = "babytreecc/Implicit-suicide-detection"
    output_dir = "data"

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("EXPORTING DATASETS TO JSONL")
    print("="*70)

    splits = ['train', 'validation', 'test']

    for split in splits:
        try:
            print(f"\nLoading {split} split from cache...")
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir
            )

            output_file = f"{output_dir}/implicit_suicide_{split}.jsonl"

            print(f"Exporting to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in dataset:
                    f.write(json.dumps(example) + '\n')

            print(f"  [OK] {split}: {len(dataset)} examples -> {output_file}")

        except Exception as e:
            error_msg = str(e)
            if "Unknown split" in error_msg:
                print(f"  [SKIP] {split}: Not available")
            else:
                print(f"  [FAIL] {split}: {error_msg[:100]}")

    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print("\nDatasets are now available as JSONL files in data/")
    print("\nYou can load them directly:")
    print("  from src.suicide_dataset_loader import load_jsonl")
    print("  dataset = load_jsonl('data/implicit_suicide_train.jsonl')")
    print("\n" + "="*70)

if __name__ == '__main__':
    export_to_jsonl()
