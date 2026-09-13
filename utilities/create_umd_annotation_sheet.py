"""
Create an Excel sheet with random equal samples of UMD data for LLM annotation.

Outputs: data/umd_annotation_sheet.xlsx with columns:
- sample_id
- umd_label (original a/b/c/d/None)
- binary_label (suicidal/not_suicidal) - pre-filled
- text
- llm_prediction (to be filled)
- llm_confidence (to be filled)
- llm_reasoning (to be filled)
"""

import pandas as pd
import random
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis_utils import _load_umd

# Binary mapping (conservative: c+d = suicidal, a+b+None = not suicidal)
SUICIDAL_LABELS = {"c", "d"}  # moderate, severe
NOT_SUICIDAL_LABELS = {"a", "b", "None"}  # no risk, low risk, control

def load_umd_binary(source="expert", max_chars_per_user=4000):
    """Load UMD data and return flat list with binary labels."""
    categories = _load_umd(split="train", source=source, max_chars_per_user=max_chars_per_user)

    records = []
    for cat, examples in categories.items():
        records.extend(examples)

    # Add binary_label to each record
    for record in records:
        umd_label = record.get("umd_label", "None")
        if umd_label in SUICIDAL_LABELS:
            record["binary_label"] = "suicidal"
        else:
            record["binary_label"] = "not_suicidal"

    print(f"\nTotal records: {len(records)}")
    suicidal = [r for r in records if r["binary_label"] == "suicidal"]
    not_suicidal = [r for r in records if r["binary_label"] == "not_suicidal"]
    print(f"  Suicidal (c+d): {len(suicidal)}")
    print(f"  Not suicidal (a+b+None): {len(not_suicidal)}")

    return records


def sample_equal(records, n_per_class=50, seed=42):
    """Sample equal numbers from suicidal and not_suicidal classes."""
    random.seed(seed)

    suicidal = [r for r in records if r["binary_label"] == "suicidal"]
    not_suicidal = [r for r in records if r["binary_label"] == "not_suicidal"]

    min_available = min(len(suicidal), len(not_suicidal))
    if n_per_class > min_available:
        print(f"Warning: requested {n_per_class} per class, but only {min_available} available.")
        n_per_class = min_available

    suicidal_sample = random.sample(suicidal, n_per_class)
    not_suicidal_sample = random.sample(not_suicidal, n_per_class)

    return suicidal_sample + not_suicidal_sample


def create_annotation_excel(sampled_records, output_path):
    """Create Excel file with annotation columns."""
    rows = []
    for i, record in enumerate(sampled_records):
        rows.append({
            "sample_id": i + 1,
            "umd_label": record.get("umd_label", "None"),
            "severity": record.get("severity", 0),
            "binary_label": record["binary_label"],  # Ground truth (for validation)
            "user_id": record.get("user_id", "unknown"),
            "n_posts": record.get("n_posts", 1),
            "text": record.get("text", "")[:4000],  # Truncate for Excel
            # Columns to be filled by LLM
            "llm_prediction": "",  # suicidal / not_suicidal
            "llm_confidence": "",  # high / medium / low
            "llm_reasoning": "",
            "llm_correct": "",  # YES / NO (calculated after)
        })

    df = pd.DataFrame(rows)

    # Reorder columns for better workflow
    cols = ["sample_id", "umd_label", "severity", "binary_label", "text",
            "llm_prediction", "llm_confidence", "llm_reasoning", "llm_correct"]
    df = df[cols]

    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n[OK] Exported {len(df)} rows to {output_path}")

    # Print summary
    print("\n[STATS] Sample distribution:")
    print(df.groupby(["binary_label", "umd_label"]).size().unstack(fill_value=0))

    return df



def _data_dir():
    """Data directory: $DATA_DIR if set, else <repo root>/data.

    Keeps this script runnable on any machine; the UMD files it reads are
    access-controlled and must be placed there by the user.
    """
    import os
    from pathlib import Path
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"

def main():
    output_dir = _data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "umd_annotation_sheet.xlsx"

    print("Loading UMD dataset...")
    records = load_umd_binary(source="expert", max_chars_per_user=2000)

    print("\nSampling equal distribution...")
    sampled = sample_equal(records, n_per_class=50, seed=42)

    print("\nCreating Excel annotation sheet...")
    df = create_annotation_excel(sampled, output_path)

    print(f"\n[DONE] Next steps:")
    print(f"1. Open the Excel file: {output_path}")
    print(f"2. Use the 'text' column as input to your LLM")
    print(f"3. Fill in: llm_prediction, llm_confidence, llm_reasoning")
    print(f"4. Compare llm_prediction vs binary_label for accuracy")


if __name__ == "__main__":
    main()
