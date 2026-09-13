"""
Load and process the Implicit Suicide Detection dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/babytreecc/Implicit-suicide-detection

This dataset contains real examples of implicit and explicit suicide-related
text, perfect for mechanistic interpretability research on safety failures.
"""

from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import json


class ImplicitSuicideDataset(Dataset):
    """
    Load HuggingFace 'Implicit-suicide-detection' dataset.

    Dataset structure (expected):
    - text: The input text/dialogue
    - label: 0=non-suicidal, 1=suicidal ideation
    - implicit_level: 0=explicit, 1=implicit (if available)

    Features:
    - Automatic download from HuggingFace
    - Stratified splits (train/val/test)
    - Clean/corrupt prompt generation
    - Integration with mechanistic analysis
    """

    def __init__(
        self,
        tokenizer=None,
        split: str = "train",
        cache_dir: str = None,
        implicit_only: bool = False,
    ):
        """
        Load dataset from HuggingFace.

        Args:
            tokenizer: HuggingFace tokenizer (optional, for tokenization)
            split: 'train', 'validation', or 'test'
            cache_dir: Where to cache downloaded dataset
            implicit_only: Filter to only implicit cases (if label available)
        """
        self.tokenizer = tokenizer
        self.implicit_only = implicit_only

        print(f"Loading Implicit Suicide Detection dataset ({split} split)...")

        try:
            # Load from HuggingFace
            self.dataset = load_dataset(
                "babytreecc/Implicit-suicide-detection",
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you have internet connection and HuggingFace datasets library installed")
            print("Install with: pip install datasets")
            raise

        print(f"Loaded {len(self.dataset)} examples")

        # Explore structure
        self._explore_dataset()

        # Filter if needed
        if implicit_only:
            self._filter_implicit_only()

    def _explore_dataset(self):
        """Print dataset structure and statistics."""
        print("\n=== Dataset Structure ===")
        print(f"Total examples: {len(self.dataset)}")
        print(f"Columns: {self.dataset.column_names}")

        # Show sample
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            print(f"\nSample example:")
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

        # Label distribution
        if "label" in self.dataset.column_names:
            labels = self.dataset["label"]
            print(f"\nLabel distribution:")
            print(f"  Class 0 (non-suicidal): {labels.count(0)}")
            print(f"  Class 1 (suicidal): {labels.count(1)}")

    def _filter_implicit_only(self):
        """Filter to implicit cases if implicit_level available."""
        if "implicit_level" in self.dataset.column_names:
            original_size = len(self.dataset)
            # Keep only high implicit level (>0.5)
            self.dataset = self.dataset.filter(
                lambda x: x.get("implicit_level", 0) > 0.5
            )
            print(f"Filtered to implicit cases: {len(self.dataset)}/{original_size}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def build_clean_corrupt_pair(
        self,
        idx: int,
        prompt_template: str = "standard",
    ) -> Tuple[str, str]:
        """
        Create clean/corrupt prompt pair for activation patching.

        Clean: Dialogue with suicide ideation present
        Corrupt: Same dialogue with suicide cues removed

        Args:
            idx: Dataset index
            prompt_template: "standard" or "with_labels"

        Returns:
            Tuple of (clean_prompt, corrupt_prompt) strings
        """
        example = self.dataset[idx]

        # Extract text (column name varies)
        text_key = self._find_text_column()
        text = example[text_key]

        # Determine label (clean=suicidal, corrupt=non-suicidal)
        label = example.get("label", 1)  # Default to suicidal

        if prompt_template == "standard":
            if label == 1:
                # Already suicidal: clean stays, corrupt removes cues
                clean = f"Text: {text}\nTask: Identify if this shows suicidal ideation.\nAnswer: Yes"
                corrupt = self._remove_suicide_cues(text)
                corrupt = f"Text: {corrupt}\nTask: Identify if this shows suicidal ideation.\nAnswer: No"
            else:
                # Non-suicidal case: use as corrupt, generate clean
                corrupt = f"Text: {text}\nTask: Identify if this shows suicidal ideation.\nAnswer: No"
                clean = self._add_suicide_cues(text)
                clean = f"Text: {clean}\nTask: Identify if this shows suicidal ideation.\nAnswer: Yes"

        elif prompt_template == "with_labels":
            implicit_level = example.get("implicit_level", 0.5)
            level_desc = "implicit" if implicit_level > 0.5 else "explicit"

            if label == 1:
                clean = (
                    f"Type: {level_desc}\n"
                    f"Text: {text}\n"
                    f"Assessment: Shows suicidal ideation"
                )
                corrupt = (
                    f"Type: {level_desc}\n"
                    f"Text: {self._remove_suicide_cues(text)}\n"
                    f"Assessment: No suicidal ideation"
                )
            else:
                corrupt = (
                    f"Type: {level_desc}\n"
                    f"Text: {text}\n"
                    f"Assessment: No suicidal ideation"
                )
                clean = (
                    f"Type: {level_desc}\n"
                    f"Text: {self._add_suicide_cues(text)}\n"
                    f"Assessment: Shows suicidal ideation"
                )
        else:
            raise ValueError(f"Unknown template: {prompt_template}")

        return clean, corrupt

    def _find_text_column(self) -> str:
        """Find the text column (varies by dataset version)."""
        possible_names = ["text", "content", "message", "utterance", "input"]
        for name in possible_names:
            if name in self.dataset.column_names:
                return name
        # Fallback to first string column
        for col in self.dataset.column_names:
            if isinstance(self.dataset[0][col], str):
                return col
        raise ValueError("Could not find text column")

    def _remove_suicide_cues(self, text: str) -> str:
        """Remove suicide-related keywords from text."""
        import re

        suicide_keywords = [
            r"\bdie\b", r"\bdead\b", r"\bdeath\b", r"\bsuicide\b",
            r"\bkill\b", r"\bkilling\b", r"\bharm\b",
            r"\bburden\b", r"\bworthless\b", r"\bhopeless\b",
            r"\bcan't\s+go\s+on\b", r"\bdon't\s+want\s+to\s+live\b",
            r"\bbetter\s+off\s+dead\b", r"\bwish\s+i\s+was\s+dead\b",
            r"\bnoose\b", r"\bpills\b", r"\bslash\b", r"\bjump\b",
            r"\bforever\b", r"\bend\s+it\b", r"\blast\b", r"\bfinal\b",
        ]

        result = text
        for keyword in suicide_keywords:
            result = re.sub(keyword, "", result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = " ".join(result.split())

        return result if result else "[removed]"

    def _add_suicide_cues(self, text: str) -> str:
        """Add suicide-related context to non-suicidal text (for testing)."""
        # For non-suicidal cases, prepend context
        cues = [
            "I've been feeling like a burden lately. ",
            "Sometimes I think about giving up. ",
            "I don't see a future for myself. ",
        ]
        import random
        return random.choice(cues) + text

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total_examples": len(self.dataset),
            "columns": self.dataset.column_names,
        }

        if "label" in self.dataset.column_names:
            labels = self.dataset["label"]
            stats["suicidal_count"] = labels.count(1)
            stats["non_suicidal_count"] = labels.count(0)
            stats["suicidal_ratio"] = labels.count(1) / len(labels)

        if "implicit_level" in self.dataset.column_names:
            implicit_levels = self.dataset["implicit_level"]
            stats["mean_implicit_level"] = sum(implicit_levels) / len(implicit_levels)

        return stats

    def to_jsonl(self, output_path: str):
        """Export dataset to JSONL format (for caching)."""
        with open(output_path, "w") as f:
            for example in self.dataset:
                f.write(json.dumps(example) + "\n")
        print(f"Exported {len(self.dataset)} examples to {output_path}")


def load_huggingface_suicide_dataset(
    tokenizer=None,
    split: str = "train",
    cache_dir: str = "data/huggingface_cache",
    implicit_only: bool = False,
) -> ImplicitSuicideDataset:
    """
    Convenience function to load the dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        split: 'train', 'validation', 'test'
        cache_dir: Cache directory for HuggingFace
        implicit_only: Filter to implicit cases only

    Returns:
        ImplicitSuicideDataset instance
    """
    return ImplicitSuicideDataset(
        tokenizer=tokenizer,
        split=split,
        cache_dir=cache_dir,
        implicit_only=implicit_only,
    )


if __name__ == "__main__":
    """Test dataset loading and exploration."""

    print("=" * 80)
    print("Testing Implicit Suicide Detection Dataset")
    print("=" * 80)

    # Load train set
    dataset = ImplicitSuicideDataset(split="train")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Show examples
    print("\n=== Sample Clean/Corrupt Pairs ===")
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i+1}:")
        clean, corrupt = dataset.build_clean_corrupt_pair(i)
        print(f"Clean:\n  {clean[:100]}...")
        print(f"Corrupt:\n  {corrupt[:100]}...")

    # Test export
    print("\n=== Exporting to JSONL ===")
    dataset.to_jsonl("data/implicit_suicide_detection_train.jsonl")

    print("\n✓ Dataset loaded and tested successfully!")
