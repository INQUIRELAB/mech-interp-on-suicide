#!/usr/bin/env python
"""Concept registry for the Gated Causal-Ladder (GCL) framework.

The framework is concept-agnostic: every rung of the ladder (gate, probe,
diff-of-means, ablation, rank-k, steering) depends only on (a) two text classes,
(b) a yes/no prompt, (c) the two answer tokens. A *concept* bundles exactly that.
Adding a new concept = registering one ConceptConfig; the driver (run_gcl.py)
does the rest unchanged. That is what makes this a framework rather than a
one-off analysis.

Registered concepts:
  suicide   -- binary suicide vs non-suicide  (07_Suicide_Watch_232K)   [reproduces the paper]
  toxicity  -- toxic vs clean comments        (civil_comments)          [second concept]

Each loader returns (pos_texts, neg_texts) already shuffled and length-filtered.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MIN_CHARS = 80


@dataclass
class ConceptConfig:
    name: str
    display_name: str
    # loader(n_per_class, seed) -> (pos_texts, neg_texts)
    loader: Callable[[int, int], Tuple[List[str], List[str]]]
    prompt_prefix: str          # text before {text}
    question: str               # the yes/no question line
    pos_answer: str = " Yes"    # answer token for the positive class
    neg_answer: str = " No"
    peak_layer_hint: int = 16   # where to read the direction (Llama-3.1-8B)
    notes: str = ""

    def build_prompt(self, text: str) -> str:
        return (f"{self.prompt_prefix}\n\nText: {text}\n\n{self.question} "
                f"Answer with Yes or No.\nAnswer:")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _clean(texts, n, seed):
    rng = np.random.default_rng(seed)
    out = [t for t in texts if t and len(t) > MIN_CHARS]
    rng.shuffle(out)
    return out[:n]


def _load_suicide(n_per_class, seed=0):
    from datasets import load_from_disk
    ds = load_from_disk(str(PROJECT_ROOT / "data" / "datasets" /
                            "07_Suicide_Watch_232K" / "hf_cache"))["train"]
    texts, labels = ds["text"], ds["label"]
    pos = _clean([t for t, l in zip(texts, labels) if l == "suicide"], n_per_class, seed)
    neg = _clean([t for t, l in zip(texts, labels) if l == "non-suicide"], n_per_class, seed + 1)
    return pos, neg


def _load_sentiment(n_per_class, seed=0):
    """Positive vs negative movie reviews (IMDB). Affective valence; long texts."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train")
    txt, lab = ds["text"], ds["label"]
    pos = _clean([t for t, l in zip(txt, lab) if l == 1], n_per_class, seed)
    neg = _clean([t for t, l in zip(txt, lab) if l == 0], n_per_class, seed + 1)
    return pos, neg


def _load_topic_sports(n_per_class, seed=0):
    """Sports vs other news topics (AG News). A LOW-affect, neutral-topic concept:
    the critical control for whether the steering-salience confound is affect-specific."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="train")
    txt, lab = ds["text"], ds["label"]      # 0 World, 1 Sports, 2 Business, 3 Sci/Tech
    pos = _clean([t for t, l in zip(txt, lab) if l == 1], n_per_class, seed)
    neg = _clean([t for t, l in zip(txt, lab) if l != 1], n_per_class, seed + 1)
    return pos, neg


def _load_topic_business(n_per_class, seed=0):
    """Business vs other news topics (AG News). Second neutral-topic control."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="train")
    txt, lab = ds["text"], ds["label"]
    pos = _clean([t for t, l in zip(txt, lab) if l == 2], n_per_class, seed)
    neg = _clean([t for t, l in zip(txt, lab) if l != 2], n_per_class, seed + 1)
    return pos, neg


def _load_suicide_vs_depression(n_per_class, seed=0):
    """MATCHED-NEGATIVE: r/SuicideWatch vs r/depression, both from SWMH.

    Reviewer #1: the 232K task pits r/SuicideWatch (adult crisis register) against
    r/teenagers (casual teen register), so its headline AUCs are inflated by a
    register/population gap, not pure suicidality. Here BOTH classes are distressed
    mental-health Reddit, so the register is matched and the contrast isolates
    suicidality-vs-depression rather than crisis-adult-vs-casual-teen. Running the full
    ladder on THIS concept is the primary matched-negative analysis the review asks for.
    """
    import json
    path = PROJECT_ROOT / "data" / "datasets" / "01_SWMH_Reddit" / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"SWMH not found: {path}")
    sui, dep = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lab = r.get("label", "")
            txt = (r.get("text") or "").strip()
            if lab == "self.SuicideWatch":
                sui.append(txt)
            elif lab == "self.depression":
                dep.append(txt)
    return _clean(sui, n_per_class, seed), _clean(dep, n_per_class, seed + 1)


def _load_toxicity(n_per_class, seed=0):
    """Toxic vs clean comments from civil_comments (HF).

    pos = toxicity >= 0.5 (clearly toxic); neg = toxicity == 0 (clearly clean).
    A streamed scan keeps memory bounded; we stop once both pools are full.
    """
    from datasets import load_dataset
    need = n_per_class * 4  # over-collect before length filter + shuffle
    pos_raw, neg_raw = [], []
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    for row in ds:
        tox = row.get("toxicity", 0.0)
        txt = (row.get("text") or "").strip()
        if not txt or len(txt) <= MIN_CHARS:
            continue
        if tox >= 0.5 and len(pos_raw) < need:
            pos_raw.append(txt)
        elif tox == 0.0 and len(neg_raw) < need:
            neg_raw.append(txt)
        if len(pos_raw) >= need and len(neg_raw) >= need:
            break
    pos = _clean(pos_raw, n_per_class, seed)
    neg = _clean(neg_raw, n_per_class, seed + 1)
    return pos, neg


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONCEPTS = {
    "suicide": ConceptConfig(
        name="suicide",
        display_name="Suicidality (binary, 232K)",
        loader=_load_suicide,
        prompt_prefix=("Read the following text and determine if the person shows "
                       "signs of suicidal ideation."),
        question="Does this text show signs of suicidal ideation?",
        peak_layer_hint=16,
        notes="Primary concept; reproduces paper Sec 4.2-4.5.",
    ),
    "suicide_depression": ConceptConfig(
        name="suicide_depression",
        display_name="Suicide vs depression (matched distress, SWMH)",
        loader=_load_suicide_vs_depression,
        prompt_prefix=("Read the following text and determine if the person shows "
                       "signs of suicidal ideation."),
        question="Does this text show signs of suicidal ideation?",
        peak_layer_hint=16,
        notes="Matched-negative control for the register confound (reviewer #1): "
              "both classes are distressed mental-health Reddit.",
    ),
    "toxicity": ConceptConfig(
        name="toxicity",
        display_name="Toxicity (civil_comments)",
        loader=_load_toxicity,
        prompt_prefix=("Read the following comment and determine if it is toxic "
                       "(rude, disrespectful, or hateful)."),
        question="Is this comment toxic?",
        peak_layer_hint=16,
        notes="Second concept; outward harmful language vs. inward distress.",
    ),
    "sentiment": ConceptConfig(
        name="sentiment",
        display_name="Sentiment (IMDB, positive)",
        loader=_load_sentiment,
        prompt_prefix="Read the following movie review and determine if it is positive.",
        question="Is this movie review positive?",
        peak_layer_hint=16,
        notes="Affective-valence concept; long texts.",
    ),
    "sports": ConceptConfig(
        name="sports",
        display_name="Topic: sports (AG News)",
        loader=_load_topic_sports,
        prompt_prefix="Read the following news text and determine if it is about sports.",
        question="Is this text about sports?",
        peak_layer_hint=16,
        notes="LOW-affect neutral topic; critical control for the salience confound.",
    ),
    "business": ConceptConfig(
        name="business",
        display_name="Topic: business (AG News)",
        loader=_load_topic_business,
        prompt_prefix="Read the following news text and determine if it is about business.",
        question="Is this text about business?",
        peak_layer_hint=16,
        notes="Second neutral-topic control.",
    ),
}


def get_concept(name: str) -> ConceptConfig:
    if name not in CONCEPTS:
        raise ValueError(f"Unknown concept '{name}'. Registered: {list(CONCEPTS)}")
    return CONCEPTS[name]
