"""Concept registry for the audit: the paper's concepts plus published concept directions.

Reuses the loaders in analysis/sanity/gcl/concept_config.py and adds the published
claims Paper 1 audits. Each concept also carries what the audit's extra readouts need:
a negated question (polarity test) and a short description for the multiple-choice
readout.

Published-claim concepts (data as close to the original paper as a yes/no task allows):
  refusal     Arditi et al. 2024: harmful (AdvBench) vs harmless (Alpaca) instructions
  truth       Marks & Tegmark 2023: true vs false statements (geometry-of-truth cities)
  sycophancy  Rimsky et al. 2024 (CAA): behavior-matching vs non-matching answers
  anger       emotion direction in the spirit of Zou et al. 2023 (RepE): anger vs other
              emotions (dair-ai/emotion)
"""
import io
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(os.environ.get("MI_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "sanity" / "gcl"))
CACHE = PROJECT_ROOT / "data" / "datasets" / "audit_cache"


@dataclass
class AuditConcept:
    name: str
    display_name: str
    loader: Callable[[int, int], Tuple[List[str], List[str]]]
    prefix: str
    question: str
    neg_question: str          # same question, polarity flipped
    mc_label: str              # "The text {mc_label}." in the multiple-choice readout
    source: str = ""
    truncate: str = "right"    # "left" keeps the END of long texts (CAA: the answer is last)
    body_tokens: int = 176     # paper's budget (256 - 80 reserved); CAA items need up to 240
    pos_answer: str = " Yes"
    neg_answer: str = " No"


def _clean(texts, n, seed, min_chars):
    rng = np.random.default_rng(seed)
    out = [t.strip() for t in texts if t and len(t.strip()) >= min_chars]
    out = list(dict.fromkeys(out))          # de-duplicate, keep order
    rng.shuffle(out)
    return out[:n]


def _fetch(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


# ---------------------------------------------------------------- new loaders
def _load_refusal(n, seed=0):
    """Arditi et al.'s own splits (harmful: AdvBench/MaliciousInstruct/TDC/HarmBench; harmless: Alpaca)."""
    base = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/"
    def read(kind):
        out = []
        for split in ("train", "val", "test"):
            path = _fetch(base + f"{kind}_{split}.json", f"arditi_{kind}_{split}.json")
            out += [r["instruction"] for r in json.loads(path.read_text(encoding="utf-8"))]
        return out
    return _clean(read("harmful"), n, seed, 10), _clean(read("harmless"), n, seed + 1, 10)


def _load_truth(n, seed=0):
    import csv
    path = _fetch("https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv",
                  "cities.csv")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    pos = [r["statement"] for r in rows if r["label"] == "1"]
    neg = [r["statement"] for r in rows if r["label"] == "0"]
    return _clean(pos, n, seed, 10), _clean(neg, n, seed + 1, 10)


def _load_sycophancy(n, seed=0):
    path = _fetch("https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/"
                  "sycophancy/generate_dataset.json", "caa_sycophancy.json")
    items = json.loads(path.read_text(encoding="utf-8"))
    pos = [f"{it['question']}\nAnswer: {it['answer_matching_behavior']}" for it in items]
    neg = [f"{it['question']}\nAnswer: {it['answer_not_matching_behavior']}" for it in items]
    return _clean(pos, n, seed, 10), _clean(neg, n, seed + 1, 10)


def _load_anger(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", split="train")   # 0 sad 1 joy 2 love 3 anger 4 fear 5 surprise
    pos = [t for t, l in zip(ds["text"], ds["label"]) if l == 3]
    neg = [t for t, l in zip(ds["text"], ds["label"]) if l != 3]
    return _clean(pos, n, seed, 30), _clean(neg, n, seed + 1, 30)


# ---------------------------------------------------------------- registry
def _from_gcl(name, neg_question, mc_label, source):
    from concept_config import get_concept
    c = get_concept(name)
    return AuditConcept(name=c.name, display_name=c.display_name, loader=c.loader,
                        prefix=c.prompt_prefix, question=c.question, neg_question=neg_question,
                        mc_label=mc_label, source=source)


def _registry():
    reg = {
        "suicide": lambda: _from_gcl(
            "suicide", "Is this text free of any signs of suicidal ideation?",
            "expresses suicidal thoughts", "this project (Suicide-Watch-232K)"),
        "suicide_depression": lambda: _from_gcl(
            "suicide_depression", "Is this text free of any signs of suicidal ideation?",
            "expresses suicidal thoughts", "this project (SWMH, matched register)"),
        "toxicity": lambda: _from_gcl(
            "toxicity", "Is this comment polite and free of toxicity?",
            "is rude or hateful", "this project (civil_comments)"),
        "sentiment": lambda: _from_gcl(
            "sentiment", "Is this movie review negative?",
            "is a positive review", "Tigges et al. 2023 (IMDB)"),
        "sports": lambda: _from_gcl(
            "sports", "Is this text about something other than sports?",
            "is about sports", "neutral-topic control (AG News)"),
        "business": lambda: _from_gcl(
            "business", "Is this text about something other than business?",
            "is about business", "neutral-topic control (AG News)"),
        "refusal": lambda: AuditConcept(
            "refusal", "Refusal: harmful vs harmless requests", _load_refusal,
            "Read the following request and determine if it asks for something harmful.",
            "Does this request ask for something harmful?",
            "Is this request harmless?",
            "asks for something harmful", "Arditi et al. 2024 (AdvBench vs Alpaca)"),
        "truth": lambda: AuditConcept(
            "truth", "Truth: true vs false statements", _load_truth,
            "Read the following statement and determine if it is true.",
            "Is this statement true?",
            "Is this statement false?",
            "is a true statement", "Marks & Tegmark 2023 (cities)"),
        "sycophancy": lambda: AuditConcept(
            "sycophancy", "Sycophancy (CAA A/B answers)", _load_sycophancy,
            "Read the following question and answer and determine if the answer is sycophantic.",
            "Does the answer tell the person what they want to hear rather than what is accurate?",
            "Does the answer give an accurate view even if the person may not want to hear it?",
            "tells the person what they want to hear", "Rimsky et al. 2024 (CAA)",
            truncate="left", body_tokens=256),
        "anger": lambda: AuditConcept(
            "anger", "Emotion: anger vs other emotions", _load_anger,
            "Read the following text and determine if the writer is angry.",
            "Does this text express anger?",
            "Is this text free of anger?",
            "expresses anger", "emotion direction, cf. Zou et al. 2023 (dair-ai/emotion)"),
    }
    return reg


def get_concept(name) -> AuditConcept:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown concept '{name}'; registered: {sorted(reg)}")
    return reg[name]()


def list_concepts():
    return sorted(_registry())


# Off-target readouts. A concept whose own topic matches one of these never uses it.
OFFTARGET_QUESTIONS = [
    ("food", "Read the following text and determine if it is about food.",
     "Is this text primarily about food or cooking?"),
    ("weather", "Read the following text and determine if it is about the weather.",
     "Is this text about the weather?"),
    ("programming", "Read the following text and determine if it is about computer programming.",
     "Is this text about computer programming?"),
]
OFFTARGET_MC = ["is about food or cooking", "is about the weather", "is about computer programming"]
