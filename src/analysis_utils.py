"""Shared analysis utilities for all phase scripts.

Provides:
- Model loading + wrapping in HookedHFModel
- Multi-dataset loading (DeepSuiMind, SWMH, Extended_SWMH)
- Activation patching (denoising style) that works with HookedHFModel
- Prompt building for suicide detection task
- Metric computation (logit difference)
"""

import sys
import gc
import json
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ModelLoader
from src.hf_adapter import HookedHFModel


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "deepsuimind": {
        "display_name": "DeepSuiMind",
        "description": "Implicit suicide detection (Death-Me/Life-Not-Me categories)",
    },
    "swmh": {
        "display_name": "SWMH_Reddit",
        "description": "Reddit mental health posts (SuicideWatch vs other subreddits)",
    },
    "extended_swmh": {
        "display_name": "Extended_SWMH",
        "description": "Extended SWMH with emotion annotations",
    },
    "umd": {
        "display_name": "UMD_Reddit_Suicidality",
        "description": "UMD expert-annotated 4-level risk (a/b/c/d) — CLPsych 2019 shared task",
    },
}


# UMD severity mapping: ordinal label -> numeric severity
UMD_SEVERITY = {"a": 0, "b": 1, "c": 2, "d": 3, "None": 0, None: 0}
UMD_LABEL_NAMES = {"a": "No Risk", "b": "Low Risk", "c": "Moderate Risk", "d": "Severe Risk", "None": "Control"}


def dataset_choices():
    """Return list of valid dataset names for argparse."""
    return list(DATASETS.keys())


def add_dataset_arg(parser):
    """Add --dataset argument to an argparse parser."""
    parser.add_argument(
        "--dataset", type=str, choices=dataset_choices(),
        default="deepsuimind",
        help=f"Dataset to use (default: deepsuimind). Options: {', '.join(dataset_choices())}",
    )


def get_output_dir(base_phase_dir, dataset_name):
    """Return output directory with dataset name: results/phase{N}/{dataset}/"""
    return base_phase_dir / dataset_name


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_by_category(split="train", dataset_name="deepsuimind"):
    """Load a dataset and split by category.

    Args:
        split: Data split (only used for deepsuimind/swmh).
        dataset_name: One of 'deepsuimind', 'swmh', 'extended_swmh'.

    Returns:
        dict with keys: "Death-Me", "Death-Not-Me", "Life-Not-Me"
        Each maps to a list of dataset examples with at least a text field.
    """
    loader = {
        "deepsuimind": _load_deepsuimind,
        "swmh": _load_swmh,
        "extended_swmh": _load_extended_swmh,
        "umd": _load_umd,
    }.get(dataset_name)

    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {dataset_choices()}")

    print(f"Loading dataset: {DATASETS[dataset_name]['display_name']}")
    return loader(split=split)


def _load_deepsuimind(split="train"):
    """Load DeepSuiMind dataset and split by Intention Category."""
    from datasets import load_dataset
    ds = load_dataset("babytreecc/Implicit-suicide-detection", split=split)
    print(f"Loaded {len(ds)} examples. Columns: {ds.column_names}")

    categories = {}
    for i, example in enumerate(ds):
        cat = example.get("Intention Category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(example)

    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    return categories


def _load_swmh(split="train"):
    """Load SWMH Reddit dataset. Maps SuicideWatch -> Death-Me, others -> Life-Not-Me."""
    data_path = PROJECT_ROOT / "data" / "datasets" / "01_SWMH_Reddit" / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"SWMH data not found: {data_path}")

    categories = {"Death-Me": [], "Life-Not-Me": [], "Death-Not-Me": []}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line.strip())
            label = example.get("label", "")
            # Wrap into unified format with text key
            wrapped = {"text": example.get("text", ""), "original_label": label}
            if label == "self.SuicideWatch":
                categories["Death-Me"].append(wrapped)
            elif label == "self.depression":
                # Depression posts may reference death without self-targeting
                categories["Death-Not-Me"].append(wrapped)
            else:
                categories["Life-Not-Me"].append(wrapped)

    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    return categories


def _load_extended_swmh(split="train"):
    """Load Extended SWMH CSV. Same label mapping as SWMH."""
    data_path = PROJECT_ROOT / "data" / "datasets" / "03_Extended_SWMH" / "repo" / "extendedswmh.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Extended SWMH data not found: {data_path}")

    categories = {"Death-Me": [], "Life-Not-Me": [], "Death-Not-Me": []}
    with open(data_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            wrapped = {"text": row.get("text", ""), "original_label": label}
            if label == "self.SuicideWatch":
                categories["Death-Me"].append(wrapped)
            elif label == "self.depression":
                categories["Death-Not-Me"].append(wrapped)
            else:
                categories["Life-Not-Me"].append(wrapped)

    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    return categories


def _load_umd(split="train", source="expert", max_chars_per_user=4000):
    """Load UMD Reddit Suicidality dataset (expert or crowd subset).

    Aggregates each user's SuicideWatch posts into one text blob (most recent first,
    truncated to max_chars_per_user). Maps ordinal labels -> 3-category bucket for
    compatibility with existing phases; severity is preserved in 'umd_label' field.

    Mapping:
        d (severe) + c (moderate) -> Death-Me
        b (low)                   -> Death-Not-Me
        a (no risk) + None        -> Life-Not-Me

    Args:
        split: "train" or "test" (for crowd); expert has no split.
        source: "expert" (245+245 users) or "crowd" (621+621 users).
        max_chars_per_user: cap on concatenated post text length per user.

    Returns dict with categories; each example has:
        - text: concatenated post text
        - umd_label: one of "a","b","c","d","None"
        - severity: int 0-3
        - user_id: int
        - n_posts: number of posts aggregated
    """
    base = PROJECT_ROOT / "data" / "datasets" / "04_UMD_Reddit_Suicidality" / "umd_reddit_suicidewatch_dataset_v2"
    if not base.exists():
        raise FileNotFoundError(f"UMD dataset not found at {base}")

    if source == "expert":
        labels_csv = base / "expert" / "expert.csv"
        posts_csv = base / "expert" / "expert_posts.csv"
    elif source == "crowd":
        labels_csv = base / "crowd" / "train" / "crowd_train.csv"
        posts_csv = base / "crowd" / "train" / "shared_task_posts.csv"
    else:
        raise ValueError(f"source must be 'expert' or 'crowd', got {source}")

    # Load user->label
    user_labels = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id", "").strip()
            lbl = row.get("label", "").strip() or "None"
            if uid:
                user_labels[uid] = lbl

    # Build set of target user_ids (annotated SW users)
    # Control users aren't in labels_csv; detect via negative IDs in posts
    target_users = set(user_labels.keys())
    print(f"  UMD: {len(target_users)} annotated users loaded from {labels_csv.name}")

    # Aggregate SuicideWatch posts per user (streaming — posts CSV can be large)
    user_posts = {}  # user_id -> list[(timestamp, title, body)]
    control_posts = {}  # for control users (negative IDs)
    n_rows = 0
    with open(posts_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            uid = row.get("user_id", "").strip()
            if not uid:
                continue
            subreddit = (row.get("subreddit", "") or "").strip().lower()
            # For annotated users: only keep SuicideWatch posts
            if uid in target_users:
                if subreddit != "suicidewatch":
                    continue
                bucket = user_posts
            else:
                # Control users: keep any post (limit total to keep memory sane)
                try:
                    uid_int = int(uid)
                    if uid_int >= 0:
                        continue  # only negative IDs are controls
                except ValueError:
                    continue
                bucket = control_posts
                if len(control_posts) >= 500 and uid not in control_posts:
                    continue  # cap control pool
            try:
                ts = int(row.get("timestamp", 0))
            except ValueError:
                ts = 0
            title = (row.get("post_title", "") or "").strip()
            body = (row.get("post_body", "") or "").strip()
            bucket.setdefault(uid, []).append((ts, title, body))
    print(f"  UMD: scanned {n_rows} post rows, {len(user_posts)} SW users with posts, "
          f"{len(control_posts)} control users")

    categories = {"Death-Me": [], "Death-Not-Me": [], "Life-Not-Me": []}

    def aggregate(posts, cap):
        posts_sorted = sorted(posts, key=lambda x: -x[0])
        pieces, used = [], 0
        for _, title, body in posts_sorted:
            chunk = (title + "\n" + body).strip()
            if not chunk:
                continue
            if used + len(chunk) > cap:
                chunk = chunk[: max(0, cap - used)]
            pieces.append(chunk)
            used += len(chunk)
            if used >= cap:
                break
        return "\n\n".join(pieces)

    # Annotated users
    for uid, lbl in user_labels.items():
        if uid not in user_posts:
            continue
        text = aggregate(user_posts[uid], max_chars_per_user)
        if len(text) < 50:
            continue
        record = {
            "text": text,
            "umd_label": lbl,
            "severity": UMD_SEVERITY.get(lbl, 0),
            "user_id": uid,
            "n_posts": len(user_posts[uid]),
        }
        if lbl in ("d", "c"):
            categories["Death-Me"].append(record)
        elif lbl == "b":
            categories["Death-Not-Me"].append(record)
        else:  # "a" or None
            categories["Life-Not-Me"].append(record)

    # Control users -> Life-Not-Me (label "None")
    for uid, posts in control_posts.items():
        text = aggregate(posts, max_chars_per_user)
        if len(text) < 50:
            continue
        record = {
            "text": text,
            "umd_label": "None",
            "severity": 0,
            "user_id": uid,
            "n_posts": len(posts),
        }
        categories["Life-Not-Me"].append(record)

    for cat, examples in categories.items():
        print(f"  {cat}: {len(examples)} examples")

    return categories


def load_umd_with_severity(source="expert", max_chars_per_user=4000):
    """UMD-specific accessor: returns a FLAT list of records with severity 0-3.

    Use this for phase 11 (severity axis) and any analysis that needs graded labels.
    Each record includes: text, umd_label (a/b/c/d/None), severity (0-3),
    user_id, n_posts.
    """
    cats = _load_umd(source=source, max_chars_per_user=max_chars_per_user)
    flat = []
    for examples in cats.values():
        flat.extend(examples)
    # Sort by severity for convenience
    flat.sort(key=lambda r: r["severity"])
    print(f"  UMD flat: {len(flat)} records across severity levels "
          f"{[sum(1 for r in flat if r['severity'] == s) for s in range(4)]}")
    return flat


def get_text(example):
    """Extract text from a dataset example."""
    for key in ["text", "content", "message"]:
        if key in example and example[key]:
            return example[key]
    # Fallback: first string field
    for key, val in example.items():
        if isinstance(val, str) and len(val) > 20:
            return val
    return str(example)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "standard": (
        "Read the following text and determine if the person shows signs of "
        "suicidal ideation.\n\n"
        "Text: {text}\n\n"
        "Does this text show signs of suicidal ideation? Answer with Yes or No.\n"
        "Answer:"
    ),
    "empathetic": (
        "You are a compassionate mental health professional. Read the following "
        "text carefully, paying attention to emotional undertones and implicit "
        "cues of distress.\n\n"
        "Text: {text}\n\n"
        "Does this text show signs of suicidal ideation? Answer with Yes or No.\n"
        "Answer:"
    ),
    "clinical": (
        "You are a clinical psychologist conducting a risk assessment. Analyze "
        "the following text for any indicators of suicidal ideation, including "
        "implicit associations with death.\n\n"
        "Text: {text}\n\n"
        "Does this text show signs of suicidal ideation? Answer with Yes or No.\n"
        "Answer:"
    ),
}


def build_prompt(text, style="standard"):
    """Build a suicide detection prompt."""
    template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["standard"])
    return template.format(text=text)


def build_prompt_tokens(text, tokenizer, style="standard", max_total_tokens=256, device=None):
    """Build a prompt that fits in max_total_tokens by truncating the BODY, not the suffix.

    Naive `tokenizer(prompt, max_length=256, truncation=True)` slices the END of the
    prompt — which removes the question and "Answer:" suffix. The model then sees only
    a half-finished narrative and predicts the next narrative token, not Yes/No.

    This helper budgets template prefix + suffix first, then fits as much of the user
    text as remains. Returns input_ids of shape [1, L] where L <= max_total_tokens
    (no padding — caller pads if needed).
    """
    template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["standard"])
    prefix, suffix = template.split("{text}")

    pre_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    suf_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    body_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    body_budget = max_total_tokens - len(pre_ids) - len(suf_ids)
    if body_budget <= 0:
        raise ValueError(
            f"max_total_tokens={max_total_tokens} too small for template "
            f"(prefix={len(pre_ids)}, suffix={len(suf_ids)})"
        )
    body_ids = body_ids[:body_budget]

    full = pre_ids + body_ids + suf_ids
    out = torch.tensor([full], dtype=torch.long)
    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_model_wrapped(model_name="llama", device="cuda", load_in_4bit=True):
    """Load a model and wrap in HookedHFModel.

    Honors env var MECH_INTERP_PRECISION:
        "4bit" (default on low-VRAM GPUs) — same as load_in_4bit=True
        "8bit"   — 8-bit quantization
        "fp16"   — no quantization, fp16 (needs ~16 GB VRAM)
        "bf16"   — no quantization, bf16 (A100 / H100 recommended)
        "auto"   — let the ModelLoader decide based on VRAM

    Env var wins over load_in_4bit argument.
    """
    import os
    prec = os.environ.get("MECH_INTERP_PRECISION", "").lower()
    loader = ModelLoader(device=device)

    load_fn = {
        "llama": loader.load_llama,
        "qwen": loader.load_qwen,
        "mistral": loader.load_mistral,
    }.get(model_name)

    if load_fn is None:
        raise ValueError(f"Unknown model: {model_name}. Use 'llama', 'qwen', or 'mistral'.")

    if prec == "bf16":
        hf_model, tokenizer = load_fn(auto_quantize=False, load_in_4bit=False,
                                      load_in_8bit=False, torch_dtype=torch.bfloat16)
    elif prec == "fp16":
        hf_model, tokenizer = load_fn(auto_quantize=False, load_in_4bit=False,
                                      load_in_8bit=False)
    elif prec == "8bit":
        hf_model, tokenizer = load_fn(auto_quantize=False, load_in_4bit=False,
                                      load_in_8bit=True)
    elif prec == "auto":
        hf_model, tokenizer = load_fn(auto_quantize=True)
    else:
        hf_model, tokenizer = load_fn(load_in_4bit=load_in_4bit, auto_quantize=False)
    model = HookedHFModel(hf_model, tokenizer, device=torch.device(device))

    print(f"Model wrapped: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}")
    return model, tokenizer


def unload_model(model):
    """Free GPU memory from a model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_yes_no_ids(tokenizer):
    """Get token IDs for 'Yes' and 'No'."""
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    return yes_tokens[0], no_tokens[0]


def tokenize_pair(tokenizer, clean_prompt, corrupt_prompt, max_length=256, device="cuda"):
    """Tokenize clean and corrupt prompts to same length.

    Returns:
        (clean_tokens, corrupt_tokens) - both [1, max_length] tensors
    """
    clean_enc = tokenizer(
        clean_prompt, return_tensors="pt", max_length=max_length,
        truncation=True, padding="max_length",
    )
    corrupt_enc = tokenizer(
        corrupt_prompt, return_tensors="pt", max_length=max_length,
        truncation=True, padding="max_length",
    )
    return (
        clean_enc["input_ids"].to(device),
        corrupt_enc["input_ids"].to(device),
    )


# ---------------------------------------------------------------------------
# Activation patching (denoising style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_activation_patching(
    model,
    clean_tokens,
    corrupt_tokens,
    yes_id,
    no_id,
):
    """Run denoising activation patching across all layers.

    Denoising: run on CORRUPT input, patch in CLEAN activation.
    IE = (patched - corrupt) / (clean - corrupt)
    IE = 1 means component fully restores clean behavior.

    Args:
        model: HookedHFModel
        clean_tokens: [1, seq_len] clean input
        corrupt_tokens: [1, seq_len] corrupt input
        yes_id: token ID for "Yes"
        no_id: token ID for "No"

    Returns:
        attn_ie: [n_layers] IE for attention at each layer
        mlp_ie: [n_layers] IE for MLP at each layer
        ld_clean: scalar logit diff on clean input
        ld_corrupt: scalar logit diff on corrupt input
    """
    n_layers = model.cfg.n_layers

    # Clean baseline
    clean_logits = model(clean_tokens)
    ld_clean = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).float()

    # Corrupt baseline + cache clean activations for patching
    corrupt_logits = model(corrupt_tokens)
    ld_corrupt = (corrupt_logits[0, -1, yes_id] - corrupt_logits[0, -1, no_id]).float()

    # Get clean cache (for denoising: we patch clean into corrupt)
    _, clean_cache = model.run_with_cache(clean_tokens)

    denom = (ld_clean - ld_corrupt).abs()
    MIN_LD_GAP = 0.5  # Skip pairs with tiny LD gap to avoid IE explosion

    if denom < MIN_LD_GAP:
        return torch.zeros(n_layers), torch.zeros(n_layers), ld_clean, ld_corrupt

    attn_ie = torch.zeros(n_layers)
    mlp_ie = torch.zeros(n_layers)

    for layer in range(n_layers):
        for component, ie_tensor in [("hook_attn_out", attn_ie), ("hook_mlp_out", mlp_ie)]:
            hook_name = f"blocks.{layer}.{component}"
            # TL's ActivationCache supports __contains__ / __getitem__ but not .get().
            # Using in+index keeps compatibility with the old dict-based HF cache too.
            if hook_name not in clean_cache:
                continue
            clean_act = clean_cache[hook_name]

            def make_hook(cached_act):
                def hook_fn(output, hook):
                    return cached_act.to(output.device if isinstance(output, torch.Tensor) else output[0].device)
                return hook_fn

            model.add_hook(hook_name, make_hook(clean_act))
            patched_logits = model(corrupt_tokens)
            model.remove_hooks()

            ld_patched = (patched_logits[0, -1, yes_id] - patched_logits[0, -1, no_id]).float()
            ie_tensor[layer] = ((ld_patched - ld_corrupt) / denom).item()
            # Clip IE to [-1, 1] to prevent numerical explosions
            ie_tensor[layer] = max(-1.0, min(1.0, ie_tensor[layer]))

    return attn_ie, mlp_ie, ld_clean, ld_corrupt


@torch.no_grad()
def run_head_knockout(
    model,
    corrupt_tokens,
    clean_tokens,
    yes_id,
    no_id,
    layers=None,
):
    """Zero-ablate each attention head individually and measure effect.

    Args:
        model: HookedHFModel
        corrupt_tokens: [1, seq_len] corrupt input (baseline)
        clean_tokens: [1, seq_len] clean input (for reference)
        yes_id, no_id: token IDs
        layers: list of layers to test (default: all)

    Returns:
        head_importance: [n_layers, n_heads] - drop in logit diff when head zeroed
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    if layers is None:
        layers = list(range(n_layers))

    # Baseline: run on clean input
    clean_logits = model(clean_tokens)
    ld_baseline = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).float().item()

    head_importance = np.zeros((n_layers, n_heads))

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_attn_out"

        for head in range(n_heads):
            def make_zero_head_hook(h, d_h):
                def hook_fn(output, hook):
                    modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
                    # Zero out this head's contribution
                    # Attention output is [batch, seq, d_model], each head contributes d_head dims
                    start = h * d_h
                    end = start + d_h
                    modified[:, :, start:end] = 0.0
                    return modified
                return hook_fn

            model.add_hook(hook_name, make_zero_head_hook(head, d_head))
            ablated_logits = model(clean_tokens)
            model.remove_hooks()

            ld_ablated = (ablated_logits[0, -1, yes_id] - ablated_logits[0, -1, no_id]).float().item()
            head_importance[layer, head] = abs(ld_baseline - ld_ablated)

    return head_importance


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def plot_ie_heatmap(attn_ie, mlp_ie, title="Indirect Effect by Layer",
                    save_path=None):
    """Plot attention and MLP IE side by side."""
    n_layers = len(attn_ie)

    fig, axes = plt.subplots(1, 2, figsize=(10, max(6, n_layers * 0.25)))

    # Attention IE
    ax = axes[0]
    data = attn_ie.reshape(-1, 1) if isinstance(attn_ie, np.ndarray) else attn_ie.numpy().reshape(-1, 1)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_ylabel("Layer")
    ax.set_title("Attention IE")
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.05)

    # MLP IE
    ax = axes[1]
    data = mlp_ie.reshape(-1, 1) if isinstance(mlp_ie, np.ndarray) else mlp_ie.numpy().reshape(-1, 1)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_title("MLP IE")
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.05)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_ie_curves(attn_ie, mlp_ie, title="IE by Layer", save_path=None):
    """Plot attention and MLP IE as line curves."""
    n_layers = len(attn_ie)
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(10, 5))
    a = attn_ie.numpy() if isinstance(attn_ie, torch.Tensor) else attn_ie
    m = mlp_ie.numpy() if isinstance(mlp_ie, torch.Tensor) else mlp_ie

    ax.plot(layers, a, "o-", label="Attention", color="#e74c3c", markersize=4)
    ax.plot(layers, m, "s-", label="MLP", color="#3498db", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Indirect Effect")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n_layers - 0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
