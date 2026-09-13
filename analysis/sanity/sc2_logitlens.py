#!/usr/bin/env python
"""Logit lens on the VALIDATED binary task: where the Yes/No decision crystallizes.

Companion to sc2_patching.py / sc2_head_knockout.py / sc2_sae.py. The original
Phase 10 logit-lens ran on DeepSuiMind (chance task); this re-runs it on the binary
suicide-vs-non-suicide task (07_Suicide_Watch_232K) with the exact model / prompt /
readout of sc2_faithfulness.py.

Method. For each post we take the final-position residual stream after every layer
(hidden_states from output_hidden_states), apply the model's final norm and unembed
(lm_head) -- the standard logit lens -- and read the Yes-No logit difference. Doing
this per layer shows the depth at which the model's decision forms, separately for
suicidal and non-suicidal posts. The gap between the two curves is the per-layer
"decision signal"; we expect it to open up in the mid layers, consistent with the
layer-16 probe readout and the mid-layer head-knockout / patching peaks.

Output: results/sc2_binary/logitlens.json  +  logitlens.png
Run:    python analysis/sanity/sc2_logitlens.py --n 50
Cost:   n_total forward passes (one per post). n=50 => 100 fwd (~2-4 min, 12 GB).
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Logit lens on binary 232K")
    p.add_argument("--n", type=int, default=50, help="examples per class (default 50)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui, non = sui[: args.n], non[: args.n]

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    final_norm = model.model.norm     # Llama final RMSNorm
    unembed = model.lm_head           # not quantized -> directly callable
    w_dtype = unembed.weight.dtype

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def per_layer_ld(ids):
        """Yes-No logit diff from the lens at every layer (incl. embedding=index 0)."""
        hs = model(ids, output_hidden_states=True).hidden_states  # len n_layers+1
        out = []
        for h in hs:
            v = h[0, -1, :].to(w_dtype)
            v = final_norm(v)
            lg = unembed(v)
            out.append((lg[yes_id] - lg[no_id]).item())
        return out

    sui_curves = np.array([per_layer_ld(chat_ids(t)) for t in sui])  # [n, n_layers+1]
    non_curves = np.array([per_layer_ld(chat_ids(t)) for t in non])
    depth = sui_curves.shape[1]

    sui_mean, sui_sd = sui_curves.mean(0), sui_curves.std(0)
    non_mean, non_sd = non_curves.mean(0), non_curves.std(0)
    sep = sui_mean - non_mean
    sep_top = int(np.argmax(sep))

    print("\n" + "=" * 60)
    print(f"LOGIT LENS (binary 232K, n={args.n}/class)")
    print(f"  final-layer LD: suicide {sui_mean[-1]:+.2f} vs non {non_mean[-1]:+.2f}")
    print(f"  max suicide-vs-non separation: layer {sep_top} ({sep[sep_top]:+.2f})")
    print("=" * 60)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    with open(out_dir / "logitlens.json", "w") as f:
        json.dump({
            "dataset": "07_Suicide_Watch_232K", "n_per_class": args.n,
            "depth": depth, "note": "index 0 = embeddings; index i = residual after layer i-1",
            "suicide_mean_ld": sui_mean.tolist(), "suicide_sd_ld": sui_sd.tolist(),
            "nonsuicide_mean_ld": non_mean.tolist(), "nonsuicide_sd_ld": non_sd.tolist(),
            "separation": sep.tolist(),
            "max_separation_layer": sep_top, "max_separation": float(sep[sep_top]),
        }, f, indent=2)

    # ---- figure ----
    xs = list(range(depth))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(xs, sui_mean, "-o", ms=3, color="#e74c3c", label="suicidal posts")
    ax1.fill_between(xs, sui_mean - sui_sd, sui_mean + sui_sd, color="#e74c3c", alpha=0.15)
    ax1.plot(xs, non_mean, "-o", ms=3, color="#3498db", label="non-suicidal posts")
    ax1.fill_between(xs, non_mean - non_sd, non_mean + non_sd, color="#3498db", alpha=0.15)
    ax1.axhline(0, ls="--", c="grey", lw=1)
    ax1.set_xlabel("layer (0 = embeddings)"); ax1.set_ylabel("lens Yes-No logit difference")
    ax1.set_title("Logit lens: decision by depth"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(xs, sep, "-o", ms=3, color="C2")
    ax2.axhline(0, ls="--", c="grey", lw=1)
    ax2.axvline(sep_top, ls=":", c="grey", lw=1)
    ax2.set_xlabel("layer"); ax2.set_ylabel("suicide - non-suicide lens LD")
    ax2.set_title(f"Decision signal opens mid-network (peak L{sep_top})")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Logit lens on binary suicide detection, 232K", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / "logitlens.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved results/sc2_binary/logitlens.json + logitlens.png")


if __name__ == "__main__":
    main()
