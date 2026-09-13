#!/usr/bin/env python
"""Which attention heads COMPUTE intent (not topic)?  -- the mechanism.

sc2_intent_minpairs.py asks *whether* the layer-16 feature tracks intent. This
script asks *how* the model gets there: which attention heads are responsible for
distinguishing genuine self-directed intent from lexically-matched mention.

Method (per-head knockout against an intent contrast). For each query head (L, h)
we zero its slice of the o_proj input -- the d_head-wide block that head writes to
the residual stream (the same decomposition used by sc2_head_knockout / DLA) --
and re-measure the Yes-No logit difference on two groups of the minimal-pairs set:
  - genuine_intent items
  - control items (default: topical + counterfactual, i.e. same words, no intent)

    drop_genuine(L,h)  = mean(LD_baseline - LD_ablated) over genuine items
    drop_control(L,h)  = mean(LD_baseline - LD_ablated) over control items
    intent_effect(L,h) = drop_genuine - drop_control

A head with large positive intent_effect carries the "Yes" signal specifically for
genuine intent while leaving topical mention untouched -- a candidate node in the
intent circuit. A head that drops both equally is a generic death-topic head. This
separates the intent mechanism from the topic mechanism the binary paper found.

This is the discovery half of Track A; the writer heads it surfaces are the inputs
to a focused path-patching follow-up (see research_docs/INTENT_MECHANISM_PLAN.md).

Output: results/sc2_binary/intent_circuit.json + intent_circuit.png
Run:    python analysis/sanity/sc2_intent_circuit.py --layers 4-26
"""
import sys, json, argparse
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MAXLEN = 256
device = "cuda"
MINPAIRS = PROJECT_ROOT / "data" / "intent_minpairs" / "intent_minpairs_v1.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Per-head intent-circuit knockout")
    p.add_argument("--layers", type=str, default="4-26",
                   help="layer range to scan, e.g. '4-26' or 'all'")
    p.add_argument("--control_conditions", type=str, default="topical,counterfactual",
                   help="comma-separated control conditions to contrast against")
    p.add_argument("--n_items", type=int, default=0,
                   help="cap items per group (0 = use all)")
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in MINPAIRS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["text"])

    ctrl_conds = [c.strip() for c in args.control_conditions.split(",") if c.strip()]
    genuine = by_cond["genuine_intent"]
    control = [t for c in ctrl_conds for t in by_cond.get(c, [])]
    if args.n_items:
        genuine = genuine[: args.n_items]
        control = control[: args.n_items]
    print(f"genuine items: {len(genuine)}   control items ({'+'.join(ctrl_conds)}): {len(control)}")

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads
    print(f"model: {n_layers} layers, {n_heads} heads, d_head={d_head}")

    if args.layers.lower() == "all":
        layer_list = list(range(n_layers))
    else:
        lo, hi = args.layers.split("-"); layer_list = list(range(int(lo), int(hi) + 1))

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # pre-tokenize once
    g_ids = [chat_ids(t) for t in genuine]
    c_ids = [chat_ids(t) for t in control]

    @torch.no_grad()
    def ld(ids):
        out = model(ids)
        return (out.logits[0, -1, yes_id] - out.logits[0, -1, no_id]).item()

    # ---- baselines ----
    print("baseline LDs...")
    g_base = np.array([ld(i) for i in g_ids])
    c_base = np.array([ld(i) for i in c_ids])
    print(f"  mean LD  genuine {g_base.mean():+.2f}   control {c_base.mean():+.2f}")

    # ---- per-head knockout via o_proj input masking ----
    # forward_pre_hook on o_proj zeros columns [h*d_head:(h+1)*d_head] of its input,
    # which is exactly head h's contribution to the residual stream. Every layer's
    # hook is registered once; a hook only fires when state names its own layer, so
    # exactly one head at one layer is ablated per forward pass.
    state = {"layer": None, "head": None}

    def make_prehook(layer_idx):
        def prehook(module, inputs):
            if state["layer"] != layer_idx or state["head"] is None:
                return None
            x = inputs[0].clone()
            h = state["head"]
            x[..., h * d_head:(h + 1) * d_head] = 0.0
            return (x,) + tuple(inputs[1:])
        return prehook

    handles = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_prehook(L))
               for L in layer_list]

    drop_g = np.full((n_layers, n_heads), np.nan)
    drop_c = np.full((n_layers, n_heads), np.nan)

    @torch.no_grad()
    def mean_ld(ids_list):
        return np.array([ld(i) for i in ids_list]).mean()

    for L in layer_list:
        state["layer"] = L
        for h in range(n_heads):
            state["head"] = h
            drop_g[L, h] = g_base.mean() - mean_ld(g_ids)
            drop_c[L, h] = c_base.mean() - mean_ld(c_ids)
        print(f"  layer {L}: max intent_effect head = "
              f"{int(np.nanargmax(drop_g[L] - drop_c[L]))} "
              f"({np.nanmax(drop_g[L] - drop_c[L]):+.3f})")
    state["layer"] = state["head"] = None

    for hd in handles:
        hd.remove()

    intent_effect = drop_g - drop_c

    # rank heads
    flat = [((L, h), float(intent_effect[L, h])) for L in layer_list for h in range(n_heads)]
    flat.sort(key=lambda x: -x[1])
    top = flat[:15]
    print("\n" + "=" * 64)
    print("TOP INTENT-CIRCUIT HEADS (drop_genuine - drop_control)")
    print("=" * 64)
    for (L, h), v in top:
        print(f"  L{L:2d}H{h:2d}: intent_effect={v:+.3f}  "
              f"(drop_g={drop_g[L,h]:+.3f}, drop_c={drop_c[L,h]:+.3f})")
    print("=" * 64)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    json.dump({
        "dataset": "intent_minpairs_v1", "method": "per_head_knockout_intent_contrast",
        "control_conditions": ctrl_conds, "layers_scanned": layer_list,
        "n_genuine": len(genuine), "n_control": len(control),
        "baseline_ld": {"genuine": float(g_base.mean()), "control": float(c_base.mean())},
        "top_heads": [{"layer": L, "head": h, "intent_effect": v,
                       "drop_genuine": float(drop_g[L, h]), "drop_control": float(drop_c[L, h])}
                      for (L, h), v in top],
        "intent_effect_grid": np.nan_to_num(intent_effect).tolist(),
    }, open(out_dir / "intent_circuit.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(10, max(4, len(layer_list) * 0.22)))
    sub = intent_effect[layer_list, :]
    vmax = np.nanmax(np.abs(sub)) or 1.0
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(layer_list))); ax.set_yticklabels(layer_list)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title("Intent-circuit map: per-head (drop_genuine - drop_control)\n"
                 "red = ablation hurts genuine intent more than topical mention")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(str(out_dir / "intent_circuit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/sc2_binary/intent_circuit.json + intent_circuit.png")


if __name__ == "__main__":
    main()
