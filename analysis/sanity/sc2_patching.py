#!/usr/bin/env python
"""Layer-wise denoising activation patching on the VALIDATED binary task.

This is the circuit-tracing companion to sc2_head_knockout.py / sc2_sae.py. The
original Phase 1/2 indirect-effect plots ("Denoising IE: Death-Me vs Life-Not-Me")
were computed on DeepSuiMind, the task the model performs at chance, so they
attribute a routing of behavior that does not exist. This re-runs the same idea on
the binary suicide-vs-non-suicide task (07_Suicide_Watch_232K), where the behavior
is real (AUC ~0.95), using the exact model / prompt / readout of
sc2_faithfulness.py so the result is comparable to the direction-level causal
claims.

Method (denoising). For each pair (suicidal post = CLEAN, non-suicidal post =
CORRUPT) we cache, on the clean run, the per-layer contribution that each component
adds to the residual stream:
  - attention contribution  = output of layers[L].self_attn.o_proj
  - MLP contribution        = output of layers[L].mlp.down_proj
Both are plain Linear outputs of shape [1, seq, d_model], so hooking them is robust
across transformers versions (no attention-return-signature dependence). We then
run the corrupt input and, one component at a time, REPLACE that component's output
with the cached clean activation and re-read the Yes-No logit difference. The
indirect effect is

    IE = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt),   clipped to [-1, 1]

so IE = 1 means restoring that one component fully recovers the clean (suicidal)
behavior. Clean and corrupt are left-padded to a common length so the final
"Answer:" position aligns at index -1 for both; pad positions are masked.

A peak in attention IE or MLP IE at particular layers localizes where the
suicidality signal is written into the residual stream. We expect a mid-layer
concentration consistent with the L16 probe readout and the head-knockout map.

Output: results/sc2_binary/patching.json  +  patching_ie.png
Run:    python analysis/sanity/sc2_patching.py --n 20
        python analysis/sanity/sc2_patching.py --n 30 --seed 1
Cost:   per pair ~ (2 + 2*n_layers) forward passes. n=20 => ~1.3k fwd (~5-10 min, 12 GB).
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
MIN_LD_GAP = 1.0  # skip pairs whose clean-corrupt LD gap is too small to normalize
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise activation patching on binary 232K")
    p.add_argument("--n", type=int, default=20, help="clean/corrupt pairs (default 20)")
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
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    layers = model.model.layers
    n_layers = len(layers)
    print(f"{n_layers} layers; {args.n} clean/corrupt pairs")

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False)["input_ids"]

    def left_pad(a, b):
        """Left-pad two 1D id lists to a common length; return ids+mask tensors."""
        L = max(len(a), len(b))
        def pad(x):
            k = L - len(x)
            ids = [pad_id] * k + x
            mask = [0] * k + [1] * len(x)
            return ids, mask
        ai, am = pad(a); bi, bm = pad(b)
        T = lambda v: torch.tensor([v], device=device)
        return (T(ai), T(am)), (T(bi), T(bm))

    @torch.no_grad()
    def ld(ids, mask):
        lg = model(input_ids=ids, attention_mask=mask).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    # component module accessors (plain Linear outputs added to the residual stream)
    def comp_module(L, comp):
        return layers[L].self_attn.o_proj if comp == "attn" else layers[L].mlp.down_proj

    attn_ie = np.zeros(n_layers)
    mlp_ie = np.zeros(n_layers)
    n_used = 0

    for pi, (s_text, n_text) in enumerate(zip(sui, non)):
        (cids, cmask), (nids, nmask) = left_pad(chat_ids(s_text), chat_ids(n_text))

        # ---- cache clean component outputs ----
        cache = {}  # (L, comp) -> tensor [1, seq, d]
        handles = []
        for L in range(n_layers):
            for comp in ("attn", "mlp"):
                def mk(L=L, comp=comp):
                    def hook(module, inp, out):
                        cache[(L, comp)] = (out[0] if isinstance(out, tuple) else out).detach().clone()
                    return hook
                handles.append(comp_module(L, comp).register_forward_hook(mk()))
        with torch.no_grad():
            ld_clean = ld(cids, cmask)
        for h in handles:
            h.remove()

        with torch.no_grad():
            ld_corrupt = ld(nids, nmask)

        denom = ld_clean - ld_corrupt
        if abs(denom) < MIN_LD_GAP:
            print(f"  pair {pi}: |clean-corrupt|={abs(denom):.2f} < {MIN_LD_GAP}, skipped")
            continue

        # ---- patch each component on the corrupt run ----
        for L in range(n_layers):
            for comp, store in (("attn", attn_ie), ("mlp", mlp_ie)):
                clean_act = cache[(L, comp)]
                def mk(clean_act=clean_act):
                    def hook(module, inp, out):
                        if isinstance(out, tuple):
                            return (clean_act.to(out[0].dtype),) + out[1:]
                        return clean_act.to(out.dtype)
                    return hook
                hd = comp_module(L, comp).register_forward_hook(mk())
                try:
                    ld_patched = ld(nids, nmask)
                finally:
                    hd.remove()
                ie = (ld_patched - ld_corrupt) / denom
                store[L] += max(-1.0, min(1.0, ie))
        n_used += 1
        print(f"  pair {pi}: clean LD={ld_clean:+.2f} corrupt LD={ld_corrupt:+.2f} [{n_used} used]")

    if n_used == 0:
        raise RuntimeError("No usable pairs (all had tiny clean-corrupt gap). Try more --n.")
    attn_ie /= n_used
    mlp_ie /= n_used

    a_top = int(np.argmax(attn_ie)); m_top = int(np.argmax(mlp_ie))
    print("\n" + "=" * 60)
    print(f"PATCHING IE (binary 232K, {n_used} pairs)")
    print(f"  peak attention IE: L{a_top} = {attn_ie[a_top]:+.3f}")
    print(f"  peak MLP IE:       L{m_top} = {mlp_ie[m_top]:+.3f}")
    print("=" * 60)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    with open(out_dir / "patching.json", "w") as f:
        json.dump({
            "dataset": "07_Suicide_Watch_232K", "n_pairs": args.n, "n_used": n_used,
            "n_layers": n_layers, "min_ld_gap": MIN_LD_GAP,
            "attn_ie": attn_ie.tolist(), "mlp_ie": mlp_ie.tolist(),
            "peak_attn_layer": a_top, "peak_attn_ie": float(attn_ie[a_top]),
            "peak_mlp_layer": m_top, "peak_mlp_ie": float(mlp_ie[m_top]),
        }, f, indent=2)

    # ---- figure: IE curves by layer (the circuit-tracing plot) ----
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = list(range(n_layers))
    ax.plot(xs, attn_ie, "-o", ms=3, color="#e74c3c", label="Attention")
    ax.plot(xs, mlp_ie, "-s", ms=3, color="#3498db", label="MLP")
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("indirect effect (denoising)")
    ax.set_title("Where suicidality is written: layer-wise activation patching\n"
                 "binary suicide detection, 232K")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "patching_ie.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved results/sc2_binary/patching.json + patching_ie.png")


if __name__ == "__main__":
    main()
