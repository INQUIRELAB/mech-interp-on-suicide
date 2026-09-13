#!/usr/bin/env python
"""Attribution Patching (AtP*) on the validated binary task.

The exact denoising patch in sc2_patching.py costs ~(2 + 2*n_layers) forward
passes per pair, and the per-head knockout (sc2_head_knockout.py) is only
affordable at n=20/class -> wide CIs, no individually significant head. AtP
(Nanda 2023; Syed et al. 2023; Kramar et al. AtP* 2024) approximates the patching
indirect effect with a FIRST-ORDER Taylor term, so ONE forward + ONE backward
pass yields an effect estimate for EVERY layer and EVERY one of the 1024 heads at
once. We can therefore run more pairs and get a dense, powered causal map.

Denoising AtP. For a node with activation a, replacing its corrupt value with the
clean value changes the metric (Yes-No logit diff measured on the corrupt run) by
approximately

    Delta_metric ~= (a_clean - a_corrupt) . d(metric)/d(a) |_{a_corrupt}

We normalize by (LD_clean - LD_corrupt), exactly like the real IE, so AtP_IE is on
the same [-1,1]-ish scale and directly comparable to sc2_patching.json.

Graph correctness: we make ONLY the token embedding a leaf requiring grad (one
detach at the input), so the entire downstream network is a single connected
differentiable graph with NO internal cuts. Component activations are captured as
non-leaf tensors via hooks with retain_grad(); a single backward populates all
their .grad. Head-level effects come from the o_proj INPUT (the concatenated
per-head outputs, sliced by head_dim -- GQA-safe, same slice the knockout zeroes);
summing a layer's head attributions equals that layer's attention attribution
because o_proj is linear.

Validation: we report Spearman/Pearson correlation of the AtP per-layer curves
against the exact patching IE (results/sc2_binary/patching.json) and the AtP
per-head map against the knockout map (head_knockout.json), if present.

Output: results/sc2_binary/attribution_patching.json + attribution_patching.png
Run:    python analysis/sanity/sc2_attribution_patching.py --n 40
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

MAXLEN = 200          # modest to keep the backward graph inside 12 GB
MIN_LD_GAP = 1.0
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Attribution patching (AtP*) on binary 232K")
    p.add_argument("--n", type=int, default=40, help="clean/corrupt pairs (default 40)")
    p.add_argument("--maxlen", type=int, default=MAXLEN)
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    maxlen = args.maxlen

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
    nL = len(layers)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    print(f"{nL} layers, {n_heads} heads, head_dim={head_dim}; {args.n} pairs, maxlen={maxlen}")

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: maxlen - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False)["input_ids"]

    def left_pad(a, b):
        L = max(len(a), len(b))
        def pad(x):
            k = L - len(x)
            return [pad_id] * k + x, [0] * k + [1] * len(x)
        ai, am = pad(a); bi, bm = pad(b)
        T = lambda v: torch.tensor([v], device=device)
        return (T(ai), T(am)), (T(bi), T(bm))

    # ---- hook plumbing -------------------------------------------------------
    # capture store: keys ('zin', L) = o_proj input ; ('mlp', L) = down_proj output
    cap = {}
    grad_mode = {"on": False}  # when on, retain_grad on captured (corrupt) tensors

    def embed_leaf_hook(module, inp, out):
        # one detach at the input so the whole net is a connected grad graph
        return out.detach().requires_grad_(True)

    def make_zin_prehook(L):
        def prehook(module, args):
            z = args[0]
            cap[("zin", L)] = z
            if grad_mode["on"]:
                z.retain_grad()
            return args
        return prehook

    def make_mlp_hook(L):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[("mlp", L)] = o
            if grad_mode["on"]:
                o.retain_grad()
            return out
        return hook

    h_embed = model.model.embed_tokens.register_forward_hook(embed_leaf_hook)
    handles = [h_embed]
    for L in range(nL):
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_zin_prehook(L)))
        handles.append(layers[L].mlp.down_proj.register_forward_hook(make_mlp_hook(L)))

    def metric(ids, mask):
        lg = model(input_ids=ids, attention_mask=mask).logits[0, -1, :]
        return lg[yes_id] - lg[no_id]

    attn_head = np.zeros((nL, n_heads))
    attn_layer = np.zeros(nL)
    mlp_layer = np.zeros(nL)
    n_used = 0

    for pi, (s_text, n_text) in enumerate(zip(sui, non)):
        (cids, cmask), (nids, nmask) = left_pad(chat_ids(s_text), chat_ids(n_text))

        # ---- clean run (no grad): cache clean activations + LD_clean ----
        grad_mode["on"] = False
        cap.clear()
        with torch.no_grad():
            ld_clean = metric(cids, cmask).item()
        zin_clean = {L: cap[("zin", L)].detach().clone() for L in range(nL)}
        mlp_clean = {L: cap[("mlp", L)].detach().clone() for L in range(nL)}

        # ---- corrupt run WITH grad ----
        grad_mode["on"] = True
        cap.clear()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            m_corrupt = metric(nids, nmask)
            ld_corrupt = m_corrupt.item()
            denom = ld_clean - ld_corrupt
            if abs(denom) < MIN_LD_GAP:
                print(f"  pair {pi}: |gap|={abs(denom):.2f} < {MIN_LD_GAP}, skipped")
                continue
            m_corrupt.backward()

        # ---- accumulate AtP attributions ----
        for L in range(nL):
            z = cap[("zin", L)]
            o = cap[("mlp", L)]
            if z.grad is None or o.grad is None:
                continue
            zc = zin_clean[L]; oc = mlp_clean[L]
            # head-level: slice the concatenated head outputs by head_dim
            dz = (zc - z.detach()).float()          # [1, seq, n_heads*head_dim]
            gz = z.grad.float()
            contrib = (dz * gz)[0]                   # [seq, n_heads*head_dim]
            contrib = contrib.view(contrib.shape[0], n_heads, head_dim).sum(dim=(0, 2))  # [n_heads]
            attn_head[L] += (contrib / denom).cpu().numpy()
            attn_layer[L] += float(contrib.sum().item() / denom)
            # mlp layer
            dm = (oc - o.detach()).float()
            mlp_layer[L] += float((dm * o.grad.float()).sum().item() / denom)

        n_used += 1
        print(f"  pair {pi}: clean LD={ld_clean:+.2f} corrupt LD={ld_corrupt:+.2f} [{n_used} used]")

    for h in handles:
        h.remove()
    if n_used == 0:
        raise RuntimeError("No usable pairs; increase --n.")
    attn_head /= n_used; attn_layer /= n_used; mlp_layer /= n_used

    # ---- top heads ----
    flat = [(L, h, float(attn_head[L, h])) for L in range(nL) for h in range(n_heads)]
    flat.sort(key=lambda r: -abs(r[2]))
    top_heads = [{"layer": L, "head": h, "atp_ie": v} for L, h, v in flat[:15]]
    a_top = int(np.argmax(np.abs(attn_layer))); m_top = int(np.argmax(np.abs(mlp_layer)))

    # ---- validation vs exact patching + knockout, if present ----
    def corr(a, b):
        from scipy.stats import spearmanr, pearsonr
        a, b = np.asarray(a), np.asarray(b)
        return {"spearman": float(spearmanr(a, b).statistic), "pearson": float(pearsonr(a, b)[0])}

    validation = {}
    pj = PROJECT_ROOT / "results" / "sc2_binary" / "patching.json"
    if pj.exists():
        p = json.load(open(pj))
        if len(p.get("attn_ie", [])) == nL:
            validation["vs_patching_attn"] = corr(attn_layer, p["attn_ie"])
            validation["vs_patching_mlp"] = corr(mlp_layer, p["mlp_ie"])
    kj = PROJECT_ROOT / "results" / "sc2_binary" / "head_knockout.json"
    if kj.exists():
        k = json.load(open(kj))
        kd = np.array(k["auc_drop"]) if "auc_drop" in k else None
        if kd is not None and kd.shape == (nL, n_heads):
            validation["vs_knockout_head"] = corr(np.abs(attn_head).flatten(), np.abs(kd).flatten())

    print("\n" + "=" * 64)
    print(f"ATTRIBUTION PATCHING (AtP*, binary 232K, {n_used} pairs)")
    print(f"  peak attn-layer IE: L{a_top} = {attn_layer[a_top]:+.3f}")
    print(f"  peak mlp-layer  IE: L{m_top} = {mlp_layer[m_top]:+.3f}")
    print("  top heads by |AtP IE|:")
    for t in top_heads[:8]:
        print(f"    L{t['layer']:2d}H{t['head']:2d}: {t['atp_ie']:+.4f}")
    if validation:
        print("  validation (correlation with exact methods):")
        for k_, v in validation.items():
            print(f"    {k_}: spearman={v['spearman']:+.3f} pearson={v['pearson']:+.3f}")
    print("=" * 64)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    json.dump({"dataset": "07_Suicide_Watch_232K", "method": "attribution_patching_AtP",
               "n_pairs": args.n, "n_used": n_used, "n_layers": nL, "n_heads": n_heads,
               "maxlen": maxlen, "attn_layer_ie": attn_layer.tolist(),
               "mlp_layer_ie": mlp_layer.tolist(), "attn_head_ie": attn_head.tolist(),
               "peak_attn_layer": a_top, "peak_mlp_layer": m_top,
               "top_heads": top_heads, "validation": validation},
              open(out_dir / "attribution_patching.json", "w"), indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = list(range(nL))
    ax[0].plot(xs, attn_layer, "-o", ms=3, color="#e74c3c", label="Attention (AtP)")
    ax[0].plot(xs, mlp_layer, "-s", ms=3, color="#3498db", label="MLP (AtP)")
    ax[0].axhline(0, ls="--", c="grey", lw=1)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("AtP indirect effect")
    ax[0].set_title("Per-layer attribution (one fwd+bwd/pair)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    im = ax[1].imshow(attn_head, aspect="auto", cmap="RdBu_r",
                      vmin=-np.abs(attn_head).max(), vmax=np.abs(attn_head).max())
    ax[1].set_xlabel("head"); ax[1].set_ylabel("layer")
    ax[1].set_title("Per-head AtP IE (all 1024 heads)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(out_dir / "attribution_patching.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/sc2_binary/attribution_patching.json + .png")


if __name__ == "__main__":
    main()
