#!/usr/bin/env python
"""Sparse autoencoder at L16 on the VALIDATED binary task (07_Suicide_Watch_232K).

The original Phase 8 SAE ran on DeepSuiMind MLP activations (a below-baseline
task). This re-runs the SAE on the residual stream at layer 16 -- the layer the
diff-of-means direction, probe, and ablation all peak at -- on the real binary
task, so the discovered features explain a behavior the model actually performs.

Pipeline (matches sc2_faithfulness.py for model/prompt/readout):
  1. Collect residual-stream states at L16 (hidden_states[16], final token) for
     suicide and non-suicide posts.
  2. Train an overcomplete SAE (d_sae = expansion * d_model) with an L1 penalty.
  3. Rank features by suicide-selectivity (mean act on suicide - non-suicide).
  4. Causally validate: inject the top suicide-selective features into the L16
     residual of NON-suicide posts and measure how far the behavioral logit
     difference moves and how many posts flip past the class midpoint.

A small set of features carrying most of the selectivity, whose injection flips
non-suicide posts, is monosemantic-feature-level evidence for the same low-rank
circuit found by rank-k ablation.

Output: results/sc2_binary/sae_L16.json
        results/sc2_binary/sae_weights_L16.pt
        results/sc2_binary/sae_features_L16.png

Run:  python analysis/sanity/sc2_sae.py --n-collect 150 --epochs 200
Needs the 4-bit Llama on a >=12 GB GPU. SAE training itself is fast.
"""
import sys, json, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, torch.nn as nn, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

LAYER = 16          # hidden_states[16] == output of decoder layer 15
HOOK_LAYER = 15     # intervene on layers[15] output (feeds hidden_states[16])
MAXLEN = 256
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="SAE at L16 on binary 232K")
    p.add_argument("--n-collect", type=int, default=150, help="examples/class for SAE (default 150)")
    p.add_argument("--n-causal", type=int, default=50, help="non-suicide posts to steer (default 50)")
    p.add_argument("--expansion", type=int, default=8, help="SAE overcomplete factor (default 8)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--l1", type=float, default=5e-3)
    p.add_argument("--steer-scale", type=float, default=2.0, help="feature injection scale")
    p.add_argument("--top-k", type=int, default=10, help="top features to inject")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


class SparseAutoencoder(nn.Module):
    def __init__(self, d_input, d_sae):
        super().__init__()
        self.d_input, self.d_sae = d_input, d_sae
        self.W_enc = nn.Parameter(torch.randn(d_sae, d_input) * (1.0 / d_input ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_input, d_sae) * (1.0 / d_sae ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(d_input))
        with torch.no_grad():
            self.W_dec.data = self.W_enc.data.T.clone()
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True) + 1e-8

    def forward(self, x):
        z = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat, z


def train_sae(acts, d_model, d_sae, epochs, l1, lr=1e-3, bs=32, dev="cuda"):
    mean = acts.mean(0); std = acts.std(0).clamp(min=1e-8)
    x = ((acts - mean) / std).to(dev)
    sae = SparseAutoencoder(d_model, d_sae).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = x.shape[0]
    hist = {"recon": [], "l1": [], "l0": []}
    for ep in range(epochs):
        perm = torch.randperm(n)
        er = el = e0 = nb = 0
        for s in range(0, n, bs):
            b = x[perm[s:s + bs]]
            x_hat, z = sae(b)
            recon = ((b - x_hat) ** 2).mean()
            l1pen = z.abs().mean()
            loss = recon + l1 * l1pen
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True) + 1e-8
            er += recon.item(); el += l1pen.item(); e0 += (z > 0).float().sum(1).mean().item(); nb += 1
        sched.step()
        hist["recon"].append(er / nb); hist["l1"].append(el / nb); hist["l0"].append(e0 / nb)
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{epochs}: recon={er/nb:.5f} L1={el/nb:.5f} L0={e0/nb:.1f}")
    return sae, mean, std, hist


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui_c, non_c = sui[: args.n_collect], non[: args.n_collect]
    non_causal = non[args.n_collect: args.n_collect + args.n_causal]

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers
    d_model = model.config.hidden_size

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def hidden_and_ld(text):
        out = model(chat_ids(text), output_hidden_states=True)
        h = out.hidden_states[LAYER][0, -1, :].float().cpu()
        lg = out.logits[0, -1, :]
        return h, (lg[yes_id] - lg[no_id]).item()

    # ---- Step 1: collect activations ----
    print(f"Collecting L{LAYER} activations ({args.n_collect}/class)...")
    h_sui, ld_sui = [], []
    h_non, ld_non = [], []
    for t in sui_c:
        h, ld = hidden_and_ld(t); h_sui.append(h); ld_sui.append(ld)
    for t in non_c:
        h, ld = hidden_and_ld(t); h_non.append(h); ld_non.append(ld)
    H_sui = torch.stack(h_sui); H_non = torch.stack(h_non)
    acts = torch.cat([H_sui, H_non], 0)
    mid_threshold = (np.mean(ld_sui) + np.mean(ld_non)) / 2
    print(f"  behavioral LD: suicide mean={np.mean(ld_sui):.2f}, non mean={np.mean(ld_non):.2f}, "
          f"midpoint={mid_threshold:.2f}")

    # ---- Step 2: train SAE ----
    d_sae = args.expansion * d_model
    print(f"Training SAE d_input={d_model} d_sae={d_sae}...")
    sae_dev = "cuda" if torch.cuda.is_available() else "cpu"
    sae, a_mean, a_std, hist = train_sae(acts, d_model, d_sae, args.epochs, args.l1, dev=sae_dev)

    # ---- Step 3: feature selectivity ----
    sae.eval()
    with torch.no_grad():
        _, z_sui = sae(((H_sui - a_mean) / a_std).to(sae_dev))
        _, z_non = sae(((H_non - a_mean) / a_std).to(sae_dev))
    z_sui, z_non = z_sui.cpu(), z_non.cpu()
    mean_sui, mean_non = z_sui.mean(0), z_non.mean(0)
    diff = mean_sui - mean_non                      # >0 => suicide-selective
    frac_active = (torch.cat([z_sui, z_non], 0) > 0).float().mean(0)
    n_dead = int((frac_active == 0).sum())
    l0 = float((torch.cat([z_sui, z_non], 0) > 0).float().sum(1).mean())
    top = diff.topk(args.top_k)
    top_idx = top.indices.tolist()
    print(f"\n  d_sae={d_sae}  dead={n_dead}  mean L0={l0:.1f}")
    print(f"  top {args.top_k} suicide-selective features:")
    for i in top_idx:
        print(f"    F{i:5d}: diff={diff[i]:+.4f}  active(sui)={ (z_sui[:,i]>0).float().mean():.0%}"
              f"  active(non)={ (z_non[:,i]>0).float().mean():.0%}")

    # ---- Step 4: causal validation (inject features into non-suicide posts) ----
    print(f"\nCausal validation: inject top {args.top_k} features (scale={args.steer_scale}) "
          f"into {len(non_causal)} non-suicide posts...")
    sui_feat_target = mean_sui.to(sae_dev)         # target feature values from suicide class
    W_dec = sae.W_dec.data.to(sae_dev)
    b_dec = sae.b_dec.data.to(sae_dev)
    a_mean_d, a_std_d = a_mean.to(sae_dev), a_std.to(sae_dev)
    top_idx_t = torch.tensor(top_idx, device=sae_dev)

    def make_inject_hook():
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            last = h[0, -1, :].float()
            norm = ((last - a_mean_d.to(last.device)) / a_std_d.to(last.device)).to(sae_dev)
            _, z = sae(norm.unsqueeze(0)); z = z.squeeze(0)
            z[top_idx_t] = sui_feat_target[top_idx_t] * args.steer_scale
            x_hat = (z @ W_dec.T + b_dec)
            recon = (x_hat * a_std_d + a_mean_d).to(h.device, h.dtype)
            h2 = h.clone(); h2[0, -1, :] = recon
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    @torch.no_grad()
    def ld_of(ids):
        lg = model(ids).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    base_lds, steer_lds = [], []
    for t in non_causal:
        ids = chat_ids(t)
        base_lds.append(ld_of(ids))
        hd = layers[HOOK_LAYER].register_forward_hook(make_inject_hook())
        try:
            steer_lds.append(ld_of(ids))
        finally:
            hd.remove()
    base_lds, steer_lds = np.array(base_lds), np.array(steer_lds)
    mean_shift = float((steer_lds - base_lds).mean())
    flip_rate = float((steer_lds > mid_threshold).mean())
    base_above = float((base_lds > mid_threshold).mean())
    print(f"  mean LD: {base_lds.mean():.2f} -> {steer_lds.mean():.2f}  (shift {mean_shift:+.2f})")
    print(f"  fraction above class midpoint: {base_above:.0%} -> {flip_rate:.0%}")

    # ---- save ----
    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    torch.save({"state_dict": sae.state_dict(), "d_input": d_model, "d_sae": d_sae,
                "act_mean": a_mean, "act_std": a_std, "history": hist,
                "top_features": top_idx}, PROJECT_ROOT / "results" / "sc2_binary" / "sae_weights_L16.pt")
    with open(PROJECT_ROOT / "results" / "sc2_binary" / "sae_L16.json", "w") as f:
        json.dump({
            "dataset": "07_Suicide_Watch_232K", "layer": LAYER,
            "n_collect": args.n_collect, "d_model": d_model, "d_sae": d_sae,
            "expansion": args.expansion, "l1": args.l1, "epochs": args.epochs,
            "n_dead": n_dead, "mean_l0": l0,
            "top_features": [{"index": int(i), "selectivity": float(diff[i]),
                              "frac_active_suicide": float((z_sui[:, i] > 0).float().mean()),
                              "frac_active_nonsuicide": float((z_non[:, i] > 0).float().mean())}
                             for i in top_idx],
            "causal": {"steer_scale": args.steer_scale, "top_k": args.top_k,
                       "n_causal": len(non_causal), "class_midpoint_ld": mid_threshold,
                       "baseline_mean_ld": float(base_lds.mean()),
                       "steered_mean_ld": float(steer_lds.mean()),
                       "mean_ld_shift": mean_shift,
                       "frac_above_midpoint_baseline": base_above,
                       "frac_above_midpoint_steered": flip_rate},
        }, f, indent=2)

    # ---- figure ----
    # Rendered at the Computational Linguistics print column (5.31 in wide) with serif
    # 8 pt type, so \includegraphics does no scaling and the in-figure text matches the
    # body text height. Three panels stacked vertically so each is full column width
    # (large and readable); panel meaning lives in the caption (Top/Middle/Bottom), so
    # the in-figure titles are dropped to keep the panels uncluttered.
    COLW = 5.31
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    fig, ax = plt.subplots(3, 1, figsize=(COLW, 5.7))
    ax[0].plot(hist["recon"], label="recon", lw=1.2); ax[0].plot(hist["l1"], label="L1", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    alive = frac_active.numpy() > 0
    ax[1].scatter(mean_non.numpy()[alive], mean_sui.numpy()[alive], s=5, alpha=0.25, c="#9aa1a9", lw=0)
    ax[1].scatter(mean_non.numpy()[top_idx], mean_sui.numpy()[top_idx], s=24, c="#b0383d",
                  marker="^", label="top suicide-selective")
    lim = max(mean_sui.numpy()[alive].max(), mean_non.numpy()[alive].max()) * 1.1
    ax[1].plot([0, lim], [0, lim], "--", c="grey", lw=0.8, alpha=0.6)
    ax[1].set_xlabel("mean act (non-suicide)"); ax[1].set_ylabel("mean act (suicide)")
    ax[1].legend(loc="upper left")
    ax[2].hist(base_lds, bins=20, alpha=0.65, label="baseline", color="#9aa1a9")
    ax[2].hist(steer_lds, bins=20, alpha=0.65, label="feature-injected", color="#3e7d4f")
    ax[2].axvline(mid_threshold, color="k", ls="--", lw=0.8, label="class midpoint")
    ax[2].set_xlabel("logit diff (Yes$-$No)"); ax[2].set_ylabel("count")
    ax[2].legend(loc="upper right")
    fig.tight_layout(h_pad=1.0)
    fig.savefig(str(PROJECT_ROOT / "results" / "sc2_binary" / "sae_features_L16.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved sae_L16.json + sae_weights_L16.pt + sae_features_L16.png")


if __name__ == "__main__":
    main()
