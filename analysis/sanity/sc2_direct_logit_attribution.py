#!/usr/bin/env python
"""Direct Logit Attribution (DLA) on the validated binary task.

The logit lens (sc2_logitlens.py) reads the residual stream at each layer through
the unembedding, showing WHEN the Yes-No decision forms. DLA goes one level finer:
it decomposes the FINAL Yes-No logit difference into the additive contribution of
each component that writes to the residual stream (the embedding, and every
layer's attention and MLP output), so we can say WHICH components push the model
toward "suicide".

Linearization. With a final RMSNorm and unembedding, the metric is linear in the
final residual h_L:
    LD = logit(Yes) - logit(No) = h_L . d_eff,
    d_eff = (W_U[Yes] - W_U[No]) * norm_weight / rms(h_L).
Because the residual stream is exactly the sum of component outputs
(h_L = embed + sum_l attn_l + mlp_l), each component c contributes c . d_eff to LD,
and the contributions sum to LD exactly (we report the reconstruction error as a
sanity check; the only approximation is treating rms(h_L) as a fixed scale, the
standard DLA convention). All read at the final ("Answer:") token position.

We report, per component, the mean DLA on suicidal vs non-suicidal posts and their
DIFFERENCE (suicide - non): a positive difference means the component differentially
drives a "Yes (suicidal)" call. This locates the decision-writing components and
should agree with the mid-layer hotspot from patching / AtP / head knockout.

Output: results/sc2_binary/dla.json + dla.png
Run:    python analysis/sanity/sc2_direct_logit_attribution.py --n 50
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
    p = argparse.ArgumentParser(description="Direct logit attribution on binary 232K")
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

    layers = model.model.layers
    nL = len(layers)

    # unembedding direction w = W_U[yes] - W_U[no]
    lm_w = model.lm_head.weight
    if lm_w.dtype in (torch.uint8,) or lm_w.__class__.__name__ == "Params4bit":
        raise RuntimeError("lm_head is quantized; DLA needs an fp16 unembedding (re-load with lm_head unquantized).")
    w = (lm_w[yes_id].float() - lm_w[no_id].float()).detach()      # [d_model]
    norm_w = model.model.norm.weight.float().detach()              # [d_model]
    eps = getattr(model.model.norm, "variance_epsilon", getattr(model.config, "rms_norm_eps", 1e-5))

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    cap = {}

    def mk_hook(key):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[key] = o[0, -1, :].detach().float()   # last-position contribution
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(mk_hook("embed"))]
    for L in range(nL):
        handles.append(layers[L].self_attn.o_proj.register_forward_hook(mk_hook(("attn", L))))
        handles.append(layers[L].mlp.down_proj.register_forward_hook(mk_hook(("mlp", L))))

    # The TRUE pre-final-norm residual is the INPUT to model.model.norm. (Note:
    # output_hidden_states[nL] is the POST-norm state, which would break both the
    # rms scale and the additive component decomposition -- hence we hook here.)
    def norm_prehook(module, args):
        cap["prenorm"] = args[0][0, -1, :].detach().float()
        return args
    handles.append(model.model.norm.register_forward_pre_hook(norm_prehook))

    @torch.no_grad()
    def dla_one(text):
        cap.clear()
        out = model(chat_ids(text))
        h_final = cap["prenorm"]                                   # pre-final-norm residual
        lg = out.logits[0, -1, :]
        ld_true = (lg[yes_id] - lg[no_id]).item()
        rms = torch.sqrt((h_final ** 2).mean() + eps)
        d_eff = (w * norm_w / rms)                                 # [d_model]
        attn = np.array([float(cap[("attn", L)] @ d_eff) for L in range(nL)])
        mlp = np.array([float(cap[("mlp", L)] @ d_eff) for L in range(nL)])
        emb = float(cap["embed"] @ d_eff)
        recon = emb + attn.sum() + mlp.sum()
        return ld_true, recon, emb, attn, mlp

    def collect(texts, tag):
        LDt, LDr, EMB, A, M = [], [], [], [], []
        for i, t in enumerate(texts):
            ld, rc, e, a, m = dla_one(t)
            LDt.append(ld); LDr.append(rc); EMB.append(e); A.append(a); M.append(m)
            if (i + 1) % 25 == 0:
                print(f"    {tag} {i+1}/{len(texts)}")
        return (np.array(LDt), np.array(LDr), np.array(EMB), np.stack(A), np.stack(M))

    print("DLA on suicidal posts..."); sLDt, sLDr, sE, sA, sM = collect(sui, "sui")
    print("DLA on non-suicidal posts..."); nLDt, nLDr, nE, nA, nM = collect(non, "non")
    for h in handles:
        h.remove()

    recon_err = float(np.mean(np.abs(np.r_[sLDt, nLDt] - np.r_[sLDr, nLDr])))
    attn_diff = sA.mean(0) - nA.mean(0)     # per-layer attn: suicide - non
    mlp_diff = sM.mean(0) - nM.mean(0)
    a_top = int(np.argmax(attn_diff)); m_top = int(np.argmax(mlp_diff))

    print("\n" + "=" * 64)
    print(f"DIRECT LOGIT ATTRIBUTION (binary 232K, n={args.n}/class)")
    print(f"  reconstruction |LD_true - LD_recon| mean = {recon_err:.3e}  (should be ~0)")
    print(f"  mean LD: suicide {sLDt.mean():+.2f}  non {nLDt.mean():+.2f}")
    print(f"  top attn layer (suicide-non): L{a_top} = {attn_diff[a_top]:+.3f}")
    print(f"  top mlp  layer (suicide-non): L{m_top} = {mlp_diff[m_top]:+.3f}")
    print("  per-layer attn DLA difference (suicide - non), L8-L20:")
    for L in range(8, 21):
        print(f"    L{L:2d}: attn {attn_diff[L]:+.3f}  mlp {mlp_diff[L]:+.3f}")
    print("=" * 64)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    json.dump({"dataset": "07_Suicide_Watch_232K", "method": "direct_logit_attribution",
               "n_per_class": args.n, "n_layers": nL,
               "reconstruction_abs_error_mean": recon_err,
               "mean_ld": {"suicide": float(sLDt.mean()), "non": float(nLDt.mean())},
               "embed_dla": {"suicide": float(sE.mean()), "non": float(nE.mean())},
               "attn_dla_suicide": sA.mean(0).tolist(), "attn_dla_non": nA.mean(0).tolist(),
               "mlp_dla_suicide": sM.mean(0).tolist(), "mlp_dla_non": nM.mean(0).tolist(),
               "attn_dla_diff": attn_diff.tolist(), "mlp_dla_diff": mlp_diff.tolist(),
               "peak_attn_diff_layer": a_top, "peak_mlp_diff_layer": m_top},
              open(out_dir / "dla.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = list(range(nL))
    ax.plot(xs, attn_diff, "-o", ms=3, color="#e74c3c", label="Attention (suicide - non)")
    ax.plot(xs, mlp_diff, "-s", ms=3, color="#3498db", label="MLP (suicide - non)")
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("DLA toward 'Yes' (logit units)")
    ax.set_title("Direct logit attribution: which components decide 'suicidal'\n"
                 "binary 232K, suicide minus non-suicide")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "dla.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/sc2_binary/dla.json + dla.png")


if __name__ == "__main__":
    main()
