#!/usr/bin/env python
"""Causal intent test — is the layer-16 direction what DRIVES the intent call?

sc2_intent_minpairs.py showed the layer-16 suicidality direction *separates*
genuine intent from lexically-matched mention (correlational). This script asks
whether that direction is what the model causally USES to make the distinction,
mirroring the faithfulness/sufficiency ladder used for the binary task.

NECESSITY (ablation). Project the direction out of the residual stream
(layers ABLATE_FROM..end, all positions) and re-measure the genuine-vs-control
behavioral AUC. If it collapses toward chance while a random-direction control
does not, the direction is causally necessary for the intent distinction.

SUFFICIENCY (steering). Add the direction (natural difference-of-means scale,
swept by a coefficient) to the residual stream of the NEGATIVE-projecting control
items (counterfactual / topical) and measure whether the model flips toward "Yes
(suicidal)". If genuine-intent steering flips mention toward a suicidal call while
a random direction does not, the direction is causally sufficient -- and, more
pointedly, the model can be made to treat "I would never kill myself" as intent by
injecting this one direction.

Direction = difference-of-means at layer 16, fit on the 232K binary TRAIN split
(the paper's central object), evaluated on the minimal-pairs set.

Output: results/sc2_binary/intent_causal.json + intent_causal.png
Run:    python analysis/sanity/sc2_intent_causal.py --controls topical,counterfactual
"""
import sys, json, argparse
from collections import defaultdict
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
from sklearn.metrics import roc_auc_score
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

N_TRAIN = 100
LAYER = 16
MAXLEN = 256
device = "cuda"
MINPAIRS = PROJECT_ROOT / "data" / "intent_minpairs" / "intent_minpairs_v1.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Causal intent test (ablation + steering)")
    p.add_argument("--controls", type=str, default="topical,counterfactual",
                   help="control conditions to contrast genuine intent against")
    p.add_argument("--steer_targets", type=str, default="counterfactual,topical",
                   help="conditions to steer toward 'Yes'")
    p.add_argument("--coefs", type=str, default="0,1,2,4",
                   help="steering coefficients (× natural diff-of-means)")
    p.add_argument("--ranks", type=str, default="1,2,4",
                   help="subspace ranks to ablate for the necessity test")
    p.add_argument("--ablate_from", type=int, default=11,
                   help="ablate the subspace at decoder layers ablate_from..end "
                        "(default 11: Track A's intent-writer heads are L11-15)")
    p.add_argument("--data", type=str, default=str(MINPAIRS),
                   help="path to the intent jsonl (synthetic minpairs or real set)")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def deflate_subspace(A, B, k):
    """k diff-of-means directions via iterative deflation; orthonormal [d,k].
    (Same construction as sc2_rankk.py.)"""
    A, B = A.copy(), B.copy()
    V = []
    for _ in range(k):
        d = A.mean(0) - B.mean(0)
        d = d / (np.linalg.norm(d) + 1e-8)
        V.append(d)
        A = A - np.outer(A @ d, d)
        B = B - np.outer(B @ d, d)
    Q, _ = np.linalg.qr(np.stack(V, axis=1))
    return Q[:, :k]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["text"])
    ctrl_conds = [c.strip() for c in args.controls.split(",") if c.strip()]
    steer_conds = [c.strip() for c in args.steer_targets.split(",") if c.strip()]
    coefs = [float(c) for c in args.coefs.split(",")]
    ranks = [int(r) for r in args.ranks.split(",")]
    ABLATE_FROM = args.ablate_from

    genuine = by_cond["genuine_intent"]
    control = [t for c in ctrl_conds for t in by_cond.get(c, [])]
    print(f"genuine n={len(genuine)}  control ({'+'.join(ctrl_conds)}) n={len(control)}")

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def hidden16(text):
        out = model(chat_ids(text), output_hidden_states=True)
        return out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()

    @torch.no_grad()
    def logit_diff(text):
        lg = model(chat_ids(text)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def auc_gc(genuine_scores, control_scores):
        y = np.array([1] * len(genuine_scores) + [0] * len(control_scores))
        return float(roc_auc_score(y, np.array(genuine_scores + control_scores)))

    # ---- fit direction on 232K TRAIN ----
    print("Fitting layer-16 direction on 232K train...")
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    h_sui = np.stack([hidden16(t) for t in sui[:N_TRAIN]])
    h_non = np.stack([hidden16(t) for t in non[:N_TRAIN]])
    d_raw = h_sui.mean(0) - h_non.mean(0)              # natural scale (for steering)
    r_raw = torch.tensor(d_raw, device=device, dtype=torch.float16)
    g = rng.standard_normal(d_raw.shape[0]); g = g / np.linalg.norm(g)
    r_rand_raw = torch.tensor(g * np.linalg.norm(d_raw), device=device, dtype=torch.float16)

    # rank-k intent subspace (deflation) + matched random subspace
    maxk = max(ranks)
    V_intent = deflate_subspace(h_sui, h_non, maxk)                      # [d, maxk]
    V_rand = np.linalg.qr(rng.standard_normal((d_raw.shape[0], maxk)))[0][:, :maxk]

    # ---- hooks: ablate a subspace (remove components) or steer (add vector) ----
    def make_subspace_ablate_hook(Vt):  # Vt: torch [d,k] orthonormal
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            V = Vt.to(h.dtype)
            h2 = h - (h @ V) @ V.T
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def make_steer_hook(vec, coef):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + coef * vec.to(h.dtype)
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def with_hooks(make):
        return [layers[i].register_forward_hook(make) for i in range(ABLATE_FROM, len(layers))]

    def measure(make=None):
        handles = with_hooks(make) if make else []
        gs = [logit_diff(t) for t in genuine]
        cs = [logit_diff(t) for t in control]
        for x in handles: x.remove()
        auc = auc_gc(gs, cs)
        gap = float(np.mean(gs) - np.mean(cs))   # the metric that matters under Yes-bias
        return auc, gap, float(np.mean(gs)), float(np.mean(cs))

    # ===================== NECESSITY (rank-k subspace ablation) =====================
    # NOTE: the genuine-vs-control AUC is near-saturated because the model's safety
    # Yes-bias keeps BOTH classes' logits positive, so ranking is preserved even when
    # the separation shrinks. The genuine-control LD GAP is the sensitive readout.
    print("Baseline...")
    base_auc, base_gap, base_g, base_c = measure(None)
    print(f"baseline: AUC {base_auc:.3f}  gap {base_gap:+.2f}  (genuine {base_g:+.2f}, control {base_c:+.2f})")

    necessity = {"baseline": {"auc": base_auc, "gap": base_gap, "ld": [base_g, base_c]},
                 "ablate_from": ABLATE_FROM, "by_rank": {}}
    print("\n" + "=" * 72)
    print(f"NECESSITY  (genuine vs {'+'.join(ctrl_conds)}; rank-k ablation from L{ABLATE_FROM})")
    print("=" * 72)
    print(f"{'k':>3}{'intent AUC':>12}{'intent gap':>12}{'gap collapse':>14}"
          f"{'rand gap':>10}{'rand collapse':>15}")
    best_collapse = 0.0
    for k in ranks:
        Vi = torch.tensor(V_intent[:, :k], device=device, dtype=torch.float16)
        Vr = torch.tensor(V_rand[:, :k], device=device, dtype=torch.float16)
        ia, ig, igg, igc = measure(make_subspace_ablate_hook(Vi))
        ra, rg, rgg, rgc = measure(make_subspace_ablate_hook(Vr))
        ic = 1.0 - ig / base_gap if base_gap else float("nan")   # fraction of gap removed
        rc = 1.0 - rg / base_gap if base_gap else float("nan")
        best_collapse = max(best_collapse, ic - max(rc, 0.0))
        necessity["by_rank"][k] = {"intent_auc": ia, "intent_gap": ig, "intent_gap_collapse": ic,
                                   "random_auc": ra, "random_gap": rg, "random_gap_collapse": rc,
                                   "intent_ld": [igg, igc], "random_ld": [rgg, rgc]}
        print(f"{k:>3}{ia:>12.3f}{ig:>12.2f}{ic:>13.0%}{rg:>10.2f}{rc:>14.0%}")
    necessary = best_collapse > 0.5
    necessity["necessary"] = bool(necessary)
    print("=" * 72)
    print(f"VERDICT: {'NECESSARY' if necessary else 'WEAK'} — intent ablation collapses the "
          f"genuine-control LD gap by up to {best_collapse:.0%} more than random")

    # ===================== SUFFICIENCY (steering) =====================
    steer_items = [t for c in steer_conds for t in by_cond.get(c, [])]
    base_steer_ld = np.array([logit_diff(t) for t in steer_items])
    base_flip = float((base_steer_ld > 0).mean())
    print("\n" + "=" * 64)
    print(f"SUFFICIENCY  (steer {'+'.join(steer_conds)} toward 'Yes'; n={len(steer_items)})")
    print("=" * 64)
    print(f"baseline: mean LD {base_steer_ld.mean():+.2f}, frac LD>0 = {base_flip:.2f}")
    sweep = {}
    for coef in coefs:
        if coef == 0:
            ld_i = base_steer_ld; ld_r = base_steer_ld
        else:
            h_ = with_hooks(make_steer_hook(r_raw, coef))
            ld_i = np.array([logit_diff(t) for t in steer_items])
            for x in h_: x.remove()
            h_ = with_hooks(make_steer_hook(r_rand_raw, coef))
            ld_r = np.array([logit_diff(t) for t in steer_items])
            for x in h_: x.remove()
        sweep[coef] = {
            "intent_mean_ld": float(ld_i.mean()), "intent_frac_yes": float((ld_i > 0).mean()),
            "random_mean_ld": float(ld_r.mean()), "random_frac_yes": float((ld_r > 0).mean()),
        }
        print(f"  coef={coef:>4}: intent LD {ld_i.mean():+.2f} (frac>0 {(ld_i>0).mean():.2f})   "
              f"random LD {ld_r.mean():+.2f} (frac>0 {(ld_r>0).mean():.2f})")
    print("=" * 64)

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"
    ensure_dir(out_dir)
    # tag by full file stem so paired runs (e.g. v2_raw vs v2_lemma) don't clobber;
    # fall back to the legacy real/synthetic tags for the original two files.
    stem = Path(args.data).stem
    if stem == "intent_minpairs_v1":
        tag = "synthetic"
    elif stem == "intent_real_v1":
        tag = "real"
    else:
        tag = stem.replace("intent_", "")
    json.dump({
        "dataset": Path(args.data).name, "layer": LAYER, "ablate_from": ABLATE_FROM,
        "controls": ctrl_conds, "n_genuine": len(genuine), "n_control": len(control),
        "necessity": necessity,
        "sufficiency": {"steer_conditions": steer_conds, "n_steer": len(steer_items),
                        "baseline_frac_yes": base_flip, "sweep": sweep},
    }, open(out_dir / f"intent_causal_{tag}.json", "w"), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = ranks
    ax1.plot(ks, [necessity["by_rank"][k]["intent_gap_collapse"] for k in ks], "-o",
             color="#e74c3c", label="intent subspace")
    ax1.plot(ks, [necessity["by_rank"][k]["random_gap_collapse"] for k in ks], "-s",
             color="#bbb", label="random subspace")
    ax1.set_xlabel("subspace rank k"); ax1.set_ylabel("genuine-control LD gap collapse")
    ax1.set_ylim(-0.1, 1.05); ax1.set_title(f"Necessity (rank-k ablation from L{ABLATE_FROM})")
    ax1.legend()
    cs = sorted(sweep)
    ax2.plot(cs, [sweep[c]["intent_frac_yes"] for c in cs], "-o", color="#e74c3c", label="intent dir")
    ax2.plot(cs, [sweep[c]["random_frac_yes"] for c in cs], "-s", color="#bbb", label="random dir")
    ax2.set_xlabel("steering coefficient"); ax2.set_ylabel("frac of mention flipped to 'Yes'")
    ax2.set_ylim(0, 1.05); ax2.set_title("Sufficiency (steering mention)"); ax2.legend()
    for a in (ax1, ax2): a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"intent_causal_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results/sc2_binary/intent_causal_{tag}.json + .png")


if __name__ == "__main__":
    main()
