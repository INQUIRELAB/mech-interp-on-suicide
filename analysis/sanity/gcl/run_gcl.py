#!/usr/bin/env python
"""Gated Causal-Ladder (GCL) — the reusable framework driver.

Runs the full validation-gated ladder on ANY registered concept (concept_config.py):

  GATE       behavioral + probe AUC vs lexical baseline  -> GO / NO-GO (stop if NO-GO)
  LOCATE     per-layer linear probe -> peak layer
  EXTRACT    per-layer difference-of-means direction
  NECESSARY  directional ablation (all layers)   vs. random direction
  LOW-RANK   rank-k subspace ablation             vs. random rank-k subspace
  SUFFICIENT steering on negative posts           vs. random dir + off-target readout

Every rung is paired with a matched control. The SAME code runs suicide and
toxicity; that the pipeline (not just the suicidality feature) transfers across
concepts is the framework claim. The gate stopping on a below-baseline concept
(the implicit-intent NO-GO, sc2c) is the proof it has teeth.

Usage:
  python analysis/sanity/gcl/run_gcl.py --concept suicide
  python analysis/sanity/gcl/run_gcl.py --concept toxicity
  python analysis/sanity/gcl/run_gcl.py --concept toxicity --model Qwen/Qwen2.5-7B-Instruct

Output: results/gcl/{concept}/{model_tag}.json
Needs a 4-bit 7-8B model on a >=12 GB GPU.
"""
import sys, json, argparse, re
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.models import ModelLoader
from concept_config import get_concept

MAXLEN = 256
BOOT = 2000
RANK_SWEEP = [1, 2, 3, 4, 6, 8]
CHANCE_HI = 0.60          # AUC at/below this counts as "driven to chance"
OFFTARGET_Q = "Is this text primarily about food or cooking?"
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="GCL framework driver")
    p.add_argument("--concept", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n", type=int, default=150, help="examples/class total")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def boot_ci(score_pos, score_neg, seed=None, B=BOOT):
    if seed is None:
        seed = get_seed() + 12345
    brng = np.random.default_rng(seed)
    sp, sn = np.asarray(score_pos), np.asarray(score_neg)
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    aucs = np.empty(B)
    for b in range(B):
        aucs[b] = roc_auc_score(y, np.r_[sp[brng.integers(0, len(sp), len(sp))],
                                         sn[brng.integers(0, len(sn), len(sn))]])
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def deflate_subspace(Hs, Hn, k):
    """k diff-of-means directions via iterative deflation, orthonormalized. [k, d]."""
    rs, rn = Hs.copy(), Hn.copy()
    dirs = []
    for _ in range(k):
        d = rs.mean(0) - rn.mean(0)
        u = d / (np.linalg.norm(d) + 1e-8)
        dirs.append(u)
        rs = rs - np.outer(rs @ u, u)
        rn = rn - np.outer(rn @ u, u)
    Q, _ = np.linalg.qr(np.stack(dirs, 1))   # [d, k] orthonormal
    return Q.T                                # [k, d]


def main():
    args = parse_args()
    concept = get_concept(args.concept)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", args.model.split("/")[-1]).lower()
    rng = np.random.default_rng(args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())

    print(f"=== GCL :: concept={concept.display_name} :: model={args.model} ===")
    pos, neg = concept.loader(args.n, args.seed)
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    print(f"loaded {n}/class")

    # ---- lexical baseline (the GATE reference) ----
    texts = pos + neg
    y = np.array([1] * n + [0] * n)
    Xtf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit_transform(texts)
    tfidf_auc = roc_auc_score(y, cross_val_predict(
        LogisticRegression(C=1.0, max_iter=2000), Xtf, y, cv=cv, method="predict_proba")[:, 1])
    print(f"lexical TF-IDF baseline AUC = {tfidf_auc:.3f}")

    # ---- model ----
    loader = ModelLoader(device=device)
    model, tok = loader.load_model(args.model, load_in_4bit=True, auto_quantize=False)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = tok.chat_template is not None
    yes_id = tok.encode(concept.pos_answer, add_special_tokens=False)[0]
    no_id = tok.encode(concept.neg_answer, add_special_tokens=False)[0]
    layers = model.model.layers
    nL = len(layers)
    print(f"{nL} layers; prompt={'chat' if use_chat else 'raw'}")

    def make_ids(text, question=None):
        prefix = concept.prompt_prefix
        q = question or concept.question
        body_ids = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = f"{prefix}\n\nText: {tok.decode(body_ids)}\n\n{q} Answer with Yes or No.\nAnswer:"
        if use_chat:
            s = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return tok(content, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def feat(text):
        out = model(make_ids(text), output_hidden_states=True)
        lg = out.logits[0, -1, :]
        ld = (lg[yes_id] - lg[no_id]).item()
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
        return ld, hs

    @torch.no_grad()
    def logit_diff(text, question=None):
        lg = model(make_ids(text, question)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def collect(ts, name):
        lds, hs = [], []
        for i, t in enumerate(ts):
            ld, h = feat(t); lds.append(ld); hs.append(h)
            if (i + 1) % 50 == 0:
                print(f"    {name} {i+1}/{len(ts)}")
        return np.array(lds), np.stack(hs)

    print("collecting activations...")
    ld_p, H_p = collect(pos, "pos")
    ld_n, H_n = collect(neg, "neg")

    # ================= GATE + LOCATE =================
    best = (0, 0.0); best_proba = None; per_layer = []
    for L in range(H_p.shape[1]):
        X = np.concatenate([H_p[:, L, :], H_n[:, L, :]])
        proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                                  X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba); per_layer.append({"layer": L, "auc": float(auc)})
        if auc > best[1]:
            best = (L, auc); best_proba = proba.copy()
    behav_auc = roc_auc_score(y, np.concatenate([ld_p, ld_n]))
    probe_ci = boot_ci(best_proba[:n], best_proba[n:])
    behav_ci = boot_ci(ld_p, ld_n)
    peak = best[0]
    go = (best[1] > tfidf_auc) and (behav_auc > CHANCE_HI)
    print(f"\nGATE: behavioral AUC {behav_auc:.3f}  probe best AUC {best[1]:.3f} @L{peak}  "
          f"vs baseline {tfidf_auc:.3f} -> {'GO' if go else 'NO-GO'}")

    report = {
        "concept": concept.name, "display_name": concept.display_name, "model": args.model,
        "model_tag": tag, "n_per_class": n, "n_layers": nL, "prompt": "chat" if use_chat else "raw",
        "lexical_baseline_auc": float(tfidf_auc),
        "gate": {"behavioral_auc": float(behav_auc), "behavioral_ci95": behav_ci,
                 "probe_best_auc": float(best[1]), "probe_best_layer": peak,
                 "probe_best_ci95": probe_ci, "decision": "GO" if go else "NO-GO"},
        "probe_per_layer": per_layer,
    }

    if not go:
        report["note"] = "Gate failed: behavior does not clear baseline. Ladder not run (this is the framework working)."
        _save(report, concept.name, tag)
        print("NO-GO -> ladder halted. (The gate doing its job.)")
        return

    # ================= EXTRACT (per-layer diff-of-means) =================
    trs = slice(0, args.n_train)
    dirs = H_p[trs].mean(0) - H_n[trs].mean(0)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)   # [nL+1, d]
    rand = rng.standard_normal(dirs.shape); rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)

    pos_te, neg_te = pos[args.n_train:args.n_train + args.n_test], neg[args.n_train:args.n_train + args.n_test]
    yte = np.array([1] * len(pos_te) + [0] * len(neg_te))

    def behav_auc_te():
        s = [logit_diff(t) for t in pos_te]; nlds = [logit_diff(t) for t in neg_te]
        return roc_auc_score(yte, np.array(s + nlds)), float(np.mean(s) - np.mean(nlds))

    def proj_hook(vecs):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = vecs.to(h.dtype); h2 = h - (h @ v).unsqueeze(-1) * v
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def subspace_hook(R):  # R: [d, k] torch on device
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            Rt = R.to(h.dtype)
            h2 = h - (h @ Rt) @ Rt.T
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def add_hooks(hook_factory, per_layer_arg):
        return [layers[i].register_forward_hook(hook_factory(per_layer_arg[i])) for i in range(nL)]

    # ================= NECESSARY (directional ablation) =================
    Rdir = {i: torch.tensor(dirs[i + 1], device=device) for i in range(nL)}
    Rrnd = {i: torch.tensor(rand[i + 1], device=device) for i in range(nL)}
    base_auc, base_gap = behav_auc_te()
    h_ = add_hooks(proj_hook, Rdir); abl_auc, abl_gap = behav_auc_te(); [x.remove() for x in h_]
    h_ = add_hooks(proj_hook, Rrnd); rnd_auc, rnd_gap = behav_auc_te(); [x.remove() for x in h_]
    print(f"NECESSARY: base {base_auc:.3f} -> ablate {abl_auc:.3f} (random {rnd_auc:.3f}); "
          f"gap {base_gap:.2f}->{abl_gap:.2f}")
    report["necessary"] = {"baseline_auc": base_auc, "ablate_auc": abl_auc, "random_auc": rnd_auc,
                           "baseline_gap": base_gap, "ablate_gap": abl_gap, "random_gap": rnd_gap}

    # ================= LOW-RANK (rank-k subspace ablation) =================
    rank_rows = []
    kstar = None
    for k in RANK_SWEEP:
        subs = np.stack([deflate_subspace(H_p[trs, L, :], H_n[trs, L, :], k) for L in range(nL + 1)])  # [nL+1, k, d]
        Rk = {i: torch.tensor(subs[i + 1].T, device=device) for i in range(nL)}                        # [d,k]
        # random orthonormal rank-k control
        rsub = np.stack([np.linalg.qr(rng.standard_normal((dirs.shape[1], k)))[0] for _ in range(nL)])  # [nL,d,k]
        Rkr = {i: torch.tensor(rsub[i], device=device) for i in range(nL)}
        h_ = add_hooks(subspace_hook, Rk); auc_k, _ = behav_auc_te(); [x.remove() for x in h_]
        h_ = add_hooks(subspace_hook, Rkr); auc_kr, _ = behav_auc_te(); [x.remove() for x in h_]
        rank_rows.append({"k": k, "auc": auc_k, "random_auc": auc_kr})
        print(f"LOW-RANK k={k}: AUC {auc_k:.3f} (random {auc_kr:.3f})")
        if kstar is None and auc_k <= CHANCE_HI:
            kstar = k
    report["low_rank"] = {"sweep": rank_rows, "k_star": kstar, "chance_threshold": CHANCE_HI}

    # ================= SUFFICIENT (steering + specificity) =================
    delta = ((H_p[trs].mean(0) - H_n[trs].mean(0)) * dirs).sum(1)  # [nL+1]
    WIN = [i for i in range(max(8, peak - 4), min(nL, peak + 5))]

    def add_dir_hooks(coef, vecs):
        amap = {i: torch.tensor(coef * delta[i + 1] * vecs[i + 1], device=device) for i in WIN}
        def mk(i):
            add = amap[i]
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h2 = h + add.to(h.dtype)
                return (h2,) + out[1:] if isinstance(out, tuple) else h2
            return hook
        return [layers[i].register_forward_hook(mk(i)) for i in WIN]

    base_on = np.array([logit_diff(t) for t in neg_te])
    base_off = np.array([logit_diff(t, OFFTARGET_Q) for t in neg_te])
    threshold = (np.mean([logit_diff(t) for t in pos_te]) + base_on.mean()) / 2

    suff_rows = []
    for c in [1.0, 2.0]:
        h_ = add_dir_hooks(c, dirs); on = np.array([logit_diff(t) for t in neg_te]); [x.remove() for x in h_]
        h_ = add_dir_hooks(c, dirs); off = np.array([logit_diff(t, OFFTARGET_Q) for t in neg_te]); [x.remove() for x in h_]
        suff_rows.append({"coef": c, "on_target_ld": float(on.mean()), "off_target_ld": float(off.mean()),
                          "on_shift": float(on.mean() - base_on.mean()),
                          "off_shift": float(off.mean() - base_off.mean()),
                          "frac_over_threshold": float((on > threshold).mean())})
        print(f"SUFFICIENT c={c}: on {on.mean():.2f} (flip {(on>threshold).mean():.0%})  off {off.mean():.2f}")
    h_ = add_dir_hooks(1.0, rand); rnd_on = np.array([logit_diff(t) for t in neg_te]); [x.remove() for x in h_]

    # Judge specificity at the clean dose c=1 (paper notes c>=2 saturates off-manifold).
    c1 = suff_rows[0]
    rnd_shift = float(rnd_on.mean() - base_on.mean())
    specific = (c1["on_shift"] > 3 * abs(c1["off_shift"]) and c1["on_shift"] > 2.0
                and c1["on_shift"] > 3 * abs(rnd_shift))
    report["sufficient"] = {
        "window": WIN, "threshold": float(threshold),
        "baseline_on_ld": float(base_on.mean()), "baseline_off_ld": float(base_off.mean()),
        "sweep": suff_rows, "random_c1_on_ld": float(rnd_on.mean()), "random_c1_shift": rnd_shift,
        "specificity_dose": 1.0,
        "on_target_shift": c1["on_shift"], "off_target_shift": c1["off_shift"],
        "specific": bool(specific),
    }
    print(f"SPECIFICITY @c=1: on-shift {c1['on_shift']:+.2f}  off-shift {c1['off_shift']:+.2f}  "
          f"random-shift {rnd_shift:+.2f} -> {'specific' if specific else 'NOT specific'}")

    # ================= SCORECARD =================
    sc = {
        "gate_GO": go,
        "necessary": base_auc - abl_auc > 3 * max(base_auc - rnd_auc, 0.01),
        "low_rank": kstar is not None,
        "sufficient": report["sufficient"]["specific"] and suff_rows[0]["frac_over_threshold"] > 0.5,
    }
    report["scorecard"] = sc
    print("\n" + "=" * 64)
    print(f"GCL SCORECARD :: {concept.display_name} :: {args.model}")
    for k, v in sc.items():
        print(f"  {k:12s}: {'PASS' if v else 'fail'}")
    print("=" * 64)
    _save(report, concept.name, tag)


def _save(report, concept_name, tag):
    outdir = PROJECT_ROOT / "results" / "gcl" / concept_name
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{tag}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
