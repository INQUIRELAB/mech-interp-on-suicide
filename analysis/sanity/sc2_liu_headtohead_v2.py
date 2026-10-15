#!/usr/bin/env python
"""Liu-style head patching vs. train-fit direction ablation, rerun on the
manuscript checkpoint (meta-llama/Llama-3.1-8B-Instruct, fp16).

Replacement for sc2_liu_headtohead.py (original script not recoverable; its
output is results/sc2_binary/liu_headtohead_unsloth.json). The design follows
the recorded metadata of that run, with the sample sizes raised and the
sampling made identical to sc2_faithfulness.py so the manuscript's direction-
ablation test set is a strict subset of this one.

Direction ablation (same protocol as sc2_faithfulness.py):
  - pool: 07_Suicide_Watch_232K, len(text) > 80, shuffled with SEED (default 0)
  - train = first N_TRAIN/class  -> difference-of-means direction at hidden_states[16]
  - test  = next N_TEST/class    -> the first 100/class are exactly the manuscript test set
  - ablate the unit direction at decoder layers 15..31, all positions
  - controls: the manuscript's random direction (same RNG draw) + extra random directions
  - per-example LD saved; stratified paired bootstrap CIs for AUC and AUC drops

Liu-style head patching (denoising, z = per-head o_proj input):
  - pairs drawn only from held-out posts after the ablation test block (never train)
  - exact full-prompt token-length match, no truncation
  - IE = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt), signed, NOT clipped
  - primary set: first N_PAIRS pairs with |LD_clean - LD_corrupt| >= MIN_GAP
    (screened/excluded pairs are counted and saved); raw LD recovery saved for all
  - per-head pair-bootstrap CIs, one-sample t-tests with BH-FDR, split-half
    stability, and the old run's top-20 heads evaluated as pre-specified heads

Run (from project root):
  HF_TOKEN=... python analysis/sanity/sc2_liu_headtohead_v2.py
Outputs: results/sc2_binary/liu_headtohead_metallama_v2{.json,_ablation_examples.csv,
         _patching_pairs.csv,_ie.npz}
"""
import sys, os, json, csv, hashlib, argparse, time, platform
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from scipy import stats
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16        # hidden_states[16] == output of decoder layer 15
ABLATE_FROM = 15  # hook decoder layers 15..last (sc2_faithfulness.py)
MAXLEN = 256
MANUSCRIPT = {"baseline_auc": 0.975, "direction_ablation_auc": 0.832,
              "random_ablation_auc": 0.975, "n_test_per_class": 100,
              "source": "paper/npjai_main.tex lines 75, 160-161, 213-214"}
OUT = PROJECT_ROOT / "results" / "sc2_binary"
STEM = "liu_headtohead_metallama_v2"
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-pairs", type=int, default=60, help="primary pairs with |gap|>=min-gap")
    p.add_argument("--max-screen", type=int, default=150)
    p.add_argument("--min-gap", type=float, default=1.0)
    p.add_argument("--pair-pool", type=int, default=6000, help="held-out posts/class scanned for length matches")
    p.add_argument("--n-random", type=int, default=5, help="extra random-direction controls")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def h16(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def boot_auc(y, scores_by_cond, rng, n_boot):
    """Stratified paired bootstrap: same resampled examples for every condition."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: np.empty(n_boot) for k in scores_by_cond}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores_by_cond.items():
            out[k][b] = roc_auc_score(y[idx], s[idx])
    return out


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    args = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    ensure_dir(OUT)

    # ---------------- data: identical draw to sc2_faithfulness.py ----------------
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    NT, NE = args.n_train, args.n_test
    sui_tr, non_tr = sui[:NT], non[:NT]
    sui_te, non_te = sui[NT:NT + NE], non[NT:NT + NE]
    train_hash = {h16(t) for t in sui_tr + non_tr}
    test_hash = [h16(t) for t in sui_te + non_te]
    overlap_train_test = sum(h in train_hash for h in test_hash)
    dup_within_test = len(test_hash) - len(set(test_hash))

    # ---------------- model ----------------
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    resolved_rev = getattr(model.config, "_commit_hash", None)
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers
    cfg = model.config
    n_layers, n_heads = len(layers), cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")

    def chat_ids(text):
        body_ids = tok(text, add_special_tokens=False)["input_ids"]
        truncated = len(body_ids) > MAXLEN - 80
        content = prefix + tok.decode(body_ids[: MAXLEN - 80]) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"], truncated

    @torch.no_grad()
    def ld_batch(ids):
        lg = model(ids.to(device), logits_to_keep=1).logits[:, -1, :].float()
        return (lg[:, yes_id] - lg[:, no_id]).cpu().numpy()

    @torch.no_grad()
    def hidden16(text):
        out = model(chat_ids(text)[0].to(device), output_hidden_states=True)
        return out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()

    # ---------------- direction ablation ----------------
    print(f"[{time.time()-t0:.0f}s] fitting direction on train ({NT}/class)", flush=True)
    hs = np.stack([hidden16(t) for t in sui_tr]); hn = np.stack([hidden16(t) for t in non_tr])
    d = hs.mean(0) - hn.mean(0); d /= np.linalg.norm(d)
    g = rng.standard_normal(d.shape[0]); g /= np.linalg.norm(g)   # same draw as faithfulness.py
    rng_extra = np.random.default_rng(args.seed + 1)
    rand_dirs = [g] + [(lambda v: v / np.linalg.norm(v))(rng_extra.standard_normal(d.shape[0]))
                       for _ in range(args.n_random)]

    def make_hook(vec):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = vec.to(h.dtype)
            h2 = h - (h @ v).unsqueeze(-1) * v
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    test_texts = sui_te + non_te
    y = np.array([1] * len(sui_te) + [0] * len(non_te))
    test_ids = [chat_ids(t) for t in test_texts]

    def score_all(vec=None):
        hs_ = []
        if vec is not None:
            v = torch.tensor(vec, device=device, dtype=torch.float16)
            hs_ = [layers[i].register_forward_hook(make_hook(v)) for i in range(ABLATE_FROM, n_layers)]
        try:
            return np.array([ld_batch(i)[0] for i, _ in test_ids])
        finally:
            for x in hs_: x.remove()

    print(f"[{time.time()-t0:.0f}s] scoring test ({NE}/class) x {2 + len(rand_dirs)} conditions", flush=True)
    sc = {"baseline": score_all(), "direction_ablation": score_all(d)}
    for k, v in enumerate(rand_dirs):
        sc[f"random_{k}"] = score_all(v)

    def ablation_summary(mask, tag):
        yy = y[mask]; s = {k: v[mask] for k, v in sc.items()}
        auc = {k: float(roc_auc_score(yy, v)) for k, v in s.items()}
        bt = boot_auc(yy, {k: s[k] for k in ("baseline", "direction_ablation", "random_0")},
                      np.random.default_rng(args.seed + 7), args.n_boot)
        rand_aucs = [auc[f"random_{k}"] for k in range(len(rand_dirs))]
        return {
            "subset": tag, "n_per_class": int(yy.sum()),
            "auc": auc,
            "auc_ci95": {k: ci(v) for k, v in bt.items()},
            "auc_drop_direction": auc["baseline"] - auc["direction_ablation"],
            "auc_drop_direction_ci95": ci(bt["baseline"] - bt["direction_ablation"]),
            "auc_drop_random0": auc["baseline"] - auc["random_0"],
            "auc_drop_random0_ci95": ci(bt["baseline"] - bt["random_0"]),
            "drop_direction_minus_drop_random0_ci95": ci(bt["random_0"] - bt["direction_ablation"]),
            "random_auc_range_all_controls": [min(rand_aucs), max(rand_aucs)],
            "mean_ld": {k: {"suicide": float(v[yy == 1].mean()), "non_suicide": float(v[yy == 0].mean())}
                        for k, v in s.items()},
        }

    first100 = np.r_[np.arange(min(100, NE)), len(sui_te) + np.arange(min(100, NE))]
    m100 = np.zeros(len(y), bool); m100[first100] = True
    abl_full = ablation_summary(np.ones(len(y), bool), f"full_test_{NE}_per_class")
    abl_ms = ablation_summary(m100, "manuscript_test_set_first_100_per_class")

    with open(OUT / f"{STEM}_ablation_examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "label", "in_manuscript_test_subset", "n_tokens", "truncated", *sc.keys()])
        for i, t in enumerate(test_texts):
            w.writerow([test_hash[i], int(y[i]), int(m100[i]), test_ids[i][0].shape[1],
                        int(test_ids[i][1]), *[f"{sc[k][i]:.5f}" for k in sc]])
    print(f"[{time.time()-t0:.0f}s] ablation AUC full: " +
          ", ".join(f"{k}={abl_full['auc'][k]:.3f}" for k in ("baseline", "direction_ablation", "random_0")), flush=True)

    # ---------------- Liu-style head patching ----------------
    excluded_hash = train_hash | set(test_hash)
    pool_s = [t for t in sui[NT + NE: NT + NE + args.pair_pool] if h16(t) not in excluded_hash]
    pool_n = [t for t in non[NT + NE: NT + NE + args.pair_pool] if h16(t) not in excluded_hash]
    by_len_s, by_len_n = {}, {}
    for t in pool_s:
        ids, tr = chat_ids(t)
        if not tr: by_len_s.setdefault(ids.shape[1], []).append((t, ids))
    for t in pool_n:
        ids, tr = chat_ids(t)
        if not tr: by_len_n.setdefault(ids.shape[1], []).append((t, ids))
    cand = []
    for L in sorted(by_len_s.keys() & by_len_n.keys()):
        for a, b in zip(by_len_s[L], by_len_n[L]):
            cand.append((L, a, b))
    order = rng.permutation(len(cand))
    cand = [cand[i] for i in order]
    print(f"[{time.time()-t0:.0f}s] {len(cand)} exact-length candidate pairs", flush=True)

    o_projs = [layers[L].self_attn.o_proj for L in range(n_layers)]

    @torch.no_grad()
    def cache_clean(ids):
        store = {}
        def mk(L):
            def pre(m, inp): store[L] = inp[0][0].detach().clone()
            return pre
        hs_ = [o_projs[L].register_forward_pre_hook(mk(L)) for L in range(n_layers)]
        try:
            ld = ld_batch(ids)[0]
        finally:
            for x in hs_: x.remove()
        return ld, store

    @torch.no_grad()
    def patch_layer(ids_c, clean_z, L):
        """Batch row h patches head h; row n_heads is an unpatched consistency row."""
        B = n_heads + 1
        def pre(m, inp):
            x = inp[0].clone()
            for h in range(n_heads):
                x[h, :, h * hd:(h + 1) * hd] = clean_z[:, h * hd:(h + 1) * hd]
            return (x,) + tuple(inp[1:])
        hk = o_projs[L].register_forward_pre_hook(pre)
        try:
            out = ld_batch(ids_c.repeat(B, 1))
        finally:
            hk.remove()
        return out[:n_heads], out[n_heads]

    pairs, ie_list, raw_list = [], [], []
    n_screened = n_excluded = 0
    max_consistency_err = 0.0
    for L_tok, (ts, ids_s), (tn, ids_n) in cand:
        if len([p for p in pairs if p["primary"]]) >= args.n_pairs or n_screened >= args.max_screen:
            break
        n_screened += 1
        ld_clean, z = cache_clean(ids_s)
        ld_corr = ld_batch(ids_n)[0]
        gap = ld_clean - ld_corr
        primary = abs(gap) >= args.min_gap
        n_excluded += int(not primary)
        raw = np.zeros((n_layers, n_heads), np.float32)
        for L in range(n_layers):
            lp, ld_unp = patch_layer(ids_n, z[L], L)
            raw[L] = lp - ld_corr
            max_consistency_err = max(max_consistency_err, abs(ld_unp - ld_corr))
        ie = raw / gap if gap != 0 else np.full_like(raw, np.nan)
        pairs.append({"pair_index": len(pairs), "hash_suicide": h16(ts), "hash_non_suicide": h16(tn),
                      "n_tokens": int(L_tok), "ld_clean": float(ld_clean), "ld_corrupt": float(ld_corr),
                      "gap": float(gap), "primary": bool(primary)})
        ie_list.append(ie); raw_list.append(raw)
        np.savez_compressed(OUT / f"{STEM}_ie.partial.npz", ie=np.stack(ie_list), raw=np.stack(raw_list))
        print(f"[{time.time()-t0:.0f}s] pair {len(pairs)} len={L_tok} gap={gap:+.2f} "
              f"{'primary' if primary else 'EXCLUDED'}  max IE={np.nanmax(ie):+.3f}", flush=True)

    IE, RAW = np.stack(ie_list), np.stack(raw_list)
    prim = np.array([p["primary"] for p in pairs])
    IEp, RAWp, gaps = IE[prim], RAW[prim], np.array([p["gap"] for p in pairs])[prim]
    nP = len(IEp)
    np.savez_compressed(OUT / f"{STEM}_ie.npz", ie=IE, raw_ld_recovery=RAW, primary=prim,
                        gap=np.array([p["gap"] for p in pairs]))
    (OUT / f"{STEM}_ie.partial.npz").unlink(missing_ok=True)

    brng = np.random.default_rng(args.seed + 11)
    bidx = brng.integers(0, nP, size=(args.n_boot, nP))
    mean_ie = IEp.mean(0)
    boot_mean = np.stack([IEp[b].mean(0) for b in bidx])                      # [B, L, H]
    ratio_ie = RAWp.mean(0) / gaps.mean()                                      # ratio-of-means IE
    boot_ratio = np.stack([RAWp[b].mean(0) / gaps[b].mean() for b in bidx])
    tstat, pval = stats.ttest_1samp(IEp.reshape(nP, -1), 0.0, axis=0)
    qval = bh(pval).reshape(n_layers, n_heads); pval = pval.reshape(n_layers, n_heads)
    lo, hi = np.percentile(boot_mean, 2.5, 0), np.percentile(boot_mean, 97.5, 0)
    rlo, rhi = np.percentile(boot_ratio, 2.5, 0), np.percentile(boot_ratio, 97.5, 0)

    def head_row(L, h):
        return {"layer": int(L), "head": int(h), "mean_ie": float(mean_ie[L, h]),
                "ci95": [float(lo[L, h]), float(hi[L, h])],
                "ratio_of_means_ie": float(ratio_ie[L, h]), "ratio_ci95": [float(rlo[L, h]), float(rhi[L, h])],
                "p_ttest": float(pval[L, h]), "q_bh": float(qval[L, h])}

    flat = np.argsort(-mean_ie.ravel())
    top20 = [head_row(*divmod(int(i), n_heads)) for i in flat[:20]]
    layer_boot = boot_mean.mean(2)
    layer_rows = [{"layer": L, "mean_ie": float(mean_ie[L].mean()), "ci95": ci(layer_boot[:, L]),
                   "sum_ie_over_heads": float(mean_ie[L].sum()), "sum_ci95": ci(boot_mean[:, L].sum(1))}
                  for L in range(n_layers)]

    # split-half stability of the head ranking
    perm = np.random.default_rng(args.seed + 13).permutation(nP)
    A, Bh = IEp[perm[: nP // 2]].mean(0).ravel(), IEp[perm[nP // 2:]].mean(0).ravel()
    topA, topB = set(np.argsort(-A)[:20]), set(np.argsort(-Bh)[:20])
    split_half = {"spearman_all_heads": float(stats.spearmanr(A, Bh)[0]),
                  "top20_overlap": len(topA & topB),
                  "top1_half_a": list(divmod(int(np.argmax(A)), n_heads)),
                  "top1_half_b": list(divmod(int(np.argmax(Bh)), n_heads))}

    # old run's top-20 heads as pre-specified hypotheses
    old_path = OUT / "liu_headtohead_unsloth.json"
    prespec = None
    if old_path.exists():
        old = json.load(open(old_path))
        olds = old["liu_style_head_patching"]["top20_heads_descriptive"]
        prespec = [{**head_row(o["layer"], o["head"]), "old_mean_ie": o["mean_ie"], "old_ci95": o["ci95"],
                    "q_bh_within_prespecified_20": None} for o in olds]
        q20 = bh([r["p_ttest"] for r in prespec])
        for r, q in zip(prespec, q20): r["q_bh_within_prespecified_20"] = float(q)

    with open(OUT / f"{STEM}_patching_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) + ["max_head_ie", "argmax_head"])
        w.writeheader()
        for p, ie in zip(pairs, IE):
            k = int(np.nanargmax(ie)); w.writerow({**p, "max_head_ie": float(np.nanmax(ie)),
                                                  "argmax_head": f"L{k // n_heads}H{k % n_heads}"})

    # ---------------- comparison against manuscript ----------------
    def mismatch(name, rerun, ms, tol):
        return {"quantity": name, "rerun": rerun, "manuscript": ms, "abs_diff": abs(rerun - ms),
                "tolerance": tol, "mismatch": abs(rerun - ms) > tol}
    cmp = [mismatch(f"{k} AUC ({abl_ms['subset']})", abl_ms["auc"][a], MANUSCRIPT[k], 0.02)
           for k, a in (("baseline_auc", "baseline"), ("direction_ablation_auc", "direction_ablation"),
                        ("random_ablation_auc", "random_0"))]
    for c, a in zip(cmp, ("baseline", "direction_ablation", "random_0")):
        c["manuscript_value_inside_rerun_ci95"] = abl_ms["auc_ci95"][a][0] <= c["manuscript"] <= abl_ms["auc_ci95"][a][1]

    res = {
        "design": "Liu-style single-head z (o_proj input) denoising patching vs train-fit direction "
                  "ablation; same model and prompt; held-out test only. Different estimands; not a "
                  "method-superiority test. Replacement implementation of sc2_liu_headtohead.py.",
        "dataset": "07_Suicide_Watch_232K",
        "model": MODEL_ID, "model_revision_requested": args.revision, "model_revision_resolved": resolved_rev,
        "manuscript_model_revision": "not recorded in src/models.py (unpinned); exact match cannot be verified",
        "precision": "fp16", "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "host": platform.node(), "seed": args.seed,
        "runtime_s": round(time.time() - t0, 1),
        "split": {"direction_fit": f"train: first {NT}/class of seeded shuffle (as sc2_faithfulness.py)",
                  "direction_ablation_eval": f"test: next {NE}/class",
                  "patching": f"held-out posts after the test block (next {args.pair_pool}/class scanned), "
                              "train and test hashes excluded",
                  "train_test_hash_overlap": overlap_train_test, "duplicate_hashes_within_test": dup_within_test,
                  "test_truncated_examples": int(sum(tr for _, tr in test_ids))},
        "direction_ablation": {"layer_fit": LAYER, "ablate_from": ABLATE_FROM, "n_random_controls": len(rand_dirs),
                               "full_test": abl_full, "manuscript_test_subset": abl_ms},
        "liu_style_head_patching": {
            "metric": "signed IE=(LD_patched-LD_corrupt)/(LD_clean-LD_corrupt); raw, no clipping",
            "site": "per-head slice of o_proj input (z), all positions, clean->corrupt",
            "pair_matching": "exact full-prompt token length, no truncation",
            "min_abs_gap_primary": args.min_gap, "pairs_screened": n_screened,
            "pairs_excluded_small_gap": n_excluded, "n_pairs_primary": nP,
            "candidate_pairs_available": len(cand),
            "batch_consistency_max_abs_ld_err": float(max_consistency_err),
            "top20_heads": top20,
            "n_heads_q_bh_lt_0.05_positive": int(((qval < 0.05) & (mean_ie > 0)).sum()),
            "n_heads_q_bh_lt_0.05_negative": int(((qval < 0.05) & (mean_ie < 0)).sum()),
            "split_half_stability": split_half,
            "old_run_top20_as_prespecified": prespec,
            "layers": layer_rows,
            "per_pair": pairs,
            "per_pair_ie_file": f"{STEM}_ie.npz",
        },
        "comparison_with_manuscript": cmp,
        "comparison_with_old_unsloth_run": None if not old_path.exists() else {
            "old_n_test_per_class": old["test_examples_per_class"], "old_pairs": old["patching_pairs_used"],
            "old_auc": old["direction_ablation"]["auc"],
            "old_top_head": old["liu_style_head_patching"]["top20_heads_descriptive"][0],
            "rerun_top_head": top20[0]},
        "manuscript_claims_changed": False,
    }
    json.dump(res, open(OUT / f"{STEM}.json", "w"), indent=2)
    print(json.dumps({"ablation_manuscript_subset": abl_ms["auc"], "ablation_full": abl_full["auc"],
                      "patching_top5": top20[:5], "split_half": split_half,
                      "comparison": cmp}, indent=1), flush=True)
    print(f"Saved {OUT / (STEM + '.json')}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
