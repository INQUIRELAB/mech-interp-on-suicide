#!/usr/bin/env python
"""Downstream-task ablation table (the analogue of Liu et al. 2025, Table 2).

Liu et al. validate their components by ablating them and remeasuring a real
downstream IR metric (relevance-judgment F1, reranking NDCG@10) against a
Random-N control, with significance stars. Our faithfulness test (sc2_faithfulness)
shows the suicidality DIRECTION is necessary, but reports AUC only. This script
reports the same ablations as a deployed-classifier degradation table: F1, recall,
and precision at a FIXED decision threshold (calibrated once on the unablated model,
as a deployment would), plus threshold-free AUC, each with a 2000-sample bootstrap
95% CI and a paired-bootstrap significance test of the drop vs. the full model.

Rows (mirroring Liu's Random vs. targeted structure):
  Full model                  -- no ablation (baseline)
  Random-direction ablation   -- equal-norm random direction (the control)
  Suicidality-direction abl.  -- the rank-1 difference-of-means direction
  Rank-4 subspace ablation    -- the 2-4 dim subspace from sc2_rankk

Ablation = project the direction/subspace out of the residual stream at layers
ABLATE_FROM..end, all positions (identical to sc2_faithfulness). The decision
threshold is the F1-maximizing threshold on the Full-model TRAIN logit differences,
then frozen and applied to every condition's TEST outputs.

Output: results/sc2_binary/downstream_table.json + downstream_table.png
Run:    python analysis/sanity/sc2_downstream_table.py --n_train 100 --n_test 100
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
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir

LAYER = 16
ABLATE_FROM = 15
MAXLEN = 256
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description="Downstream ablation table (Liu Table 2 analogue)")
    p.add_argument("--n_train", type=int, default=100, help="per class, fit direction + threshold")
    p.add_argument("--n_test", type=int, default=100, help="per class, evaluation")
    p.add_argument("--rank", type=int, default=4, help="subspace rank for the rank-k row")
    p.add_argument("--seed", type=int, default=get_seed())
    return p.parse_args()


def deflate_subspace(A, B, k):
    A, B = A.copy(), B.copy(); V = []
    for _ in range(k):
        d = A.mean(0) - B.mean(0); d = d / (np.linalg.norm(d) + 1e-8); V.append(d)
        A = A - np.outer(A @ d, d); B = B - np.outer(B @ d, d)
    return np.linalg.qr(np.stack(V, axis=1))[0][:, :k]


def best_f1_threshold(scores, y):
    """Threshold on scores maximizing F1 (deployment calibration)."""
    cand = np.unique(scores)
    mids = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand
    best_t, best_f1 = 0.0, -1.0
    for t in mids:
        f = f1_score(y, (scores > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    texts, labels = ds["text"], ds["label"]
    sui = [t for t, l in zip(texts, labels) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(texts, labels) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    sui_tr, sui_te = sui[: args.n_train], sui[args.n_train: args.n_train + args.n_test]
    non_tr, non_te = non[: args.n_train], non[args.n_train: args.n_train + args.n_test]

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
        return model(chat_ids(text), output_hidden_states=True).hidden_states[LAYER][0, -1, :].float().cpu().numpy()

    @torch.no_grad()
    def logit_diff(text):
        lg = model(chat_ids(text)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    # ---- fit direction + subspace on TRAIN ----
    print("Fitting direction + subspace on train...")
    h_sui = np.stack([hidden16(t) for t in sui_tr])
    h_non = np.stack([hidden16(t) for t in non_tr])
    d = h_sui.mean(0) - h_non.mean(0); d_unit = d / np.linalg.norm(d)
    r_dir = torch.tensor(d_unit, device=device, dtype=torch.float16)
    g = rng.standard_normal(d.shape[0]); g = g / np.linalg.norm(g)
    r_rand = torch.tensor(g, device=device, dtype=torch.float16)
    V_sub = torch.tensor(deflate_subspace(h_sui, h_non, args.rank), device=device, dtype=torch.float16)

    # threshold calibrated on FULL-model train LDs
    tr_ld = np.array([logit_diff(t) for t in sui_tr] + [logit_diff(t) for t in non_tr])
    tr_y = np.array([1] * len(sui_tr) + [0] * len(non_tr))
    thr = best_f1_threshold(tr_ld, tr_y)
    print(f"calibrated decision threshold (train, max-F1): LD>{thr:.3f}")

    # ---- ablation hooks (identical convention to sc2_faithfulness) ----
    def dir_hook(vec):
        def h(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            v = vec.to(o.dtype); o2 = o - (o @ v).unsqueeze(-1) * v
            return (o2,) + out[1:] if isinstance(out, tuple) else o2
        return h

    def sub_hook(Vt):
        def h(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            V = Vt.to(o.dtype); o2 = o - (o @ V) @ V.T
            return (o2,) + out[1:] if isinstance(out, tuple) else o2
        return h

    def with_hooks(mk):
        return [layers[i].register_forward_hook(mk) for i in range(ABLATE_FROM, len(layers))]

    yte = np.array([1] * len(sui_te) + [0] * len(non_te))

    def eval_condition(mk=None):
        handles = with_hooks(mk) if mk else []
        ld = np.array([logit_diff(t) for t in sui_te] + [logit_diff(t) for t in non_te])
        for x in handles: x.remove()
        return ld

    conditions = [
        ("Full model", None),
        ("Random-direction ablation", dir_hook(r_rand)),
        ("Suicidality-direction ablation", dir_hook(r_dir)),
        (f"Rank-{args.rank} subspace ablation", sub_hook(V_sub)),
    ]

    def metrics(ld):
        pred = (ld > thr).astype(int)
        return {"auc": float(roc_auc_score(yte, ld)), "f1": float(f1_score(yte, pred, zero_division=0)),
                "recall": float(recall_score(yte, pred, zero_division=0)),
                "precision": float(precision_score(yte, pred, zero_division=0)),
                "accuracy": float(accuracy_score(yte, pred))}

    def boot(ld_full, ld_cond, n_boot=2000):
        """Paired bootstrap: metric CIs for cond + p(drop<=0) vs full for f1 & auc."""
        n = len(yte); br = np.random.default_rng(get_seed() + 1)
        f1s, aucs, df1, dauc = [], [], [], []
        for _ in range(n_boot):
            idx = br.integers(0, n, n)
            y = yte[idx]
            if len(set(y.tolist())) < 2: continue
            pc = (ld_cond[idx] > thr).astype(int); pf = (ld_full[idx] > thr).astype(int)
            f1c = f1_score(y, pc, zero_division=0); ac = roc_auc_score(y, ld_cond[idx])
            f1f = f1_score(y, pf, zero_division=0); af = roc_auc_score(y, ld_full[idx])
            f1s.append(f1c); aucs.append(ac); df1.append(f1f - f1c); dauc.append(af - ac)
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        # one-sided p: fraction of bootstrap where the drop is <= 0 (i.e. no degradation)
        p_f1 = float(np.mean(np.array(df1) <= 0)); p_auc = float(np.mean(np.array(dauc) <= 0))
        return {"f1_ci": ci(f1s), "auc_ci": ci(aucs), "p_drop_f1": p_f1, "p_drop_auc": p_auc}

    print("Evaluating conditions on held-out test...")
    results = {}
    ld_full = None
    for name, mk in conditions:
        ld = eval_condition(mk)
        if name == "Full model": ld_full = ld
        m = metrics(ld)
        b = boot(ld_full, ld) if ld_full is not None else {}
        results[name] = {**m, **b}
        print(f"  {name:34} AUC {m['auc']:.3f}  F1 {m['f1']:.3f}  "
              f"R {m['recall']:.3f}  P {m['precision']:.3f}")

    # ---- print table ----
    print("\n" + "=" * 88)
    print(f"DOWNSTREAM SUICIDE-DETECTION ABLATION TABLE (test n={args.n_test}/class, thr LD>{thr:.2f})")
    print("=" * 88)
    print(f"{'condition':34}{'AUC':>8}{'F1':>8}{'Recall':>9}{'Prec':>8}{'F1 drop':>10}{'sig':>5}")
    full = results["Full model"]
    for name, _ in conditions:
        r = results[name]
        drop = full["f1"] - r["f1"]
        sig = "*" if r.get("p_drop_f1", 1.0) < 0.05 and name != "Full model" else ""
        print(f"{name:34}{r['auc']:>8.3f}{r['f1']:>8.3f}{r['recall']:>9.3f}"
              f"{r['precision']:>8.3f}{drop:>+10.3f}{sig:>5}")
    print("=" * 88)
    print("* drop in F1 vs full model significant at p<0.05 (paired bootstrap)")

    out_dir = PROJECT_ROOT / "results" / "sc2_binary"; ensure_dir(out_dir)
    json.dump({"dataset": "07_Suicide_Watch_232K", "layer": LAYER, "ablate_from": ABLATE_FROM,
               "rank": args.rank, "n_train": args.n_train, "n_test": args.n_test,
               "threshold": thr, "metric_at": "fixed train-calibrated threshold",
               "conditions": results}, open(out_dir / "downstream_table.json", "w"), indent=2)

    # plot: F1 with CI per condition
    names = [n for n, _ in conditions]
    f1s = [results[n]["f1"] for n in names]
    los = [results[n].get("f1_ci", [results[n]["f1"], results[n]["f1"]])[0] for n in names]
    his = [results[n].get("f1_ci", [results[n]["f1"], results[n]["f1"]])[1] for n in names]
    err = [[f - l for f, l in zip(f1s, los)], [h - f for f, h in zip(f1s, his)]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#3498db", "#bbb", "#e74c3c", "#c0392b"]
    ax.bar(range(len(names)), f1s, yerr=err, capsize=4, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Full", "Random\ndir", "Suicidality\ndir", f"Rank-{args.rank}"], fontsize=9)
    ax.set_ylabel("downstream F1 (suicide detection)"); ax.set_ylim(0, 1.05)
    ax.set_title("Ablating the suicidality feature degrades downstream detection\n"
                 "(fixed deployment threshold; random-direction control)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(str(out_dir / "downstream_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/sc2_binary/downstream_table.json + downstream_table.png")


if __name__ == "__main__":
    main()
