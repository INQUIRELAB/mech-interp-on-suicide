#!/usr/bin/env python
"""Does reading the internals catch risk the model's own output misses?

The scale result says models encode suicidality they do not act on. This asks
the deployment-facing version of that question at 7-8B: on the cases where the
model itself answers wrongly, is the internal axis still right?

Protocol, all on held-out data:
  1. fit a probe and a difference-of-means direction on N_FIT/class of 07-232K,
     at the layer the gate selected;
  2. score N_EVAL/class of held-out text -- one forward per text yields both the
     behavioral logit difference and the hidden states, so the two readouts see
     exactly the same inputs;
  3. calibrate one decision threshold per readout on a disjoint split (the
     model has a strong affirmative bias, so argmax is not the right comparison
     -- see the gate script);
  4. report the head-to-head, then restrict to the model's own errors and ask
     what the probe says there.

The number that matters is FN recovery: of the suicidal posts the model's
behavioral readout calls non-suicidal, what fraction does the internal readout
place above its own threshold? A monitor is only worth its complexity if that
is well above zero.

Usage:
    MI_PRECISION=fp16 python analysis/sanity/sc2_internal_monitor.py
    MI_PRECISION=fp16 python analysis/sanity/sc2_internal_monitor.py --n-eval 250
"""
import sys, json, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from src.models import ModelLoader
from src.analysis_utils import (
    PROMPT_TEMPLATES, ensure_dir, load_dataset_by_category, get_text,
)

MAXLEN = 256
device = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=None,
                   help="probe layer; default: read it from the gate results")
    p.add_argument("--n-fit", type=int, default=500, help="per class, to fit")
    p.add_argument("--n-eval", type=int, default=1000, help="per class, held out")
    p.add_argument("--n-cal", type=int, default=200, help="per class, threshold split")
    p.add_argument("--n-ext", type=int, default=500, help="per class, per external set")
    p.add_argument("--out", default="results/sc2_binary/internal_monitor.json")
    return p.parse_args()


def clean(pool, want, rng):
    pool = [t for t in pool if t and len(t) > 80]
    rng.shuffle(pool)
    return pool[:want]


def load_232k(rng, want):
    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    t, l = ds["text"], ds["label"]
    return (clean([x for x, y in zip(t, l) if y == "suicide"], want, rng),
            clean([x for x, y in zip(t, l) if y == "non-suicide"], want, rng))


def load_external(name, rng, want):
    """Same loaders and label conventions as sc2_generalization.py."""
    if name == "SDCNL":
        ds = load_from_disk("data/datasets/08_SDCNL/hf_cache")["train"]
        t, c = ds["text"], ds["class"]
        return (clean([x for x, y in zip(t, c) if y == "suicide"], want, rng),
                clean([x for x, y in zip(t, c) if y == "non-suicide"], want, rng))
    if name == "SWMH":
        d = load_dataset_by_category(dataset_name="swmh")
        return (clean([get_text(x) for x in d.get("Death-Me", [])], want, rng),
                clean([get_text(x) for x in d.get("Life-Not-Me", [])], want, rng))
    raise ValueError(name)


def main():
    args = parse_args()
    rng = np.random.default_rng(get_seed())

    layer = args.layer
    if layer is None:
        for cand in ("results/sc2_binary_fp16/gate.json", "results/sc2_binary/gate.json"):
            p = PROJECT_ROOT / cand
            if p.exists():
                layer = json.loads(p.read_text())["best_probe_layer"]
                print(f"layer {layer} (from {cand})")
                break
    if layer is None:
        raise SystemExit("no gate.json found; pass --layer")

    need = args.n_fit + args.n_cal + args.n_eval
    sui, non = load_232k(rng, need)
    if min(len(sui), len(non)) < need:
        raise SystemExit(f"need {need}/class, have {len(sui)}/{len(non)}")

    a, b = args.n_fit, args.n_fit + args.n_cal
    split = {
        "fit": (sui[:a],  non[:a]),
        "cal": (sui[a:b], non[a:b]),
        "eval": (sui[b:need], non[b:need]),
    }

    loader = ModelLoader(device=device)
    model, tok = loader.load_llama(auto_quantize=False)
    model.eval()
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]

    def chat_ids(text):
        tmpl = PROMPT_TEMPLATES["standard"]
        prefix, suffix = tmpl.split("{text}")
        body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
        content = prefix + tok.decode(body) + suffix
        s = tok.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    @torch.no_grad()
    def score(text):
        """One forward -> (behavioral logit diff, hidden state at `layer`)."""
        out = model(chat_ids(text), output_hidden_states=True)
        lg = out.logits[0, -1, :]
        ld = (lg[yes_id] - lg[no_id]).item()
        h = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
        return ld, h

    def collect(texts, tag):
        lds, hs = [], []
        for i, t in enumerate(texts):
            ld, h = score(t)
            lds.append(ld); hs.append(h)
            if (i + 1) % 100 == 0:
                print(f"    {tag} {i+1}/{len(texts)}", flush=True)
        return np.array(lds), np.stack(hs)

    acts = {}
    for name, (s, n) in split.items():
        print(f"\nscoring {name}: {len(s)}+{len(n)}", flush=True)
        s_ld, s_h = collect(s, f"{name}/sui")
        n_ld, n_h = collect(n, f"{name}/non")
        acts[name] = {
            "ld": np.concatenate([s_ld, n_ld]),
            "h":  np.concatenate([s_h, n_h]),
            "y":  np.concatenate([np.ones(len(s)), np.zeros(len(n))]),
        }

    # --- fit both readouts on the fit split only ---
    fit = acts["fit"]
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000))
    probe.fit(fit["h"], fit["y"])

    d = fit["h"][fit["y"] == 1].mean(0) - fit["h"][fit["y"] == 0].mean(0)
    d = d / (np.linalg.norm(d) + 1e-8)

    def readouts(d_):
        return (probe.predict_proba(d_["h"])[:, 1], d_["h"] @ d, d_["ld"])

    # --- one threshold per readout, calibrated on the disjoint cal split ---
    def best_threshold(scores, y):
        order = np.unique(scores)
        cands = (order[:-1] + order[1:]) / 2 if len(order) > 1 else order
        accs = [(((scores > t).astype(int) == y).mean(), t) for t in cands]
        return max(accs)[1]

    cal = acts["cal"]
    c_probe, c_proj, c_ld = readouts(cal)
    thr = {
        "probe": best_threshold(c_probe, cal["y"]),
        "proj":  best_threshold(c_proj, cal["y"]),
        "behav": best_threshold(c_ld, cal["y"]),
    }
    print(f"\nthresholds (calibrated on {args.n_cal}/class): "
          + "  ".join(f"{k}={v:.3f}" for k, v in thr.items()))

    # --- head to head on the eval split ---
    ev = acts["eval"]
    e_probe, e_proj, e_ld = readouts(ev)
    y = ev["y"]

    def summarize(s, t):
        pred = (s > t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        return {"auc": float(roc_auc_score(y, s)),
                "acc": float((pred == y).mean()),
                "recall": float(tp / max(tp + fn, 1)),
                "precision": float(tp / max(tp + fp, 1)),
                "n_false_neg": int(fn)}

    head = {"behavioral": summarize(e_ld, thr["behav"]),
            "probe":      summarize(e_probe, thr["probe"]),
            "projection": summarize(e_proj, thr["proj"])}

    # --- the question: what do the internals say on the model's own errors? ---
    behav_pred = (e_ld > thr["behav"]).astype(int)
    fn_mask = (behav_pred == 0) & (y == 1)          # missed suicidal posts
    fp_mask = (behav_pred == 1) & (y == 0)          # false alarms
    err_mask = behav_pred != y

    recovery = {}
    for name, s, t in (("probe", e_probe, thr["probe"]),
                       ("projection", e_proj, thr["proj"])):
        rec = float((s[fn_mask] > t).mean()) if fn_mask.sum() else None
        rej = float((s[fp_mask] <= t).mean()) if fp_mask.sum() else None
        # AUC restricted to the cases the model got wrong: can the internal
        # readout separate them when the behavioral readout could not?
        err_auc = (float(roc_auc_score(y[err_mask], s[err_mask]))
                   if err_mask.sum() and len(np.unique(y[err_mask])) > 1 else None)
        recovery[name] = {"fn_recovered": rec, "fp_rejected": rej,
                          "auc_on_model_errors": err_auc}

    # --- matched-operating-point control -----------------------------------
    # The recovery figures above compare each readout at its OWN calibrated
    # threshold, and the probe's happens to sit at higher recall. A readout that
    # simply says "yes" more often would recover false negatives for free, so the
    # comparison has to be re-run with the operating point held fixed. Two ways
    # to hold it: match the behavioral readout's precision, or match its alarm
    # rate (the number of posts flagged). Recovery that survives both is not an
    # artifact of where the threshold happened to land.
    def op_point(scores, thr_):
        pred = (scores > thr_).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        return {"precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "n_flagged": int(pred.sum()),
                "fn_recovered": float((scores[fn_mask] > thr_).mean()) if fn_mask.sum() else None,
                "fp_rejected": float((scores[fp_mask] <= thr_).mean()) if fp_mask.sum() else None,
                "threshold": float(thr_)}

    behav_op = op_point(e_ld, thr["behav"])
    matched = {}
    for name, s in (("probe", e_probe), ("projection", e_proj)):
        grid = np.unique(s)
        grid = (grid[:-1] + grid[1:]) / 2 if len(grid) > 1 else grid
        pts = [op_point(s, t) for t in grid]

        # (a) strictest threshold whose precision is at least the behavioral one
        feasible = [q for q in pts if q["precision"] >= behav_op["precision"]]
        by_prec = max(feasible, key=lambda q: q["recall"]) if feasible else None

        # (b) threshold flagging the same number of posts as the model does
        by_rate = min(pts, key=lambda q: abs(q["n_flagged"] - behav_op["n_flagged"]))

        matched[name] = {"matched_precision": by_prec, "matched_alarm_rate": by_rate}

    print("\n  matched-operating-point control (behavioral reference: "
          f"precision {behav_op['precision']:.3f}, recall {behav_op['recall']:.3f}, "
          f"{behav_op['n_flagged']} flagged)")
    for name, m in matched.items():
        for how, q in m.items():
            if q is None:
                print(f"    {name:11s} {how:18s} unreachable at this precision")
                continue
            fr = "n/a" if q["fn_recovered"] is None else f"{q['fn_recovered']:.3f}"
            print(f"    {name:11s} {how:18s} precision {q['precision']:.3f}  "
                  f"recall {q['recall']:.3f}  flagged {q['n_flagged']:4d}  "
                  f"FN recovered {fr}")

    print("\n" + "=" * 62)
    print(f"INTERNAL MONITOR vs BEHAVIORAL READOUT  (layer {layer}, "
          f"n={args.n_eval}/class held out)")
    print("=" * 62)
    for k, v in head.items():
        print(f"  {k:12s} AUC {v['auc']:.3f}  acc {v['acc']:.3f}  "
              f"recall {v['recall']:.3f}  FN {v['n_false_neg']}")
    print(f"\n  model false negatives: {int(fn_mask.sum())}  "
          f"false positives: {int(fp_mask.sum())}")
    for k, v in recovery.items():
        fn_r = "n/a" if v["fn_recovered"] is None else f"{v['fn_recovered']:.3f}"
        ea = "n/a" if v["auc_on_model_errors"] is None else f"{v['auc_on_model_errors']:.3f}"
        print(f"  {k:12s} recovers {fn_r} of them;  AUC on model errors {ea}")

    # --- transfer: same direction and probe, untouched, on external corpora ---
    external = {}
    for name in ("SDCNL", "SWMH"):
        try:
            s, n = load_external(name, rng, args.n_ext)
        except Exception as e:
            print(f"\n[skip] {name}: {type(e).__name__}: {e}")
            continue
        if min(len(s), len(n)) < 20:
            print(f"\n[skip] {name}: too few examples")
            continue
        print(f"\nscoring {name}: {len(s)}+{len(n)}", flush=True)
        s_ld, s_h = collect(s, f"{name}/sui")
        n_ld, n_h = collect(n, f"{name}/non")
        dx = {"ld": np.concatenate([s_ld, n_ld]),
              "h": np.concatenate([s_h, n_h]),
              "y": np.concatenate([np.ones(len(s)), np.zeros(len(n))])}
        x_probe, x_proj, x_ld = readouts(dx)
        yx = dx["y"]
        bp = (x_ld > thr["behav"]).astype(int)
        fnx = (bp == 0) & (yx == 1)
        external[name] = {
            "n_per_class": int(min(len(s), len(n))),
            "behavioral_auc": float(roc_auc_score(yx, x_ld)),
            "probe_auc": float(roc_auc_score(yx, x_probe)),
            "projection_auc": float(roc_auc_score(yx, x_proj)),
            "n_false_neg": int(fnx.sum()),
            "fn_recovered_probe": (float((x_probe[fnx] > thr["probe"]).mean())
                                   if fnx.sum() else None),
        }
        e = external[name]
        print(f"  {name}: behav {e['behavioral_auc']:.3f}  probe {e['probe_auc']:.3f}"
              f"  FN {e['n_false_neg']}  recovered "
              + ("n/a" if e["fn_recovered_probe"] is None
                 else f"{e['fn_recovered_probe']:.3f}"))

    out = PROJECT_ROOT / args.out
    ensure_dir(out.parent)
    out.write_text(json.dumps({
        "layer": layer,
        "n_fit": args.n_fit, "n_cal": args.n_cal, "n_eval": args.n_eval,
        "thresholds": {k: float(v) for k, v in thr.items()},
        "head_to_head": head,
        "n_model_false_neg": int(fn_mask.sum()),
        "n_model_false_pos": int(fp_mask.sum()),
        "error_recovery": recovery,
        "behavioral_operating_point": behav_op,
        "matched_operating_point": matched,
        # raw held-out scores, so any other operating point can be checked
        # post hoc without re-running the model
        "eval_scores": {"y": y.astype(int).tolist(),
                        "behavioral_ld": e_ld.tolist(),
                        "probe": e_probe.tolist(),
                        "projection": e_proj.tolist()},
        "external": external,
    }, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
