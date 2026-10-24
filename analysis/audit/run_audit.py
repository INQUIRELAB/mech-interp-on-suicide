#!/usr/bin/env python
"""Concept-direction audit driver (Paper 1, Phase 2).

Runs the evidence ladder for ONE (concept, model, seed) and writes an evidence profile:

  GATE        behavioral AUC vs a TF-IDF baseline on held-out data (paired bootstrap)
  DECODABLE   per-layer probe (fit split -> eval split) vs TF-IDF and a shuffled-label probe
  NECESSARY   rank-1 ablation at every layer (zero- and mean-projection)
              vs 4 null families x M draws, IUT gate over {AUC drop, LD-gap drop}
              + kl_matched (zero mode): each null ablation is scaled down (h - beta (h.v) v,
              beta <= 1) until its off-target damage (reference-text KL) equals the target's;
              IUT gate on the LD-gap drop at equal damage. The null directions are first made
              orthogonal to the target: ablation is sign-invariant, so a sign-randomized or
              label-shuffled DoM (|cos| ~0.6-0.7 with the target) is a noisy +-target, not a null.
  SUFFICIENT  steering on negative-class texts at a matched off-target KL budget
              vs 4 null families x M draws, IUT gate over {selectivity, selectivity tail,
              polarity}; readouts: target Yes/No, 3 off-target Yes/No, negated question,
              4-way multiple choice
  (RANK       Phase 3: analysis/audit/rank_ident.py)

Changes from analysis/sanity/gcl/run_gcl.py (kept as-is to reproduce the submitted paper):
fp16/bf16 instead of 4-bit, disjoint fit/eval splits at n in the thousands, four null
families instead of one random direction, a KL-matched operating point instead of c=1,
polarity and multiple-choice readouts, realized edit norms, and an IUT verdict per rung.

Usage:
  python run_audit.py --concept suicide --model meta-llama/Llama-3.1-8B-Instruct
  python run_audit.py --concept refusal --model Qwen/Qwen2.5-7B-Instruct --seed 1
  python run_audit.py --concept suicide --model Qwen/Qwen2.5-1.5B-Instruct --smoke
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import Backend
from concepts import OFFTARGET_MC, OFFTARGET_QUESTIONS, PROJECT_ROOT, get_concept
from interventions import (NULL_FAMILIES, NullSampler, ablation_hooks, calibrate_alpha,
                           diff_of_means, erase_fn, mean_kl, steering_hooks)
from stats import auc, boot_auc_ci, boot_auc_diff_ci, iut_gate, test_vs_null


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="float16", help="float16 | bfloat16 (use bfloat16 for Gemma)")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--n-fit", type=int, default=2000, help="per class, for directions/probes")
    p.add_argument("--n-eval", type=int, default=1000, help="per class, held out")
    p.add_argument("--n-nec", type=int, default=300, help="per class, eval subset for necessity")
    p.add_argument("--n-steer", type=int, default=300, help="negative-class eval texts steered")
    p.add_argument("--n-ref", type=int, default=48, help="neutral prompts for KL calibration")
    p.add_argument("--n-null", type=int, default=10, help="draws per null family")
    p.add_argument("--no-kl-matched-nec", action="store_true", help="skip the KL-matched necessity test")
    p.add_argument("--kappa", type=float, default=0.1, help="matched off-target KL budget")
    p.add_argument("--alpha-level", type=float, default=0.01, help="IUT alpha (Holm-corrected)")
    p.add_argument("--site", type=int, default=None, help="hidden index for steering; default: "
                   "best probe layer within 25-75%% depth")
    p.add_argument("--half-window", type=int, default=4)
    p.add_argument("--nec-modes", default="zero,mean")
    p.add_argument("--families", default=",".join(NULL_FAMILIES))
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-ladder", action="store_true", help="run causal rungs even on NO-GO")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "audit"))
    p.add_argument("--smoke", action="store_true", help="tiny sizes for a pipeline check")
    a = p.parse_args()
    if a.smoke:
        a.n_fit, a.n_eval, a.n_nec, a.n_steer, a.n_ref, a.n_null = 48, 32, 24, 16, 8, 3
    return a


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    a = parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    concept = get_concept(a.concept)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", a.model.split("/")[-1]).lower()
    families = [f for f in a.families.split(",") if f]
    log(f"AUDIT concept={concept.name} model={a.model} seed={a.seed}")

    # ------------------------------------------------------------ data
    pos, neg = concept.loader(a.n_fit + a.n_eval, a.seed)
    n = min(len(pos), len(neg))
    n_fit, n_eval = a.n_fit, a.n_eval
    if n < n_fit + n_eval:                       # small datasets: keep a 60/40 split
        n_fit = int(round(0.6 * n)); n_eval = n - n_fit
        log(f"only {n}/class available -> n_fit={n_fit}, n_eval={n_eval}")
    fit_p, fit_n = pos[:n_fit], neg[:n_fit]
    ev_p, ev_n = pos[n_fit:n_fit + n_eval], neg[n_fit:n_fit + n_eval]

    report = {"concept": concept.name, "display_name": concept.display_name,
              "source": concept.source, "model": a.model, "model_tag": tag, "seed": a.seed,
              "dtype": "int8" if a.load_in_8bit else a.dtype,
              "n_fit": n_fit, "n_eval": n_eval, "config": vars(a)}

    # ------------------------------------------------------------ lexical baseline
    tf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                       LogisticRegression(C=1.0, max_iter=2000))
    tf.fit(fit_p + fit_n, [1] * len(fit_p) + [0] * len(fit_n))
    tf_p, tf_n = tf.predict_proba(ev_p)[:, 1], tf.predict_proba(ev_n)[:, 1]
    tfidf_auc = auc(tf_p, tf_n)
    log(f"TF-IDF baseline AUC (held out) = {tfidf_auc:.3f}")

    # ------------------------------------------------------------ model + activations
    be = Backend(a.model, dtype=a.dtype, load_in_8bit=a.load_in_8bit)
    be.truncate_side, be.body_budget = concept.truncate, concept.body_tokens
    nL = be.n_layers
    yes_id, no_id = be.token_id(concept.pos_answer), be.token_id(concept.neg_answer)
    letter_ids = [be.token_id(" " + c) for c in "ABCD"]
    P = lambda ts, q=concept.question, pre=concept.prefix: [be.polar_prompt(t, pre, q) for t in ts]
    log(f"{nL} layers, chat={be.use_chat}; collecting activations")
    _, Hfp = be.collect(P(fit_p), yes_id, no_id, a.batch)
    _, Hfn = be.collect(P(fit_n), yes_id, no_id, a.batch)
    ld_ep, Hep = be.collect(P(ev_p), yes_id, no_id, a.batch)
    ld_en, Hen = be.collect(P(ev_n), yes_id, no_id, a.batch)

    # ------------------------------------------------------------ GATE
    behav = auc(ld_ep, ld_en)
    gate_go = bool(behav > tfidf_auc and behav > 0.6)
    report["gate"] = {
        "behavioral_auc": behav, "behavioral_ci95": boot_auc_ci(ld_ep, ld_en, seed=a.seed),
        "tfidf_auc": tfidf_auc, "tfidf_ci95": boot_auc_ci(tf_p, tf_n, seed=a.seed),
        "behavior_minus_tfidf_ci95": boot_auc_diff_ci(ld_ep, ld_en, tf_p, tf_n, seed=a.seed),
        "decision": "GO" if gate_go else "NO-GO"}
    log(f"GATE behavioral {behav:.3f} vs TF-IDF {tfidf_auc:.3f} -> {report['gate']['decision']}")

    # ------------------------------------------------------------ DECODABLE
    yfit = np.r_[np.ones(len(fit_p)), np.zeros(len(fit_n))]
    per_layer, best = [], (None, -1.0, None)
    for j in range(nL + 1):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        clf.fit(np.vstack([Hfp[:, j], Hfn[:, j]]).astype(np.float32), yfit)
        sp = clf.predict_proba(Hep[:, j].astype(np.float32))[:, 1]
        sn = clf.predict_proba(Hen[:, j].astype(np.float32))[:, 1]
        s = auc(sp, sn)
        per_layer.append({"hidden_index": j, "auc": s})
        if s > best[1]:
            best = (j, s, (sp, sn))
    lo, hi = max(1, int(0.25 * nL)), min(nL, int(round(0.75 * nL)))
    site = a.site if a.site is not None else max(range(lo, hi + 1), key=lambda j: per_layer[j]["auc"])
    shuf = []
    Xb = np.vstack([Hfp[:, best[0]], Hfn[:, best[0]]]).astype(np.float32)
    for _ in range(max(a.n_null, 10)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        clf.fit(Xb, rng.permutation(yfit))
        shuf.append(auc(clf.predict_proba(Hep[:, best[0]].astype(np.float32))[:, 1],
                        clf.predict_proba(Hen[:, best[0]].astype(np.float32))[:, 1]))
    report["decodable"] = {
        "probe_per_layer": per_layer, "best_hidden_index": best[0], "best_auc": best[1],
        "best_ci95": boot_auc_ci(*best[2], seed=a.seed), "site": site,
        "site_auc": per_layer[site]["auc"],
        "probe_minus_tfidf_ci95": boot_auc_diff_ci(best[2][0], best[2][1], tf_p, tf_n, seed=a.seed),
        "shuffled_label_null": test_vs_null(best[1], shuf)}
    report["decodable"]["pass"] = bool(report["decodable"]["probe_minus_tfidf_ci95"][0] > 0
                                       and report["decodable"]["shuffled_label_null"]["p"] < a.alpha_level)
    log(f"DECODABLE best L{best[0]} AUC {best[1]:.3f}; site L{site} AUC {per_layer[site]['auc']:.3f}")

    if not gate_go and not a.force_ladder:
        report["note"] = "NO-GO: causal rungs not run (use --force-ladder to run them anyway)."
        return _save(report, a, tag, t0)

    # ------------------------------------------------------------ directions + nulls
    dirs, gap = diff_of_means(Hfp, Hfn)                                   # [nL+1, d], [nL+1]
    mu = (np.vstack([Hfp.mean(0, dtype=np.float64)[None], Hfn.mean(0, dtype=np.float64)[None]])
          .mean(0).astype(np.float32))                                     # pooled mean [nL+1, d]
    all_j = list(range(1, nL + 1))
    window = [j for j in range(site - a.half_window, site + a.half_window + 1) if 1 <= j <= nL]
    sampler = NullSampler(Hfp, Hfn, all_j)
    null_draws = {f: [sampler.sample(f, rng) for _ in range(a.n_null)] for f in families}
    log(f"null draws ready ({len(families)} families x {a.n_null})")
    np.savez_compressed(_out_dir(a) / f"{tag}__seed{a.seed}_dirs.npz", dirs=dirs, gap=gap, mu=mu,
                        site=site, window=np.array(window))

    # ------------------------------------------------------------ NECESSARY
    nec_p, nec_n = ev_p[: a.n_nec], ev_n[: a.n_nec]
    Pnp, Pnn = P(nec_p), P(nec_n)
    ref_texts = ev_n[-a.n_ref:]
    ref_prompts = [be._wrap_chat(f"Continue the following text.\n\n{be._truncate(t)}") for t in ref_texts]
    base_lp = torch.log_softmax(be.last_logits(ref_prompts, a.batch), -1)

    def scaled_ablation(d, beta):  # zero-projection with strength beta (beta = 1: full ablation)
        return {j: erase_fn(be.to_dev(beta * np.asarray(v).reshape(-1, 1)), be.to_dev(np.asarray(v).reshape(-1, 1)))
                for j, v in d.items()}

    def nec_stats(hooks):
        with be.hooked(hooks):
            lp, ln = be.logit_diff(Pnp, yes_id, no_id, a.batch), be.logit_diff(Pnn, yes_id, no_id, a.batch)
        return auc(lp, ln), float(lp.mean() - ln.mean())

    base_auc, base_gap = nec_stats({})
    report["necessary"] = {"baseline_auc": base_auc, "baseline_ld_gap": base_gap, "modes": {}}
    for mode in [m for m in a.nec_modes.split(",") if m]:
        mu_j = {j: mu[j] for j in all_j}
        tgt_hooks = ablation_hooks(be, {j: dirs[j] for j in all_j}, mode, mu_j)
        tgt_auc, tgt_gap = nec_stats(tgt_hooks)
        tgt_kl = mean_kl(be, ref_prompts, tgt_hooks, base_lp, a.batch)
        tests, null_summary = {}, {}
        for f in families:
            vals, kls = [], []
            for d in null_draws[f]:
                hk = ablation_hooks(be, d, mode, mu_j)
                vals.append(nec_stats(hk))
                kls.append(mean_kl(be, ref_prompts, hk, base_lp, a.batch))
            drops_auc = [base_auc - v[0] for v in vals]
            drops_gap = [base_gap - v[1] for v in vals]
            tests[("auc_drop", f)] = test_vs_null(base_auc - tgt_auc, drops_auc)
            tests[("gap_drop", f)] = test_vs_null(base_gap - tgt_gap, drops_gap)
            null_summary[f] = {"auc_mean": float(np.mean([v[0] for v in vals])),
                               "auc_min": float(np.min([v[0] for v in vals])),
                               "ld_gap_mean": float(np.mean([v[1] for v in vals])),
                               "ld_gap_min": float(np.min([v[1] for v in vals])),
                               "kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls))}
        edit = [float(np.abs(np.r_[Hep[:, j], Hen[:, j]].astype(np.float32)
                             @ dirs[j] - (mu[j] @ dirs[j] if mode == "mean" else 0)).mean())
                for j in all_j]
        verdict = iut_gate(tests, a.alpha_level)
        report["necessary"]["modes"][mode] = {
            "ablated_auc": tgt_auc, "ablated_ld_gap": tgt_gap, "ablated_kl": tgt_kl, "null_auc": null_summary,
            "realized_edit_norm_mean_per_layer": edit, "gate": verdict}
        log(f"NECESSARY[{mode}] {base_auc:.3f} -> {tgt_auc:.3f} (gap {base_gap:.2f} -> {tgt_gap:.2f}, KL {tgt_kl:.3f}); "
            f"nulls {[round(v['auc_mean'], 3) for v in null_summary.values()]} "
            f"KL {[round(v['kl_mean'], 3) for v in null_summary.values()]} -> "
            f"{'PASS' if verdict['pass'] else 'fail'}")

        if mode != "zero" or a.no_kl_matched_nec:
            continue
        # KL-matched: shrink each null ablation until its reference KL equals the target's
        km_tests, km_summary = {}, {}
        for f in families:
            rows = []
            for d_raw in null_draws[f]:
                d, cos_raw = {}, []
                for j, v in d_raw.items():
                    t_ = dirs[j] / (np.linalg.norm(dirs[j]) + 1e-12)
                    cos_raw.append(abs(float(v @ t_)))
                    o = v - (v @ t_) * t_
                    d[j] = (o / (np.linalg.norm(o) + 1e-12)).astype(np.float32)
                lo, hi = 0.0, 1.0
                if mean_kl(be, ref_prompts, scaled_ablation(d, 1.0), base_lp, a.batch) > tgt_kl:
                    for _ in range(7):  # bisection on beta; KL rises with beta for a rank-1 edit
                        mid = (lo + hi) / 2
                        if mean_kl(be, ref_prompts, scaled_ablation(d, mid), base_lp, a.batch) > tgt_kl:
                            hi = mid
                        else:
                            lo = mid
                    beta = (lo + hi) / 2
                else:
                    beta = 1.0  # the null does less damage even at full strength
                hk = scaled_ablation(d, beta)
                au, gp = nec_stats(hk)
                rows.append({"beta": beta, "kl": mean_kl(be, ref_prompts, hk, base_lp, a.batch),
                             "auc": au, "ld_gap": gp, "raw_abs_cos_with_target": float(np.mean(cos_raw))})
            km_tests[("gap_drop", f)] = test_vs_null(base_gap - tgt_gap, [base_gap - r["ld_gap"] for r in rows])
            km_summary[f] = {"beta_mean": float(np.mean([r["beta"] for r in rows])),
                             "kl_mean": float(np.mean([r["kl"] for r in rows])),
                             "ld_gap_mean": float(np.mean([r["ld_gap"] for r in rows])),
                             "auc_mean": float(np.mean([r["auc"] for r in rows])),
                             "raw_abs_cos_mean": float(np.mean([r["raw_abs_cos_with_target"] for r in rows])),
                             "draws": rows}
        km_verdict = iut_gate(km_tests, a.alpha_level)
        report["necessary"]["kl_matched"] = {"nulls_orthogonalized_to_target": True,
                                             "target_kl": tgt_kl, "target_ld_gap": tgt_gap,
                                             "nulls": km_summary, "gate": km_verdict}
        log(f"NECESSARY[kl-matched @ KL {tgt_kl:.3f}] gap drop {base_gap - tgt_gap:.2f} vs nulls "
            f"{[round(base_gap - v['ld_gap_mean'], 2) for v in km_summary.values()]} "
            f"(beta {[round(v['beta_mean'], 2) for v in km_summary.values()]}) -> "
            f"{'PASS' if km_verdict['pass'] else 'fail'}")

    # ------------------------------------------------------------ SUFFICIENT
    items = ev_n[: a.n_steer]
    offq = [(name, pre, q) for name, pre, q in OFFTARGET_QUESTIONS if name not in concept.name]
    mc_rng = np.random.default_rng(a.seed + 7)
    mc_opts, mc_tgt = [], []
    for _ in items:
        order = mc_rng.permutation(4)
        labels = [concept.mc_label] + OFFTARGET_MC
        mc_opts.append([labels[k] for k in order]); mc_tgt.append(int(np.where(order == 0)[0][0]))
    mc_tgt = np.array(mc_tgt)
    prompts = {"on": P(items), "neg": P(items, concept.neg_question),
               **{f"off_{name}": P(items, q, pre) for name, pre, q in offq},
               "mc": [be.mc_prompt(t, o) for t, o in zip(items, mc_opts)]}

    def readouts(hooks):
        out = {}
        with be.hooked(hooks):
            for k, pr in prompts.items():
                if k == "mc":
                    lp = be.letter_logprobs(pr, letter_ids, a.batch)
                    tgt = lp[np.arange(len(lp)), mc_tgt]
                    off = (lp.sum(1) - tgt) / 3.0
                    out["mc_target"], out["mc_off"] = tgt, off
                else:
                    out[k] = be.logit_diff(pr, yes_id, no_id, a.batch)
        return out

    base = readouts({})
    thr = float((np.median(ld_ep) + np.median(ld_en)) / 2)
    off_keys = [k for k in base if k.startswith("off_")]

    def summarize(r, alpha, kl, dset, edit):
        d_on = r["on"] - base["on"]
        d_off = np.mean([r[k] - base[k] for k in off_keys], 0)
        d_neg = r["neg"] - base["neg"]
        s = d_on - d_off
        mc_s = (r["mc_target"] - base["mc_target"]) - (r["mc_off"] - base["mc_off"])
        cos = float(np.mean([abs(float(dset[j] @ dirs[j])) for j in window]))
        return {"alpha": float(alpha), "kl": float(kl), "on_shift": float(d_on.mean()),
                "off_shift": float(d_off.mean()), "neg_shift": float(d_neg.mean()),
                "selectivity": float(s.mean()), "selectivity_tail_p05": float(np.percentile(s, 5)),
                "polarity": float((d_on - d_neg).mean()), "mc_selectivity": float(mc_s.mean()),
                "flip_rate": float((r["on"] > thr).mean()), "abs_cos_with_target": cos,
                "realized_edit_norm_per_layer": edit,
                "off_shift_by_question": {k: float((r[k] - base[k]).mean()) for k in off_keys}}

    def run_direction(dset, alpha0):
        make = lambda al: steering_hooks(be, dset, {j: gap[j] for j in window}, al)
        alpha, kl = calibrate_alpha(be, ref_prompts, make, base_lp, a.kappa, alpha0, batch=a.batch)
        edit = [float(abs(alpha * gap[j])) for j in window]
        return summarize(readouts(make(alpha)), alpha, kl, dset, edit), alpha

    tgt_dirs = {j: dirs[j] for j in window}
    tgt, tgt_alpha = run_direction(tgt_dirs, 1.0)
    c1 = summarize(readouts(steering_hooks(be, tgt_dirs, {j: gap[j] for j in window}, 1.0)),
                   1.0, float("nan"), tgt_dirs, [float(abs(gap[j])) for j in window])
    log(f"SUFFICIENT target @KL={tgt['kl']:.3f}: on {tgt['on_shift']:+.2f} off {tgt['off_shift']:+.2f} "
        f"neg {tgt['neg_shift']:+.2f} sel {tgt['selectivity']:+.2f} mc {tgt['mc_selectivity']:+.2f}")
    null_rows, tests, tests_mc = {}, {}, {}
    for f in families:
        rows = []
        for d in null_draws[f]:
            r, _ = run_direction({j: d[j] for j in window}, tgt_alpha)
            rows.append(r)
        null_rows[f] = rows
        for stat in ("selectivity", "selectivity_tail_p05", "polarity"):
            tests[(stat, f)] = test_vs_null(tgt[stat], [r[stat] for r in rows])
        tests_mc[("mc_selectivity", f)] = test_vs_null(tgt["mc_selectivity"], [r["mc_selectivity"] for r in rows])
        log(f"  null[{f}] sel {np.mean([r['selectivity'] for r in rows]):+.2f} "
            f"on {np.mean([r['on_shift'] for r in rows]):+.2f} "
            f"|cos| {np.mean([r['abs_cos_with_target'] for r in rows]):.3f}")
    report["sufficient"] = {
        "site": site, "window": window, "kappa": a.kappa, "flip_threshold": thr,
        "baseline": {k: float(v.mean()) for k, v in base.items()},
        "target": tgt, "target_at_c1": c1, "nulls": null_rows,
        "gate_polar": iut_gate(tests, a.alpha_level), "gate_mc": iut_gate(tests_mc, a.alpha_level)}
    log(f"SUFFICIENT polar -> {'PASS' if report['sufficient']['gate_polar']['pass'] else 'fail'}; "
        f"mc -> {'PASS' if report['sufficient']['gate_mc']['pass'] else 'fail'}")
    return _save(report, a, tag, t0)


def _out_dir(a):
    d = Path(a.out_dir) / a.concept
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(report, a, tag, t0):
    report["runtime_min"] = round((time.time() - t0) / 60, 2)
    path = _out_dir(a) / f"{tag}__seed{a.seed}.json"
    path.write_text(json.dumps(report, indent=1, default=float))
    log(f"saved {path} ({report['runtime_min']} min)")


if __name__ == "__main__":
    main()
