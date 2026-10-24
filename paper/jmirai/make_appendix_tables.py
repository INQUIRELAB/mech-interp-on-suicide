"""Write the LaTeX tables of Multimedia Appendix 1 from the result files in ./data (no hand-typed numbers).

Run:  python make_appendix_tables.py      (writes tables/*.tex, included by jmirai_appendix1.tex)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D, OUT = HERE / "data", HERE / "tables"
NAMES = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B", "mistral": "Mistral-7B", "gemma": "Gemma-2-9B"}


def load(p): return json.load(open(p, encoding="utf8"))
def key(n): return next(k for k in NAMES if k in n.lower())
def f3(x): return "--" if x is None else f"{x:.3f}"
def num(x): return f"{x:.3f}".replace("-", "$-$")
def ci(c): return "--" if c is None else f"{num(c[0])} to {num(c[1])}"


def table(name, caption, header, rows, colspec, size="footnotesize"):
    OUT.mkdir(exist_ok=True)
    minus = lambda s: "$-$" + s[1:] if len(s) > 1 and s[0] == "-" and s[1].isdigit() else s
    body = "\n".join(" & ".join(minus(c) for c in r) + r" \\" for r in rows)
    sep = "\\setlength{\\tabcolsep}{3pt}" if size == "scriptsize" else ""
    tex = (f"\\begin{{table}}[htbp]\\centering\\{size}{sep}\n\\caption{{{caption}}}\n\\label{{tab:{name}}}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n"
           f"\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf8")


def prompts():
    rows = []
    labels = {"base": "Base", "neg_flip": "Negated question", "clinical": "Clinical definition", "contrast": "Contrast",
              "escalate": "Crisis triage", "fewshot": "Four-shot", "rating": "1--5 rating"}
    trn = load(D / "tfidf_truncation/tfidf_truncation.json")["models"]
    for f in sorted((D / "paper2/prompt_invariance").glob("*.json")):
        a = load(f); t = trn[f.name.split("__")[1]]["variants"]
        for v, lab in labels.items():
            r = a["variants"][v]
            assert abs(t[v]["auc"] - r["auc"]) < 1e-9   # same per-post scores in both files
            yes = "--" if r.get("frac_yes_neg") is None else f"{r['frac_yes_neg']:.1%}".replace("%", r"\%")
            rows.append([NAMES[key(a["model"])], lab, f3(r["auc"]), ci(r["ci95"]), ci(t[v]["minus_tfidf_trunc_ci95"]),
                         ci(r["minus_tfidf_ci95"]), f3(t[v]["stack_trunc_cv_auc"]), yes])
    table("prompts", "Register-matched screening, every model and prompt. Differences are paired bootstrap 95\\% CIs "
          "against TF-IDF on the same truncated text and on full posts; the stack uses TF-IDF on the same text.",
          ["Model", "Prompt", "AUC", "95\\% CI", "$-$ TF-IDF, same text", "$-$ TF-IDF, full post", "Stack AUC",
           "``Yes'' on depression"], rows, "llcccccc", size="scriptsize")


def tfidf_trunc():
    rows = []
    trn = load(D / "tfidf_truncation/tfidf_truncation.json")
    pct = lambda x: f"{x:.1%}".replace("%", r"\%")
    for tag, r in trn["models"].items():
        t = r["tfidf_trunc"]; fr = r["frac_truncated"]
        rows.append([NAMES[key(tag)], pct(fr["SuicideWatch"]), pct(fr["depression"]), f3(t["auc"]), ci(t["ci95"]),
                     f3(t["sens90"]), ci(t["sens90_ci95"]), f3(t["calibrated_n100"]["sensitivity"])])
    # full-post row from the saved scores of the original run (same bootstrap as the Figure 2 script)
    pi = sorted((D / "paper2/prompt_invariance").glob("*.json"))[0]
    a = load(pi); z = np.load(pi.with_suffix(".npz")); y = z["y"].astype(int); s = z["tfidf"]
    sens = lambda i: float((s[i][y[i] == 1] > np.quantile(s[i][y[i] == 0], 0.9)).mean())
    rng = np.random.default_rng(1); P_, N_ = np.where(y == 1)[0], np.where(y == 0)[0]
    boot = [sens(np.r_[rng.choice(P_, len(P_)), rng.choice(N_, len(N_))]) for _ in range(1000)]
    cal = [m["tfidf"]["cal100"]["sensitivity"][0] for m in load(D / "threshold_calibration/threshold_calibration.json")["models"].values()]
    rows.append(["Full post", "--", "--", f3(a["tfidf_auc"]), ci(a["tfidf_ci95"]), f3(sens(np.arange(len(y)))),
                 ci(np.percentile(boot, [2.5, 97.5])), f"{min(cal):.3f}--{max(cal):.3f}"])
    table("tfidftrunc", "TF-IDF given the same text as each model (register-matched set). Truncated: share of "
          "evaluation posts longer than 176 tokens. Calibrated: mean sensitivity with the threshold set on 100 posts "
          "per class (500 splits). The full-post row repeats the original analysis (saved scores).",
          ["Tokenizer", "Truncated, r/SW", "Truncated, r/dep", "AUC", "95\\% CI", "Sens. at 90\\% spec.", "95\\% CI",
           "Calibrated sens."], rows, "lccccccc", size="scriptsize")


def umd():
    rows = []
    order = [("proj_site", "Direction projection"), ("behav", "Model answer"), ("probe_cv", "Probe (CV)"),
             ("tfidf_cv_seen", "TF-IDF, in-domain CV"), ("tfidf_cv_full", "TF-IDF, in-domain CV, full text"),
             ("tfidf232_seen", "TF-IDF, 232K transferred"), ("lexicon_seen", "Explicit-term lexicon")]
    for f in sorted((D / "umd_v2").glob("*.json")):
        a = load(f); s = a["scores"]
        for k, lab in order:
            r = s[k]
            rows.append([NAMES[key(a["model"])], lab, f3(r["auc_hi_lo"]), ci(r["auc_hi_lo_ci95"]),
                         f"{r['spearman']:.2f}", f"{r['partial_spearman_given_length']:.2f}"])
    table("umd", "Expert-rated risk, UMD users.",
          ["Model", "Readout", "AUC", "95\\% CI", "$\\rho$", "Partial $\\rho$"], rows, "llcccc")


def minpairs():
    rows = []
    conds = [("counterfactual", "Denial"), ("third_person", "Third person"), ("topical", "Topical"),
             ("fiction", "Fiction"), ("recovery_past", "Past/recovered")]
    for f in sorted((D / "intent_minpairs").glob("*.json")):
        a = load(f)
        for c, lab in conds:
            r = a["per_condition"][c]
            rows.append([NAMES[key(a["model"])], f"{lab} ({r['n_control']})", f3(r["behav_base"]["auc"]),
                         f3(r["behav_clinical"]["auc"]), f3(r["proj_site"]["auc"]), f3(r["tfidf232"]["auc"]),
                         f3(r["lexicon"]["auc"]), f"{a['yes_rate'][c]['base']:.0%}".replace("%", r"\%"),
                         f"{a['yes_rate'][c]['clinical']:.0%}".replace("%", r"\%")])
    table("minpairs", "Minimal pairs, by control version.",
          ["Model", "Control (n)", "Answer, base", "Answer, clinical", "Projection", "TF-IDF 232K", "Lexicon",
           "Yes, base", "Yes, clinical"], rows, "llccccccc", size="scriptsize")


def necessity():
    rows, auc_rows = [], []
    files = [(f, "Suicide-Watch-232K") for f in sorted((D / "audit_klnec/suicide").glob("*.json"))] + \
            [(f, "register-matched") for f in sorted((D / "audit_swmh/suicide_depression").glob("*.json"))]
    for f, fitset in files:
        a = load(f); n = a["necessary"]; km = n["kl_matched"]; base = n["baseline_ld_gap"]
        modes = n["modes"]
        auc_rows.append([NAMES[key(a["model"])], fitset, f3(n["baseline_auc"]), f3(modes["zero"]["ablated_auc"]),
                         f"{modes['zero']['ablated_kl']:.2f}",
                         f3(modes["mean"]["ablated_auc"]) if "mean" in modes else "--",
                         f"{modes['mean']['ablated_kl']:.2f}" if "mean" in modes else "--"])
        rows.append([NAMES[key(a["model"])] + ("" if fitset.startswith("Suicide") else ", register-matched"),
                     "suicide direction", f"{km['target_kl']:.2f}", "--",
                     f"{km['target_ld_gap']:.2f} ({km['target_ld_gap'] / base:.0%})".replace("%", r"\%"), "--", "--"])
        for fam, lab in [("isotropic", "isotropic"), ("pca", "top-PC"), ("sign_random", "sign-random"),
                         ("shuffled_label", "shuffled-label")]:
            r = km["nulls"][fam]; gaps = [d["ld_gap"] for d in r["draws"]]
            n_le = sum(g <= km["target_ld_gap"] for g in gaps)
            rows.append(["", lab, f"{r['kl_mean']:.2f}", f"{r['beta_mean']:.2f}",
                         f"{r['ld_gap_mean']:.2f} ({r['ld_gap_mean'] / base:.0%})".replace("%", r"\%"),
                         f"{min(gaps):.2f}", f"{n_le}/{len(gaps)}"])
        all_gaps = [d["ld_gap"] for r in km["nulls"].values() for d in r["draws"]]
        verdict = "pass" if all(g > km["target_ld_gap"] for g in all_gaps) else "fail"
        rows.append(["", f"baseline gap {base:.2f}; {verdict}", "", "", "", "", ""])
    table("necessity", "Necessity against damage-matched nulls (direction fit on Suicide-Watch-232K unless marked "
          "register-matched).",
          ["Model", "Direction", "KL", "$\\beta$", "Remaining gap, mean (share)", "Smallest", "Draws $\\le$ target"],
          rows, "llccccc")
    table("necessity_auc", "Answer AUC on the held-out necessity posts before and after projecting the direction out of "
          "every layer: zero projection (as in the main text) and mean projection, with the KL divergence of each edit.",
          ["Model", "Direction fit on", "AUC before", "Zero proj.", "KL", "Mean proj.", "KL"], auc_rows, "llccccc")


def rank_of_use():
    rows = []
    for f in sorted((D / "gsa").glob("rank_*.json")):
        a = load(f); c = a["checks"]
        rows.append([NAMES[key(f.name)], a["concept"], f"L{a['site']} ({a['scan_full_swap_share'][str(a['site'])]:.2f})",
                     f"{a['class_gap_val']:.1f}", f"{c['dom']['share_of_full']:.2f}", f"{c['white_top1']['share_of_full']:.2f}",
                     f"{c['grad_mean']['share_of_full']:.2f}", f"{c['random1']['share_of_full']:.3f}",
                     f"{a['illusion_index']:.2f}"])
    table("rank", "Where the answer is formed, and in how many directions.",
          ["Model", "Concept", "Site", "Class gap", "DoM", "Whitened top-1", "Gradient", "Random", "Illusion"],
          rows, "llccccccc")


def steering():
    rows = []
    for sub in ["steering_topic", "steering_topic_loo"]:
        for f in sorted((D / sub).glob("*.json")):
            a = load(f)
            for var in (["orig"] if sub == "steering_topic" else ["minus_shared_loo"]):
                if var not in a["concepts"]["suicide"]["variants"]:
                    continue
                r = next(x for x in a["concepts"]["suicide"]["variants"][var] if abs(x["kappa"] - 0.1) < 1e-9)
                ratios = []
                for cv in a["concepts"].values():
                    q = next(x for x in cv["variants"][var] if abs(x["kappa"] - 0.1) < 1e-9)
                    ratios.append(q["nontopic_shift"] / q["on_shift"])
                lab = "raw" if var == "orig" else "shared component removed (LOO)"
                rows.append([NAMES[key(a["model"])], lab, f"{r['on_shift']:+.2f}", f"{r['off_shift']:+.2f}",
                             f"{r['nontopic_shift']:+.2f}", f"{r['neg_shift']:+.2f}",
                             f"{r['nontopic_shift'] / r['on_shift']:.2f}", f"{np.median(ratios):.2f}"])
    table("steering", "Steering along the suicide direction at equal collateral damage.",
          ["Model", "Variant", "Target", "Topical", "Non-topical", "Negated", "Ratio (suicide)", "Median ratio (8)"],
          rows, "llcccccc", size="scriptsize")


FU = D / "followups_20261024"
TAGS = {"llama": "llama_3_1_8b_instruct", "qwen": "qwen2_5_7b_instruct", "mistral": "mistral_7b_instruct_v0_3",
        "gemma": "gemma_2_9b_it"}
PLAB = {"base": "Base", "clinical": "Clinical", "contrast": "Contrast"}


def followups():
    """Register-matched follow-ups computed on CPU (swmh_followups.py, kaggle_composition.py)."""
    a = load(FU / "swmh_followups.json")
    lc = a["learning_curve"]
    rows = [[n, str(r["reps"]), f3(r["auc_mean"]), f"{num(r['auc_p2_5'])} to {num(r['auc_p97_5'])}", f3(r["sens90_mean"])]
            for n, r in lc["tfidf"].items()]
    table("learning_curve", "TF-IDF fit on random subsets of the fitting split (full posts), scored on the 1000 + 1000 "
          "evaluation posts: mean AUC and the central 95\\% of repetitions.",
          ["Posts per class", "Repetitions", "AUC, mean", "Central 95\\%", "Sens. at 90\\% spec., mean"], rows, "ccccc")
    eq = lc["labels_per_class_equivalent"]
    prompt_cols = [f"{lab} prompt" for lab in PLAB.values()]
    fmt = lambda v: v.replace(">", "$>$").replace("<=", "$\\le$") if isinstance(v, str) else f"{v:.0f}"
    rows = [[NAMES[k]] + [f"{f3(lc['llm_auc_full_posts'][t][p])} ({fmt(eq[t][p])})" for p in PLAB] for k, t in TAGS.items()]
    table("labels_equiv", "Model answer AUC on full posts and, in brackets, the number of labeled posts per class at which "
          "the mean TF-IDF AUC of the learning curve reaches it (interpolated on the logarithm of the subset size).",
          ["Model"] + prompt_cols, rows, "lccc")
    d = a["duplicates"]
    rows = [["Posts in SWMH training file, r/SuicideWatch / r/depression (more than 80 characters)",
             f"{d['swmh_train_posts']['SuicideWatch']} / {d['swmh_train_posts']['depression']}"],
            ["Texts posted verbatim in both forums (whole file)", str(d["exact_text_in_both_forums_full_train"])],
            ["Evaluation posts near-duplicating a fitting post (of the other class)",
             f"{d['eval_near_dup_of_fit_post']} ({d['eval_near_dup_of_fit_post_other_class']})"],
            ["Evaluation posts near-duplicating an evaluation post of the other class",
             str(d["eval_near_dup_across_classes_within_eval"])],
            ["Evaluation posts flagged, r/SuicideWatch / r/depression",
             f"{d['eval_flagged_by_class']['SuicideWatch']} / {d['eval_flagged_by_class']['depression']}"],
            ["TF-IDF AUC without flagged posts (all posts 0.843)", f3(d["auc_without_flagged"]["tfidf_full"])]]
    table("duplicates", "Exact and near-duplicate posts in the register-matched set (token containment of at least 90\\%, "
          "at least 15 word types).", ["", "Count / value"], rows, "p{0.62\\linewidth}c")
    rows = [[NAMES[k]] + [f"{f3(r['auc'])} ({num(r['minus_tfidf_ci95'][0])} to {num(r['minus_tfidf_ci95'][1])})"
                          for r in (d["auc_without_flagged"]["models"][t][p] for p in PLAB)] for k, t in TAGS.items()]
    table("dup_auc", "Model answer AUC on full posts without the flagged evaluation posts, with the 95\\% CI of the paired "
          "difference from TF-IDF on the same posts.", ["Model"] + prompt_cols, rows, "lccc",
          size="scriptsize")
    sc = a["stack_control"]
    names = {"char_tfidf": "Character n-gram TF-IDF", "lexicon": "Explicit-term lexicon",
             "char_tfidf+lexicon": "Character n-gram TF-IDF + lexicon"}
    rows = [[names[n], f3(sc["stacks"][n]["stack_auc"]), ci(sc["stacks"][n]["gain_ci95"])] for n in names]
    rows += [[f"{NAMES[k]}, {PLAB[p].lower()} prompt", f3(sc["stacks"][f'{t}|{p}']['stack_auc']),
              ci(sc["stacks"][f"{t}|{p}"]["gain_ci95"])] for k, t in TAGS.items() for p in ("base", "contrast")]
    table("stack_control", f"Stacking TF-IDF (cross-validated AUC {sc['tfidf_cv_auc']:.3f}) with a second score in the "
          "same 5-fold cross-validated logistic regression (full posts).",
          ["Second score", "Stack AUC", "Gain over TF-IDF, 95\\% CI"], rows, "lcc")
    rows = [[NAMES[k], PLAB[p], f"{c['mean_p_yes_SuicideWatch']:.2f}", f"{c['mean_p_yes_depression']:.2f}",
             f"{c['calibration_slope']:.3f}", f"{c['calibration_intercept']:.2f}", f"{c['brier']:.3f}"]
            for k, t in TAGS.items() for p, c in ((p, a["calibration_full_posts"][t][p]) for p in PLAB)]
    table("calibration", "The default answer read as a probability, P(yes) = 1/(1 + exp($-$(Yes $-$ No logit))), on full "
          "posts: mean P(yes) by forum, and the calibration slope and intercept from a logistic regression of the forum "
          "label on the logit difference (ideal slope 1).",
          ["Model", "Prompt", "P(yes), r/SW", "P(yes), r/dep", "Slope", "Intercept", "Brier"], rows, "llccccc")
    k = load(FU / "kaggle_composition.json")
    rows = []
    for forum, r in k["swmh"].items():
        for kind, lab in (("full_text", "whole text"), ("prefix25", "first 25 words")):
            x = r[kind]
            rows.append([forum.replace("self.", "r/"), lab, str(x["n_keyed"]),
                         f"{x.get('in_suicide', 0)} ({x.get('share_in_suicide', 0):.2%})".replace("%", r"\%"),
                         f"{x.get('in_non-suicide', 0)}"])
    table("kaggle", f"SWMH posts found in the two classes of the raw-text Kaggle release ({k['kaggle_rows']} posts), "
          "matched on normalized text.", ["SWMH forum", "Matched on", "Posts", "In suicide class", "In nonsuicide class"],
          rows, "llccc")


def steering_controls():
    """Suicide-direction steering: answer-polarity removal and interchange (paper runs), plus the follow-up runs
    with directions fit on the text alone (steering_post) and with the register-matched direction (steering_q9)."""
    rows = []
    def row(a, cname, var, lab):
        v = a["concepts"].get(cname, {}).get("variants", {}).get(var)
        if not v:
            return None
        r = v[0] if var == "interchange" else next(x for x in v if x["kappa"] is not None and abs(x["kappa"] - 0.1) < 1e-9)
        return [NAMES[key(a["model"])], lab, f"{r['on_shift']:+.2f}", f"{r['off_shift']:+.2f}",
                f"{r['nontopic_shift']:+.2f}", f"{r['on_shift'] - r['nontopic_shift']:+.2f}"]
    for f in sorted((D / "steering_topic").glob("*.json")):
        a = load(f)
        for var, lab in (("orig", "232K, question fit"), ("minus_yes", "232K, answer polarity removed"),
                         ("interchange", "232K, interchange")):
            r = row(a, "suicide", var, lab)
            if r:
                rows.append(r)
        for sub, cname, var, lab in (("steering_q9", "suicide_depression", "orig", "register-matched, question fit"),
                                     ("steering_q9", "suicide_depression", "interchange", "register-matched, interchange"),
                                     ("steering_post", "suicide", "orig", "232K, text-only fit"),
                                     ("steering_post", "suicide_depression", "orig", "register-matched, text-only fit")):
            g = [x for x in sorted((D / sub).glob("*.json")) if key(x.name) == key(f.name)] if (D / sub).exists() else []
            if g:
                r = row(load(g[0]), cname, var, lab)
                if r:
                    rows.append(r)
    table("steering_controls", "Steering along the suicide direction at KL 0.1 (interchange: no KL scaling): shift in the "
          "Yes$-$No logit difference for the target, topical off-target and non-topical (always-no) questions, and the "
          "target minus non-topical shift.",
          ["Model", "Direction and variant", "Target", "Topical", "Non-topical", "Target $-$ non-topical"], rows,
          "llcccc", size="scriptsize")
    # median generic ratio over all concepts, question fit vs text-only fit
    rows = []
    for sub, lab in (("steering_q9", "question fit"), ("steering_post", "text-only fit")):
        for f in sorted((D / sub).glob("*.json")) if (D / sub).exists() else []:
            a = load(f); ratios, diffs = [], []
            for cv in a["concepts"].values():
                q = next(x for x in cv["variants"]["orig"] if abs(x["kappa"] - 0.1) < 1e-9)
                ratios.append(q["nontopic_shift"] / q["on_shift"]); diffs.append(q["on_shift"] - q["nontopic_shift"])
            rows.append([NAMES[key(a["model"])], lab, str(len(ratios)), f"{np.median(ratios):.2f}",
                         f"{np.median(diffs):+.2f}"])
    if rows:
        table("steering_fit", "Generic push across all steered concepts: median ratio of non-topical to target shift and "
              "median target minus non-topical shift, for directions fit at the answer position and on the text alone. "
              "Follow-up runs with nine concepts (the eight of Table~\\ref{tab:steering} plus the register-matched "
              "suicide direction), run separately, so the question-fit medians are not identical to those in "
              "Table~\\ref{tab:steering}.",
              ["Model", "Directions", "Concepts", "Median ratio", "Median target $-$ non-topical"], rows, "llccc")


def offtarget():
    """Off-target necessity (offtarget_necessity.py): share of each task's class gap left after ablating a direction."""
    src = D / "offtarget_necessity"
    if not src.exists():
        return
    tasks = ["suicide", "suicide_depression", "sentiment", "toxicity", "sports", "truth"]
    tl = {"suicide": "Suicide (232K)", "suicide_depression": "Suicide (matched)", "sentiment": "Sentiment",
          "toxicity": "Toxicity", "sports": "Sports", "truth": "Truth"}
    for f in sorted(src.glob("*.json")):
        a = load(f); rows = []
        for scope, slab in (("all", "every layer"), ("upper", "upper half")):
            for d in tasks + ["answer_polarity"]:
                r = a["ablations"][f"{d}|{scope}"]
                rows.append([tl.get(d, "Answer polarity"), slab, f"{r['kl']:.2f}"] +
                            [f"{r['tasks'][t]['gap_share']:.2f}" for t in tasks])
        base = "; ".join(f"{tl[t]} {a['baseline'][t]['auc']:.3f}" for t in tasks)
        table(f"offtarget_{key(a['model'])}", f"Off-target necessity, {NAMES[key(a['model'])]}: share of each task's "
              f"Yes$-$No class gap left after zero-projecting a direction out of every layer or of the upper half "
              f"(layers {a['upper_layers'][0]}--{a['upper_layers'][-1]}); KL on neutral reference prompts. "
              f"Baseline AUCs: {base}.",
              ["Direction ablated", "Layers", "KL"] + [tl[t] for t in tasks], rows, "llcrrrrrr", size="scriptsize")
        rows = [["None (baseline)", "--", "--"] + [f"{a['baseline'][t]['auc']:.2f}" for t in tasks]]
        for scope, slab in (("all", "every layer"), ("upper", "upper half")):
            for d in tasks + ["answer_polarity"]:
                r = a["ablations"][f"{d}|{scope}"]
                rows.append([tl.get(d, "Answer polarity"), slab, f"{r['kl']:.2f}"] +
                            [f"{r['tasks'][t]['auc']:.2f}" for t in tasks])
        table(f"offtarget_auc_{key(a['model'])}", f"Off-target necessity, {NAMES[key(a['model'])]}: AUC of each task's "
              f"Yes$-$No logit difference (ranking) on the same held-out texts, before and after the edits of "
              f"Table~\\ref{{tab:offtarget_{key(a['model'])}}}.",
              ["Direction ablated", "Layers", "KL"] + [tl[t] for t in tasks], rows, "llcrrrrrr", size="scriptsize")


def benchmark():
    a = load(D / "audit/suicide/llama_3_1_8b_instruct__seed0.json"); g, d = a["gate"], a["decodable"]
    nz = a["necessary"]["modes"]["zero"]
    rows = [["Model answer", f3(g["behavioral_auc"]), ci(g["behavioral_ci95"])],
            ["TF-IDF", f3(g["tfidf_auc"]), ci(g["tfidf_ci95"])],
            ["Answer $-$ TF-IDF", "--", ci(g["behavior_minus_tfidf_ci95"])],
            [f"Probe, layer {d['site']}", f3(d["site_auc"]), "--"],
            ["Probe $-$ TF-IDF", "--", ci(d["probe_minus_tfidf_ci95"])],
            ["Answer after ablating the direction", f3(nz["ablated_auc"]), f"baseline {a['necessary']['baseline_auc']:.3f}"]]
    table("benchmark", "Suicide-Watch-232K, Llama-3.1-8B.", ["Readout", "AUC", "95\\% CI / note"], rows, "lcc")


if __name__ == "__main__":
    benchmark(); prompts(); tfidf_trunc(); umd(); minpairs(); necessity(); rank_of_use(); steering()
    followups(); steering_controls(); offtarget()
    print(sorted(p.name for p in OUT.glob("*.tex")))
