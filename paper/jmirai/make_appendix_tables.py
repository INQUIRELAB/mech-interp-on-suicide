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
    table("umd", "Clinician-rated risk, UMD users.",
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
    rows = []
    for f in sorted((D / "audit_klnec/suicide").glob("*.json")):
        a = load(f); n = a["necessary"]; km = n["kl_matched"]; base = n["baseline_ld_gap"]
        rows.append([NAMES[key(a["model"])], "suicide direction", f"{km['target_kl']:.2f}", "--",
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
    table("necessity", "Necessity against damage-matched nulls.",
          ["Model", "Direction", "KL", "$\\beta$", "Remaining gap, mean (share)", "Smallest", "Draws $\\le$ target"],
          rows, "llccccc")


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
    print(sorted(p.name for p in OUT.glob("*.tex")))
