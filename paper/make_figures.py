#!/usr/bin/env python
"""Render paper figures from the saved result JSONs into paper/figures/*.png.

Style: journal conventions for Computational Linguistics. Figures are rendered at
the actual print column width (5.31 in) with ~8 pt serif type matching the LaTeX
body, so nothing is scaled at \\includegraphics time. The claim lives in the
caption, not in an in-figure title. One semantic palette throughout:
red = suicidality feature, grey = matched control, blue = baseline/behavior.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RES = ROOT / "results" / "sc2_binary"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

COLW = 5.31  # CL text-column width in inches; every figure is this wide
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "serif",
    # Match the CL body typeface (clv2025.cls uses Palatino), not Times.
    "font.serif": ["Palatino Linotype", "Palatino", "URW Palladio L",
                   "P052", "TeX Gyre Pagella", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "errorbar.capsize": 2.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # --- readability: every text element black by default, never a light grey ---
    "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "axes.edgecolor": "black", "legend.labelcolor": "black",
})
# semantic palette, muted for print
RED = "#b0383d"      # the suicidality feature / targeted intervention
BLUE = "#34558b"     # baseline / behavioral readout
GREY = "#9aa1a9"     # matched control
GREY3 = "#4a5058"    # secondary/reference lines (darkened for print contrast)
OCHRE = "#c98a2e"    # second model family (Qwen)
GREEN = "#3e7d4f"    # third model family (Mistral)
LIGHTBLUE = "#7da1cc"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


# ---- Fig 1: probe AUC by (fractional) depth, three models ----
def fig1():
    gate = load("gate.json"); qw = load("secondmodel_qwen.json"); mi = load("secondmodel_mistral.json")
    if not (gate or qw or mi):
        return  # no results synced: keep the existing PNG rather than blanking it
    fig, ax = plt.subplots(figsize=(COLW, 2.7))
    if gate:
        L = [d["layer"] for d in gate["probe_per_layer"]]; a = [d["auc"] for d in gate["probe_per_layer"]]
        ax.plot([x / (len(L) - 1) for x in L], a, "-o", color=BLUE, ms=2.5,
                label="Llama-3.1-8B (0.968)")
    for d, lab, c, mk in [(qw, "Qwen2.5-7B (0.975)", OCHRE, "s"),
                          (mi, "Mistral-7B (0.982)", GREEN, "^")]:
        if d:
            L = [x["layer"] for x in d["per_layer_auc"]]; a = [x["auc"] for x in d["per_layer_auc"]]
            ax.plot([x / (len(L) - 1) for x in L], a, "-" + mk, ms=2.5, label=lab, color=c)
    ax.axhline(0.5, ls="--", c=GREY3, lw=0.7, label="chance")
    ax.set_xlabel("relative network depth (layer / total)")
    ax.set_ylabel("linear-probe ROC-AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig1_probe_by_layer.png"); plt.close(fig)


# ---- Fig 2: faithfulness bars + rank-k curve ----
def fig2():
    fa = load("faithfulness_perlayer.json"); rk = load("rankk.json")
    if not (fa or rk):
        return  # no results synced: keep the existing PNG rather than blanking it
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLW, 2.3))
    if fa:
        names = ["baseline", "ablate\nsuicidality", "ablate\nrandom"]
        vals = [fa["baseline"]["auc"], fa["suicidality_ablation"]["auc"], fa["random_ablation"]["auc"]]
        bars = ax1.bar(names, vals, color=[BLUE, RED, GREY], width=0.6, zorder=3)
        ax1.axhline(0.5, ls="--", c=GREY3, lw=0.7)
        ax1.text(2.45, 0.515, "chance", ha="right", va="bottom", fontsize=7.5, color="black")
        ax1.set_ylabel("behavioral ROC-AUC"); ax1.set_ylim(0, 1.05)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=8)
    if rk:
        ks = [s["k"] for s in rk["sweep"]]
        suic = [s["suic_auc"] for s in rk["sweep"]]; rnd = [s["random_auc"] for s in rk["sweep"]]
        ax2.plot(ks, rnd, "-o", color=GREY, label="random subspace", zorder=3)
        ax2.plot(ks, suic, "-o", color=RED, label="suicidality subspace", zorder=4)
        ax2.axhline(rk["baseline_auc"], ls=(0, (1, 1.2)), c=BLUE, lw=0.9, label="baseline")
        ax2.axhline(0.5, ls="--", c=GREY3, lw=0.7)
        ax2.text(ks[-1], 0.515, "chance", ha="right", va="bottom", fontsize=7.5, color="black")
        for k, v in zip(ks, suic):
            ax2.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, -10),
                         ha="center", fontsize=7.5, color="black")
        ax2.set_xlabel("subspace rank $k$ (per layer)"); ax2.set_ylabel("behavioral ROC-AUC")
        ax2.set_xticks(ks)
        ax2.set_ylim(0.3, 1.03); ax2.legend(loc="center right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig2_faithfulness_rankk.png"); plt.close(fig)


# ---- Fig 3: steering dose-response ----
def fig3():
    su = load("sufficiency.json")
    if not su:
        return
    fig, ax = plt.subplots(figsize=(COLW, 2.2))
    cs = [s["coef"] for s in su["sweep"]]; lds = [s["mean_ld"] for s in su["sweep"]]
    fr = [s["frac_over_threshold"] for s in su["sweep"]]
    thr = su["threshold"]
    # reference lines with inline labels (kept out of the legend to reduce clutter)
    ax.axhline(su["baseline_sui_ld"], ls=(0, (1, 1.2)), c=RED, lw=0.9)
    ax.axhline(thr, ls="--", c=GREY3, lw=0.7)
    ax.text(3.3, su["baseline_sui_ld"] + 0.2, f"genuine suicidal baseline ({su['baseline_sui_ld']:.1f})",
            ha="right", va="bottom", fontsize=7.5, color=RED)
    ax.text(3.3, thr - 0.3, f"decision threshold ({thr:.1f})",
            ha="right", va="top", fontsize=7.5, color=GREY3)
    # main steering curve
    ax.plot(cs, lds, "-o", color=BLUE, label="non-suicidal + suicidality dir", zorder=5)
    # random-direction control (distinct, never hidden)
    rc = su["random_c2"]
    ax.scatter([2], [rc["mean_ld"]], marker="D", s=18, color=GREY, edgecolor=GREY3,
               linewidth=0.6, zorder=6,
               label=f"+ random dir, $c$=2 ({rc['frac_over_threshold']*100:.0f}% flipped)")
    # %-flipped labels above each steering point
    for c, ld, f in zip(cs, lds, fr):
        ax.annotate(f"{f*100:.0f}%", (c, ld), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5, color=BLUE)
    ax.set_xlim(-0.22, 3.4); ax.set_ylim(0.5, 15.8)
    ax.set_xlabel("steering coefficient $c$")
    ax.set_ylabel("mean logit difference (Yes$-$No)")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "fig3_steering.png"); plt.close(fig)


# ---- Fig 4: severity gradient ----
def fig4():
    gr = load("gradient.json")
    if not gr:
        return
    order = ["neutral", "depression", "implicit_ideation", "explicit_suicide"]
    proj = gr["gradient_projection"]
    means = [proj[k][0] for k in order]
    lo = [proj[k][0] - proj[k][1] for k in order]; hi = [proj[k][2] - proj[k][0] for k in order]
    fig, ax = plt.subplots(figsize=(COLW, 2.1))
    # ordinal severity encoded as a single-hue ramp (neutral in grey)
    cols = [GREY, "#dcaeae", "#c97676", RED]
    ax.bar(range(len(order)), means, yerr=[lo, hi], color=cols, width=0.62, zorder=3,
           error_kw=dict(ecolor="#3a3a3a", lw=0.8, capthick=0.8))
    ax.axhline(0, c=GREY3, lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["neutral", "depression", "implicit\nideation", "explicit\nsuicide"])
    ax.set_ylabel("projection onto\nsuicidality direction")
    ax.set_ylim(-2.35, 1.9)
    for i, m in enumerate(means):
        if m >= 0:
            ax.annotate(f"{m:+.2f}", (i, m + hi[i]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color="black")
        else:
            ax.annotate(f"{m:+.2f}", (i, m - lo[i]), textcoords="offset points",
                        xytext=(0, -10), ha="center", fontsize=8, color="black")
    fig.savefig(FIG / "fig4_severity_gradient.png"); plt.close(fig)


# ---- Fig 5: generalization (transfer + causal) ----
def fig5():
    ge = load("generalization.json"); ca = load("crossablation.json")
    bc = load("bootstrap_ci.json")
    if not (ge or ca):
        return  # no results synced: keep the existing PNG rather than blanking it
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLW, 2.3))
    if ge:
        ds = list(ge["results"].keys())
        within = [ge["results"][d]["within_probe_auc"] for d in ds]
        # Use the bootstrap point estimate for transfer AUC so the bars match the text
        # and the framework diagram (which report bootstrap means + CIs), not the raw
        # single-pass transfer_auc (differs by ~0.001).
        trans = [bc[f"transfer_auc_{d}"][0] if bc and f"transfer_auc_{d}" in bc
                 else ge["results"][d]["transfer_auc"] for d in ds]
        x = range(len(ds)); w = 0.35
        b1 = ax1.bar([i - w/2 for i in x], within, w, label="within-dataset probe", color=BLUE)
        b2 = ax1.bar([i + w/2 for i in x], trans, w, label="transfer (232K direction)", color=LIGHTBLUE)
        ax1.axhline(0.5, ls="--", c=GREY3, lw=0.7)
        ax1.set_xticks(list(x)); ax1.set_xticklabels(ds); ax1.set_ylabel("ROC-AUC")
        ax1.set_ylim(0, 1.45)
        ax1.legend(loc="upper center", ncol=1, fontsize=7.5,
                   bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
        ax1.bar_label(b1, fmt="%.3f", fontsize=7, padding=1)
        ax1.bar_label(b2, fmt="%.3f", fontsize=7, padding=1)
    if ca:
        ds = list(ca["results"].keys())
        base = [ca["results"][d]["baseline"]["auc"] for d in ds]
        abl = [ca["results"][d]["ablate07"]["auc"] for d in ds]
        rnd = [ca["results"][d]["random"]["auc"] for d in ds]
        x = range(len(ds)); w = 0.27
        ax2.bar([i - w for i in x], base, w, label="baseline", color=BLUE)
        ax2.bar(list(x), abl, w, label="ablate 232K dir", color=RED)
        ax2.bar([i + w for i in x], rnd, w, label="ablate random", color=GREY)
        ax2.axhline(0.5, ls="--", c=GREY3, lw=0.7)
        ax2.set_xticks(list(x)); ax2.set_xticklabels(ds); ax2.set_ylabel("behavioral ROC-AUC")
        ax2.set_ylim(0, 1.55)
        ax2.legend(loc="upper center", ncol=1, fontsize=7.5,
                   bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig5_generalization.png"); plt.close(fig)


# ---- Fig 6: per-head knockout (heatmap + layer profile) ----
def fig6():
    hk = load("head_knockout.json")
    if not hk:
        return
    import numpy as np
    ad = np.array(hk["auc_drop"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLW, 2.4),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax1.imshow(ad, cmap="Reds", aspect="auto", vmin=0)
    ax1.set_xlabel("attention head"); ax1.set_ylabel("layer")
    cb = fig.colorbar(im, ax=ax1, label="baseline AUC $-$ knockout AUC", fraction=0.046)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("baseline AUC $-$ knockout AUC", fontsize=7.5)
    # annotate top-3 heads
    for t in hk["top_heads"][:3]:
        ax1.text(t["head"], t["layer"], "$\\times$", ha="center", va="center",
                 color=BLUE, fontsize=8)
    layers = list(range(ad.shape[0]))
    ax2.plot(ad.max(1), layers, "-o", ms=2, color=RED, label="max head")
    ax2.plot(ad.mean(1), layers, "-o", ms=2, color=BLUE, label="mean head")
    ax2.axvline(0, ls="--", c=GREY3, lw=0.7)
    ax2.invert_yaxis(); ax2.set_xlabel("AUC drop"); ax2.set_ylabel("layer")
    ax2.legend()
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig6_head_knockout.png"); plt.close(fig)


# ---- Fig 6b: head-knockout forest plot with bootstrap CIs (paper figure) ----
def fig6_forest():
    st = load("head_knockout_stats.json")
    if not st:
        return
    import numpy as np
    rows = st["top_heads"]
    fig, ax = plt.subplots(figsize=(COLW, max(2.2, 0.28 * len(rows))))
    ys = np.arange(len(rows))[::-1]
    labels_ = [f"L{r['layer']}H{r['head']}" + ("*" if r["sig_fdr_0.05"] else "") for r in rows]
    drops = [r["auc_drop"] for r in rows]
    errs = [[r["auc_drop"] - r["ci95"][0] for r in rows],
            [r["ci95"][1] - r["auc_drop"] for r in rows]]
    ax.errorbar(drops, ys, xerr=errs, fmt="o", ms=3, color=RED, elinewidth=0.8)
    ax.axvspan(0, st["null_all_heads"]["p95"], color=GREY, alpha=0.25,
               label="across-head null (0–95th pct)")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(labels_, fontsize=7.5)
    ax.set_xlabel("AUC drop (baseline $-$ knockout), 95% CI")
    ax.legend()
    fig.savefig(FIG / "fig6_head_knockout_forest.png"); plt.close(fig)


def _norm(auc, lo=0.5):
    return max(0.0, min(1.0, (auc - lo) / (1 - lo)))


# ---- Fig 7: validation scorecard (forest plot; replaces the radar and the body table) ----
def fig7():
    """Forest-style scorecard: per rung of the causal ladder, the suicidality feature
    (red) vs. its matched control (grey) on a normalized 0--1 axis, with the raw metric
    printed on the right so the figure carries the evidence table's numbers too. The
    failed steering rung sits at zero, below a divider, where it belongs."""
    import numpy as np
    sp = load("specificity.json"); fa = load("faithfulness_perlayer.json")
    rk = load("rankk.json"); ge = load("generalization.json")
    gr = load("gradient.json"); ss = load("steering_specificity.json")
    if not all([sp, fa, rk, ge, gr, ss]):
        return

    def removed(auc, base):  # share of above-chance signal an ablation removes
        return max(0.0, min(1.0, (base - auc) / (base - 0.5)))

    base = fa["baseline"]["auc"]
    k4 = {s["k"]: s for s in rk["sweep"]}[4]
    transfer = float(np.mean([v["transfer_auc"] for v in ge["results"].values()]))
    dep = gr["depression_specificity"]["probe_auc"][0]
    sem_f = sp["probe_recall_masked"] / sp["probe_recall_orig"]
    sem_c = sp["tfidf_recall_masked"] / sp["tfidf_recall_orig"]
    sw = {round(s["coef"], 1): s for s in ss["sweep"]}
    on_rise = sw[1.0]["on_target_ld"] - sw[0.0]["on_target_ld"]
    off_rise = sw[1.0]["off_target_ld"] - sw[0.0]["off_target_ld"]
    selectivity = max(0.0, (on_rise - off_rise) / on_rise) if on_rise > 0 else 0.0

    # built bottom-to-top; reading top->bottom traces the causal ladder
    # (label, eff_norm, ctl_norm, feature_str, control_str, holds?)
    arr = "$\\rightarrow$"
    rows = [
        ("Selective\nsteering", selectivity, 0.0, "not selective", "off-target rises more", False),
        ("General\n(transfer)", _norm(transfer), 0.0, f"AUC {transfer:.2f}", "random $\\approx$ chance", True),
        ("Low-rank\n(rank-4)", removed(k4["suic_auc"], rk["baseline_auc"]),
         removed(k4["random_auc"], rk["baseline_auc"]), f"rank-4 {arr} 0.40", "random 0.95", True),
        ("Necessary\n(ablation)", removed(fa["suicidality_ablation"]["auc"], base),
         removed(fa["random_ablation"]["auc"], base), f"0.97 {arr} 0.66", "random 0.97", True),
        ("Specific\n(vs. depression)", _norm(dep), 0.0, f"AUC {dep:.2f}", "chance 0.50", True),
        ("Semantic\n(keywords masked)", sem_f, sem_c, "recall 0.74", "TF-IDF 0.26", True),
    ]
    n = len(rows)
    cf, cc = 1.28, 1.62  # x of the feature / control value columns
    fig, ax = plt.subplots(figsize=(COLW, 3.0))
    for i, (lab, eff, ctl, fs, cs, ok) in enumerate(rows):
        ax.plot([0, 1.0], [i, i], color="#e8eaec", lw=3, solid_capstyle="butt", zorder=0)
        ax.plot([ctl, eff], [i, i], color="#c2c7cf", lw=1.2, zorder=1)
        ax.scatter([ctl], [i], s=18, color=GREY, zorder=2)
        if ok:
            ax.scatter([eff], [i], s=38, color=RED, zorder=3)
            ax.text(cf, i, fs, va="center", ha="center", fontsize=8, color=RED)
            ax.text(cc, i, cs, va="center", ha="center", fontsize=8, color="black")
        else:
            ax.scatter([eff], [i], s=38, facecolors="none", edgecolors=RED, lw=1.0, zorder=3)
            ax.text(cf, i, fs, va="center", ha="center", fontsize=8, color=RED, style="italic")
            ax.text(cc, i, cs, va="center", ha="center", fontsize=8, color="black", style="italic")
    # column headers: feature in red (ties to the red feature dots), control/rest body-black
    ax.text(0.5, n - 0.35, "normalized score", ha="center", fontsize=8, color="black")
    ax.text(cf, n - 0.35, "feature", ha="center", fontsize=8, color=RED)
    ax.text(cc, n - 0.35, "control", ha="center", fontsize=8, color="black")
    ax.axhline(0.5, color="#d0d4d9", lw=0.7, ls=":")  # divider above the failed rung
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.06, 1.80); ax.set_ylim(-0.85, n - 0.05)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.text(0.5, -0.7, "0 = control / chance floor;   1 = ideal", ha="center",
            fontsize=7.5, color="black")
    fig.savefig(FIG / "fig7_validation_scorecard.png"); plt.close(fig)


# ---- Fig 8: cross-model replication (grouped bars) ----
def fig8():
    import numpy as np
    gate = load("gate.json"); fa = load("faithfulness_perlayer.json")
    qw = load("crossmodel_qwen2_5_7b_instruct.json")
    mi = load("crossmodel_mistral_7b_instruct_v0_3.json")
    if not all([gate, fa, qw, mi]):
        return

    def nec(b, a):
        return max(0.0, min(1.0, (b - a) / (b - 0.5)))

    metrics = ["probe AUC", "behavioral AUC", "necessity\n(signal ablated)"]
    models = [
        ("Llama-3.1-8B-Inst", [gate["best_probe_auc"], gate["behav_auc"],
                               nec(fa["baseline"]["auc"], fa["suicidality_ablation"]["auc"])], BLUE),
        ("Qwen2.5-7B-Inst", [qw["best_probe_auc"], qw["behavioral_auc"],
                             nec(qw["faithfulness"]["baseline_auc"], qw["faithfulness"]["ablate_auc"])], OCHRE),
        ("Mistral-7B-Inst", [mi["best_probe_auc"], mi["behavioral_auc"],
                             nec(mi["faithfulness"]["baseline_auc"], mi["faithfulness"]["ablate_auc"])], GREEN),
    ]
    y = np.arange(len(metrics)); h = 0.26
    fig, ax = plt.subplots(figsize=(COLW, 2.6))
    for j, (name, vals, c) in enumerate(models):
        ax.barh(y + (j - 1) * h, vals, height=h, color=c, label=name, zorder=2)
        for yi, v in zip(y + (j - 1) * h, vals):
            ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(3, 0),
                        va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=8.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel("ROC-AUC (probe, behavioral); share ablated (necessity)")
    ax.invert_yaxis()
    # legend above the axes so the long model names never collide with the bars
    # or run off the right edge (was loc="center right", which overprinted the
    # behavioral-AUC cluster and clipped at the figure border).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=7.5, columnspacing=1.2, handlelength=1.4,
              handletextpad=0.5, borderaxespad=0.0)
    fig.savefig(FIG / "fig8_crossmodel_bars.png"); plt.close(fig)


# ---- Fig 9: keyword-masking slopegraph (semantic vs lexical) ----
def fig9():
    sp = load("specificity.json")
    if not sp:
        return
    fig, ax = plt.subplots(figsize=(COLW, 2.6))
    x = [0, 1]
    series = [("TF-IDF (lexical)", [sp["tfidf_recall_orig"], sp["tfidf_recall_masked"]], GREY3),
              ("layer-16 probe", [sp["probe_recall_orig"], sp["probe_recall_masked"]], RED)]
    # left-end labels (original text) sit at near-equal recall (1.00 vs 0.97), so
    # push the higher series' label up and the lower one's down to avoid collision
    hi0 = max(s[1][0] for s in series)
    for label, y, c in series:
        ax.plot(x, y, "o-", color=c, label=label)
        dy0 = 6 if y[0] >= hi0 else -12
        ax.annotate(f"{y[0]:.2f}", (0, y[0]), textcoords="offset points", xytext=(-24, dy0),
                    fontsize=8, color=c)
        ax.annotate(f"{y[1]:.2f}", (1, y[1]), textcoords="offset points", xytext=(8, -2),
                    fontsize=8, color=c)
        ax.annotate(f"$-${y[0]-y[1]:.2f}", (1, y[1]), textcoords="offset points",
                    xytext=(8, 8), fontsize=7.5, color=c)
    ax.set_xticks(x); ax.set_xticklabels(["original text", f"top-{sp['n_keywords']} keywords masked"])
    ax.set_ylabel("recall on suicidal posts"); ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, 1.4)
    ax.legend(loc="lower left")
    fig.savefig(FIG / "fig9_keyword_slope.png"); plt.close(fig)


# ---- copy figures rendered directly by the analysis scripts (mechanism, depth) ----
def _copy(name, src_dir=RES):
    import shutil
    src = src_dir / name
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copyfile(src, FIG / name)


def fig_sae():       _copy("sae_features_L16.png")
def fig_logitlens(): _copy("logitlens.png")
def fig_atp():       _copy("attribution_patching.png")
def fig_eap():       _copy("edge_attribution.png")


# ---- UMD expert-severity: replotted from JSON in the house style (was a raw
#      matplotlib-default copy; now matches the other figures) ----
def fig_umd():
    import numpy as np
    d = load("umd_severity.json")
    if not d:
        try:
            _copy("umd_severity.png")   # fall back to the analysis-script render
        except FileNotFoundError:
            pass
        return
    lv = ["a (none)", "b (low)", "c (moderate)", "d (severe)"]
    P = d["projection_by_level"]
    means = [P[k][0] for k in lv]
    lo = [P[k][0] - P[k][1] for k in lv]
    hi = [P[k][2] - P[k][0] for k in lv]
    tauc = d["transfer_auc_hi_vs_lo"]
    bauc = d.get("behavioral_auc_hi_vs_lo", tauc)
    pu = d.get("per_user")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLW, 2.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    # left: mean projection onto the suicidality direction by clinician risk level
    x = np.arange(4)
    ax1.errorbar(x, means, yerr=[lo, hi], fmt="o-", color=RED, lw=1.1, ms=3.5,
                 capsize=2.5, ecolor=GREY3, elinewidth=0.8, zorder=3)
    ax1.axhline(0, c=GREY3, lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["a\nnone", "b\nlow", "c\nmod.", "d\nsevere"])
    ax1.set_xlabel("clinician risk level")
    ax1.set_ylabel("mean projection onto\nsuicidality direction")
    ax1.set_xlim(-0.4, 3.4)

    # right: per-user separation (strip if raw scores are in the JSON, else AUC bars)
    if pu and "proj" in pu and "severity" in pu:
        rng = np.random.default_rng(0)
        proj = np.asarray(pu["proj"], dtype=float)
        sev = np.asarray(pu["severity"], dtype=float)
        jit = sev + rng.normal(0, 0.06, len(sev))
        ax2.scatter(jit, proj, s=6, alpha=0.35, color=RED, edgecolors="none", zorder=2)
        ax2.set_xticks(range(4)); ax2.set_xticklabels(list("abcd"))
        ax2.set_xlabel("clinician risk level")
        ax2.set_ylabel("per-user projection")
        ax2.text(0.03, 0.97, f"transfer AUC (c,d vs a,b) = {tauc[0]:.2f}",
                 transform=ax2.transAxes, va="top", fontsize=7.5)
    else:
        yerr = [[tauc[0] - tauc[1], bauc[0] - bauc[1]],
                [tauc[2] - tauc[0], bauc[2] - bauc[0]]]
        ax2.bar([0, 1], [tauc[0], bauc[0]], yerr=yerr, color=[RED, BLUE], width=0.55,
                zorder=3, error_kw=dict(ecolor=GREY3, lw=0.8, capsize=2.5))
        ax2.axhline(0.5, ls="--", c=GREY3, lw=0.7)
        ax2.text(1.45, 0.515, "chance", ha="right", va="bottom", fontsize=7.5, color="black")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["feature\nprojection", "model\nreadout"])
        ax2.set_ylabel("high- vs low-risk ROC-AUC")
        ax2.set_ylim(0, 1.05)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIG / "umd_severity.png"); plt.close(fig)


# ---- mechanism figures replotted from cached JSON (vector-grade, not 150-dpi copies) ----
def fig_patching():
    d = load("patching.json")
    L = list(range(d["n_layers"]))
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    ax.plot(L, d["attn_ie"], "-o", color=RED, ms=2.5, label="attention")
    ax.plot(L, d["mlp_ie"], "-s", color=BLUE, ms=2.5, label="MLP")
    ax.axhline(0, ls="--", c=GREY3, lw=0.7)
    ax.set_xlabel("layer"); ax.set_ylabel("normalized indirect effect")
    ax.legend(loc="upper right")
    fig.savefig(FIG / "patching_ie.png"); plt.close(fig)


def fig_dla():
    d = load("dla.json")
    L = list(range(d["n_layers"]))
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    ax.plot(L, d["attn_dla_diff"], "-o", color=RED, ms=2.5, label="attention")
    ax.plot(L, d["mlp_dla_diff"], "-s", color=BLUE, ms=2.5, label="MLP")
    ax.axhline(0, ls="--", c=GREY3, lw=0.7)
    ax.set_xlabel("layer")
    ax.set_ylabel("logit-diff contribution (suicide $-$ non)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "dla.png"); plt.close(fig)


def fig_salience():
    import numpy as np
    d = json.loads((ROOT / "results" / "gcl" / "steering_salience.json").read_text())
    rows = list(d["matrix"].keys())      # 5 concept directions + "random"
    qs = d["questions"]                  # 6 readout questions (incl. off-target "food")
    base = d["baselines"]
    # matrix stores the per-question SHIFT at c=1; the readout LEVEL is baseline + shift.
    M = np.array([[base[q] + d["matrix"][r]["1.0"][q] for q in qs] for r in rows])
    spread = M.max(axis=1) - M.min(axis=1)
    y = np.arange(len(rows))

    # constrained_layout keeps the right-panel x-label and the colorbar from being
    # clipped (the previous manual wspace + tight bbox cut the label).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLW, 2.7),
                                   gridspec_kw={"width_ratios": [1.85, 1.05]},
                                   constrained_layout=True)

    im = ax1.imshow(M, aspect="auto", cmap="viridis")
    ax1.set_xticks(range(len(qs))); ax1.set_xticklabels(qs, rotation=40, ha="right")
    ax1.set_yticks(y); ax1.set_yticklabels(rows)
    ax1.set_xlabel("readout question"); ax1.set_ylabel("steering direction")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label("Yes$-$No readout level", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # right: within-row spread. Label every bar (was blank), no in-figure title.
    colors = [GREY if r == "random" else RED for r in rows]
    ax2.barh(y, spread, color=colors, height=0.68, zorder=3)
    ax2.set_yticks(y); ax2.set_yticklabels(rows)
    ax2.invert_yaxis()                      # match the heatmap row order (first row on top)
    ax2.set_xlim(0, float(spread.max()) * 1.22)
    # two lines: the single-line label overflowed the narrow panel and was cut
    ax2.set_xlabel("within-row readout\nspread (logits)", linespacing=1.25)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)
    for yi, s in zip(y, spread):
        ax2.text(s + float(spread.max()) * 0.03, yi, f"{s:.1f}",
                 va="center", fontsize=7, color="black")

    fig.savefig(FIG / "steering_salience.png"); plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig6_forest, fig7, fig8, fig9,
          fig_sae, fig_patching, fig_logitlens, fig_umd, fig_atp, fig_dla, fig_eap,
          fig_salience):
    try:
        f(); print(f"ok: {f.__name__}")
    except Exception as e:
        print(f"FAIL {f.__name__}: {e}")
print(f"figures in {FIG}")
