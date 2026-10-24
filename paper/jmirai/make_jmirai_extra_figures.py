"""Figures restored from the npj manuscript for the JMIR AI version, redrawn in the same house
style as make_jmirai_figures.py and limited to analyses that survived the corrected audit.

  figures/fig6_heads           main text: attention heads write the direction, but not only for suicide
  figures/figA2_localization   Appendix 1: where the answer forms, and in how many directions
  figures/figA3_transfer       Appendix 1: the 232K direction is used on other corpora
  figures/figA1_geometry       Appendix 1: what ablation and steering do to the activation

Sources. Localization: data/audit (probe per layer, n=1000/class) and data/gsa/rank_* (state
swaps). Transfer: data/transfer/transfer_dedup.json (near-duplicates of 232K posts removed).
Head sets: data/heads (revision_controls.ipynb rerun on yorha, 2026-10-16, Llama-3.1-8B-Instruct
at commit 0e9e39f2, fp16); these reproduce the values earlier typed from ../npjai_supp.tex.
Run:  python make_jmirai_extra_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np

import make_jmirai_figures as base   # rcParams, colours, markers, helpers
from make_jmirai_figures import INK, INK2, CTRL, CTRL_LT, FAINT, RED, FULL, MODELS, MARK, D, panel, dot, save

RED_LT = "#e9c3c5"


def load(p):
    return json.load(open(D / p, encoding="utf8"))


# ---------------------------------------------------------------- Figure 6 (main text)
def fig_heads():
    w = load("heads/heads_write_direction.json"); spec = load("heads/specificity_mean_ablation.json")
    ks = ["5", "10", "20", "40"]; k = np.array([int(x) for x in ks])
    proj_restored = [100 * w["patching"][x]["proj_ie_top"] for x in ks]   # joint patching, % of L16 projection
    gap_restored = [100 * w["patching"][x]["ld_ie_top"] for x in ks]      # joint patching, % of LD gap
    sep_left = [100 * w["knockout"]["mean"]["by_k"][x]["proj_gap_frac_remaining"] for x in ks]  # mean-ablation
    # mean-ablation AUC by task, and the range over layer-matched random head sets on suicide
    auc = {lab: ([0] + list(k), [spec[f"{t}/mean"]["baseline_auc"]] + [spec[f"{t}/mean"]["by_k"][x]["auc_top"] for x in ks])
           for t, lab in [("suicidality", "suicide"), ("sentiment", "sentiment (SST-2)"), ("topic_sports", "sports (AG News)")]}
    rand_lo = min(spec["suicidality/mean"]["by_k"][x]["auc_rand_range"][0] for x in ks)
    rand_hi = max(spec["suicidality/mean"]["by_k"][x]["auc_rand_range"][1] for x in ks)

    fig = plt.figure(figsize=(FULL, 2.3))
    gs = fig.add_gridspec(1, 2, wspace=0.42)
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])

    ax.plot(k, gap_restored, "-", color=INK, lw=0.9)
    ax.plot(k, gap_restored, "s", ms=3.0, color=INK, mfc="white", mew=0.7, zorder=4)
    ax.plot(k, proj_restored, "-", color=RED, lw=0.9)
    ax.plot(k, proj_restored, "o", ms=3.2, color=RED, mec="white", mew=0.5, zorder=4)
    ax.plot(k, sep_left, "--", color=RED, lw=0.9, dashes=(3, 1.6))
    ax.plot(k, sep_left, "o", ms=3.2, color=RED, mfc="white", mew=0.7, zorder=4)
    ax.text(41.5, gap_restored[-1] + 4.5, "patch in: logit gap restored", ha="right", fontsize=5.8, color=INK)
    ax.text(41.5, proj_restored[-1] - 9.5, "patch in: direction restored", ha="right", fontsize=5.8, color=RED)
    ax.text(41.5, sep_left[-1] + 17, "ablate: direction left", ha="right", fontsize=5.8, color=RED)
    ax.text(41.5, sep_left[-1] + 12.2, "(random head sets: 75–100% left)", ha="right", fontsize=5.4, color=INK2)
    ax.text(41.5, proj_restored[-1] - 14.5, "(random head sets: below 0)", ha="right", fontsize=5.4, color=INK2)
    ax.set_xticks(k); ax.set_xlim(2, 43)
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xlabel("top-$k$ heads (ranked by patching)")
    ax.set_ylabel("% of effect at layer 16")

    bx.axhspan(rand_lo, rand_hi, color=FAINT, lw=0, zorder=0)
    bx.text(24, 0.983, "random head sets", fontsize=5.6, color=INK2, va="bottom")
    bx.axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)))
    bx.text(1.0, 0.505, "chance", fontsize=5.6, color=INK2, ha="left", va="bottom")
    styles = {"suicide": (RED, "o", True), "sentiment (SST-2)": (INK, "o", False), "sports (AG News)": (INK, "^", True)}
    for name, (xs, ys) in auc.items():
        c, m, filled = styles[name]
        bx.plot(xs, ys, "-", color=c, lw=0.9)
        bx.plot(xs, ys, m, ms=3.2, color=c, mfc=c if filled else "white", mec=c, mew=0.7, zorder=4)
    bx.text(41.2, auc["suicide"][1][-1], "suicide", fontsize=5.8, color=RED, va="center", ha="left")
    bx.text(41.2, auc["sentiment (SST-2)"][1][-1], "sentiment", fontsize=5.8, color=INK, va="center", ha="left")
    bx.text(41.2, auc["sports (AG News)"][1][-1], "sports", fontsize=5.8, color=INK, va="center", ha="left")
    bx.set_xticks([0, 5, 10, 20, 40]); bx.set_xlim(-1.5, 47)
    bx.spines["bottom"].set_bounds(0, 40)
    bx.set_ylim(0.42, 1.02); bx.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bx.set_xlabel("top-$k$ heads mean-ablated")
    bx.set_ylabel("ROC-AUC, same yes/no format")

    panel(fig, ax, "A", "Llama-3.1-8B, heads ranked by patching")
    panel(fig, bx, "B")
    save(fig, "fig6_heads")


# ---------------------------------------------------------------- Figure A1
def fig_localization():
    fig = plt.figure(figsize=(FULL, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.0], wspace=0.85)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))

    pr = load("audit/suicide/llama_3_1_8b_instruct__seed0.json")["decodable"]
    L = [p["hidden_index"] for p in pr["probe_per_layer"]]
    A = [p["auc"] for p in pr["probe_per_layer"]]
    ax.plot(L, A, "-", color=INK, lw=0.9)
    ax.plot(pr["site"], pr["site_auc"], "o", ms=3.4, color=RED, mec="white", mew=0.5, zorder=4)
    ax.text(pr["site"] + 1, pr["site_auc"] - 0.035, f"layer {pr['site']}: {pr['site_auc']:.3f}", fontsize=5.8,
            color=RED, va="top")
    ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)))
    ax.text(32, 0.51, "chance", fontsize=5.6, color=INK2, ha="right", va="bottom")
    ax.set_xlim(-1, 33); ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_ylim(0.45, 1.02); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("layer (Llama-3.1-8B)"); ax.set_ylabel("linear-probe ROC-AUC")

    files = {"llama": "gsa/rank_suicide_meta_llama_3_1_8b_instruct__seed0.json",
             "qwen": "gsa/rank_suicide_qwen2_5_7b_instruct__seed0.json",
             "mistral": "gsa/rank_suicide_mistral_7b_instruct_v0_3__seed0.json"}
    G = {m: load(f) for m, f in files.items()}
    bx.axhline(0.8, color=INK2, lw=0.5, ls=(0, (3, 2)))
    bx.text(1, 0.78, "0.8: site\nchosen here", fontsize=5.6, color=INK2, va="top", linespacing=1.1)
    for m, g in G.items():
        s = g["scan_full_swap_share"]; xs = np.array(sorted(map(int, s))); ys = np.array([s[str(x)] for x in xs])
        bx.plot(xs, ys, "-", color=INK, lw=0.6)
        for xx, yy in zip(xs, ys):
            dot(bx, xx, yy, m, ms=2.8)
    bx.set_xlim(0, 33); bx.set_xticks([0, 8, 16, 24, 32])
    bx.set_ylim(-0.08, 1.12); bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xlabel("layer of the state swap"); bx.set_ylabel("share of class gap moved")
    for i, m in enumerate(G):
        mk, filled = MARK[m]
        bx.plot(21, 0.40 - i * 0.085, mk, ms=2.8, color=INK, mfc=INK if filled else "white", mec=INK, mew=0.7)
        bx.text(22.3, 0.40 - i * 0.085, MODELS[m], fontsize=5.6, va="center")

    rows = [("dom", "difference of means"), ("white_top1", "whitened, top 1"), ("grad_mean", "mean readout gradient"),
            ("random1", "random direction"), ("proj_top1", "projection, top 1")]
    cx.axvline(1.0, color=INK2, lw=0.5, ls=(0, (3, 2)))
    cx.text(1.03, -0.45, "full swap", fontsize=5.6, color=INK2, va="center")
    for j, m in enumerate(G):
        for i, (key, _) in enumerate(rows):
            v = G[m]["checks"][key]["share_of_full"]
            c = RED if key == "dom" else (CTRL if key == "random1" else INK)
            dot(cx, v, i + (j - 1) * 0.2, m, color=c, ms=2.8)
    cx.set_yticks(range(len(rows))); cx.set_yticklabels([r[1] for r in rows])
    cx.get_yticklabels()[0].set_color(RED)
    cx.set_ylim(len(rows) - 0.4, -0.6); cx.tick_params(axis="y", length=0); cx.spines["left"].set_visible(False)
    cx.set_xlim(-0.05, 1.9); cx.set_xticks([0, 0.5, 1.0, 1.5])
    cx.set_xlabel("one direction / full-state swap")
    cx.annotate("> 1: an illusion", xy=(1.55, 4.0), xytext=(1.25, 3.25), fontsize=5.6, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4))

    panel(fig, ax, "A", "Decodable early")
    panel(fig, bx, "B", "Used late")
    panel(fig, cx, "C", "Carried by one direction")
    save(fig, "figA2_localization")


# ---------------------------------------------------------------- Figure A2
def fig_transfer():
    d = json.load(open(D / "transfer/transfer_dedup.json", encoding="utf8"))
    rows = [("Kaggle-raw", d["SDCNL"]), ("SWMH, r/SuicideWatch\nvs other forums", d["SWMH"])]
    fig, ax = plt.subplots(figsize=(FULL * 0.6, 1.35))
    for y, (lab, r) in zip([0, 1], rows):
        c = r["causal"]
        ax.annotate("", xy=(c["ablate_232K_dirs"], y), xytext=(c["baseline"], y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color=RED, lw=0.9,
                                    shrinkA=1.5, shrinkB=0))
        ax.plot(c["baseline"], y, "o", color=INK, ms=3.2, mec="white", mew=0.4, zorder=4)
        ax.plot(c["random"], y - 0.22, "o", ms=3.0, mfc="white", mec=CTRL, mew=0.7, zorder=4)
        ax.text(c["baseline"] + 0.008, y, f"{c['baseline']:.3f}", va="center", fontsize=5.6)
        ax.text(c["ablate_232K_dirs"] - 0.008, y, f"{c['ablate_232K_dirs']:.3f}", va="center", ha="right",
                fontsize=5.6, color=RED)
        lo, hi = c["drop_ci95"]
        ax.text((c["baseline"] + c["ablate_232K_dirs"]) / 2, y - 0.17,
                f"drop {c['baseline'] - c['ablate_232K_dirs']:.3f} ({lo:.3f}–{hi:.3f})", ha="center",
                va="bottom", fontsize=5.4, color=RED)
        ax.text(c["random"] + 0.008, y - 0.22, "random direction", va="center", fontsize=5.4, color=INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], linespacing=1.1)
    ax.set_ylim(1.45, -0.55); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.6, 1.02); ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("model-answer ROC-AUC (Llama-3.1-8B), 80 posts per class")
    ax.text(0.6, -0.5, "● unedited   ", fontsize=5.6, va="center")
    ax.text(0.665, -0.5, "→ ablate the 232K direction", fontsize=5.6, va="center", color=RED)
    save(fig, "figA3_transfer")


# ---------------------------------------------------------------- Figure A3
def fig_geometry():
    fig = plt.figure(figsize=(FULL * 0.62, 2.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 112); ax.set_ylim(0, 64); ax.axis("off")
    rng = np.random.default_rng(3)
    neg = np.c_[rng.normal(18, 5, 16), rng.normal(36, 6, 16)]
    pos = np.c_[rng.normal(54, 5, 16), rng.normal(38, 5, 16)]
    ax.add_patch(Polygon([[33, 20], [41, 26], [41, 56], [33, 50]], closed=True, fc=FAINT, ec=CTRL, lw=0.5))
    ax.plot(neg[:, 0], neg[:, 1], "o", ms=2.6, mfc="white", mec=CTRL, mew=0.6)
    ax.plot(pos[:, 0], pos[:, 1], "o", ms=2.6, color=INK, mec="white", mew=0.3)
    ax.add_patch(FancyArrowPatch((30, 38), (76, 38), arrowstyle="-|>,head_length=3,head_width=1.4", color=RED,
                                 lw=0.9, mutation_scale=1))
    ax.text(77.5, 38, "$\\mathbf{r}$", fontsize=7, color=RED, va="center")
    h = (60, 50)
    ax.plot(*h, "o", ms=3.4, color=INK, zorder=5)
    ax.text(h[0] + 1.5, h[1] + 1.2, "$\\mathbf{h}$", fontsize=6.5)
    ax.add_patch(FancyArrowPatch(h, (38.5, 50), arrowstyle="-|>,head_length=2.4,head_width=1.1", color=RED,
                                 lw=0.7, mutation_scale=1, shrinkB=0))
    ax.plot(37.6, 50, "o", ms=3.0, mfc="white", mec=RED, mew=0.7, zorder=5)
    ax.text(36, 58, "$\\mathbf{h}\\cdot\\mathbf{r}=0$", fontsize=5.8, color=INK2, ha="center")
    ax.add_patch(FancyArrowPatch((62, 26), (74, 26), arrowstyle="-|>,head_length=2.4,head_width=1.1", color=RED,
                                 lw=0.7, mutation_scale=1))
    ax.text(68, 22.5, "$+c\\,\\mathbf{r}$", fontsize=5.8, color=RED, ha="center")
    ax.plot(2, 60, "o", ms=2.6, mfc="white", mec=CTRL, mew=0.6); ax.text(4, 60, "non-suicidal", fontsize=5.8, va="center")
    ax.plot(2, 56, "o", ms=2.6, color=INK); ax.text(4, 56, "suicidal", fontsize=5.8, va="center")
    y = 15.0
    for name, eq, what in [("ablate", "$\\mathbf{h}\\leftarrow\\mathbf{h}-(\\mathbf{h}\\cdot\\mathbf{r})\\,\\mathbf{r}$",
                            "onto the plane: is the direction needed? vs nulls with equal damage"),
                           ("steer", "$\\mathbf{h}\\leftarrow\\mathbf{h}+c\\,\\mathbf{r}$",
                            "along r: does it move only the suicide answer? vs unrelated questions")]:
        ax.text(2, y, name, fontsize=6.2, color=RED, fontweight="bold")
        ax.text(12, y, eq, fontsize=6.2)
        ax.text(30, y, what, fontsize=5.8, color=INK2)
        y -= 5.5
    ax.text(2, 2.2, "Two of 4096 dimensions are drawn; $\\mathbf{r}$ is a unit vector.", fontsize=5.6, color=INK2,
            style="italic")
    save(fig, "figA1_geometry")


if __name__ == "__main__":
    fig_heads(); fig_localization(); fig_transfer(); fig_geometry()
    print("ok")
