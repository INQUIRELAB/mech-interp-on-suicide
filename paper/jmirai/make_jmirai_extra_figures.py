"""Figures restored from the npj manuscript for the JMIR AI version, redrawn in the same house
style as make_jmirai_figures.py and limited to analyses that survived the corrected audit.

  figures/fig6_heads           main text: attention heads write the direction, but not only for suicide
  figures/figA2_localization   Appendix 1, Figure A1: where the answer forms, and in how many directions
  figures/figA3_transfer       Appendix 1, Figure A2: the 232K direction is used on other corpora
Figures A1 and A2 use the plain style of Figures 2-5. The old geometry sketch (figA1_geometry) was dropped on
2026-10-25: main-text Figure 1 now shows the edits.

Sources. Localization: data/audit (probe per layer, n=1000/class) and data/gsa/rank_* (state
swaps). Transfer: data/transfer/transfer_dedup.json (near-duplicates of 232K posts removed).
Head sets: data/heads (revision_controls.ipynb rerun on yorha, 2026-10-16, Llama-3.1-8B-Instruct
at commit 0e9e39f2, fp16); these reproduce the values earlier typed from ../npjai_supp.tex.
Run:  python make_jmirai_extra_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np

from make_jmirai_figures import INK, INK2, CTRL, FAINT, RED, FULL, MODELS, MARK, D, panel, dot, save  # also sets rcParams
from make_jmirai_figures import PLAIN, TEXT_W, COLOR, title, save_plain

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
    # mean-ablation: share of each task's logit-difference gap left, top-k heads and layer-matched random heads
    tasks = [("suicidality", "Suicide"), ("sentiment", "Sentiment (SST-2)"), ("topic_sports", "Sports (AG News)")]
    left = {lab: ([0] + list(k), [100.0] + [100 * spec[f"{t}/mean"]["by_k"][x]["gap_top"] / spec[f"{t}/mean"]["baseline_gap"]
                                            for x in ks]) for t, lab in tasks}
    rand = np.array([[100 * spec[f"{t}/mean"]["by_k"][x]["gap_rand_mean"] / spec[f"{t}/mean"]["baseline_gap"] for x in ks]
                     for t, _ in tasks])
    rand_lo, rand_hi = np.r_[100.0, rand.min(0)], np.r_[100.0, rand.max(0)]
    C = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    with plt.rc_context(PLAIN):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_W, 3.2), layout="constrained")
        ax.plot(k, gap_restored, "-s", ms=5, color=C[0], label="Patching: logit gap")
        ax.plot(k, proj_restored, "-o", ms=5, color=C[1], label="Patching: layer-16 projection")
        ax.plot(k, sep_left, "--o", ms=5, color=C[2], mfc="white", mew=1.2, label="Ablation: direction left")
        ax.set_xticks(k); ax.set_xlim(0, 43)
        ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Top-$k$ heads (ranked by patching)")
        ax.set_ylabel("% of effect")
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.64), frameon=False, handlelength=1.8,
                  borderaxespad=0.2)
        title(ax, "A", "Patching and ablation")

        xr = np.r_[0, k]
        bx.fill_between(xr, rand_lo, rand_hi, color="0.85", lw=0, zorder=0, label="Random head sets")
        for (name, (xs, ys)), c in zip(left.items(), C):
            bx.plot(xs, ys, "-o", ms=5, color=c, label=name)
        bx.set_xticks([0, 5, 10, 20, 40]); bx.set_xlim(-1.5, 43)
        bx.set_ylim(0, 105); bx.set_yticks([0, 25, 50, 75, 100])
        bx.set_xlabel("Top-$k$ heads mean-ablated")
        bx.set_ylabel("% of logit gap left")
        h, l = bx.get_legend_handles_labels()
        bx.legend(h[1:] + h[:1], l[1:] + l[:1], loc="center right", frameon=False, handlelength=2.2,
                  borderaxespad=0.2)
        title(bx, "B", "Same ablation, three tasks")
        save_plain(fig, "fig6_heads")



# ---------------------------------------------------------------- Appendix figures
def fig_localization():
    """Figure A1 (plain style): probe by layer, state swap by layer, single-direction swaps at the site."""
    pr = load("audit/suicide/llama_3_1_8b_instruct__seed0.json")["decodable"]
    files = {"llama": "gsa/rank_suicide_meta_llama_3_1_8b_instruct__seed0.json",
             "qwen": "gsa/rank_suicide_qwen2_5_7b_instruct__seed0.json",
             "mistral": "gsa/rank_suicide_mistral_7b_instruct_v0_3__seed0.json"}
    G = {m: load(f) for m, f in files.items()}
    with plt.rc_context(PLAIN):
        fig = plt.figure(figsize=(TEXT_W, 5.6), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95])
        ax, bx, cx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

        L = [p["hidden_index"] for p in pr["probe_per_layer"]]
        A = [p["auc"] for p in pr["probe_per_layer"]]
        ax.plot(L, A, "-", color=COLOR["llama"])
        ax.plot(pr["site"], pr["site_auc"], "o", ms=5, color=COLOR["llama"])
        ax.annotate(f"layer {pr['site']}: {pr['site_auc']:.3f}", xy=(pr["site"], pr["site_auc"]),
                    xytext=(pr["site"], 0.93), fontsize=10, ha="center", va="top")
        ax.axhline(0.5, color="0.5", lw=1.0, ls=":")
        ax.text(32, 0.51, "Chance", fontsize=10, color="0.4", ha="right", va="bottom")
        ax.set_xlim(-1, 33); ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_ylim(0.45, 1.02); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel("Layer"); ax.set_ylabel("Linear-probe AUC")
        title(ax, "A", "Probe, Llama-3.1-8B, 232K")

        for m, g in G.items():
            s = g["scan_full_swap_share"]
            xs = np.array(sorted(map(int, s))); ys = np.array([s[str(x)] for x in xs])
            bx.plot(xs, ys, "-o", ms=4, color=COLOR[m], label=MODELS[m])
        bx.axhline(0.8, color="0.5", lw=1.0, ls="--")
        bx.text(1, 0.82, "0.8: site", fontsize=10, color="0.4", va="bottom")
        bx.set_xlim(0, 33); bx.set_xticks([0, 8, 16, 24, 32])
        bx.set_ylim(-0.08, 1.12); bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        bx.set_xlabel("Layer of the state swap"); bx.set_ylabel("Share of class gap moved")
        bx.legend(loc="lower right", frameon=False, handlelength=1.5)
        title(bx, "B", "State swap, 232K")

        rows = [("dom", "Difference of means"), ("white_top1", "Whitened, top 1"),
                ("grad_mean", "Mean readout gradient"), ("random1", "Random direction"),
                ("proj_top1", "Projection, top 1")]
        off = dict(zip(G, [-0.2, 0.0, 0.2]))
        cx.axvline(1.0, color="black", lw=1.0, ls="--")
        cx.text(1.02, -0.55, "Full-state swap", fontsize=10, va="center")
        for m in G:
            for i, (key, _) in enumerate(rows):
                cx.plot(G[m]["checks"][key]["share_of_full"], i + off[m], "o", ms=5, color=COLOR[m])
        cx.set_yticks(range(len(rows)), [r[1] for r in rows])
        cx.set_ylim(len(rows) - 0.5, -0.8); cx.tick_params(axis="y", length=0)
        cx.set_xlim(-0.05, 1.9); cx.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        cx.set_xlabel("Gap moved by one direction / gap moved by the full-state swap")
        cx.text(1.88, 3.0, "Above 1: more than the whole\nstate carries (an illusion)", fontsize=10, ha="right",
                va="center")
        title(cx, "C", "One-direction swaps at the site, 232K")
        save_plain(fig, "figA2_localization")


def fig_transfer():
    """Figure A2 (plain style): ablating the 232K direction on two other corpora."""
    d = json.load(open(D / "transfer/transfer_dedup.json", encoding="utf8"))
    rows = [("Kaggle-raw", d["SDCNL"]), ("SWMH, r/SuicideWatch\nvs other forums", d["SWMH"])]
    with plt.rc_context(PLAIN):
        fig, ax = plt.subplots(figsize=(TEXT_W, 2.7), layout="constrained")
        for y, (_, r) in enumerate(rows):
            c = r["causal"]
            ax.annotate("", xy=(c["ablate_232K_dirs"], y), xytext=(c["baseline"], y),
                        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.2, shrinkA=4, shrinkB=0))
            ax.plot(c["baseline"], y, "o", ms=6, color="black", zorder=4)
            ax.plot(c["random"], y + 0.25, "o", ms=6, mfc="white", mec="0.45", mew=1.2, zorder=4)
            ax.text(c["baseline"] + 0.006, y - 0.17, f"{c['baseline']:.3f}", fontsize=10, va="center")
            ax.text(c["ablate_232K_dirs"] - 0.006, y, f"{c['ablate_232K_dirs']:.3f}", fontsize=10,
                    ha="right", va="center", color="#d62728")
            lo, hi = c["drop_ci95"]
            ax.text((c["baseline"] + c["ablate_232K_dirs"]) / 2, y - 0.12,
                    f"Drop {c['baseline'] - c['ablate_232K_dirs']:.3f} (95% CI {lo:.3f}–{hi:.3f})",
                    ha="center", va="bottom", fontsize=10, color="#d62728")
        ax.set_yticks([0, 1], [r[0] for r in rows])
        ax.set_ylim(1.5, -0.55); ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.6, 1.0); ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel("Model-answer AUC, 80 posts per class")
        h = [plt.Line2D([], [], marker="o", ms=6, ls="", color="black"),
             plt.Line2D([], [], color="#d62728", lw=1.2, marker=">", ms=5),
             plt.Line2D([], [], marker="o", ms=6, ls="", mfc="white", mec="0.45", mew=1.2)]
        fig.legend(h, ["Unedited", "232K direction ablated", "Random direction ablated"],
                   loc="outside upper center", ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.5)
        ax.set_title("Llama-3.1-8B; near-duplicates of 232K posts removed", loc="left", pad=6)
        save_plain(fig, "figA3_transfer")


if __name__ == "__main__":
    fig_heads(); fig_localization(); fig_transfer()
    print("ok")
