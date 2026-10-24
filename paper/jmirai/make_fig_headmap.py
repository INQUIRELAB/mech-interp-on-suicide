"""Head map figure: where the suicide direction is written, and what removing those heads does to each post.

  figures/fig_headmap   A: all 1,024 Llama-3.1-8B attention heads at their (layer, head) position, sized by
                        mean patching IE; the top-20 heads write into the layer-16 residual stream.
                        B: projection onto r at layer 16 for each of the 200 test posts, unedited vs after
                        mean-ablating the top-20 heads or a layer-matched random set of 20.
                        C: per-post change (ablated - unedited).

Sources. Head IE: liu_headtohead_metallama_v2_ie.npz (60 primary pairs, fp16, commit 0e9e39f2); the
ranking is the one revision_controls.ipynb falls back to (argsort of mean IE over primary pairs).
Per-post projections: data/heads/controls_per_example.npz (suicidality, mean ablation).
Run:  python make_fig_headmap.py
"""
import os
import pathlib
from pathlib import Path

# Z: is a mapped drive that rejects realpath(); make_jmirai_figures calls Path(__file__).resolve() on import
pathlib.Path.resolve = lambda self, strict=False: Path(os.path.abspath(self))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_jmirai_figures as base   # rcParams, colours, helpers
from make_jmirai_figures import INK, INK2, CTRL, CTRL_LT, FAINT, RED, FULL, D, panel, save

IE_NPZ = (Path(__file__).resolve().parents[2]
          / "results/sc2_binary/liu_headtohead_metallama_v2_ie.npz")   # from sc2_liu_headtohead_v2.py
NL = NH = 32
K = 20
SITE = 16          # hidden_states[16]: written by layers 0-15


def fig_headmap():
    z = np.load(IE_NPZ)
    ie = z["ie"][z["primary"].astype(bool)].mean(0)                    # (layer, head)
    rank = np.argsort(-ie.ravel())[:K]
    top = [(int(i) // NH, int(i) % NH) for i in rank]

    c = np.load(D / "heads/controls_per_example.npz")
    p = "suicidality__mean__"
    y, b, P = c[p + "y"], c[p + "base_proj"], c[p + "proj"]
    sets = list(c["sets"])
    i_top = sets.index(f"top{K}")
    i_rand = [i for i, s in enumerate(sets) if s == f"rand{K}"]
    gap = lambda v: v[y == 1].mean() - v[y == 0].mean()
    gaps = np.array([gap(P[:, i]) for i in i_rand])
    i_med = i_rand[int(np.argsort(gaps)[len(gaps) // 2])]               # median random set

    order = np.r_[np.where(y == 0)[0][np.argsort(b[y == 0])], np.where(y == 1)[0][np.argsort(b[y == 1])]]
    x = np.arange(len(order))

    fig = plt.figure(figsize=(FULL, 82 / 25.4))
    ax = fig.add_axes([0.045, 0.10, 0.36, 0.78])
    bx = fig.add_axes([0.53, 0.43, 0.45, 0.45])
    cx = fig.add_axes([0.53, 0.10, 0.45, 0.22], sharex=bx)

    # ---- A: head map
    ax.axhspan(7.5, 15.5, color=FAINT, lw=0, zorder=0)
    pos = np.clip(ie, 0, None)
    L, H = np.mgrid[0:NL, 0:NH]
    ax.scatter(H.ravel(), L.ravel(), s=1.2, color=CTRL_LT, lw=0, zorder=1)
    m = pos.ravel() > 0.05
    ax.scatter(H.ravel()[m], L.ravel()[m], s=2 + 70 * pos.ravel()[m], facecolor="white", edgecolor=CTRL,
               lw=0.45, zorder=2)
    ax.axhline(SITE - 0.5, color=RED, lw=0.9, zorder=3)
    ax.text(31.6, SITE - 0.1, r"layer-16 residual stream, where $\mathbf{r}$ is read", color=RED,
            fontsize=5.8, ha="right", va="bottom", zorder=5, bbox=dict(fc="white", ec="none", pad=0.3))
    for k, (l, h) in enumerate(top):
        s = 2 + 70 * ie[l, h]
        if l < SITE:
            ax.annotate("", xy=(h, SITE - 0.62), xytext=(h, l + 0.35), zorder=3,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=0.45,
                                        color=RED, shrinkA=0, shrinkB=0, alpha=0.75))
            ax.scatter(h, l, s=s, color=RED, lw=0, zorder=4)
        else:
            ax.scatter(h, l, s=s, facecolor="white", edgecolor=RED, lw=0.6, zorder=4)
    for k, (l, h) in enumerate(top[:7]):
        ax.text(h + 0.75, l - 0.15, f"L{l}H{h}", fontsize=5.2, color=INK, va="center", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85))
    above = [(l, h) for l, h in top if l >= SITE]
    for j, (l, h) in enumerate(above):
        ax.annotate(f"{len(above)} of the top {K} sit above layer 16\nand cannot write to it" if j == 0 else "",
                    xy=(h, l), xytext=(24.5, 21.2), fontsize=5.4, color=INK2, ha="right", va="center",
                    linespacing=1.1, arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkB=2.5,
                                                     relpos=(1, 0.5)))
    ax.text(32.4, 11.5, "layers 8–15", rotation=-90, fontsize=5.4, color=INK2, ha="left", va="center")
    ax.set_xlim(-0.8, 31.8); ax.set_ylim(-0.8, 31.8)
    ax.set_xticks([0, 8, 16, 24, 31]); ax.set_yticks([0, 8, 16, 24, 31])
    ax.set_xlabel("head index"); ax.set_ylabel("layer", labelpad=6)
    ax.set_aspect("equal")
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 2))
    ax.text(0, -5.2, "dot area: mean share of the logit gap restored by patching that one head (60 pairs)",
            fontsize=5.2, color=INK2, transform=ax.transData, va="top", clip_on=False)

    # ---- B: per-post projection
    bx.plot(x, b[order], color=INK, lw=0.9, zorder=3)
    bx.plot(x, P[order, i_med], color=CTRL, lw=0.7, ls=(0, (2.2, 1.2)), zorder=4)
    bx.plot(x, P[order, i_top], color=RED, lw=0.7, zorder=5)
    bx.axvline(99.5, color=INK2, lw=0.4, ls=":")
    bx.axhline(0, color=CTRL_LT, lw=0.4, zorder=1)
    bx.text(49.5, 2.95, "non-suicidal posts", ha="center", fontsize=5.8, color=INK2)
    bx.text(149.5, 2.95, "suicidal posts", ha="center", fontsize=5.8, color=INK2)
    bx.text(3, 1.55, "unedited", fontsize=5.8, color=INK)
    bx.text(3, 1.15, f"random {K} heads ablated", fontsize=5.8, color=CTRL)
    bx.text(3, 0.75, f"top {K} heads ablated", fontsize=5.8, color=RED)
    bx.text(197, -2.05, f"class gap: unedited {gap(b):.2f}, random {gap(P[:, i_med]):.2f}, "
            f"top {K} {gap(P[:, i_top]):.2f}", fontsize=5.4, color=INK2, ha="right")
    bx.set_ylim(-2.4, 3.3)
    bx.set_ylabel(r"$\mathbf{h}\cdot\mathbf{r}$ at layer 16")
    plt.setp(bx.get_xticklabels(), visible=False)

    # ---- C: change per post
    d_rand = np.stack([P[order, i] - b[order] for i in i_rand])
    cx.fill_between(x, d_rand.min(0), d_rand.max(0), color=FAINT, lw=0, step="mid", zorder=1)
    cx.plot(x, P[order, i_med] - b[order], color=CTRL, lw=0.6, zorder=3)
    cx.plot(x, P[order, i_top] - b[order], color=RED, lw=0.7, zorder=4)
    cx.axhline(0, color=INK, lw=0.4, zorder=2)
    cx.axvline(99.5, color=INK2, lw=0.4, ls=":")
    cx.text(197, d_rand.max() + 0.12, f"band: all {len(i_rand)} random sets", fontsize=5.4, color=INK2,
            ha="right", va="bottom")
    cx.set_ylabel("ablated − unedited")
    cx.set_xlabel("post (sorted by class, then by unedited projection)")
    cx.set_xlim(-1, 200); cx.set_xticks([0, 50, 100, 150, 199])

    panel(fig, ax, "A", f"{K} heads, ranked by patching, write the direction")
    panel(fig, bx, "B", "Each post's projection onto the direction")
    panel(fig, cx, "C", "Change per post")
    save(fig, "fig_headmap")


if __name__ == "__main__":
    fig_headmap()
