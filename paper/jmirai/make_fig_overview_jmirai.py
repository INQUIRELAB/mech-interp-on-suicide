"""Figure 1 (overview) for the JMIR AI manuscript, in the register of the npj overview
(../make_fig_overview.py): Arial, millimetre coordinates on a 180 mm canvas, red only for the
suicide direction and the edits made to it.

A single panel, drawn as a left-to-right pipeline: what the model is asked, where the answer
forms, and what is done to it. Results are shown in Figures 2-6, not previewed here.
Usage: python make_fig_overview_jmirai.py            (writes figures/fig1_overview.*)
"""
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(os.path.abspath(__file__)).parent    # resolve() fails on the Z: network drive
OUT = HERE / "figures"

INK = "#1d1f22"
INK2 = "#454a50"
LINE = "#9aa0a6"
PALE = "#eceef0"
MID = "#c3c8cd"
RED = "#b0383d"
REDF = "#d9898d"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

W, Y0, Y1 = 180.0, 65.5, 116.0                      # panel occupies y = Y0..Y1 mm
fig = plt.figure(figsize=(W / 25.4, (Y1 - Y0) / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(Y0, Y1); ax.axis("off")

S_PANEL, S_HEAD, S_BODY, S_SMALL = 8, 7, 6.5, 6


def t(x, y, s, size=S_BODY, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def arrow(x0, y0, x1, y1, color=INK, lw=0.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=2.2,head_width=1.1",
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=1))


def hline(x0, x1, y, color=INK, lw=0.6, **kw):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", **kw)


# =============================================================================
# the screening question, the model, and the edits: icon, thick labelled arrow, icon;
# boxed node labels; no title bar (the caption names the figure)
# =============================================================================


def flow(x0, x1, y, above, below, color=INK):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.55",
                lw=2.4, color=color, shrinkA=0, shrinkB=0, mutation_scale=4))
    xm = (x0 + x1) / 2
    for i, s in enumerate(above):
        t(xm, y + 3.0 + 3.0 * (len(above) - 1 - i), s, S_SMALL, INK, ha="center")
    for i, s in enumerate(below):
        t(xm, y - 5.2 - 3.0 * i, s, S_SMALL, INK, ha="center")


def node_label(xc, y, s, color=INK):
    tx = t(xc, y, s, S_BODY, color, ha="center", va="center", zorder=6)
    fig.canvas.draw()
    bb = tx.get_window_extent().transformed(ax.transData.inverted())
    ax.add_patch(Rectangle((bb.x0 - 1.4, y - 2.4), bb.width + 2.8, 4.8, fc="white", ec=color, lw=0.7, zorder=5))


YC, YL = 104.0, 90.5          # icon centre line, node-label line

# -- 1 post icon: page with folded corner and text lines ---------------------------------
PX, PW, PH = 7.0, 11.0, 14.0
py0 = YC - PH / 2
ax.fill([PX, PX + PW - 3, PX + PW, PX + PW, PX], [py0, py0, py0 + 3, py0 + PH, py0 + PH],
        fc="white", ec=INK, lw=1.0, zorder=3)
ax.plot([PX + PW - 3, PX + PW - 3, PX + PW], [py0, py0 + 3, py0 + 3], color=INK, lw=1.0, zorder=3)
for k, ln in enumerate([7.5, 6.0, 7.5, 4.5]):
    hline(PX + 1.8, PX + 1.8 + ln, py0 + PH - 3.0 - 2.6 * k, color=INK2, lw=0.9, zorder=4)
node_label(PX + PW / 2, YL, "Post")

# -- 2 model: 32 layers, 13-19 highlighted ----------------------------------------------------
flow(21.5, 44.5, YC, ["Does this text show signs", "of suicidal ideation? Yes/No"],
     ["score = logit(Yes) − logit(No)", "at the final token"])
X0, PITCH, SW, SH = 47.5, 0.86, 0.56, 16.0
xl = [X0 + i * PITCH for i in range(32)]
for i, x in enumerate(xl):
    if 13 <= i <= 19:           # the late step in which the answer reaches the final token (Llama)
        fc, ec = REDF, RED
    else:
        fc, ec = PALE, LINE
    ax.add_patch(Rectangle((x, YC - SH / 2), SW, SH, fc=fc, ec=ec, lw=0.3, zorder=3))
t((xl[13] + xl[19] + SW) / 2, YC + SH / 2 + 1.0, "13–19", S_SMALL, RED, weight="bold", ha="center")
MXC = (xl[0] + xl[31] + SW) / 2
node_label(MXC, YL, "Llama-3.1-8B, 32 layers")

# -- 3 final-token state h with direction r ---------------------------------------------------
flow(77.5, 100.5, YC, ["final token"], ["the answer forms", "in one late step"])
NC, CW, CH = 9, 3.0, 1.7
HX = 103.0
hy0 = YC - NC * CH / 2
for k in range(NC):
    red = k in (4, 5)
    ax.add_patch(Rectangle((HX, hy0 + k * CH), CW, CH * 0.62, fc=REDF if red else MID,
                           ec=RED if red else "none", lw=0.5, zorder=3))
t(HX + CW + 1.2, hy0 + 5 * CH, "$\mathbf{r}$", S_BODY, RED, va="center")
node_label(HX + CW / 2 + 1, YL, "$\mathbf{h}$, 4096 numbers")

# -- 4 edits to r (layers differ by analysis; Multimedia Appendix 1) ------------------------------
flow(111.5, 134.5, YC, ["swap · ablate · steer r", "inside the model"],
     ["r = mean(suicidal) −", "mean(non-suicidal)"], color=RED)
# icon: Yes/No logit bars, before (outline) and after (filled)
BX = 138.0
for k, (lab, before, after) in enumerate([("Yes", 9.0, 3.5), ("No", 3.5, 8.0)]):
    yb = YC + 2.2 - k * 5.4
    t(BX, yb + 1.2, lab, S_SMALL, INK, va="center")
    ax.add_patch(Rectangle((BX + 5.0, yb), before, 2.4, fc="white", ec=LINE, lw=0.6, zorder=3))
    ax.add_patch(Rectangle((BX + 5.0, yb + 0.5), after, 1.4, fc=RED, ec="none", zorder=4))
node_label(BX + 7.0, YL, "Change in answer")

# -- controls, test sets and other models -------------------------------------------------------
t(134.0, 84.0, "each edit vs a control:", S_SMALL, INK2)
t(134.0, 81.0, "equal-KL null directions, other", S_SMALL, INK2)
t(134.0, 78.0, "tasks' directions, always-No questions", S_SMALL, INK2)
t(0, 78.0, "Test sets", S_SMALL, weight="bold")
t(0, 75.0, "Suicide-Watch-232K (benchmark) · SWMH, r/SuicideWatch vs r/depression ·", S_SMALL, INK2)
t(0, 72.0, "UMD, 245 users, expert-rated risk · 64 minimal pairs (same words, other meaning)", S_SMALL, INK2)
t(0, 67.6, "Also Qwen2.5-7B, Mistral-7B, Gemma-2-9B (16-bit, one forward pass) · text baselines: "
           "TF-IDF classifier, explicit-term lexicon", S_SMALL, INK2)

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig1_overview.pdf")
fig.savefig(OUT / "fig1_overview.png", dpi=600)
print("ok")
