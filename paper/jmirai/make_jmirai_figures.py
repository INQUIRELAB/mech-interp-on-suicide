"""Figures for the JMIR AI manuscript, rendered only from the result files in ./data.

Run:  python make_jmirai_figures.py            (writes figures/fig*.png and .pdf)
Every plotted value is read from a JSON/NPZ result file; nothing is typed in by hand.

House style is the one used for the npj figures (../make_figures_npj.py): Arial at
6-7 pt, bold panel letters (uppercase, as JMIR captions cite "Figure 1A"), 0.5 pt axes, direct labels instead of legend
boxes, one accent hue. Red is reserved for the suicidality direction and the edits
made to it; grey is a baseline or matched control; everything else is ink. Models are
told apart by marker shape, not by colour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
OUT = HERE / "figures"

MM = 1 / 25.4
FULL = 180 * MM

INK = "#1d1f22"
INK2 = "#55595f"
CTRL = "#8f959c"
CTRL_LT = "#c9cdd1"
FAINT = "#eef0f2"
RED = "#b0383d"

MODELS = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B", "mistral": "Mistral-7B", "gemma": "Gemma-2-9B"}
# marker, filled?  (one ink hue; shape carries model identity)
MARK = {"llama": ("o", True), "qwen": ("s", False), "mistral": ("^", True), "gemma": ("D", False)}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8, "axes.labelpad": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "lines.linewidth": 0.9, "lines.solid_capstyle": "butt",
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "figure.dpi": 150,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def load(p):
    return json.load(open(D / p, encoding="utf8"))


def model_key(name):
    n = name.lower()
    return next(k for k in MODELS if k in n)


def sens_at_spec(y, s, spec=0.90):
    """Share of positives above the 90th percentile of negative-class scores."""
    return float((s[y == 1] > np.quantile(s[y == 0], spec)).mean())


def stack_cv(y, a, b, k=5, seed=0):
    """5-fold cross-validated logistic regression on [TF-IDF, LLM] scores."""
    X = np.c_[a, b]; X = (X - X.mean(0)) / X.std(0)
    idx = np.random.default_rng(seed).permutation(len(y)); out = np.empty(len(y))
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        Xt = np.c_[np.ones(len(tr)), X[tr]]; w = np.zeros(3)
        for _ in range(3000):
            w -= 0.1 * Xt.T @ (1 / (1 + np.exp(-Xt @ w)) - y[tr]) / len(tr)
        out[te] = np.c_[np.ones(len(te)), X[te]] @ w
    return out


def boot_ci(y, fn, B=1000, seed=1):
    """Class-stratified bootstrap percentile interval."""
    rng = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [fn(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def panel(fig, ax, letter, title=""):
    """Bold panel letter flush with the left edge of the tick labels, short title after it."""
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    top = ax.get_position().y1 + 0.035
    t = fig.text(bb.x0, top, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")
    if title:
        fig.canvas.draw()
        tb = t.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(tb.x1 + 0.008, top + 0.002, title, fontsize=6.5, ha="left", va="bottom", color=INK)


def dot(ax, x, y, key, color=INK, ms=3.2, **kw):
    m, filled = MARK[key]
    ax.plot(x, y, m, ms=ms, color=color, mfc=color if filled else "white", mec=color,
            mew=0.7, zorder=5, **kw)


def key_row(ax, keys, y, x0, dx, color=INK):
    """One-row marker key in axes coordinates, drawn above the plot."""
    for i, k in enumerate(keys):
        m, filled = MARK[k]
        x = x0 + i * dx
        ax.plot(x, y, m, ms=3.0, color=color, mfc=color if filled else "white", mec=color, mew=0.7,
                transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.025, y, MODELS[k], transform=ax.transAxes, va="center", fontsize=5.8)


def hrows(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, linespacing=1.1)
    ax.set_ylim(len(labels) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig_benchmark_vs_matched():
    fig = plt.figure(figsize=(FULL, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.25, 1.0], wspace=0.62)
    axa, axb, axc = (fig.add_subplot(gs[i]) for i in range(3))

    # a: the standard benchmark (232K), Llama, n=1000/class
    g = load("audit/suicide/llama_3_1_8b_instruct__seed0.json")["gate"]
    rows = [("model\nanswer", g["behavioral_auc"], g["behavioral_ci95"], INK),
            ("TF-IDF", g["tfidf_auc"], g["tfidf_ci95"], CTRL)]
    for i, (_, v, ci, c) in enumerate(rows):
        axa.hlines(i, ci[0], ci[1], color=c, lw=0.9)
        axa.plot(v, i, "o", color=c, ms=3.4, mec="white", mew=0.5, zorder=4)
        axa.text(v, i - 0.13, f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, color=c)
    hrows(axa, [r[0] for r in rows])
    axa.set_ylim(1.45, -0.75)
    axa.set_xlim(0.93, 0.97); axa.set_xticks([0.93, 0.95, 0.97])
    axa.set_xlabel("ROC-AUC")
    axa.text(0.5, -0.17, "Llama-3.1-8B", transform=axa.transAxes, ha="center", va="top",
             fontsize=5.6, color=INK2)

    # b: register-matched SWMH, AUC per prompt, 3 models
    variants = ["base", "neg_flip", "clinical", "contrast", "escalate", "fewshot", "rating"]
    labels = ["base", "negated", "clinical definition", "contrast", "triage", "few-shot", "1-5 rating"]
    files = sorted((D / "paper2/prompt_invariance_full").glob("suicide_depression__*.json"))
    # Full posts for both systems (4096-token budget; one evaluation post cut). TF-IDF is fit and scored on
    # full posts in every run, so it is drawn once: band = 95% CI, line = AUC.
    first = load(files[0].relative_to(D)); tfa_full = first["tfidf_auc"]; t_lo, t_hi = first["tfidf_ci95"]
    axb.axvspan(t_lo, t_hi, color=FAINT, lw=0, zorder=0)
    axb.axvline(tfa_full, color=CTRL, lw=0.8, zorder=1)
    axb.text(tfa_full - 0.002, -0.75, f"TF-IDF {tfa_full:.3f}", ha="right", va="bottom", fontsize=5.8, color=INK2)
    keys = []
    for j, f in enumerate(files):
        a = load(f.relative_to(D)); k = model_key(a["model"]); keys.append(k)
        y = np.arange(len(variants)) + (j - (len(files) - 1) / 2) * 0.72 / len(files)
        for yi, t in zip(y, variants):
            v = a["variants"][t]["auc"]; lo, hi = a["variants"][t]["ci95"]
            axb.hlines(yi, lo, hi, color=INK, lw=0.5, zorder=3)
            dot(axb, v, yi, k)
    hrows(axb, labels)
    axb.set_ylim(len(variants) - 0.5, -0.85)
    axb.set_xlim(0.68, 0.88); axb.set_xticks([0.70, 0.75, 0.80, 0.85])
    axb.set_xlabel("ROC-AUC, suicide vs depression (SWMH)")

    # c: sensitivity at 90% specificity (models with per-post scores)
    npzs = sorted((D / "paper2/prompt_invariance_full").glob("*.npz"))
    cats = ["base", "clinical", "contrast", "TF-IDF +\ncontrast (stacked)"]
    # TF-IDF drawn once, as in B (the same full-post fit in every run)
    z0 = np.load(npzs[0]); y0 = z0["y"].astype(int)
    tv = sens_at_spec(y0, z0["tfidf"]); ci = boot_ci(y0, lambda i: sens_at_spec(y0[i], z0["tfidf"][i]))
    axc.axvspan(ci[0], ci[1], color=FAINT, lw=0, zorder=0)
    axc.axvline(tv, color=CTRL, lw=0.8, zorder=1)
    axc.text(tv - 0.004, -0.7, f"TF-IDF {tv:.0%}", ha="right", va="bottom", fontsize=5.8, color=INK2)
    for j, f in enumerate(npzs):
        z = np.load(f); y = z["y"].astype(int); k = model_key(f.name)
        assert (y == y0).all()
        st = stack_cv(y, z["tfidf"], z["contrast"])
        scores = [z["base"], z["clinical"], z["contrast"], st]
        yy = np.arange(len(cats)) + (j - (len(npzs) - 1) / 2) * 0.72 / len(npzs)
        for yi, s in zip(yy, scores):
            v = sens_at_spec(y, s); ci = boot_ci(y, lambda i: sens_at_spec(y[i], s[i]))
            axc.hlines(yi, ci[0], ci[1], color=INK, lw=0.5, zorder=3)
            dot(axc, v, yi, k)
    hrows(axc, cats)
    axc.set_ylim(len(cats) - 0.5, -0.85)
    axc.set_xlim(0.3, 0.7); axc.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    axc.set_xticklabels(["30%", "40", "50", "60", "70%"])
    axc.set_xlabel("sensitivity at 90% specificity")

    panel(fig, axa, "A", "Suicide-Watch-232K")
    panel(fig, axb, "B", "SWMH, by prompt")
    panel(fig, axc, "C")
    kx = 0.30
    for k in ["llama", "qwen", "mistral", "gemma"]:
        m, filled = MARK[k]
        fig.add_artist(plt.Line2D([kx], [-0.075], marker=m, ms=3.0, color=INK, mfc=INK if filled else "white",
                                  mec=INK, mew=0.7, transform=fig.transFigure, ls=""))
        fig.text(kx + 0.012, -0.075, MODELS[k], va="center", fontsize=5.8)
        kx += 0.12
    save(fig, "fig2_benchmark_vs_matched")


# ---------------------------------------------------------------- Figure 2
def fig_umd():
    fig, ax = plt.subplots(figsize=(FULL * 0.66, 2.55))
    readouts = [("proj_site", "direction projection"), ("behav", "model answer"),
                ("probe_cv", "linear probe (in-domain CV)"), ("tfidf_cv_seen", "TF-IDF (in-domain CV)"),
                ("tfidf232_seen", "TF-IDF (trained on 232K)"), ("lexicon_seen", "explicit-term lexicon")]
    order = ["llama", "qwen", "mistral", "gemma"]
    files = {model_key(f.name): f for f in (D / "umd_v2").glob("*.json")}
    ax.axhline(2.5, color=CTRL_LT, lw=0.5, zorder=0)
    ax.text(0.51, 2.62, "text baselines (same truncated text)", ha="left", va="top", fontsize=5.2,
            color=INK2, style="italic")
    ax.axvline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0.505, -0.5, "chance", fontsize=5.6, color=INK2, va="top")
    for j, k in enumerate(order):
        a = load(files[k].relative_to(D))["scores"]
        y = np.arange(len(readouts)) + (j - 1.5) * 0.17
        for yi, (key, _) in zip(y, readouts):
            v = a[key]["auc_hi_lo"]; lo, hi = a[key]["auc_hi_lo_ci95"]
            c = RED if key == "proj_site" else (CTRL if key in ("tfidf_cv_seen", "tfidf232_seen", "lexicon_seen") else INK)
            ax.hlines(yi, lo, hi, color=c, lw=0.5, zorder=3)
            dot(ax, v, yi, k, color=c, ms=3.0)
    hrows(ax, [r[1] for r in readouts])
    ax.get_yticklabels()[0].set_color(RED)
    ax.set_ylim(len(readouts) - 0.45, -0.55)
    ax.set_xlim(0.4, 0.95); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("ROC-AUC, expert-rated high risk (c, d) vs low risk (a, b), UMD")
    # stacked key in the empty lower-right corner
    for i, k in enumerate(order):
        m, filled = MARK[k]
        ax.plot(0.79, 4.25 + i * 0.3, m, ms=2.8, color=INK, mfc=INK if filled else "white", mec=INK, mew=0.6)
        ax.text(0.805, 4.25 + i * 0.3, MODELS[k], va="center", fontsize=5.6)
    save(fig, "fig3_umd_expert_risk")


# ---------------------------------------------------------------- Figure 3
def fig_minpairs():
    order = ["llama", "qwen", "mistral", "gemma"]
    conds = [("genuine_intent", "genuine intent"), ("counterfactual", "denial (“I would never”)"),
             ("third_person", "third person"), ("recovery_past", "past / recovered"),
             ("fiction", "fiction"), ("topical", "topical mention")]
    files = {model_key(f.name): f for f in (D / "intent_minpairs").glob("*.json")}
    cmap = LinearSegmentedColormap.from_list("ink", ["#ffffff", "#c4c8cc", "#3a3e44"])
    fig = plt.figure(figsize=(FULL * 0.82, 2.1))
    gs = fig.add_gridspec(1, 2, wspace=0.06)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for ax, prompt in zip(axes, ["base", "clinical"]):
        M = np.array([[load(files[k].relative_to(D))["yes_rate"][c][prompt] for k in order] for c, _ in conds])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=5.8,
                        color="white" if M[i, j] > 0.55 else INK)
        ax.axhline(0.5, color=INK, lw=0.4, xmin=0, xmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([MODELS[k] for k in order])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([c[1] for c in conds] if ax is axes[0] else [])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    # values are printed in every cell, so no colour bar
    fig.canvas.draw()
    for ax, letter, title in zip(axes, "AB", ["base prompt", "clinical definition"]):
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        x0 = bb.x0 if ax is axes[1] else ax.get_position().x0 - 0.0
        t = fig.text(x0 if ax is axes[1] else bb.x0, bb.y1 + 0.02, letter, fontsize=8,
                     fontweight="bold", ha="left", va="bottom")
        fig.canvas.draw()
        tb = t.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(tb.x1 + 0.008, bb.y1 + 0.022, title, fontsize=6.5, ha="left", va="bottom")
    save(fig, "fig4_minimal_pairs")


# ---------------------------------------------------------------- Figure 4
def fig_mechanism():
    fig = plt.figure(figsize=(FULL, 4.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 0.82], wspace=0.32, hspace=0.78)
    ax = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[0, 1]); cx = fig.add_subplot(gs[1, :])
    fig_offtarget(cx)

    fams = [("isotropic", "isotropic"), ("pca", "top-PC"), ("sign_random", "sign-random"),
            ("shuffled_label", "shuffled-label")]
    order = ["llama", "qwen", "mistral"]
    files = {model_key(f.name): f for f in (D / "audit_klnec/suicide").glob("*.json")}
    rng = np.random.default_rng(0)
    for j, k in enumerate(order):
        n = load(files[k].relative_to(D))["necessary"]; km = n["kl_matched"]; base = n["baseline_ld_gap"]
        x0 = j * 6
        ax.bar(x0, km["target_ld_gap"] / base, color=RED, width=0.72, zorder=3)
        for i, (fk, _) in enumerate(fams):
            draws = np.array([dd["ld_gap"] for dd in km["nulls"][fk]["draws"]]) / base
            ax.bar(x0 + 1 + i, draws.mean(), color=CTRL_LT, width=0.72, lw=0, zorder=2)
            ax.plot(x0 + 1 + i + rng.uniform(-0.2, 0.2, len(draws)), draws, "o", ms=1.6, color=INK2,
                    mew=0, zorder=3)
        ax.text(x0 + 2.0, 1.86, MODELS[k], ha="center", va="bottom", fontsize=6.2)
        ax.text(x0 + 2.0, 1.80, f"edit KL {km['target_kl']:.2f}", ha="center", va="top", fontsize=5.6,
                color=INK2)
    ax.axhline(1, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(16.6, 1.015, "no change", ha="right", va="bottom", fontsize=5.6, color=INK2)
    ticks = [j * 6 + i for j in range(3) for i in range(5)]
    ax.set_xticks(ticks)
    ax.set_xticklabels((["suicide direction"] + [f[1] for f in fams]) * 3, rotation=55, ha="right",
                       rotation_mode="anchor", fontsize=5.6)
    for t in ax.get_xticklabels()[::5]:
        t.set_color(RED)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, 16.7)
    ax.set_ylim(0, 2.0); ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.spines["left"].set_bounds(0, 1.5)
    ax.set_ylabel("logit-difference gap remaining\n(fraction of baseline)", linespacing=1.1)

    order = ["llama", "mistral", "gemma", "qwen"]
    files = {model_key(f.name): f for f in (D / "steering_topic").glob("*.json")}
    for j, k in enumerate(order):
        a = load(files[k].relative_to(D)); ratios = []; sui = None
        for cname, cv in a["concepts"].items():
            rec = next(r for r in cv["variants"]["orig"] if abs(r["kappa"] - 0.1) < 1e-9)
            ratios.append(rec["nontopic_shift"] / rec["on_shift"])
            if cname == "suicide":
                sui = rec["nontopic_shift"] / rec["on_shift"]
        ratios = np.array(ratios)
        bx.plot(j + np.linspace(-0.14, 0.14, len(ratios)), np.clip(ratios, 0.05, None), "o", ms=2.8,
                mfc="white", mec=CTRL, mew=0.6, zorder=2)
        bx.plot([j - 0.24, j + 0.24], [np.median(ratios)] * 2, color=INK, lw=0.9, zorder=3)
        bx.plot(j, sui, "D", ms=3.6, color=RED, mec="white", mew=0.4, zorder=4)
    bx.axhline(1, color=INK2, lw=0.5, ls=(0, (3, 2)))
    bx.set_xticks(range(len(order)))
    bx.set_xticklabels([MODELS[k] for k in order], rotation=25, ha="right", rotation_mode="anchor")
    bx.tick_params(axis="x", length=0)
    bx.set_xlim(-0.6, len(order) - 0.4)
    bx.set_yscale("log"); bx.set_ylim(0.05, 10)
    bx.set_ylabel("shift on unrelated always-No questions\n/ shift on target", linespacing=1.1)
    # direct labels on the data instead of a key row
    q = order.index("qwen"); m_ = order.index("mistral")
    bx.annotate("suicide", xy=(q + 0.05, 0.17), xytext=(q - 0.55, 0.075), fontsize=5.6, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.4, shrinkA=0, shrinkB=2))
    bx.annotate("median of\n8 concepts", xy=(m_ + 0.24, 1.31), xytext=(m_ + 0.42, 5.5), fontsize=5.4,
                color=INK2, linespacing=1.0,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=1, shrinkB=1))
    bx.text(len(order) - 0.45, 1.06, "as much as\nthe target", fontsize=5.2, color=INK2, ha="right",
            va="bottom", linespacing=1.0)

    panel(fig, ax, "A")
    panel(fig, bx, "B", "steering, equal KL (0.1)")
    panel(fig, cx, "C", "off-target necessity")
    save(fig, "fig5_mechanism")


def fig_offtarget(cx):
    """AUC of the suicide task (red) and mean AUC of four unrelated yes/no tasks (grey) after a direction is projected
    out (data/offtarget_necessity), every layer or the upper half; for the unrelated tasks' own directions, the lowest
    suicide AUC of the four."""
    other = ["sentiment", "toxicity", "sports", "truth"]
    rows = [("none", None, "no ablation"), ("suicide", "all", "suicide direction, every layer"),
            ("suicide", "upper", "suicide direction, upper half"),
            ("others", "upper", "unrelated tasks' directions (lowest)"),
            ("answer_polarity", "upper", "answer polarity, upper half")]
    files = {model_key(f.name): f for f in (D / "offtarget_necessity").glob("*.json")}
    order = [k for k in ["llama", "qwen", "mistral", "gemma"] if k in files]
    for j, k in enumerate(order):
        a = load(files[k].relative_to(D))
        for i, (d, scope, _) in enumerate(rows):
            yi = i + (j - (len(order) - 1) / 2) * 0.16
            if d == "none":
                s, o = a["baseline"]["suicide"]["auc"], np.mean([a["baseline"][t]["auc"] for t in other])
            elif d == "others":
                s = min(a["ablations"][f"{x}|{scope}"]["tasks"]["suicide"]["auc"] for x in other); o = None
            else:
                r = a["ablations"][f"{d}|{scope}"]["tasks"]
                s, o = r["suicide"]["auc"], np.mean([r[t]["auc"] for t in other])
            dot(cx, s, yi, k, color=RED, ms=3.0)
            if o is not None:
                m, _ = MARK[k]
                cx.plot(o, yi, m, ms=3.0, color=CTRL, mfc="white", mec=CTRL, mew=0.7, zorder=4)
    hrows(cx, [r[2] for r in rows])
    cx.set_ylim(len(rows) - 0.45, -0.55)
    cx.axvline(0.5, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    cx.set_xlim(0.3, 1.02); cx.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cx.set_xlabel("ROC-AUC after the direction is projected out")
    for i, k in enumerate(order):                       # key in the empty lower-left part of the panel
        m, filled = MARK[k]
        cx.plot(0.315, 2.62 + 0.42 * i, m, ms=2.8, color=INK, mfc=INK if filled else "white", mec=INK, mew=0.6)
        cx.text(0.33, 2.62 + 0.42 * i, MODELS[k], va="center", fontsize=5.6)
    cx.text(0.52, 2.62, "red: suicide task", va="center", fontsize=5.6, color=RED)
    cx.text(0.52, 3.04, "grey: mean of sentiment, toxicity,", va="center", fontsize=5.6, color=CTRL)
    cx.text(0.52, 3.46, "sports and truth tasks", va="center", fontsize=5.6, color=CTRL)


if __name__ == "__main__":
    fig_benchmark_vs_matched(); fig_umd(); fig_minpairs(); fig_mechanism()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
