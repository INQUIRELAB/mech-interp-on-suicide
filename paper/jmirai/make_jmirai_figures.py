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
from matplotlib.patches import Patch
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


# ---------------------------------------------------------------- Figures 2, 3 and 5 (plain style)
# Arial at the manuscript's 11 pt, matplotlib's default colours (one per model), drawn at the printed
# text width (6.5 in) and saved without cropping, so the text prints at its true size.
PLAIN = {
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.pad": 3, "ytick.major.pad": 3,
    "axes.labelpad": 4, "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
    "text.color": "black", "axes.labelcolor": "black", "lines.linewidth": 1.0,
}
COLOR = {"llama": "#1f77b4", "qwen": "#ff7f0e", "mistral": "#2ca02c", "gemma": "#d62728"}
ORDER = ["llama", "qwen", "mistral", "gemma"]
OFFSET = dict(zip(ORDER, [-0.27, -0.09, 0.09, 0.27]))
TEXT_W = 6.5


def ci_dot(ax, x, v, lo, hi, color, label=None):
    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o", ms=5, color=color, elinewidth=1.0,
                capsize=2.5, capthick=1.0, label=label, zorder=3)


def tfidf_ref(ax, v, lo, hi):
    ax.axhspan(lo, hi, color="0.88", lw=0, zorder=0)
    ax.axhline(v, color="black", lw=1.0, ls="--", zorder=1)


def title(ax, letter, text):
    ax.set_title(rf"$\mathbf{{{letter}}}$   {text}", loc="left", pad=6)


def save_plain(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600)
    plt.close(fig)


def fig_benchmark_vs_matched():
    with plt.rc_context(PLAIN):
        fig = plt.figure(figsize=(TEXT_W, 6.2), layout="constrained")
        rows = fig.add_gridspec(2, 1)
        t = rows[0].subgridspec(1, 2, width_ratios=[1.1, 3])
        b = rows[1].subgridspec(1, 2, width_ratios=[2.3, 1])
        axa, axb = fig.add_subplot(t[0]), fig.add_subplot(t[1])
        axc, axl = fig.add_subplot(b[0]), fig.add_subplot(b[1])

        # A: the standard benchmark (Suicide-Watch-232K), Llama only, 1000 posts per class
        g = load("audit/suicide/llama_3_1_8b_instruct__seed0.json")["gate"]
        for x, v, ci, c in [(0, g["behavioral_auc"], g["behavioral_ci95"], COLOR["llama"]),
                            (1, g["tfidf_auc"], g["tfidf_ci95"], "black")]:
            ci_dot(ax=axa, x=x, v=v, lo=ci[0], hi=ci[1], color=c)
        axa.set_xticks([0, 1], ["Llama-3.1-8B", "TF-IDF"], rotation=30, ha="right", rotation_mode="anchor")
        axa.set_xlim(-0.6, 1.6)
        axa.set_ylim(0.93, 0.97); axa.set_yticks([0.93, 0.94, 0.95, 0.96, 0.97])
        axa.set_ylabel("AUC")
        title(axa, "A", "Suicide-Watch-232K")

        # B: SWMH (register-matched), AUC per prompt, full posts for both systems
        variants = ["base", "neg_flip", "clinical", "contrast", "escalate", "fewshot", "rating"]
        labels = ["Base", "Negated", "Clinical definition", "Contrast", "Triage", "Few-shot", "1-5 rating"]
        files = {model_key(f.name): f for f in (D / "paper2/prompt_invariance_full").glob("suicide_depression__*.json")}
        first = load(files["llama"].relative_to(D))
        tfidf_ref(axb, first["tfidf_auc"], *first["tfidf_ci95"])
        for k in ORDER:
            a = load(files[k].relative_to(D))
            for i, t_ in enumerate(variants):
                lo, hi = a["variants"][t_]["ci95"]
                ci_dot(axb, i + OFFSET[k], a["variants"][t_]["auc"], lo, hi, COLOR[k])
        axb.set_xticks(range(len(variants)), labels, rotation=30, ha="right", rotation_mode="anchor")
        axb.set_xlim(-0.6, len(variants) - 0.4)
        axb.set_ylim(0.70, 0.90); axb.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
        axb.set_ylabel("AUC")
        title(axb, "B", "SWMH (r/SuicideWatch vs r/depression)")

        # C: SWMH, sensitivity at 90% specificity
        cats = ["Base", "Clinical\ndefinition", "Contrast", "TF-IDF +\ncontrast"]
        npzs = {model_key(f.name): f for f in (D / "paper2/prompt_invariance_full").glob("*.npz")}
        z0 = np.load(npzs["llama"]); y0 = z0["y"].astype(int)
        tv = sens_at_spec(y0, z0["tfidf"]); tci = boot_ci(y0, lambda i: sens_at_spec(y0[i], z0["tfidf"][i]))
        tfidf_ref(axc, 100 * tv, 100 * tci[0], 100 * tci[1])
        for k in ORDER:
            z = np.load(npzs[k]); y = z["y"].astype(int)
            assert (y == y0).all()
            st = stack_cv(y, z["tfidf"], z["contrast"])
            for i, s in enumerate([z["base"], z["clinical"], z["contrast"], st]):
                v = sens_at_spec(y, s); ci = boot_ci(y, lambda j: sens_at_spec(y[j], s[j]))
                ci_dot(axc, i + OFFSET[k], 100 * v, 100 * ci[0], 100 * ci[1], COLOR[k])
        axc.set_xticks(range(len(cats)), cats)
        axc.set_xlim(-0.6, len(cats) - 0.4)
        axc.set_ylim(30, 70); axc.set_yticks([30, 40, 50, 60, 70])
        axc.set_ylabel("Sensitivity at 90%\nspecificity (%)")
        title(axc, "C", "SWMH (r/SuicideWatch vs r/depression)")

        # Shared key
        axl.axis("off")
        h = [plt.Line2D([], [], marker="o", ms=5, ls="", color=COLOR[k]) for k in ORDER]
        h += [plt.Line2D([], [], color="black", ls="--", lw=1.0),
              Patch(color="0.88", lw=0)]
        axl.legend(h, [MODELS[k] for k in ORDER] + ["TF-IDF", "TF-IDF 95% CI"], loc="center left",
                   frameon=False, handlelength=1.8)
        save_plain(fig, "fig2_benchmark_vs_matched")


def fig_umd():
    with plt.rc_context(PLAIN):
        fig, ax = plt.subplots(figsize=(TEXT_W, 3.7), layout="constrained")
        readouts = [("proj_site", "Direction\nprojection"), ("behav", "Model\nanswer"),
                    ("probe_cv", "Linear\nprobe"), ("tfidf_cv_seen", "TF-IDF"),
                    ("tfidf232_seen", "TF-IDF,\n232K-trained"), ("lexicon_seen", "Explicit-term\nlexicon")]
        files = {model_key(f.name): f for f in (D / "umd_v2").glob("*.json")}
        ax.axhline(0.5, color="0.5", lw=1.0, ls=":", zorder=1)
        ax.axvline(2.5, color="0.6", lw=0.8, zorder=1)
        for k in ORDER:
            a = load(files[k].relative_to(D))["scores"]
            for i, (key, _) in enumerate(readouts):
                lo, hi = a[key]["auc_hi_lo_ci95"]
                ci_dot(ax, i + OFFSET[k], a[key]["auc_hi_lo"], lo, hi, COLOR[k])
        ax.set_xticks(range(len(readouts)), [r[1] for r in readouts])
        ax.set_xlim(-0.6, len(readouts) - 0.4)
        ax.set_ylim(0.4, 1.0); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel("AUC, high vs low\nexpert-rated risk")
        ax.text(1.0, 0.97, "From the LLM", ha="center", va="top", fontsize=10)
        ax.text(4.0, 0.97, "Text baselines", ha="center", va="top", fontsize=10)
        ax.text(-0.55, 0.505, "Chance", ha="left", va="bottom", fontsize=10, color="0.4")
        h = [plt.Line2D([], [], marker="o", ms=5, ls="", color=COLOR[k]) for k in ORDER]
        fig.legend(h, [MODELS[k] for k in ORDER], loc="outside upper center", ncol=4, frameon=False,
                   handletextpad=0.3, columnspacing=1.5)
        ax.set_title("UMD Reddit Suicidality Dataset, 245 users", loc="left", pad=6)
        save_plain(fig, "fig3_umd_expert_risk")


# ---------------------------------------------------------------- Figure 3
def fig_minpairs():
    conds = [("genuine_intent", "Genuine intent"), ("counterfactual", "Denial (“I would never”)"),
             ("third_person", "Third person"), ("recovery_past", "Past or recovered"),
             ("fiction", "Fiction"), ("topical", "Topical mention")]
    files = {model_key(f.name): f for f in (D / "intent_minpairs").glob("*.json")}
    with plt.rc_context(PLAIN):
        fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, 5.2), layout="constrained")
        for ax, prompt, letter, name in zip(axes, ["base", "clinical"], "AB",
                                            ["Base prompt", "Clinical-definition prompt"]):
            M = np.array([[load(files[k].relative_to(D))["yes_rate"][c][prompt] for k in ORDER] for c, _ in conds])
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=10,
                            color="white" if M[i, j] > 0.6 else "black")
            ax.set_xticks(range(len(ORDER)), [MODELS[k] for k in ORDER])
            ax.set_yticks(range(len(conds)), [c[1] for c in conds])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            title(ax, letter, name)
        # values are printed in every cell, so no colour bar
        save_plain(fig, "fig4_minimal_pairs")


# ---------------------------------------------------------------- Figure 5 (plain style)
def fig_mechanism():
    with plt.rc_context(PLAIN):
        fig = plt.figure(figsize=(TEXT_W, 7.4), layout="constrained")
        rows = fig.add_gridspec(2, 1, height_ratios=[1, 1.05])
        top = rows[0].subgridspec(1, 2, width_ratios=[1.45, 1])
        ax, bx = fig.add_subplot(top[0]), fig.add_subplot(top[1])
        cx = fig.add_subplot(rows[1])

        # A: remaining answer gap after the suicide direction or a damage-matched null direction is projected out
        fams = [("isotropic", "Isotropic"), ("pca", "Top PC"), ("sign_random", "Sign-\nrandom"),
                ("shuffled_label", "Shuffled\nlabel")]
        order = ["llama", "qwen", "mistral"]
        off3 = dict(zip(order, [-0.22, 0.0, 0.22]))
        files = {model_key(f.name): f for f in (D / "audit_klnec/suicide").glob("*.json")}
        ax.axhline(1, color="0.5", lw=1.0, ls=":", zorder=1)
        handles = []
        for k in order:
            n = load(files[k].relative_to(D))["necessary"]; km = n["kl_matched"]; base = n["baseline_ld_gap"]
            v = km["target_ld_gap"] / base
            ax.plot(off3[k], v, "o", ms=5, color=COLOR[k], zorder=3)
            for i, (fk, _) in enumerate(fams):
                draws = np.array([dd["ld_gap"] for dd in km["nulls"][fk]["draws"]]) / base
                m = draws.mean()
                ax.errorbar(i + 1 + off3[k], m, yerr=[[m - draws.min()], [draws.max() - m]], fmt="o", ms=5,
                            color=COLOR[k], elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3)
            handles.append(plt.Line2D([], [], marker="o", ms=5, ls="", color=COLOR[k],
                                      label=f"{MODELS[k]} (KL {km['target_kl']:.2f})"))
        ax.set_xticks(range(5), ["Suicide\ndirection"] + [f[1] for f in fams])
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 2.0); ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_ylabel("Remaining answer gap\n(fraction of baseline)")
        ax.text(-0.45, 1.02, "No change", ha="left", va="bottom", fontsize=10, color="0.4")
        ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
        title(ax, "A", "Necessity against null directions")

        # B: steering at equal KL; ratio of the shift on always-"no" questions to the shift on the target question
        order = ["llama", "qwen", "mistral", "gemma"]
        files = {model_key(f.name): f for f in (D / "steering_topic").glob("*.json")}
        for j, k in enumerate(order):
            a = load(files[k].relative_to(D)); ratios = []; sui = None
            for cname, cv in a["concepts"].items():
                rec = next(r for r in cv["variants"]["orig"] if abs(r["kappa"] - 0.1) < 1e-9)
                ratios.append(rec["nontopic_shift"] / rec["on_shift"])
                if cname == "suicide":
                    sui = ratios[-1]
            others = [r for r, c in zip(ratios, a["concepts"]) if c != "suicide"]
            bx.plot(j + np.linspace(-0.15, 0.15, len(others)), others, "o", ms=5, mfc="white", mec="0.5",
                    mew=1.0, ls="", zorder=2)
            bx.plot([j - 0.28, j + 0.28], [np.median(ratios)] * 2, color="black", lw=1.5, zorder=3)
            bx.plot(j, sui, "o", ms=6, color=COLOR[k], zorder=4)
        bx.axhline(1, color="0.5", lw=1.0, ls=":", zorder=1)
        bx.set_yscale("log"); bx.set_ylim(0.05, 100)
        bx.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
        bx.set_xticks(range(len(order)), [MODELS[k] for k in order], rotation=30, ha="right",
                      rotation_mode="anchor")
        bx.set_xlim(-0.6, len(order) - 0.4)
        bx.set_ylabel("Off-target shift / target shift")
        h = [plt.Line2D([], [], marker="o", ms=6, ls="", color="0.3", label="Suicide"),
             plt.Line2D([], [], marker="o", ms=5, ls="", mfc="white", mec="0.5", mew=1.0, label="Other concepts"),
             plt.Line2D([], [], color="black", lw=1.5, label="Median")]
        bx.legend(handles=h, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
        title(bx, "B", "Steering at equal KL")

        # C: off-target necessity; AUC of the suicide task and mean AUC of four unrelated tasks
        other = ["sentiment", "toxicity", "sports", "truth"]
        conds = [("none", None, "No\nablation"), ("suicide", "all", "Suicide\ndirection,\nevery layer"),
                 ("suicide", "upper", "Suicide\ndirection,\nupper half"),
                 ("others", "upper", "Unrelated tasks'\ndirections,\nupper half (lowest)"),
                 ("answer_polarity", "upper", "Answer\npolarity,\nupper half")]
        files = {model_key(f.name): f for f in (D / "offtarget_necessity").glob("*.json")}
        cx.axhline(0.5, color="0.5", lw=1.0, ls=":", zorder=1)
        for k in order:
            a = load(files[k].relative_to(D))
            for i, (d, scope, _) in enumerate(conds):
                x = i + OFFSET[k]
                if d == "none":
                    s, o = a["baseline"]["suicide"]["auc"], np.mean([a["baseline"][t]["auc"] for t in other])
                elif d == "others":
                    s = min(a["ablations"][f"{t}|{scope}"]["tasks"]["suicide"]["auc"] for t in other); o = None
                else:
                    r = a["ablations"][f"{d}|{scope}"]["tasks"]
                    s, o = r["suicide"]["auc"], np.mean([r[t]["auc"] for t in other])
                cx.plot(x, s, "o", ms=6, color=COLOR[k], zorder=4)
                if o is not None:
                    cx.plot(x, o, "o", ms=6, mfc="white", mec=COLOR[k], mew=1.2, zorder=3)
        cx.set_xticks(range(len(conds)), [c[2] for c in conds])
        cx.set_xlim(-0.5, len(conds) - 0.5)
        cx.set_ylim(0.15, 1.02); cx.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cx.set_ylabel("AUC after the edit")
        cx.text(-0.45, 0.505, "Chance", ha="left", va="bottom", fontsize=10, color="0.4")
        h = [plt.Line2D([], [], marker="o", ms=6, ls="", color=COLOR[k], label=MODELS[k]) for k in order]
        h += [plt.Line2D([], [], marker="o", ms=6, ls="", color="0.3", label="Suicide task"),
              plt.Line2D([], [], marker="o", ms=6, ls="", mfc="white", mec="0.3", mew=1.2,
                         label="Unrelated tasks (mean)")]
        cx.legend(handles=h, loc="lower left", ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2,
                  borderaxespad=0.2)
        title(cx, "C", "Off-target necessity")
        save_plain(fig, "fig5_mechanism")


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
