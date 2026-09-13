#!/usr/bin/env python
"""UMD expert-severity validation of the suicidality direction.

The severity gradient in sc2_gradient.py uses text-type PROXIES
(neutral/depression/implicit/explicit). UMD (CLPsych-2019 expert subset) gives
REAL clinician ordinal risk labels per user: a (none) < b (low) < c (moderate)
< d (severe). This is the strongest possible severity test: does projection onto
the 232K suicidality direction track expert-assigned risk?

Tests:
  1. SEVERITY MONOTONICITY — mean projection onto the 07 direction rises across
     a<b<c<d, with bootstrap 95% CIs, plus Spearman rho (projection vs ordinal
     severity over all users) and the behavioral logit-diff for comparison.
  2. TRANSFER AUC — does the 07 direction separate high-risk (c+d) from no/low
     risk (a+b) on UMD with NO retraining? (decode transfer to a new dataset)
  3. cosine between the 07 direction and UMD's own high-vs-low diff-of-means.

Output: results/sc2_binary/umd_severity.json + umd_severity.png
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.models import ModelLoader
from src.analysis_utils import PROMPT_TEMPLATES, ensure_dir, load_umd_with_severity

N_DIR = 100      # per class to fit the 07 reference direction
LAYER = 16
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())
LEVELS = {0: "a (none)", 1: "b (low)", 2: "c (moderate)", 3: "d (severe)"}


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


# ---- 07 reference pools ----
ds07 = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
s07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "suicide"], N_DIR)
n07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "non-suicide"], N_DIR)

# ---- UMD expert users with ordinal severity ----
umd = load_umd_with_severity(source="expert")
umd_texts = [r["text"] for r in umd]
umd_sev = np.array([r["severity"] for r in umd])
print(f"UMD: {len(umd_texts)} users, severity counts "
      f"{[int((umd_sev==s).sum()) for s in range(4)]}")

# ---- model ----
loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]


def chat_ids(text):
    tmpl = PROMPT_TEMPLATES["standard"]
    prefix, suffix = tmpl.split("{text}")
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = prefix + tok.decode(body) + suffix
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def feat(text):
    out = model(chat_ids(text), output_hidden_states=True)
    lg = out.logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item(), out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()


def collect(texts, tag):
    lds, hs = [], []
    for i, t in enumerate(texts):
        ld, h = feat(t); lds.append(ld); hs.append(h)
        if (i + 1) % 50 == 0:
            print(f"    {tag} {i+1}/{len(texts)}")
    return np.array(lds), np.stack(hs)


def boot_mean(x, B=2000):
    if len(x) == 0:
        return (float("nan"),) * 3
    ms = [rng.choice(x, len(x), replace=True).mean() for _ in range(B)]
    return float(np.mean(x)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def boot_auc(y, s, B=2000):
    aucs, idx = [], np.arange(len(y))
    for _ in range(B):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        aucs.append(roc_auc_score(y[bi], s[bi]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


print("Extracting 07 reference direction...")
_, h07s = collect(s07, "07-sui"); _, h07n = collect(n07, "07-non")
dir07 = h07s.mean(0) - h07n.mean(0); dir07 = dir07 / np.linalg.norm(dir07)

print("Extracting UMD activations...")
umd_ld, umd_h = collect(umd_texts, "umd")
proj = umd_h @ dir07                       # projection onto 07 suicidality direction

# ---- (1) severity monotonicity ----
per_level_proj = {s: boot_mean(proj[umd_sev == s]) for s in range(4)}
per_level_ld = {s: boot_mean(umd_ld[umd_sev == s]) for s in range(4)}
rho_proj, p_proj = spearmanr(umd_sev, proj)
rho_ld, p_ld = spearmanr(umd_sev, umd_ld)

# ---- (1b) volume control -------------------------------------------------
# UMD annotates the USER, not the post, and each record here is one user's
# posts concatenated. Higher-risk users post more, so aggregated length is
# itself correlated with severity: a projection that rises with severity could
# be tracking verbosity. Partial out length (Spearman on ranks, residualized)
# and report the posts/length-vs-severity correlations alongside.
n_posts = np.array([r.get("n_posts", np.nan) for r in umd], dtype=float)
n_chars = np.array([len(r["text"]) for r in umd], dtype=float)


def partial_spearman(x, y, z):
    """rho(x, y | z) via rank residuals; nan-safe on z."""
    from scipy.stats import rankdata
    ok = ~np.isnan(z)
    rx, ry, rz = (rankdata(v[ok]) for v in (x, y, z))
    def resid(a):
        A = np.vstack([rz, np.ones_like(rz)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return spearmanr(resid(rx), resid(ry))


rho_len, p_len = spearmanr(n_chars, umd_sev)
rho_np, p_np = spearmanr(n_posts[~np.isnan(n_posts)], umd_sev[~np.isnan(n_posts)])
rho_proj_len, _ = spearmanr(n_chars, proj)
rho_partial_chars, p_partial_chars = partial_spearman(proj, umd_sev.astype(float), n_chars)
rho_partial_posts, p_partial_posts = partial_spearman(proj, umd_sev.astype(float), n_posts)
volume_control = {
    "n_users": int(len(umd)),
    "n_unique_users": int(len({r["user_id"] for r in umd})),
    "spearman_chars_vs_severity": [float(rho_len), float(p_len)],
    "spearman_posts_vs_severity": [float(rho_np), float(p_np)],
    "spearman_chars_vs_projection": [float(rho_proj_len)],
    "partial_rho_proj_severity_given_chars": [float(rho_partial_chars), float(p_partial_chars)],
    "partial_rho_proj_severity_given_posts": [float(rho_partial_posts), float(p_partial_posts)],
    "frac_at_truncation_cap": float((n_chars >= 3990).mean()),
}
order = [per_level_proj[s][0] for s in range(4)]
mono = all(order[i] <= order[i + 1] for i in range(3))

# ---- (2) transfer AUC: high-risk (c,d) vs low/none (a,b) ----
hi = umd_sev >= 2
y = hi.astype(int)
transfer_auc = boot_auc(y, proj)
behav_auc = boot_auc(y, umd_ld)
# (3) cosine of 07 dir vs UMD's own hi-vs-lo direction
own = umd_h[hi].mean(0) - umd_h[~hi].mean(0); own = own / (np.linalg.norm(own) + 1e-8)
cosine = float(own @ dir07)

print("\n" + "=" * 72)
print("UMD EXPERT-SEVERITY VALIDATION (projection onto 07 suicidality direction)")
print("=" * 72)
print(f"{'expert level':16} {'n':>4} {'mean proj':>10} {'95% CI':>20} {'behav LD':>9}")
for s in range(4):
    m, lo, hi_ = per_level_proj[s]; n = int((umd_sev == s).sum()); bl = per_level_ld[s][0]
    print(f"{LEVELS[s]:16} {n:>4} {m:10.2f}  [{lo:6.2f},{hi_:6.2f}]  {bl:8.2f}")
print("-" * 72)
print(f"  projection monotonic a<=b<=c<=d : {mono}")
print(f"  Spearman rho (proj vs severity) : {rho_proj:+.3f}  (p={p_proj:.1e})")
print(f"  Spearman rho (behav LD vs sev)  : {rho_ld:+.3f}  (p={p_ld:.1e})")
print(f"  TRANSFER AUC hi(c,d) vs lo(a,b) : {transfer_auc[0]:.3f}  [{transfer_auc[1]:.3f},{transfer_auc[2]:.3f}]")
print(f"  behavioral AUC hi vs lo         : {behav_auc[0]:.3f}  [{behav_auc[1]:.3f},{behav_auc[2]:.3f}]")
print(f"  cos(07 dir, UMD hi-lo dir)      : {cosine:+.3f}")
print("-" * 72)
print("  volume control (annotations are per USER; posts are longitudinal)")
print(f"    unique users / records        : {volume_control['n_unique_users']}/{volume_control['n_users']}")
print(f"    rho(chars, severity)          : {rho_len:+.3f}  (p={p_len:.1e})")
print(f"    rho(posts, severity)          : {rho_np:+.3f}  (p={p_np:.1e})")
print(f"    rho(chars, projection)        : {rho_proj_len:+.3f}")
print(f"    rho(proj, sev | chars)        : {rho_partial_chars:+.3f}  (p={p_partial_chars:.1e})")
print(f"    rho(proj, sev | posts)        : {rho_partial_posts:+.3f}  (p={p_partial_posts:.1e})")
print(f"    users at 4000-char cap        : {volume_control['frac_at_truncation_cap']:.1%}")
print("=" * 72)
verdict = rho_proj > 0.2 and transfer_auc[1] > 0.5
print(f"VERDICT: {'EXPERT-VALIDATED severity axis (proj tracks clinician risk)' if verdict else 'weak — see numbers'}")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xs = list(range(4))
means = [per_level_proj[s][0] for s in xs]
los = [per_level_proj[s][0] - per_level_proj[s][1] for s in xs]
his = [per_level_proj[s][2] - per_level_proj[s][0] for s in xs]
ax[0].errorbar(xs, means, yerr=[los, his], fmt="o-", color="C3", capsize=4)
ax[0].set_xticks(xs); ax[0].set_xticklabels([LEVELS[s] for s in xs], rotation=20, ha="right")
ax[0].set_ylabel("mean projection onto 07 direction")
ax[0].set_title(f"UMD expert severity (Spearman rho={rho_proj:+.2f})")
ax[0].grid(alpha=0.3)
# scatter proj vs severity (jittered)
jit = umd_sev + rng.normal(0, 0.06, len(umd_sev))
ax[1].scatter(jit, proj, s=12, alpha=0.5, color="C0")
ax[1].set_xticks(xs); ax[1].set_xticklabels([s for s in "abcd"])
ax[1].set_xlabel("expert risk level"); ax[1].set_ylabel("projection")
ax[1].set_title(f"transfer AUC (c,d vs a,b) = {transfer_auc[0]:.2f}")
ax[1].grid(alpha=0.3)
fig.tight_layout()
ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
fig.savefig(str(PROJECT_ROOT / "results" / "sc2_binary" / "umd_severity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

with open(PROJECT_ROOT / "results" / "sc2_binary" / "umd_severity.json", "w") as f:
    json.dump({"dataset": "04_UMD_Reddit_Suicidality (expert)", "layer": LAYER,
               "n_users": len(umd_texts), "reference_dataset": "07_Suicide_Watch_232K",
               "severity_counts": {LEVELS[s]: int((umd_sev == s).sum()) for s in range(4)},
               "projection_by_level": {LEVELS[s]: per_level_proj[s] for s in range(4)},
               "behavioral_ld_by_level": {LEVELS[s]: per_level_ld[s] for s in range(4)},
               "spearman_proj_vs_severity": {"rho": float(rho_proj), "p": float(p_proj)},
               "spearman_behav_vs_severity": {"rho": float(rho_ld), "p": float(p_ld)},
               "monotonic": bool(mono),
               "transfer_auc_hi_vs_lo": transfer_auc, "behavioral_auc_hi_vs_lo": behav_auc,
               "cosine_07_vs_umd_hilo": cosine,
               "volume_control": volume_control,
               "per_user": [{"user_id": r["user_id"], "severity": int(s),
                             "n_posts": r.get("n_posts"), "n_chars": len(r["text"]),
                             "projection": float(p), "behavioral_ld": float(b)}
                            for r, s, p, b in zip(umd, umd_sev, proj, umd_ld)],
               "expert_validated": bool(verdict)}, f, indent=2)
print("Saved results/sc2_binary/umd_severity.json + umd_severity.png")
