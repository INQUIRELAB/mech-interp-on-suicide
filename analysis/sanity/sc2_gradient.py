#!/usr/bin/env python
"""Depression specificity + severity gradient.

Two questions the binary tests left open:
  (1) Is the layer-16 circuit suicidality-SPECIFIC or just a general-distress
      detector? Test: SWMH SuicideWatch vs SWMH depression (both distressed).
  (2) Does the circuit encode a SEVERITY GRADIENT? Project four levels onto the
      07 suicidality direction:
        neutral  <  depression  <  implicit ideation (DeepSuiMind)  <  explicit suicide
      If the projection rises monotonically, the direction recovers the
      severity axis the original project wanted.
Bootstrap 95% CIs on every headline number.
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.repro import get_seed, set_seed
set_seed()

import torch, numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.models import ModelLoader
from src.analysis_utils import (
    PROMPT_TEMPLATES, ensure_dir, load_dataset_by_category, get_text,
)

N = 100
LAYER = 16
MAXLEN = 256
device = "cuda"
rng = np.random.default_rng(get_seed())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())

NEUTRAL = [
    "I had a wonderful day at the park with my friends today. The weather was perfect and we played for hours.",
    "Just finished baking cookies for the bake sale. They turned out great and the kitchen smells amazing.",
    "Our team won the soccer match this weekend. Everyone played well and we celebrated with pizza.",
    "So excited for my vacation next month. We're going to the beach and I've started planning activities.",
    "The new restaurant downtown is fantastic. I tried their pasta and it was the best meal in ages.",
    "My garden is finally blooming. The tomatoes are coming in nicely and the roses look beautiful.",
    "Had a productive day at work and finished my project early. Treating myself to a movie tonight.",
    "Learning guitar has been rewarding. I can finally play my favorite song all the way through.",
    "We adopted a puppy last week and he's the sweetest. He loves chasing his tail around the room.",
    "The hiking trip was incredible. We reached the summit just in time to watch the sunrise.",
    "I got a promotion at work today. All the hard work this year finally paid off and I'm thrilled.",
    "Spent the afternoon reading a great novel in the sun. Sometimes the simple things are best.",
    "My sister is getting married next spring and I'll be the maid of honor. So happy for her.",
    "The farmers market had the freshest produce today. I picked up berries and homemade bread.",
    "Fixed up my bike and went for a long ride along the river. Felt great to be outside.",
    "We had a lovely family dinner tonight. Mom made her famous lasagna and we laughed for hours.",
    "Planted herbs on my balcony and they're thriving. Fresh basil for cooking is the best.",
    "The concert last night was amazing. The band played my favorite songs and the crowd was electric.",
    "Started a yoga class this morning and I feel refreshed. Looking forward to going again.",
    "We repainted the living room and it looks brand new. So happy with how it turned out.",
    "Went apple picking today and came home with a huge basket. Apple pie is on the menu.",
    "My friend surprised me with museum tickets. The new exhibit was absolutely fascinating.",
    "The morning jog through the park was perfect. Crisp air, birds singing, a great start.",
    "Tried a new board game tonight and it was hilarious. Game nights with friends are the best.",
    "Baked fresh bread for the first time and it worked. The whole house smells wonderful.",
    "Spent the day volunteering at the animal shelter. The dogs were happy to go on walks.",
    "The road trip was full of great memories. We sang along to music and stopped at overlooks.",
    "Learned to make sushi at home and it came out delicious. Dinner was a big success.",
    "We watched the meteor shower from the rooftop. Counted over twenty shooting stars.",
    "Had brunch with old college friends today. It felt like no time had passed at all.",
]


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]
    rng.shuffle(out)
    return out[:n]


ds07 = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
s07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "suicide"], N)
n07 = clean([t for t, l in zip(ds07["text"], ds07["label"]) if l == "non-suicide"], N)
sw = load_dataset_by_category(dataset_name="swmh")
sw_sui = clean([get_text(x) for x in sw["Death-Me"]], N)        # SuicideWatch
sw_dep = clean([get_text(x) for x in sw["Death-Not-Me"]], N)    # depression
dsm = load_dataset_by_category(dataset_name="deepsuimind")
implicit = clean([get_text(x) for x in dsm["Death-Me"]], N)     # implicit ideation
neutral = NEUTRAL

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


def boot_auc(y, s, B=2000):
    aucs = []
    idx = np.arange(len(y))
    for _ in range(B):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        aucs.append(roc_auc_score(y[bi], s[bi]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def boot_mean(x, B=2000):
    ms = [rng.choice(x, len(x), replace=True).mean() for _ in range(B)]
    return float(np.mean(x)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


print("Collecting groups...")
ld07s, h07s = collect(s07, "07-sui"); ld07n, h07n = collect(n07, "07-non")
ldsws, hsws = collect(sw_sui, "sw-sui"); ldswd, hswd = collect(sw_dep, "sw-dep")
ldimp, himp = collect(implicit, "implicit"); ldneu, hneu = collect(neutral, "neutral")

dir07 = h07s.mean(0) - h07n.mean(0); dir07 = dir07 / np.linalg.norm(dir07)

# ---- (1) depression specificity: SWMH suicide vs depression ----
y = np.array([1] * len(hsws) + [0] * len(hswd))
X = np.concatenate([hsws, hswd])
proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                          X, y, cv=cv, method="predict_proba")[:, 1]
probe_auc = boot_auc(y, proba)
behav_auc = boot_auc(y, np.concatenate([ldsws if False else ldsws, ldswd]))
proj_auc = boot_auc(y, X @ dir07)

# ---- (2) severity gradient: projection onto dir07 ----
groups = [("neutral", hneu), ("depression", hswd), ("implicit_ideation", himp),
          ("explicit_suicide", hsws)]
grad = {name: boot_mean(H @ dir07) for name, H in groups}
behav = {"neutral": boot_mean(ldneu), "depression": boot_mean(ldswd),
         "implicit_ideation": boot_mean(ldimp), "explicit_suicide": boot_mean(ldsws)}

print("\n" + "=" * 70)
print("(1) DEPRESSION SPECIFICITY — SWMH SuicideWatch vs depression (n=%d/class)" % N)
print("=" * 70)
print(f"  probe AUC:      {probe_auc[0]:.3f}  [{probe_auc[1]:.3f}, {probe_auc[2]:.3f}]")
print(f"  07-dir AUC:     {proj_auc[0]:.3f}  [{proj_auc[1]:.3f}, {proj_auc[2]:.3f}]")
print(f"  behavioral AUC: {behav_auc[0]:.3f}  [{behav_auc[1]:.3f}, {behav_auc[2]:.3f}]")
spec = probe_auc[1] > 0.6
print(f"  -> {'SUICIDALITY-SPECIFIC (separates suicide from depression)' if spec else 'GENERAL-DISTRESS (cannot separate suicide from depression)'}")

print("\n" + "=" * 70)
print("(2) SEVERITY GRADIENT — projection onto 07 suicidality direction")
print("=" * 70)
print(f"{'level':20} {'mean proj':>10} {'95% CI':>22} {'behav LD':>10}")
for name, _ in groups:
    m, lo, hi = grad[name]; bm = behav[name][0]
    print(f"{name:20} {m:10.2f}   [{lo:7.2f}, {hi:7.2f}]   {bm:8.2f}")
order = [grad[n][0] for n, _ in groups]
mono = all(order[i] <= order[i + 1] for i in range(len(order) - 1))
print(f"  -> projection monotonic increasing: {mono}")

ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
with open(PROJECT_ROOT / "results" / "sc2_binary" / "gradient.json", "w") as f:
    json.dump({"n": N, "layer": LAYER,
               "depression_specificity": {"probe_auc": probe_auc, "dir07_auc": proj_auc,
                                          "behavioral_auc": behav_auc, "specific": bool(spec)},
               "gradient_projection": {k: grad[k] for k, _ in groups},
               "gradient_behavioral_ld": {k: behav[k] for k in behav},
               "monotonic": bool(mono)}, f, indent=2)
print("\nSaved results/sc2_binary/gradient.json")
