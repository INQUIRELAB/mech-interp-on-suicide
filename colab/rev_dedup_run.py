# Auto-extracted from nb_src.py: sections 0, 1, 7, 12, 12b only.

# ---- section 0 ----

# 0.1  Runtime + dependencies
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.56.2",
                    "accelerate>=0.33", "datasets>=2.19", "datasketch"], check=True)


# ---- section 0 ----

# 0.2  Paths, Drive, Hugging Face login, fixed settings
from pathlib import Path
import json, time, zipfile, copy
if IN_COLAB:
    from google.colab import drive, userdata
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    # Same Drive project as llama_all_phases_colab.ipynb
    DRIVE_PROJECT = Path("/content/drive/MyDrive/MI on suicide")
    assert DRIVE_PROJECT.exists(), f"{DRIVE_PROJECT} not found"
    BASE = DRIVE_PROJECT / "Dataset" / "Outputs"
    WORK = Path("/content/work")
    # Reuse the Llama weights already cached on Drive (no 16 GB re-download).
    # Must be set before transformers / huggingface_hub are imported.
    os.environ["HF_HOME"] = str(DRIVE_PROJECT / "hf_cache")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(DRIVE_PROJECT / "hf_cache" / "hub")
    try:
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
    BUNDLE_CANDIDATES = [DRIVE_PROJECT / "colab_data_bundle.zip",
                         DRIVE_PROJECT / "Dataset" / "Inputs" / "colab_data_bundle.zip",
                         DRIVE_PROJECT / "Dataset" / "colab_data_bundle.zip",
                         Path("/content/drive/MyDrive/colab_data_bundle.zip")]
else:
    BASE = Path(os.environ.get("REV_BASE", "."))
    WORK = Path(os.environ.get("REV_WORK", str(BASE / "work")))
    HF_TOKEN = os.environ.get("HF_TOKEN")
    BUNDLE_CANDIDATES = [BASE / "colab_data_bundle.zip"]

BUNDLE = next((p for p in BUNDLE_CANDIDATES if p.exists()), BUNDLE_CANDIDATES[0])
OUT = BASE / "revision_colab"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
if not (WORK / "data").exists():
    assert BUNDLE.exists(), "Upload colab_data_bundle.zip to one of: " + ", ".join(map(str, BUNDLE_CANDIDATES))
    print("unzipping bundle ..."); zipfile.ZipFile(BUNDLE).extractall(WORK)
DATA = WORK / "data" / "datasets"
RERUN = WORK / "results" / "sc2_binary"        # fp16 rerun at commit 0e9e39f2 (liu_headtohead_metallama_v2*)
OLD = WORK / "results" / "sc2_binary_fp16"     # the fp16 results the manuscript currently cites

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
else:
    print("No HF_TOKEN secret. Run:  from huggingface_hub import notebook_login; notebook_login()")

MODEL_ID = os.environ.get("REV_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
REVISION = os.environ.get("REV_REVISION", "0e9e39f249a16976918f6564b8830bc894c89659")
SEED = 0
LAYER = 16          # hidden_states[16] == output of decoder layer 15
ABLATE_FROM = 15    # direction ablation hooks decoder layers 15..31
MAXLEN = 256        # body capped at MAXLEN-80 = 176 tokens (manuscript protocol)
TEMPLATE = ("Read the following text and determine if the person shows signs of "
            "suicidal ideation.\n\n"
            "Text: {text}\n\n"
            "Does this text show signs of suicidal ideation? Answer with Yes or No.\n"
            "Answer:")
PREFIX, SUFFIX = TEMPLATE.split("{text}")
print("OUT ->", OUT)


# ---- section 0 ----

# 0.3  Shared helpers (hashing, AUC, bootstrap, BH, JSON output)
import hashlib, re
import numpy as np
from scipy import stats
from scipy.stats import rankdata

ENV = {"seed": SEED, "model": MODEL_ID, "revision_requested": REVISION}

def h16(t):
    """Same hash as sc2_liu_headtohead_v2.py."""
    return hashlib.sha256(" ".join(t.split()).encode()).hexdigest()[:16]

def ci(a, axis=0):
    return np.percentile(a, [2.5, 97.5], axis=axis)

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)

def auc_cols(y, S):
    """ROC-AUC of every column of S (ties -> average rank). y: 0/1 vector."""
    S = np.asarray(S, float)
    S = S[:, None] if S.ndim == 1 else S
    r = rankdata(S, axis=0)
    n1 = int(y.sum()); n0 = len(y) - n1
    out = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out

def gap_cols(y, S):
    S = np.asarray(S, float); S = S[:, None] if S.ndim == 1 else S
    return S[y == 1].mean(0) - S[y == 0].mean(0)

def strat_boot(y, B=2000, seed=SEED + 7):
    rng = np.random.default_rng(seed)
    P, N = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(B):
        yield np.concatenate([rng.choice(P, len(P)), rng.choice(N, len(N))])

def boot_p(d):
    """Two-sided bootstrap p-value for H0: effect = 0 (rows = resamples)."""
    return np.minimum(1.0, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))

def _js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def save_json(name, obj):
    path = OUT / name
    path.write_text(json.dumps({**obj, "env": ENV}, indent=2, default=_js))
    print("saved", path)

def fmt_ci(x, lo, hi, d=3):
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ---- section 1 ----

# 1.1  Suicide-Watch-232K draw (sc2_faithfulness.py / sc2_liu_headtohead_v2.py)
from datasets import load_from_disk
import csv
ds07 = load_from_disk(str(DATA / "07_Suicide_Watch_232K/hf_cache/train"))
T07, L07 = ds07["text"], ds07["label"]

def draw_main(seed=SEED):
    rng = np.random.default_rng(seed)
    sui = [t for t, l in zip(T07, L07) if l == "suicide" and t and len(t) > 80]
    non = [t for t, l in zip(T07, L07) if l == "non-suicide" and t and len(t) > 80]
    rng.shuffle(sui); rng.shuffle(non)
    return sui, non, rng

SUI, NON, _rng = draw_main()
RNG_AFTER_SHUFFLE = copy.deepcopy(_rng)   # the random-control direction is the next draw from this
SUI_TR, NON_TR = SUI[:100], NON[:100]          # direction fit
SUI_TE, NON_TE = SUI[100:300], NON[100:300]    # held-out test; first 100/class == manuscript test set

abl_rows = list(csv.DictReader(open(RERUN / "liu_headtohead_metallama_v2_ablation_examples.csv")))
assert [r["hash"] for r in abl_rows] == [h16(t) for t in SUI_TE + NON_TE], "test draw differs from the rerun!"
print(f"pools: {len(SUI)} suicide / {len(NON)} non-suicide;  test set matches the fp16 rerun (400/400 hashes)")


# ---- section 1 ----

# 1.2  External corpora: exact rng sequence of sc2_crossablation.py / sc2_generalization.py / sc2_internal_monitor.py
dssd = load_from_disk(str(DATA / "08_SDCNL/hf_cache/train"))
print("columns  232K:", ds07.column_names, "| 'SDCNL':", dssd.column_names, f"({len(dssd)} rows)")

def load_swmh(split):
    with open(DATA / "01_SWMH_Reddit" / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]
SWMH_TRAIN = load_swmh("train")
print("SWMH columns:", sorted(SWMH_TRAIN[0].keys()), f"({len(SWMH_TRAIN)} train rows)")

def draw_external(n_ext=500):
    """The 80/class (cross-ablation) and 120/class (generalization) samples are prefixes of this draw."""
    rng = np.random.default_rng(SEED)
    def clean(texts, n):
        out = [t for t in texts if t and len(t) > 80]; rng.shuffle(out); return out[:n]
    s07 = clean([t for t, l in zip(T07, L07) if l == "suicide"], 100)
    n07 = clean([t for t, l in zip(T07, L07) if l == "non-suicide"], 100)
    sd = (clean([t for t, c in zip(dssd["text"], dssd["class"]) if c == "suicide"], n_ext),
          clean([t for t, c in zip(dssd["text"], dssd["class"]) if c == "non-suicide"], n_ext))
    sw = (clean([r.get("text", "") for r in SWMH_TRAIN if r.get("label", "") == "self.SuicideWatch"], n_ext),
          clean([r.get("text", "") for r in SWMH_TRAIN
                 if r.get("label", "") not in ("self.SuicideWatch", "self.depression")], n_ext))
    return {"s07": s07, "n07": n07, "SDCNL": sd, "SWMH": sw}, rng

EXT, RNG_AFTER_EXT = draw_external(500)
assert EXT["s07"] == SUI_TR and EXT["n07"] == NON_TR
print("external draws ok; 232K direction-train set identical across scripts")


# ---- section 1 ----

# 1.3  Tokenizer + prompt builder (identical to the manuscript scripts)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
YES = tok.encode(" Yes", add_special_tokens=False)[0]
NO = tok.encode(" No", add_special_tokens=False)[0]

def chat_ids(text, maxlen=MAXLEN):
    body = tok(text, add_special_tokens=False)["input_ids"]
    truncated = len(body) > maxlen - 80
    content = PREFIX + tok.decode(body[: maxlen - 80]) + SUFFIX
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"], truncated

def body_len(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])

ids0, tr0 = chat_ids(SUI_TE[0])
assert ids0.shape[1] == int(abl_rows[0]["n_tokens"]), "prompt length differs from the rerun"
print("prompt builder matches the rerun (token count check)")


# ---- section 7 ----

import torch, transformers
from transformers import AutoModelForCausalLM
DEV = "cuda"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float16,
                                             device_map={"": 0})
model.eval()
LAYERS = model.model.layers
NL, NH = len(LAYERS), model.config.num_attention_heads
HD = getattr(model.config, "head_dim", None) or model.config.hidden_size // NH
DM = model.config.hidden_size
ENV.update(revision_resolved=getattr(model.config, "_commit_hash", None), precision="fp16",
           transformers=transformers.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
print(ENV)

@torch.inference_mode()
def ld_batch(ids):
    lg = model(ids.to(DEV), logits_to_keep=1).logits[:, -1, :].float()
    return (lg[:, YES] - lg[:, NO]).cpu().numpy()

@torch.inference_mode()
def hidden_states(ids):
    out = model(ids.to(DEV), output_hidden_states=True, logits_to_keep=1)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy()   # [NL+1, DM]

def make_ablate_hook(vec):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(h.dtype)
        h2 = h - (h @ v).unsqueeze(-1) * v
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    return hook

def score(ids_list, vec=None, layers=None):
    """LD per prompt, optionally ablating `vec` (np array) at decoder layers `layers` (default 15..31)."""
    handles = []
    if vec is not None:
        layers = range(ABLATE_FROM, NL) if layers is None else layers
        v = torch.tensor(vec, device=DEV, dtype=torch.float16)
        handles = [LAYERS[i].register_forward_hook(make_ablate_hook(v)) for i in layers]
    try:
        return np.array([ld_batch(i)[0] for i in ids_list])
    finally:
        for x in handles: x.remove()

# direction: difference of means at hidden_states[16] over the training posts
H_TR = np.stack([hidden_states(chat_ids(t)[0]) for t in SUI_TR + NON_TR])     # [200, NL+1, DM]
DIR = H_TR[:100, LAYER].mean(0) - H_TR[100:, LAYER].mean(0); DIR /= np.linalg.norm(DIR)
_r = copy.deepcopy(RNG_AFTER_SHUFFLE)
RAND0 = _r.standard_normal(DM); RAND0 /= np.linalg.norm(RAND0)                 # same draw as sc2_faithfulness.py
np.save(OUT / "direction_L16.npy", DIR)

TEST_TEXTS = SUI_TE + NON_TE
Y_TEST = np.array([1] * len(SUI_TE) + [0] * len(NON_TE))
TEST_IDS = [chat_ids(t)[0] for t in TEST_TEXTS]
m100 = np.r_[np.arange(100), 200 + np.arange(100)]
sc7 = {"baseline": score([TEST_IDS[i] for i in m100]),
       "direction": score([TEST_IDS[i] for i in m100], DIR),
       "random0": score([TEST_IDS[i] for i in m100], RAND0)}
y100 = Y_TEST[m100]
ref = {k: np.array([float(abl_rows[i][c]) for i in m100])
       for k, c in (("baseline", "baseline"), ("direction", "direction_ablation"), ("random0", "random_0"))}
for k in sc7:
    print(f"{k:10s} AUC {auc_cols(y100, sc7[k])[0]:.4f} (rerun {auc_cols(y100, ref[k])[0]:.4f}; manuscript "
          f"{ {'baseline': 0.975, 'direction': 0.832, 'random0': 0.975}[k] })   "
          f"max |LD - rerun LD| = {np.abs(sc7[k] - ref[k]).max():.3f}")


# ---- section 12 ----

DIRS = H_TR[:100].mean(0) - H_TR[100:].mean(0)
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True) + 1e-8                      # [NL+1, DM]
_r = copy.deepcopy(RNG_AFTER_EXT)
RANDS = _r.standard_normal(DIRS.shape); RANDS /= np.linalg.norm(RANDS, axis=1, keepdims=True)
flag = globals().get("FLAGGED", set())
TRANSFER = {}
for name in ("SDCNL", "SWMH"):
    s, n = EXT[name][0][:120], EXT[name][1][:120]
    texts = s + n; y = np.array([1] * len(s) + [0] * len(n))
    ids = [chat_ids(t)[0] for t in texts]
    Hx = np.stack([hidden_states(i)[LAYER] for i in ids])
    proj = Hx @ DIRS[LAYER]
    lds = score(ids)
    sub80 = np.r_[np.arange(80), 120 + np.arange(80)]
    TRANSFER[name] = {"proj": proj, "ld": lds, "y": y, "ids": ids, "texts": texts, "sub80": sub80}

def score_perlayer(ids_list, dirs):
    """Ablate dirs[i+1] at decoder layer i, for every layer (sc2_crossablation.py)."""
    vs = [torch.tensor(dirs[i + 1], device=DEV, dtype=torch.float16) for i in range(NL)]
    hs = [LAYERS[i].register_forward_hook(make_ablate_hook(vs[i])) for i in range(NL)]
    try:
        return np.array([ld_batch(i)[0] for i in ids_list])
    finally:
        for x in hs: x.remove()

TRANSFER_OUT = {}
for name, T in TRANSFER.items():
    sub = T["sub80"]; ids80 = [T["ids"][i] for i in sub]; y80 = T["y"][sub]
    base80 = T["ld"][sub]; abl80 = score_perlayer(ids80, DIRS); rnd80 = score_perlayer(ids80, RANDS)
    keep120 = np.array([h16(t) not in flag for t in T["texts"]]); keep80 = keep120[sub]
    res = {}
    for tag, k120, k80 in (("all", np.ones(240, bool), np.ones(160, bool)), ("excluding_overlap", keep120, keep80)):
        yk = T["y"][k120]; S3 = np.stack([base80, abl80, rnd80], 1)[k80]; y3 = y80[k80]
        a3 = auc_cols(y3, S3); b3 = np.array([auc_cols(y3[ix], S3[ix]) for ix in strat_boot(y3, 2000)])
        dec = auc_cols(yk, np.stack([T["proj"][k120], T["ld"][k120]], 1))
        res[tag] = {"n_per_class_decoding": [int(yk.sum()), int((1 - yk).sum())],
                    "decoding_transfer_auc": dec[0], "behavioral_auc_120": dec[1],
                    "n_per_class_causal": [int(y3.sum()), int((1 - y3).sum())],
                    "causal": {"baseline": a3[0], "ablate_232K_dirs": a3[1], "random": a3[2],
                               "drop_ci95": ci(b3[:, 0] - b3[:, 1]), "random_drop_ci95": ci(b3[:, 0] - b3[:, 2])}}
        print(f"{name} [{tag}] decoding transfer AUC {dec[0]:.3f} (n={int(yk.sum())}+{int((1 - yk).sum())}); "
              f"causal: baseline {a3[0]:.3f} -> ablated {a3[1]:.3f} (drop {fmt_ci(a3[0] - a3[1], *ci(b3[:, 0] - b3[:, 1]))}),"
              f" random {a3[2]:.3f}")
    old = json.load(open(OLD / "crossablation.json"))["results"][name]
    res["old_fp16_crossablation_json"] = {c: old[c]["auc"] for c in old}
    TRANSFER_OUT[name] = res
save_json("transfer_fp16.json", TRANSFER_OUT)


# ---- section 12b ----

import collections as _co
_tok = lambda x: set(re.findall(r"[a-z0-9]+", (x or "").lower()))
_A = [_tok(t) for t in T07]
_df = _co.Counter(w for s in _A for w in s)
_idx = _co.defaultdict(list)
for i, s in enumerate(_A):
    if len(s) < 15: continue
    for w in sorted(s, key=_df.__getitem__)[:2]: _idx[w].append(i)
FIT = set(SUI_TR + NON_TR)

def match_232k(text, thr=0.9):
    T = _tok(text); cands = set()
    for w in T:
        L = _idx.get(w)
        if L and len(L) <= 200: cands.update(L)
    if not cands: return None
    c, i = max((len(_A[i] & T) / len(_A[i]), i) for i in cands)
    return i if c >= thr else None

DEDUP = {}
for name in ("SDCNL", "SWMH"):
    cls_texts, info = [], {}
    for cls, texts in zip(("suicide", "non_suicide"), EXT[name]):
        m = [match_232k(t) for t in texts]
        info[cls] = {"flagged_first80": sum(x is not None for x in m[:80]),
                     "flagged_first120": sum(x is not None for x in m[:120]),
                     "flagged_of_500": sum(x is not None for x in m),
                     "flagged_direction_fit_posts": sum(x is not None and T07[x] in FIT for x in m)}
        cls_texts.append([t for t, x in zip(texts, m) if x is None])
    s, n = cls_texts[0][:120], cls_texts[1][:120]
    print(name, info, "| clean posts available:", len(cls_texts[0]), len(cls_texts[1]))
    texts = s + n; y = np.array([1] * len(s) + [0] * len(n))
    ids = [chat_ids(t)[0] for t in texts]
    proj = np.stack([hidden_states(i)[LAYER] for i in ids]) @ DIRS[LAYER]
    lds = score(ids)
    sub = np.r_[np.arange(80), len(s) + np.arange(80)]
    ids80 = [ids[i] for i in sub]; y80 = y[sub]
    S3 = np.stack([lds[sub], score_perlayer(ids80, DIRS), score_perlayer(ids80, RANDS)], 1)
    a3 = auc_cols(y80, S3); b3 = np.array([auc_cols(y80[ix], S3[ix]) for ix in strat_boot(y80, 2000)])
    dec = auc_cols(y, np.stack([proj, lds], 1))
    DEDUP[name] = {"flags": info, "n_per_class_decoding": [len(s), len(n)], "decoding_transfer_auc": dec[0],
                   "behavioral_auc_120": dec[1], "n_per_class_causal": [80, 80],
                   "causal": {"baseline": a3[0], "ablate_232K_dirs": a3[1], "random": a3[2],
                              "drop_ci95": ci(b3[:, 0] - b3[:, 1]), "random_drop_ci95": ci(b3[:, 0] - b3[:, 2])}}
    print(f"{name} [deduplicated] decoding {dec[0]:.3f}; causal {a3[0]:.3f} -> {a3[1]:.3f} "
          f"(drop {fmt_ci(a3[0] - a3[1], *ci(b3[:, 0] - b3[:, 1]))}), random {a3[2]:.3f}")
save_json("transfer_dedup.json", DEDUP)

