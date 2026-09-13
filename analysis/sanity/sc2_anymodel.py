#!/usr/bin/env python
"""Generalized cross-model replication — probe + behavioral + faithfulness for ANY model.

Generalizes sc2_secondmodel.py beyond the hardcoded base Qwen/Mistral so the paper
can report (a) cross-SCALE replication (Llama-3.2-1B/3B-Instruct) and (b)
INSTRUCT-tuned Qwen/Mistral, addressing two gaps vs Liu et al. (model-size
generalization, §4.4.2) and the paper's own limitation #4 (base vs instruct).

Instruct models use the chat template (matching the primary Llama-3.1-8B-Instruct
methodology); base models fall back to raw prompting. The layer-16 anchor does not
transfer across depths, so the probe sweeps all layers and reports the best layer.

Usage:
  python sc2_anymodel.py --model meta-llama/Llama-3.2-3B-Instruct
  python sc2_anymodel.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen_instruct
  python sc2_anymodel.py --model mistralai/Mistral-7B-Instruct-v0.3 --prompt chat
"""
import sys, json, argparse, re
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
from src.analysis_utils import build_prompt_tokens, PROMPT_TEMPLATES, ensure_dir

N = 150
N_TRAIN = 100
N_TEST = 50
MAXLEN = 256
BOOT = 2000
device = "cuda"
rng = np.random.default_rng(get_seed())
brng = np.random.default_rng(get_seed() + 12345)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=get_seed())


def parse_args():
    p = argparse.ArgumentParser(description="Generalized cross-model replication")
    p.add_argument("--model", required=True, help="full HF id, e.g. meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--tag", default=None, help="output tag (default: derived from model id)")
    p.add_argument("--prompt", choices=["auto", "chat", "raw"], default="auto",
                   help="auto: chat template if available else raw")
    return p.parse_args()


def boot_ci(score_pos, score_neg, B=BOOT):
    sp, sn = np.asarray(score_pos), np.asarray(score_neg)
    np_, nn = len(sp), len(sn); y = np.r_[np.ones(np_), np.zeros(nn)]
    aucs = np.empty(B)
    for b in range(B):
        aucs[b] = roc_auc_score(y, np.r_[sp[brng.integers(0, np_, np_)], sn[brng.integers(0, nn, nn)]])
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def clean(texts, n):
    out = [t for t in texts if t and len(t) > 80]; rng.shuffle(out); return out[:n]


def main():
    args = parse_args()
    tag = args.tag or re.sub(r"[^a-zA-Z0-9]+", "_", args.model.split("/")[-1]).lower()

    ds = load_from_disk("data/datasets/07_Suicide_Watch_232K/hf_cache")["train"]
    sui = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "suicide"], N)
    non = clean([t for t, l in zip(ds["text"], ds["label"]) if l == "non-suicide"], N)

    loader = ModelLoader(device=device)
    model, tok = loader.load_model(args.model, load_in_4bit=True, auto_quantize=False)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = (args.prompt == "chat") or (args.prompt == "auto" and tok.chat_template is not None)
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    layers = model.model.layers
    nL = len(layers)
    print(f"{args.model}: {nL} layers; prompt={'chat' if use_chat else 'raw'}; yes={yes_id} no={no_id}")

    def make_ids(text):
        if use_chat:
            tmpl = PROMPT_TEMPLATES["standard"]; prefix, suffix = tmpl.split("{text}")
            body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
            content = prefix + tok.decode(body) + suffix
            s = tok.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return build_prompt_tokens(text, tok, style="standard", max_total_tokens=MAXLEN, device=device)

    @torch.no_grad()
    def feat(text):
        out = model(make_ids(text), output_hidden_states=True)
        lg = out.logits[0, -1, :]
        ld = (lg[yes_id] - lg[no_id]).item()
        hs = np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])
        return ld, hs

    @torch.no_grad()
    def logit_diff(text):
        lg = model(make_ids(text)).logits[0, -1, :]
        return (lg[yes_id] - lg[no_id]).item()

    def collect(texts, t):
        lds, hs = [], []
        for i, x in enumerate(texts):
            ld, h = feat(x); lds.append(ld); hs.append(h)
            if (i + 1) % 50 == 0: print(f"    {t} {i+1}/{len(texts)}")
        return np.array(lds), np.stack(hs)

    print("Extracting activations...")
    ld_s, H_s = collect(sui, "sui"); ld_n, H_n = collect(non, "non")

    y = np.array([1] * N + [0] * N)
    best = (0, 0.0); best_proba = None; per_layer = []
    for L in range(H_s.shape[1]):
        X = np.concatenate([H_s[:, L, :], H_n[:, L, :]])
        proba = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                                  X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba); per_layer.append((L, auc))
        if auc > best[1]: best = (L, auc); best_proba = proba.copy()
    behav_auc = roc_auc_score(y, np.concatenate([ld_s, ld_n]))
    probe_ci = boot_ci(best_proba[:N], best_proba[N:]); behav_ci = boot_ci(ld_s, ld_n)

    # faithfulness: per-layer ablation
    trs = slice(0, N_TRAIN)
    dirs = H_s[trs].mean(0) - H_n[trs].mean(0)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    rand = rng.standard_normal(dirs.shape); rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
    R = {i: torch.tensor(dirs[i + 1], device=device, dtype=torch.float16) for i in range(nL)}
    Rr = {i: torch.tensor(rand[i + 1], device=device, dtype=torch.float16) for i in range(nL)}

    def make_hook(vec):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = vec.to(h.dtype); h2 = h - (h @ v).unsqueeze(-1) * v
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook

    def with_hooks(vmap):
        return [layers[i].register_forward_hook(make_hook(vmap[i])) for i in range(nL)]

    sui_te, non_te = sui[N_TRAIN:N_TRAIN + N_TEST], non[N_TRAIN:N_TRAIN + N_TEST]
    yte = np.array([1] * N_TEST + [0] * N_TEST)

    def behav_auc_te():
        s = [logit_diff(t) for t in sui_te]; n = [logit_diff(t) for t in non_te]
        return roc_auc_score(yte, np.array(s + n)), float(np.mean(s) - np.mean(n))

    print("Faithfulness baseline / ablate / random...")
    base_f = behav_auc_te()
    h_ = with_hooks(R); abl_f = behav_auc_te(); [x.remove() for x in h_]
    h_ = with_hooks(Rr); rnd_f = behav_auc_te(); [x.remove() for x in h_]

    drop, cdrop = base_f[0] - abl_f[0], base_f[0] - rnd_f[0]
    ok = best[1] > 0.75 and drop > 3 * max(cdrop, 0.01)
    print("\n" + "=" * 68)
    print(f"CROSS-MODEL: {args.model}  ({nL} layers, {'chat' if use_chat else 'raw'}, n={N}/class)")
    print("=" * 68)
    print(f"  best probe layer: {best[0]}/{nL}  AUC={best[1]:.3f}  CI=[{probe_ci[0]:.3f},{probe_ci[1]:.3f}]")
    print(f"  behavioral AUC:   {behav_auc:.3f}  CI=[{behav_ci[0]:.3f},{behav_ci[1]:.3f}]")
    print(f"  faithfulness: base AUC {base_f[0]:.3f} -> ablate {abl_f[0]:.3f} (random {rnd_f[0]:.3f})")
    print(f"  AUC drop suic {drop:+.3f} vs random {cdrop:+.3f}")
    print(f"VERDICT: {'REPLICATES' if ok else 'partial'} -- {args.model}")

    ensure_dir(PROJECT_ROOT / "results" / "sc2_binary")
    out = PROJECT_ROOT / "results" / "sc2_binary" / f"crossmodel_{tag}.json"
    json.dump({"model": args.model, "tag": tag, "n_layers": nL, "prompt": "chat" if use_chat else "raw",
               "n": N, "best_probe_layer": best[0], "best_probe_layer_frac": best[0] / nL,
               "best_probe_auc": best[1], "best_probe_auc_ci95": probe_ci,
               "behavioral_auc": behav_auc, "behavioral_auc_ci95": behav_ci,
               "faithfulness": {"baseline_auc": base_f[0], "ablate_auc": abl_f[0], "random_auc": rnd_f[0],
                                "baseline_gap": base_f[1], "ablate_gap": abl_f[1], "random_gap": rnd_f[1]},
               "per_layer_auc": [{"layer": L, "auc": a} for L, a in per_layer],
               "replicates": bool(ok)}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
