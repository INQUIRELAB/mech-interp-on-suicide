#!/usr/bin/env python
"""Can answer-orthogonalization rescue the sufficiency rung?

specificity_matrix.py showed the diff-of-means steering vector is dominated by a
generic "answer Yes" component (it raises off-target questions as much as the
on-target one). The diff-of-means is computed at the answer token on a polar
question, so it necessarily includes the Yes-No answer direction.

Fix: estimate the answer direction y = unembed(Yes) - unembed(No) in residual
space, project it out of each per-layer concept direction, and re-run the
cross-concept steering matrix with the answer-orthogonalized directions. If
suicide_perp now raises the suicide question MORE than food/toxicity (and likewise
toxicity_perp), sufficiency is real once the answer component is removed. If not,
diff-of-means cannot carry a clean sufficiency claim and the paper must say so.

Output: results/gcl/specificity_orthogonal.json
"""
import sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch, numpy as np
from src.repro import get_seed, set_seed
set_seed()
from src.models import ModelLoader
from concept_config import get_concept

MAXLEN = 256
WIN = list(range(12, 21))
N_DIR_TRAIN = 100
N_SUBSTRATE = 40
COEFS = [0.5, 1.0]
device = "cuda"
rng = np.random.default_rng(get_seed())

QUESTIONS = {
    "suicide": ("Read the following text and determine if the person shows signs of suicidal ideation.",
                "Does this text show signs of suicidal ideation?"),
    "toxicity": ("Read the following comment and determine if it is toxic (rude, disrespectful, or hateful).",
                 "Is this comment toxic?"),
    "food": ("Read the following text and determine if it is primarily about food or cooking.",
             "Is this text primarily about food or cooking?"),
}

loader = ModelLoader(device=device)
model, tok = loader.load_llama(load_in_4bit=True, auto_quantize=False)
model.eval()
yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
no_id = tok.encode(" No", add_special_tokens=False)[0]
layers = model.model.layers
nL = len(layers)
d_model = model.config.hidden_size


# ---- answer direction from the unembedding (robust to 4-bit lm_head) ----
def get_answer_direction():
    w = model.lm_head.weight
    try:
        if w.dtype in (torch.float16, torch.float32, torch.bfloat16):
            row_yes = w[yes_id].float().detach().cpu().numpy()
            row_no = w[no_id].float().detach().cpu().numpy()
            y = row_yes - row_no
            print("  answer direction from lm_head (dense)")
        else:
            raise TypeError(w.dtype)
    except Exception as e:
        # fallback: dequantize via a forward of one-hot? simplest: use embed diff is wrong.
        # Empirical fallback below.
        print(f"  lm_head not dense ({e}); using empirical answer direction")
        return None
    return y / (np.linalg.norm(y) + 1e-8)


def fit_direction(concept_name):
    cc = get_concept(concept_name)
    pos, neg = cc.loader(N_DIR_TRAIN, seed=get_seed())
    pos, neg = pos[:N_DIR_TRAIN], neg[:N_DIR_TRAIN]
    Hp = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in pos])
    Hn = np.stack([hidden_lastpos(t, cc.prompt_prefix, cc.question) for t in neg])
    raw = Hp.mean(0) - Hn.mean(0)
    unit = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    return unit, Hp, Hn


def build_ids(text, prefix, question):
    body = tok(text, add_special_tokens=False)["input_ids"][: MAXLEN - 80]
    content = f"{prefix}\n\nText: {tok.decode(body)}\n\n{question} Answer with Yes or No.\nAnswer:"
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def hidden_lastpos(text, prefix, question):
    out = model(build_ids(text, prefix, question), output_hidden_states=True)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states])


@torch.no_grad()
def logit_diff(text, prefix, question):
    lg = model(build_ids(text, prefix, question)).logits[0, -1, :]
    return (lg[yes_id] - lg[no_id]).item()


print("Fitting concept directions...")
sui_unit, Hp_s, Hn_s = fit_direction("suicide")
tox_unit, Hp_t, Hn_t = fit_direction("toxicity")

y_single = get_answer_direction()
if y_single is None:
    # empirical: shared component of the two concept directions ~ answer direction
    shared = (sui_unit + tox_unit)
    y_layer = shared / (np.linalg.norm(shared, axis=1, keepdims=True) + 1e-8)
else:
    y_layer = np.tile(y_single, (nL + 1, 1))  # same unembed direction at every layer


def orthogonalize(unit, y):
    proj = (unit * y).sum(1, keepdims=True) * y
    perp = unit - proj
    return perp / (np.linalg.norm(perp, axis=1, keepdims=True) + 1e-8)


sui_perp = orthogonalize(sui_unit, y_layer)
tox_perp = orthogonalize(tox_unit, y_layer)

# cosine of raw direction with answer direction (how entangled it was), at WIN layers
cos_sui = float(np.mean([abs(np.dot(sui_unit[l], y_layer[l])) for l in WIN]))
cos_tox = float(np.mean([abs(np.dot(tox_unit[l], y_layer[l])) for l in WIN]))
print(f"  |cos(concept dir, answer dir)| over WIN: suicide={cos_sui:.3f} toxicity={cos_tox:.3f}")


def delta_for(unit, Hp, Hn):
    return ((Hp.mean(0) - Hn.mean(0)) * unit).sum(1)


DIRS = {
    "suicide_raw":  (sui_unit, delta_for(sui_unit, Hp_s, Hn_s)),
    "suicide_perp": (sui_perp, delta_for(sui_perp, Hp_s, Hn_s)),
    "toxicity_raw": (tox_unit, delta_for(tox_unit, Hp_t, Hn_t)),
    "toxicity_perp":(tox_perp, delta_for(tox_perp, Hp_t, Hn_t)),
}

sub_pos, sub_neg = get_concept("suicide").loader(N_SUBSTRATE + 50, seed=get_seed() + 7)
substrate = sub_neg[:N_SUBSTRATE]


def add_hooks(unit, delta, coef):
    amap = {i: torch.tensor(coef * delta[i + 1] * unit[i + 1], device=device) for i in WIN}
    def mk(i):
        add = amap[i]
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = h + add.to(h.dtype)
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        return hook
    return [layers[i].register_forward_hook(mk(i)) for i in WIN]


def mean_ld(qk):
    prefix, q = QUESTIONS[qk]
    return float(np.mean([logit_diff(t, prefix, q) for t in substrate]))


print("Baselines...")
base = {qk: mean_ld(qk) for qk in QUESTIONS}
print("  " + "  ".join(f"{k}{v:+.2f}" for k, v in base.items()))

matrix = {}
for dname, (unit, delta) in DIRS.items():
    matrix[dname] = {}
    for c in COEFS:
        h_ = add_hooks(unit, delta, c)
        try:
            shifts = {qk: mean_ld(qk) - base[qk] for qk in QUESTIONS}
        finally:
            [x.remove() for x in h_]
        matrix[dname][str(c)] = shifts
        print(f"  {dname:13s} c={c}: " + "  ".join(f"{qk}{shifts[qk]:+.2f}" for qk in QUESTIONS))

# verdict: does perp direction raise its OWN question most?
def own(dname, c):
    s = matrix[dname][str(c)]
    own_q = "suicide" if dname.startswith("suicide") else "toxicity"
    others = [q for q in QUESTIONS if q != own_q]
    return s[own_q] > max(s[o] for o in others)

verdict = {}
for c in COEFS:
    verdict[str(c)] = {
        "suicide_perp_specific": bool(own("suicide_perp", c)),
        "toxicity_perp_specific": bool(own("toxicity_perp", c)),
        "suicide_raw_specific": bool(own("suicide_raw", c)),
        "toxicity_raw_specific": bool(own("toxicity_raw", c)),
    }
rescued = any(v["suicide_perp_specific"] and v["toxicity_perp_specific"] for v in verdict.values())

print("\n" + "=" * 64)
print("ANSWER-ORTHOGONALIZATION VERDICT")
for c in COEFS:
    v = verdict[str(c)]
    print(f"  c={c}: perp specific  suicide={v['suicide_perp_specific']} toxicity={v['toxicity_perp_specific']}"
          f"   (raw: sui={v['suicide_raw_specific']} tox={v['toxicity_raw_specific']})")
print(f"  => sufficiency {'RESCUED by answer-orthogonalization' if rescued else 'NOT rescued: diff-of-means cannot carry clean sufficiency'}")

json.dump({"window": WIN, "coefs": COEFS, "cos_concept_answer": {"suicide": cos_sui, "toxicity": cos_tox},
           "baselines": base, "matrix": matrix, "verdict": verdict, "rescued": bool(rescued)},
          open(PROJECT_ROOT / "results" / "gcl" / "specificity_orthogonal.json", "w"), indent=2)
print("Saved results/gcl/specificity_orthogonal.json")
