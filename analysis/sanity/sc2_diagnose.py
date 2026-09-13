#!/usr/bin/env python
"""SC-2 diagnostic: why does Llama always answer Yes?

Tests 4 conditions on one model load to isolate the fix:
  A. raw prompt   + "Yes"/"No"  tokens (current SC-2 behavior)
  B. raw prompt   + " Yes"/" No" tokens (leading-space fix)
  C. chat template + " Yes"/" No" tokens (instruct fix)
  D. chat template + "Yes"/"No"  tokens

Reports per-class accuracy + mean logit-diff for each condition so we can
see which combination actually discriminates Death-Me from Life-Not-Me.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.analysis_utils import (
    load_model_wrapped, load_dataset_by_category, get_text, PROMPT_TEMPLATES,
)

N = 30
MAXLEN = 256
device = "cuda"

cats = load_dataset_by_category(dataset_name="deepsuimind")
death = cats["Death-Me"][:N]
life = cats["Life-Not-Me"][:N]

model, tok = load_model_wrapped("llama", device=device)

def tid(s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0], len(ids)

for label in ["Yes", "No", " Yes", " No"]:
    print(f"  token {label!r:6} -> id {tid(label)}")

yes_ns, no_ns = tid("Yes")[0], tid("No")[0]      # no space
yes_sp, no_sp = tid(" Yes")[0], tid(" No")[0]    # leading space


def raw_ids(text, style="standard"):
    tmpl = PROMPT_TEMPLATES[style]
    prefix, suffix = tmpl.split("{text}")
    pre = tok(prefix, add_special_tokens=True)["input_ids"]
    suf = tok(suffix, add_special_tokens=False)["input_ids"]
    body = tok(text, add_special_tokens=False)["input_ids"]
    budget = MAXLEN - len(pre) - len(suf)
    full = pre + body[:budget] + suf
    return torch.tensor([full], device=device)


def chat_ids(text, style="standard"):
    tmpl = PROMPT_TEMPLATES[style]
    # Split so we can budget the body, then re-assemble as a single user msg.
    prefix, suffix = tmpl.split("{text}")
    # Budget the raw body against a rough template-token estimate.
    body = tok(text, add_special_tokens=False)["input_ids"]
    body = body[: MAXLEN - 80]
    text_trunc = tok.decode(body)
    content = prefix + text_trunc + suffix
    msgs = [{"role": "user", "content": content}]
    prompt_str = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    ids = tok(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return ids


def run(builder, yes_id, no_id, style="standard"):
    res = {}
    for name, data, want_yes in [("death", death, True), ("life", life, False)]:
        correct, diffs = 0, []
        for ex in data:
            ids = builder(get_text(ex), style)
            with torch.no_grad():
                logits = model(ids)[0, -1, :]
            d = (logits[yes_id] - logits[no_id]).item()
            diffs.append(d)
            pred_yes = d > 0
            if pred_yes == want_yes:
                correct += 1
        res[name] = (correct / len(data), float(np.mean(diffs)))
    return res


conditions = [
    ("A raw  + no-space", raw_ids,  yes_ns, no_ns, "standard"),
    ("B raw  + space   ", raw_ids,  yes_sp, no_sp, "standard"),
    ("C chat + space   ", chat_ids, yes_sp, no_sp, "standard"),
    ("D chat + no-space", chat_ids, yes_ns, no_ns, "standard"),
    ("E chat + space/clin", chat_ids, yes_sp, no_sp, "clinical"),
]

print("\n" + "=" * 72)
print(f"{'condition':22} {'death_acc':>9} {'life_acc':>9} {'d_ld':>7} {'l_ld':>7}")
print("-" * 72)
for name, builder, yid, nid, style in conditions:
    r = run(builder, yid, nid, style)
    da, dl = r["death"]
    la, ll = r["life"]
    print(f"{name:22} {da:9.3f} {la:9.3f} {dl:7.2f} {ll:7.2f}")
print("=" * 72)
