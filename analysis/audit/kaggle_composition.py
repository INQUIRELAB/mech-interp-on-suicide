#!/usr/bin/env python
"""Which subreddits make up the two classes of the Kaggle suicide corpus (Suicide-Watch-232K / Kaggle-raw)?

The Kaggle page says posts were collected from r/SuicideWatch and r/depression and that non-suicide posts come
from r/teenagers, while the current release has only "suicide" and "non-suicide" labels. SWMH holds posts
from the same subreddits and years with their subreddit of origin. This script matches every SWMH
r/SuicideWatch and r/depression post (more than 80 characters) against the raw-text Kaggle release
(Ram07/Detection-for-Suicide) by normalized full text and by normalized 25-word prefix, and reports the share
of each SWMH forum found in each Kaggle class. If r/depression posts appear in the "suicide" class about as
often as r/SuicideWatch posts do, the "suicide" class pools both forums.

Usage: python kaggle_composition.py <Kaggle-raw load_from_disk dir> <SWMH train.jsonl> <out.json>
"""
import hashlib
import json
import re
import sys
from collections import Counter

from datasets import load_from_disk, concatenate_datasets

KAGGLE, TRAIN, OUT = sys.argv[1:4]


def toks(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def key_full(t):
    return hashlib.md5(" ".join(toks(t)).encode()).hexdigest()


def key_prefix(t, n=25):
    w = toks(t)
    return hashlib.md5(" ".join(w[:n]).encode()).hexdigest() if len(w) >= n else None


dd = load_from_disk(KAGGLE)
ds = concatenate_datasets([dd[s] for s in dd]) if hasattr(dd, "keys") else dd
cols = ds.column_names
tcol = "text" if "text" in cols else cols[0]
lcol = "label" if "label" in cols else [c for c in cols if c != tcol][0]
labels = Counter(ds[lcol])
full_k, pref_k = {}, {}
for t, lab in zip(ds[tcol], ds[lcol]):
    full_k.setdefault(key_full(t), set()).add(str(lab))
    kp = key_prefix(t)
    if kp:
        pref_k.setdefault(kp, set()).add(str(lab))

swmh = {"self.SuicideWatch": [], "self.depression": []}
for line in open(TRAIN, encoding="utf-8"):
    r = json.loads(line)
    if r.get("label") in swmh and len((r.get("text") or "").strip()) > 80:
        swmh[r["label"]].append(r["text"].strip())

res = {"kaggle_rows": len(ds), "kaggle_columns": cols, "kaggle_label_counts": {str(k): v for k, v in labels.items()},
       "swmh": {}}
for forum, texts in swmh.items():
    row = {"n": len(texts)}
    for kind, index, keyf in (("full_text", full_k, key_full), ("prefix25", pref_k, key_prefix)):
        hits = Counter()
        n_keyed = 0
        for t in texts:
            k = keyf(t)
            if k is None:
                continue
            n_keyed += 1
            for lab in index.get(k, ()):
                hits[lab] += 1
        row[kind] = {"n_keyed": n_keyed, **{f"in_{lab}": hits[lab] for lab in sorted(hits)},
                     **{f"share_in_{lab}": hits[lab] / max(n_keyed, 1) for lab in sorted(hits)}}
    res["swmh"][forum] = row
    print(forum, json.dumps(row), flush=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("saved", OUT)
