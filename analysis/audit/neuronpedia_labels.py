#!/usr/bin/env python
"""Fetch Neuronpedia explanations for the SAE features named in sae_shared / sae_ts_link results.

Neuronpedia's gemma-2-9b-it/20-gemmascope-res-16k is Gemma Scope layer_20/width_16k/average_l0_91,
the default in sae_shared.py, so feature indices match. Run where there is internet access:
    python neuronpedia_labels.py ../../results/sae_shared/gemma_2_9b_it__gemma_l20__seed0.json \
        [../../results/sae_shared/gemma_2_9b_it__sae_ts_link__seed0.json]
Writes <first json>_labels.json: {feature index: [explanations]} plus the groups each came from.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

SOURCE = "gemma-2-9b-it/20-gemmascope-res-16k"


def fetch(idx):
    url = f"https://www.neuronpedia.org/api/feature/{SOURCE}/{idx}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            return [e.get("description") for e in d.get("explanations", []) if e.get("description")]
        except Exception as e:  # rate limit or transient error
            time.sleep(2 + 3 * attempt)
            err = str(e)
    return [f"<fetch failed: {err}>"]


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    groups = {}
    for p in paths:
        d = json.loads(p.read_text())
        if "decomposition" in d:
            dec = d["decomposition"]
            groups["shared_features"] = dec["shared_features"]
            groups["shared_pc1_decoder_top"] = dec["shared_pc1"]["top_features"]
            groups["yes_dir_decoder_top"] = dec["yes_dir"]["top_features"]
            for c, v in dec["concepts_feature"].items():
                groups[f"concept_{c}_dF_top10"] = v["top_features"][:10]
        if "b_hat_top_features" in d:
            groups["sae_ts_bhat_top"] = d["b_hat_top_features"]
    idxs = sorted({int(i) for v in groups.values() for i in v})
    print(f"{len(idxs)} features to fetch")
    labels = {}
    for i, idx in enumerate(idxs):
        labels[idx] = fetch(idx)
        time.sleep(0.3)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(idxs)}")
    out = paths[0].with_name(paths[0].stem + "_labels.json")
    out.write_text(json.dumps({"source": SOURCE, "groups": groups,
                               "labels": {str(k): v for k, v in labels.items()}}, indent=1))
    for g, ids in groups.items():
        print(f"\n== {g}")
        for idx in ids[:10]:
            print(f"  {idx:6d}  {(labels[int(idx)] or ['-'])[0]}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
