#!/usr/bin/env python
"""Aggregate audit runs into evidence profiles (claims x rungs), Paper 1's headline table.

Reads results/audit/<concept>/<model>__seed<k>.json and writes
results/audit/evidence_profile.csv and .md. A cell is PASS / fail / gated (NO-GO, rung not
run) / n/a (rung absent, e.g. rank before Phase 3). With several seeds a cell reports the
pass fraction, e.g. "2/3".

Usage: python evidence_profile.py [--root results/audit]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from concepts import PROJECT_ROOT

RUNGS = ["gate", "decodable", "necessary_klm", "necessary_zero", "necessary_mean", "sufficient_polar",
         "sufficient_mc", "rank_identified"]


def rung_values(r):
    out = {"gate": r["gate"]["decision"] == "GO", "decodable": r["decodable"]["pass"]}
    gated = "necessary" not in r
    for mode in ("zero", "mean"):
        m = r.get("necessary", {}).get("modes", {}).get(mode)
        out[f"necessary_{mode}"] = "gated" if gated else (m["gate"]["pass"] if m else None)
    # KL-matched, target-orthogonal nulls (F9); the unmatched zero/mean gates use invalid nulls
    klm = r.get("necessary", {}).get("kl_matched")
    out["necessary_klm"] = "gated" if gated else (klm["gate"]["pass"] if klm else None)
    s = r.get("sufficient")
    out["sufficient_polar"] = "gated" if gated else (s["gate_polar"]["pass"] if s else None)
    out["sufficient_mc"] = "gated" if gated else (s["gate_mc"]["pass"] if s else None)
    rk = r.get("rank")
    out["rank_identified"] = rk.get("identified") if rk else None
    return out


def cell(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if all(v == "gated" for v in vals):
        return "gated"
    b = [v for v in vals if isinstance(v, bool)]
    if len(b) == 1:
        return "PASS" if b[0] else "fail"
    return f"{sum(b)}/{len(b)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "results" / "audit"))
    root = Path(ap.parse_args().root)
    groups = defaultdict(list)
    for f in sorted(root.glob("*/*__seed*.json")):
        r = json.loads(f.read_text())
        groups[(r["concept"], r["model_tag"])].append(r)
    rows = []
    for (concept, model), runs in sorted(groups.items()):
        per = [rung_values(r) for r in runs]
        row = {"concept": concept, "source": runs[0].get("source", ""), "model": model,
               "seeds": len(runs)}
        row.update({k: cell([p[k] for p in per]) for k in RUNGS})
        tgt = [r["sufficient"]["target"] for r in runs if "sufficient" in r]
        if tgt:
            row["on_shift"] = round(sum(t["on_shift"] for t in tgt) / len(tgt), 2)
            row["off_shift"] = round(sum(t["off_shift"] for t in tgt) / len(tgt), 2)
            row["neg_shift"] = round(sum(t["neg_shift"] for t in tgt) / len(tgt), 2)
        rows.append(row)
    if not rows:
        print(f"no runs under {root}")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "evidence_profile.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (root / "evidence_profile.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
