#!/usr/bin/env python
"""Run any result-producing script across several seeds and report mean +/- std.

Each sanity/gcl script routes its randomness through src.repro.get_seed(), which
reads the SEED env var. This driver runs the target script once per seed (setting
SEED in the child environment), captures the JSON it writes, and aggregates every
numeric field across seeds.

Usage
-----
    # seeds from the SEEDS env var / .env (default), e.g. SEEDS=0,1,2,3,4
    python analysis/sanity/seed_sweep.py \
        --script analysis/sanity/sc2_steering_specificity.py \
        --out results/sc2_binary/steering_specificity.json

    # explicit seeds, plus args forwarded to the target script after `--`
    python analysis/sanity/seed_sweep.py --seeds 0,1,2 \
        --script analysis/sanity/sc2_head_knockout.py \
        --out results/sc2_binary/head_knockout.json -- --n 20

Outputs
-------
* <out>.seed{S}.json   per-seed copy of the script's output
* <out>.sweep.json     {path: {mean, std, n, values}} for every numeric leaf
and prints a mean +/- std table to stdout.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.repro import get_seeds


def flatten(obj, prefix=""):
    """Yield (dotted_path, float) for every numeric leaf (bools excluded)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path to the target script")
    ap.add_argument("--out", required=True, help="JSON file the script writes")
    ap.add_argument("--seeds", default=None, help="comma-separated; default = SEEDS env")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="args after `--` are forwarded to the target script")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else get_seeds()
    fwd = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    out = Path(args.out)

    per_seed = {}
    for s in seeds:
        env = {**os.environ, "SEED": str(s)}
        print(f"\n===== seed {s} =====", flush=True)
        r = subprocess.run([sys.executable, args.script, *fwd], env=env)
        if r.returncode != 0:
            print(f"  seed {s} FAILED (exit {r.returncode}); skipping", flush=True)
            continue
        if not out.exists():
            print(f"  seed {s}: expected output {out} not found; skipping", flush=True)
            continue
        snap = out.with_suffix(out.suffix + f".seed{s}.json")
        shutil.copy(out, snap)
        per_seed[s] = dict(flatten(json.loads(out.read_text())))

    if len(per_seed) < 1:
        sys.exit("no successful runs to aggregate")

    keys = sorted(set().union(*[d.keys() for d in per_seed.values()]))
    agg = {}
    for k in keys:
        vals = [per_seed[s][k] for s in per_seed if k in per_seed[s]]
        agg[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "values": vals,
        }

    sweep = out.with_suffix(out.suffix + ".sweep.json")
    sweep.write_text(json.dumps({"seeds": list(per_seed), "metrics": agg}, indent=2))

    print(f"\n===== sweep over seeds {list(per_seed)} =====")
    print(f"{'metric':<48} {'mean':>12} {'std':>10}  n")
    for k in keys:
        a = agg[k]
        print(f"{k:<48} {a['mean']:>12.4f} {a['std']:>10.4f}  {a['n']}")
    print(f"\nsaved {sweep}")


if __name__ == "__main__":
    main()
