#!/usr/bin/env python
"""Re-run the paper-backing sc2 suite in fp16 and archive the results separately.

Every sc2_* script hardcodes load_in_4bit=True (they were written for a 12 GB
3060) and writes into results/sc2_binary/. This runner sets MI_PRECISION=fp16 --
which src.models.ModelLoader honours over the caller's arguments -- and, after
each script, copies whatever files that script touched into

    results/sc2_binary_fp16/

so the existing 4-bit JSONs stay intact for the precision comparison table.
Skips the scripts that need packages the deimos venv does not carry
(bitsandbytes for the quantized paths, sae_lens for the SAE).

Usage:
    python analysis/sanity/run_fp16_verify.py                  # core suite
    python analysis/sanity/run_fp16_verify.py --only faithfulness rankk
    python analysis/sanity/run_fp16_verify.py --precision bf16
    python analysis/sanity/run_fp16_verify.py --dry-run
"""
import sys, os, json, time, shutil, argparse, subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent.parent
SANITY = PROJECT_ROOT / "analysis" / "sanity"
SRC_RESULTS = PROJECT_ROOT / "results" / "sc2_binary"

# The rungs the paper actually reports, cheapest first so a broken environment
# surfaces in minutes rather than hours. head_knockout last: it is the slow one.
SUITE = [
    ("gate",                  "sc2_binary_gate.py",              []),
    ("specificity",           "sc2_specificity.py",              []),
    ("faithfulness_perlayer", "sc2_faithfulness_perlayer.py",    []),
    ("faithfulness",          "sc2_faithfulness.py",             []),
    ("rankk",                 "sc2_rankk.py",                    []),
    ("sufficiency",           "sc2_sufficiency.py",              []),
    ("gradient",              "sc2_gradient.py",                 []),
    ("generalization",        "sc2_generalization.py",           []),
    ("crossablation",         "sc2_crossablation.py",            []),
    ("downstream_table",      "sc2_downstream_table.py",         []),
    ("ci",                    "sc2_ci.py",                       []),
    ("umd_severity",          "sc2_umd_severity.py",             []),
    ("steering_specificity",  "sc2_steering_specificity.py",     []),
    ("steering_salience",     "gcl/steering_salience.py",        []),
    ("secondmodel_qwen",      "sc2_secondmodel.py",              ["qwen"]),
    ("secondmodel_mistral",   "sc2_secondmodel.py",              ["mistral"]),
    ("patching",              "sc2_patching.py",                 []),
    ("attribution_patching",  "sc2_attribution_patching.py",     []),
    # --n 60 matches the sample size the reported null was measured at; the
    # script's own default is 20, which would silently weaken it.
    ("head_knockout",         "sc2_head_knockout.py",            ["--n", "60"]),
    # --n MUST match the sweep above: the stats script re-uses the same examples
    # by seed, and the CIs are meaningless if the two disagree.
    ("head_knockout_stats",   "sc2_head_knockout_stats.py",      ["--n", "60"]),
]

# Needs sae_lens, which the deimos venv does not have. Run elsewhere or pip it.
SKIP_BY_DEFAULT = {"sae"}

PER_SCRIPT_TIMEOUT = 9000  # 2.5 h


def snapshot(d):
    """{path: mtime_ns} for everything currently in the results dir."""
    if not d.exists():
        return {}
    return {p: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these names")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these names")
    ap.add_argument("--precision", default="fp16",
                    choices=["fp16", "bf16", "8bit", "4bit"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without loading a model")
    args = ap.parse_args()

    dest = PROJECT_ROOT / "results" / f"sc2_binary_{args.precision}"
    logdir = dest / "logs"
    skip = set(args.skip) | SKIP_BY_DEFAULT
    suite = [s for s in SUITE
             if (not args.only or s[0] in args.only) and s[0] not in skip]

    print(f"precision : {args.precision}")
    print(f"results   : {dest}")
    print(f"scripts   : {len(suite)}")
    for name, script, extra in suite:
        print(f"   {name:24s} {script} {' '.join(extra)}")
    if args.dry_run:
        return

    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MI_PRECISION=args.precision, PYTHONUNBUFFERED="1")

    manifest, t_suite = [], time.perf_counter()
    for name, script, extra in suite:
        path = SANITY / script
        if not path.exists():
            print(f"[skip] {name}: {script} not found")
            continue

        before = snapshot(SRC_RESULTS)
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        log = logdir / f"{name}.log"
        with log.open("w", encoding="utf-8") as fh:
            try:
                rc = subprocess.call(
                    [sys.executable, str(path), *extra],
                    cwd=str(PROJECT_ROOT), env=env,
                    stdout=fh, stderr=subprocess.STDOUT,
                    timeout=PER_SCRIPT_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                rc = "timeout"
        dt = time.perf_counter() - t0

        # Anything the script created or rewrote belongs to this precision run.
        after = snapshot(SRC_RESULTS)
        moved = []
        for p, mtime in after.items():
            if before.get(p) != mtime:
                rel = p.relative_to(SRC_RESULTS)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
                moved.append(str(rel))

        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {status}  {dt/60:.1f} min  {len(moved)} file(s) -> {dest.name}/")
        if rc != 0:
            print(f"  see {log}")
        manifest.append({"name": name, "script": script, "args": extra,
                         "rc": rc, "minutes": round(dt / 60, 2), "files": moved})

    total = (time.perf_counter() - t_suite) / 60
    (dest / "manifest.json").write_text(json.dumps({
        "precision": args.precision,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_minutes": round(total, 1),
        "runs": manifest,
    }, indent=2))

    ok = sum(1 for m in manifest if m["rc"] == 0)
    print(f"\n{ok}/{len(manifest)} succeeded in {total:.0f} min")
    print(f"manifest: {dest / 'manifest.json'}")
    print(f"\nnext: python analysis/sanity/compare_precision.py "
          f"--a results/sc2_binary --b {dest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
