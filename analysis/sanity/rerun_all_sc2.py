#!/usr/bin/env python
"""Sequential re-run of the full binary-task (sc2_binary) experiment suite.

Runs each paper-backing script as its own subprocess (one model in VRAM at a
time, 12 GB-safe), captures stdout/stderr to a per-script log, records timing
and exit status, and writes a manifest + markdown summary at the end. Continues
on failure so one broken script does not abort the whole sweep.

Usage:
    python analysis/sanity/rerun_all_sc2.py            # full suite
    python analysis/sanity/rerun_all_sc2.py --only gate faithfulness_perlayer
    python analysis/sanity/rerun_all_sc2.py --skip head_knockout sae
"""
import sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
SANITY = PROJECT_ROOT / "analysis" / "sanity"
LOGDIR = PROJECT_ROOT / "results" / "sc2_binary" / "rerun_logs"

# (name, script, args). Ordered fast/core first, heavy last.
SUITE = [
    ("gate",                 "sc2_binary_gate.py",          []),
    ("specificity",          "sc2_specificity.py",          []),
    ("faithfulness_perlayer","sc2_faithfulness_perlayer.py",[]),
    ("faithfulness",         "sc2_faithfulness.py",         []),
    ("rankk",                "sc2_rankk.py",                []),
    ("sufficiency",          "sc2_sufficiency.py",          []),
    ("gradient",             "sc2_gradient.py",             []),
    ("generalization",       "sc2_generalization.py",       []),
    ("crossablation",        "sc2_crossablation.py",        []),
    ("ci",                   "sc2_ci.py",                   []),
    ("patching",             "sc2_patching.py",             []),
    ("logitlens",            "sc2_logitlens.py",            []),
    ("dump_examples",        "dump_examples.py",            []),
    ("secondmodel_qwen",     "sc2_secondmodel.py",          ["qwen"]),
    ("secondmodel_mistral",  "sc2_secondmodel.py",          ["mistral"]),
    ("sae",                  "sc2_sae.py",                  []),
    ("head_knockout",        "sc2_head_knockout.py",        []),
    ("head_knockout_stats",  "sc2_head_knockout_stats.py",  []),
    # --- new MI methods (written this session) ---
    ("umd_severity",         "sc2_umd_severity.py",         []),
    ("attribution_patching", "sc2_attribution_patching.py", []),
    ("direct_logit_attr",    "sc2_direct_logit_attribution.py", []),
]

PER_SCRIPT_TIMEOUT = 9000  # 2.5 h ceiling per script (head knockout is the slow one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only these names")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these names")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    suite = [s for s in SUITE if (not args.only or s[0] in args.only)
             and s[0] not in args.skip]
    # Skip new-method scripts gracefully if not yet written.
    suite = [s for s in suite if (SANITY / s[1]).exists()]

    manifest = {"started": datetime.now().isoformat(), "results": []}
    print(f"Re-running {len(suite)} scripts; logs -> {LOGDIR}")

    for name, script, extra in suite:
        log_path = LOGDIR / f"{name}.log"
        cmd = [sys.executable, str(SANITY / script), *extra]
        print(f"\n{'='*64}\n[{datetime.now():%H:%M:%S}] {name}: {' '.join(cmd)}\n{'='*64}")
        t0 = time.time()
        status, rc = "ok", 0
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=lf,
                                      stderr=subprocess.STDOUT, text=True,
                                      timeout=PER_SCRIPT_TIMEOUT)
            rc = proc.returncode
            status = "ok" if rc == 0 else "fail"
        except subprocess.TimeoutExpired:
            status, rc = "timeout", -1
        except Exception as e:
            status, rc = f"error:{e}", -2
        dt = time.time() - t0
        # tail the log for a quick verdict line
        tail = ""
        try:
            tail = "".join(open(log_path, encoding="utf-8").readlines()[-3:]).strip()
        except Exception:
            pass
        manifest["results"].append({"name": name, "script": script, "args": extra,
                                    "status": status, "returncode": rc,
                                    "seconds": round(dt, 1), "log": str(log_path),
                                    "tail": tail})
        print(f"[{datetime.now():%H:%M:%S}] {name}: {status} in {dt/60:.1f} min")
        # checkpoint manifest after every script
        manifest["updated"] = datetime.now().isoformat()
        json.dump(manifest, open(LOGDIR / "manifest.json", "w"), indent=2)

    # markdown summary
    ok = sum(1 for r in manifest["results"] if r["status"] == "ok")
    lines = [f"# sc2_binary full re-run — {manifest['started']}",
             f"\n{ok}/{len(manifest['results'])} scripts OK\n",
             "| script | status | minutes |", "|---|---|---|"]
    for r in manifest["results"]:
        lines.append(f"| {r['name']} | {r['status']} | {r['seconds']/60:.1f} |")
    (LOGDIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE. {ok}/{len(manifest['results'])} OK. See {LOGDIR/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
