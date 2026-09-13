#!/usr/bin/env python
"""Driver script to run all sanity checks (SC-1 through SC-4).

Loads Llama once, runs all four sanity checks sequentially, and writes
a single summary with GO/NO-GO recommendation for Tier 1.

Output:
    results/sanity/summary.md - consolidated summary with decisions
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_script(script_path, output_dir, dataset="deepsuimind", extra_args=None):
    """Run a sanity check script and capture output.

    Args:
        script_path: path to the Python script
        output_dir: output directory for results
        dataset: dataset name
        extra_args: list of additional args for the script

    Returns:
        (success, output) tuple
    """
    cmd = [sys.executable, str(script_path), "--dataset", dataset, "--output-dir", str(output_dir)]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,  # 10 minute timeout per script
        )

        if result.returncode == 0:
            print(result.stdout)
            return True, result.stdout
        else:
            print(f"ERROR: Script failed with return code {result.returncode}")
            print(result.stderr)
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print("ERROR: Script timed out after 10 minutes")
        return False, "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def extract_sc1_results(sc1_dir):
    """Extract SC-1 results from JSON."""
    json_path = sc1_dir / "baseline_results.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    return {
        "accuracy": data.get("mean_accuracy", None),
        "top_death_me": data.get("top_death_me_ngrams", [])[:5],
        "top_life_not_me": data.get("top_life_not_me_ngrams", [])[:5],
    }


def extract_sc2_results(sc2_dir):
    """Extract SC-2 results from JSON."""
    json_path = sc2_dir / "accuracy.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    return {
        "death_acc": data.get("death_me_accuracy", None),
        "life_acc": data.get("life_not_me_accuracy", None),
        "overall_acc": data.get("overall_accuracy", None),
        "confusion": data.get("confusion_matrix", None),
    }


def extract_sc3_results(sc3_dir):
    """Extract SC-3 results from JSON."""
    json_path = sc3_dir / "confounds.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    # Check for major confounds
    has_length_confound = False
    if "Death-Me" in data and "Life-Not-Me" in data:
        death_len = data["Death-Me"]["char_length_mean"]
        life_len = data["Life-Not-Me"]["char_length_mean"]
        ratio = death_len / life_len if life_len > 0 else 1.0
        has_length_confound = ratio > 1.5 or ratio < 1/1.5

    return {
        "has_length_confound": has_length_confound,
        "death_me_length": data.get("Death-Me", {}).get("char_length_mean") if "Death-Me" in data else None,
        "life_not_me_length": data.get("Life-Not-Me", {}).get("char_length_mean") if "Life-Not-Me" in data else None,
    }


def extract_sc4_results(sc4_dir):
    """Extract SC-4 results from bug report."""
    report_path = sc4_dir / "position_bug_report.md"
    if not report_path.exists():
        return None

    with open(report_path) as f:
        content = f.read()

    # Check for Phase 11 bug
    has_phase11_bug = "CONFIRMED" in content and "Phase 11" in content

    return {
        "has_phase11_bug": has_phase11_bug,
    }


def make_decision(sc1, sc2, sc3, sc4):
    """Make GO/NO-GO decision based on all sanity checks.

    Returns:
        (decision, reasons) tuple
        decision is one of: "GO", "PIVOT", "NO-GO"
    """
    reasons = []

    # SC-1: High accuracy means we should pivot
    if sc1 and sc1["accuracy"] is not None:
        if sc1["accuracy"] >= 0.90:
            return "PIVOT", [
                "SC-1: TF-IDF baseline accuracy >= 90% - the 'circuit' is just keyword detection",
                "Recommendation: Pivot to Tier 3 (refusal-circuit reframing)"
            ]
        elif sc1["accuracy"] < 0.75:
            reasons.append("SC-1: Low baseline accuracy - task genuinely requires inference")

    # SC-2: Model must be able to do the task
    if sc2:
        min_acc = float('inf')
        if sc2["death_acc"] is not None:
            min_acc = min(min_acc, sc2["death_acc"])
        if sc2["life_acc"] is not None:
            min_acc = min(min_acc, sc2["life_acc"])

        if min_acc < 0.60:
            return "NO-GO", [
                f"SC-2: Model accuracy < 60% - model cannot do the task reliably",
                "Recommendation: Try different prompt or change model"
            ]
        elif min_acc < 0.85:
            reasons.append(f"SC-2: Model accuracy {min_acc:.2%} - metrics noisy, flag in downstream plots")

    # SC-3: Document confounds
    if sc3 and sc3["has_length_confound"]:
        reasons.append("SC-3: MAJOR CONFOUND - length differs by >1.5x between categories")
        reasons.append("     All length-sensitive metrics are contaminated")

    # SC-4: Phase 11 bug
    if sc4 and sc4["has_phase11_bug"]:
        reasons.append("SC-4: Phase 11 bug confirmed - [-1, :] indexes pad token")
        reasons.append("     All Phase 11 severity axis results are invalid")

    if not reasons or "Low baseline accuracy" in reasons[0]:
        return "GO", reasons if reasons else ["All sanity checks passed or within acceptable range"]

    return "GO", reasons


def write_summary(results_dir, sc1, sc2, sc3, sc4, decision, reasons):
    """Write the summary markdown file."""
    summary_path = results_dir / "summary.md"

    with open(summary_path, "w") as f:
        f.write("# Sanity Check Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Verdict\n\n")
        f.write(f"**{decision}**\n\n")

        f.write("### Reasons\n\n")
        for reason in reasons:
            f.write(f"- {reason}\n")
        f.write("\n")

        f.write("## SC-1: Lexical Baseline\n\n")
        if sc1:
            f.write(f"- Accuracy: **{sc1['accuracy']:.4f}**\n")
            if sc1['accuracy'] >= 0.90:
                f.write("- **WARNING**: Baseline too high - circuit may just be keyword detection\n")
            elif sc1['accuracy'] < 0.75:
                f.write("- Task genuinely requires inference\n")

            f.write(f"\nTop 5 Death-Me n-grams:\n")
            for word, coef in sc1['top_death_me']:
                f.write(f"  - {coef:>8.4f} {word}\n")

            f.write(f"\nTop 5 Life-Not-Me n-grams:\n")
            for word, coef in sc1['top_life_not_me']:
                f.write(f"  - {coef:>8.4f} {word}\n")
        else:
            f.write("- Results not available\n")
        f.write("\n")

        f.write("## SC-2: Model Task Accuracy\n\n")
        if sc2:
            f.write(f"- Death-Me accuracy: **{sc2['death_acc']:.4f}**\n")
            f.write(f"- Life-Not-Me accuracy: **{sc2['life_acc']:.4f}**\n")
            f.write(f"- Overall accuracy: **{sc2['overall_acc']:.4f}**\n")
            f.write(f"\nConfusion matrix:\n")
            f.write(f"```\n")
            f.write(f"              Pred Yes  Pred No\n")
            f.write(f"Death-Me         {sc2['confusion'][0][0]:4d}     {sc2['confusion'][0][1]:4d}\n")
            f.write(f"Life-Not-Me      {sc2['confusion'][1][0]:4d}     {sc2['confusion'][1][1]:4d}\n")
            f.write(f"```\n")
        else:
            f.write("- Results not available\n")
        f.write("\n")

        f.write("## SC-3: Surface Confounds\n\n")
        if sc3:
            if sc3['has_length_confound']:
                f.write("- **MAJOR CONFOUND**: Length differs by >1.5x between categories\n")
                f.write("- All length-sensitive metrics (residual norm, attention entropy) are contaminated\n")
            else:
                f.write("- No major length confound detected\n")
            if sc3['death_me_length'] and sc3['life_not_me_length']:
                f.write(f"- Death-Me mean length: {sc3['death_me_length']:.1f} chars\n")
                f.write(f"- Life-Not-Me mean length: {sc3['life_not_me_length']:.1f} chars\n")
        else:
            f.write("- Results not available\n")
        f.write("\n")

        f.write("## SC-4: Position Effect\n\n")
        if sc4:
            if sc4['has_phase11_bug']:
                f.write("- **BUG CONFIRMED**: Phase 11 uses [-1, :] indexing on padded tensors\n")
                f.write("- This indexes the LAST PAD TOKEN, not the answer token\n")
                f.write("- All Phase 11 'severity axis' results are invalid\n")
            else:
                f.write("- No Phase 11 bug detected\n")
        else:
            f.write("- Results not available\n")
        f.write("\n")

        f.write("## Next Steps\n\n")
        if decision == "PIVOT":
            f.write("1. Do not proceed to Tier 1\n")
            f.write("2. Pivot to Tier 3 (refusal-circuit reframing) as per plan/00_sanity_checks.md\n")
        elif decision == "NO-GO":
            f.write("1. Fix the blocking issue (model accuracy) before continuing\n")
            f.write("2. Re-run sanity checks after fix\n")
        else:
            f.write("1. Proceed to Tier 1 (circuit validation) per plan/01_circuit_validation.md\n")
            f.write("2. Keep confound warnings in mind for all Tier 1 experiments\n")
        f.write("\n")

    print(f"\nSaved summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run all sanity checks")
    parser.add_argument(
        "--dataset", type=str, default="deepsuimind",
        choices=["deepsuimind", "swmh", "extended_swmh"],
        help="Dataset to use (default: deepsuimind)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sanity/)"
    )
    parser.add_argument(
        "--skip", type=str, nargs="*",
        help="Skip specific sanity checks (e.g., --skip sc1 sc3)"
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "sanity"
    else:
        output_dir = Path(args.output_dir)
    from src.analysis_utils import ensure_dir
    ensure_dir(output_dir)

    print("="*60)
    print("Running All Sanity Checks")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")

    sanity_dir = PROJECT_ROOT / "analysis" / "sanity"

    # Track which checks to skip
    skip_set = set(args.skip) if args.skip else set()

    # Run each sanity check
    results = {}

    if "sc1" not in skip_set:
        sc1_dir = output_dir / "sc1"
        ensure_dir(sc1_dir)
        success, _ = run_script(
            sanity_dir / "sc1_lexical_baseline.py",
            sc1_dir,
            args.dataset,
        )
        results["sc1"] = extract_sc1_results(sc1_dir) if success else None
    else:
        print("\n*** SKIPPING SC-1 ***")
        results["sc1"] = None

    if "sc2" not in skip_set:
        sc2_dir = output_dir / "sc2"
        ensure_dir(sc2_dir)
        success, _ = run_script(
            sanity_dir / "sc2_task_accuracy.py",
            sc2_dir,
            args.dataset,
        )
        results["sc2"] = extract_sc2_results(sc2_dir) if success else None
    else:
        print("\n*** SKIPPING SC-2 ***")
        results["sc2"] = None

    if "sc3" not in skip_set:
        sc3_dir = output_dir / "sc3"
        ensure_dir(sc3_dir)
        success, _ = run_script(
            sanity_dir / "sc3_surface_confounds.py",
            sc3_dir,
            args.dataset,
        )
        results["sc3"] = extract_sc3_results(sc3_dir) if success else None
    else:
        print("\n*** SKIPPING SC-3 ***")
        results["sc3"] = None

    if "sc4" not in skip_set:
        sc4_dir = output_dir / "sc4"
        ensure_dir(sc4_dir)
        success, _ = run_script(
            sanity_dir / "sc4_position_effect.py",
            sc4_dir,
            args.dataset,
            ["--tokenizer", "meta-llama/Llama-3.1-8B"],
        )
        results["sc4"] = extract_sc4_results(sc4_dir) if success else None
    else:
        print("\n*** SKIPPING SC-4 ***")
        results["sc4"] = None

    # Make decision
    decision, reasons = make_decision(
        results["sc1"], results["sc2"], results["sc3"], results["sc4"]
    )

    # Write summary
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"**{decision}**")
    print("\nReasons:")
    for reason in reasons:
        print(f"  - {reason}")

    write_summary(
        output_dir,
        results["sc1"],
        results["sc2"],
        results["sc3"],
        results["sc4"],
        decision,
        reasons,
    )

    print("="*60)

    return 0 if decision == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
