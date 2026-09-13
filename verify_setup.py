#!/usr/bin/env python
"""Verify that the environment can run the analyses in this repository.

    python verify_setup.py

Checks third-party dependencies, the project's own modules, GPU availability,
the seeding configuration, and (optionally) the dataset loader. Exits 0 when
everything required is present.

A missing GPU or an unavailable dataset is reported as a warning, not a
failure: the figure-rendering path (paper/make_figures.py) runs from committed
result files on CPU and needs neither.
"""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "[OK]  ", "[FAIL]", "[WARN]"


def check_dependencies():
    """Third-party packages required by the analysis scripts."""
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("dotenv", "python-dotenv"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
    ]
    ok = True
    for module, label in required:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "")
            print(f"  {OK} {label}{' ' + version if version else ''}")
        except Exception as exc:
            print(f"  {FAIL} {label}: {exc}")
            ok = False
    return ok


def check_project_modules():
    """This repository's own modules."""
    modules = [
        ("src.repro", "seeding (src.repro)"),
        ("src.models", "model loading (src.models)"),
        ("src.hf_adapter", "HuggingFace adapter (src.hf_adapter)"),
        ("src.analysis_utils", "analysis helpers (src.analysis_utils)"),
    ]
    ok = True
    for module, label in modules:
        try:
            importlib.import_module(module)
            print(f"  {OK} {label}")
        except Exception as exc:
            print(f"  {FAIL} {label}: {str(exc)[:110]}")
            ok = False
    return ok


def check_gpu():
    """Report GPUs. Absence is a warning: CPU is enough to render figures."""
    try:
        import torch
    except Exception as exc:
        print(f"  {FAIL} torch not importable: {exc}")
        return False

    try:
        if not torch.cuda.is_available():
            print(f"  {WARN} No CUDA GPU visible. Model experiments need one;")
            print("         paper/make_figures.py runs on CPU from result files.")
            return True
        count = torch.cuda.device_count()
        print(f"  {OK} CUDA available, {count} device(s), torch {torch.__version__}")
        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"         [{i}] {props.name}, {props.total_memory / 1e9:.0f} GB")
            except Exception as exc:
                print(f"  {WARN} device {i} could not be queried: {str(exc)[:70]}")
        largest = max(
            (torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(count)),
            default=0,
        )
        if largest < 16:
            print(f"  {WARN} Largest GPU is {largest:.0f} GB. An 8B model at fp16 needs")
            print("         about 16 GB; use --precision 4bit or a smaller --model.")
        return True
    except Exception as exc:
        print(f"  {WARN} GPU query failed: {str(exc)[:110]}")
        return True


def check_directories():
    """src/ and analysis/ must ship; results/ is created on demand."""
    ok = True
    for name in ("src", "analysis", "paper"):
        if (ROOT / name).is_dir():
            print(f"  {OK} {name}/")
        else:
            print(f"  {FAIL} {name}/ missing from the repository")
            ok = False
    results = ROOT / "results"
    if results.is_dir():
        print(f"  {OK} results/")
    else:
        results.mkdir(parents=True, exist_ok=True)
        print(f"  {OK} results/ created")
    return ok


def check_seeding():
    """Report the seed the run will use."""
    try:
        from src.repro import get_seed
        seed = get_seed()
        print(f"  {OK} SEED={seed}" + ("  (paper values reproduce at SEED=0)" if seed == 0 else ""))
        if not (ROOT / ".env").exists():
            print(f"  {WARN} No .env file. Copy .env.example to .env to set SEED,")
            print("         HF_TOKEN and data paths. Defaults are used meanwhile.")
        return True
    except Exception as exc:
        print(f"  {FAIL} seeding unavailable: {str(exc)[:110]}")
        return False


def check_dataset():
    """Optional: the loader may download on first use. Never fatal."""
    try:
        from src.suicide_dataset_loader import ImplicitSuicideDataset
    except Exception as exc:
        print(f"  {WARN} loader not importable: {str(exc)[:110]}")
        return True
    try:
        dataset = ImplicitSuicideDataset(split="train")
        if len(dataset) > 0:
            print(f"  {OK} dataset loaded, {len(dataset)} cases")
            clean, corrupt = dataset.build_clean_corrupt_pair(0)
            print(f"  {OK} clean/corrupt pairs built" if clean and corrupt
                  else f"  {WARN} clean/corrupt pair came back empty")
        else:
            print(f"  {WARN} dataset loaded but empty")
    except Exception as exc:
        print(f"  {WARN} dataset unavailable: {str(exc)[:110]}")
        print("         Datasets are not bundled; see README section 'Data'.")
    return True


def main():
    print("=" * 70)
    print("SETUP VERIFICATION")
    print("=" * 70)

    checks = {}
    print("\n1. Dependencies")
    checks["dependencies"] = check_dependencies()
    print("\n2. Project modules")
    checks["modules"] = check_project_modules()
    print("\n3. GPU")
    checks["gpu"] = check_gpu()
    print("\n4. Directories")
    checks["directories"] = check_directories()
    print("\n5. Reproducibility")
    checks["seeding"] = check_seeding()
    print("\n6. Dataset (optional)")
    checks["dataset"] = check_dataset()

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    print("\n" + "=" * 70)
    if passed == total:
        print(f"All {total} checks passed.")
        print("\nReproduce a single claim, for example the behavioral gate:")
        print("  MI_PRECISION=fp16 python analysis/sanity/sc2_binary_gate.py")
        print("Or the full binary-task suite:")
        print("  python analysis/sanity/run_fp16_verify.py")
        print("\nMI_PRECISION=fp16 matters: the scripts default to 4-bit.")
        print("See README.md for the claim-to-script mapping.")
        return 0

    print(f"{passed}/{total} checks passed. Failing:")
    for name, result in checks.items():
        if not result:
            print(f"  - {name}")
    print("\nInstall dependencies with: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
