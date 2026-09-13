"""Centralized reproducibility / seeding for the whole project.

Every stochastic script should, near the top (after imports), do:

    from src.repro import set_seed, get_seed
    set_seed()                      # seeds python / numpy / torch / cuda

and then derive any *local* generator from the master seed instead of a
hardcoded constant, e.g.:

    rng = np.random.default_rng(get_seed())
    clf = LogisticRegression(random_state=get_seed())
    boot_rng = np.random.default_rng(get_seed() + 12345)   # independent stream

The single source of truth is the ``SEED`` environment variable (loaded from a
``.env`` file if present; see ``.env.example``). It defaults to ``0`` so that the
numbers reported in the paper reproduce exactly. Change ``SEED`` to re-run the
whole pipeline under a different draw; set ``SEEDS=0,1,2,3,4`` and use
``analysis/sanity/seed_sweep.py`` to obtain mean +/- std across seeds.

Design notes
------------
* Offsets (``get_seed() + 12345`` etc.) keep independent RNG streams *and* keep
  the seed-0 values identical to the previously hardcoded constants, so wiring
  this in does not move any already-published number.
* ``DETERMINISTIC=1`` additionally forces torch's deterministic algorithms. It
  is off by default because it can raise on a few CUDA ops and slow things down;
  for inference (the bulk of this project) cudnn determinism is enough.
"""

import os
import random
from typing import List, Optional

# Load .env once, mirroring src/models.py. Safe if python-dotenv is missing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a declared dependency
    pass

import numpy as np

DEFAULT_SEED = 0


def _truthy(val: Optional[str]) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def get_seed() -> int:
    """Master seed from the ``SEED`` env var (default ``0``)."""
    raw = os.environ.get("SEED")
    if raw is None or raw.strip() == "":
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"SEED must be an integer, got {raw!r}")


def get_seeds() -> List[int]:
    """List of seeds for a multi-seed sweep.

    Reads ``SEEDS`` (comma-separated, e.g. ``0,1,2,3,4``); falls back to
    ``[get_seed()]`` so single-seed runs need no extra config.
    """
    raw = os.environ.get("SEEDS")
    if raw is None or raw.strip() == "":
        return [get_seed()]
    return [int(x) for x in raw.replace(" ", "").split(",") if x != ""]


def set_seed(seed: Optional[int] = None, deterministic: Optional[bool] = None) -> int:
    """Seed python, numpy, and torch (CPU + CUDA). Returns the seed used.

    Parameters
    ----------
    seed: explicit seed; if ``None``, uses :func:`get_seed`.
    deterministic: force torch deterministic algorithms; if ``None``, reads the
        ``DETERMINISTIC`` env var (default off).
    """
    if seed is None:
        seed = get_seed()
    if deterministic is None:
        deterministic = _truthy(os.environ.get("DETERMINISTIC"))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cudnn determinism is cheap and safe for the inference workloads here.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except ImportError:
        pass

    return seed
