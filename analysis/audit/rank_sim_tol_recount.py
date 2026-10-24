"""Recount the gsa spectral rank in the planted-rank grid with numpy's matrix_rank tolerance.

The 2026-10-07 grid (rank_sim_gsa_nuis{0,0.3,1.0}.json) counted eigenvalues above the parallel-
analysis null without a numerical floor. At nuisance 0 the interchange matrix has rank <= 2k*, so
the count ran through numerical zeros. This recounts from the stored top-20 spectra (a recount
that reaches 20 is flagged as truncated) and writes rank_sim_gsa_tol_summary.json.

  python rank_sim_tol_recount.py <results/rank_ident dir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = np.finfo(float).eps


def recount(ev, thr, d):
    ev, thr = np.asarray(ev), np.asarray(thr)
    tol = np.abs(ev).max() * d * EPS
    k = 0
    while k < len(ev) and ev[k] > max(thr[k], tol):
        k += 1
    return k


def main():
    root = Path(sys.argv[1])
    out = {}
    for g in ("0", "0.3", "1.0"):
        d = json.loads((root / f"rank_sim_gsa_nuis{g}.json").read_text())
        D = d["config"]["d"]
        cells = defaultdict(lambda: {"raw": [], "tol": []})
        for r in d["rows"]:
            if "k_spec" not in r:
                continue
            c = cells[(r["kstar"], r["n"])]
            c["raw"].append(r["k_spec"])
            c["tol"].append(recount(r["evals_top"], r["null_top"], D))
        rows = []
        for (ks, n), c in sorted(cells.items()):
            rows.append({"kstar": ks, "n": n, "seeds": len(c["tol"]),
                         "raw_median": float(np.median(c["raw"])),
                         "raw_frac_correct": float(np.mean([k == ks for k in c["raw"]])),
                         "tol_median": float(np.median(c["tol"])), "tol_min": min(c["tol"]),
                         "tol_max": max(c["tol"]),
                         "tol_frac_correct": float(np.mean([k == ks for k in c["tol"]])),
                         "truncated": max(c["tol"]) >= 20})
            print(f"nuis {g:>3} k*={ks} n={n:5d} raw {rows[-1]['raw_median']:5.0f} "
                  f"ok {rows[-1]['raw_frac_correct']:.1f} | tol {rows[-1]['tol_median']:3.0f} "
                  f"[{rows[-1]['tol_min']}-{rows[-1]['tol_max']}] ok {rows[-1]['tol_frac_correct']:.1f}")
        out[g] = rows
    (root / "rank_sim_gsa_tol_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
