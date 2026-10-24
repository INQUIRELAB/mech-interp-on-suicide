#!/usr/bin/env python
"""Is iterative difference-of-means deflation degenerate after k = 1?

After projecting u1 = dhat/|dhat| out of both classes, the residual difference of means is
dhat - (dhat.u1) u1 = 0 exactly in the same sample, so directions 2..k of deflation are
normalized floating-point residue. This script measures that on real activations, using the
same deflation code as analysis/sanity/sc2_rank_sweep.py (float32, eps=1e-8), and checks
where the residue directions put their mass: on the few massive-activation coordinates, or
spread like a random direction.

Usage: python deflation_diagnostic.py --model Qwen/Qwen2.5-1.5B-Instruct --concept suicide --n 150
"""
import argparse
import json

import numpy as np

from backend import Backend
from concepts import get_concept


def paper_deflate(A, B, k):
    """Verbatim logic of sc2_rank_sweep.deflate_subspace, also returning pre-normalization norms."""
    A, B = A.copy(), B.copy()
    V, norms = [], []
    for _ in range(k):
        d = A.mean(0) - B.mean(0)
        norms.append(float(np.linalg.norm(d)))
        d = d / (np.linalg.norm(d) + 1e-8)
        V.append(d)
        A = A - np.outer(A @ d, d)
        B = B - np.outer(B @ d, d)
    Q, _ = np.linalg.qr(np.stack(V, axis=1))
    return Q[:, :k], norms


def mass_on(v, idx):
    return float((v[idx] ** 2).sum() / (v ** 2).sum())


def participation_ratio(v):
    p = v ** 2 / (v ** 2).sum()
    return float(1.0 / (p ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concept", default="suicide")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="deflation_diagnostic.json")
    a = ap.parse_args()
    c = get_concept(a.concept)
    pos, neg = c.loader(a.n, 0)
    be = Backend(a.model)
    yes, no = be.token_id(c.pos_answer), be.token_id(c.neg_answer)
    P = lambda ts: [be.polar_prompt(t, c.prefix, c.question) for t in ts]
    _, Hp = be.collect(P(pos), yes, no, verbose=False)
    _, Hn = be.collect(P(neg), yes, no, verbose=False)
    nL = be.n_layers
    rng = np.random.default_rng(0)
    rows = []
    for j in sorted({nL // 4, nL // 2, (3 * nL) // 4}):
        A, B = Hp[:, j].astype(np.float32), Hn[:, j].astype(np.float32)
        Q, norms = paper_deflate(A, B, a.k)
        scale = np.abs(np.vstack([A, B])).mean(0)
        top = np.argsort(scale)[::-1][:8]                  # massive-activation coordinates
        r = rng.standard_normal(A.shape[1])
        row = {"hidden_index": j, "d": A.shape[1],
               "prenorm_residual_norms": norms,
               "ratio_k2_to_k1": norms[1] / norms[0],
               "top8_coord_share_of_activation_scale": float(scale[top].sum() / scale.sum()),
               "dir_mass_on_top8": [mass_on(Q[:, i], top) for i in range(a.k)],
               "dir_participation_ratio": [participation_ratio(Q[:, i]) for i in range(a.k)],
               "random_mass_on_top8": mass_on(r, top),
               "random_participation_ratio": participation_ratio(r)}
        # fresh float64 deflation for comparison: is direction 2 the same "direction"?
        Q64, n64 = paper_deflate(A.astype(np.float64), B.astype(np.float64), a.k)
        row["float64_prenorm_norms"] = n64
        row["abs_cos_dir2_f32_vs_f64"] = float(abs(Q[:, 1] @ Q64[:, 1]))
        rows.append(row)
        print(f"L{j}: |dDoM| k=1 {norms[0]:.3g}, k=2 {norms[1]:.3g} (ratio {row['ratio_k2_to_k1']:.1e}); "
              f"mass on top-8 coords: dir1 {row['dir_mass_on_top8'][0]:.3f}, "
              f"dirs2-{a.k} {np.mean(row['dir_mass_on_top8'][1:]):.3f}, random {row['random_mass_on_top8']:.4f}; "
              f"PR dir1 {row['dir_participation_ratio'][0]:.0f}, dirs2+ "
              f"{np.mean(row['dir_participation_ratio'][1:]):.0f}, random {row['random_participation_ratio']:.0f}; "
              f"|cos| dir2 f32 vs f64 {row['abs_cos_dir2_f32_vs_f64']:.3f}")
    json.dump({"model": a.model, "concept": a.concept, "n": a.n, "rows": rows},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
