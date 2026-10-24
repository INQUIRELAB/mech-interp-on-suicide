"""Subspace estimators for the rank-identifiability study (Phase 3).

Every estimator returns an erasure in the low-rank affine form used by interventions.erase_fn,
    x' = x - ((x - mu) @ V) @ U.T        U, V: [d, k], mu: [d] or None,
so orthogonal projection (U = V = R) and oblique erasure (LEACE, U != V) share one hook.

  deflation   iterative difference of means, orthonormalized (what the paper used)
  inlp        iterative nullspace projection with a closed-form ridge classifier
  leace       LEACE (Belrose et al. 2023). For a binary concept the least-squares eraser is
              rank 1, so it is only defined at k = 1: if it removes the behavior, the
              behavior's linear information is one-dimensional under oblique erasure,
              whatever rank orthogonal projection needs
  random      random orthonormal subspace (control)
  das         distributed alignment search (learned; rank_ident.py, needs the model)
"""
import numpy as np


def _orth(cols):
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q.astype(np.float32)                     # [d, k]


def deflation(Xp, Xn, k):
    """WARNING: degenerate for k >= 2. After projecting u1 = dhat/|dhat| out, the in-sample
    residual difference of means is exactly zero, so directions 2..k are normalized
    floating-point residue (measured: ~1e-5 of |dhat|, 20-35% of mass on the 8 largest
    activation coordinates; deflation_diagnostic.py). Kept only to reproduce the paper."""
    rp, rn = Xp.astype(np.float64), Xn.astype(np.float64)
    cols = []
    for _ in range(k):
        u = rp.mean(0) - rn.mean(0)
        u /= np.linalg.norm(u) + 1e-12
        cols.append(u)
        rp = rp - np.outer(rp @ u, u)
        rn = rn - np.outer(rn @ u, u)
    R = _orth(cols)
    return R, R


def inlp(Xp, Xn, k, ridge=1e-2):
    X = np.vstack([Xp, Xn]).astype(np.float64)
    y = np.r_[np.ones(len(Xp)), -np.ones(len(Xn))]
    X = X - X.mean(0)
    lam = ridge * np.trace(X.T @ X) / X.shape[1]
    cols = []
    for _ in range(k):
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        w /= np.linalg.norm(w) + 1e-12
        for c in cols:                               # keep the basis orthogonal
            w -= (w @ c) * c
        w /= np.linalg.norm(w) + 1e-12
        cols.append(w)
        X = X - np.outer(X @ w, w)
    R = _orth(cols)
    return R, R


def leace(Xp, Xn, shrink=1e-3):
    """Rank-1 LEACE eraser for a binary concept. Returns (U, V, mu) with U, V: [d, 1]."""
    X = np.vstack([Xp, Xn]).astype(np.float64)
    z = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / len(X)
    cov += shrink * np.trace(cov) / len(cov) * np.eye(len(cov))
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    W = (evecs * evals ** -0.5) @ evecs.T            # Sigma^{-1/2}
    W_pinv = (evecs * evals ** 0.5) @ evecs.T        # Sigma^{1/2}
    sxz = Xc.T @ (z - z.mean()) / len(X)             # cross-covariance [d]
    u = W @ sxz
    u /= np.linalg.norm(u) + 1e-12
    U = (W_pinv @ u)[:, None]
    V = (W @ u)[:, None]
    return U.astype(np.float32), V.astype(np.float32), mu.astype(np.float32)


def random_subspace(d, k, rng):
    R = np.linalg.qr(rng.standard_normal((d, k)))[0].astype(np.float32)
    return R, R


def top_coordinates(X, k):
    """The k largest mean-|activation| coordinates as a basis (massive-activation control)."""
    idx = np.argsort(np.abs(X).mean(0))[::-1][:k]
    E = np.zeros((X.shape[1], k), np.float32)
    E[idx, np.arange(k)] = 1.0
    return E, E


def cluster_dom(Xp, Xn, k, seed=0):
    """Subtype difference of means: k-means the positive class into k clusters and take each
    cluster mean minus the negative-class mean. k = 1 is the ordinary DoM direction. Unlike
    deflation it returns k informative directions when the concept has subtypes."""
    if k == 1:
        u = Xp.mean(0) - Xn.mean(0)
        R = _orth([u / (np.linalg.norm(u) + 1e-12)])
        return R, R
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Xp.astype(np.float32))
    mn = Xn.mean(0)
    cols = [Xp[lab == c].mean(0) - mn for c in range(k) if (lab == c).any()]
    R = _orth([c / (np.linalg.norm(c) + 1e-12) for c in cols])
    return R, R


def erase_np(X, U, V, mu=None):
    Xc = X if mu is None else X - mu
    return X - (Xc @ V) @ U.T


ESTIMATORS = ("deflation", "inlp", "cluster_dom", "leace", "random", "top_coords")


def estimate(name, Xp, Xn, k, rng=None):
    """Uniform entry point. Returns (U, V) for orthogonal estimators, (U, V, mu) for leace."""
    if name == "deflation":
        return deflation(Xp, Xn, k)
    if name == "inlp":
        return inlp(Xp, Xn, k)
    if name == "leace":
        if k != 1:
            raise ValueError("LEACE for a binary concept is rank 1")
        return leace(Xp, Xn)
    if name == "random":
        return random_subspace(Xp.shape[1], k, rng)
    if name == "top_coords":
        return top_coordinates(np.vstack([Xp, Xn]), k)
    if name == "cluster_dom":
        return cluster_dom(Xp, Xn, k)
    raise ValueError(name)
