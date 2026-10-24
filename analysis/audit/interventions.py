"""Directions, null families, and interventions for the audit.

Null families for every causal rung (each produces unit directions per layer):
  isotropic       random unit directions                               (geometric baseline)
  pca             random directions inside the top-k PCA subspace       (SteerCheck)
  sign_random     difference of means with per-pair random signs         (SteerCheck; strongest)
  shuffled_label  difference of means after permuting the class labels   (What Would Falsify It?)

Erasure is written as a low-rank affine edit, h <- h - ((h - mu) V) U^T, which covers
zero-projection (mu=0, U=V=R), mean-projection (mu=pooled mean, U=V=R) and LEACE-style
oblique erasure (U != V) with one hook.
"""
import numpy as np
import torch

NULL_FAMILIES = ("isotropic", "pca", "sign_random", "shuffled_label")


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def diff_of_means(Hp, Hn):
    """Hp, Hn: [n, L+1, d] -> unit directions [L+1, d] and class-mean projection gap [L+1]."""
    raw = Hp.mean(0, dtype=np.float64) - Hn.mean(0, dtype=np.float64)
    dirs = unit(raw)
    gap = (raw * dirs).sum(-1)
    return dirs.astype(np.float32), gap.astype(np.float32)


class NullSampler:
    """Draws null unit directions per layer. Per-layer data and PCA bases are computed once."""

    def __init__(self, Hp, Hn, hidden_idx, pca_k=8):
        self.idx = list(hidden_idx)
        self.Hp = {j: Hp[:, j, :].astype(np.float32) for j in self.idx}
        self.Hn = {j: Hn[:, j, :].astype(np.float32) for j in self.idx}
        self.pca_k = pca_k
        self._pca = {}

    def _pca_basis(self, j):
        if j not in self._pca:
            X = np.vstack([self.Hp[j], self.Hn[j]])
            X = X - X.mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            self._pca[j] = Vt[: self.pca_k]
        return self._pca[j]

    def _one(self, family, j, rng):
        Hp_l, Hn_l = self.Hp[j], self.Hn[j]
        if family == "isotropic":
            return unit(rng.standard_normal(Hp_l.shape[1]))
        if family == "pca":
            return unit(rng.standard_normal(self.pca_k) @ self._pca_basis(j))
        if family == "sign_random":
            m = min(len(Hp_l), len(Hn_l))
            D = Hp_l[:m] - Hn_l[:m]                 # arbitrary pairing; classes are pre-shuffled
            eps = rng.choice([-1.0, 1.0], size=m)
            return unit(eps @ D / m)
        if family == "shuffled_label":
            X = np.vstack([Hp_l, Hn_l])
            perm = rng.permutation(len(X))
            return unit(X[perm[: len(Hp_l)]].mean(0) - X[perm[len(Hp_l):]].mean(0))
        raise ValueError(family)

    def sample(self, family, rng):
        """One draw: {hidden index j: unit direction [d]}."""
        return {j: self._one(family, j, rng).astype(np.float32) for j in self.idx}


# ---------------------------------------------------------------- hooks
def _follow(*tensors):
    """Per-device copies, so hooks work when a model is split across GPUs."""
    cache = {}
    def get(dev):
        if dev not in cache:
            cache[dev] = [None if t is None else t.to(dev) for t in tensors]
        return cache[dev]
    return get


def erase_fn(U, V, mu=None):
    """h <- h - ((h - mu) @ V) @ U^T. U, V: torch [d, k]; mu: torch [d] or None."""
    # computed in float32: oblique (LEACE) edits overflow or lose precision in fp16 (NaN, rank_ident)
    get = _follow(*(None if t is None else t.float() for t in (U, V, mu)))
    def fn(h):
        U_, V_, mu_ = get(h.device)
        x = h.float() if mu_ is None else h.float() - mu_
        return (h.float() - (x @ V_) @ U_.T).to(h.dtype)
    return fn


def add_fn(vec):
    get = _follow(vec)
    def fn(h):
        return h + get(h.device)[0]
    return fn


def ablation_hooks(backend, dirs_by_j, mode="zero", mu_by_j=None):
    """Rank-1 ablation of dirs_by_j[j] at each hidden index j. mode: zero | mean."""
    hooks = {}
    for j, v in dirs_by_j.items():
        R = backend.to_dev(np.asarray(v).reshape(-1, 1))
        mu = backend.to_dev(mu_by_j[j]) if mode == "mean" else None
        hooks[j] = erase_fn(R, R, mu)
    return hooks


def steering_hooks(backend, dirs_by_j, scale_by_j, alpha):
    """Add alpha * scale_j * v_j at each hidden index j."""
    return {j: add_fn(backend.to_dev(alpha * float(scale_by_j[j]) * np.asarray(v)))
            for j, v in dirs_by_j.items()}


# ---------------------------------------------------------------- KL calibration
@torch.no_grad()
def mean_kl(backend, prompts, hooks, base_logprobs, batch=16):
    with backend.hooked(hooks):
        lg = backend.last_logits(prompts, batch)
    lp = torch.log_softmax(lg, -1)
    return float((base_logprobs.exp() * (base_logprobs - lp)).sum(-1).mean())


def calibrate_alpha(backend, prompts, make_hooks, base_logprobs, kappa, alpha0=1.0,
                    iters=4, tol=0.15, batch=16):
    """Find alpha with mean KL(base || steered) ~= kappa on neutral reference prompts.

    KL is close to quadratic in alpha for small edits, so alpha <- alpha * sqrt(kappa/KL)
    converges in a few steps. Returns (alpha, achieved_kl).
    """
    alpha, kl = alpha0, None
    for _ in range(iters):
        kl = mean_kl(backend, prompts, make_hooks(alpha), base_logprobs, batch)
        if kl <= 0:
            alpha *= 4.0
            continue
        if abs(kl - kappa) / kappa < tol:
            break
        alpha *= float(np.clip(np.sqrt(kappa / kl), 0.1, 10.0))
    return alpha, kl
