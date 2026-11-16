"""Weakly incompatible qubit pairs.

m qubits, each carrying two unbiased observables a.sigma and b.sigma at angle theta (K = 2m).
Task: estimate all K correlations to accuracy U with probability >= 0.9 (maximally mixed
data, uniform labels). Learners:
  grouped  two sharp contexts (cover number G = 2), B/2 copies each;
  joint    the joint POVM of Proposition 4 at the threshold sharpness, on every copy;
  oracle   non-physical, both sharp outcomes on every copy.

All functions take the random generator explicitly. Calling them in the order of
notebooks/03_qubit_pairs.ipynb with seed 7 reproduces the paper data.
"""
from __future__ import annotations

import numpy as np

M_DEFAULT, U_DEFAULT, T_DEFAULT = 10, 0.05, 6000


def max_error(B: int, theta: float, learner: str, rng: np.random.Generator,
              m: int = M_DEFAULT, T: int = T_DEFAULT) -> np.ndarray:
    """T Monte Carlo draws of max_k |estimate_k - mu_k| (mu_k = 0). theta in radians."""
    if learner == "oracle":
        za = 2 * rng.binomial(B, 0.5, (T, m)) - B
        zb = 2 * rng.binomial(B, 0.5, (T, m)) - B
        return np.maximum(np.abs(za), np.abs(zb)).max(1) / B
    if learner == "grouped":
        n = B // 2
        za = 2 * rng.binomial(n, 0.5, (T, m)) - n
        zb = 2 * rng.binomial(B - n, 0.5, (T, m)) - (B - n)
        return np.maximum(np.abs(za) / n, np.abs(zb) / (B - n)).max(1)
    if learner == "joint":
        # P(j,k) = (1 + jk x)/4 on the maximally mixed state; estimator = outcome / eta
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        eta = 1 / (c + s)
        x = eta * (c - s)
        probs = np.array([(1 + x) / 4, (1 - x) / 4, (1 - x) / 4, (1 + x) / 4])  # ++, +-, -+, --
        N = rng.multinomial(B, probs, size=(T, m))
        sa = N[..., 0] + N[..., 1] - N[..., 2] - N[..., 3]
        sb = N[..., 0] - N[..., 1] + N[..., 2] - N[..., 3]
        return np.maximum(np.abs(sa), np.abs(sb)).max(1) / (B * eta)
    raise ValueError(learner)


def b90(theta: float, learner: str, rng: np.random.Generator, U: float = U_DEFAULT,
        lo: int = 50, hi: int = 200_000) -> int:
    """Smallest budget (1% bisection resolution) with Pr[max error <= U] >= 0.9."""
    while hi / lo > 1.01:
        mid = int(np.sqrt(lo * hi))
        if np.mean(max_error(mid, theta, learner, rng) <= U) >= 0.9:
            hi = mid
        else:
            lo = mid
    return hi
