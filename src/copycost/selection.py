"""Selection on the hard instances of the anticommuting family H_{m,G}.

K = mG candidates, hidden index h, label s uniform, state rho_{h,s} = (I + s eps A_h)/d.
For every learner the statistic z = s*b of candidate k is a +-1 variable with mean
eps*[k==h] (sharp) or eps*[k==h]/sqrt(G) (joint POVM at sharpness 1/sqrt(G)); the counts
are therefore binomial and the simulation needs no state vectors. By symmetry h = 0.

All functions take the random generator explicitly. Calling them in the order of
notebooks/02_anticommuting_selection.ipynb with seed 20261114 reproduces the paper data.
"""
from __future__ import annotations

import numpy as np

K_DEFAULT = 90
EPS_DEFAULT = 0.10


def success_nonadaptive(B: int, G: int, learner: str, rng: np.random.Generator,
                        T: int = 4000, K: int = K_DEFAULT, eps: float = EPS_DEFAULT) -> float:
    """Pr[argmax empirical correlation = h] for a non-adaptive learner with budget B.

    learner = 'grouped': round-robin over G sharp contexts; context c measures element c of every block.
              'joint'  : one joint POVM per block at sharpness 1/sqrt(G) on every copy.
              'oracle' : non-physical, all K sharp outcomes on every copy (the G = 1 baseline).
    """
    m = K // G
    if learner == "grouped":
        n = np.full(G, B // G)
        n[:B % G] += 1                                  # copies per context
        nk = np.repeat(n[None, :], m, axis=0).ravel()   # candidate (j, c) -> index j*G + c
        bias = eps
    elif learner == "joint":
        nk = np.full(K, B)
        bias = eps / np.sqrt(G)
    elif learner == "oracle":
        nk = np.full(K, B)
        bias = eps
    else:
        raise ValueError(learner)
    p = np.full(K, 0.5)
    p[0] = (1 + bias) / 2
    S = rng.binomial(np.broadcast_to(nk, (T, K)), np.broadcast_to(p, (T, K))) / np.maximum(nk, 1)
    S = S + 1e-9 * rng.random((T, K))                   # uniform tie-breaking
    return float(np.mean(S.argmax(1) == 0))


def success_seqhalving(B: int, G: int, rng: np.random.Generator,
                       T: int = 4000, K: int = K_DEFAULT, eps: float = EPS_DEFAULT) -> float:
    """Adaptive reference: sequential halving. Each of ceil(log2 K) rounds spends B/R copies
    round-robin on the contexts that still contain a surviving candidate, then keeps the better
    half (fresh data each round)."""
    ctx = np.tile(np.arange(G), K // G)                 # context of candidate k = j*G + c
    R = int(np.ceil(np.log2(K)))
    Br = B // R
    alive = np.ones((T, K), bool)
    p = np.full(K, 0.5)
    p[0] = (1 + eps) / 2
    for _ in range(R):
        need = np.zeros((T, G), bool)
        for c in range(G):
            need[:, c] = (alive & (ctx == c)).any(1)
        nc = need.sum(1)                                # contexts in use per trial
        base = Br // nc
        nk = np.where(alive, base[:, None], 0)
        S = rng.binomial(nk, np.broadcast_to(p, (T, K))) / np.maximum(nk, 1) + 1e-9 * rng.random((T, K))
        S[~alive] = -np.inf
        keep = int(np.ceil(alive[0].sum() / 2))
        idx = np.argsort(-S, 1)[:, :keep]
        alive = np.zeros((T, K), bool)
        np.put_along_axis(alive, idx, True, 1)
        if keep == 1:
            break
    return float(np.mean(alive[:, 0]))


def budget_grid() -> np.ndarray:
    """34 log-spaced budgets between 60 and 2.5e5 copies (Fig. 2)."""
    return np.unique(np.round(np.logspace(np.log10(60), np.log10(2.5e5), 34)).astype(int))


def b90(fun, G: int, lo: float = 10, hi: float = 2e6) -> float:
    """Budget at which fun(B, G) reaches 0.9: bisection in log B on a monotone (noisy) curve,
    stopped at 2% relative resolution. Returns the upper end of the bracket."""
    for _ in range(26):
        mid = int(np.sqrt(lo * hi))
        if fun(mid, G) >= 0.9:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.02:
            break
    return hi


def b90_repeatability(learner: str, G_values=(1, 2, 3, 5, 6, 9, 10, 15, 18), reps: int = 5, T: int = 6000,
                      K: int = K_DEFAULT, eps: float = EPS_DEFAULT) -> dict:
    """Repeatability of B_0.9 (Section 6): the bisection of ``b90`` rerun with independent seeds
    1000 + 17 r + G for run r and context number G. Reproduces reference/b90_reps.json."""
    G_values = list(G_values)
    vals = np.zeros((reps, len(G_values)))
    for r in range(reps):
        for i, G in enumerate(G_values):
            rng = np.random.default_rng(1000 + 17 * r + G)
            vals[r, i] = b90(lambda B, G_: success_nonadaptive(B, G_, learner, rng, T=T, K=K, eps=eps), G)
    g = np.array(G_values, float)
    norm = vals * eps ** 2 / (g * np.log(K))
    slopes = [(g @ v) / (g @ g) for v in vals]
    rel = vals.std(0, ddof=1) / vals.mean(0)
    return {"b90": vals.tolist(), "slope_mean": float(np.mean(slopes)), "slope_sd": float(np.std(slopes, ddof=1)),
            "max_rel_sd": float(rel.max()), "norm_min": float(norm.min()), "norm_max": float(norm.max())}
