"""Incompatibility number kappa (Definition 3, Algorithm 1 of the paper).

Algorithm 1 has two stages:
  1. for every candidate group C, the joint-measurability threshold eta*(C): 1 for a single
     observable or a commuting group (Proposition 1(ii)), otherwise a semidefinite program;
  2. the covering linear program
         kappa = min sum_C w_C   s.t.   sum_{C contains k} w_C eta*(C)^2 >= 1 for every k,  w >= 0.

This module also computes the partition version kappa_part, the cover number G (chromatic
number of the conflict graph), Lambda = max_psi sum_k <psi|A_k|psi>^2 and the subfamily
lower bound max_{A'} |A'| / Lambda(A') of Proposition 5, and reruns the 5-cycle example
(Section 5) and the random-family test of Open problem 2 (Section 7).

All randomised functions take the generator explicitly. Calling them in the order of
notebooks/05_kappa_and_checks.ipynb with seed 11 reproduces reference/kappa_cover.json.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1])}
FIVE_CYCLE = ["IX", "XI", "IY", "ZY", "YX"]   # neighbours commute, non-neighbours anticommute


def pauli(label: str) -> np.ndarray:
    """Pauli string, e.g. 'XZ' -> X (x) Z."""
    out = np.array([[1]])
    for c in label:
        out = np.kron(out, PAULI[c])
    return out


def commute(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.linalg.norm(A @ B - B @ A) < tol)


def is_commuting(A: list[np.ndarray], S) -> bool:
    return all(commute(A[i], A[j]) for i, j in itertools.combinations(S, 2))


def eta_star_group(A: list[np.ndarray], S) -> float:
    """eta*(S) by semidefinite programming (Definition 3). Checks the solver status and falls
    back to SCS if Clarabel does not report an optimal solution."""
    import cvxpy as cp
    if len(S) == 1:
        return 1.0
    d = A[0].shape[0]
    outs = list(itertools.product([1, -1], repeat=len(S)))
    F = {b: cp.Variable((d, d), hermitian=True) for b in outs}
    eta = cp.Variable()
    cons = [F[b] >> 0 for b in outs] + [sum(F.values()) == np.eye(d)]
    for i, k in enumerate(S):
        cons.append(sum(F[b] for b in outs if b[i] == 1) == (np.eye(d) + eta * A[k]) / 2)
    prob = cp.Problem(cp.Maximize(eta), cons)
    with warnings.catch_warnings():               # an inaccurate solve is caught by the status check below
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        prob.solve(solver=cp.CLARABEL)
        if prob.status != cp.OPTIMAL:
            prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"SDP for {S} ended with status {prob.status}")
    return float(eta.value)


def thresholds(A: list[np.ndarray], groups=None) -> dict:
    """Stage 1 of Algorithm 1: eta*(C) for every group C (all nonempty subsets by default)."""
    K = len(A)
    if groups is None:
        groups = [S for r in range(1, K + 1) for S in itertools.combinations(range(K), r)]
    return {tuple(S): (1.0 if is_commuting(A, S) else eta_star_group(A, S)) for S in groups}


def kappa_lp(K: int, eta: dict) -> tuple[float, dict]:
    """Stage 2 of Algorithm 1: the covering LP. Returns kappa_w and the cover {group: weight}.
    With all groups this is kappa(A); with a shorter list it is a feasible cover, kappa_w >= kappa."""
    from scipy.optimize import linprog
    groups = list(eta)
    A_ub = np.zeros((K, len(groups)))
    for j, S in enumerate(groups):
        for k in S:
            A_ub[k, j] = -eta[S] ** 2
    res = linprog(np.ones(len(groups)), A_ub=A_ub, b_ub=-np.ones(K), bounds=[(0, None)] * len(groups))
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(res.fun), {S: float(w) for S, w in zip(groups, res.x) if w > 1e-9}


def partitions(items: list):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def kappa_part_and_G(A: list[np.ndarray], eta: dict) -> tuple[float, int]:
    """Partition version kappa_part = min over partitions of sum_l eta*(C_l)^-2, and the cover
    number G = smallest partition into commuting groups (brute force; small families only)."""
    kp, G = np.inf, np.inf
    for p in partitions(list(range(len(A)))):
        kp = min(kp, sum(eta[tuple(sorted(g))] ** -2 for g in p))
        if all(eta[tuple(sorted(g))] == 1.0 and is_commuting(A, g) for g in p):
            G = min(G, len(p))
    return float(kp), int(G)


def Lambda(A: list[np.ndarray], rng: np.random.Generator, starts: int = 200, iters: int = 200) -> float:
    """Lambda = max over pure states of sum_k <A_k>^2, by fixed-point iteration from random starts
    (psi <- top eigenvector of sum_k <A_k> A_k). A lower estimate that is exact at convergence."""
    d = A[0].shape[0]
    best = 0.0
    for _ in range(starts):
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v /= np.linalg.norm(v)
        for _ in range(iters):
            e = [np.real(v.conj() @ a @ v) for a in A]
            _, U = np.linalg.eigh(sum(x * a for x, a in zip(e, A)))
            v = U[:, -1]
        best = max(best, sum(np.real(v.conj() @ a @ v) ** 2 for a in A))
    return float(best)


def subfamily_bound(A: list[np.ndarray], labels: list[str], rng: np.random.Generator):
    """max over nonempty subfamilies A' of |A'| / Lambda(A') (Proposition 5) and the maximiser."""
    best, arg = 0.0, None
    for m in range(1, len(A) + 1):
        for S in itertools.combinations(range(len(A)), m):
            val = m / Lambda([A[i] for i in S], rng, starts=60, iters=120)
            if val > best + 1e-9:
                best, arg = val, [labels[i] for i in S]
    return float(best), arg


def analyse(labels: list[str], rng: np.random.Generator, with_subfamilies: bool = True) -> dict:
    """All quantities of Section 5 for a family of Pauli strings."""
    A = [pauli(s) for s in labels]
    K = len(labels)
    eta = thresholds(A)
    kap, cover = kappa_lp(K, eta)
    kp, G = kappa_part_and_G(A, eta)
    lam = Lambda(A, rng)
    out = {"family": list(labels), "K": K, "G": G, "kappa": kap, "kappa_part": kp, "Lambda": lam,
           "K_over_Lambda": K / lam, "eta_full": eta[tuple(range(K))],
           "cover": [{"group": [labels[k] for k in S], "weight": w, "eta": eta[S]} for S, w in cover.items()]}
    sharp = {S: 1.0 for S in eta if is_commuting(A, S)}   # randomised sharp covers only
    out["kappa_sharp_only"] = kappa_lp(K, sharp)[0]
    if with_subfamilies:
        out["subfamily_bound"], out["subfamily"] = subfamily_bound(A, labels, rng)
    return out


def random_families(rng: np.random.Generator, sizes=(5,) * 20 + (6,) * 10, verbose: bool = False) -> list[dict]:
    """The test of Open problem 2: random families of distinct non-identity two-qubit Pauli strings."""
    labels2 = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]
    fams = []
    for K in sizes:
        fam = [str(s) for s in rng.choice(labels2, size=K, replace=False)]
        r = analyse(fam, rng)
        fams.append(r)
        if verbose:
            print(f"{' '.join(fam):24s} G={r['G']} kappa={r['kappa']:.3f} kappa_part={r['kappa_part']:.3f} "
                  f"K/Lambda={r['K_over_Lambda']:.3f} max_sub={r['subfamily_bound']:.3f}", flush=True)
    return fams
