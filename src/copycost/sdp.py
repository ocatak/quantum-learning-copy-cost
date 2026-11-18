"""Joint-measurability threshold by semidefinite programming.

eta*(A_1..A_G) = max eta  s.t. there is a POVM {F_b}_{b in {+-1}^G} with marginals (I + eta b_k A_k)/2.
For a single group, 1/eta*^2 is its cost in the covering program of Definition 3; the incompatibility
number kappa of a family is the value of that program over all groups (see ``copycost.kappa``).
"""
from __future__ import annotations

import itertools

import numpy as np

from .operators import X, Z


def eta_star(A: list[np.ndarray]) -> float:
    import cvxpy as cp
    d, G = A[0].shape[0], len(A)
    eta = cp.Variable()
    outs = list(itertools.product([1, -1], repeat=G))
    F = {b: cp.Variable((d, d), hermitian=True) for b in outs}
    cons = [F[b] >> 0 for b in outs] + [sum(F.values()) == np.eye(d)]
    for k in range(G):
        cons.append(sum(F[b] for b in outs if b[k] == 1) == (np.eye(d) + eta * A[k]) / 2)
    cp.Problem(cp.Maximize(eta), cons).solve(solver=cp.CLARABEL)
    return float(eta.value)


def pair_observables(theta_deg: float) -> list[np.ndarray]:
    """Z and cos(theta) Z + sin(theta) X: two qubit observables at angle theta on the Bloch sphere."""
    t = np.radians(theta_deg)
    return [Z, np.cos(t) * Z + np.sin(t) * X]


def pair_threshold(theta_deg):
    """Closed form of Proposition 4: 1/(cos(theta/2) + sin(theta/2))."""
    t = np.radians(theta_deg)
    return 1 / (np.cos(t / 2) + np.sin(t / 2))
