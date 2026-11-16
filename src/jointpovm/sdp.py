"""Joint-measurability threshold eta* by semidefinite programming (needs cvxpy).

Same formulation as copycost.sdp: maximise eta subject to a POVM {F_b}
whose marginals are (I + eta A_k)/2.
"""
from __future__ import annotations

import itertools

import numpy as np


def eta_star(observables) -> float:
    import cvxpy as cp
    d = observables[0].shape[0]
    G = len(observables)
    eta = cp.Variable()
    outs = list(itertools.product([1, -1], repeat=G))
    F = {b: cp.Variable((d, d), hermitian=True) for b in outs}
    cons = [F[b] >> 0 for b in outs] + [sum(F.values()) == np.eye(d)]
    for k in range(G):
        cons.append(sum(F[b] for b in outs if b[k] == 1) == (np.eye(d) + eta * observables[k]) / 2)
    prob = cp.Problem(cp.Maximize(eta), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    return float(eta.value)
