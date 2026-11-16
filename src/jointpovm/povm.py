"""Analytic joint POVMs and their reconstruction from the actual circuits."""
from __future__ import annotations

import itertools

import numpy as np

from .axes import I2, PAULIS, dot_sigma, pair_vectors
from .circuits import measurement_projectors
from .schemes import Scheme, pair_eta


def effective_povm(scheme: Scheme) -> dict:
    """Effective POVM of a joint scheme: E_v = sum_axis p_axis sum_o [table(axis,o) = v] P_o.

    Keys are tuples of reported +-1 outcomes (one per observable)."""
    if scheme.kind != "joint":
        raise ValueError("effective POVM is defined for joint schemes")
    E: dict = {}
    for ai, (n, p) in enumerate(zip(scheme.axes, scheme.probs)):
        P = measurement_projectors(n)
        for oi in (0, 1):
            key = tuple(int(v) for v in scheme.table[ai, oi])
            E[key] = E.get(key, np.zeros((2, 2), complex)) + p * P[oi]
    for key in itertools.product([1, -1], repeat=scheme.n_obs):
        E.setdefault(key, np.zeros((2, 2), complex))
    return E


def marginal(E: dict, k: int, value: int = 1) -> np.ndarray:
    return sum(e for key, e in E.items() if key[k] == value)


def analytic_pair_povm(theta_deg: float) -> dict:
    """E_jk = 1/4 [ (1 + j k x) I + eta (j a + k b) . sigma ], x = eta (c - s)."""
    a, b = pair_vectors(theta_deg)
    eta = pair_eta(theta_deg)
    t = np.radians(theta_deg)
    x = eta * (np.cos(t / 2) - np.sin(t / 2))
    return {(j, k): 0.25 * ((1 + j * k * x) * I2 + eta * dot_sigma(j * a + k * b))
            for j in (1, -1) for k in (1, -1)}


def analytic_xyz_povm() -> dict:
    """F_b = 2^-3 (I + b . sigma / sqrt 3), Eq. (4) of the paper with G = 3."""
    return {bb: (I2 + sum(bi * P for bi, P in zip(bb, PAULIS)) / np.sqrt(3)) / 8
            for bb in itertools.product([1, -1], repeat=3)}
