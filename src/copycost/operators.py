"""Anticommuting Pauli families and exact numerical checks of the joint POVM.

Checks (for each block size G):
  (i)   pairwise anticommutation of the Jordan-Wigner operators;
  (ii)  the joint POVM E_s = (I + sum_i s_i A_i / sqrt(G)) / 2^G is positive, complete and
        has marginals (I + A_i/sqrt(G))/2;
  (iii) sum_i <A_i>^2 <= 1 on random pure states (Lemma on anticommuting observables);
  (iv)  the Born rule of the hard instance rho = (I + s eps A_0)/d under the joint POVM is
        2^{-G} (1 + s eps b_0 / sqrt(G)), the law used by the selection simulator.
"""
from __future__ import annotations

import itertools

import numpy as np

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1]).astype(complex)


def kron(*ops):
    out = np.array([[1.0]], complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def anticommuting_set(q: int) -> list[np.ndarray]:
    """2q+1 pairwise-anticommuting Pauli strings on q qubits (Jordan-Wigner Majoranas plus parity)."""
    ops = []
    for j in range(q):
        ops.append(kron(*([Z] * j + [X] + [I2] * (q - j - 1))))
        ops.append(kron(*([Z] * j + [Y] + [I2] * (q - j - 1))))
    ops.append(kron(*([Z] * q)))
    return ops


def joint_povm(A: list[np.ndarray]) -> dict[tuple, np.ndarray]:
    """Joint POVM of G anticommuting observables at sharpness 1/sqrt(G)."""
    G, d = len(A), A[0].shape[0]
    return {s: (np.eye(d) + sum(si * Ai for si, Ai in zip(s, A)) / np.sqrt(G)) / 2 ** G
            for s in itertools.product([1, -1], repeat=G)}


def verify_block(G: int, q: int, rng: np.random.Generator, n_states: int = 2000, eps: float = 0.3) -> dict:
    """Run checks (i)-(iv) for the first G operators on q qubits. Raises AssertionError on failure."""
    A = anticommuting_set(q)[:G]
    d = 2 ** q
    for i in range(G):
        for j in range(G):
            assert np.allclose(A[i] @ A[j] + A[j] @ A[i], 2 * np.eye(d) * (i == j))
    E = joint_povm(A)
    min_eig = min(np.linalg.eigvalsh(e).min() for e in E.values())
    assert min_eig > -1e-12
    complete_err = np.abs(sum(E.values()) - np.eye(d)).max()
    assert complete_err < 1e-12
    marg_err = 0.0
    for i in range(G):
        marg = sum(e for s, e in E.items() if s[i] == 1)
        marg_err = max(marg_err, np.abs(marg - (np.eye(d) + A[i] / np.sqrt(G)) / 2).max())
    assert marg_err < 1e-12
    worst = 0.0
    for _ in range(n_states):
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v /= np.linalg.norm(v)
        worst = max(worst, sum(np.real(v.conj() @ Ai @ v) ** 2 for Ai in A))
    assert worst <= 1 + 1e-12
    born_err = 0.0
    for s0 in (1, -1):
        rho = (np.eye(d) + s0 * eps * A[0]) / d
        for b, e in E.items():
            born_err = max(born_err, abs(np.real(np.trace(e @ rho)) - 2 ** -G * (1 + s0 * eps * b[0] / np.sqrt(G))))
    assert born_err < 1e-12
    return {"G": G, "qubits": q, "dim": d, "min_eigenvalue": float(min_eig),
            "completeness_err": float(complete_err), "marginal_err": float(marg_err),
            "max_sum_sq_expectations": float(worst), "born_rule_err": float(born_err)}
