"""Born-rule and typical-instance checks of Section 6, family H_{m,3} with q = 1 (X, Y, Z on each of m qubits).

(1) Hard instances. ``selection`` draws the statistic of each candidate as an independent binomial with
    bias eps (grouped) or eps/sqrt(G) (joint). Here the outcome probabilities are computed from the
    measurement operators by the Born rule: the sharp Pauli projectors for the grouped learner and the eight
    cube-diagonal effects F_b = (I + b.sigma/sqrt 3)/8 for the joint learner (Algorithm 2, JointXYZ).
    Counts are drawn per qubit (binomial for a sharp context, multinomial over the 8 joint outcomes), which
    keeps the correlations between the three outcomes of the joint POVM.
(2) Typical instances. Random product data: y uniform, rho_y = tensor_j (I + y r_j.sigma)/2 with r_j uniform
    in the ball of radius 0.6, so mu_{j,i} = r_{j,i}. Mean excess risk of both learners against the budget.

Copies are i.i.d., so outcome counts are sufficient statistics and drawing counts is exact. Multiplying every
outcome of a copy by its label y gives the same law as measuring the state with Bloch vector y r, so the label
is folded into r.

All functions take the generator explicitly. Calling ``hard_instance_check`` and then ``typical_instances``
with one generator seeded 20261117 reproduces reference/born_rule_check.json.
"""
from __future__ import annotations

import numpy as np

SIG = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
I2 = np.eye(2)
BS = np.array([[b0, b1, b2] for b0 in (1, -1) for b1 in (1, -1) for b2 in (1, -1)])   # joint outcomes
F = [(I2 + sum(b[i] * SIG[i] for i in range(3)) / np.sqrt(3)) / 8 for b in BS]          # eq. (4), G = 3
ETA = 1 / np.sqrt(3)
assert np.allclose(sum(F), I2) and all(np.linalg.eigvalsh(f).min() > -1e-12 for f in F)


def rho(r) -> np.ndarray:
    return (I2 + sum(r[i] * SIG[i] for i in range(3))) / 2


def p_plus_sharp(r, i: int) -> float:
    """Born rule: probability of +1 for a sharp measurement of sigma_i."""
    return float(np.real(np.trace((I2 + SIG[i]) / 2 @ rho(r))))


def p_joint(r) -> np.ndarray:
    return np.array([float(np.real(np.trace(f @ rho(r)))) for f in F])


def estimates(Rb: np.ndarray, B: int, learner: str, T: int, rng: np.random.Generator) -> np.ndarray:
    """Estimates mu_hat (T x m x 3) for Bloch vectors Rb (m x 3) from B copies (Algorithm 3)."""
    m = Rb.shape[0]
    mu = np.zeros((T, m, 3))
    if learner == "grouped":           # round robin over the 3 contexts; context i measures sigma_i on every qubit
        n = np.full(3, B // 3)
        n[:B % 3] += 1
        for j in range(m):
            for i in range(3):
                k = rng.binomial(n[i], p_plus_sharp(Rb[j], i), size=T)
                mu[:, j, i] = (2 * k - n[i]) / max(n[i], 1)
        return mu
    for j in range(m):
        N = rng.multinomial(B, p_joint(Rb[j]), size=T)          # T x 8 counts
        mu[:, j, :] = (N @ BS) / (B * ETA)                      # z = b_i / eta, averaged
    return mu


def binomial_model_success(B: int, m: int, eps: float, learner: str, T: int, rng: np.random.Generator) -> float:
    """The binomial model of ``selection.success_nonadaptive`` with G = 3, for comparison."""
    K = 3 * m
    if learner == "grouped":
        n = np.full(3, B // 3)
        n[:B % 3] += 1
        nk, bias = np.tile(n, m), eps
    else:
        nk, bias = np.full(K, B), eps / np.sqrt(3)
    p = np.full(K, 0.5)
    p[0] = (1 + bias) / 2
    S = rng.binomial(np.broadcast_to(nk, (T, K)), np.broadcast_to(p, (T, K))) / np.maximum(nk, 1)
    S = S + 1e-9 * rng.random((T, K))
    return float(np.mean(S.argmax(1) == 0))


def hard_instance_check(rng: np.random.Generator, m: int = 10, eps: float = 0.1, T: int = 6000,
                        grid=(60, 150, 300, 600, 1200, 2400, 4800, 9600)) -> dict:
    """Success probability by the Born rule vs. the binomial model on the hard instance (h = sigma_x, qubit 0)."""
    Rb = np.zeros((m, 3))
    Rb[0, 0] = eps
    out = {"m": m, "K": 3 * m, "eps": eps, "T": T, "B": list(grid), "born": {}, "model": {}}
    for L in ["grouped", "joint"]:
        sb, sm = [], []
        for B in grid:
            mu = estimates(Rb, B, L, T, rng).reshape(T, -1) + 1e-9 * rng.random((T, 3 * m))
            sb.append(float(np.mean(mu.argmax(1) == 0)))
            sm.append(binomial_model_success(B, m, eps, L, T, rng))
        out["born"][L], out["model"][L] = sb, sm
        out[f"max_abs_diff_{L}"] = float(np.abs(np.array(sb) - np.array(sm)).max())
    out["two_se_bound"] = float(2 * np.sqrt(2 * 0.25 / T))
    return out


def typical_instances(rng: np.random.Generator, m: int = 10, n_inst: int = 400, T: int = 50,
                      radius: float = 0.6, grid=(90, 300, 900, 3000, 9000)) -> dict:
    """Mean excess risk of the grouped and joint learners on random product data."""
    res = {"m": m, "K": 3 * m, "instances": n_inst, "trials_per_instance": T, "radius": radius, "B": list(grid),
           "grouped": [], "joint": [], "grouped_se": [], "joint_se": []}
    insts = []
    for _ in range(n_inst):
        v = rng.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        insts.append(v * radius * rng.random((m, 1)) ** (1 / 3))   # uniform in the ball
    for L in ["grouped", "joint"]:
        for B in grid:
            ex = []
            for Rb in insts:
                mu_true = Rb.ravel()
                mu = estimates(Rb, B, L, T, rng).reshape(T, -1)
                ex.append(np.mean((mu_true.max() - mu_true[mu.argmax(1)]) / 2))
            ex = np.array(ex)
            res[L].append(float(ex.mean()))
            res[L + "_se"].append(float(ex.std(ddof=1) / np.sqrt(len(ex))))
    return res
