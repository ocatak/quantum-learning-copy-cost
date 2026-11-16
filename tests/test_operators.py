"""Exact operator checks (tests 1-4): POVM reconstruction from circuits, rotations, threshold, edge cases."""
import numpy as np
import pytest

from jointpovm import schemes as S
from jointpovm.axes import I2, PAULIS, dot_sigma, pair_vectors
from jointpovm.circuits import measurement_projectors, rotation_unitary
from jointpovm.povm import analytic_pair_povm, analytic_xyz_povm, effective_povm, marginal

THETAS = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
TOL = 1e-10


def _check_povm(E):
    for e in E.values():
        assert np.allclose(e, e.conj().T, atol=TOL)
        assert np.linalg.eigvalsh(e).min() > -TOL
    assert np.allclose(sum(E.values()), I2, atol=TOL)


@pytest.mark.parametrize("theta", THETAS)
def test_pair_povm_from_circuits(theta):
    sch = S.pair_joint(theta)
    E = effective_povm(sch)
    _check_povm(E)
    a, b = pair_vectors(theta)
    eta = S.pair_eta(theta)
    assert np.allclose(marginal(E, 0), (I2 + eta * dot_sigma(a)) / 2, atol=TOL)
    assert np.allclose(marginal(E, 1), (I2 + eta * dot_sigma(b)) / 2, atol=TOL)
    Ea = analytic_pair_povm(theta)
    for key in Ea:
        assert np.allclose(E[key], Ea[key], atol=TOL), key
    # state-independent joint statistic: sum_jk jk E_jk = x I
    assert np.allclose(sum(j * k * e for (j, k), e in E.items()), S.pair_x(theta) * I2, atol=TOL)


def test_xyz_povm_from_circuits():
    E = effective_povm(S.xyz_joint())
    _check_povm(E)
    for k, P in enumerate(PAULIS):
        assert np.allclose(marginal(E, k), (I2 + P / np.sqrt(3)) / 2, atol=TOL)
    Ea = analytic_xyz_povm()
    for key in Ea:
        assert np.allclose(E[key], Ea[key], atol=TOL), key


def _all_axes():
    axes = []
    for th in THETAS + [120.0, 180.0]:
        axes += S.pair_joint(th).axes + S.pair_grouped(th).axes
    axes += S.xyz_joint().axes + S.xyz_grouped().axes
    rng = np.random.default_rng(0)
    axes += list(v / np.linalg.norm(v) for v in rng.normal(size=(20, 3)))
    axes += [np.array([0, 0, -1.0]), np.array([0, -1.0, 0]), np.array([-1.0, 0, 0])]
    return axes


def test_rotations():
    Z = PAULIS[2]
    for n in _all_axes():
        U = rotation_unitary(n)
        assert np.allclose(U.conj().T @ Z @ U, dot_sigma(n), atol=1e-12, rtol=0), n
        P = measurement_projectors(n)
        assert np.allclose(P[0] - P[1], dot_sigma(n), atol=1e-12, rtol=0)


@pytest.mark.parametrize("theta", THETAS)
def test_threshold_closed_form(theta):
    t = np.radians(theta)
    assert np.isclose(S.pair_eta(theta), 1 / (np.cos(t / 2) + np.sin(t / 2)), atol=1e-15)
    assert np.isclose(1 / S.pair_eta(theta) ** 2, 1 + np.sin(t), atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 20.0, 90.0])
def test_threshold_sdp(theta):
    pytest.importorskip("cvxpy")
    from jointpovm.sdp import eta_star
    a, b = pair_vectors(theta)
    assert abs(eta_star([dot_sigma(a), dot_sigma(b)]) - S.pair_eta(theta)) < 1e-6


def test_sdp_xyz():
    pytest.importorskip("cvxpy")
    from jointpovm.sdp import eta_star
    assert abs(eta_star(list(PAULIS)) - 1 / np.sqrt(3)) < 1e-6


def test_edge_cases():
    s0 = S.pair_joint(0.0)
    assert s0.axis_labels == ["n1"] and np.allclose(s0.probs, [1.0])
    assert s0.n_axes == 1                       # no n2 circuit is built at theta = 0
    s180 = S.pair_joint(180.0)
    assert s180.axis_labels == ["n2"]
    assert np.isclose(S.pair_eta(90.0), 1 / np.sqrt(2), atol=1e-15)
    assert np.isclose(S.pair_x(90.0), 0.0, atol=1e-15)
    for bad in (-1.0, 180.5, np.nan, np.inf, "abc", None):
        with pytest.raises(ValueError):
            S.pair_joint(bad)
        with pytest.raises(ValueError):
            S.pair_eta(bad)


def test_multinomial_allocation_not_stratified():
    """Shot counts per axis must vary between seeds (i.i.d. axis choice), and sum to N."""
    from jointpovm.sampling import Config, allocate
    cfg = Config("x", S.pair_joint(45.0), "ry35", 10_000)
    counts = []
    for seed in range(5):
        sp = allocate(cfg, np.random.default_rng(seed))
        assert sum(s.shots for s in sp) == 10_000
        counts.append(tuple(s.shots for s in sp))
    assert len(set(counts)) > 1
    g = allocate(Config("g", S.pair_grouped(45.0), "ry35", 10_001), np.random.default_rng(0))
    assert [s.shots for s in g] == [5001, 5000]
