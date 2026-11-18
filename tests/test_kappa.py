"""Incompatibility number (Algorithm 1) against the closed forms of the paper."""
import numpy as np
import pytest

from copycost import kappa as kp

X, Z = kp.pauli("X"), kp.pauli("Z")


def kappa_of(A):
    return kp.kappa_lp(len(A), kp.thresholds(A))[0]


@pytest.mark.parametrize("theta_deg", [0, 20, 60, 90])
def test_qubit_pair_kappa_is_one_plus_sin(theta_deg):
    """Proposition 4: kappa = 1 + sin(theta)."""
    t = np.radians(theta_deg)
    A = [Z, np.cos(t) * Z + np.sin(t) * X]
    assert kappa_of(A) == pytest.approx(1 + np.sin(t), abs=1e-5)


def test_anticommuting_block_kappa_equals_G():
    """Proposition 3: X, Y, Z on one qubit, kappa = G = 3."""
    A = [kp.pauli(s) for s in "XYZ"]
    assert kappa_of(A) == pytest.approx(3, abs=1e-5)
    _, G = kp.kappa_part_and_G(A, kp.thresholds(A))
    assert G == 3


def test_commuting_family_kappa_is_one():
    A = [kp.pauli(s) for s in ["ZI", "IZ", "ZZ"]]
    assert kappa_of(A) == pytest.approx(1, abs=1e-9)


def test_restricted_group_list_gives_feasible_upper_bound():
    """Algorithm 1 with a shorter group list returns a feasible cover with kappa_w >= kappa."""
    A = [kp.pauli(s) for s in kp.FIVE_CYCLE]
    singletons = kp.thresholds(A, groups=[(k,) for k in range(5)])
    assert kp.kappa_lp(5, singletons)[0] == pytest.approx(5)


@pytest.mark.slow
def test_five_cycle():
    """Section 5: kappa = 5/2 = K/Lambda < kappa_part = 2.639 < G = 3."""
    A = [kp.pauli(s) for s in kp.FIVE_CYCLE]
    eta = kp.thresholds(A)
    kap, cover = kp.kappa_lp(5, eta)
    kpart, G = kp.kappa_part_and_G(A, eta)
    assert kap == pytest.approx(2.5, abs=1e-6)
    assert kpart == pytest.approx(2.6393, abs=1e-3)
    assert G == 3
    assert eta[tuple(range(5))] == pytest.approx(0.6155, abs=1e-3)
    assert kp.Lambda(A, np.random.default_rng(0), starts=40, iters=100) == pytest.approx(2, abs=1e-6)
