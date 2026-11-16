"""Measurement axes, Pauli matrices and basis-rotation angles.

Convention: outcome o = +1 <-> computational bit '0', o = -1 <-> bit '1'.
For a unit vector n we build U_n with U_n^dagger Z U_n = n . sigma, so that measuring Z
after U_n is a sharp measurement of n . sigma.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (X, Y, Z)

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])

CUBE_DIAGONALS = tuple(
    np.array(v, dtype=float) / np.sqrt(3)
    for v in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
)


def dot_sigma(n) -> np.ndarray:
    """n . sigma for a real 3-vector n."""
    n = np.asarray(n, dtype=float)
    return n[0] * X + n[1] * Y + n[2] * Z


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    nrm = np.linalg.norm(v)
    if not np.isfinite(nrm) or nrm < 1e-14:
        raise ValueError("cannot normalise a zero vector")
    return v / nrm


def rotation_angles(n) -> tuple[float, float]:
    """(alpha, beta) with n = (sin b cos a, sin b sin a, cos b).

    The circuit rz(-alpha) followed by ry(-beta) implements U_n = Ry(-beta) Rz(-alpha),
    and U_n^dagger Z U_n = Rz(alpha) Ry(beta) Z Ry(-beta) Rz(-alpha) = n . sigma.
    """
    n = unit(n)
    beta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    alpha = float(np.arctan2(n[1], n[0])) if np.hypot(n[0], n[1]) > 1e-15 else 0.0
    return alpha, beta


def check_theta_deg(theta_deg: float) -> float:
    """Validate an angle in degrees, theta in [0, 180]; return it as float."""
    try:
        t = float(theta_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid theta {theta_deg!r}") from exc
    if not np.isfinite(t) or t < 0.0 or t > 180.0:
        raise ValueError(f"theta must lie in [0, 180] degrees, got {theta_deg!r}")
    return t


def pair_vectors(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """a = Z axis, b = (sin theta, 0, cos theta)."""
    t = np.radians(check_theta_deg(theta_deg))
    return EZ.copy(), np.array([np.sin(t), 0.0, np.cos(t)])
