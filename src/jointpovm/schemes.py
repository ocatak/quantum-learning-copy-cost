"""Measurement schemes (joint randomized-basis POVMs and grouped sharp measurements).

A scheme is a list of measurement axes plus a classical post-processing table
    table[axis, o_index, obs] in {+1, -1, 0},
which maps the sharp outcome o (o_index 0 <-> o=+1, 1 <-> o=-1) on an axis to the
reported +-1 outcome of each observable (0 = observable not reported on that axis).

* joint schemes draw the axis at random for every copy (i.i.d., probabilities `probs`);
  every observable is reported on every copy, and the unbiased estimator is outcome/eta.
* grouped schemes split the copies deterministically and evenly over the axes; each
  observable is read sharply (eta = 1) on its own share of the copies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .axes import CUBE_DIAGONALS, EX, EY, EZ, check_theta_deg, pair_vectors, unit


@dataclass
class Scheme:
    name: str
    kind: str                      # 'joint' or 'grouped'
    axes: list                     # list of unit 3-vectors
    axis_labels: list
    probs: np.ndarray | None       # joint: i.i.d. axis probabilities; grouped: None
    table: np.ndarray              # (n_axes, 2, n_obs)
    eta: float                     # nominal sharpness of each reported outcome
    obs_names: list
    obs_vectors: list              # target Bloch vectors t_k (observable = t_k . sigma)
    params: dict = field(default_factory=dict)

    @property
    def n_axes(self) -> int:
        return len(self.axes)

    @property
    def n_obs(self) -> int:
        return len(self.obs_names)

    def mask(self) -> np.ndarray:
        """(n_axes, n_obs) bool: is observable k reported on axis a."""
        return self.table[:, 0, :] != 0

    def signs(self) -> np.ndarray:
        """(n_axes, n_obs): reported outcome = sign * o."""
        return self.table[:, 0, :].astype(float)


def pair_eta(theta_deg: float) -> float:
    """Threshold sharpness eta(theta) = 1 / (cos(theta/2) + sin(theta/2))."""
    t = np.radians(check_theta_deg(theta_deg))
    return float(1.0 / (np.cos(t / 2) + np.sin(t / 2)))


def pair_x(theta_deg: float) -> float:
    """x = eta (cos(theta/2) - sin(theta/2)) = E[j k] for every input state."""
    t = np.radians(check_theta_deg(theta_deg))
    return float(pair_eta(theta_deg) * (np.cos(t / 2) - np.sin(t / 2)))


def pair_joint(theta_deg: float) -> Scheme:
    """Randomized bisector / anti-bisector scheme realising the threshold joint POVM.

    axis n1 = (a+b)/|a+b| with prob c/(c+s): (j, k) = (o, o)
    axis n2 = (a-b)/|a-b| with prob s/(c+s): (j, k) = (o, -o)
    Axes with probability zero (n2 at theta = 0, n1 at theta = 180) are not built.
    """
    th = check_theta_deg(theta_deg)
    a, b = pair_vectors(th)
    t = np.radians(th)
    c, s = np.cos(t / 2), np.sin(t / 2)
    axes, labels, probs, rows = [], [], [], []
    if c > 1e-12:
        axes.append(unit(a + b)); labels.append("n1"); probs.append(c / (c + s))
        rows.append([[+1, +1], [-1, -1]])
    if s > 1e-12:
        axes.append(unit(a - b)); labels.append("n2"); probs.append(s / (c + s))
        rows.append([[+1, -1], [-1, +1]])
    probs = np.array(probs) / np.sum(probs)
    return Scheme(
        name=f"pair_joint_{th:g}", kind="joint", axes=axes, axis_labels=labels,
        probs=probs, table=np.array(rows, dtype=int), eta=pair_eta(th),
        obs_names=["A", "A'"], obs_vectors=[a, b],
        params={"theta_deg": th, "c": float(c), "s": float(s), "x": pair_x(th)},
    )


def pair_grouped(theta_deg: float) -> Scheme:
    """Half the copies measure A = Z sharply, half measure A' = b . sigma sharply."""
    th = check_theta_deg(theta_deg)
    a, b = pair_vectors(th)
    table = np.array([[[+1, 0], [-1, 0]], [[0, +1], [0, -1]]], dtype=int)
    return Scheme(
        name=f"pair_grouped_{th:g}", kind="grouped", axes=[a, b], axis_labels=["a", "b"],
        probs=None, table=table, eta=1.0, obs_names=["A", "A'"], obs_vectors=[a, b],
        params={"theta_deg": th},
    )


def xyz_joint() -> Scheme:
    """Uniform cube-diagonal scheme: outcome vector b = o sqrt(3) d, sharpness 1/sqrt(3)."""
    rows = []
    for d in CUBE_DIAGONALS:
        sg = np.sign(d).astype(int)
        rows.append([list(sg), list(-sg)])
    return Scheme(
        name="xyz_joint", kind="joint", axes=list(CUBE_DIAGONALS),
        axis_labels=["d+++", "d+--", "d-+-", "d--+"], probs=np.full(4, 0.25),
        table=np.array(rows, dtype=int), eta=1 / np.sqrt(3),
        obs_names=["X", "Y", "Z"], obs_vectors=[EX, EY, EZ], params={"G": 3},
    )


def xyz_grouped() -> Scheme:
    table = np.zeros((3, 2, 3), dtype=int)
    for k in range(3):
        table[k, 0, k], table[k, 1, k] = +1, -1
    return Scheme(
        name="xyz_grouped", kind="grouped", axes=[EX, EY, EZ], axis_labels=["x", "y", "z"],
        probs=None, table=table, eta=1.0, obs_names=["X", "Y", "Z"],
        obs_vectors=[EX, EY, EZ], params={"G": 3},
    )
