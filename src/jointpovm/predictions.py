"""Exact predictions for every estimator, with or without (asymmetric) readout error.

Readout model: P(read 1 | o=+1) = p01, P(read 0 | o=-1) = p10, i.e. the read outcome o'
satisfies E[o'] = (1 - p01 - p10) E[o] + (p10 - p01). Symmetric case p01 = p10 = p:
sharpness of every reported outcome is multiplied by (1 - 2p).

Estimator modes (applied to the read outcome o' before the scheme's sign table):
  raw       o'
  cal       o' / (1 - p01 - p10)                  symmetric calibration, p = (p01+p10)/2
  cal_asym  (o' - (p10 - p01)) / (1 - p01 - p10)  full (asymmetric) calibration
The estimator of <t_k . sigma> is (reported value) / eta.

NOTE on variances. For a raw +-1/eta-valued estimator the per-copy variance is
(1 - m1^2)/eta^2, where m1 is the mean reported outcome. The variance therefore carries no
information beyond the mean; at mu = 0 (m1 = 0 in the noiseless case) it equals 1/eta^2 by
construction, whatever the device does. Variance points are consistency checks only; the
hardware content of the experiment lies in the means (effective sharpness, marginals).
"""
from __future__ import annotations

import numpy as np

from .schemes import Scheme

MODES = ("raw", "cal", "cal_asym")


def outcome_values(mode: str, p01, p10):
    """Values assigned to read outcomes o' = +1, -1. Broadcasts over leading dims of p01/p10."""
    p01 = np.asarray(p01, float); p10 = np.asarray(p10, float)
    o = np.array([1.0, -1.0])
    if mode == "raw":
        return np.broadcast_to(o, p01.shape + (2,)).copy()
    gain = (1.0 - p01 - p10)[..., None]
    if mode == "cal":
        return o / gain
    if mode == "cal_asym":
        return (o - (p10 - p01)[..., None]) / gain
    raise ValueError(mode)


def axis_weights(scheme: Scheme) -> np.ndarray:
    """(n_axes, n_obs): fraction of the copies used for observable k that go to axis a."""
    m = scheme.mask().astype(float)
    if scheme.kind == "joint":
        return m * scheme.probs[:, None]
    return m / m.sum(axis=0, keepdims=True)


def copies_factor(scheme: Scheme) -> np.ndarray:
    """N_total / N_used per observable (1 for joint, n_axes/#axes-used for grouped)."""
    if scheme.kind == "joint":
        return np.ones(scheme.n_obs)
    m = scheme.mask().astype(float)
    return scheme.n_axes / m.sum(axis=0)


def mu_exact(scheme: Scheme, r) -> np.ndarray:
    return np.array([float(np.dot(t, r)) for t in scheme.obs_vectors])


def predict(scheme: Scheme, r, p01: float = 0.0, p10: float = 0.0) -> dict:
    """Exact expectation of every statistic for Bloch vector r and readout errors (p01, p10)."""
    r = np.asarray(r, float)
    mu = mu_exact(scheme, r)
    w = axis_weights(scheme)
    sg = scheme.signs()
    # read-outcome probabilities per axis
    p_plus = np.array([(1 + np.dot(n, r)) / 2 for n in scheme.axes])
    pr_plus = p_plus * (1 - p01) + (1 - p_plus) * p10
    P = np.stack([pr_plus, 1 - pr_plus], axis=1)                      # (n_axes, 2)
    out = {"mu": mu, "eta": scheme.eta}
    with np.errstate(divide="ignore", invalid="ignore"):
        for mode in MODES:
            ov = outcome_values(mode, p01, p10)                       # (2,)
            m1 = np.einsum("ak,ao,o->k", w * sg, P, ov)
            m2 = np.einsum("ak,ao,o->k", w, P, ov ** 2)
            var_z = (m2 - m1 ** 2) / scheme.eta ** 2
            ideal = 1 - mu ** 2
            out[mode] = {
                "m1": m1, "mean_est": m1 / scheme.eta,
                "eta_eff": np.where(np.abs(mu) > 1e-9, m1 / mu, np.nan),
                "var_per_copy": var_z,
                "cost_rel_ideal": np.where(ideal > 1e-9, var_z * copies_factor(scheme) / ideal, np.nan),
            }
    if scheme.kind == "joint" and scheme.n_obs == 2:
        # E[j k] = sum_axis p_axis * (sign_j sign_k) : fixed by the classical axis choice.
        out["E_jk"] = float(np.sum(scheme.probs * sg[:, 0] * sg[:, 1]))
    return out
