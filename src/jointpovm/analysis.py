"""Estimators, nonparametric bootstrap and delta-method intervals.

Data of one configuration are per-shot records (axis index, read outcome o' = +-1). All
statistics are functions of the count table C[axis, o_index], so a nonparametric bootstrap
(resampling copies with replacement) is exactly a multinomial draw of the count table with
the empirical cell frequencies. Joint schemes are resampled as a single stratum (the axis is
part of each copy's random outcome); grouped schemes are resampled per axis, because their
split is deterministic by design. Calibration shots are resampled in the same draw, so the
calibrated intervals include calibration uncertainty.
"""
from __future__ import annotations

import numpy as np

from .predictions import MODES, copies_factor, outcome_values
from .schemes import Scheme


def count_table(axis: np.ndarray, o: np.ndarray, n_axes: int) -> np.ndarray:
    C = np.zeros((n_axes, 2), dtype=np.int64)
    np.add.at(C, (axis.astype(int), (o < 0).astype(int)), 1)
    return C


def strata(scheme: Scheme) -> list[list[int]]:
    if scheme.kind == "joint":
        return [list(range(scheme.n_axes))]
    return [[a] for a in range(scheme.n_axes)]


def bootstrap_tables(C: np.ndarray, scheme: Scheme, B: int, rng) -> np.ndarray:
    """(B, n_axes, 2) resampled count tables."""
    out = np.zeros((B,) + C.shape, dtype=np.int64)
    for st in strata(scheme):
        cells = C[st].ravel()
        n = int(cells.sum())
        if n == 0:
            continue
        draw = rng.multinomial(n, cells / n, size=B)
        out[:, st, :] = draw.reshape(B, len(st), 2)
    return out


def calibration_from_counts(c0: np.ndarray, c1: np.ndarray):
    """c0 = counts (o=+1, o=-1) after preparing |0>; c1 likewise for |1>. Returns p01, p10."""
    c0 = np.asarray(c0, float); c1 = np.asarray(c1, float)
    p01 = c0[..., 1] / c0.sum(axis=-1)
    p10 = c1[..., 0] / c1.sum(axis=-1)
    return p01, p10


def statistics(C: np.ndarray, scheme: Scheme, mu: np.ndarray, p01, p10) -> dict:
    """All statistics, vectorised over a leading batch axis of C (shape (..., n_axes, 2))."""
    C = np.asarray(C, float)
    sg = scheme.signs(); m = scheme.mask().astype(float)
    n_used = np.einsum("...ao,ak->...k", C, m)
    n_tot = C.sum(axis=(-1, -2))
    fac = copies_factor(scheme)
    ideal = 1 - mu ** 2
    res = {"n_used": n_used}
    with np.errstate(divide="ignore", invalid="ignore"):
        for mode in MODES:
            ov = outcome_values(mode, p01, p10)                     # (..., 2)
            S = [np.einsum("...ao,ak,...o->...k", C, sg ** q if q % 2 else m, ov ** q) / n_used
                 for q in (1, 2, 3, 4)]
            m1, m2, m3, m4 = S
            var = m2 - m1 ** 2
            c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
            var_z = var / scheme.eta ** 2
            se_var_z = np.sqrt(np.maximum(c4 - var ** 2, 0) / n_used) / scheme.eta ** 2
            res[mode] = {
                "m1": m1,
                "mean_est": m1 / scheme.eta,
                "se_mean_est": np.sqrt(var / n_used) / scheme.eta,
                "eta_eff": np.where(np.abs(mu) > 1e-9, m1 / np.where(np.abs(mu) > 1e-9, mu, 1), np.nan),
                "var_per_copy": var_z,
                "cost_rel_ideal": np.where(ideal > 1e-9, var_z * fac / np.where(ideal > 1e-9, ideal, 1), np.nan),
                "cost_rel_ideal_delta_se": np.where(ideal > 1e-9, se_var_z * fac / np.where(ideal > 1e-9, ideal, 1), np.nan),
            }
        if scheme.kind == "joint" and scheme.n_obs == 2:
            jk = sg[:, 0] * sg[:, 1]
            res["E_jk"] = np.einsum("...ao,a->...", C, jk) / n_tot
    return res


def ci(samples, level=0.95):
    a = (1 - level) / 2
    s = np.asarray(samples, float)
    return np.nanquantile(s, [a, 1 - a], axis=0)


def summarise(point: dict, boot: dict, levels=(0.95, 0.99)) -> dict:
    """Attach percentile intervals from bootstrap arrays to the point estimates (JSON-ready)."""
    out = {}
    for key, val in point.items():
        if isinstance(val, dict):
            out[key] = summarise(val, boot[key], levels)
        else:
            entry = {"value": np.asarray(val).tolist()}
            for lv in levels:
                entry[f"ci{int(lv * 100)}"] = ci(boot[key], lv).tolist()
            out[key] = entry
    return out
