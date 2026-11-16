"""Experiment plan, execution and analysis shared by all run scripts."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import platform
from pathlib import Path

import numpy as np

from . import schemes as S
from .analysis import (bootstrap_tables, calibration_from_counts, count_table, statistics,
                       summarise)
from .axes import EZ
from .predictions import predict
from .sampling import Config, allocate, build_circuit
from .states import CARDINAL, bloch

DEFAULT_THETAS = (0.0, 20.0, 45.0, 90.0)
PAIR_GROUPED_STATES = ("plus_i", "ry35")
PAIR_JOINT_STATES = ("ry35",) + tuple(CARDINAL)        # cardinal states = marginal tomography
XYZ_STATES = ("mixed", "xyz")
PACKAGES = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy", "matplotlib",
            "cvxpy", "pytest")


def cal_scheme() -> S.Scheme:
    return S.Scheme(name="calZ", kind="grouped", axes=[EZ], axis_labels=["z"], probs=None,
                    table=np.array([[[1], [-1]]]), eta=1.0, obs_names=["Z"], obs_vectors=[EZ])


def build_plan(thetas=DEFAULT_THETAS, n=10_000, n_cal=None, xyz=True, tomo=True) -> list[Config]:
    n_cal = n if n_cal is None else n_cal
    plan = [Config("cal|zero", cal_scheme(), "zero", n_cal, "cal"),
            Config("cal|one", cal_scheme(), "one", n_cal, "cal")]
    for th in thetas:
        th = float(th)
        for st in PAIR_JOINT_STATES:
            if st != "ry35" and not tomo and st != "plus_i":
                continue
            plan.append(Config(f"pair|{th:g}|joint|{st}", S.pair_joint(th), st, n,
                               "main" if st in ("ry35", "plus_i") else "tomo"))
        for st in PAIR_GROUPED_STATES:
            plan.append(Config(f"pair|{th:g}|grouped|{st}", S.pair_grouped(th), st, n, "main"))
    if xyz:
        for st in XYZ_STATES:
            plan.append(Config(f"xyz|joint|{st}", S.xyz_joint(), st, n, "main"))
            plan.append(Config(f"xyz|grouped|{st}", S.xyz_grouped(), st, n, "main"))
    return plan


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            out[p] = None
    return out


def make_pubs(plan, seed):
    """Seeded multinomial allocation -> (specs, logical circuits)."""
    rng = np.random.default_rng(seed)
    specs, circs = [], []
    by_id = {c.cid: c for c in plan}
    for cfg in plan:
        for sp in allocate(cfg, rng):
            specs.append(sp)
            circs.append(build_circuit(sp, by_id[sp.config].scheme))
    return specs, circs


def run_plan(plan, executor, seed: int = 1234) -> dict:
    specs, circs = make_pubs(plan, seed)
    tcircs = executor.prepare(circs)
    outs = executor.run(tcircs, [s.shots for s in specs])
    raw = {}
    for sp, o in zip(specs, outs):
        if len(o) != sp.shots:
            raise RuntimeError(f"pub {sp} returned {len(o)} shots")
        d = raw.setdefault(sp.config, {"axis": [], "o": [], "prep": []})
        d["axis"].append(np.full(sp.shots, sp.axis, np.int8))
        d["o"].append(o)
        d["prep"].append(np.full(sp.shots, sp.prep))
    for d in raw.values():
        for k in d:
            d[k] = np.concatenate(d[k])
    depths = [qc.depth() for qc in tcircs]
    ops = {}
    for qc in tcircs:
        for k, v in qc.count_ops().items():
            ops[k] = ops.get(k, 0) + v
    meta = {"n_circuits": len(specs), "n_shots": int(sum(s.shots for s in specs)),
            "transpiled_depth_min": int(min(depths)), "transpiled_depth_max": int(max(depths)),
            "transpiled_op_counts_total": ops, "allocation_seed": seed,
            "pubs": [[s.config, s.axis, s.prep, s.shots] for s in specs]}
    return {"raw": raw, "meta": meta}


def analyse(plan, raw: dict, n_boot: int = 2000, seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)
    cal0 = count_table(raw["cal|zero"]["axis"], raw["cal|zero"]["o"], 1)[0]
    cal1 = count_table(raw["cal|one"]["axis"], raw["cal|one"]["o"], 1)[0]
    p01, p10 = calibration_from_counts(cal0, cal1)
    b0 = rng.multinomial(cal0.sum(), cal0 / cal0.sum(), size=n_boot)
    b1 = rng.multinomial(cal1.sum(), cal1 / cal1.sum(), size=n_boot)
    p01_b, p10_b = calibration_from_counts(b0, b1)
    calib = {"p01": float(p01), "p10": float(p10), "p_sym": float((p01 + p10) / 2),
             "asymmetry_p10_minus_p01": float(p10 - p01),
             "p01_ci95": np.quantile(p01_b, [0.025, 0.975]).tolist(),
             "p10_ci95": np.quantile(p10_b, [0.025, 0.975]).tolist(),
             "counts_prep0": cal0.tolist(), "counts_prep1": cal1.tolist()}
    results, boots = {}, {}
    for cfg in plan:
        if cfg.role == "cal":
            continue
        sch = cfg.scheme
        r = bloch(cfg.state)
        C = count_table(raw[cfg.cid]["axis"], raw[cfg.cid]["o"], sch.n_axes)
        mu = np.array([float(np.dot(t, r)) for t in sch.obs_vectors])
        point = statistics(C, sch, mu, p01, p10)
        Cb = bootstrap_tables(C, sch, n_boot, rng)
        bst = statistics(Cb, sch, mu, p01_b, p10_b)
        boots[cfg.cid] = bst
        results[cfg.cid] = {
            "scheme": sch.name, "kind": sch.kind, "state": cfg.state, "role": cfg.role,
            "bloch": r.tolist(), "obs": sch.obs_names, "mu_exact": mu.tolist(),
            "eta_nominal": sch.eta, "params": sch.params, "n_copies": int(C.sum()),
            "counts": C.tolist(), "axis_labels": sch.axis_labels,
            "estimates": summarise(point, bst),
            "pred_noiseless": _jsonify(predict(sch, r)),
            "pred_readout_symmetric": _jsonify(predict(sch, r, calib["p_sym"], calib["p_sym"])),
            "pred_readout_asymmetric": _jsonify(predict(sch, r, p01, p10)),
        }
    derived = {"tomography": _tomography(plan, raw, boots, p01, p10),
               "ratio_joint_over_grouped_mu0": _ratios(plan, results, boots)}
    return {"calibration": calib, "configs": results, "derived": derived,
            "n_boot": n_boot, "boot_seed": seed}


def _tomography(plan, raw, boots, p01, p10):
    """Fit each marginal effect (alpha I + v.sigma)/2 of the joint pair scheme from the six
    cardinal states: v_i = (E[j|+e_i] - E[j|-e_i])/2, alpha - 1 = mean of the six E[j]."""
    out = {}
    thetas = sorted({c.scheme.params["theta_deg"] for c in plan if c.cid.startswith("pair|")})
    for th in thetas:
        ids = {st: f"pair|{th:g}|joint|{st}" for st in CARDINAL}
        if not all(i in boots for i in ids.values()):
            continue
        sch = S.pair_joint(th)
        entry = {}
        for k, name in enumerate(sch.obs_names):
            t = sch.obs_vectors[k]
            for mode in ("raw", "cal", "cal_asym"):
                def fit(get):
                    E = {st: get(ids[st]) for st in CARDINAL}
                    v = np.stack([(E["plus"] - E["minus"]) / 2, (E["plus_i"] - E["minus_i"]) / 2,
                                  (E["zero"] - E["one"]) / 2], axis=-1)
                    off = sum(E.values()) / 6
                    nrm = np.linalg.norm(v, axis=-1)
                    ang = np.degrees(np.arccos(np.clip(v @ t / nrm, -1, 1)))
                    return nrm, ang, off, v
                bn, ba, bo, _ = fit(lambda cid: boots[cid][mode]["m1"][..., k])
                pn, pa, po, pv = fit(lambda cid: _point_m1(plan, raw, cid, k, mode, p01, p10))
                entry[f"{name}|{mode}"] = {
                    "sharpness": _ci_entry(pn, bn), "angle_to_target_deg": _ci_entry(pa, ba),
                    "offset_alpha_minus_1": _ci_entry(po, bo), "v": np.asarray(pv).tolist(),
                    "target": (sch.eta * t).tolist()}
        out[f"{th:g}"] = entry
    return out


def _point_m1(plan, raw, cid, k, mode, p01, p10):
    cfg = next(c for c in plan if c.cid == cid)
    C = count_table(raw[cid]["axis"], raw[cid]["o"], cfg.scheme.n_axes)
    mu = np.zeros(cfg.scheme.n_obs)
    return statistics(C, cfg.scheme, mu, p01, p10)[mode]["m1"][k]


def _ratios(plan, results, boots):
    """Joint/grouped per-copy variance ratio at mu = 0 (state plus_i); consistency check only."""
    out = {}
    for c in plan:
        if c.cid.startswith("pair|") and c.cid.endswith("|joint|plus_i"):
            g = c.cid.replace("|joint|", "|grouped|")
            if g not in boots:
                continue
            th = c.scheme.params["theta_deg"]
            e = {}
            for mode in ("raw", "cal"):
                num_b = boots[c.cid][mode]["cost_rel_ideal"]; den_b = boots[g][mode]["cost_rel_ideal"]
                num = np.array(results[c.cid]["estimates"][mode]["cost_rel_ideal"]["value"])
                den = np.array(results[g]["estimates"][mode]["cost_rel_ideal"]["value"])
                e[mode] = _ci_entry(num / den, num_b / den_b)
            e["prediction"] = (1 + np.sin(np.radians(th))) / 2
            out[f"{th:g}"] = e
    return out


def _ci_entry(point, boot):
    point = np.asarray(point, float); boot = np.asarray(boot, float)
    return {"value": point.tolist(),
            "ci95": np.nanquantile(boot, [0.025, 0.975], axis=0).tolist(),
            "ci99": np.nanquantile(boot, [0.005, 0.995], axis=0).tolist()}


def _jsonify(x):
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def save(path, payload: dict, raw: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("date_utc", _dt.datetime.now(_dt.timezone.utc).isoformat())
    payload.setdefault("versions", package_versions())
    path.write_text(json.dumps(_jsonify(payload), indent=1))
    if raw is not None:
        arrs = {}
        for cid, d in raw.items():
            for k, v in d.items():
                arrs[f"{cid}::{k}"] = v
        np.savez_compressed(path.with_name("raw_" + path.stem + ".npz"), **arrs)
    return path
