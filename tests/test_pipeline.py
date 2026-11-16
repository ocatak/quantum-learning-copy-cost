"""Asymmetric readout, fake-backend pipeline (test 8) and runtime-sampler code path. No network."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jointpovm.execute import AerExecutor, BackendExecutor, readout_noise_model
from jointpovm.experiment import analyse, build_plan, run_plan, save

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_asymmetric_readout():
    """Full calibration is unbiased; symmetric calibration is biased by the predicted amount;
    the tomographic sharpness is offset-immune: raw |v| = eta (1 - p01 - p10)."""
    p01, p10 = 0.01, 0.08
    plan = build_plan(thetas=(20.0, 90.0), n=10_000)
    out = run_plan(plan, AerExecutor(seed=3, noise_model=readout_noise_model(p01, p10)), seed=4)
    res = analyse(plan, out["raw"], n_boot=2000, seed=5)
    for cid, r in res["configs"].items():
        if r["role"] != "main":
            continue
        for k, mu in enumerate(r["mu_exact"]):
            e = r["estimates"]["cal_asym"]["mean_est"]
            se = (e["ci99"][1][k] - e["ci99"][0][k]) / (2 * 2.5758)
            assert abs(e["value"][k] - mu) <= 4 * se, (cid, k)
            raw = r["estimates"]["raw"]
            pred_raw = r["pred_readout_asymmetric"]["raw"]["mean_est"][k]
            assert abs(raw["mean_est"]["value"][k] - pred_raw) <= 4 * raw["se_mean_est"]["value"][k] + 1e-12
    for th, d in res["derived"]["tomography"].items():
        for obs in ("A", "A'"):
            v = d[f"{obs}|raw"]["sharpness"]
            eta = np.linalg.norm(d[f"{obs}|raw"]["target"])
            lo, hi = v["ci99"]
            assert lo <= eta * (1 - p01 - p10) <= hi, (th, obs, v["value"], eta * (1 - p01 - p10))


@pytest.fixture(scope="module")
def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


@pytest.mark.slow
def test_fake_backend_pipeline(tmp_path, fake_backend):
    """Transpile, run, calibrate, estimate, CIs, JSON and figure on a fake Heron backend.
    Deviations from the prediction are printed, not asserted."""
    plan = build_plan(thetas=(0.0, 90.0), n=3000)
    ex = BackendExecutor(fake_backend, qubit=None, seed=7, chunk=500)
    out = run_plan(plan, ex, seed=8)
    res = analyse(plan, out["raw"], n_boot=2000, seed=9)
    path = save(tmp_path / "fake.json", {"run": dict(ex.meta, **out["meta"]), **res}, raw=out["raw"])
    R = json.loads(path.read_text())
    assert R["run"]["shots_discarded"] >= 0 and R["run"]["qubit"] == ex.qubit
    assert set(R["versions"]) >= {"qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy"}
    for cid, r in R["configs"].items():
        assert r["n_copies"] == 3000
        for k in range(len(r["obs"])):
            assert np.isfinite(r["estimates"]["raw"]["mean_est"]["value"][k])
        if cid.endswith("|ry35") or cid.endswith("|xyz"):
            for k, o in enumerate(r["obs"]):
                e = r["estimates"]["cal_asym"]["eta_eff"]
                print(f"{cid:22s} {o:3s} eta={r['eta_nominal']:.4f} eta_eff(cal)={e['value'][k]:.4f} "
                      f"95% [{e['ci95'][0][k]:.4f}, {e['ci95'][1][k]:.4f}]")
    figdir = tmp_path / "fig"
    cmd = [sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--data", str(path),
           "--sdp", str(ROOT / "results" / "sdp_eta.json"), "--outdir", str(figdir),
           "--label", "fake backend (test)"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    assert (figdir / "fig_hardware.pdf").stat().st_size > 1000
    assert (figdir / "fig_hardware.png").stat().st_size > 1000


@pytest.mark.slow
def test_fake_backend_pubs_independent(fake_backend):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1); qc.h(0); qc.measure(0, 0)
    vals = []
    for rep in range(10):
        ex = BackendExecutor(fake_backend, qubit=109, seed=600 + rep, chunk=500)
        t = ex.prepare([qc])[0]
        outs = ex.run([t] * 30, [500] * 30)
        z = np.array([o.mean() * np.sqrt(len(o)) for o in outs])
        vals.append(z.mean() * np.sqrt(len(z)))
    vals = np.array(vals) - np.mean(vals)            # remove the common readout offset
    assert 0.4 < np.std(vals) < 1.8, np.std(vals)


def test_runtime_sampler_code_path(fake_backend):
    """The hardware path (qiskit_ibm_runtime.SamplerV2, equal-shot chunks, truncation),
    exercised locally on the fake backend."""
    from jointpovm.circuits import measure_circuit
    ex = BackendExecutor(fake_backend, seed=1, chunk=200, runtime_sampler=True)
    circs = ex.prepare([measure_circuit("zero", [0, 0, 1]), measure_circuit("one", [0, 0, 1])])
    outs = ex.run(circs, [250, 130])
    assert [len(o) for o in outs] == [250, 130]
    assert ex.meta["n_pubs_submitted"] == 3 and ex.meta["shots_discarded"] == 600 - 380
    assert outs[0].mean() > 0.9 and outs[1].mean() < -0.8
