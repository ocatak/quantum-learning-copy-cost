"""Statistical tests on the exact and readout-noisy Aer simulators (tests 5-7, 9).

Variance-law tests are consistency checks: for +-1/eta-valued estimators the per-copy
variance is (1 - m1^2)/eta^2 and follows from the mean. At mu = 0 the variance estimate is
pinned to its maximum 1/eta^2 minus m1_hat^2/eta^2 >= 0, so the bootstrap percentile interval
never contains the prediction (the prediction is the edge of the support). For mu = 0 we use
the exact sampling law instead: N_used * m1_hat^2 ~ chi^2_1 asymptotically.
"""
import numpy as np
import pytest

from jointpovm import schemes as S
from jointpovm.execute import AerExecutor, readout_noise_model
from jointpovm.experiment import analyse, build_plan, run_plan
from jointpovm.states import bloch

FULL_THETAS = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
VAR_THETAS = (0.0, 20.0, 45.0, 90.0)
N = 10_000
CHI2_1_99 = 6.634896601021214


@pytest.fixture(scope="module")
def exact_run():
    plan = build_plan(thetas=FULL_THETAS, n=N)
    out = run_plan(plan, AerExecutor(seed=11), seed=21)
    return plan, out, analyse(plan, out["raw"], n_boot=2000, seed=31)


@pytest.mark.slow
def test_unbiased_exact(exact_run):
    plan, _, res = exact_run
    assert res["calibration"]["p01"] == 0 and res["calibration"]["p10"] == 0
    for cid, r in res["configs"].items():
        est = r["estimates"]["raw"]
        for k, mu in enumerate(r["mu_exact"]):
            m, se = est["mean_est"]["value"][k], est["se_mean_est"]["value"][k]
            if se == 0:                      # deterministic outcome (eigenstate, eta = 1)
                assert np.isclose(m, mu, atol=1e-12), (cid, k)
            else:
                assert abs(m - mu) <= 4 * se, (cid, k, m, mu, se)


@pytest.mark.slow
def test_joint_statistic_E_jk(exact_run):
    """E[jk] = x for every input state. NOTE: jk is fixed by the classical axis choice
    (+1 on n1, -1 on n2), so this checks the sampler, not the quantum measurement."""
    plan, _, res = exact_run
    for cid, r in res["configs"].items():
        if r["kind"] == "joint" and cid.startswith("pair|"):
            x = r["params"]["x"]
            v = r["estimates"]["E_jk"]["value"]
            assert abs(v - x) <= 4 * np.sqrt(max(1 - x ** 2, 1e-12) / r["n_copies"]) + 1e-12, cid


@pytest.mark.slow
def test_variance_law_exact(exact_run):
    """Each check uses a 99% interval; across the family of checks we allow the number of
    misses expected by chance (<= 2 of 38; P(>=3 | Bin(38, 0.01)) ~ 0.6%), and every miss
    must still agree with the prediction within 4 delta-method standard errors."""
    plan, _, res = exact_run
    checks = []                                  # (label, inside_99, within_4se_delta)

    def interval_check(label, est, p, k):
        lo, hi = est["cost_rel_ideal"]["ci99"][0][k], est["cost_rel_ideal"]["ci99"][1][k]
        v, se = est["cost_rel_ideal"]["value"][k], est["cost_rel_ideal_delta_se"]["value"][k]
        checks.append((label, lo <= p <= hi, abs(v - p) <= 4 * se))

    for th in VAR_THETAS:
        for kind in ("joint", "grouped"):
            for st in ("plus_i", "ry35"):
                r = res["configs"][f"pair|{th:g}|{kind}|{st}"]
                est, pred = r["estimates"]["raw"], r["pred_noiseless"]["raw"]
                for k, mu in enumerate(r["mu_exact"]):
                    p = pred["cost_rel_ideal"][k]
                    label = (th, kind, st, r["obs"][k])
                    if abs(mu) < 1e-12:
                        # mu = 0: estimate = prediction - fac * m1_hat^2 / eta^2 (edge of support)
                        n_used = r["estimates"]["n_used"]["value"][k]
                        fac = 1.0 if kind == "joint" else 2.0
                        stat = (p - est["cost_rel_ideal"]["value"][k]) * r["eta_nominal"] ** 2 * n_used / fac
                        assert stat >= -1e-9, label
                        checks.append((label, stat <= CHI2_1_99, stat <= 4 ** 2))
                    else:
                        interval_check(label, est, p, k)
    for kind in ("joint", "grouped"):
        r = res["configs"][f"xyz|{kind}|xyz"]
        for k in range(3):
            interval_check(("xyz", kind, r["obs"][k]), r["estimates"]["raw"],
                           r["pred_noiseless"]["raw"]["cost_rel_ideal"][k], k)
    assert len(checks) == len(VAR_THETAS) * 2 * 2 * 2 + 6
    misses = [c[0] for c in checks if not c[1]]
    print(f"variance-law checks: {len(checks)}, outside 99% interval: {misses}")
    assert len(misses) <= 2, misses
    assert all(c[2] for c in checks), [c[0] for c in checks if not c[2]]


def _analytic_pair_sample(theta, r, n, rng):
    """Multinomial-free sampler: draw (j, k) i.i.d. from P(j,k) = 1/4[(1+jkx) + eta(j mu_a + k mu_b)]."""
    sch = S.pair_joint(theta)
    eta, x = sch.eta, sch.params["x"]
    mu_a, mu_b = (float(np.dot(t, r)) for t in sch.obs_vectors)
    keys = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    p = np.array([0.25 * ((1 + j * k * x) + eta * (j * mu_a + k * mu_b)) for j, k in keys])
    idx = rng.choice(4, size=n, p=p / p.sum())
    jk = np.array(keys)[idx]
    return jk[:, 0] / eta, jk[:, 1] / eta


@pytest.mark.slow
@pytest.mark.parametrize("theta", [20.0, 45.0, 90.0])
def test_analytic_sampler_agrees_with_circuits(exact_run, theta):
    plan, out, res = exact_run
    rng = np.random.default_rng(int(theta) + 5)
    r = bloch("ry35")
    za, zb = _analytic_pair_sample(theta, r, N, rng)
    est = res["configs"][f"pair|{theta:g}|joint|ry35"]["estimates"]["raw"]
    for k, z in enumerate((za, zb)):
        m_c, se_c = est["mean_est"]["value"][k], est["se_mean_est"]["value"][k]
        assert abs(z.mean() - m_c) <= 4 * np.hypot(se_c, z.std() / np.sqrt(N))
        v_c, sev_c = est["var_per_copy"]["value"][k], est["cost_rel_ideal_delta_se"]["value"][k]
        mu = res["configs"][f"pair|{theta:g}|joint|ry35"]["mu_exact"][k]
        sev_c *= (1 - mu ** 2)                      # back to per-copy variance scale
        c4 = np.mean((z - z.mean()) ** 4)
        sev_a = np.sqrt(max(c4 - z.var() ** 2, 0) / N)
        assert abs(z.var() - v_c) <= 4 * np.hypot(sev_c, sev_a)


@pytest.fixture(scope="module", params=[0.02, 0.05])
def noisy_run(request):
    p = request.param
    plan = build_plan(thetas=VAR_THETAS, n=N, tomo=False)
    out = run_plan(plan, AerExecutor(seed=13, noise_model=readout_noise_model(p)), seed=23)
    return p, plan, analyse(plan, out["raw"], n_boot=2000, seed=33)


def _se_from_ci99(entry, k):
    lo, hi = entry["ci99"][0][k], entry["ci99"][1][k]
    return (hi - lo) / (2 * 2.5758)


@pytest.mark.slow
def test_readout_noise(noisy_run):
    p, plan, res = noisy_run
    cal = res["calibration"]
    assert abs(cal["p01"] - p) < 4 * np.sqrt(p * (1 - p) / N)
    assert abs(cal["p10"] - p) < 4 * np.sqrt(p * (1 - p) / N)
    for cid, r in res["configs"].items():
        for k, mu in enumerate(r["mu_exact"]):
            raw = r["estimates"]["raw"]
            m, se = raw["mean_est"]["value"][k], raw["se_mean_est"]["value"][k]
            assert abs(m - (1 - 2 * p) * mu) <= 4 * se, (cid, k, m, mu)      # shrinkage (1-2p)
            cal_e = r["estimates"]["cal"]["mean_est"]
            se_c = _se_from_ci99(cal_e, k)
            assert abs(cal_e["value"][k] - mu) <= 4 * se_c, (cid, k)         # calibrated: unbiased
    for th, e in res["derived"]["ratio_joint_over_grouped_mu0"].items():
        for mode in ("raw", "cal"):
            for k in range(2):
                lo, hi = e[mode]["ci99"][0][k], e[mode]["ci99"][1][k]
                assert lo <= e["prediction"] <= hi, (p, th, mode, k, lo, e["prediction"], hi)


@pytest.mark.slow
def test_calibrated_variance_inflation(noisy_run):
    """Calibrated estimators: per-copy variance inflates to 1/(eta^2 (1-2p)^2) - mu^2."""
    p, plan, res = noisy_run
    for th in VAR_THETAS:
        r = res["configs"][f"pair|{th:g}|joint|plus_i"]
        for k in range(2):
            v = r["estimates"]["cal"]["var_per_copy"]
            pred = 1 / (r["eta_nominal"] ** 2 * (1 - 2 * p) ** 2)
            assert abs(v["value"][k] - pred) <= 4 * _se_from_ci99(v, k) + 1e-12, (th, k)


def test_reproducible():
    plan = build_plan(thetas=(20.0,), n=2000, xyz=True, tomo=False)
    a = run_plan(plan, AerExecutor(seed=5), seed=6)
    b = run_plan(plan, AerExecutor(seed=5), seed=6)
    c = run_plan(plan, AerExecutor(seed=5), seed=7)
    for cid in a["raw"]:
        assert np.array_equal(a["raw"][cid]["o"], b["raw"][cid]["o"])
        assert np.array_equal(a["raw"][cid]["axis"], b["raw"][cid]["axis"])
    assert a["meta"]["pubs"] == b["meta"]["pubs"] != c["meta"]["pubs"]
    ra = analyse(plan, a["raw"], n_boot=200, seed=1)
    rb = analyse(plan, b["raw"], n_boot=200, seed=1)
    import json
    assert json.dumps(ra, default=float) == json.dumps(rb, default=float)   # NaN-safe comparison


@pytest.mark.slow
def test_pubs_statistically_independent():
    """Regression test for the qiskit-aer seeding pitfall: sampling errors of different pubs
    (distinct shot counts) must be independent. The standardised mean z over K pubs should
    have s.d. ~ 1 across replicates (it was ~6 with a single sampler seed)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1); qc.h(0); qc.measure(0, 0)
    shots = np.random.default_rng(0).integers(500, 3000, size=30)
    vals = []
    for rep in range(25):
        outs = AerExecutor(seed=100 + rep).run([qc] * len(shots), shots)
        z = np.array([o.mean() / np.sqrt(1 / len(o)) for o in outs])
        vals.append(z.mean() * np.sqrt(len(z)))
    sd = np.std(vals)
    assert 0.6 < sd < 1.5, sd
