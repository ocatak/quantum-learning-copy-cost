"""Print compact tables from one or more results JSON files.

    python scripts/report.py results/simulator_exact.json results/fake_backend.json
"""
import json
import sys

import numpy as np


def fmt(e, k=None, nd=3):
    v = e["value"] if k is None else e["value"][k]
    lo, hi = (e["ci95"][0], e["ci95"][1]) if k is None else (e["ci95"][0][k], e["ci95"][1][k])
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"


def report(path):
    R = json.load(open(path))
    run = R["run"]
    print(f"\n=== {path}: {run.get('executor')} {run.get('backend', '')} qubit={run.get('qubit', '-')}"
          f"  circuits={run['n_circuits']} shots={run['n_shots']:,}  date={R['date_utc'][:19]}")
    c = R["calibration"]
    print(f"readout: p01={c['p01']:.4f} {np.round(c['p01_ci95'], 4).tolist()}  "
          f"p10={c['p10']:.4f} {np.round(c['p10_ci95'], 4).tolist()}  p_sym={c['p_sym']:.4f}  "
          f"p10-p01={c['asymmetry_p10_minus_p01']:+.4f}")
    if "reported_prob_meas1_prep0" in run:
        print(f"backend-reported: P(1|0)={run['reported_prob_meas1_prep0']:.4f} "
              f"P(0|1)={run['reported_prob_meas0_prep1']:.4f}")
    print("\n[1] effective sharpness eta_eff = mean outcome / exact mu  (state ry35 for the pair, xyz for X,Y,Z)")
    print(f"{'config':24s} {'obs':4s} {'eta':>6s} {'eta(1-2p)':>9s}  {'raw':>24s}  {'cal (sym)':>24s}  {'cal (asym)':>24s}")
    for cid, r in R["configs"].items():
        if not (cid.endswith("|ry35") or cid.endswith("|xyz")):
            continue
        for k, o in enumerate(r["obs"]):
            ps = r["pred_readout_symmetric"]["raw"]["eta_eff"][k]
            e = r["estimates"]
            print(f"{cid:24s} {o:4s} {r['eta_nominal']:6.4f} {ps:9.4f}  {fmt(e['raw']['eta_eff'], k, 4):>24s}  "
                  f"{fmt(e['cal']['eta_eff'], k, 4):>24s}  {fmt(e['cal_asym']['eta_eff'], k, 4):>24s}")
    print("\n[2] marginal detector tomography of the joint pair scheme (6 cardinal states)")
    for th, d in R["derived"]["tomography"].items():
        for key, v in d.items():
            if key.endswith("|raw") or key.endswith("|cal"):
                eta = np.linalg.norm(v["target"])
                print(f"theta={th:>3s} {key:9s} |v|={fmt(v['sharpness'], nd=4)} (eta={eta:.4f})  "
                      f"angle={fmt(v['angle_to_target_deg'], nd=2)} deg  offset={fmt(v['offset_alpha_minus_1'], nd=4)}")
    print("\n[3] E[jk] (fixed by the classical axis choice; checks the sampler only)")
    for cid, r in R["configs"].items():
        if "E_jk" in r["estimates"] and cid.endswith("|ry35"):
            print(f"{cid:24s} x={r['params']['x']:.4f}  E[jk]={fmt(r['estimates']['E_jk'], nd=4)}")
    print("\n[4] per-copy variance relative to ideal (consistency check; follows from the mean)")
    print(f"{'config':26s} {'obs':4s} {'pred':>7s}  {'raw':>24s}  {'cal (sym)':>24s}")
    for cid, r in R["configs"].items():
        if r["role"] != "main":
            continue
        for k, o in enumerate(r["obs"]):
            e = r["estimates"]
            print(f"{cid:26s} {o:4s} {r['pred_noiseless']['raw']['cost_rel_ideal'][k]:7.3f}  "
                  f"{fmt(e['raw']['cost_rel_ideal'], k):>24s}  {fmt(e['cal']['cost_rel_ideal'], k):>24s}")
    print("\n[5] joint/grouped variance ratio at mu = 0 (identity for zero-mean +-1 data)")
    for th, e in R["derived"]["ratio_joint_over_grouped_mu0"].items():
        print(f"theta={th:>3s} pred={e['prediction']:.4f}  raw A={fmt(e['raw'], 0, 4)}  cal A={fmt(e['cal'], 0, 4)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
