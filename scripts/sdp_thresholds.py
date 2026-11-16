"""Joint-measurability thresholds by SDP for the qubit pair and the X, Y, Z block.

    python scripts/sdp_thresholds.py      # -> results/sdp_eta.json
"""
import json
from pathlib import Path

import numpy as np

from jointpovm.axes import PAULIS, dot_sigma, pair_vectors
from jointpovm.experiment import package_versions
from jointpovm.schemes import pair_eta
from jointpovm.sdp import eta_star

THETAS = [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90]


def main():
    res = {"pair": {"theta_deg": [], "eta_sdp": [], "eta_closed_form": []}, "versions": package_versions()}
    for th in THETAS:
        a, b = pair_vectors(th)
        e = eta_star([dot_sigma(a), dot_sigma(b)])
        res["pair"]["theta_deg"].append(th); res["pair"]["eta_sdp"].append(e)
        res["pair"]["eta_closed_form"].append(pair_eta(th))
        print(f"theta={th:3d}  SDP {e:.8f}  closed form {pair_eta(th):.8f}")
    e = eta_star(list(PAULIS))
    res["xyz"] = {"eta_sdp": e, "eta_closed_form": 1 / np.sqrt(3)}
    print(f"X,Y,Z  SDP {e:.8f}  1/sqrt3 {1 / np.sqrt(3):.8f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/sdp_eta.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
