"""Shared command-line helpers for the run scripts."""
from __future__ import annotations

import argparse

import numpy as np

from .experiment import DEFAULT_THETAS, build_plan, make_pubs


def base_parser(desc: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--thetas", type=float, nargs="+", default=list(DEFAULT_THETAS),
                   help="pair angles in degrees (default: %(default)s)")
    p.add_argument("--n", type=int, default=10_000,
                   help="copies per (scheme, theta, state) configuration (default: %(default)s)")
    p.add_argument("--n-cal", type=int, default=None, help="shots per calibration state (default: = --n)")
    p.add_argument("--no-xyz", action="store_true", help="skip the X, Y, Z flagship")
    p.add_argument("--no-tomo", action="store_true", help="skip the 6-state marginal tomography")
    p.add_argument("--seed", type=int, default=1234, help="allocation seed")
    p.add_argument("--sim-seed", type=int, default=4321, help="simulator seed")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (>= 2000)")
    p.add_argument("--chunk", type=int, default=500,
                   help="shots per pub on IBM runtime backends (equal-shots constraint)")
    p.add_argument("--out", default=default_out)
    return p


def plan_from_args(args):
    if args.boot < 2000:
        raise SystemExit("--boot must be >= 2000")
    return build_plan(thetas=args.thetas, n=args.n, n_cal=args.n_cal,
                      xyz=not args.no_xyz, tomo=not args.no_tomo)


def circuit_duration_s(backend, tqc) -> float:
    """Sum of calibrated instruction durations on the circuit's qubits (rz is virtual)."""
    target = backend.target
    tot = 0.0
    for inst in tqc.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qargs = tuple(tqc.find_bit(q).index for q in inst.qubits)
        try:
            d = target[name][qargs].duration
        except Exception:
            d = None
        tot += d or 0.0
    return tot


def rep_delay_s(backend) -> float:
    try:
        conf = backend._get_conf_dict_from_json()
        return float(conf["default_rep_delay"]) * 1e-6
    except Exception:
        pass
    try:
        return float(backend.configuration().default_rep_delay)
    except Exception:
        return 250e-6


def job_summary(plan, executor, seed, per_job_overhead_s=(2.0, 10.0)) -> dict:
    """Transpile the full plan (no execution) and estimate QPU time."""
    specs, circs = make_pubs(plan, seed)
    tcircs = executor.prepare(circs)
    shots = np.array([s.shots for s in specs])
    durs = np.array([circuit_duration_s(executor.backend, t) for t in tcircs])
    rd = rep_delay_s(executor.backend)
    per_circ = np.array([-(-int(n) // executor.chunk) * executor.chunk for n in shots])
    loop = float(np.sum(per_circ * (durs + rd)))
    depths = [t.depth() for t in tcircs]
    return {"backend": executor.backend.name, "qubit": executor.qubit,
            "n_configs": len(plan), "n_circuits": len(specs), "n_shots": int(shots.sum()),
            "shot_chunk": executor.chunk, "n_pubs": int(sum(per_circ) // executor.chunk),
            "n_shots_executed": int(per_circ.sum()),
            "rep_delay_us": rd * 1e6, "circuit_duration_us_max": float(durs.max() * 1e6),
            "depth_min": int(min(depths)), "depth_max": int(max(depths)),
            "shot_loop_seconds": loop,
            "estimate_seconds_range": [loop + per_job_overhead_s[0], loop + per_job_overhead_s[1]],
            "note": ("shot-loop time = sum_shots (circuit duration + default rep_delay), from the "
                     "backend target; the per-job overhead range is an assumption. IBM reports "
                     "the billed quantum seconds only after execution.")}


def print_summary(s: dict):
    print(f"backend {s['backend']}  physical qubit {s['qubit']}")
    print(f"configurations {s['n_configs']}  circuits {s['n_circuits']}  shots used {s['n_shots']:,}")
    print(f"pubs {s['n_pubs']} x {s['shot_chunk']} shots = {s['n_shots_executed']:,} shots executed "
          f"({s['n_shots_executed'] - s['n_shots']:,} discarded by truncation)")
    print(f"transpiled depth {s['depth_min']}-{s['depth_max']}  max circuit duration "
          f"{s['circuit_duration_us_max']:.2f} us  rep_delay {s['rep_delay_us']:.0f} us")
    lo, hi = s["estimate_seconds_range"]
    print(f"shot-loop QPU time {s['shot_loop_seconds']:.1f} s; estimate with job overhead "
          f"{lo:.0f}-{hi:.0f} s ({lo / 60:.1f}-{hi / 60:.1f} min)")
    print("note:", s["note"])
