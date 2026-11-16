"""Exact (noiseless) Aer run, optionally with a symmetric readout-error model.

    python scripts/run_simulator.py                       # exact
    python scripts/run_simulator.py --readout-p 0.05      # symmetric readout flips
    python scripts/run_simulator.py --readout-p 0.0085 --readout-p10 0.098   # asymmetric
"""
from jointpovm.cli import base_parser, plan_from_args
from jointpovm.execute import AerExecutor, readout_noise_model
from jointpovm.experiment import analyse, run_plan, save


def main(argv=None):
    p = base_parser(__doc__, "results/simulator_exact.json")
    p.add_argument("--readout-p", type=float, default=None,
                   help="readout flip probability P(1|0) (default: noiseless)")
    p.add_argument("--readout-p10", type=float, default=None,
                   help="P(0|1) if different from --readout-p (asymmetric readout)")
    args = p.parse_args(argv)
    if args.readout_p is not None and args.out == "results/simulator_exact.json":
        args.out = (f"results/simulator_readout_p{args.readout_p:g}.json" if args.readout_p10 is None
                    else f"results/simulator_readout_p01_{args.readout_p:g}_p10_{args.readout_p10:g}.json")
    plan = plan_from_args(args)
    nm = readout_noise_model(args.readout_p, args.readout_p10) if args.readout_p is not None else None
    ex = AerExecutor(seed=args.sim_seed, noise_model=nm)
    out = run_plan(plan, ex, seed=args.seed)
    res = analyse(plan, out["raw"], n_boot=args.boot)
    meta = dict(ex.meta, readout_p01=args.readout_p, readout_p10=args.readout_p10, sim_seed=args.sim_seed, **out["meta"])
    path = save(args.out, {"run": meta, "args": vars(args), **res}, raw=out["raw"])
    print(f"wrote {path}  ({meta['n_circuits']} circuits, {meta['n_shots']:,} shots)")
    return path


if __name__ == "__main__":
    main()
