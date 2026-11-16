"""Full pipeline on a qiskit_ibm_runtime fake backend (local Aer with the device noise model).

    python scripts/run_fake_backend.py --backend FakeKingston
"""
from jointpovm.cli import base_parser, job_summary, plan_from_args, print_summary
from jointpovm.execute import BackendExecutor
from jointpovm.experiment import analyse, run_plan, save


def get_fake(name: str):
    from qiskit_ibm_runtime import fake_provider
    if not name.startswith("Fake"):
        name = "Fake" + name.removeprefix("ibm_").capitalize()
    return getattr(fake_provider, name)()


def main(argv=None):
    p = base_parser(__doc__, "results/fake_backend.json")
    p.add_argument("--backend", default="FakeKingston")
    p.add_argument("--qubit", type=int, default=None, help="physical qubit (default: best readout)")
    p.add_argument("--runtime-sampler", action="store_true",
                   help="use qiskit_ibm_runtime.SamplerV2 local mode (hardware code path, ~1 s per pub)")
    args = p.parse_args(argv)
    plan = plan_from_args(args)
    backend = get_fake(args.backend)
    ex = BackendExecutor(backend, qubit=args.qubit, seed=args.sim_seed, chunk=args.chunk,
                         runtime_sampler=args.runtime_sampler)
    summ = job_summary(plan, ex, args.seed)
    print_summary(summ)
    out = run_plan(plan, ex, seed=args.seed)
    res = analyse(plan, out["raw"], n_boot=args.boot)
    meta = dict(ex.meta, sim_seed=args.sim_seed, dry_run_summary=summ, **out["meta"])
    path = save(args.out, {"run": meta, "args": vars(args), **res}, raw=out["raw"])
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
