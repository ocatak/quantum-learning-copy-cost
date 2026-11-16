"""IBM hardware run. DEFAULT IS A DRY RUN: transpile and print the job summary only.

    python scripts/run_hardware.py --backend ibm_kingston                 # dry run
    python scripts/run_hardware.py --backend ibm_kingston --submit \
        --confirm ibm_kingston                                           # spends QPU time

Credentials come from a saved QiskitRuntimeService account or from the environment variables
QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE (and optionally QISKIT_IBM_CHANNEL). Tokens are never
printed or written to results.
"""
import os
import sys

from jointpovm.cli import base_parser, job_summary, plan_from_args, print_summary
from jointpovm.execute import BackendExecutor
from jointpovm.experiment import analyse, run_plan, save


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if tok:
        return QiskitRuntimeService(channel=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
                                    token=tok, instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    return QiskitRuntimeService()


def main(argv=None):
    p = base_parser(__doc__, "results/hardware.json")
    p.add_argument("--backend", default=None, help="device name (default: least busy operational)")
    p.add_argument("--qubit", type=int, default=None, help="physical qubit (default: best readout)")
    p.add_argument("--submit", action="store_true", help="actually submit (spends QPU time)")
    p.add_argument("--confirm", default=None, help="must equal the backend name when --submit is given")
    args = p.parse_args(argv)
    plan = plan_from_args(args)
    try:
        svc = get_service()
        backend = svc.backend(args.backend) if args.backend else svc.least_busy(simulator=False, operational=True)
        real = True
    except Exception as exc:  # no account: dry run on the matching fake backend only
        print(f"[no IBM account available: {type(exc).__name__}] dry run uses the fake backend instead")
        if args.submit:
            sys.exit("cannot submit without an IBM account")
        from run_fake_backend import get_fake
        backend = get_fake(args.backend or "FakeKingston")
        real = False
    ex = BackendExecutor(backend, qubit=args.qubit, seed=args.seed, is_hardware=real, chunk=args.chunk)
    summ = job_summary(plan, ex, args.seed)
    print_summary(summ)
    if not args.submit:
        print("DRY RUN: nothing submitted. Re-run with --submit --confirm <backend> to execute.")
        return summ
    if args.confirm != backend.name:
        sys.exit(f"refusing to submit: --confirm must equal '{backend.name}'")
    out = run_plan(plan, ex, seed=args.seed)
    res = analyse(plan, out["raw"], n_boot=args.boot)
    meta = dict(ex.meta, job_id=getattr(ex, "last_job_id", None), dry_run_summary=summ, **out["meta"])
    path = save(args.out, {"run": meta, "args": vars(args), **res}, raw=out["raw"])
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
