"""Figure of the implementation check (command-line wrapper around jointpovm.figures).

    python scripts/make_figures.py --data results/fake_backend_q72.json --name fig_noisy_sim \
        --label "simulated|(noise model of ibm_kingston, April 2026)" --usetex   # v1 figure
    python scripts/make_figures.py --data results/hardware.json --usetex          # v2 (hardware)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
from jointpovm.figures import noisy_sim_figure, setup  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="results JSON (simulator, fake backend or hardware)")
    p.add_argument("--sdp", default="results/sdp_eta.json")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--usetex", action="store_true", help="typeset with LaTeX (paper style)")
    p.add_argument("--cal", choices=["cal_asym", "cal"], default="cal_asym",
                   help="calibration shown: full (asymmetric, default) or symmetric")
    p.add_argument("--label", default=None, help="data-source note printed in both panels ('|' = line break)")
    p.add_argument("--name", default="fig_hardware", help="output basename (.pdf and .png)")
    args = p.parse_args(argv)
    setup(args.usetex)
    fig = noisy_sim_figure(json.load(open(args.data)), json.load(open(args.sdp)),
                           cal_mode=args.cal, label=args.label, usetex=args.usetex)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.name}.pdf")
    fig.savefig(out / f"{args.name}.png", dpi=300)
    print(f"wrote {out / (args.name + '.pdf')} and .png (usetex={args.usetex}, calibration={args.cal})")


if __name__ == "__main__":
    main()
