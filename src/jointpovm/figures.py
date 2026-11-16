"""Figure of the implementation check of the joint POVMs (fig_noisy_sim.pdf in v1).

(a) Effective sharpness of the two marginals of the qubit-pair joint POVM, from 6-state
    marginal tomography (|v| of the fitted effect (alpha I + v.sigma)/2), raw and
    readout-calibrated, against the threshold eta(theta) and the SDP values.
(b) X, Y, Z flagship: effective sharpness of each Pauli marginal of the cube-diagonal POVM
    (state Rz(30)Ry(55)|0>), raw and calibrated, against 1/sqrt(3); sharp grouped
    measurements (eta = 1) for comparison.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Same style as copycost.style (paper figures)
PAPER_RC = {
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 3.6,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02}
# Fallback without a LaTeX installation: matplotlib's bundled Computer Modern (cmr10 + mathtext cm)
FALLBACK_RC = {'text.usetex': False, 'font.family': 'serif', 'font.serif': ['cmr10'],
               'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False}
W = 6.4


def setup(usetex: bool):
    plt.rcParams.update(PAPER_RC)
    if not usetex:
        plt.rcParams.update(FALLBACK_RC)


def panel(ax, lab, usetex):
    if usetex:
        ax.text(-0.16, 1.04, r'\textbf{(' + lab + ')}', transform=ax.transAxes, fontsize=9.5, va='bottom')
    else:
        ax.text(-0.16, 1.04, '(' + lab + ')', transform=ax.transAxes, fontsize=9.5, va='bottom',
                family='cmb10')


def err(entry, k=None):
    v = entry["value"] if k is None else entry["value"][k]
    lo, hi = (entry["ci95"] if k is None else (entry["ci95"][0][k], entry["ci95"][1][k]))
    return v, [[v - lo], [hi - v]]


def noisy_sim_figure(R: dict, S: dict, cal_mode: str = "cal_asym", label: str | None = None,
                     usetex: bool = False):
    """Draw the two-panel figure from a results dict R (run_plan + analyse) and SDP dict S
    (scripts/sdp_thresholds.py). Call setup(usetex) first. Returns the figure."""
    if label:  # cmr10 has no '_' glyph; LaTeX needs it escaped. '|' starts a new line.
        label = label.replace("_", r"\_" if usetex else r"$\_$").replace("|", "\n")
    cal = R["calibration"]; gain = 1 - cal["p01"] - cal["p10"]
    tomo = R["derived"]["tomography"]

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.35)); fig.subplots_adjust(wspace=0.32)
    # ---------------- (a) qubit pair
    ax = axs[0]
    tt = np.linspace(0, 90, 400); t = np.radians(tt); eta = 1 / (np.cos(t / 2) + np.sin(t / 2))
    ax.plot(tt, eta, '-', color='0', label=r'threshold $1/(\cos\frac{\theta}{2}+\sin\frac{\theta}{2})$')
    ax.plot(tt, gain * eta, ':', color='0.45', lw=0.9, label=r'$\times(1-p_{01}-p_{10})$')
    ax.plot(S["pair"]["theta_deg"], S["pair"]["eta_sdp"], 'o', mfc='white', mec='0', mew=0.8, ms=4.2,
            label='SDP', zorder=2)
    mk = {"A": "^", "A'": "v"}; dx = {"A": -1.3, "A'": 1.3}
    for obs in ("A", "A'"):
        th = np.array(sorted(float(k) for k in tomo))
        for mode, col in (("raw", "0.6"), (cal_mode, "0")):
            vals = [err(tomo[f"{x:g}"][f"{obs}|{mode}"]["sharpness"]) for x in th]
            y = np.array([v for v, _ in vals]); e = np.hstack([q for _, q in vals])
            ax.errorbar(th + dx[obs], y, yerr=e, fmt=mk[obs], color=col, mfc=col, mec=col, ms=3.8,
                        elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
    h = [Line2D([], [], ls='none', marker='^', color='0', ms=3.8), Line2D([], [], ls='none', marker='v', color='0', ms=3.8),
         Line2D([], [], ls='none', marker='s', color='0.6', ms=3.6), Line2D([], [], ls='none', marker='s', color='0', ms=3.6)]
    leg1 = ax.legend(loc='upper right', borderaxespad=0.2, handlelength=1.8)
    ax.add_artist(leg1)
    ax.legend(h, [r'$A$', r"$A'$", 'raw', 'calibrated'], loc='lower left', ncol=2, borderaxespad=0.2,
              handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.0, 0.13 if label else 0.08))
    ax.set_xlim(-4, 94); ax.set_ylim(0.55, 1.04)
    ax.xaxis.set_major_locator(MultipleLocator(15)); ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r'angle $\theta$ between the two observables (deg)')
    ax.set_ylabel(r'effective sharpness $\eta_{\mathrm{eff}}$')
    if label:
        ax.text(0.03, 0.04, label, transform=ax.transAxes, fontsize=6.5, color='0.35', ha='left', va='bottom')
    panel(ax, 'a', usetex)

    # ---------------- (b) X, Y, Z flagship
    ax = axs[1]
    rj = R["configs"]["xyz|joint|xyz"]; rg = R["configs"]["xyz|grouped|xyz"]
    x = np.arange(3)
    ax.axhline(1 / np.sqrt(3), color='0', lw=1.0)
    ax.axhline(gain / np.sqrt(3), color='0.45', ls=':', lw=0.9)
    ax.axhline(1.0, color='0.55', ls='--', lw=0.9)
    ax.axhline(gain, color='0.45', ls=':', lw=0.9)
    for r, m, off in ((rj, 'o', 0.0), (rg, 's', 0.0)):
        for mode, col, d in (("raw", "0.6", -0.12), (cal_mode, "0", 0.12)):
            vals = [err(r["estimates"][mode]["eta_eff"], k) for k in range(3)]
            y = np.array([v for v, _ in vals]); e = np.hstack([q for _, q in vals])
            ax.errorbar(x + d + off, y, yerr=e, fmt=m, color=col, mfc=col, mec=col, ms=3.8,
                        elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
    h = [Line2D([], [], ls='none', marker='o', color='0', ms=3.8), Line2D([], [], color='0', lw=1.0),
         Line2D([], [], ls='none', marker='s', color='0', ms=3.8), Line2D([], [], color='0.55', ls='--', lw=0.9)]
    ax.legend(h, ['joint POVM', r'$1/\sqrt{3}$', 'grouped (sharp)', r'$\eta=1$'], loc='center',
              ncol=2, borderaxespad=0.2, handlelength=1.4, handletextpad=0.5, columnspacing=0.7, bbox_to_anchor=(0.5, 0.42))
    ax.set_xticks(x); ax.set_xticklabels([r'$X$', r'$Y$', r'$Z$'])
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0.33 if label else 0.40, 1.12)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if label:
        ax.text(0.03, 0.04, label, transform=ax.transAxes, fontsize=6.5, color='0.35', ha='left', va='bottom')
    ax.set_xlabel(r'Pauli observable')
    ax.set_ylabel(r'effective sharpness $\eta_{\mathrm{eff}}$')
    panel(ax, 'b', usetex)

    return fig
