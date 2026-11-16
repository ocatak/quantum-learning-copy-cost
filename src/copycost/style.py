"""Matplotlib style of the paper figures (grayscale, 6.4 in wide, Computer Modern).

With a LaTeX installation (``latex`` on PATH) text is typeset by LaTeX, exactly as in the
paper. Without one, matplotlib's bundled Computer Modern fonts and mathtext are used.
"""
from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt

WIDTH = 6.4

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

FALLBACK_RC = {'text.usetex': False, 'font.family': 'serif', 'font.serif': ['cmr10'],
               'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False}


def find_latex() -> bool:
    """True if a latex binary is available; adds the macOS TeX Live path if needed."""
    if shutil.which("latex"):
        return True
    for d in ("/Library/TeX/texbin", "/usr/local/texlive/bin"):
        if os.path.exists(os.path.join(d, "latex")):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def setup(usetex: bool | None = None) -> bool:
    """Apply the paper style. usetex=None: use LaTeX if it is installed. Returns the usetex flag."""
    if usetex is None:
        usetex = find_latex()
    plt.rcParams.update(PAPER_RC)
    if not usetex:
        plt.rcParams.update(FALLBACK_RC)
    return usetex


def bold(s: str) -> str:
    return r'\textbf{' + s + '}' if plt.rcParams['text.usetex'] else s


def panel(ax, lab: str):
    """Panel label (a), (b) in the upper-left corner, outside the axes."""
    kw = {} if plt.rcParams['text.usetex'] else {'family': 'cmb10'}
    ax.text(-0.16, 1.04, bold('(' + lab + ')'), transform=ax.transAxes, fontsize=9.5, va='bottom', **kw)
