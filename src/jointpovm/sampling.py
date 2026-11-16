"""Shot allocation: turn (scheme, state, N copies) into concrete circuits with shot counts.

Joint schemes are i.i.d. randomized POVMs: every copy independently picks an axis with
probability p_axis (and, for a mixed input, a pure component with its weight). We therefore
draw the number of shots per (component, axis) circuit from a MULTINOMIAL with a seeded RNG.
Deterministic proportional allocation (round(N p_axis) shots per axis) would be stratified
sampling: it removes the axis-choice randomness, has a different (smaller) variance, and no
longer realises the joint POVM copy by copy. We do not use it for joint schemes.

Grouped schemes are, by definition, deterministic splits: N/|axes| copies per axis
(remainder spread over the first axes). A mixed input is still drawn i.i.d. per copy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import measure_circuit
from .schemes import Scheme
from .states import MIXTURES


@dataclass
class PubSpec:
    config: str          # configuration id
    axis: int            # axis index within the scheme
    prep: str            # pure component prepared
    shots: int


@dataclass
class Config:
    cid: str
    scheme: Scheme
    state: str
    n_copies: int
    role: str = "main"   # 'main', 'tomo', 'cal'


def allocate(cfg: Config, rng: np.random.Generator) -> list[PubSpec]:
    comps = MIXTURES[cfg.state]
    w = np.array([cw for _, cw in comps], dtype=float)
    sch = cfg.scheme
    specs: list[PubSpec] = []
    if sch.kind == "joint":
        probs = np.outer(w, sch.probs).ravel()          # (component, axis) pairs
        counts = rng.multinomial(cfg.n_copies, probs / probs.sum())
        for idx, n in enumerate(counts):
            ci, ai = divmod(idx, sch.n_axes)
            if n > 0:
                specs.append(PubSpec(cfg.cid, ai, comps[ci][0], int(n)))
    elif sch.kind == "grouped":
        per_axis = [len(x) for x in np.array_split(np.arange(cfg.n_copies), sch.n_axes)]
        for ai, n_ax in enumerate(per_axis):
            counts = rng.multinomial(n_ax, w / w.sum())
            for ci, n in enumerate(counts):
                if n > 0:
                    specs.append(PubSpec(cfg.cid, ai, comps[ci][0], int(n)))
    else:
        raise ValueError(sch.kind)
    return specs


def build_circuit(spec: PubSpec, scheme: Scheme):
    return measure_circuit(spec.prep, scheme.axes[spec.axis],
                           name=f"{spec.config}|{spec.axis}|{spec.prep}")
