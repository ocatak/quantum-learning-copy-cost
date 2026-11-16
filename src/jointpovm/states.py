"""Input states: preparation circuits and exact Bloch vectors.

A (possibly mixed) state is a list of (pure-prep name, weight). Mixtures are realised as
i.i.d. mixtures: each copy's component is drawn at random (multinomial shot counts).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

PHI_PAIR_DEG = 35.0           # Ry(35 deg)|0>
XYZ_RY_DEG, XYZ_RZ_DEG = 55.0, 30.0   # Rz(30 deg) Ry(55 deg)|0>


def _bloch_ry_rz(ry_deg, rz_deg=0.0):
    t, p = np.radians(ry_deg), np.radians(rz_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


PURE = {
    "zero": np.array([0.0, 0.0, 1.0]),
    "one": np.array([0.0, 0.0, -1.0]),
    "plus": np.array([1.0, 0.0, 0.0]),
    "minus": np.array([-1.0, 0.0, 0.0]),
    "plus_i": np.array([0.0, 1.0, 0.0]),
    "minus_i": np.array([0.0, -1.0, 0.0]),
    "ry35": _bloch_ry_rz(PHI_PAIR_DEG),
    "xyz": _bloch_ry_rz(XYZ_RY_DEG, XYZ_RZ_DEG),
}

MIXTURES = {name: [(name, 1.0)] for name in PURE}
MIXTURES["mixed"] = [("zero", 0.5), ("one", 0.5)]

CARDINAL = ["zero", "one", "plus", "minus", "plus_i", "minus_i"]


def prep_circuit(name: str) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    if name == "zero":
        pass
    elif name == "one":
        qc.x(0)
    elif name == "plus":
        qc.h(0)
    elif name == "minus":
        qc.x(0); qc.h(0)
    elif name == "plus_i":
        qc.h(0); qc.s(0)               # S H |0> : Bloch vector +y
    elif name == "minus_i":
        qc.h(0); qc.sdg(0)
    elif name == "ry35":
        qc.ry(np.radians(PHI_PAIR_DEG), 0)
    elif name == "xyz":
        qc.ry(np.radians(XYZ_RY_DEG), 0); qc.rz(np.radians(XYZ_RZ_DEG), 0)
    else:
        raise KeyError(f"unknown state {name!r}")
    return qc


def bloch(state: str) -> np.ndarray:
    """Exact Bloch vector of a (possibly mixed) state."""
    return sum(w * PURE[c] for c, w in MIXTURES[state])
