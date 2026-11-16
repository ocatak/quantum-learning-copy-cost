"""Circuits: state preparation + basis rotation U_n + Z measurement."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .axes import rotation_angles
from .states import prep_circuit


def rotation_circuit(n) -> QuantumCircuit:
    """U_n with U_n^dagger Z U_n = n . sigma (no measurement)."""
    alpha, beta = rotation_angles(n)
    qc = QuantumCircuit(1)
    if abs(alpha) > 0:
        qc.rz(-alpha, 0)
    if abs(beta) > 0:
        qc.ry(-beta, 0)
    return qc


def rotation_unitary(n) -> np.ndarray:
    return Operator(rotation_circuit(n)).data


def measurement_projectors(n) -> list[np.ndarray]:
    """[P_{o=+1}, P_{o=-1}] = U^dagger |o><o| U built from the actual rotation circuit."""
    U = rotation_unitary(n)
    out = []
    for bit in (0, 1):
        ket = np.zeros(2, dtype=complex); ket[bit] = 1.0
        out.append(U.conj().T @ np.outer(ket, ket.conj()) @ U)
    return out


def measure_circuit(prep: str, n, name: str | None = None) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1, name=name or f"{prep}")
    qc.compose(prep_circuit(prep), inplace=True)
    qc.compose(rotation_circuit(n), inplace=True)
    qc.measure(0, 0)
    return qc
