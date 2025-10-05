from __future__ import annotations

import pathlib
import sys

import numpy as np

from qiskit import QuantumCircuit

# Ensure the repository root is on sys.path when running from source.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasar.backends.sv import StatevectorBackend
from quasar.backends import tableau
from quasar.backends.tableau import TableauBackend


def _build_sample_clifford_circuit(num_qubits: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.s(0)
    qc.cx(0, 1)
    qc.h(1)
    qc.sdg(1)
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.swap(0, 2)
    qc.h(2)
    qc.s(2)
    qc.cx(2, 1)
    qc.cz(1, 0)
    return qc


def test_tableau_backend_executes_with_stim():
    assert tableau.stim_available(), "Stim backend is required for this test"

    circuit = _build_sample_clifford_circuit()

    backend = TableauBackend()
    stim_result = backend.run(circuit, want_statevector=True)

    assert stim_result is not None, "Stim backend should return a statevector for Clifford circuits"

    reference = StatevectorBackend().run(circuit)

    assert reference is not None

    nz = np.flatnonzero(np.abs(reference) > 1e-12)
    phase = 1.0 if len(nz) == 0 else stim_result[nz[0]] / reference[nz[0]]
    assert np.allclose(stim_result, reference * phase, atol=1e-8)
