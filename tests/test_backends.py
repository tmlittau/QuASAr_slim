from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit

# Ensure the repository root is on sys.path when running from source.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XGate
from qiskit.circuit.exceptions import CircuitError

from quasar.backends import DecisionDiagramBackend, StatevectorBackend
from quasar.backends.sv import StatevectorQubitMappingError


def _bell_state_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _expected_bell_state() -> np.ndarray:
    vec = np.zeros(4, dtype=np.complex128)
    amp = 1 / math.sqrt(2)
    vec[0] = amp
    vec[3] = amp
    return vec


def test_statevector_backend_produces_bell_state():
    backend = StatevectorBackend()

    result = backend.run(_bell_state_circuit())

    assert result is not None, "Statevector backend should return a statevector"
    assert result.shape == (4,)
    assert np.allclose(result, _expected_bell_state())


def test_statevector_backend_supports_initial_state():
    backend = StatevectorBackend()

    qc = QuantumCircuit(1)
    qc.x(0)

    initial_state = np.array([0.0, 1.0], dtype=np.complex128)

    result = backend.run(qc, initial_state=initial_state)

    assert result is not None, "Statevector backend should evolve custom initial states"
    assert result.shape == (2,)
    assert np.allclose(result, np.array([1.0, 0.0], dtype=np.complex128))


def test_decision_diagram_backend_matches_bell_state():
    backend = DecisionDiagramBackend()

    result = backend.run(_bell_state_circuit())

    assert result is not None, "DDSIM backend should return a decision diagram"

    vector = np.array(result.get_vector(), dtype=np.complex128)

    assert vector.shape == (4,)
    assert np.allclose(vector, _expected_bell_state())


def test_statevector_backend_flags_mismapped_qubits():
    class MiswiredCircuit:
        def __init__(self) -> None:
            good_reg = QuantumRegister(1, "good")
            bad_reg = QuantumRegister(1, "bad")
            self.qubits = [good_reg[0]]
            self.num_qubits = 1
            self.partition_id = "test-partition"
            self.data = [(XGate(), [bad_reg[0]], [])]

        def find_bit(self, bit):  # type: ignore[override]
            raise CircuitError("bit not found")

    backend = StatevectorBackend()

    with pytest.raises(StatevectorQubitMappingError) as exc_info:
        backend.run(MiswiredCircuit())

    message = str(exc_info.value)
    assert "test-partition" in message
    assert "x" in message.lower()
    assert "bad" in message
