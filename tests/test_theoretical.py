from __future__ import annotations

from qiskit import QuantumCircuit

from quasar.theoretical import (
    estimate_decision_diagram,
    estimate_quasar,
    estimate_statevector,
    estimate_tableau,
)


def _clifford_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _non_clifford_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)
    return qc


def test_statevector_estimate_counts_amp_ops() -> None:
    qc = _clifford_circuit()
    estimate = estimate_statevector(qc)
    assert estimate.backend == "sv"
    assert estimate.work_unit_label == "amp_ops"
    assert estimate.work_units == 20.0  # 4 amplitudes * (1 one-q + 4 two-q)
    assert estimate.memory_bytes == 64


def test_decision_diagram_estimate_positive_cost() -> None:
    qc = _clifford_circuit()
    estimate = estimate_decision_diagram(qc)
    assert estimate.backend == "dd"
    assert estimate.work_units > 0.0
    assert estimate.memory_bytes > 0


def test_tableau_estimate_handles_clifford_and_non_clifford() -> None:
    clifford = estimate_tableau(_clifford_circuit())
    assert clifford.ok
    assert clifford.work_units > 0.0
    non_clifford = estimate_tableau(_non_clifford_circuit())
    assert not non_clifford.ok
    assert "non-Clifford" in (non_clifford.reason or "")


def test_quasar_estimate_reports_single_backend_reason() -> None:
    qc = _clifford_circuit()
    result = estimate_quasar(qc)
    assert result.ok
    assert result.single_backend == "tableau"
    assert result.single_backend_reason and "is_clifford" in result.single_backend_reason
    assert "tableau_ops" in result.work_units_by_label
    partition = result.partitions[0]
    assert partition.backend == "tableau"
    assert partition.estimate.memory_bytes == 64
