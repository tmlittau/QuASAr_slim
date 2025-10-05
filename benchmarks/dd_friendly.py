"""Decision-diagram friendly benchmark circuits."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["dd_friendly_prefix_diag_tail", "CIRCUIT_REGISTRY", "build"]


def _line_graph_layer(qc: QuantumCircuit) -> None:
    num_qubits = qc.num_qubits
    for qubit in range(num_qubits):
        qc.h(qubit)
    for index in range(num_qubits - 1):
        qc.cz(index, index + 1)


def _sparse_diag_tail_layer(
    qc: QuantumCircuit,
    rng: np.random.Generator,
    *,
    angle_scale: float,
    sparsity: float,
    bandwidth: int,
) -> None:
    num_qubits = qc.num_qubits
    k = max(1, int(round(sparsity * num_qubits)))
    idxs = rng.choice(num_qubits, size=k, replace=False)
    for qubit in idxs:
        theta = float(rng.uniform(-angle_scale, angle_scale))
        qc.rz(theta, int(qubit))
    m_pairs = max(1, k // 2)
    for _ in range(m_pairs):
        i = int(rng.integers(0, num_qubits - 1))
        j = int(min(num_qubits - 1, i + int(rng.integers(1, max(2, bandwidth + 1)))))
        if i != j:
            qc.cz(min(i, j), max(i, j))


def dd_friendly_prefix_diag_tail(
    *,
    num_qubits: int,
    depth: int,
    cutoff: float = 0.8,
    angle_scale: float = 0.1,
    tail_sparsity: float = 0.05,
    tail_bandwidth: int = 2,
    seed: int = 1,
) -> QuantumCircuit:
    cutoff = max(0.0, min(1.0, float(cutoff)))
    depth = int(depth)
    rng = np.random.default_rng(int(seed))
    qc = QuantumCircuit(int(num_qubits))
    d_pre = int(round(depth * cutoff))
    d_tail = max(0, depth - d_pre)

    for _ in range(d_pre):
        _line_graph_layer(qc)

    for _ in range(d_tail):
        _sparse_diag_tail_layer(
            qc,
            rng,
            angle_scale=angle_scale,
            sparsity=tail_sparsity,
            bandwidth=tail_bandwidth,
        )
    return qc


CIRCUIT_REGISTRY: Dict[str, Any] = {
    "dd_friendly_prefix_diag_tail": dd_friendly_prefix_diag_tail,
}


def build(kind: str, /, **kwargs: Any) -> QuantumCircuit:
    try:
        builder = CIRCUIT_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown DD-friendly circuit kind: {kind}") from exc
    return builder(**kwargs)
