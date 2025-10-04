
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit

def _line_graph_layer(qc: QuantumCircuit):
    n = qc.num_qubits
    for q in range(n):
        qc.h(q)
    for i in range(n-1):
        qc.cz(i, i+1)

def _sparse_diag_tail_layer(qc: QuantumCircuit, rng: np.random.Generator, *, angle_scale: float,
                            sparsity: float, bandwidth: int):
    n = qc.num_qubits
    k = max(1, int(round(sparsity * n)))
    idxs = rng.choice(n, size=k, replace=False)
    for q in idxs:
        theta = float(rng.uniform(-angle_scale, angle_scale))
        qc.rz(theta, int(q))
    m_pairs = max(1, k // 2)
    for _ in range(m_pairs):
        i = int(rng.integers(0, n-1))
        j = int(min(n-1, i + int(rng.integers(1, max(2, bandwidth+1)))))
        if i != j:
            qc.cz(min(i, j), max(i, j))

def dd_friendly_prefix_diag_tail(*, num_qubits: int, depth: int, cutoff: float = 0.8,
                                 angle_scale: float = 0.1, tail_sparsity: float = 0.05,
                                 tail_bandwidth: int = 2, seed: int = 1) -> QuantumCircuit:
    cutoff = max(0.0, min(1.0, float(cutoff)))
    depth = int(depth)
    rng = np.random.default_rng(int(seed))
    qc = QuantumCircuit(int(num_qubits))
    d_pre = int(round(depth * cutoff))
    d_tail = max(0, depth - d_pre)

    for _ in range(d_pre):
        _line_graph_layer(qc)

    for _ in range(d_tail):
        _sparse_diag_tail_layer(qc, rng, angle_scale=angle_scale,
                                sparsity=tail_sparsity, bandwidth=tail_bandwidth)
    return qc

def register_into(reg: dict):
    reg["dd_friendly_prefix_diag_tail"] = dd_friendly_prefix_diag_tail
