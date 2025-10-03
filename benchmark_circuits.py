
from __future__ import annotations
from typing import List
from qiskit import QuantumCircuit
import numpy as np

def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    if n <= 0:
        return qc
    qc.h(0)
    for i in range(1, n):
        qc.cx(i-1, i)
    return qc

def ghz_clusters_random(num_qubits: int, block_size: int = 8, depth: int = 200, seed: int = 1) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        sub = ghz(end - start)
        qc.compose(sub, qubits=list(range(start, end)), inplace=True)
    oneq = ["h","rx","ry","rz","s","sdg","x","z"]
    twoq = ["cx","cz","swap","rzz"]
    for _ in range(depth):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            for q in range(start, end):
                g = rng.choice(oneq)
                if g in {"rx","ry","rz"}:
                    getattr(qc, g)(float(rng.uniform(0, 2*np.pi)), q)
                else:
                    getattr(qc, g)(q)
            block = list(range(start, end))
            rng.shuffle(block)
            for a, b in zip(block[::2], block[1::2]):
                g = rng.choice(twoq)
                if g == "rzz":
                    qc.rzz(float(rng.uniform(0, 2*np.pi)), a, b)
                else:
                    getattr(qc, g)(a, b)
    return qc

def random_clifford(num_qubits: int, depth: int = 200, seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    cliff1 = ["h","s","sdg","x","z"]
    cliff2 = ["cx","cz","swap"]
    for _ in range(depth):
        for q in range(num_qubits):
            getattr(qc, rng.choice(cliff1))(q)
        order = list(range(num_qubits))
        rng.shuffle(order)
        for a,b in zip(order[::2], order[1::2]):
            getattr(qc, rng.choice(cliff2))(a,b)
    return qc
