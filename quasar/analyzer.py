
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from .qusd import Plan, QuSD
from .gate_metrics import circuit_metrics

from qiskit import QuantumCircuit


@dataclass
class AnalysisResult:
    plan: Plan
    metrics_global: Dict[str, Any]


def _union_find_components(circ: QuantumCircuit) -> List[List[int]]:
    n = circ.num_qubits
    parent = list(range(n))
    rank = [0]*n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a,b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[rb] < rank[ra]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for inst, qargs, _ in circ.data:
        qubits = [circ.find_bit(q).index for q in qargs]
        if len(qubits) > 1:
            base = qubits[0]
            for q in qubits[1:]:
                union(base, q)

    comp_map: Dict[int, List[int]] = {}
    for q in range(n):
        r = find(q)
        comp_map.setdefault(r, []).append(q)
    return list(comp_map.values())

def _extract_subcircuit(circ: QuantumCircuit, qubit_subset: List[int]) -> QuantumCircuit:
    subset_sorted = sorted(qubit_subset)
    mapping = {g: i for i, g in enumerate(subset_sorted)}
    sub = QuantumCircuit(len(subset_sorted))
    for inst, qargs, cargs in circ.data:
        q_idxs = [circ.find_bit(q).index for q in qargs]
        if all(q in mapping for q in q_idxs):
            sub.append(inst, [sub.qubits[mapping[q]] for q in q_idxs], cargs)
    sub.metadata = {"global_qubits": subset_sorted, "mapping": mapping}
    return sub

def analyze(circuit: QuantumCircuit) -> AnalysisResult:
    comps = _union_find_components(circuit)
    plan = Plan(meta={"total_qubits": circuit.num_qubits, "components": len(comps)})
    for pid, qubits in enumerate(comps):
        sub = _extract_subcircuit(circuit, qubits)
        metrics = circuit_metrics(sub)
        qusd = QuSD(id=pid, qubits=qubits, circuit=sub, metrics=metrics)
        plan.add(qusd)

    global_metrics = circuit_metrics(circuit)
    return AnalysisResult(plan=plan, metrics_global=global_metrics)
