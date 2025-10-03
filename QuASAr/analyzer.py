
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from .SSD import SSD, PartitionNode

from qiskit import QuantumCircuit


CLIFFORD = {
    "i","x","y","z","h","s","sdg","cx","cz","swap"
}

ROTATION_GATES = {"rx","ry","rz","rxx","ryy","rzz","crx","cry","crz","rzx"}

@dataclass
class AnalysisResult:
    ssd: SSD
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

    # connect qubits that share any multi-qubit gate
    for inst, qargs, _ in circ.data:
        qubits = [circ.find_bit(q).index for q in qargs]
        if len(qubits) > 1:
            base = qubits[0]
            for q in qubits[1:]:
                union(base, q)

    # components
    comp_map: Dict[int, List[int]] = {}
    for q in range(n):
        r = find(q)
        comp_map.setdefault(r, []).append(q)
    return list(comp_map.values())

def _extract_subcircuit(circ: QuantumCircuit, qubit_subset: List[int]) -> QuantumCircuit:
    # preserve order; remap to local indices [0..k-1]
    subset_sorted = sorted(qubit_subset)
    mapping = {g: i for i, g in enumerate(subset_sorted)}
    sub = QuantumCircuit(len(subset_sorted))
    for inst, qargs, cargs in circ.data:
        q_idxs = [circ.find_bit(q).index for q in qargs]
        if all(q in mapping for q in q_idxs):
            sub.append(inst, [sub.qubits[mapping[q]] for q in q_idxs], cargs)
    sub.metadata = {"global_qubits": subset_sorted, "mapping": mapping}
    return sub

def _gate_name(inst) -> str:
    try:
        return inst.name.lower()
    except Exception:
        return str(inst).lower()

def _is_clifford_inst(inst_name: str) -> bool:
    return inst_name in CLIFFORD

def _metrics_for(circ: QuantumCircuit) -> Dict[str, Any]:
    total = 0
    cliff = 0
    twoq = 0
    t_count = 0
    rotations = 0
    for inst, qargs, _ in circ.data:
        name = _gate_name(inst)
        total += 1
        if name in {"t","tdg"}:
            t_count += 1
        if len(qargs) >= 2:
            twoq += 1
        if name in ROTATION_GATES:
            rotations += 1
        if _is_clifford_inst(name):
            cliff += 1
    is_clifford = (total > 0 and cliff == total and t_count == 0 and rotations == 0)
    return {
        "num_qubits": circ.num_qubits,
        "num_gates": total,
        "clifford_gates": cliff,
        "two_qubit_gates": twoq,
        "t_count": t_count,
        "rotation_count": rotations,
        "is_clifford": is_clifford,
        "depth": circ.depth(),
    }

def analyze(circuit: QuantumCircuit) -> AnalysisResult:
    comps = _union_find_components(circuit)
    ssd = SSD(meta={"total_qubits": circuit.num_qubits, "components": len(comps)})
    for pid, qubits in enumerate(comps):
        sub = _extract_subcircuit(circuit, qubits)
        metrics = _metrics_for(sub)
        node = PartitionNode(id=pid, qubits=qubits, circuit=sub, metrics=metrics)
        ssd.add(node)

    global_metrics = _metrics_for(circuit)
    return AnalysisResult(ssd=ssd, metrics_global=global_metrics)
