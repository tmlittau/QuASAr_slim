
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Iterable
import hashlib
import json

from qiskit import QuantumCircuit

from .cost_estimator import CostEstimator

def _fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

@dataclass
class PartitionNode:
    id: int
    qubits: List[int]
    circuit: QuantumCircuit
    metrics: Dict[str, Any] = field(default_factory=dict)
    backend: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def set_backend(self, name: str) -> None:
        self.backend = name

    def clone(self) -> "PartitionNode":
        return PartitionNode(
            id=int(self.id),
            qubits=list(self.qubits),
            circuit=self.circuit,
            metrics=dict(self.metrics),
            backend=self.backend,
            meta=dict(self.meta),
        )

    def compute_fingerprint(self) -> str:
        try:
            qasm = self.circuit.qasm()  # deprecated in some qiskit versions
        except Exception:
            qasm = str(self.circuit)
        return _fingerprint(qasm)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.compute_fingerprint()
        d["num_qubits"] = int(self.circuit.num_qubits) if hasattr(self.circuit, "num_qubits") else len(self.qubits)
        d["depth"] = int(self.circuit.depth()) if hasattr(self.circuit, "depth") else None
        d["backend"] = self.backend
        d.pop("circuit", None)  # don't serialize full circuit
        return d

@dataclass
class SSD:
    partitions: List[PartitionNode] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    partition_cache: Dict[str, PartitionNode] = field(default_factory=dict)
    estimated_cost: float = 0.0
    decision_trace: List[str] = field(default_factory=list)
    _cached_fingerprint: Optional[str] = field(default=None, init=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "partitions": [p.to_dict() for p in self.partitions],
            "estimated_cost": float(self.estimated_cost),
            "decision_trace": list(self.decision_trace),
        }

    def add(self, node: PartitionNode) -> None:
        fingerprint = node.compute_fingerprint()
        node.meta.setdefault("fingerprint", fingerprint)
        cached = self.partition_cache.get(fingerprint)
        if cached is not None:
            node.meta["cache_hit"] = True
            node.meta["cache_source"] = cached.id
        else:
            node.meta["cache_hit"] = False
            node.meta["cache_source"] = node.id
            self.partition_cache[fingerprint] = node
        self.partitions.append(node)
        self._cached_fingerprint = None

    def __len__(self) -> int:
        return len(self.partitions)

    def fork(self) -> "SSD":
        new_partitions = [node.clone() for node in self.partitions]
        new_cache: Dict[str, PartitionNode] = {}
        for node in new_partitions:
            fingerprint = node.meta.get("fingerprint") if isinstance(node.meta, dict) else None
            if not fingerprint:
                fingerprint = node.compute_fingerprint()
                node.meta["fingerprint"] = fingerprint
            new_cache[str(fingerprint)] = node
        clone = SSD(
            partitions=new_partitions,
            meta=dict(self.meta),
            partition_cache=new_cache,
            estimated_cost=self.estimated_cost,
            decision_trace=list(self.decision_trace),
        )
        clone._cached_fingerprint = self._cached_fingerprint
        return clone

    def extend_plan(self, nodes: Iterable[PartitionNode], estimator: CostEstimator) -> None:
        for node in nodes:
            clone = node.clone()
            self.add(clone)
            self.estimated_cost += self._estimate_node_cost(clone, estimator)
            reason = clone.meta.get("planner_reason") if isinstance(clone.meta, dict) else None
            if reason:
                self.decision_trace.append(str(reason))

    @property
    def fingerprint(self) -> str:
        if not self._cached_fingerprint:
            self._cached_fingerprint = self._plan_fingerprint()
        return self._cached_fingerprint

    def _plan_fingerprint(self) -> str:
        payload: List[Dict[str, Any]] = []
        for node in self.partitions:
            fingerprint = None
            if isinstance(node.meta, dict):
                fingerprint = node.meta.get("fingerprint")
            if not fingerprint:
                fingerprint = node.compute_fingerprint()
            payload.append(
                {
                    "id": int(node.id),
                    "qubits": tuple(node.qubits),
                    "backend": node.backend,
                    "fingerprint": fingerprint,
                }
            )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()[:16]

    @staticmethod
    def _estimate_node_cost(node: PartitionNode, estimator: CostEstimator) -> float:
        metrics = node.metrics or {}
        backend = node.backend or "sv"
        n = int(metrics.get("num_qubits", 0) or 0)
        twoq = int(metrics.get("two_qubit_gates", 0) or 0)
        total_gates = int(metrics.get("num_gates", 0) or 0)
        oneq = max(total_gates - twoq, 0)
        rotations = int(metrics.get("rotation_count", 0) or 0)
        sparsity = float(metrics.get("sparsity", 0.0) or 0.0)

        if backend == "sv":
            return estimator.sv_cost(n, oneq, twoq)
        if backend == "dd":
            return estimator.decision_diagram_cost(
                n=n,
                num_gates=total_gates,
                twoq=twoq,
                rotation_count=rotations,
                sparsity=sparsity,
            )
        if backend == "tableau":
            return estimator.tableau_prefix_cost(n, oneq, twoq)
        return estimator.sv_cost(n, oneq, twoq)
