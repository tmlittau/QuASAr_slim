
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable
import hashlib
import json

from qiskit import QuantumCircuit

from .cost_estimator import CostEstimator


def _fingerprint(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class QuSD:
    """Representation of a quantum subcircuit."""

    id: int
    qubits: List[int]
    circuit: QuantumCircuit
    metrics: Dict[str, Any] = field(default_factory=dict)
    backend: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    out_state: Optional[Any] = None
    successors: List["QuSD"] = field(default_factory=list)
    predecessors: List["QuSD"] = field(default_factory=list)

    def set_backend(self, name: str) -> None:
        self.backend = name

    def attach_successor(self, successor: "QuSD") -> None:
        if successor not in self.successors:
            self.successors.append(successor)
        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def detach_successor(self, successor: "QuSD") -> None:
        if successor in self.successors:
            self.successors.remove(successor)
        if self in successor.predecessors:
            successor.predecessors.remove(self)

    def clone(self, include_state: bool = False) -> "QuSD":
        return QuSD(
            id=int(self.id),
            qubits=list(self.qubits),
            circuit=self.circuit,
            metrics=dict(self.metrics),
            backend=self.backend,
            meta=dict(self.meta),
            out_state=self.out_state if include_state else None,
        )

    def compute_fingerprint(self) -> str:
        try:
            qasm = self.circuit.qasm()
        except Exception:
            qasm = str(self.circuit)
        return _fingerprint(qasm)

    def to_dict(self) -> Dict[str, Any]:
        num_qubits = (
            int(self.circuit.num_qubits)
            if hasattr(self.circuit, "num_qubits")
            else len(self.qubits)
        )
        depth = int(self.circuit.depth()) if hasattr(self.circuit, "depth") else None
        return {
            "id": int(self.id),
            "qubits": list(self.qubits),
            "metrics": dict(self.metrics),
            "backend": self.backend,
            "meta": dict(self.meta),
            "fingerprint": self.meta.get("fingerprint")
            or self.compute_fingerprint(),
            "num_qubits": num_qubits,
            "depth": depth,
            "has_out_state": self.out_state is not None,
            "successors": [s.id for s in self.successors],
        }


@dataclass
class Plan:
    qusds: List[QuSD] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    qusd_cache: Dict[str, QuSD] = field(default_factory=dict)
    estimated_cost: float = 0.0
    decision_trace: List[str] = field(default_factory=list)
    _cached_fingerprint: Optional[str] = field(default=None, init=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        payload = [q.to_dict() for q in self.qusds]
        return {
            "meta": dict(self.meta),
            "qusds": payload,
            "estimated_cost": float(self.estimated_cost),
            "decision_trace": list(self.decision_trace),
        }

    def add(self, qusd: QuSD, *, predecessors: Optional[Iterable[QuSD]] = None) -> None:
        fingerprint = qusd.meta.get("fingerprint") or qusd.compute_fingerprint()
        qusd.meta.setdefault("fingerprint", fingerprint)
        cached = self.qusd_cache.get(fingerprint)
        if cached is not None:
            qusd.meta["cache_hit"] = True
            qusd.meta["cache_source"] = cached.id
        else:
            qusd.meta["cache_hit"] = False
            qusd.meta["cache_source"] = qusd.id
            self.qusd_cache[fingerprint] = qusd
        if predecessors:
            for pred in predecessors:
                pred.attach_successor(qusd)
        elif self.qusds:
            self.qusds[-1].attach_successor(qusd)
        self.qusds.append(qusd)
        self._cached_fingerprint = None

    def __len__(self) -> int:
        return len(self.qusds)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.qusds)

    def fork(self) -> "Plan":
        clone_map: Dict[int, QuSD] = {}
        new_qusds: List[QuSD] = []
        for qusd in self.qusds:
            clone = qusd.clone(include_state=False)
            clone.meta.pop("cache_hit", None)
            clone.meta.pop("cache_source", None)
            clone_map[id(qusd)] = clone
            new_qusds.append(clone)

        for qusd in self.qusds:
            clone = clone_map[id(qusd)]
            for succ in qusd.successors:
                clone.attach_successor(clone_map[id(succ)])

        new_cache: Dict[str, QuSD] = {}
        for qusd in new_qusds:
            fingerprint = qusd.meta.get("fingerprint") or qusd.compute_fingerprint()
            new_cache[str(fingerprint)] = qusd

        clone_plan = Plan(
            qusds=new_qusds,
            meta=dict(self.meta),
            qusd_cache=new_cache,
            estimated_cost=self.estimated_cost,
            decision_trace=list(self.decision_trace),
        )
        clone_plan._cached_fingerprint = self._cached_fingerprint
        return clone_plan

    def extend_plan(self, qusds: Iterable[QuSD], estimator: CostEstimator) -> None:
        for qusd in qusds:
            clone = qusd.clone(include_state=False)
            self.add(clone)
            self.estimated_cost += self._estimate_qusd_cost(clone, estimator)
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
        for qusd in self.qusds:
            fingerprint = None
            if isinstance(qusd.meta, dict):
                fingerprint = qusd.meta.get("fingerprint")
            if not fingerprint:
                fingerprint = qusd.compute_fingerprint()
            payload.append(
                {
                    "id": int(qusd.id),
                    "qubits": tuple(qusd.qubits),
                    "backend": qusd.backend,
                    "fingerprint": fingerprint,
                    "successors": sorted(s.id for s in qusd.successors),
                }
            )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()[:16]

    @staticmethod
    def _estimate_qusd_cost(qusd: QuSD, estimator: CostEstimator) -> float:
        metrics = qusd.metrics or {}
        backend = qusd.backend or "sv"
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


__all__ = ["QuSD", "Plan"]
