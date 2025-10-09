
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import hashlib

try:
    from qiskit import QuantumCircuit
except Exception:  # pragma: no cover
    QuantumCircuit = Any  # type: ignore

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

    def compute_fingerprint(self) -> str:
        try:
            qasm = self.circuit.qasm()  # deprecated in some qiskit versions
        except Exception:
            qasm = str(self.circuit)
        return _fingerprint(qasm + str(self.qubits))

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

    def to_dict(self) -> Dict[str, Any]:
        return {"meta": dict(self.meta), "partitions": [p.to_dict() for p in self.partitions]}

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

    def __len__(self) -> int:
        return len(self.partitions)
