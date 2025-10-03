
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import hashlib

from qiskit import QuantumCircuit


def _fingerprint(text: str) -> str:
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
            qasm = self.circuit.qasm()  # deprecated in new qiskit, but fine here
        except Exception:
            qasm = str(self.circuit)
        return _fingerprint(qasm + str(self.qubits))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.compute_fingerprint()
        d["num_qubits"] = int(self.circuit.num_qubits) if hasattr(self.circuit, "num_qubits") else len(self.qubits)
        d["depth"] = int(self.circuit.depth()) if hasattr(self.circuit, "depth") else None
        d["backend"] = self.backend
        # Avoid dumping the full circuit object
        d.pop("circuit", None)
        return d

@dataclass
class SSD:
    partitions: List[PartitionNode] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "partitions": [p.to_dict() for p in self.partitions],
        }

    def add(self, node: PartitionNode) -> None:
        self.partitions.append(node)

    def __len__(self) -> int:
        return len(self.partitions)
