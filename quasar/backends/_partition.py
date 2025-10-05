from __future__ import annotations

"""Helpers for converting partition objects into backend-agnostic operations."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class Operation:
    """Light-weight gate descriptor extracted from a partition circuit."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()


def _coerce_quantum_circuit(circuit: Any) -> QuantumCircuit:
    """Return a :class:`QuantumCircuit` representation of *circuit*."""

    if isinstance(circuit, QuantumCircuit):
        return circuit
    if hasattr(circuit, "to_qiskit"):
        converted = circuit.to_qiskit()
        if isinstance(converted, QuantumCircuit):
            return converted
    raise TypeError("Partition cannot be converted to a QuantumCircuit instance")


def extract_operations(circuit: Any) -> Tuple[int, List[Operation]]:
    """Extract a list of backend-agnostic operations from *circuit*."""

    qc = _coerce_quantum_circuit(circuit)
    if qc.num_qubits == 0:
        return 0, []

    qubit_indices = {qubit: index for index, qubit in enumerate(qc.qubits)}
    operations: List[Operation] = []

    def _iter_instructions(data: Iterable[Any]) -> Iterable[Tuple[Any, Sequence[Any]]]:
        for entry in data:
            operation = getattr(entry, "operation", None)
            qubits = getattr(entry, "qubits", None)
            if operation is not None and qubits is not None:
                yield operation, qubits
                continue
            # Legacy tuple layout: (instruction, qargs, cargs)
            if isinstance(entry, tuple) and len(entry) >= 2:
                yield entry[0], entry[1]
                continue
            raise TypeError("Unrecognised circuit data entry: {!r}".format(entry))

    for inst, qargs in _iter_instructions(qc.data):
        name = getattr(inst, "name", "").lower()
        if name in {"barrier"}:
            continue
        qubits = tuple(qubit_indices[q] for q in qargs)
        raw_params = getattr(inst, "params", ())
        try:
            params = tuple(float(complex(p)) for p in raw_params)
        except TypeError as exc:  # pragma: no cover - unexpected symbolic parameter
            raise TypeError(f"Unsupported symbolic parameter in instruction '{name}'") from exc
        operations.append(Operation(name=name, qubits=qubits, params=params))

    return qc.num_qubits, operations
