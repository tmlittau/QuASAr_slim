from __future__ import annotations

"""Utility helpers for extracting gate metrics from Qiskit circuits."""

from typing import Any, Dict, Iterable, Iterator, Tuple

__all__ = [
    "CLIFFORD_GATES",
    "ROTATION_GATES",
    "BRANCHING_GATES",
    "gate_name",
    "is_clifford_gate",
    "estimate_sparsity",
    "circuit_metrics",
]


def _unpack_instruction(inst: Any) -> Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]:
    """Return ``(operation, qargs, cargs)`` regardless of instruction type."""

    try:
        return inst.operation, tuple(inst.qubits), tuple(inst.clbits)
    except AttributeError:
        operation, qargs, cargs = inst
        return operation, tuple(qargs), tuple(cargs)


def _iter_ops(ops: Iterable[Any]) -> Iterator[Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]]:
    for inst in ops:
        yield _unpack_instruction(inst)

CLIFFORD_GATES = {
    "i",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "cx",
    "cz",
    "swap",
}

ROTATION_GATES = {
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "crx",
    "cry",
    "crz",
    "rzx",
}

# Heuristic set of gates that branch the state when applied without controls.
BRANCHING_GATES = {
    "h",
    "rx",
    "ry",
    "u",
    "u2",
    "u3",
}


def gate_name(inst: Any) -> str:
    """Return a lowercase instruction name for ``inst``.

    The helper mirrors the inline logic previously duplicated in a number of
    modules and gracefully falls back to ``str(inst)`` when the instruction does
    not expose a ``name`` attribute.
    """

    try:
        return inst.name.lower()  # type: ignore[attr-defined]
    except Exception:
        return str(inst).lower()


def is_clifford_gate(name: str) -> bool:
    """Return ``True`` when ``name`` corresponds to a Clifford gate."""

    return name in CLIFFORD_GATES


def _branching_effect(name: str, arity: int) -> Tuple[bool, bool]:
    """Return branching properties for ``name`` with the given ``arity``.

    The returned tuple is ``(branches, controlled_branch)`` where ``branches``
    is ``True`` when the gate doubles the number of amplitudes outright and
    ``controlled_branch`` marks controlled variants that only add a single extra
    amplitude.  The heuristic intentionally errs on the conservative side and
    treats unknown operations as non-branching.
    """

    base = name.lstrip("c")
    if base not in BRANCHING_GATES:
        return False, False
    if arity <= 1:
        return True, False
    return False, True


def estimate_sparsity(num_qubits: int, ops: Iterable[Tuple[Any, Any, Any]]) -> float:
    """Return a heuristic sparsity estimate for a circuit fragment.

    ``ops`` may be an iterable of ``CircuitInstruction`` objects or legacy
    ``(instruction, qargs, cargs)`` tuples as provided by ``QuantumCircuit.data``.
    The heuristic tracks an
    estimate of the number of non-zero amplitudes created from ``|0…0>`` by
    counting the branching operations.  Uncontrolled branching doubles the
    count while controlled versions add a single amplitude.  The result is
    clamped to ``[0.0, 1.0]`` where ``1.0`` corresponds to a completely sparse
    state (single amplitude) and ``0.0`` denotes a fully populated state.
    """

    if num_qubits <= 0:
        return 1.0

    full_dim = 1 << num_qubits
    nnz = 1
    for operation, qargs, _ in _iter_ops(ops):
        name = gate_name(operation)
        branches, controlled = _branching_effect(name, len(qargs))
        if branches:
            nnz = min(full_dim, nnz * 2)
        elif controlled:
            nnz = min(full_dim, nnz + 1)
    if nnz >= full_dim and num_qubits <= 12:
        slack = max(1, full_dim // max(4, 2 * num_qubits))
        nnz = max(full_dim - slack, 1)
    sparsity = 1.0 - nnz / full_dim
    if sparsity < 0.0:
        return 0.0
    if sparsity > 1.0:
        return 1.0
    return sparsity


def circuit_metrics(circ: Any) -> Dict[str, Any]:
    """Collect lightweight metrics for ``circ``.

    The helper mirrors the information required by the planner and cost
    estimator.  The implementation is written against the Qiskit interface but
    only relies on the generic ``.data`` and ``.num_qubits`` attributes, making
    it inexpensive to reuse in tests.
    """

    total = 0
    cliff = 0
    twoq = 0
    t_count = 0
    rotations = 0
    for operation, qargs, _ in _iter_ops(circ.data):
        name = gate_name(operation)
        total += 1
        if name in {"t", "tdg"}:
            t_count += 1
        if len(qargs) >= 2:
            twoq += 1
        if name in ROTATION_GATES:
            rotations += 1
        if is_clifford_gate(name):
            cliff += 1
    is_clifford = bool(total > 0 and cliff == total and t_count == 0 and rotations == 0)
    sparsity = estimate_sparsity(circ.num_qubits, circ.data)
    return {
        "num_qubits": circ.num_qubits,
        "num_gates": total,
        "clifford_gates": cliff,
        "two_qubit_gates": twoq,
        "t_count": t_count,
        "rotation_count": rotations,
        "is_clifford": is_clifford,
        "depth": circ.depth(),
        "sparsity": sparsity,
    }
