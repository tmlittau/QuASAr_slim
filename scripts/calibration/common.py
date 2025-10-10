"""Shared helpers for calibration scripts."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quasar.gate_metrics import circuit_metrics, gate_name


def require_dependency(module: str, install_hint: str) -> None:
    """Abort execution when *module* cannot be imported."""

    if importlib.util.find_spec(module) is None:
        raise SystemExit(install_hint)


require_dependency("qiskit", "qiskit is required for calibration. Install with: pip install qiskit")

from qiskit import QuantumCircuit  # noqa: E402  (import after dependency check)


CLIFFORD_1Q_GATES: Tuple[str, ...] = ("h", "s", "sdg", "x", "z")
CLIFFORD_2Q_GATES: Tuple[str, ...] = ("cx", "cz", "swap")
ROTATION_GATES: Tuple[str, ...] = ("rx", "ry", "rz")
CONTROLLED_ROTATIONS: Tuple[str, ...] = ("crx", "cry", "crz", "cp")
DIAGONAL_GATES: Tuple[str, ...] = ("p", "t", "tdg", "s", "sdg", "z")


@dataclass
class TailSplit:
    """Representation of a circuit split into Clifford prefix and non-Clifford tail."""

    prefix: QuantumCircuit
    tail: QuantumCircuit
    prefix_ops: Sequence[Tuple[object, Sequence[object], Sequence[object]]]
    tail_ops: Sequence[Tuple[object, Sequence[object], Sequence[object]]]


def _iter_instructions(
    data: Iterable[Tuple[object, Sequence[object], Sequence[object]]]
) -> Iterable[Tuple[object, Sequence[object], Sequence[object]]]:
    for inst, qargs, cargs in data:
        yield inst, qargs, cargs


def count_ops(ops: Iterable[Tuple[object, Sequence[object], Sequence[object]]]) -> Tuple[int, int]:
    oneq = 0
    twoq = 0
    for inst, qargs, _ in _iter_instructions(ops):
        if len(qargs) >= 2:
            twoq += 1
        else:
            oneq += 1
    return oneq, twoq


def build_subcircuit_like(parent: QuantumCircuit, ops: Sequence[Tuple[object, Sequence[object], Sequence[object]]]) -> QuantumCircuit:
    sub = QuantumCircuit(parent.num_qubits)
    for inst, qargs, cargs in ops:
        sub.append(inst, qargs, cargs)
    return sub


def split_at_first_nonclifford(qc: QuantumCircuit) -> Optional[TailSplit]:
    prefix_ops: List[Tuple[object, Sequence[object], Sequence[object]]] = []
    tail_ops: List[Tuple[object, Sequence[object], Sequence[object]]] = []
    seen_tail = False
    for entry in qc.data:
        inst, qargs, cargs = entry
        name = gate_name(inst)
        is_cliff = name in {"i", "id"} or name in CLIFFORD_1Q_GATES or name in CLIFFORD_2Q_GATES
        if not seen_tail and is_cliff:
            prefix_ops.append((inst, qargs, cargs))
            continue
        if not seen_tail and not is_cliff:
            seen_tail = True
        tail_ops.append((inst, qargs, cargs))

    if not prefix_ops or not tail_ops:
        return None

    prefix = build_subcircuit_like(qc, prefix_ops)
    tail = build_subcircuit_like(qc, tail_ops)
    return TailSplit(prefix=prefix, tail=tail, prefix_ops=prefix_ops, tail_ops=tail_ops)


def build_clifford_tail(
    *,
    n: int,
    depth_cliff: int,
    tail_layers: int,
    angle_scale: float,
    seed: int,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth_cliff):
        for q in range(n):
            getattr(qc, rng.choice(CLIFFORD_1Q_GATES))(q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            getattr(qc, rng.choice(CLIFFORD_2Q_GATES))(a, b)

    for _ in range(tail_layers):
        for q in range(n):
            theta = float(rng.uniform(-angle_scale, angle_scale))
            qc.rx(theta, q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            qc.cz(a, b)
    return qc


def random_clifford_circuit(*, n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(CLIFFORD_1Q_GATES)
            getattr(qc, gate)(q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            gate = rng.choice(CLIFFORD_2Q_GATES)
            getattr(qc, gate)(a, b)
    return qc


def random_tail_circuit(
    *,
    n: int,
    layers: int,
    angle_scale: float,
    rotation_prob: float,
    twoq_prob: float,
    branch_prob: float,
    diag_prob: float,
    seed: int,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            r = rng.random()
            if r < rotation_prob:
                gate = rng.choice(ROTATION_GATES)
                getattr(qc, gate)(float(rng.uniform(-angle_scale, angle_scale)), q)
            elif r < rotation_prob + branch_prob:
                qc.h(q)
            elif r < rotation_prob + branch_prob + diag_prob:
                gate = rng.choice(DIAGONAL_GATES)
                if gate in {"p", "cp"}:
                    qc.p(float(rng.uniform(-angle_scale, angle_scale)), q)
                else:
                    getattr(qc, gate)(q)
        order = list(range(n))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            if rng.random() >= twoq_prob:
                continue
            gate = rng.choice(CLIFFORD_2Q_GATES + CONTROLLED_ROTATIONS)
            if gate in CONTROLLED_ROTATIONS:
                getattr(qc, gate)(float(rng.uniform(-angle_scale, angle_scale)), a, b)
            else:
                getattr(qc, gate)(a, b)
    return qc


def collect_metrics(circuit: QuantumCircuit) -> dict:
    return circuit_metrics(circuit)
