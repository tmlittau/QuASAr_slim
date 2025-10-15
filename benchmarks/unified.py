"""Unified benchmark circuit generator for QuASAr experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np
from qiskit import QuantumCircuit

CircuitFamily = Literal["disjoint", "hybrid", "mixed"]
TailType = Literal["random", "sparse"]


@dataclass(frozen=True)
class CircuitOptions:
    """Optional parameters for the unified circuit generator."""

    block_size: int = 8
    cutoff: float = 0.8
    tail_type: TailType = "random"


def _partition_blocks(num_qubits: int, block_size: int) -> List[List[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks: List[List[int]] = []
    for start in range(0, num_qubits, block_size):
        stop = min(num_qubits, start + block_size)
        block = list(range(start, stop))
        if block:
            blocks.append(block)
    return blocks


def _prepare_ghz_block(qc: QuantumCircuit, block: Sequence[int]) -> None:
    if not block:
        return
    first = block[0]
    qc.h(first)
    for target in block[1:]:
        qc.cx(first, target)


def _apply_block_clifford_layer(
    qc: QuantumCircuit, block: Sequence[int], rng: np.random.Generator
) -> None:
    if not block:
        return
    single = ("h", "s", "sdg", "x", "z")
    for qubit in block:
        gate = rng.choice(single)
        getattr(qc, gate)(qubit)
    shuffled = list(block)
    rng.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if rng.random() < 0.5:
            qc.cx(a, b)
        else:
            qc.cz(a, b)


def _apply_block_random_tail(
    qc: QuantumCircuit, block: Sequence[int], rng: np.random.Generator, *, angle_scale: float
) -> None:
    if not block:
        return
    for qubit in block:
        theta = float(rng.uniform(-angle_scale, angle_scale))
        phi = float(rng.uniform(-angle_scale, angle_scale))
        lam = float(rng.uniform(-angle_scale, angle_scale))
        qc.u(theta, phi, lam, qubit)
    shuffled = list(block)
    rng.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if rng.random() < 0.5:
            qc.cx(a, b)
        else:
            qc.cz(a, b)


def _apply_block_sparse_tail(
    qc: QuantumCircuit,
    block: Sequence[int],
    rng: np.random.Generator,
    *,
    angle_scale: float,
    sparsity: float,
) -> None:
    if not block:
        return
    size = len(block)
    count = max(1, int(round(sparsity * size))) if sparsity > 0 else 0
    count = min(size, count)
    if count > 0:
        targets = list(rng.choice(block, size=count, replace=False))
        for qubit in targets:
            theta = float(rng.uniform(-angle_scale, angle_scale))
            qc.rz(theta, qubit)
    rng.shuffle(block := list(block))
    for a, b in zip(block[::2], block[1::2]):
        qc.cz(a, b)


def _apply_global_clifford_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    single = ("h", "s", "sdg", "x", "z")
    twoq = ("cx", "cz", "swap")
    for qubit in range(qc.num_qubits):
        getattr(qc, rng.choice(single))(qubit)
    order = list(range(qc.num_qubits))
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        getattr(qc, rng.choice(twoq))(a, b)


def _apply_global_random_tail(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    for qubit in range(qc.num_qubits):
        theta = float(rng.uniform(-0.3, 0.3))
        qc.rx(theta, qubit)
    order = list(range(qc.num_qubits))
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        qc.cz(a, b)


def _apply_global_sparse_tail(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    num_qubits = qc.num_qubits
    if num_qubits == 0:
        return
    count = max(1, num_qubits // 4)
    idxs = rng.choice(num_qubits, size=count, replace=False)
    for qubit in idxs:
        theta = float(rng.uniform(-0.2, 0.2))
        qc.rz(theta, int(qubit))
    if num_qubits < 2:
        return
    for _ in range(count // 2 or 1):
        a = int(rng.integers(0, num_qubits - 1))
        b = min(num_qubits - 1, a + int(rng.integers(1, 3)))
        if a != b:
            qc.cz(a, b)


def _validate_inputs(num_qubits: int, depth: int) -> None:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if depth < 0:
        raise ValueError("depth must be non-negative")


def generate_benchmark_circuit(
    *,
    num_qubits: int,
    depth: int,
    family: CircuitFamily,
    block_size: int | None = None,
    cutoff: float | None = None,
    tail_type: TailType = "random",
    seed: int | None = None,
) -> QuantumCircuit:
    """Generate a parameterised benchmark circuit for QuASAr."""

    _validate_inputs(num_qubits, depth)
    rng = np.random.default_rng(seed)
    family_normalized = family.lower()
    if family_normalized not in {"disjoint", "hybrid", "mixed"}:
        raise ValueError(f"Unknown circuit family '{family}'")

    opts = CircuitOptions(
        block_size=block_size or 8,
        cutoff=cutoff if cutoff is not None else 0.8,
        tail_type=tail_type,
    )
    cutoff_clamped = max(0.0, min(1.0, float(opts.cutoff)))
    tail_normalized = tail_type.lower()
    if tail_normalized not in {"random", "sparse"}:
        raise ValueError(f"Unsupported tail type '{tail_type}'")

    qc = QuantumCircuit(num_qubits)

    if family_normalized == "disjoint":
        blocks = _partition_blocks(num_qubits, opts.block_size)
        for block in blocks:
            _prepare_ghz_block(qc, block)
        for _ in range(depth):
            for block in blocks:
                _apply_block_clifford_layer(qc, block, rng)
            for block in blocks:
                if tail_normalized == "random":
                    _apply_block_random_tail(qc, block, rng, angle_scale=0.3)
                else:
                    _apply_block_sparse_tail(qc, block, rng, angle_scale=0.2, sparsity=0.25)
        return qc

    if family_normalized == "hybrid":
        clifford_depth = int(round(depth * cutoff_clamped))
        tail_depth = max(0, depth - clifford_depth)
        for _ in range(clifford_depth):
            _apply_global_clifford_layer(qc, rng)
        for _ in range(tail_depth):
            if tail_normalized == "random":
                _apply_global_random_tail(qc, rng)
            else:
                _apply_global_sparse_tail(qc, rng)
        return qc

    blocks = _partition_blocks(num_qubits, opts.block_size)
    for block in blocks:
        _prepare_ghz_block(qc, block)

    clifford_depth = int(round(depth * cutoff_clamped))
    tail_depth = max(0, depth - clifford_depth)

    for _ in range(clifford_depth):
        for block in blocks:
            _apply_block_clifford_layer(qc, block, rng)

    for _ in range(tail_depth):
        if tail_normalized == "random":
            _apply_global_random_tail(qc, rng)
        else:
            _apply_global_sparse_tail(qc, rng)

    return qc


__all__ = ["generate_benchmark_circuit", "CircuitFamily", "TailType", "CircuitOptions"]
