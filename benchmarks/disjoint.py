"""Disjoint block benchmark circuit builders."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["disjoint_preps_plus_tails", "CIRCUIT_REGISTRY", "build"]


def _apply_ry_via_rz_h(qc: QuantumCircuit, qubit: int, theta: float) -> None:
    """Apply an ``RY(theta)`` rotation using only ``H`` and ``RZ`` gates."""

    qc.rz(-math.pi / 2, qubit)
    qc.h(qubit)
    qc.rz(theta, qubit)
    qc.h(qubit)
    qc.rz(math.pi / 2, qubit)


def _prepare_ghz_block(qc: QuantumCircuit, block: list[int]) -> None:
    if not block:
        return
    first = block[0]
    qc.h(first)
    for target in block[1:]:
        qc.cx(first, target)


def _prepare_w_block(qc: QuantumCircuit, block: list[int]) -> None:
    size = len(block)
    if size == 0:
        return
    if size == 1:
        qc.x(block[0])
        return
    for index in range(size - 1):
        remaining = size - index
        theta = 2.0 * math.acos(math.sqrt((remaining - 1) / remaining))
        _apply_ry_via_rz_h(qc, block[index], theta)
        qc.cx(block[index], block[index + 1])
    qc.x(block[-1])


def _clifford_tail_layer(
    qc: QuantumCircuit, block: list[int], rng: np.random.Generator
) -> None:
    if not block:
        return
    one_qubit = ("h", "s", "x", "z")
    for qubit in block:
        gate = rng.choice(one_qubit)
        getattr(qc, gate)(qubit)
    shuffled = block.copy()
    rng.shuffle(shuffled)
    for a, b in zip(shuffled[::2], shuffled[1::2]):
        if rng.random() < 0.5:
            qc.cx(a, b)
        else:
            qc.cx(b, a)


def _diag_tail_layer(
    qc: QuantumCircuit,
    block: list[int],
    rng: np.random.Generator,
    *,
    angle_scale: float,
    sparsity: float,
    bandwidth: int,
) -> None:
    size = len(block)
    if size == 0:
        return
    if sparsity <= 0:
        num_rz = 0
    else:
        num_rz = math.ceil(sparsity * size)
        num_rz = max(1, min(size, num_rz))
    if num_rz > 0:
        targets = list(rng.choice(block, size=num_rz, replace=False))
        for qubit in targets:
            theta = float(rng.uniform(-angle_scale, angle_scale))
            qc.rz(theta, qubit)
    num_pairs = num_rz // 2
    if num_pairs <= 0 or bandwidth <= 0:
        return
    edges: list[tuple[int, int]] = []
    for index, control in enumerate(block):
        for offset in range(1, bandwidth + 1):
            partner_index = index + offset
            if partner_index < size:
                edges.append((control, block[partner_index]))
    if not edges:
        return
    num_pairs = min(num_pairs, len(edges))
    chosen = rng.choice(len(edges), size=num_pairs, replace=False)
    for edge_index in np.atleast_1d(chosen):
        a, b = edges[int(edge_index)]
        qc.cz(a, b)


def disjoint_preps_plus_tails(
    *,
    num_qubits: int,
    num_blocks: int,
    block_prep: str = "mixed",
    tail_kind: str = "mixed",
    tail_depth: int = 20,
    angle_scale: float = 0.1,
    sparsity: float = 0.05,
    bandwidth: int = 2,
    seed: int = 7,
) -> QuantumCircuit:
    """Build a block-disjoint circuit with configurable preparations and tails."""

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > num_qubits:
        raise ValueError("num_blocks cannot exceed num_qubits")

    block_prep = block_prep.lower()
    if block_prep not in {"ghz", "w", "mixed"}:
        raise ValueError(f"Unsupported block_prep '{block_prep}'")

    tail_kind = tail_kind.lower()
    if tail_kind not in {"clifford", "diag", "none", "mixed"}:
        raise ValueError(f"Unsupported tail_kind '{tail_kind}'")

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    base = num_qubits // num_blocks
    remainder = num_qubits % num_blocks
    blocks: list[list[int]] = []
    start = 0
    for index in range(num_blocks):
        size = base + (1 if index < remainder else 0)
        block = list(range(start, start + size))
        if not block:
            raise ValueError("Encountered empty block; check num_qubits/num_blocks")
        blocks.append(block)
        start += size

    prep_choices = []
    for index in range(num_blocks):
        if block_prep == "mixed":
            prep_choices.append("ghz" if index % 2 == 0 else "w")
        else:
            prep_choices.append(block_prep)

    tail_choices = []
    for index in range(num_blocks):
        if tail_kind == "mixed":
            tail_choices.append("clifford" if index % 2 == 0 else "diag")
        else:
            tail_choices.append(tail_kind)

    for index, block in enumerate(blocks):
        if prep_choices[index] == "ghz":
            _prepare_ghz_block(qc, block)
        else:
            _prepare_w_block(qc, block)

        tail = tail_choices[index]
        if tail == "none" or tail_depth <= 0:
            continue
        for _ in range(tail_depth):
            if tail == "clifford":
                _clifford_tail_layer(qc, block, rng)
            elif tail == "diag":
                _diag_tail_layer(
                    qc,
                    block,
                    rng,
                    angle_scale=angle_scale,
                    sparsity=sparsity,
                    bandwidth=bandwidth,
                )
            else:
                raise ValueError(
                    f"Unexpected tail kind '{tail}' after normalization"
                )

    return qc


CIRCUIT_REGISTRY: Dict[str, Any] = {
    "disjoint_preps_plus_tails": disjoint_preps_plus_tails,
}


def build(kind: str, /, **kwargs: Any) -> QuantumCircuit:
    try:
        builder = CIRCUIT_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown disjoint circuit kind: {kind}") from exc
    return builder(**kwargs)
