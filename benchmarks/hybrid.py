"""Hybrid and general-purpose benchmark circuit generators."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from qiskit import QuantumCircuit

__all__ = [
    "ghz_clusters_random",
    "random_clifford",
    "stitched_disjoint_rand_bandedqft_rand",
    "stitched_disjoint_diag_bandedqft_diag",
    "clifford_plus_random_rotations",
    "clifford_prefix_rot_tail",
    "CIRCUIT_REGISTRY",
    "build",
]


def ghz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    if num_qubits <= 0:
        return qc
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(i - 1, i)
    return qc


def ghz_clusters_random(
    num_qubits: int,
    block_size: int = 8,
    depth: int = 200,
    seed: int = 1,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        sub = ghz(end - start)
        qc.compose(sub, qubits=list(range(start, end)), inplace=True)
    oneq = ["h", "rx", "ry", "rz", "s", "sdg", "x", "z"]
    twoq = ["cx", "cz", "swap", "rzz"]
    for _ in range(depth):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            for qubit in range(start, end):
                gate = rng.choice(oneq)
                if gate in {"rx", "ry", "rz"}:
                    getattr(qc, gate)(float(rng.uniform(0, 2 * np.pi)), qubit)
                else:
                    getattr(qc, gate)(qubit)
            block = list(range(start, end))
            rng.shuffle(block)
            for a, b in zip(block[::2], block[1::2]):
                gate = rng.choice(twoq)
                if gate == "rzz":
                    qc.rzz(float(rng.uniform(0, 2 * np.pi)), a, b)
                else:
                    getattr(qc, gate)(a, b)
    return qc


def random_clifford(num_qubits: int, depth: int = 200, seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    cliff1 = ["h", "s", "sdg", "x", "z"]
    cliff2 = ["cx", "cz", "swap"]
    for _ in range(depth):
        for qubit in range(num_qubits):
            getattr(qc, rng.choice(cliff1))(qubit)
        order = list(range(num_qubits))
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            getattr(qc, rng.choice(cliff2))(a, b)
    return qc


def _banded_qft_block(
    qc: QuantumCircuit, qubits: list[int], bandwidth: int = 3
) -> None:
    size = len(qubits)
    for i in range(size):
        qc.h(qubits[i])
        for j in range(1, bandwidth + 1):
            if i + j < size:
                angle = np.pi / (2**j)
                qc.cp(angle, qubits[i + j], qubits[i])


def _random_rotation_layer_block(
    qc: QuantumCircuit, qubits: list[int], rng: np.random.Generator, angle_scale: float = 2 * np.pi
) -> None:
    for qubit in qubits:
        qc.rx(float(rng.uniform(0, angle_scale)), qubit)
        qc.ry(float(rng.uniform(0, angle_scale)), qubit)
        qc.rz(float(rng.uniform(0, angle_scale)), qubit)


def _diagonal_layer_block(qc: QuantumCircuit, qubits: list[int], rng: np.random.Generator) -> None:
    for qubit in qubits:
        qc.rz(float(rng.uniform(0, 2 * np.pi)), qubit)
    order = qubits.copy()
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        qc.cz(a, b)


def _neighbor_bridge_layer(
    qc: QuantumCircuit,
    start: int,
    end: int,
    next_start: int | None,
    rng: np.random.Generator,
) -> None:
    if next_start is None:
        return
    a = end - 1
    b = next_start
    if rng.random() < 0.5:
        qc.cz(a, b)
    else:
        qc.cx(a, b)


def stitched_disjoint_rand_bandedqft_rand(
    *,
    num_qubits: int,
    block_size: int = 8,
    depth_pre: int = 100,
    depth_post: int = 100,
    qft_bandwidth: int = 3,
    neighbor_bridge_layers: int = 0,
    seed: int = 1,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        qc.compose(ghz(end - start), qubits=list(range(start, end)), inplace=True)
    for _ in range(depth_pre):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            _random_rotation_layer_block(qc, list(range(start, end)), rng)
    for _ in range(neighbor_bridge_layers):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            next_start = start + block_size if end < num_qubits else None
            _neighbor_bridge_layer(qc, start, end, next_start, rng)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        _banded_qft_block(qc, list(range(start, end)), bandwidth=qft_bandwidth)
    for _ in range(depth_post):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            _random_rotation_layer_block(qc, list(range(start, end)), rng)
    return qc


def stitched_disjoint_diag_bandedqft_diag(
    *,
    num_qubits: int,
    block_size: int = 8,
    depth_pre: int = 100,
    depth_post: int = 100,
    qft_bandwidth: int = 3,
    neighbor_bridge_layers: int = 0,
    seed: int = 2,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        qc.compose(ghz(end - start), qubits=list(range(start, end)), inplace=True)
    for _ in range(depth_pre):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            _diagonal_layer_block(qc, list(range(start, end)), rng)
    for _ in range(neighbor_bridge_layers):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            next_start = start + block_size if end < num_qubits else None
            _neighbor_bridge_layer(qc, start, end, next_start, rng)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        _banded_qft_block(qc, list(range(start, end)), bandwidth=qft_bandwidth)
    for _ in range(depth_post):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            _diagonal_layer_block(qc, list(range(start, end)), rng)
    return qc


CLIFFORD_ONEQ = ("h", "s", "sdg", "x", "z")
CLIFFORD_TWOQ = ("cx", "cz", "swap")
ROT_GATES = ("rx", "ry", "rz")


def _apply_random_clifford_layer(qc: QuantumCircuit, rng: np.random.Generator) -> None:
    num_qubits = qc.num_qubits
    for qubit in range(num_qubits):
        getattr(qc, rng.choice(CLIFFORD_ONEQ))(qubit)
    order = list(range(num_qubits))
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        getattr(qc, rng.choice(CLIFFORD_TWOQ))(a, b)


def _apply_rotation_tail_layer(
    qc: QuantumCircuit, rng: np.random.Generator, angle_scale: float
) -> None:
    num_qubits = qc.num_qubits
    for qubit in range(num_qubits):
        theta = float(rng.uniform(-angle_scale, angle_scale))
        qc.rx(theta, qubit)
    order = list(range(num_qubits))
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        qc.cz(a, b)


def clifford_plus_random_rotations(
    *,
    num_qubits: int,
    depth: int = 200,
    rot_prob: float = 0.2,
    angle_scale: float = 0.1,
    seed: int = 3,
    pair_scope: str = "global",
    block_size: int = 8,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            getattr(qc, rng.choice(CLIFFORD_ONEQ))(qubit)
            if rng.random() < rot_prob:
                theta = float(rng.normal(0.0, angle_scale))
                getattr(qc, rng.choice(ROT_GATES))(theta, qubit)
        if pair_scope == "block":
            for start in range(0, num_qubits, block_size):
                end = min(num_qubits, start + block_size)
                order = list(range(start, end))
                rng.shuffle(order)
                for a, b in zip(order[::2], order[1::2]):
                    getattr(qc, rng.choice(CLIFFORD_TWOQ))(a, b)
        else:
            order = list(range(num_qubits))
            rng.shuffle(order)
            for a, b in zip(order[::2], order[1::2]):
                getattr(qc, rng.choice(CLIFFORD_TWOQ))(a, b)
    return qc


def clifford_prefix_rot_tail(
    *,
    num_qubits: int,
    depth: int,
    cutoff: float,
    angle_scale: float = 0.1,
    seed: int = 1,
) -> QuantumCircuit:
    cutoff = max(0.0, min(1.0, float(cutoff)))
    depth = int(depth)
    rng = np.random.default_rng(int(seed))
    qc = QuantumCircuit(int(num_qubits))
    d_pre = int(round(depth * cutoff))
    d_tail = max(0, depth - d_pre)
    for _ in range(d_pre):
        _apply_random_clifford_layer(qc, rng)
    for _ in range(d_tail):
        _apply_rotation_tail_layer(qc, rng, angle_scale)
    return qc


CIRCUIT_REGISTRY: Dict[str, Any] = {
    "ghz_clusters_random": ghz_clusters_random,
    "random_clifford": random_clifford,
    "stitched_rand_bandedqft_rand": stitched_disjoint_rand_bandedqft_rand,
    "stitched_diag_bandedqft_diag": stitched_disjoint_diag_bandedqft_diag,
    "clifford_plus_rot": clifford_plus_random_rotations,
    "clifford_prefix_rot_tail": clifford_prefix_rot_tail,
}


def build(kind: str, /, **kwargs: Any) -> QuantumCircuit:
    try:
        builder = CIRCUIT_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown hybrid circuit kind: {kind}") from exc
    return builder(**kwargs)
