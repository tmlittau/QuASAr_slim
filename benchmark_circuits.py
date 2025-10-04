
from __future__ import annotations
from typing import List, Dict, Any, Optional
from qiskit import QuantumCircuit
import numpy as np
import math

def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    if n <= 0:
        return qc
    qc.h(0)
    for i in range(1, n):
        qc.cx(i-1, i)
    return qc

def ghz_clusters_random(num_qubits: int, block_size: int = 8, depth: int = 200, seed: int = 1) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for start in range(0, num_qubits, block_size):
        end = min(num_qubits, start + block_size)
        sub = ghz(end - start)
        qc.compose(sub, qubits=list(range(start, end)), inplace=True)
    oneq = ["h","rx","ry","rz","s","sdg","x","z"]
    twoq = ["cx","cz","swap","rzz"]
    for _ in range(depth):
        for start in range(0, num_qubits, block_size):
            end = min(num_qubits, start + block_size)
            for q in range(start, end):
                g = rng.choice(oneq)
                if g in {"rx","ry","rz"}:
                    getattr(qc, g)(float(rng.uniform(0, 2*np.pi)), q)
                else:
                    getattr(qc, g)(q)
            block = list(range(start, end))
            rng.shuffle(block)
            for a, b in zip(block[::2], block[1::2]):
                g = rng.choice(twoq)
                if g == "rzz":
                    qc.rzz(float(rng.uniform(0, 2*np.pi)), a, b)
                else:
                    getattr(qc, g)(a, b)
    return qc

def random_clifford(num_qubits: int, depth: int = 200, seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    cliff1 = ["h","s","sdg","x","z"]
    cliff2 = ["cx","cz","swap"]
    for _ in range(depth):
        for q in range(num_qubits):
            getattr(qc, rng.choice(cliff1))(q)
        order = list(range(num_qubits))
        rng.shuffle(order)
        for a,b in zip(order[::2], order[1::2]):
            getattr(qc, rng.choice(cliff2))(a,b)
    return qc

def _banded_qft_block(qc: QuantumCircuit, qubits: List[int], bandwidth: int = 3):
    n = len(qubits)
    for i in range(n):
        qc.h(qubits[i])
        for j in range(1, bandwidth+1):
            if i + j < n:
                angle = np.pi / (2**j)
                qc.cp(angle, qubits[i+j], qubits[i])

def _random_rotation_layer_block(qc: QuantumCircuit, qubits: List[int], rng, angle_scale: float = 2*np.pi):
    for q in qubits:
        qc.rx(float(rng.uniform(0, angle_scale)), q)
        qc.ry(float(rng.uniform(0, angle_scale)), q)
        qc.rz(float(rng.uniform(0, angle_scale)), q)

def _diagonal_layer_block(qc: QuantumCircuit, qubits: List[int], rng):
    for q in qubits:
        qc.rz(float(rng.uniform(0, 2*np.pi)), q)
    order = qubits.copy()
    rng.shuffle(order)
    for a, b in zip(order[::2], order[1::2]):
        qc.cz(a, b)

def _neighbor_bridge_layer(qc: QuantumCircuit, start: int, end: int, next_start: Optional[int], rng):
    if next_start is None:
        return
    a = end - 1
    b = next_start
    if rng.random() < 0.5:
        qc.cz(a, b)
    else:
        qc.cx(a, b)

def stitched_disjoint_rand_bandedqft_rand(*, num_qubits: int, block_size: int = 8,
                                          depth_pre: int = 100, depth_post: int = 100,
                                          qft_bandwidth: int = 3, neighbor_bridge_layers: int = 0,
                                          seed: int = 1) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for s in range(0, num_qubits, block_size):
        e = min(num_qubits, s+block_size)
        qc.compose(ghz(e - s), qubits=list(range(s, e)), inplace=True)
    for _ in range(depth_pre):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            _random_rotation_layer_block(qc, list(range(s, e)), rng)
    for _ in range(neighbor_bridge_layers):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            ns = s + block_size if e < num_qubits else None
            _neighbor_bridge_layer(qc, s, e, ns, rng)
    for s in range(0, num_qubits, block_size):
        e = min(num_qubits, s+block_size)
        _banded_qft_block(qc, list(range(s, e)), bandwidth=qft_bandwidth)
    for _ in range(depth_post):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            _random_rotation_layer_block(qc, list(range(s, e)), rng)
    return qc

def stitched_disjoint_diag_bandedqft_diag(*, num_qubits: int, block_size: int = 8,
                                          depth_pre: int = 100, depth_post: int = 100,
                                          qft_bandwidth: int = 3, neighbor_bridge_layers: int = 0,
                                          seed: int = 2) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for s in range(0, num_qubits, block_size):
        e = min(num_qubits, s+block_size)
        qc.compose(ghz(e - s), qubits=list(range(s, e)), inplace=True)
    for _ in range(depth_pre):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            _diagonal_layer_block(qc, list(range(s, e)), rng)
    for _ in range(neighbor_bridge_layers):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            ns = s + block_size if e < num_qubits else None
            _neighbor_bridge_layer(qc, s, e, ns, rng)
    for s in range(0, num_qubits, block_size):
        e = min(num_qubits, s+block_size)
        _banded_qft_block(qc, list(range(s, e)), bandwidth=qft_bandwidth)
    for _ in range(depth_post):
        for s in range(0, num_qubits, block_size):
            e = min(num_qubits, s+block_size)
            _diagonal_layer_block(qc, list(range(s, e)), rng)
    return qc

def clifford_plus_random_rotations(*, num_qubits: int, depth: int = 200,
                                   rot_prob: float = 0.2, angle_scale: float = 0.1,
                                   seed: int = 3,
                                   pair_scope: str = "global",   # "global" or "block"
                                   block_size: int = 8) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    cliff1 = ["h","s","sdg","x","z"]
    cliff2 = ["cx","cz","swap"]
    for _ in range(depth):
        for q in range(num_qubits):
            getattr(qc, rng.choice(cliff1))(q)
            if rng.random() < rot_prob:
                theta = float(rng.normal(0.0, angle_scale))
                getattr(qc, rng.choice(["rx","ry","rz"]))(theta, q)
        if pair_scope == "block":
            for start in range(0, num_qubits, block_size):
                end = min(num_qubits, start + block_size)
                order = list(range(start, end))
                rng.shuffle(order)
                for a, b in zip(order[::2], order[1::2]):
                    getattr(qc, rng.choice(cliff2))(a, b)
        else:
            order = list(range(num_qubits))
            rng.shuffle(order)
            for a, b in zip(order[::2], order[1::2]):
                getattr(qc, rng.choice(cliff2))(a, b)
    return qc

# --- add helper to pair qubits locally (block scope) or globally ---
def _pairing(order):
    for a,b in zip(order[::2], order[1::2]):
        yield a,b

def _apply_random_clifford_layer(qc: QuantumCircuit, rng: np.random.Generator):
    oneq = ["h","s","sdg","x","z"]
    twoq = ["cx","cz","swap"]
    n = qc.num_qubits
    for q in range(n):
        getattr(qc, rng.choice(oneq))(q)
    order = list(range(n))
    rng.shuffle(order)
    for a,b in _pairing(order):
        getattr(qc, rng.choice(twoq))(a,b)

def _apply_rotation_tail_layer(qc: QuantumCircuit, rng: np.random.Generator, angle_scale: float):
    n = qc.num_qubits
    for q in range(n):
        theta = float(rng.uniform(-angle_scale, angle_scale))
        qc.rx(theta, q)
    order = list(range(n))
    rng.shuffle(order)
    for a,b in _pairing(order):
        qc.cz(a,b)


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
    for i in range(size - 1):
        remaining = size - i
        theta = 2.0 * math.acos(math.sqrt((remaining - 1) / remaining))
        _apply_ry_via_rz_h(qc, block[i], theta)
        qc.cx(block[i], block[i + 1])
    qc.x(block[-1])


def _clifford_tail_layer(qc: QuantumCircuit, block: list[int], rng: np.random.Generator) -> None:
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
    num_rz = int(round(sparsity * size))
    num_rz = max(0, min(size, num_rz))
    if num_rz > 0:
        targets = list(rng.choice(block, size=num_rz, replace=False))
        for qubit in targets:
            theta = float(rng.uniform(-angle_scale, angle_scale))
            qc.rz(theta, qubit)
    num_pairs = num_rz // 2
    if num_pairs <= 0 or bandwidth <= 0:
        return
    edges: list[tuple[int, int]] = []
    for idx, control in enumerate(block):
        for offset in range(1, bandwidth + 1):
            partner_index = idx + offset
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
) -> "QuantumCircuit":
    """Build a block-disjoint circuit with configurable preparations and tails.

    Args:
        num_qubits: Total number of qubits in the circuit.
        num_blocks: Number of contiguous, non-overlapping blocks to partition the
            qubits into. Each block has size either ``floor(n/k)`` or ``ceil(n/k)``
            and contains only consecutive qubits.
        block_prep: Preparation per block. ``"ghz"`` prepares every block as a GHZ
            state using only intra-block CX gates, ``"w"`` prepares every block as a
            W state, and ``"mixed"`` alternates between GHZ (even-indexed blocks)
            and W (odd-indexed blocks).
        tail_kind: Tail layer family applied after preparation. ``"clifford"``
            applies random single-qubit Clifford gates and sparse CX gates within
            each block, ``"diag"`` applies sparse RZ rotations and banded CZ gates
            per block, ``"none"`` adds no tail, and ``"mixed"`` alternates between
            Clifford and diagonal tails (even-indexed blocks receive Clifford).
        tail_depth: Number of tail layers applied to each block (ignored when
            ``tail_kind`` is ``"none"``).
        angle_scale: Maximum rotation magnitude for diagonal tails; RZ angles are
            sampled uniformly from ``[-angle_scale, angle_scale]``.
        sparsity: Fraction of qubits per block that receive an RZ rotation in each
            diagonal-tail layer. The number of CZ gates is approximately half the
            number of rotated qubits.
        bandwidth: Maximum separation (in indices within a block) for CZ gates in
            diagonal tails.
        seed: Seed for the pseudo-random number generator controlling gate
            selection.

    Returns:
        A ``QuantumCircuit`` whose multi-qubit interactions are restricted to be
        entirely within each contiguous block. No gate spans two different blocks.
    """

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
    for idx in range(num_blocks):
        size = base + (1 if idx < remainder else 0)
        block = list(range(start, start + size))
        if not block:
            raise ValueError("Encountered empty block; check num_qubits/num_blocks")
        blocks.append(block)
        start += size

    prep_choices = []
    for idx in range(num_blocks):
        if block_prep == "mixed":
            prep_choices.append("ghz" if idx % 2 == 0 else "w")
        else:
            prep_choices.append(block_prep)

    tail_choices = []
    for idx in range(num_blocks):
        if tail_kind == "mixed":
            tail_choices.append("clifford" if idx % 2 == 0 else "diag")
        else:
            tail_choices.append(tail_kind)

    for idx, block in enumerate(blocks):
        if prep_choices[idx] == "ghz":
            _prepare_ghz_block(qc, block)
        else:
            _prepare_w_block(qc, block)

        tail = tail_choices[idx]
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
                raise ValueError(f"Unexpected tail kind '{tail}' after normalization")

    return qc

# --- NEW: builder with a cutoff fraction ---
def clifford_prefix_rot_tail(*, num_qubits: int, depth: int, cutoff: float,
                             angle_scale: float = 0.1, seed: int = 1) -> QuantumCircuit:
    """Build a circuit with a Clifford-only prefix of floor(cutoff*depth) layers
    and a random-rotation tail of depth - prefix_layers layers."""
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

CIRCUIT_REGISTRY = {
    "ghz_clusters_random": ghz_clusters_random,
    "random_clifford": random_clifford,
    "stitched_rand_bandedqft_rand": stitched_disjoint_rand_bandedqft_rand,
    "stitched_diag_bandedqft_diag": stitched_disjoint_diag_bandedqft_diag,
    "clifford_plus_rot": clifford_plus_random_rotations,
    "clifford_prefix_rot_tail": clifford_prefix_rot_tail,
    "disjoint_preps_plus_tails": disjoint_preps_plus_tails,
}

def build(kind: str, **kwargs) -> QuantumCircuit:
    if kind not in CIRCUIT_REGISTRY:
        raise ValueError(f"Unknown circuit kind: {kind}")
    return CIRCUIT_REGISTRY[kind](**kwargs)
