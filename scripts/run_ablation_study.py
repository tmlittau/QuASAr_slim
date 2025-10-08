#!/usr/bin/env python3
"""Streamlined ablation study helper for QuASAr's partitioning strategies.

The new workflow focuses on a single configurable circuit that contains several
independent hybrid sub-circuits.  Each sub-circuit is constructed by applying a
Clifford-only prefix followed by either a dense random-rotation tail or a sparse
phase tail.  This mix stresses both the disjoint partitioning logic and the
hybrid prefix/tail splitter in a single problem instance, enabling lightweight
experiments before scaling to deeper benchmarks.

The module exposes two primary helpers:

``build_ablation_circuit``
    Construct the disjoint hybrid circuit together with metadata describing each
    sub-circuit.

``run_three_way_ablation``
    Plan (and optionally execute) the circuit under three planner variants:
    the full QuASAr configuration, a variant without disjoint partitioning, and
    a variant without the hybrid prefix/tail splitting.  The function returns a
    JSON-serialisable payload that can be written to disk or further processed
    by notebooks.

Example::

    from scripts import run_ablation_study as ras

    circuit, blocks = ras.build_ablation_circuit(num_components=3)
    summary = ras.run_three_way_ablation(circuit)
    print(summary["variants"][0]["partitions"])  # planned partitions

The ``main`` entry point offers a thin CLI around these helpers and stores the
results in JSON form for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit

from quasar.SSD import PartitionNode, SSD
from quasar.backends.sv import estimate_sv_bytes
from quasar.analyzer import analyze
from quasar.cost_estimator import CostEstimator
from quasar.gate_metrics import circuit_metrics
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd


@dataclass(frozen=True)
class HybridBlockSpec:
    """Describe one disjoint hybrid block inside the ablation circuit."""

    index: int
    qubits: Tuple[int, ...]
    tail_kind: str  # either "random" (dense rotations) or "sparse"


def _partition_qubits(num_qubits: int, num_components: int) -> List[Tuple[int, ...]]:
    if num_components <= 0:
        raise ValueError("num_components must be positive")
    if num_qubits < num_components:
        raise ValueError("num_qubits must be at least num_components")
    base = num_qubits // num_components
    remainder = num_qubits % num_components
    blocks: List[Tuple[int, ...]] = []
    start = 0
    for idx in range(num_components):
        size = base + (1 if idx < remainder else 0)
        block = tuple(range(start, start + size))
        blocks.append(block)
        start += size
    return blocks


def _apply_clifford_prefix(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    *,
    depth: int,
    rng: np.random.Generator,
) -> None:
    qubits = tuple(qubits)
    for _ in range(depth):
        for qubit in qubits:
            gate = rng.choice(["h", "s", "sdg", "x", "z"])
            getattr(qc, gate)(qubit)
        for control, target in zip(qubits[:-1], qubits[1:]):
            if rng.random() < 0.7:
                qc.cx(control, target)


def _apply_random_rotation_tail(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    *,
    depth: int,
    rng: np.random.Generator,
) -> None:
    qubits = tuple(qubits)
    for _ in range(depth):
        for qubit in qubits:
            theta, phi = rng.uniform(0.0, 2.0 * np.pi, size=2)
            qc.ry(theta, qubit)
            qc.rz(phi, qubit)
        for control, target in zip(qubits[:-1], qubits[1:]):
            if rng.random() < 0.6:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                qc.crx(angle, control, target)


def _apply_sparse_tail(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    *,
    depth: int,
    rng: np.random.Generator,
) -> None:
    qubits = tuple(qubits)
    for _ in range(depth):
        for qubit in qubits:
            angle = rng.uniform(-np.pi, np.pi)
            qc.rz(angle, qubit)
        for control, target in zip(qubits[:-1], qubits[1:]):
            if rng.random() < 0.3:
                angle = rng.uniform(-np.pi, np.pi)
                qc.cp(angle, control, target)


TAIL_SEQUENCE = ("random", "sparse")


def build_ablation_circuit(
    *,
    num_components: int = 4,
    component_size: int = 4,
    clifford_depth: int = 3,
    tail_depth: int = 3,
    tail_sequence: Optional[Sequence[str]] = None,
    seed: int = 1,
) -> Tuple[QuantumCircuit, List[HybridBlockSpec]]:
    """Create a disjoint circuit with hybrid blocks ready for the ablation study."""

    if component_size <= 0:
        raise ValueError("component_size must be positive")
    total_qubits = num_components * component_size
    qc = QuantumCircuit(total_qubits)
    rng = np.random.default_rng(seed)
    blocks = _partition_qubits(total_qubits, num_components)

    specs: List[HybridBlockSpec] = []
    sequence = tail_sequence or TAIL_SEQUENCE
    for index, block in enumerate(blocks):
        tail_kind = sequence[index % len(sequence)]
        _apply_clifford_prefix(qc, block, depth=clifford_depth, rng=rng)
        if tail_kind == "random":
            _apply_random_rotation_tail(qc, block, depth=tail_depth, rng=rng)
        elif tail_kind == "sparse":
            _apply_sparse_tail(qc, block, depth=tail_depth, rng=rng)
        else:
            raise ValueError(f"Unsupported tail kind '{tail_kind}'")
        specs.append(HybridBlockSpec(index=index, qubits=block, tail_kind=tail_kind))

    qc.metadata = {"hybrid_blocks": [asdict(spec) for spec in specs]}
    return qc, specs


def _collapse_to_single_partition(ssd: SSD, circuit: QuantumCircuit) -> SSD:
    merged = SSD(meta=dict(ssd.meta))
    merged.meta["components"] = 1
    stitched = circuit.copy()
    metrics = circuit_metrics(stitched)
    merged.add(
        PartitionNode(
            id=0,
            qubits=list(range(circuit.num_qubits)),
            circuit=stitched,
            metrics=metrics,
            meta={"collapsed": True},
        )
    )
    return merged


@dataclass
class VariantRecord:
    name: str
    planner: PlannerConfig
    partitions: List[Dict[str, object]]
    execution: Optional[Dict[str, object]] = None
    summary: Optional[Dict[str, object]] = None

    def to_json(self) -> Dict[str, object]:
        payload = {
            "name": self.name,
            "planner": {
                "max_ram_gb": self.planner.max_ram_gb,
                "prefer_dd": self.planner.prefer_dd,
                "hybrid_clifford_tail": self.planner.hybrid_clifford_tail,
                "conv_amp_ops_factor": self.planner.conv_amp_ops_factor,
                "sv_twoq_factor": self.planner.sv_twoq_factor,
            },
            "partitions": self.partitions,
        }
        if self.execution is not None:
            payload["execution"] = self.execution
        if self.summary is not None:
            payload["summary"] = self.summary
        return payload


def _partition_payload(ssd: SSD) -> List[Dict[str, object]]:
    return [partition.to_dict() for partition in ssd.partitions]


def _amp_ops_for_node(node: PartitionNode, estimator: CostEstimator) -> float:
    metrics = node.metrics or {}
    try:
        n = int(metrics.get("num_qubits", 0) or 0)
    except Exception:
        n = 0
    try:
        total = int(metrics.get("num_gates", 0) or 0)
    except Exception:
        total = 0
    try:
        twoq = int(metrics.get("two_qubit_gates", 0) or 0)
    except Exception:
        twoq = 0
    oneq = max(total - twoq, 0)
    if n <= 0 or total <= 0:
        return 0.0
    return float(estimator.sv_cost(n=n, oneq=oneq, twoq=twoq))


def _accumulate_sv_stats(
    ssd: SSD, execution: Optional[Dict[str, object]], estimator: CostEstimator
) -> Tuple[float, float]:
    if not execution or not isinstance(execution, dict):
        return 0.0, 0.0
    results = execution.get("results")
    if not isinstance(results, Iterable):
        return 0.0, 0.0
    lookup = {node.id: node for node in ssd.partitions}
    total_ops = 0.0
    total_time = 0.0
    for entry in results:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).lower()
        if status != "ok":
            continue
        backend = str(entry.get("backend", "sv")).lower()
        if backend != "sv":
            continue
        pid = entry.get("partition")
        try:
            pid_int = int(pid)
        except Exception:
            continue
        node = lookup.get(pid_int)
        if node is None:
            continue
        ops = _amp_ops_for_node(node, estimator)
        if ops <= 0:
            continue
        elapsed = float(entry.get("elapsed_s", 0.0))
        if elapsed <= 0:
            continue
        total_ops += ops
        total_time += elapsed
    return total_ops, total_time


def _variant_needs_estimate(execution: Optional[Dict[str, object]]) -> bool:
    if not execution or not isinstance(execution, dict):
        return False
    results = execution.get("results")
    if not isinstance(results, Iterable):
        return False
    for entry in results:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).lower()
        if status in {"estimated", "error", "failed"}:
            return True
    return False


def _summarise_execution(
    ssd: SSD,
    execution: Optional[Dict[str, object]],
    estimator: CostEstimator,
    *,
    fallback_time_per_amp: Optional[float] = None,
    prefer_estimate: bool = False,
) -> Optional[Dict[str, object]]:
    if not execution or not isinstance(execution, dict):
        return None

    results = execution.get("results")
    if not isinstance(results, Iterable):
        return None

    meta = execution.get("meta", {})
    try:
        wall = float(meta.get("wall_elapsed_s", 0.0))
    except Exception:
        wall = 0.0

    max_mem = 0
    max_mem_estimated = False
    used_estimated_time = False
    any_failure = False
    for entry in results:
        if not isinstance(entry, dict):
            continue
        mem = entry.get("mem_bytes")
        estimated_mem = False
        if mem is None:
            mem = entry.get("mem_bytes_estimated")
            if mem is not None:
                estimated_mem = True
        if mem is not None:
            try:
                mem_int = int(mem)
            except Exception:
                mem_int = 0
            if mem_int > max_mem:
                max_mem = mem_int
                max_mem_estimated = estimated_mem
        status = str(entry.get("status", "")).lower()
        if status in {"estimated", "error", "failed"}:
            any_failure = True

    if max_mem <= 0:
        est_mem = 0
        for node in ssd.partitions:
            try:
                n = int(node.metrics.get("num_qubits", 0) or 0)
            except Exception:
                n = 0
            est_mem = max(est_mem, int(estimate_sv_bytes(n)))
        if est_mem > 0:
            max_mem = est_mem
            max_mem_estimated = True

    sv_amp_ops = 0.0
    for node in ssd.partitions:
        backend = str(node.backend or "sv").lower()
        if backend != "sv":
            continue
        sv_amp_ops += _amp_ops_for_node(node, estimator)

    modeled_wall = None
    if fallback_time_per_amp and fallback_time_per_amp > 0 and sv_amp_ops > 0:
        modeled_wall = fallback_time_per_amp * sv_amp_ops

    need_estimate = prefer_estimate or any_failure
    if need_estimate and modeled_wall is not None:
        wall = modeled_wall
        used_estimated_time = True

    summary = {
        "wall_time_s": wall,
        "wall_time_estimated": used_estimated_time,
        "max_mem_bytes": int(max_mem),
        "max_mem_estimated": max_mem_estimated,
    }
    if sv_amp_ops > 0:
        summary["sv_amp_ops"] = sv_amp_ops
    if modeled_wall is not None:
        summary["modeled_wall_time_s"] = modeled_wall
    if need_estimate and not used_estimated_time:
        summary["estimate_unavailable"] = True
    return summary


def run_three_way_ablation(
    circuit: QuantumCircuit,
    *,
    planner_cfg: Optional[PlannerConfig] = None,
    execute: bool = False,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> Dict[str, object]:
    """Plan the circuit under three planner configurations."""

    analysis = analyze(circuit)
    base_ssd = analysis.ssd

    planner_base = planner_cfg or PlannerConfig()
    exec_cfg = exec_cfg or ExecutionConfig()

    results: List[VariantRecord] = []
    estimator = CostEstimator.from_planner_config(planner_base)
    sv_seconds_per_amp: Optional[float] = None

    planned_full = plan(base_ssd, planner_base)
    exec_full = execute_ssd(planned_full, exec_cfg) if execute else None
    fallback_time = sv_seconds_per_amp
    summary_full = _summarise_execution(planned_full, exec_full, estimator, fallback_time_per_amp=fallback_time)
    results.append(
        VariantRecord(
            name="full",
            planner=planner_base,
            partitions=_partition_payload(planned_full),
            execution=exec_full,
            summary=summary_full,
        )
    )
    if execute and sv_seconds_per_amp is None:
        ops, elapsed = _accumulate_sv_stats(planned_full, exec_full, estimator)
        if ops > 0 and elapsed > 0:
            sv_seconds_per_amp = elapsed / ops

    merged = _collapse_to_single_partition(base_ssd, circuit)
    planned_nodisjoint = plan(merged, planner_base)
    exec_nodisjoint = execute_ssd(planned_nodisjoint, exec_cfg) if execute else None
    fallback_time = sv_seconds_per_amp
    summary_nodisjoint = _summarise_execution(
        planned_nodisjoint,
        exec_nodisjoint,
        estimator,
        fallback_time_per_amp=fallback_time,
        prefer_estimate=_variant_needs_estimate(exec_nodisjoint),
    )
    results.append(
        VariantRecord(
            name="no_disjoint",
            planner=planner_base,
            partitions=_partition_payload(planned_nodisjoint),
            execution=exec_nodisjoint,
            summary=summary_nodisjoint,
        )
    )
    if execute and sv_seconds_per_amp is None:
        ops, elapsed = _accumulate_sv_stats(planned_nodisjoint, exec_nodisjoint, estimator)
        if ops > 0 and elapsed > 0:
            sv_seconds_per_amp = elapsed / ops

    nohybrid_cfg = PlannerConfig(
        max_ram_gb=planner_base.max_ram_gb,
        max_concurrency=planner_base.max_concurrency,
        prefer_dd=planner_base.prefer_dd,
        hybrid_clifford_tail=False,
        conv_amp_ops_factor=planner_base.conv_amp_ops_factor,
        sv_twoq_factor=planner_base.sv_twoq_factor,
    )
    planned_nohybrid = plan(base_ssd, nohybrid_cfg)
    exec_nohybrid = execute_ssd(planned_nohybrid, exec_cfg) if execute else None
    fallback_time = sv_seconds_per_amp
    summary_nohybrid = _summarise_execution(planned_nohybrid, exec_nohybrid, estimator, fallback_time_per_amp=fallback_time)
    results.append(
        VariantRecord(
            name="no_hybrid",
            planner=nohybrid_cfg,
            partitions=_partition_payload(planned_nohybrid),
            execution=exec_nohybrid,
            summary=summary_nohybrid,
        )
    )
    if execute and sv_seconds_per_amp is None:
        ops, elapsed = _accumulate_sv_stats(planned_nohybrid, exec_nohybrid, estimator)
        if ops > 0 and elapsed > 0:
            sv_seconds_per_amp = elapsed / ops

    return {
        "circuit": {
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "metadata": circuit.metadata,
        },
        "analysis": {
            "global_metrics": analysis.metrics_global,
            "num_partitions": len(base_ssd.partitions),
        },
        "variants": [record.to_json() for record in results],
    }


def _parse_tail_sequence(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    parts = [entry.strip().lower() for entry in text.split(",") if entry.strip()]
    if not parts:
        return None
    for entry in parts:
        if entry not in {"random", "sparse"}:
            raise ValueError(f"Unsupported tail kind '{entry}' in sequence")
    return parts


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-components", type=int, default=4, help="Number of disjoint hybrid blocks")
    parser.add_argument("--component-size", type=int, default=4, help="Number of qubits per block")
    parser.add_argument("--clifford-depth", type=int, default=3, help="Depth of the Clifford prefix in each block")
    parser.add_argument("--tail-depth", type=int, default=3, help="Depth of the non-Clifford tail in each block")
    parser.add_argument(
        "--tail-sequence",
        type=str,
        default=None,
        help="Comma separated tail pattern (values: random,sparse). Defaults to alternating random/sparse.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for the circuit constructor")
    parser.add_argument("--execute", action="store_true", help="Execute the planned SSDs via the simulation engine")
    parser.add_argument("--out", type=str, default=None, help="Optional path to store the JSON summary")
    args = parser.parse_args(argv)

    circuit, _ = build_ablation_circuit(
        num_components=args.num_components,
        component_size=args.component_size,
        clifford_depth=args.clifford_depth,
        tail_depth=args.tail_depth,
        tail_sequence=_parse_tail_sequence(args.tail_sequence),
        seed=args.seed,
    )

    summary = run_three_way_ablation(circuit, execute=args.execute)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - exercised via CLI smoke tests
    main()
