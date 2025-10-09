#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Tuple

from quasar.SSD import SSD, PartitionNode
from quasar.simulation_engine import ExecutionConfig, execute_ssd

try:
    from qiskit import QuantumCircuit
except ImportError as exc:  # pragma: no cover - script requires qiskit at runtime
    raise SystemExit("This demo requires qiskit to be installed") from exc


def _build_partition_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")

    qc = QuantumCircuit(num_qubits)
    for layer in range(depth):
        angle = 0.01 + layer * 1.0e-4
        for q in range(num_qubits):
            qc.rx(angle, q)
            qc.rz(angle * 0.5, q)
        for start in range(0, num_qubits - 1, 2):
            qc.cz(start, start + 1)
        for start in range(1, num_qubits - 1, 2):
            qc.cz(start, start + 1)
    return qc


def _build_disjoint_ssd(num_qubits: int, depth: int, backend: str) -> Tuple[SSD, int]:
    base = _build_partition_circuit(num_qubits, depth)
    gate_count = int(base.size())

    ssd = SSD()

    first = PartitionNode(
        id=0,
        qubits=list(range(num_qubits)),
        circuit=base,
        metrics={"num_qubits": num_qubits, "gate_count": gate_count},
        backend=backend,
    )
    ssd.add(first)

    second = PartitionNode(
        id=1,
        qubits=list(range(num_qubits, 2 * num_qubits)),
        circuit=base.copy(),
        metrics={"num_qubits": num_qubits, "gate_count": gate_count},
        backend=backend,
    )
    ssd.add(second)

    return ssd, gate_count


def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "unknown"
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate partition cache reuse on disjoint partitions"
    )
    parser.add_argument(
        "--qubits-per-partition",
        type=int,
        default=12,
        help="Number of qubits in each partition",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8000,
        help="Depth of each partition's circuit",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sv",
        help="Backend to simulate partitions with",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum worker threads for execution",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=5.0,
        help="Heartbeat interval passed to the execution config",
    )
    parser.add_argument(
        "--stuck-warn",
        type=float,
        default=60.0,
        help="Stuck warning interval passed to the execution config",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    backend = args.backend.lower()

    print(
        f"Building two disjoint partitions with {args.qubits_per_partition} qubits "
        f"and depth {args.depth} (backend={backend})"
    )
    ssd, gate_count = _build_disjoint_ssd(args.qubits_per_partition, args.depth, backend)
    print(
        f"Each partition contains {gate_count} gates; total circuit width is {2 * args.qubits_per_partition} qubits."
    )

    def _execute(enable_cache: bool) -> Tuple[float, dict]:
        cfg = ExecutionConfig(
            max_workers=args.max_workers,
            heartbeat_sec=args.heartbeat,
            stuck_warn_sec=args.stuck_warn,
            enable_partition_cache=enable_cache,
        )
        start = time.perf_counter()
        result = execute_ssd(ssd, cfg)
        elapsed = time.perf_counter() - start
        return elapsed, result

    elapsed_off, result_off = _execute(enable_cache=False)
    print("\nCaching disabled:")
    print(
        "  wall time: {elapsed:.2f} s (engine reported {engine:.2f} s)".format(
            elapsed=elapsed_off,
            engine=result_off["meta"].get("wall_elapsed_s", elapsed_off),
        )
    )
    print(
        "  cache hits/misses: {hits} / {misses}".format(
            hits=result_off["meta"].get("cache_hits"),
            misses=result_off["meta"].get("cache_misses"),
        )
    )
    print(f"  peak RSS: {_format_bytes(result_off['meta'].get('peak_rss_bytes'))}")

    # rebuild SSD so cached metadata does not leak between runs
    ssd, _ = _build_disjoint_ssd(args.qubits_per_partition, args.depth, backend)

    elapsed_on, result_on = _execute(enable_cache=True)
    print("\nCaching enabled:")
    print(
        "  wall time: {elapsed:.2f} s (engine reported {engine:.2f} s)".format(
            elapsed=elapsed_on,
            engine=result_on["meta"].get("wall_elapsed_s", elapsed_on),
        )
    )
    print(
        "  cache hits/misses: {hits} / {misses}".format(
            hits=result_on["meta"].get("cache_hits"),
            misses=result_on["meta"].get("cache_misses"),
        )
    )
    print(f"  peak RSS: {_format_bytes(result_on['meta'].get('peak_rss_bytes'))}")

    print("\nPer-partition summary (caching enabled run):")
    for entry in result_on["results"]:
        print(
            "  partition {partition}: backend={backend} qubits={num_qubits} cache_hit={cache_hit} "
            "elapsed={elapsed_s:.2f} s".format(
                partition=entry.get("partition"),
                num_qubits=entry.get("num_qubits"),
                cache_hit=entry.get("cache_hit"),
                elapsed_s=float(entry.get("elapsed_s") or 0.0),
            )
        )


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
