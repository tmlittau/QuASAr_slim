"""Generate a planning-overhead plot across a collection of benchmark circuits.

This script runs a selection of benchmark circuits, measures the time spent in
planning and execution, and visualises the relative planning overhead (as a
percentage of execution wall-clock time) against the actual execution runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import CIRCUIT_REGISTRY, build as build_circuit
from quasar.analyzer import analyze
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd


@dataclass
class CircuitSpec:
    """Description of a benchmark circuit to run for the overhead analysis."""

    kind: str
    params: Dict[str, object]

    def label(self) -> str:
        """Return a short label for plotting purposes."""

        num_qubits = self.params.get("num_qubits")
        if num_qubits is None:
            return self.kind
        return f"{self.kind}\n(n={num_qubits})"


DEFAULT_CIRCUITS: List[CircuitSpec] = [
    CircuitSpec(
        "ghz_clusters_random",
        {"num_qubits": 16, "block_size": 8, "depth": 30, "seed": 1},
    ),
    CircuitSpec(
        "random_clifford",
        {"num_qubits": 18, "depth": 40, "seed": 2},
    ),
    CircuitSpec(
        "stitched_rand_bandedqft_rand",
        {
            "num_qubits": 20,
            "block_size": 5,
            "depth_pre": 15,
            "depth_post": 15,
            "qft_bandwidth": 2,
            "seed": 3,
        },
    ),
    CircuitSpec(
        "clifford_plus_rot",
        {
            "num_qubits": 18,
            "depth": 35,
            "rot_prob": 0.3,
            "angle_scale": 0.2,
            "block_size": 6,
            "pair_scope": "block",
            "seed": 4,
        },
    ),
]


@dataclass
class RunResult:
    """Container for the timing data of a single circuit run."""

    circuit: CircuitSpec
    num_qubits: int
    analysis_time_s: float
    planning_time_s: float
    execution_time_s: float
    planning_overhead_pct: float

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["circuit"] = {
            "kind": self.circuit.kind,
            "params": self.circuit.params,
        }
        return payload


def _validate_circuits(specs: Iterable[CircuitSpec]) -> None:
    missing = [spec.kind for spec in specs if spec.kind not in CIRCUIT_REGISTRY]
    if missing:
        known = ", ".join(sorted(CIRCUIT_REGISTRY))
        raise ValueError(
            "Unknown circuit kinds: " + ", ".join(missing) + f". Known kinds: {known}"
        )


def _run_circuit(
    spec: CircuitSpec,
    planner_cfg: PlannerConfig,
    exec_cfg: ExecutionConfig,
) -> RunResult:
    circuit = build_circuit(spec.kind, **spec.params)

    analysis_start = perf_counter()
    analysis = analyze(circuit)
    analysis_time = perf_counter() - analysis_start

    plan_start = perf_counter()
    planned = plan(analysis.ssd, planner_cfg)
    planning_time = perf_counter() - plan_start

    execution = execute_ssd(planned, exec_cfg)
    execution_time = float(execution.get("meta", {}).get("wall_elapsed_s", 0.0))

    if execution_time <= 0:
        planning_overhead_pct = float("nan")
    else:
        planning_overhead_pct = 100.0 * planning_time / execution_time

    return RunResult(
        circuit=spec,
        num_qubits=circuit.num_qubits,
        analysis_time_s=analysis_time,
        planning_time_s=planning_time,
        execution_time_s=execution_time,
        planning_overhead_pct=planning_overhead_pct,
    )


def _plot_results(results: List[RunResult], output_path: Path) -> None:
    if not results:
        raise ValueError("No results available to plot")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r.execution_time_s)
    runtimes = [res.execution_time_s for res in sorted_results]
    overheads = [res.planning_overhead_pct for res in sorted_results]
    labels = [res.circuit.label() for res in sorted_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(runtimes, overheads, color="#1f77b4", zorder=3)

    for runtime, overhead, label in zip(runtimes, overheads, labels):
        ax.annotate(
            label,
            xy=(runtime, overhead),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
        )

    ax.set_xlabel("Execution wall time (s)")
    ax.set_ylabel("Planning overhead (%)")
    ax.set_title("Planning overhead vs execution runtime")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/planning_overhead.png"),
        help="Path where the overhead plot should be written.",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=None,
        help="Optional path to dump the raw timing data as JSON.",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=4.0,
        help="RAM budget (in GiB) for both planning and execution.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Override the number of worker threads used during execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    specs = DEFAULT_CIRCUITS
    _validate_circuits(specs)

    planner_cfg = PlannerConfig(max_ram_gb=args.max_ram_gb)
    exec_cfg = ExecutionConfig(max_ram_gb=args.max_ram_gb, max_workers=args.max_workers)

    results: List[RunResult] = []
    for spec in specs:
        result = _run_circuit(spec, planner_cfg, exec_cfg)
        results.append(result)
        print(
            f"{spec.kind}: planning={result.planning_time_s:.3f}s, "
            f"execution={result.execution_time_s:.3f}s, "
            f"overhead={result.planning_overhead_pct:.1f}%",
        )

    _plot_results(results, args.output)
    print(f"Saved plot to {args.output}")

    if args.data_out is not None:
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        with args.data_out.open("w", encoding="utf-8") as fh:
            json.dump([res.to_dict() for res in results], fh, indent=2)
        print(f"Saved timing data to {args.data_out}")


if __name__ == "__main__":
    main()
