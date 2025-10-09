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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import CIRCUIT_REGISTRY, build as build_circuit
from quasar.analyzer import analyze
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd

from plots.palette import EDGE_COLOR, PASTEL_COLORS, apply_paper_style


@dataclass
class CircuitSpec:
    """Description of a benchmark circuit to run for the overhead analysis."""

    kind: str
    params: Dict[str, object]
    depth_hint: int | None = None

    def label(self) -> str:
        """Return a short label for plotting purposes."""

        num_qubits = self.params.get("num_qubits")
        if num_qubits is None:
            return self.kind
        return f"{self.kind}\n(n={num_qubits})"

    def depth_value(self) -> int | None:
        """Best-effort extraction of an effective depth for the circuit."""

        if self.depth_hint is not None:
            return self.depth_hint

        depth = self.params.get("depth")
        if isinstance(depth, (int, float)):
            return int(depth)

        depth_pre = self.params.get("depth_pre")
        depth_post = self.params.get("depth_post")
        if isinstance(depth_pre, (int, float)) and isinstance(depth_post, (int, float)):
            return int(depth_pre) + int(depth_post)

        return None


@dataclass
class CircuitFamily:
    """Group of related benchmark circuits."""

    name: str
    variants: List[CircuitSpec]
    color_key: str

    def display_label(self) -> str:
        return self.name


DEFAULT_FAMILIES: List[CircuitFamily] = [
    CircuitFamily(
        name="GHZ clusters",
        color_key="tableau",
        variants=[
            CircuitSpec(
                "ghz_clusters_random",
                {"num_qubits": 16, "block_size": 8, "depth": 30, "seed": 1},
            ),
            CircuitSpec(
                "ghz_clusters_random",
                {"num_qubits": 24, "block_size": 8, "depth": 45, "seed": 11},
            ),
            CircuitSpec(
                "ghz_clusters_random",
                {"num_qubits": 28, "block_size": 7, "depth": 60, "seed": 21},
            ),
        ],
    ),
    CircuitFamily(
        name="Random Clifford",
        color_key="sv",
        variants=[
            CircuitSpec(
                "random_clifford",
                {"num_qubits": 18, "depth": 40, "seed": 2},
            ),
            CircuitSpec(
                "random_clifford",
                {"num_qubits": 24, "depth": 60, "seed": 12},
            ),
            CircuitSpec(
                "random_clifford",
                {"num_qubits": 30, "depth": 80, "seed": 22},
            ),
        ],
    ),
    CircuitFamily(
        name="Stitched banded QFT",
        color_key="dd",
        variants=[
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
                depth_hint=30,
            ),
            CircuitSpec(
                "stitched_rand_bandedqft_rand",
                {
                    "num_qubits": 22,
                    "block_size": 6,
                    "depth_pre": 20,
                    "depth_post": 20,
                    "qft_bandwidth": 2,
                    "seed": 13,
                },
                depth_hint=40,
            ),
            CircuitSpec(
                "stitched_rand_bandedqft_rand",
                {
                    "num_qubits": 26,
                    "block_size": 6,
                    "depth_pre": 25,
                    "depth_post": 25,
                    "qft_bandwidth": 3,
                    "seed": 23,
                },
                depth_hint=50,
            ),
        ],
    ),
    CircuitFamily(
        name="Clifford + rotations",
        color_key="conversion",
        variants=[
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
            CircuitSpec(
                "clifford_plus_rot",
                {
                    "num_qubits": 24,
                    "depth": 55,
                    "rot_prob": 0.35,
                    "angle_scale": 0.25,
                    "block_size": 6,
                    "pair_scope": "block",
                    "seed": 14,
                },
            ),
            CircuitSpec(
                "clifford_plus_rot",
                {
                    "num_qubits": 28,
                    "depth": 70,
                    "rot_prob": 0.4,
                    "angle_scale": 0.3,
                    "block_size": 7,
                    "pair_scope": "block",
                    "seed": 24,
                },
            ),
        ],
    ),
]

DEFAULT_CIRCUITS: List[CircuitSpec] = [
    spec for family in DEFAULT_FAMILIES for spec in family.variants
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


def _format_range(values: List[int | None]) -> str:
    numeric = [val for val in values if val is not None]
    if not numeric:
        return "N/A"
    lo, hi = min(numeric), max(numeric)
    if lo == hi:
        return f"{lo}"
    return f"{lo}–{hi}"


def _plot_results(
    families: List[CircuitFamily],
    family_results: Dict[str, List[RunResult]],
    output_path: Path,
) -> None:
    if not family_results:
        raise ValueError("No results available to plot")

    apply_paper_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_labels: List[str] = []
    means: List[float] = []
    errors: List[float] = []
    qubit_ranges: List[str] = []
    depth_ranges: List[str] = []
    colors: List[str] = []

    for family in families:
        results = family_results.get(family.name, [])
        if not results:
            continue

        overheads = [res.planning_overhead_pct for res in results if not np.isnan(res.planning_overhead_pct)]
        if not overheads:
            continue

        mean_overhead = float(np.mean(overheads))
        std_overhead = float(np.std(overheads, ddof=1)) if len(overheads) > 1 else 0.0

        family_labels.append(family.display_label())
        means.append(mean_overhead)
        errors.append(std_overhead)
        colors.append(PASTEL_COLORS.get(family.color_key, PASTEL_COLORS["tableau"]))

        qubits = [res.num_qubits for res in results]
        qubit_ranges.append(_format_range(qubits))
        depth_ranges.append(
            _format_range([spec.depth_value() for spec in family.variants])
        )

    if not means:
        raise ValueError("No valid overhead data to plot")

    x = np.arange(len(means))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        capsize=6,
    )

    ax.set_xticks(x, family_labels)
    ax.set_ylabel("Relative planning overhead (%)")
    ax.set_title("Planning overhead across circuit families")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    max_height = max(bar.get_height() + err for bar, err in zip(bars, errors))
    offset = max(0.05 * max_height, 1.0)

    for xpos, bar, err, qubits, depths in zip(x, bars, errors, qubit_ranges, depth_ranges):
        height = bar.get_height()
        box_text = f"Qubits: {qubits}\nDepth: {depths}"
        ax.text(
            xpos,
            height + err + offset,
            box_text,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": EDGE_COLOR,
                "linewidth": 0.8,
            },
        )

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

    families = DEFAULT_FAMILIES
    _validate_circuits(DEFAULT_CIRCUITS)

    planner_cfg = PlannerConfig(max_ram_gb=args.max_ram_gb)
    exec_cfg = ExecutionConfig(max_ram_gb=args.max_ram_gb, max_workers=args.max_workers)

    results: List[RunResult] = []
    family_results: Dict[str, List[RunResult]] = {family.name: [] for family in families}
    for family in families:
        for spec in family.variants:
            result = _run_circuit(spec, planner_cfg, exec_cfg)
            results.append(result)
            family_results[family.name].append(result)
            print(
                f"{family.name} ({spec.params.get('num_qubits', 'n/a')}q): "
                f"planning={result.planning_time_s:.3f}s, "
                f"execution={result.execution_time_s:.3f}s, "
                f"overhead={result.planning_overhead_pct:.1f}%",
            )

    _plot_results(families, family_results, args.output)
    print(f"Saved plot to {args.output}")

    if args.data_out is not None:
        args.data_out.parent.mkdir(parents=True, exist_ok=True)
        with args.data_out.open("w", encoding="utf-8") as fh:
            json.dump([res.to_dict() for res in results], fh, indent=2)
        print(f"Saved timing data to {args.data_out}")


if __name__ == "__main__":
    main()
