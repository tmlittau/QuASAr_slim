from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from quasar.backends.tableau import TableauBackend, stim_available

from scripts.calibration.common import collect_metrics, random_clifford_circuit


@dataclass
class CliffordSpec:
    n: int
    depth: int
    seed: int


@dataclass
class CliffordSample:
    spec: CliffordSpec
    metrics: dict
    elapsed: float


def _measure(spec: CliffordSpec, backend: TableauBackend) -> CliffordSample | None:
    circuit = random_clifford_circuit(n=spec.n, depth=spec.depth, seed=spec.seed)
    metrics = collect_metrics(circuit)
    total = int(metrics.get("num_gates", 0) or 0)
    if total <= 0:
        return None

    start = time.perf_counter()
    backend.run(circuit)
    end = time.perf_counter()
    elapsed = float(end - start)
    if elapsed <= 0.0:
        return None

    return CliffordSample(spec=spec, metrics=metrics, elapsed=elapsed)


def calibrate(samples: List[CliffordSample]) -> dict:
    if not samples:
        raise SystemExit("No tableau samples collected; adjust sampling parameters.")

    X = []
    y = []
    for sample in samples:
        total = float(sample.metrics.get("num_gates", 0) or 0)
        X.append([1.0, total])
        y.append(sample.elapsed)

    design = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    coeffs, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    base = float(coeffs[0])
    per_gate = float(coeffs[1])
    if per_gate <= 0.0:
        raise SystemExit("Calibration failed: negative or zero tableau unit cost.")

    predictions = design @ coeffs
    residual_sum = float(np.sum((targets - predictions) ** 2))
    total_sum = float(np.sum((targets - np.mean(targets)) ** 2)) if targets.size > 0 else 0.0
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    return {
        "tableau_prefix_unit_cost": per_gate,
        "tableau_base_overhead": base,
        "rank": int(rank),
        "residual_sum_sqr": residual_sum,
        "r2": r2,
    }


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or "Calibrate the tableau prefix unit cost.")
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--depth", type=int, default=200)
    parser.add_argument("--samples-per-n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    return parser


def _generate_specs(args: argparse.Namespace) -> List[CliffordSpec]:
    specs: List[CliffordSpec] = []
    base_seed = int(args.seed)
    for index, n in enumerate(args.n):
        for sample_index in range(args.samples_per_n):
            specs.append(CliffordSpec(n=int(n), depth=int(args.depth), seed=base_seed + 19 * index + 53 * sample_index))
    return specs


def run_from_args(args: argparse.Namespace) -> dict:
    if not stim_available():
        raise SystemExit("Stim (Tableau backend) is required. Install with: pip install stim")

    backend = TableauBackend()
    samples: List[CliffordSample] = []
    for spec in _generate_specs(args):
        sample = _measure(spec, backend)
        if sample is not None:
            samples.append(sample)

    report = calibrate(samples)
    report["num_samples"] = len(samples)
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    report = run_from_args(args)
    if args.out:
        with open(args.out, "w", encoding="utf8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
