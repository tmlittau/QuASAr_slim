from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from quasar.backends.dd import DecisionDiagramBackend, ddsim_available

from scripts.calibration.common import collect_metrics, random_tail_circuit


@dataclass
class DDSpec:
    n: int
    layers: int
    angle_scale: float
    rotation_prob: float
    twoq_prob: float
    branch_prob: float
    diag_prob: float
    seed: int


@dataclass
class DDSample:
    spec: DDSpec
    metrics: dict
    elapsed: float


def _measure(spec: DDSpec, backend: DecisionDiagramBackend) -> DDSample | None:
    circuit = random_tail_circuit(
        n=spec.n,
        layers=spec.layers,
        angle_scale=spec.angle_scale,
        rotation_prob=spec.rotation_prob,
        twoq_prob=spec.twoq_prob,
        branch_prob=spec.branch_prob,
        diag_prob=spec.diag_prob,
        seed=spec.seed,
    )
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

    return DDSample(spec=spec, metrics=metrics, elapsed=elapsed)


def _feature_row(metrics: dict) -> tuple[float, float, float, float, float]:
    n = max(int(metrics.get("num_qubits", 0) or 0), 1)
    total = max(int(metrics.get("num_gates", 0) or 0), 1)
    twoq = float(metrics.get("two_qubit_gates", 0) or 0)
    rotations = float(metrics.get("rotation_count", 0) or 0)
    sparsity = float(metrics.get("sparsity", 0.0) or 0.0)

    frontier = float(n)
    log_frontier = math.log2(frontier + 1.0)
    base_nodes = frontier * max(1.0, log_frontier)
    gate_factor = max(1.0, math.log2(total + 1.0))
    base_component = float(total) * base_nodes * gate_factor

    rotation_density = rotations / float(total)
    twoq_density = twoq / float(total)

    return (
        base_component,
        base_component * log_frontier,
        base_component * rotation_density,
        base_component * twoq_density,
        -base_component * sparsity,
    )


def calibrate(samples: List[DDSample]) -> dict:
    if not samples:
        raise SystemExit("No decision diagram samples collected; adjust sampling parameters.")

    design_rows = []
    targets = []
    feature_cache: List[dict] = []
    for sample in samples:
        base, frontier_term, rotation_term, twoq_term, sparsity_term = _feature_row(sample.metrics)
        design_rows.append([1.0, base, frontier_term, rotation_term, twoq_term, sparsity_term])
        targets.append(sample.elapsed)
        feature_cache.append(sample.metrics)

    design = np.asarray(design_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    coeffs, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    base_cost = float(coeffs[0])
    gate_node_factor = float(coeffs[1])
    if gate_node_factor <= 0.0:
        raise SystemExit("Calibration failed: negative decision diagram gate-node factor.")

    frontier_weight = float(coeffs[2] / gate_node_factor)
    rotation_weight = float(coeffs[3] / gate_node_factor)
    twoq_weight = float(coeffs[4] / gate_node_factor)
    sparsity_discount = float(-coeffs[5] / gate_node_factor)

    modifiers = []
    for metrics in feature_cache:
        total = max(int(metrics.get("num_gates", 0) or 0), 1)
        frontier = max(int(metrics.get("num_qubits", 0) or 0), 1)
        log_frontier = math.log2(frontier + 1.0)
        rotations = float(metrics.get("rotation_count", 0) or 0)
        twoq = float(metrics.get("two_qubit_gates", 0) or 0)
        sparsity = float(metrics.get("sparsity", 0.0) or 0.0)
        rotation_density = rotations / float(total)
        twoq_density = twoq / float(total)
        modifier = 1.0
        modifier += frontier_weight * log_frontier
        modifier += rotation_weight * rotation_density
        modifier += twoq_weight * twoq_density
        modifier -= sparsity_discount * sparsity
        modifiers.append(modifier)

    modifier_floor = max(0.01, float(np.percentile(modifiers, 5)))

    predictions = design @ coeffs
    residual_sum = float(np.sum((y - predictions) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2)) if y.size > 0 else 0.0
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    return {
        "dd_base_cost": base_cost,
        "dd_gate_node_factor": gate_node_factor,
        "dd_frontier_weight": frontier_weight,
        "dd_rotation_weight": rotation_weight,
        "dd_twoq_weight": twoq_weight,
        "dd_sparsity_discount": sparsity_discount,
        "dd_modifier_floor": modifier_floor,
        "rank": int(rank),
        "residual_sum_sqr": residual_sum,
        "r2": r2,
    }


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or "Calibrate the decision diagram cost model parameters.")
    parser.add_argument("--n", type=int, nargs="+", default=[10, 12, 14, 16])
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--samples-per-n", type=int, default=5)
    parser.add_argument("--angle-scale", type=float, default=0.15)
    parser.add_argument("--rotation-prob", type=float, default=0.5)
    parser.add_argument("--rotation-prob-span", type=float, default=0.3)
    parser.add_argument("--twoq-prob", type=float, default=0.5)
    parser.add_argument("--twoq-prob-span", type=float, default=0.3)
    parser.add_argument("--branch-prob", type=float, default=0.15)
    parser.add_argument("--diag-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", type=str, default=None)
    return parser


def _generate_specs(args: argparse.Namespace) -> List[DDSpec]:
    specs: List[DDSpec] = []
    base_seed = int(args.seed)
    for index, n in enumerate(args.n):
        for sample_index in range(args.samples_per_n):
            span_rot = float(args.rotation_prob_span)
            span_twoq = float(args.twoq_prob_span)
            rotation_prob = float(args.rotation_prob) + span_rot * (sample_index - args.samples_per_n / 2) / max(args.samples_per_n - 1, 1)
            twoq_prob = float(args.twoq_prob) + span_twoq * (sample_index - args.samples_per_n / 2) / max(args.samples_per_n - 1, 1)
            specs.append(
                DDSpec(
                    n=int(n),
                    layers=int(args.layers),
                    angle_scale=float(args.angle_scale),
                    rotation_prob=max(min(rotation_prob, 0.95), 0.05),
                    twoq_prob=max(min(twoq_prob, 0.95), 0.05),
                    branch_prob=float(args.branch_prob),
                    diag_prob=float(args.diag_prob),
                    seed=base_seed + 41 * index + 73 * sample_index,
                )
            )
    return specs


def run_from_args(args: argparse.Namespace) -> dict:
    if not ddsim_available():
        raise SystemExit("MQT DDSIM is required. Install with: pip install mqt.ddsim mqt.core")

    backend = DecisionDiagramBackend()
    samples: List[DDSample] = []
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
