from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from quasar.backends.sv import StatevectorBackend

from scripts.calibration.common import collect_metrics, random_tail_circuit


@dataclass
class TailSpec:
    n: int
    layers: int
    angle_scale: float
    rotation_prob: float
    twoq_prob: float
    branch_prob: float
    diag_prob: float
    seed: int


@dataclass
class TailSample:
    spec: TailSpec
    metrics: dict
    elapsed: float


def _measure(spec: TailSpec, backend: StatevectorBackend) -> TailSample | None:
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
    twoq = int(metrics.get("two_qubit_gates", 0) or 0)
    if total <= 0:
        return None

    start = time.perf_counter()
    backend.run(circuit)
    end = time.perf_counter()
    elapsed = float(end - start)
    if elapsed <= 0.0:
        return None

    return TailSample(spec=spec, metrics=metrics, elapsed=elapsed)


def _build_design_matrix(samples: Iterable[TailSample]) -> tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    targets: List[float] = []
    for sample in samples:
        total = float(sample.metrics.get("num_gates", 0) or 0)
        twoq = float(sample.metrics.get("two_qubit_gates", 0) or 0)
        oneq = max(total - twoq, 0.0)
        if oneq <= 0.0 and twoq <= 0.0:
            continue
        rows.append([oneq, twoq])
        targets.append(sample.elapsed)
    if not rows:
        raise SystemExit("No valid samples collected for statevector calibration.")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def calibrate(samples: Iterable[TailSample]) -> dict:
    X, y = _build_design_matrix(samples)
    coefs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coefs[0])
    beta = float(coefs[1])
    if alpha <= 0.0:
        raise SystemExit("Calibration failed: single-qubit coefficient is non-positive.")
    twoq_factor = beta / alpha if beta > 0.0 else 0.0

    predictions = X @ coefs
    residual_sum = float(np.sum((y - predictions) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2)) if y.size > 0 else 0.0
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0

    return {
        "alpha_oneq_cost": alpha,
        "beta_twoq_cost": beta,
        "twoq_factor": twoq_factor,
        "rank": int(rank),
        "residual_sum_sqr": float(residual_sum) if residuals.size == 0 else float(residuals[0]),
        "r2": float(r2),
    }


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or "Calibrate the statevector two-qubit weight.")
    parser.add_argument("--n", type=int, nargs="+", default=[8, 10, 12])
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--samples-per-n", type=int, default=5)
    parser.add_argument("--angle-scale", type=float, default=0.2)
    parser.add_argument("--rotation-prob", type=float, default=0.7)
    parser.add_argument("--rotation-prob-span", type=float, default=0.2)
    parser.add_argument("--twoq-prob", type=float, default=0.5)
    parser.add_argument("--twoq-prob-span", type=float, default=0.3)
    parser.add_argument("--branch-prob", type=float, default=0.1)
    parser.add_argument("--diag-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=str, default=None)
    return parser


def _generate_specs(args: argparse.Namespace) -> List[TailSpec]:
    specs: List[TailSpec] = []
    base_seed = int(args.seed)
    for index, n in enumerate(args.n):
        for sample_index in range(args.samples_per_n):
            span_rot = float(args.rotation_prob_span)
            span_twoq = float(args.twoq_prob_span)
            rotation_prob = float(args.rotation_prob) + span_rot * (sample_index - args.samples_per_n / 2) / max(args.samples_per_n - 1, 1)
            twoq_prob = float(args.twoq_prob) + span_twoq * (sample_index - args.samples_per_n / 2) / max(args.samples_per_n - 1, 1)
            specs.append(
                TailSpec(
                    n=int(n),
                    layers=int(args.layers),
                    angle_scale=float(args.angle_scale),
                    rotation_prob=max(min(rotation_prob, 0.95), 0.05),
                    twoq_prob=max(min(twoq_prob, 0.95), 0.05),
                    branch_prob=float(args.branch_prob),
                    diag_prob=float(args.diag_prob),
                    seed=base_seed + 31 * index + 97 * sample_index,
                )
            )
    return specs


def run_from_args(args: argparse.Namespace) -> dict:
    backend = StatevectorBackend()
    samples: List[TailSample] = []
    for spec in _generate_specs(args):
        sample = _measure(spec, backend)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise SystemExit("No calibration samples collected; try adjusting the sampling parameters.")
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
