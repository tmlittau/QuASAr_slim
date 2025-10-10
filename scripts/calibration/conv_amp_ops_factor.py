from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from quasar.backends.sv import StatevectorBackend
from quasar.backends.tableau import TableauBackend, stim_available

from scripts.calibration.common import (
    TailSplit,
    build_clifford_tail,
    count_ops,
    split_at_first_nonclifford,
)


@dataclass
class CalibSpec:
    n: int
    depth_cliff: int
    tail_layers: int
    angle_scale: float
    seed: int


@dataclass
class Sample:
    spec: CalibSpec
    split: TailSplit
    prefix_elapsed: float
    tail_elapsed: float
    conv_factor_estimate: float


def _measure(spec: CalibSpec, *, twoq_factor: float, tableau: TableauBackend, statevector: StatevectorBackend) -> Sample | None:
    qc = build_clifford_tail(
        n=spec.n,
        depth_cliff=spec.depth_cliff,
        tail_layers=spec.tail_layers,
        angle_scale=spec.angle_scale,
        seed=spec.seed,
    )
    split = split_at_first_nonclifford(qc)
    if split is None:
        return None

    one_tail, two_tail = count_ops(split.tail_ops)
    tail_norm = float(one_tail + twoq_factor * two_tail)
    if tail_norm <= 0.0:
        return None

    t0 = time.perf_counter()
    pre_state = tableau.run(split.prefix, want_statevector=True)
    t1 = time.perf_counter()
    if pre_state is None:
        return None

    statevector.run(split.tail, initial_state=pre_state)
    t2 = time.perf_counter()

    prefix_elapsed = float(t1 - t0)
    tail_elapsed = float(t2 - t1)
    if prefix_elapsed <= 0.0 or tail_elapsed <= 0.0:
        return None

    ratio = prefix_elapsed / tail_elapsed
    conv_estimate = ratio * tail_norm
    return Sample(
        spec=spec,
        split=split,
        prefix_elapsed=prefix_elapsed,
        tail_elapsed=tail_elapsed,
        conv_factor_estimate=float(conv_estimate),
    )


def calibrate(
    specs: Iterable[CalibSpec],
    *,
    twoq_factor: float,
) -> dict:
    if not stim_available():
        raise SystemExit("Stim (Tableau backend) is required. Install with: pip install stim")

    tableau = TableauBackend()
    statevector = StatevectorBackend()

    samples: List[Sample] = []
    for spec in specs:
        sample = _measure(spec, twoq_factor=twoq_factor, tableau=tableau, statevector=statevector)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise SystemExit("No calibration samples collected; adjust circuit parameters or ensure dependencies are installed.")

    estimates = np.array([s.conv_factor_estimate for s in samples], dtype=float)
    report = {
        "twoq_factor_used": float(twoq_factor),
        "conv_factor": {
            "median": float(np.median(estimates)),
            "mean": float(np.mean(estimates)),
            "p25": float(np.percentile(estimates, 25)),
            "p75": float(np.percentile(estimates, 75)),
        },
        "samples": [
            {
                "n": s.spec.n,
                "depth_cliff": s.spec.depth_cliff,
                "tail_layers": s.spec.tail_layers,
                "angle_scale": s.spec.angle_scale,
                "seed": s.spec.seed,
                "prefix_elapsed_s": s.prefix_elapsed,
                "tail_elapsed_s": s.tail_elapsed,
                "conv_factor_est": s.conv_factor_estimate,
            }
            for s in samples
        ],
    }
    report["num_samples"] = len(samples)
    return report


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or "Calibrate the conversion amp-ops factor.")
    parser.add_argument("--n", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument("--depth-cliff", type=int, default=150)
    parser.add_argument("--tail-layers", type=int, default=10)
    parser.add_argument("--angle-scale", type=float, default=0.1)
    parser.add_argument("--samples-per-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--twoq-factor", type=float, default=4.0)
    parser.add_argument("--out", type=str, default=None)
    return parser


def _generate_specs(args: argparse.Namespace) -> List[CalibSpec]:
    specs: List[CalibSpec] = []
    base_seed = int(args.seed)
    for index, n in enumerate(args.n):
        for sample in range(args.samples_per_n):
            specs.append(
                CalibSpec(
                    n=int(n),
                    depth_cliff=int(args.depth_cliff),
                    tail_layers=int(args.tail_layers),
                    angle_scale=float(args.angle_scale),
                    seed=base_seed + 17 * index + 37 * sample,
                )
            )
    return specs


def run_from_args(args: argparse.Namespace) -> dict:
    report = calibrate(_generate_specs(args), twoq_factor=float(args.twoq_factor))
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
