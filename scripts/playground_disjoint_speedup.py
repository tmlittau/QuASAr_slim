from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from benchmarks.disjoint import disjoint_preps_plus_tails
from quasar.cost_estimator import CostEstimator, CostParams


@dataclass(frozen=True)
class BlockSummary:
    index: int
    qubits: Tuple[int, ...]
    oneq_ops: int
    twoq_ops: int

    @property
    def size(self) -> int:
        return len(self.qubits)


def _compute_blocks(num_qubits: int, num_blocks: int) -> List[Tuple[int, ...]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > num_qubits:
        raise ValueError("num_blocks cannot exceed num_qubits")
    base = num_qubits // num_blocks
    remainder = num_qubits % num_blocks
    blocks: List[Tuple[int, ...]] = []
    start = 0
    for index in range(num_blocks):
        size = base + (1 if index < remainder else 0)
        block = tuple(range(start, start + size))
        if not block:
            raise ValueError("Encountered empty block while partitioning qubits")
        blocks.append(block)
        start += size
    return blocks


def _gate_qubits(qargs: Sequence[Any], qubit_indices: Dict[Any, int]) -> Tuple[int, ...]:
    indices: List[int] = []
    for q in qargs:
        if isinstance(q, int):
            indices.append(q)
            continue
        try:
            idx = qubit_indices[q]
        except KeyError as exc:  # pragma: no cover - defensive fallback
            raise TypeError(f"Unsupported qubit argument type: {type(q)!r}") from exc
        indices.append(idx)
    return tuple(indices)


def _summarize_blocks(qc, blocks: Sequence[Tuple[int, ...]]) -> List[BlockSummary]:
    qubit_to_block: Dict[int, int] = {}
    for block_index, qubits in enumerate(blocks):
        for qubit in qubits:
            qubit_to_block[qubit] = block_index

    counts = [[0, 0] for _ in blocks]  # [oneq, twoq]

    qubit_indices: Dict[Any, int] = {qubit: index for index, qubit in enumerate(qc.qubits)}

    for instruction in qc.data:
        qubits = _gate_qubits(instruction.qubits, qubit_indices)
        if not qubits:
            continue
        block_indices = {qubit_to_block.get(q) for q in qubits}
        if None in block_indices:
            missing = sorted(q for q in qubits if q not in qubit_to_block)
            raise ValueError(
                f"Gate {instruction.operation} targets qubits outside the declared blocks: {missing}"
            )
        if len(block_indices) != 1:
            raise ValueError(
                "Encountered multi-block gate; the disjoint generator is expected to keep blocks independent"
            )
        block_index = block_indices.pop()
        if len(qubits) >= 2:
            counts[block_index][1] += 1
        else:
            counts[block_index][0] += 1

    summaries: List[BlockSummary] = []
    for index, block in enumerate(blocks):
        oneq, twoq = counts[index]
        summaries.append(BlockSummary(index=index, qubits=tuple(block), oneq_ops=oneq, twoq_ops=twoq))
    return summaries


def _parallel_cost(costs: Sequence[float], workers: int) -> float:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not costs:
        return 0.0
    workers = min(workers, len(costs))
    loads = [0.0 for _ in range(workers)]
    for cost in sorted(costs, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        loads[index] += cost
    return max(loads)


def analyze_case(
    *,
    n: int,
    num_blocks: int,
    tail_depth: int,
    block_prep: str,
    tail_kind: str,
    angle_scale: float,
    sparsity: float,
    bandwidth: int,
    est: CostEstimator,
    seed: int | None,
    include_conversion: bool,
    workers: int,
) -> Dict[str, Any]:
    if n <= 0:
        raise ValueError("n must be positive")
    if tail_depth < 0:
        raise ValueError("tail_depth must be >= 0")

    blocks = _compute_blocks(n, num_blocks)
    params = dict(
        num_qubits=n,
        num_blocks=num_blocks,
        block_prep=block_prep,
        tail_kind=tail_kind,
        tail_depth=tail_depth,
        angle_scale=angle_scale,
        sparsity=sparsity,
        bandwidth=bandwidth,
    )
    if seed is not None:
        params["seed"] = seed

    qc = disjoint_preps_plus_tails(**params)
    summaries = _summarize_blocks(qc, blocks)

    total_one = sum(summary.oneq_ops for summary in summaries)
    total_two = sum(summary.twoq_ops for summary in summaries)
    sv_cost = est.sv_cost(n, total_one, total_two)

    block_cost = 0.0
    block_parallel_inputs: List[float] = []
    block_details: List[Dict[str, Any]] = []
    for summary in summaries:
        block_sv = est.sv_cost(summary.size, summary.oneq_ops, summary.twoq_ops)
        block_cost += block_sv
        conv_cost = est.conversion_cost(summary.size) if include_conversion else 0.0
        block_parallel_inputs.append(block_sv + conv_cost)
        block_details.append(
            {
                "index": summary.index,
                "size": summary.size,
                "oneq_ops": summary.oneq_ops,
                "twoq_ops": summary.twoq_ops,
                "sv_cost": block_sv,
                "conversion_cost": conv_cost,
            }
        )

    parallel_cost = _parallel_cost(block_parallel_inputs, workers)
    sequential_cost = sum(block_parallel_inputs)

    baseline_cost = min(sv_cost, sequential_cost)

    norm = float(1 << n)
    speedup_vs_sv = (sv_cost / parallel_cost) if parallel_cost > 0 else math.inf
    speedup_vs_best = (baseline_cost / parallel_cost) if parallel_cost > 0 else math.inf

    return {
        "num_qubits": n,
        "num_blocks": num_blocks,
        "block_sizes": [summary.size for summary in summaries],
        "tail_depth": tail_depth,
        "total_depth": qc.depth(),
        "sv_total": sv_cost,
        "sv_total_norm": sv_cost / norm,
        "block_total": block_cost,
        "block_total_norm": block_cost / norm,
        "sequential_cost": sequential_cost,
        "parallel_cost": parallel_cost,
        "baseline_cost": baseline_cost,
        "speedup_vs_sv": speedup_vs_sv,
        "speedup_vs_baseline": speedup_vs_best,
        "blocks": block_details,
    }


def _is_interactive_backend() -> bool:
    return matplotlib.get_backend() in {
        "TkAgg",
        "QtAgg",
        "Qt5Agg",
        "MacOSX",
        "GTK3Agg",
        "GTK4Agg",
        "wxAgg",
        "WebAgg",
        "nbAgg",
    }


def _format_block_sizes(block_sizes: Iterable[int]) -> str:
    return "[" + ",".join(str(size) for size in block_sizes) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Disjoint circuit playground: sweep block counts and tail depths to compare "
            "QuASAr's parallel disjoint execution against the best available whole-circuit baseline."
        )
    )
    ap.add_argument("--n", type=int, nargs="+", default=[32, 48, 64], help="Numbers of qubits to analyze")
    ap.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Candidate numbers of disjoint blocks (must be ≤ n for each case)",
    )
    ap.add_argument(
        "--tail-depth",
        type=int,
        nargs="+",
        default=[0, 10, 20, 40],
        dest="tail_depths",
        help="Tail depths (per block) to sweep",
    )
    ap.add_argument("--block-prep", "--prep", type=str, default="mixed", dest="block_prep", help="Block preparation kind")
    ap.add_argument("--tail-kind", type=str, default="mixed", help="Tail circuit kind")
    ap.add_argument("--angle-scale", type=float, default=0.1, help="Tail rotation angle scale for diagonal layers")
    ap.add_argument("--sparsity", type=float, default=0.05, help="Tail sparsity for diagonal layers")
    ap.add_argument("--bandwidth", type=int, default=2, help="Tail bandwidth for diagonal layers")
    ap.add_argument("--conv-factor", type=float, default=64.0, help="Conversion amortization factor")
    ap.add_argument("--twoq-factor", type=float, default=4.0, help="Statevector two-qubit gate factor")
    ap.add_argument("--target-speedup", type=float, default=1.0, help="Reference speedup line (SV_cost / per-block cost)")
    ap.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="Number of parallel workers available for disjoint block execution",
    )
    ap.add_argument(
        "--include-conversion",
        action="store_true",
        help="Include amplitude conversion costs for per-block execution estimates",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for circuit construction (set to -1 for nondeterministic)",
    )
    ap.add_argument("--out", type=str, default=None, help="If provided, save the plot to this path")
    ap.add_argument("--save-json", type=str, default=None, help="Optional path to store raw sweep results")
    args = ap.parse_args()

    if args.target_speedup <= 0:
        raise SystemExit("--target-speedup must be > 0")

    seed = None if args.seed is None or args.seed < 0 else int(args.seed)
    est = CostEstimator(
        CostParams(conv_amp_ops_factor=float(args.conv_factor), sv_twoq_factor=float(args.twoq_factor))
    )

    records: List[Dict[str, Any]] = []
    per_n_results: Dict[int, Dict[int, List[Tuple[int, Dict[str, Any]]]]] = {}

    print("n, blocks, tail_depth, block_sizes, speedup_best, speedup_sv, baseline_norm, parallel_norm")
    for n in sorted(set(args.n)):
        per_tail: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for tail_depth in sorted(set(args.tail_depths)):
            series: List[Tuple[int, Dict[str, Any]]] = []
            for blocks in sorted(set(args.blocks)):
                if blocks > n:
                    continue
                res = analyze_case(
                    n=n,
                    num_blocks=blocks,
                    tail_depth=tail_depth,
                    block_prep=args.block_prep,
                    tail_kind=args.tail_kind,
                    angle_scale=args.angle_scale,
                    sparsity=args.sparsity,
                    bandwidth=args.bandwidth,
                    est=est,
                    seed=seed,
                    include_conversion=args.include_conversion,
                    workers=max(1, int(args.parallel_workers)),
                )
                series.append((blocks, res))
                records.append(res)
                print(
                    f"{n},{blocks},{tail_depth},"
                    f"{_format_block_sizes(res['block_sizes'])},"
                    f"{res['speedup_vs_baseline']:.3f},{res['speedup_vs_sv']:.3f}"
                    f",{res['baseline_cost'] / (1 << n):.1f},{res['parallel_cost'] / (1 << n):.1f}"
                )
            if series:
                per_tail[tail_depth] = series
        per_n_results[n] = per_tail

    if args.save_json:
        payload = {
            "meta": {
                "timestamp": int(time.time()),
                "params": {
                    "n_list": sorted(set(args.n)),
                    "blocks_list": sorted(set(args.blocks)),
                    "tail_depths": sorted(set(args.tail_depths)),
                    "block_prep": args.block_prep,
                    "tail_kind": args.tail_kind,
                    "angle_scale": args.angle_scale,
                    "sparsity": args.sparsity,
                    "bandwidth": args.bandwidth,
                    "conv_factor": args.conv_factor,
                    "twoq_factor": args.twoq_factor,
                    "target_speedup": args.target_speedup,
                    "parallel_workers": args.parallel_workers,
                    "include_conversion": bool(args.include_conversion),
                    "seed": seed,
                },
            },
            "records": records,
        }
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote sweep JSON: {args.save_json}")

    num_ns = len(per_n_results)
    if num_ns == 0:
        print("No valid (n, blocks) combinations to plot")
        return

    cols = min(2, num_ns)
    rows = math.ceil(num_ns / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for ax, (n, per_tail) in zip(axes.flat, sorted(per_n_results.items())):
        for tail_depth, series in sorted(per_tail.items()):
            blocks_sorted = [b for b, _ in series]
            speeds = [res["speedup_vs_baseline"] for _, res in series]
            ax.plot(blocks_sorted, speeds, marker="o", linewidth=2, label=f"tail={tail_depth}")
        ax.set_title(f"n={n}")
        ax.set_xlabel("Number of blocks")
        ax.set_ylabel("Speedup (best baseline / QuASAr parallel)")
        ax.grid(True, alpha=0.3)
        ax.axhline(args.target_speedup, linestyle="--", linewidth=1, color="black")
        ax.legend()

    total_plots = rows * cols
    used_plots = len(per_n_results)
    for ax in axes.flat[used_plots:total_plots]:
        ax.axis("off")

    fig.suptitle(
        "Disjoint circuit speedups\n"
        f"conv={args.conv_factor}, twoq={args.twoq_factor}, prep={args.block_prep}, tail={args.tail_kind}, "
        f"workers={args.parallel_workers}, conv_cost={'on' if args.include_conversion else 'off'}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Wrote figure: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
