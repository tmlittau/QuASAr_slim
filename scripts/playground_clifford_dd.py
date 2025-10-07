from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.hybrid import (
    clifford_prefix_rot_tail,
    sparse_clifford_prefix_sparse_tail,
)
from quasar.cost_estimator import CostEstimator, CostParams
from quasar.gate_metrics import circuit_metrics, gate_name, CLIFFORD_GATES

try:
    from qiskit.circuit import CircuitInstruction
except Exception:  # pragma: no cover - optional dependency guard
    CircuitInstruction = None  # type: ignore[assignment]


def _as_operation_tuple(inst: Any) -> Tuple[Any, Tuple[Any, ...], Tuple[Any, ...]]:
    """Return ``(operation, qargs, cargs)`` without touching legacy tuple access."""

    if CircuitInstruction is not None and isinstance(inst, CircuitInstruction):
        return inst.operation, tuple(inst.qubits), tuple(inst.clbits)
    operation, qargs, cargs = inst
    return operation, tuple(qargs), tuple(cargs)


def _split_at_first_nonclifford(qc) -> Optional[int]:
    for idx, inst in enumerate(qc.data):
        operation, _, _ = _as_operation_tuple(inst)
        if gate_name(operation) not in CLIFFORD_GATES:
            return idx
    return None


def _build_subcircuit_like(parent, ops: List[Any]):
    from qiskit import QuantumCircuit

    sub = QuantumCircuit(parent.num_qubits)
    for inst in ops:
        operation, qargs, cargs = _as_operation_tuple(inst)
        local_qargs = [sub.qubits[parent.find_bit(q).index] for q in qargs]
        sub.append(operation, local_qargs, cargs)
    sub.metadata = dict(getattr(parent, "metadata", {}) or {})
    return sub


def analyze_case(
    *,
    n: int,
    depth: int,
    cutoff: float,
    angle_scale: float,
    estimator: CostEstimator,
    prefix_sparsity_min: float,
    tail_sparsity_min: float,
    circuit_builder: Callable[..., Any] = sparse_clifford_prefix_sparse_tail,
) -> Dict[str, Any]:
    qc = circuit_builder(
        num_qubits=n, depth=depth, cutoff=cutoff, angle_scale=angle_scale, seed=42
    )
    split_idx = _split_at_first_nonclifford(qc)
    if split_idx is None:
        return {
            "feasible": False,
            "reason": "no_nonclifford_tail",
        }

    pre_ops = qc.data[:split_idx]
    tail_ops = qc.data[split_idx:]
    prefix = _build_subcircuit_like(qc, pre_ops)
    tail = _build_subcircuit_like(qc, tail_ops)
    prefix_metrics = circuit_metrics(prefix)
    tail_metrics = circuit_metrics(tail)

    cmp = estimator.compare_clifford_prefix_dd_tail(
        n=n, prefix_metrics=prefix_metrics, tail_metrics=tail_metrics
    )
    norm = 1 << n
    dd_total_norm = cmp["dd_total"] / norm
    hybrid_norm = cmp["hybrid_total"] / norm
    speedup = (
        cmp["dd_total"] / cmp["hybrid_total"]
        if cmp["hybrid_total"] > 0
        else float("inf")
    )

    prefix_sparse_ok = prefix_metrics["sparsity"] >= prefix_sparsity_min
    tail_sparse_ok = tail_metrics["sparsity"] >= tail_sparsity_min
    feasible = bool(cmp["hybrid_better"] and prefix_sparse_ok and tail_sparse_ok)

    return {
        "feasible": feasible,
        "dd_total_norm": dd_total_norm,
        "hybrid_norm": hybrid_norm,
        "speedup_vs_dd": speedup,
        "prefix_sparsity": prefix_metrics["sparsity"],
        "tail_sparsity": tail_metrics["sparsity"],
        "prefix_sparse_ok": prefix_sparse_ok,
        "tail_sparse_ok": tail_sparse_ok,
        "split_index": split_idx,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Playground for tableau→DD hybrids: sweep depths for random circuits "
            "with a Clifford prefix and report estimated speedups."
        )
    )
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12], help="Qubit counts to sweep.")
    ap.add_argument("--min-depth", type=int, default=20, help="Minimum total depth (inclusive).")
    ap.add_argument("--max-depth", type=int, default=200, help="Maximum total depth (inclusive).")
    ap.add_argument("--step", type=int, default=1, help="Depth step size.")
    ap.add_argument(
        "--cutoff",
        type=float,
        nargs="+",
        default=[0.9],
        help="Clifford prefix fraction (e.g. 0.8 = 80% prefix).",
    )
    ap.add_argument("--angle-scale", type=float, default=0.05)
    ap.add_argument("--conv-factor", type=float, default=32.0)
    ap.add_argument("--twoq-factor", type=float, default=2.0)
    ap.add_argument("--prefix-sparsity-min", type=float, default=0.05)
    ap.add_argument("--tail-sparsity-min", type=float, default=0.05)
    ap.add_argument(
        "--target-speedup",
        type=float,
        default=1.0,
        help="Reference speedup line (DD_cost / Hybrid_cost).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="If set, save the generated plot to this path instead of displaying it.",
    )
    ap.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="If set, store the sweep data as JSON for later analysis.",
    )
    ap.add_argument(
        "--dense-tail",
        action="store_true",
        help=(
            "Use the legacy dense rotation tail generator instead of the sparse "
            "DD-oriented circuit builder."
        ),
    )
    args = ap.parse_args()

    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise SystemExit("--max-depth must be >= --min-depth and both >= 1.")
    if args.step < 1:
        raise SystemExit("--step must be >= 1")

    depths = list(range(int(args.min_depth), int(args.max_depth) + 1, int(args.step)))
    estimator = CostEstimator(
        CostParams(
            conv_amp_ops_factor=args.conv_factor,
            sv_twoq_factor=args.twoq_factor,
        )
    )

    records = []
    print(
        "n,cutoff,depth,feasible,speedup,dd_total_norm,hybrid_norm," "prefix_sparsity,tail_sparsity"
    )
    circuit_builder: Callable[..., Any]
    circuit_kind: str
    if args.dense_tail:
        circuit_builder = clifford_prefix_rot_tail
        circuit_kind = "clifford_prefix_rot_tail"
    else:
        circuit_builder = sparse_clifford_prefix_sparse_tail
        circuit_kind = "sparse_clifford_prefix_sparse_tail"

    for cutoff in args.cutoff:
        for n in args.n:
            series_speedups: List[float] = []
            series_depths: List[int] = []
            feasible_mask: List[bool] = []
            first_depth: Optional[int] = None
            first_dd_total_norm: Optional[float] = None
            first_hybrid_norm: Optional[float] = None
            first_speedup: Optional[float] = None
            first_prefix_sparsity: Optional[float] = None
            first_tail_sparsity: Optional[float] = None
            for depth in depths:
                res = analyze_case(
                    n=n,
                    depth=depth,
                    cutoff=cutoff,
                    angle_scale=args.angle_scale,
                    estimator=estimator,
                    prefix_sparsity_min=args.prefix_sparsity_min,
                    tail_sparsity_min=args.tail_sparsity_min,
                    circuit_builder=circuit_builder,
                )
                feasible = res.get("feasible", False)
                speedup = res.get("speedup_vs_dd", float("nan"))
                if not feasible:
                    series_speedups.append(float("nan"))
                else:
                    series_speedups.append(speedup)
                feasible_mask.append(bool(feasible))
                series_depths.append(depth)
                dd_norm = res.get("dd_total_norm")
                hyb_norm = res.get("hybrid_norm")
                pref_sp = res.get("prefix_sparsity")
                tail_sp = res.get("tail_sparsity")
                speedup_str = f"{speedup:.3f}" if feasible else ""
                dd_str = "" if dd_norm is None else f"{dd_norm:.3f}"
                hyb_str = "" if hyb_norm is None else f"{hyb_norm:.3f}"
                pref_str = "" if pref_sp is None else f"{pref_sp:.3f}"
                tail_str = "" if tail_sp is None else f"{tail_sp:.3f}"
                print(
                    f"{n},{cutoff},{depth},{int(feasible)},{speedup_str},"
                    f"{dd_str},{hyb_str},{pref_str},{tail_str}"
                )
                if feasible and first_depth is None:
                    first_depth = depth
                    first_dd_total_norm = dd_norm
                    first_hybrid_norm = hyb_norm
                    first_speedup = speedup
                    first_prefix_sparsity = pref_sp
                    first_tail_sparsity = tail_sp
            records.append(
                {
                    "n": n,
                    "cutoff": cutoff,
                    "depths": series_depths,
                    "speedups": series_speedups,
                    "feasible": feasible_mask,
                    "first_depth": first_depth,
                    "first_dd_total_norm": first_dd_total_norm,
                    "first_hybrid_norm": first_hybrid_norm,
                    "first_speedup": first_speedup,
                    "first_prefix_sparsity": first_prefix_sparsity,
                    "first_tail_sparsity": first_tail_sparsity,
                }
            )

    if args.save_json:
        payload = {
            "meta": {
                "timestamp": int(time.time()),
                "params": {
                    "n_list": args.n,
                    "min_depth": args.min_depth,
                    "max_depth": args.max_depth,
                    "step": args.step,
                    "cutoff_list": args.cutoff,
                    "angle_scale": args.angle_scale,
                    "conv_factor": args.conv_factor,
                    "twoq_factor": args.twoq_factor,
                    "prefix_sparsity_min": args.prefix_sparsity_min,
                    "tail_sparsity_min": args.tail_sparsity_min,
                    "target_speedup": args.target_speedup,
                    "circuit_kind": circuit_kind,
                },
            },
            "records": records,
        }
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote sweep JSON: {args.save_json}")

    plt.figure(figsize=(9, 5))
    for rec in records:
        label = f"n={rec['n']}, cutoff={float(rec['cutoff']):.2f}"
        plt.plot(rec["depths"], rec["speedups"], marker="o", linewidth=1.5, label=label)
    if args.target_speedup and args.target_speedup > 0:
        plt.axhline(args.target_speedup, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Depth (layers)")
    plt.ylabel("Speedup (DD_cost / Hybrid_cost)")
    plt.title(
        "Tableau→DD hybrid speedup vs depth\n"
        f"conv={args.conv_factor}, twoq={args.twoq_factor}, angle_scale={args.angle_scale},"
        f" prefix≥{args.prefix_sparsity_min:.2f}, tail≥{args.tail_sparsity_min:.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Wrote figure: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
