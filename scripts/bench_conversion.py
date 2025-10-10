"""Conversion microbenchmarks for QuASAr.

This script times conversions between tableau, decision diagram, and
statevector representations as a function of the number of qubits (and optional
circuit depth).

Examples
--------
Run the default sweep and write a JSON report:

    python scripts/bench_conversion.py --out conversion_bench.json

Benchmark for 4, 8, and 12 qubits at depths 4 and 12 with 5 repeats each and
write a CSV file:

    python scripts/bench_conversion.py --qubits 4 8 12 --depths 4 12 \
        --repeats 5 --out conversion_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quasar.conversion.dd2sv import dd_to_statevector
from quasar.conversion.tab2dd import tableau_to_dd
from quasar.conversion.tab2sv import tableau_to_statevector


ConversionName = str


DEFAULT_CONVERSIONS: Tuple[ConversionName, ...] = (
    "tableau_to_sv",
    "tableau_to_dd",
    "dd_to_sv",
)

SINGLE_Q_GATE_POOL: Tuple[str, ...] = ("h", "s", "sdg", "x", "y", "z")


@dataclass
class ConversionResult:
    """Container for the outcome of a timed conversion call."""

    elapsed: float
    value: Any
    success: bool
    error: Optional[str]


def _timed_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> ConversionResult:
    """Execute ``func`` while measuring wall-clock runtime."""

    start = perf_counter()
    try:
        value = func(*args, **kwargs)
        success = value is not None
        error: Optional[str] = None
    except Exception as exc:  # noqa: BLE001 - report arbitrary failure
        value = None
        success = False
        error = repr(exc)
    end = perf_counter()
    return ConversionResult(elapsed=end - start, value=value, success=success, error=error)


def _build_random_clifford_circuit(
    num_qubits: int,
    depth: int,
    rng: np.random.Generator,
    single_probability: float = 0.7,
    two_qubit_probability: float = 0.5,
) -> QuantumCircuit:
    """Construct a pseudo-random Clifford circuit.

    The circuit uses only Clifford generators (single-qubit Clifford gates and
    controlled-NOT) and therefore remains compatible with the tableau backend.
    """

    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive")

    circuit = QuantumCircuit(num_qubits)
    layers = max(1, depth)

    for _ in range(layers):
        # Apply single-qubit Clifford gates.
        for q in range(num_qubits):
            if rng.random() < single_probability:
                gate_name = rng.choice(SINGLE_Q_GATE_POOL)
                getattr(circuit, gate_name)(q)

        # Apply a random collection of controlled-NOT gates.
        indices = list(range(num_qubits))
        rng.shuffle(indices)
        for i in range(0, num_qubits - 1, 2):
            if rng.random() < two_qubit_probability:
                control, target = indices[i], indices[i + 1]
                if control == target:
                    continue
                # Randomly swap the role of control and target for variety.
                if rng.random() < 0.5:
                    control, target = target, control
                circuit.cx(control, target)

    return circuit


def _ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversion microbenchmark harness")
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12],
        help="List of qubit counts to benchmark",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[4, 8, 12],
        help="Circuit depths to benchmark (interpreted as Clifford layers)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per configuration",
    )
    parser.add_argument(
        "--conversions",
        type=str,
        nargs="+",
        default=list(DEFAULT_CONVERSIONS),
        choices=list(DEFAULT_CONVERSIONS),
        help="Conversions to benchmark",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="conversion_bench.json",
        help="Output file (JSON by default; CSV if the extension is .csv)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "json", "csv"],
        default="auto",
        help="Force the output format (default: infer from --out)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for circuit generation",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress information for each configuration",
    )
    return parser.parse_args(argv)


def _resolve_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".csv":
        return "csv"
    return "json"


def _write_json(path: Path, metadata: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    payload = {"metadata": metadata, "results": results}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    if not results:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    fieldnames = sorted({key for row in results for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _benchmark_configuration(
    num_qubits: int,
    depth: int,
    repeat_index: int,
    conversions: Sequence[ConversionName],
    rng: np.random.Generator,
    progress: bool,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "num_qubits": num_qubits,
        "depth": depth,
        "repeat": repeat_index,
    }

    circuit_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    local_rng = np.random.default_rng(circuit_seed)
    circuit = _build_random_clifford_circuit(num_qubits, depth, local_rng)
    clifford = Clifford(circuit)
    record["circuit_seed"] = circuit_seed
    record["circuit_depth"] = int(circuit.depth())
    record["circuit_two_qubit_ops"] = int(circuit.num_nonlocal_gates())

    if progress:
        print(
            f"[n={num_qubits:3d} depth={depth:3d} repeat={repeat_index:2d}]"
            f" Clifford generated with depth {record['circuit_depth']}"
        )

    conversions_set = set(conversions)

    # Tableau -> SV conversion
    if "tableau_to_sv" in conversions_set:
        tab_sv = _timed_call(tableau_to_statevector, clifford)
        record["tableau_to_sv_time_s"] = tab_sv.elapsed if tab_sv.success else None
        record["tableau_to_sv_success"] = tab_sv.success
        if tab_sv.error:
            record["tableau_to_sv_error"] = tab_sv.error
        if tab_sv.success and isinstance(tab_sv.value, np.ndarray):
            record["statevector_dimension"] = int(tab_sv.value.size)

    # Tableau -> DD conversion (also acts as preparation for DD -> SV).
    need_dd = "tableau_to_dd" in conversions_set or "dd_to_sv" in conversions_set
    dd_result: Optional[ConversionResult] = None
    if need_dd:
        dd_result = _timed_call(tableau_to_dd, clifford)
        prep_key = "tableau_to_dd_time_s" if "tableau_to_dd" in conversions_set else "dd_input_prep_time_s"
        record[prep_key] = dd_result.elapsed if dd_result.success else None
        success_key = "tableau_to_dd_success" if "tableau_to_dd" in conversions_set else "dd_input_prep_success"
        record[success_key] = dd_result.success
        if dd_result.error:
            error_key = (
                "tableau_to_dd_error" if "tableau_to_dd" in conversions_set else "dd_input_prep_error"
            )
            record[error_key] = dd_result.error

    # DD -> SV conversion.
    if "dd_to_sv" in conversions_set:
        dd_sv_result: Optional[ConversionResult] = None
        dd_source_label: Optional[str] = None
        dd_errors: List[str] = []

        if dd_result is not None and dd_result.success and dd_result.value is not None:
            candidate = _timed_call(dd_to_statevector, dd_result.value)
            if candidate.success:
                dd_sv_result = candidate
                dd_source_label = "dd"
            else:
                dd_errors.append(candidate.error or "DD source conversion failed")

        if dd_sv_result is None or not dd_sv_result.success:
            fallback_candidate = _timed_call(dd_to_statevector, circuit)
            if fallback_candidate.success:
                dd_sv_result = fallback_candidate
                dd_source_label = "circuit"
            else:
                dd_errors.append(fallback_candidate.error or "Circuit fallback conversion failed")
                dd_sv_result = fallback_candidate

        if dd_sv_result is None or not dd_sv_result.success:
            record["dd_to_sv_time_s"] = None
            record["dd_to_sv_success"] = False
            if dd_errors:
                record["dd_to_sv_error"] = "; ".join(dd_errors)
        else:
            record["dd_to_sv_time_s"] = dd_sv_result.elapsed
            record["dd_to_sv_success"] = True
            if dd_sv_result.error:
                record["dd_to_sv_error"] = dd_sv_result.error
            if dd_source_label == "circuit":
                record["dd_to_sv_note"] = "Used Clifford circuit fallback"
            if isinstance(dd_sv_result.value, np.ndarray):
                record.setdefault("statevector_dimension", int(dd_sv_result.value.size))

    return record


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.out)
    fmt = _resolve_format(out_path, args.format)

    rng = np.random.default_rng(args.seed)
    metadata = {
        "qubits": args.qubits,
        "depths": args.depths,
        "repeats": args.repeats,
        "conversions": list(args.conversions),
        "seed": args.seed,
    }

    results: List[Dict[str, Any]] = []
    for num_qubits in args.qubits:
        for depth in args.depths:
            for repeat in range(args.repeats):
                record = _benchmark_configuration(
                    num_qubits=num_qubits,
                    depth=depth,
                    repeat_index=repeat,
                    conversions=args.conversions,
                    rng=rng,
                    progress=args.progress,
                )
                results.append(record)

    _ensure_parent_dir(out_path)
    if fmt == "csv":
        _write_csv(out_path, results)
    else:
        _write_json(out_path, metadata, results)

    print(f"Wrote {out_path} with {len(results)} entries (format={fmt}).")


if __name__ == "__main__":
    main()
