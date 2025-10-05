from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

from benchmarks.disjoint import disjoint_preps_plus_tails
from quasar.analyzer import analyze
from quasar.baselines import run_baselines
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd


def _build_case_params(args: argparse.Namespace, n: int, blocks: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "num_qubits": int(n),
        "num_blocks": int(blocks),
        "block_prep": args.prep,
        "tail_kind": args.tail_kind,
        "tail_depth": int(args.tail_depth),
        "angle_scale": float(args.angle_scale),
        "sparsity": float(args.sparsity),
        "bandwidth": int(args.bandwidth),
    }
    if args.seed is not None:
        params["seed"] = int(args.seed)
    return params


def _record_error(errors: List[Dict[str, Any]], stage: str, exc: BaseException) -> None:
    errors.append(
        {
            "stage": stage,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def _ensure_wall_time(exec_payload: Dict[str, Any], elapsed: float) -> None:
    meta = exec_payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        exec_payload["meta"] = meta
    meta.setdefault("wall_elapsed_s", float(elapsed))


def _write_case_record(out_dir: str, stem: str, record: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)


def run_case(
    args: argparse.Namespace,
    n: int,
    blocks: int,
) -> Tuple[str, Dict[str, Any]]:
    params = _build_case_params(args, n, blocks)
    record: Dict[str, Any] = {
        "case": {"kind": "disjoint_preps_plus_tails", "params": params},
        "planner": {
            "conv_factor": float(args.conv_factor),
            "twoq_factor": float(args.twoq_factor),
            "max_ram_gb": float(args.max_ram_gb),
        },
        "quasar": {"analysis": {}, "execution": {}},
        "baselines": {},
    }

    errors: List[Dict[str, Any]] = []
    circ = None

    try:
        circ = disjoint_preps_plus_tails(**params)
    except Exception as exc:  # pragma: no cover - circuit construction failure is unlikely
        _record_error(errors, "build_circuit", exc)

    analysis_result = None
    if circ is not None:
        try:
            analysis_result = analyze(circ)
            record["quasar"]["analysis"]["global"] = analysis_result.metrics_global
        except Exception as exc:
            _record_error(errors, "analyze", exc)
            analysis_result = None

    planned_ssd = None
    if analysis_result is not None:
        try:
            planner_cfg = PlannerConfig(
                max_ram_gb=float(args.max_ram_gb),
                conv_amp_ops_factor=float(args.conv_factor),
                sv_twoq_factor=float(args.twoq_factor),
            )
            planned_ssd = plan(analysis_result.ssd, planner_cfg)
            record["quasar"]["analysis"]["ssd"] = planned_ssd.to_dict()
        except Exception as exc:
            _record_error(errors, "plan", exc)
            planned_ssd = None
            try:
                record["quasar"]["analysis"]["ssd"] = analysis_result.ssd.to_dict()
            except Exception:  # pragma: no cover - defensive fallback
                pass

    if planned_ssd is not None:
        try:
            exec_cfg = ExecutionConfig(max_ram_gb=float(args.max_ram_gb))
            t0 = time.time()
            exec_payload = execute_ssd(planned_ssd, exec_cfg)
            elapsed = time.time() - t0
            _ensure_wall_time(exec_payload, elapsed)
            record["quasar"]["execution"] = exec_payload
        except Exception as exc:
            _record_error(errors, "execute", exc)

    if circ is not None:
        try:
            baselines = run_baselines(
                circ,
                which=["tableau", "sv", "dd"],
                per_partition=False,
                max_ram_gb=float(args.max_ram_gb),
                sv_ampops_per_sec=args.sv_ampops_per_sec,
            )
            record["baselines"] = baselines
        except Exception as exc:
            _record_error(errors, "baselines", exc)
            record["baselines"] = {"error": str(exc)}
    else:
        record["baselines"] = {"error": "circuit_not_built"}

    if errors:
        record["errors"] = errors

    stem = f"disjoint_n-{n}_k-{blocks}"
    return stem, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QuASAr + whole-circuit baselines on disjoint prep+tail circuits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, nargs="+", required=True, help="Number of qubits to sweep")
    parser.add_argument("--blocks", "--num-blocks", type=int, nargs="+", required=True, dest="blocks", help="Block counts to sweep")
    parser.add_argument("--block-prep", "--prep", type=str, default="mixed", dest="prep", help="Preparation routine kind")
    parser.add_argument("--tail-kind", type=str, default="mixed", help="Tail circuit kind")
    parser.add_argument("--tail-depth", type=int, default=20, help="Tail depth layers")
    parser.add_argument("--angle-scale", type=float, default=0.1, help="Tail rotation angle scale")
    parser.add_argument("--sparsity", type=float, default=0.05, help="Tail sparsity for diagonal layers")
    parser.add_argument("--bandwidth", type=int, default=2, help="Tail bandwidth for diagonal layers")
    parser.add_argument("--out-dir", type=str, default="suite_disjoint", help="Output directory for JSON records")
    parser.add_argument("--conv-factor", type=float, default=64.0, help="Conversion amortization factor")
    parser.add_argument("--twoq-factor", type=float, default=4.0, help="Statevector two-qubit gate factor")
    parser.add_argument("--max-ram-gb", type=float, default=64.0, help="RAM budget for planning/execution")
    parser.add_argument("--sv-ampops-per-sec", type=float, default=None, help="Override SV amp-ops/sec baseline speed")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for circuit construction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = [(n, k) for n in args.n for k in args.blocks]
    total = len(combos)
    if total == 0:
        print("No cases to run.")
        return

    for idx, (n, blocks) in enumerate(combos, start=1):
        print(f"[{idx}/{total}] n={n} blocks={blocks}")
        stem, record = run_case(args, n, blocks)
        if "errors" in record:
            first_error = record["errors"][0]
            print(
                f"    -> encountered error at stage {first_error.get('stage')}: {first_error.get('message')}"
            )
        _write_case_record(args.out_dir, stem, record)


if __name__ == "__main__":
    main()
