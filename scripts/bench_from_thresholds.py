
from __future__ import annotations

import argparse
import json
import logging
import os
from time import perf_counter
from typing import Any, Dict, List, Optional

from benchmarks.hybrid import (
    clifford_prefix_rot_tail,
    sparse_clifford_prefix_sparse_tail,
)
from quasar.analyzer import analyze
from quasar.baselines import run_baselines
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_plan

from plots.palette import apply_paper_style


apply_paper_style()

def load_thresholds(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "records" not in data:
        raise SystemExit("Invalid thresholds JSON: missing 'records'")
    return data

def pick_meta_or_default(meta: Dict[str, Any], key: str, default):
    params = meta.get("params", {}) if isinstance(meta, dict) else {}
    val = params.get(key)
    return val if val is not None else default

def run_from_thresholds(
    thr_json: Dict[str, Any],
    *,
    cutoff: Optional[float],
    out_dir: str,
    angle_scale: Optional[float],
    conv_factor: Optional[float],
    twoq_factor: Optional[float],
    max_ram_gb: float,
    sv_ampops_per_sec: Optional[float],
    baseline_timeout_s: Optional[float],
    use_sparse_tail: bool,
    log: logging.Logger,
) -> None:
    meta = thr_json.get("meta", {})
    if cutoff is None:
        cutoffs = sorted({float(r["cutoff"]) for r in thr_json["records"] if r.get("cutoff") is not None})
        if len(cutoffs) != 1:
            raise SystemExit(f"--cutoff not provided and thresholds include {len(cutoffs)} cutoffs; please specify one of {cutoffs}")
        cutoff = cutoffs[0]

    angle_scale = float(angle_scale if angle_scale is not None else pick_meta_or_default(meta, "angle_scale", 0.1))
    conv_factor = float(conv_factor if conv_factor is not None else pick_meta_or_default(meta, "conv_factor", 64.0))
    twoq_factor = float(twoq_factor if twoq_factor is not None else pick_meta_or_default(meta, "twoq_factor", 4.0))

    recs = [r for r in thr_json["records"] if float(r.get("cutoff", -1)) == float(cutoff) and r.get("first_depth")]
    if not recs:
        raise SystemExit("No records with first_depth available for the selected cutoff")

    os.makedirs(out_dir, exist_ok=True)

    for r in sorted(recs, key=lambda x: int(x["n"])):
        n = int(r["n"])
        depth = int(r["first_depth"])
        log.info("Running threshold case: n=%d depth=%d cutoff=%.2f", n, depth, cutoff)

        build_start = perf_counter()
        builder = (
            sparse_clifford_prefix_sparse_tail if use_sparse_tail else clifford_prefix_rot_tail
        )
        kind = (
            "sparse_clifford_prefix_sparse_tail"
            if use_sparse_tail
            else "clifford_prefix_rot_tail"
        )
        circ = builder(
            num_qubits=n,
            depth=depth,
            cutoff=float(cutoff),
            angle_scale=angle_scale,
            seed=42,
        )
        log.info("Constructed circuit in %.2fs", perf_counter() - build_start)

        log.info("Analyzing circuit (n=%d, depth=%d)", n, depth)
        analyze_start = perf_counter()
        a = analyze(circ)
        log.info("Analyze completed in %.2fs", perf_counter() - analyze_start)

        cfg = PlannerConfig(
            max_ram_gb=max_ram_gb,
            conv_amp_ops_factor=conv_factor,
            sv_twoq_factor=twoq_factor,
            prefer_dd=bool(use_sparse_tail),
        )
        log.info(
            "Planning QuSD execution (max_ram_gb=%.1f, conv=%.2f, twoq=%.2f)",
            max_ram_gb,
            conv_factor,
            twoq_factor,
        )
        plan_start = perf_counter()
        planned = plan(a.plan, cfg)
        log.info("Plan completed in %.2fs", perf_counter() - plan_start)
        partition_summaries: List[str] = []
        backend_groups: Dict[str, List[str]] = {}
        backend_details: Dict[str, List[str]] = {}
        for node in planned.qusds:
            backend = node.backend or "unassigned"
            backend_groups.setdefault(backend, []).append(str(node.id))
            node_meta = node.meta if isinstance(node.meta, dict) else {}
            if hasattr(node.circuit, "data"):
                try:
                    gate_count = len(node.circuit.data)
                except TypeError:
                    gate_count = None
            elif hasattr(node.circuit, "size") and callable(getattr(node.circuit, "size")):
                gate_count = node.circuit.size()
            else:
                gate_count = None

            chain_id = node_meta.get("chain_id") if node_meta else None
            seq_index = node_meta.get("seq_index") if node_meta else None
            chain_bits: List[str] = []
            if chain_id is not None:
                chain_bits.append(f"chain={chain_id}")
            if seq_index is not None:
                chain_bits.append(f"seq={seq_index}")
            chain_info = ", ".join(chain_bits) if chain_bits else "chain=?"

            if gate_count is None:
                gate_info = "gates=?"
            else:
                gate_info = f"gates={gate_count}"

            partition_summaries.append(
                f"id={node.id} backend={backend} {chain_info} {gate_info}"
            )
            planner_reason = node_meta.get("planner_reason") if node_meta else None
            if planner_reason:
                backend_details.setdefault(backend, []).append(
                    f"id={node.id} reason={planner_reason}"
                )

        log.info("Planner produced %d QuSDs", len(planned.qusds))
        for summary in partition_summaries:
            log.info("  %s", summary)
        for backend, ids in backend_groups.items():
            log.info("Partitions using %s: %s", backend, ", ".join(ids))
            if backend in backend_details:
                for detail in backend_details[backend]:
                    log.info("    %s", detail)
        log.info("Executing plan (max_ram_gb=%.1f)", max_ram_gb)
        exec_start = perf_counter()
        exec_payload = execute_plan(planned, ExecutionConfig(max_ram_gb=max_ram_gb))
        exec_elapsed = perf_counter() - exec_start
        log.info("Execution completed in %.2fs", exec_elapsed)

        if planned.qusds:
            tail_backend = planned.qusds[-1].backend or "unassigned"
            log.info("Selecting baseline backend to match tail partition: %s", tail_backend)
        else:
            tail_backend = None

        baseline_backends: List[str]
        if tail_backend in {"sv", "statevector"}:
            baseline_backends = ["sv"]
        elif tail_backend in {"dd", "decision_diagram"}:
            baseline_backends = ["dd"]
        elif tail_backend in {"tableau"}:
            baseline_backends = ["tableau"]
        else:
            baseline_backends = ["tableau", "sv", "dd"]

        timeout_desc = (
            f"{baseline_timeout_s:.1f}s"
            if baseline_timeout_s is not None and baseline_timeout_s > 0
            else "disabled"
        )
        log.info(
            "Running baselines with timeout %s (selected: %s)",
            timeout_desc,
            ", ".join(baseline_backends),
        )
        baseline_start = perf_counter()
        bl = run_baselines(
            circ,
            which=baseline_backends,
            per_partition=False,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=baseline_timeout_s,
            log=log,
        )
        baseline_elapsed = perf_counter() - baseline_start
        log.info("Baselines completed in %.2fs", baseline_elapsed)
        for entry in bl.get("entries", []):
            res = entry.get("result", {})
            wall = res.get("wall_s_measured", res.get("elapsed_s"))
            log.info(
                "  baseline=%s ok=%s wall_s=%s error=%s",
                entry.get("which"),
                res.get("ok"),
                None if wall is None else f"{wall:.2f}",
                res.get("error"),
            )

        rec_out = {
            "case": {
                "kind": kind,
                "params": {
                    "num_qubits": n,
                    "depth": depth,
                    "cutoff": cutoff,
                    "angle_scale": angle_scale,
                },
            },
            "planner": {"conv_factor": conv_factor, "twoq_factor": twoq_factor},
            "quasar": {"wall_elapsed_s": exec_payload.get("meta", {}).get("wall_elapsed_s", None),
                       "execution": exec_payload,
                       "analysis": {"global": a.metrics_global, "plan": planned.to_dict()}},
            "baselines": bl,
        }
        stem = f"{kind}_n-{n}_d-{depth}_cut-{cutoff}"
        with open(os.path.join(out_dir, stem + ".json"), "w") as f:
            json.dump(rec_out, f, indent=2)

    try:
        from plots.bar_clifford_tail import make_plot

        plot_path = os.path.join(out_dir, "bars_from_thresholds.png")
        make_plot(
            out_dir,
            out=plot_path,
            title=f"Threshold bars (cutoff={cutoff}, conv={conv_factor}, twoq={twoq_factor})",
        )
        log.info("Wrote bar chart: %s", plot_path)
    except Exception as exc:
        log.warning(
            "Plot generation failed: %s. You can run: python plots/bar_clifford_tail.py --suite-dir %s --out bars_from_thresholds.png",
            exc,
            out_dir,
        )

def main():
    ap = argparse.ArgumentParser(description="Run benchmark cases derived from saved thresholds and plot bars.")
    ap.add_argument("--thresholds", type=str, required=True, help="Path to thresholds JSON saved by playground_cutoff.py --save-json")
    ap.add_argument("--cutoff", type=float, default=None, help="Cutoff to use from the thresholds JSON (needed if multiple cutoffs present).")
    ap.add_argument("--out-dir", type=str, default="suite_from_thresholds")
    ap.add_argument("--angle-scale", type=float, default=None, help="Override angle_scale (default: use value from thresholds meta or 0.1)")
    ap.add_argument("--conv-factor", type=float, default=None, help="Override conv_factor (default: use thresholds meta or 64.0)")
    ap.add_argument("--twoq-factor", type=float, default=None, help="Override twoq_factor (default: use thresholds meta or 4.0)")
    ap.add_argument("--max-ram-gb", type=float, default=64.0)
    ap.add_argument("--sv-ampops-per-sec", type=float, default=None)
    ap.add_argument("--baseline-timeout-s", type=float, default=None,
                    help="Timeout in seconds for each baseline backend (default: no timeout)")
    ap.add_argument(
        "--dd",
        action="store_true",
        help="Use a sparse hybrid circuit variant with a decision-diagram-friendly tail.",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("bench_from_thresholds")

    thr = load_thresholds(args.thresholds)
    run_from_thresholds(
        thr,
        cutoff=args.cutoff,
        out_dir=args.out_dir,
        angle_scale=args.angle_scale,
        conv_factor=args.conv_factor,
        twoq_factor=args.twoq_factor,
        max_ram_gb=args.max_ram_gb,
        sv_ampops_per_sec=args.sv_ampops_per_sec,
        baseline_timeout_s=args.baseline_timeout_s,
        use_sparse_tail=bool(args.dd),
        log=log,
    )

if __name__ == "__main__":
    main()
