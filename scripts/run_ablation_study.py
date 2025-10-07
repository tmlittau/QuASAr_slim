#!/usr/bin/env python3
"""Run an ablation study over QuASAr's partitioning strategies.

This helper builds a family of disjoint hybrid circuits and benchmarks three
QuASAr variants:

* **full** – standard QuASAr with disjoint partitioning and hybrid prefix/tail
  decomposition enabled.
* **no_disjoint** – QuASAr with disjoint partitioning disabled (the entire
  circuit is executed as a single partition) while hybrid decomposition remains
  available.
* **no_hybrid** – QuASAr with the hybrid prefix/tail optimisation disabled
  while keeping disjoint partitioning enabled.

For each problem size the script writes a JSON summary and produces bar charts
comparing the runtime of the three variants (in all cases normalised to the
full QuASAr baseline).

Run from the repository root, e.g.::

    python -m scripts.run_ablation_study --n 16 24 32 --num-blocks 4

The resulting artefacts are stored inside ``ablation_runs`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from benchmarks.disjoint import disjoint_preps_plus_tails
from plots.palette import EDGE_COLOR, PASTEL_COLORS, apply_paper_style
from quasar.SSD import SSD, PartitionNode
from quasar.analyzer import analyze
from quasar.baselines import run_baselines
from quasar.backends.dd import ddsim_available
from quasar.backends.sv import estimate_sv_bytes
from quasar.backends.tableau import stim_available
from quasar.cost_estimator import CostEstimator
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd


apply_paper_style()


# --------------------------------------------------------------------------------------
# Data containers and utilities


@dataclass
class VariantResult:
    name: str
    ok: bool
    wall_s: Optional[float]
    wall_estimate_s: Optional[float]
    planner_config: Dict[str, Any]
    plan: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    amp_ops: Optional[float] = None
    peak_mem_bytes: Optional[int] = None
    peak_mem_estimate_bytes: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "ok": self.ok,
            "wall_elapsed_s": self.wall_s,
            "wall_estimated_s": self.wall_estimate_s,
            "planner_config": self.planner_config,
        }
        if self.plan is not None:
            payload["plan"] = self.plan
        if self.execution is not None:
            payload["execution"] = self.execution
        if self.error is not None:
            payload["error"] = self.error
        if self.amp_ops is not None:
            payload["amp_ops"] = self.amp_ops
        if self.peak_mem_bytes is not None:
            payload["peak_mem_bytes"] = self.peak_mem_bytes
        if self.peak_mem_estimate_bytes is not None:
            payload["peak_mem_estimated_bytes"] = self.peak_mem_estimate_bytes
        return payload


def _ensure_seed(seed: Optional[int], *, index: int) -> int:
    if seed is None:
        return 7 + 17 * index
    return int(seed) + index


def _collapse_to_single_partition(
    circ,
    metrics: MutableMapping[str, Any],
    *,
    total_qubits: int,
) -> SSD:
    """Return an SSD with a single partition covering the entire circuit."""

    merged = SSD(meta={"total_qubits": int(total_qubits), "components": 1})
    node = PartitionNode(
        id=0,
        qubits=list(range(int(total_qubits))),
        circuit=circ,
        metrics=dict(metrics),
        meta={"ablation": "no_disjoint"},
    )
    merged.add(node)
    return merged


def _extract_wall_elapsed(exec_payload: Any) -> Optional[float]:
    if not isinstance(exec_payload, MutableMapping):
        return None
    meta = exec_payload.get("meta")
    if isinstance(meta, MutableMapping):
        wall = meta.get("wall_elapsed_s")
        if isinstance(wall, (int, float)):
            return float(wall)
    results = exec_payload.get("results")
    if isinstance(results, Iterable):
        total = 0.0
        found = False
        for entry in results:
            if not isinstance(entry, MutableMapping):
                continue
            elapsed = entry.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                total += float(elapsed)
                found = True
        if found:
            return total
    return None


def _execution_status(exec_payload: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(exec_payload, MutableMapping):
        return False, "no_execution_payload"
    results = exec_payload.get("results")
    if not isinstance(results, Iterable):
        return False, "no_results"
    ok_all = True
    notes: List[str] = []
    errors: List[str] = []
    for entry in results:
        if not isinstance(entry, MutableMapping):
            continue
        status_raw = entry.get("status")
        status = status_raw.lower() if isinstance(status_raw, str) else status_raw
        if status in (None, "ok"):
            continue
        if status == "estimated":
            err = entry.get("error")
            if isinstance(err, str):
                clean = err.strip()
                if clean and clean.lower() != "completed":
                    notes.append(f"p{entry.get('partition')}: {clean}")
            continue
        if status == "done":
            continue
        if status == "running":
            continue
        ok_all = False
        err = entry.get("error")
        if isinstance(err, str) and err:
            errors.append(f"p{entry.get('partition')}: {err}")
    if ok_all:
        return True, "; ".join(notes) if notes else None
    if errors:
        return False, "; ".join(errors)
    return False, "execution_failed"


def _extract_peak_memory(exec_payload: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(exec_payload, MutableMapping):
        return None, None
    results = exec_payload.get("results")
    if not isinstance(results, Iterable):
        return None, None
    measured: List[int] = []
    estimated: List[int] = []
    for entry in results:
        if not isinstance(entry, MutableMapping):
            continue
        mem = entry.get("mem_bytes")
        mem_est = entry.get("mem_bytes_estimated")
        status = entry.get("status")
        if isinstance(mem, (int, float)):
            if status in (None, "ok"):
                measured.append(int(mem))
            else:
                estimated.append(int(mem))
        if isinstance(mem_est, (int, float)):
            estimated.append(int(mem_est))
    meas_peak = max(measured) if measured else None
    est_peak = max(estimated) if estimated else None
    return meas_peak, est_peak


def _estimate_mem_for_backend(backend: str, n_qubits: int) -> int:
    b = (backend or "sv").lower()
    if b == "tableau":
        return 64 * 1024 * 1024
    if b == "dd":
        return 256 * 1024 * 1024
    return int(estimate_sv_bytes(n_qubits))


def _estimate_peak_mem_from_plan(ssd: SSD) -> Optional[int]:
    peak = 0
    for node in ssd.partitions:
        metrics = node.metrics or {}
        n = int(metrics.get("num_qubits", 0))
        need = _estimate_mem_for_backend(node.backend or "sv", n)
        peak = max(peak, need)
    return peak or None


def _estimate_amp_ops(ssd: SSD, cfg: PlannerConfig) -> float:
    est = CostEstimator.from_planner_config(cfg)
    total = 0.0
    for node in ssd.partitions:
        metrics = node.metrics or {}
        n = int(metrics.get("num_qubits", 0))
        total_gates = int(metrics.get("num_gates", 0))
        twoq = int(metrics.get("two_qubit_gates", 0))
        oneq = max(0, total_gates - twoq)
        backend = (node.backend or "sv").lower()
        if backend == "tableau":
            total += est.tableau_prefix_cost(n, oneq, twoq)
        else:
            total += est.sv_cost(n, oneq, twoq)
    return float(total)


def _validate_plan_features(plan: MutableMapping[str, Any]) -> None:
    partitions = plan.get("partitions") if isinstance(plan, MutableMapping) else None
    if not isinstance(partitions, Iterable):
        raise RuntimeError("plan does not contain a partitions list")

    backend_counts: Counter[str] = Counter()
    has_hybrid_chain = False
    max_qubits = 0
    for entry in partitions:
        if not isinstance(entry, MutableMapping):
            continue
        backend = entry.get("backend")
        if isinstance(backend, str):
            backend_counts[backend.lower()] += 1
        meta = entry.get("meta")
        if isinstance(meta, MutableMapping) and meta.get("chain_id"):
            has_hybrid_chain = True
        nq = entry.get("num_qubits")
        if isinstance(nq, (int, float)):
            max_qubits = max(max_qubits, int(nq))

    missing: List[str] = []
    if backend_counts.get("sv", 0) == 0:
        missing.append("sv")
    if stim_available() and backend_counts.get("tableau", 0) == 0:
        missing.append("tableau")
    if ddsim_available() and backend_counts.get("dd", 0) == 0:
        missing.append("dd")
    if missing:
        raise RuntimeError(
            "Expected the ablation circuit to exercise the following backends "
            f"but they were absent from the full plan: {', '.join(sorted(set(missing)))} "
            f"(largest partition has {max_qubits} qubits; consider reducing --num-blocks or"
            " increasing --n)"
        )
    if not has_hybrid_chain:
        raise RuntimeError("Expected at least one hybrid chain in the full plan")


def _run_variant(
    name: str,
    ssd: SSD,
    planner_cfg: PlannerConfig,
    exec_cfg: ExecutionConfig,
) -> VariantResult:
    try:
        planned = plan(ssd, planner_cfg)
    except Exception as exc:  # pragma: no cover - defensive guard
        return VariantResult(
            name=name,
            ok=False,
            wall_s=None,
            wall_estimate_s=None,
            planner_config=asdict(planner_cfg),
            error=f"plan_failed: {exc}",
        )

    amp_ops = _estimate_amp_ops(planned, planner_cfg)
    peak_mem_plan = _estimate_peak_mem_from_plan(planned)

    try:
        exec_payload = execute_ssd(planned, exec_cfg)
        wall = _extract_wall_elapsed(exec_payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        return VariantResult(
            name=name,
            ok=False,
            wall_s=None,
            wall_estimate_s=None,
            planner_config=asdict(planner_cfg),
            plan=planned.to_dict(),
            error=f"execute_failed: {exc}",
            amp_ops=amp_ops,
            peak_mem_bytes=None,
            peak_mem_estimate_bytes=peak_mem_plan,
        )

    ok_exec, exec_error = _execution_status(exec_payload)
    peak_measured, peak_estimated = _extract_peak_memory(exec_payload)
    if peak_estimated is None:
        peak_estimated = peak_mem_plan

    return VariantResult(
        name=name,
        ok=ok_exec,
        wall_s=wall if ok_exec else None,
        wall_estimate_s=None,
        planner_config=asdict(planner_cfg),
        plan=planned.to_dict(),
        execution=exec_payload,
        error=(exec_error if ok_exec or exec_error else "execution_failed"),
        amp_ops=amp_ops,
        peak_mem_bytes=peak_measured,
        peak_mem_estimate_bytes=peak_estimated,
    )


def _format_label(params: MutableMapping[str, Any]) -> str:
    n = int(params.get("num_qubits", 0))
    blocks = int(params.get("num_blocks", 0))
    return f"n={n}, blocks={blocks}"


def _make_runtime_plot(
    records: List[Dict[str, Any]],
    *,
    out_path: Path,
    title: str,
) -> None:
    labels = []
    rel_full, rel_nodisjoint, rel_nohybrid = [], [], []

    for rec in records:
        params = rec.get("params", {})
        variants = rec.get("variants", {})
        labels.append(_format_label(params))

        def _time(key: str) -> float:
            data = variants.get(key, {})
            if not isinstance(data, MutableMapping):
                return math.nan
            wall = data.get("wall_elapsed_s")
            if isinstance(wall, (int, float)):
                return float(wall)
            wall_est = data.get("wall_estimated_s")
            return float(wall_est) if isinstance(wall_est, (int, float)) else math.nan

        t_full = _time("full")
        t_nodisjoint = _time("no_disjoint")
        t_nohybrid = _time("no_hybrid")

        if math.isnan(t_full) or t_full == 0.0:
            rel_full.append(math.nan)
            rel_nodisjoint.append(math.nan)
            rel_nohybrid.append(math.nan)
        else:
            rel_full.append(1.0)
            rel_nodisjoint.append(
                t_nodisjoint / t_full if not math.isnan(t_nodisjoint) else math.nan
            )
            rel_nohybrid.append(
                t_nohybrid / t_full if not math.isnan(t_nohybrid) else math.nan
            )

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar(
        x - width,
        rel_full,
        width,
        label="Full QuASAr",
        color=PASTEL_COLORS.get("tableau"),
        edgecolor=EDGE_COLOR,
    )
    plt.bar(
        x,
        rel_nodisjoint,
        width,
        label="No disjoint partitioning",
        color=PASTEL_COLORS.get("sv"),
        edgecolor=EDGE_COLOR,
    )
    plt.bar(
        x + width,
        rel_nohybrid,
        width,
        label="No hybrid splitting",
        color=PASTEL_COLORS.get("dd"),
        edgecolor=EDGE_COLOR,
    )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Relative runtime vs full QuASAr")
    plt.title(title)
    plt.axhline(1.0, color=EDGE_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _make_memory_plot(
    records: List[Dict[str, Any]],
    *,
    out_path: Path,
    title: str,
) -> None:
    labels = []
    mem_full, mem_nodisjoint, mem_nohybrid = [], [], []

    for rec in records:
        params = rec.get("params", {})
        variants = rec.get("variants", {})
        labels.append(_format_label(params))

        def _mem(key: str) -> float:
            data = variants.get(key, {})
            if not isinstance(data, MutableMapping):
                return math.nan
            for candidate in ("peak_mem_bytes", "peak_mem_estimated_bytes"):
                value = data.get(candidate)
                if isinstance(value, (int, float)):
                    return float(value)
            return math.nan

        m_full = _mem("full")
        m_nd = _mem("no_disjoint")
        m_nh = _mem("no_hybrid")

        if math.isnan(m_full) or m_full == 0.0:
            mem_full.append(math.nan)
            mem_nodisjoint.append(math.nan)
            mem_nohybrid.append(math.nan)
        else:
            mem_full.append(1.0)
            mem_nodisjoint.append(m_nd / m_full if not math.isnan(m_nd) else math.nan)
            mem_nohybrid.append(m_nh / m_full if not math.isnan(m_nh) else math.nan)

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar(
        x - width,
        mem_full,
        width,
        label="Full QuASAr",
        color=PASTEL_COLORS.get("tableau"),
        edgecolor=EDGE_COLOR,
    )
    plt.bar(
        x,
        mem_nodisjoint,
        width,
        label="No disjoint partitioning",
        color=PASTEL_COLORS.get("sv"),
        edgecolor=EDGE_COLOR,
    )
    plt.bar(
        x + width,
        mem_nohybrid,
        width,
        label="No hybrid splitting",
        color=PASTEL_COLORS.get("dd"),
        edgecolor=EDGE_COLOR,
    )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Relative peak memory vs full QuASAr")
    plt.title(title)
    plt.axhline(1.0, color=EDGE_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
    values = [
        val
        for series in (mem_full, mem_nodisjoint, mem_nohybrid)
        for val in series
        if isinstance(val, (int, float)) and not math.isnan(val) and val > 0
    ]
    if values:
        vmax = max(values)
        vmin = min(values)
        if vmax / vmin >= 20:
            ax = plt.gca()
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# CLI


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ablation study on QuASAr's partitioning components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", "--num-qubits", nargs="+", type=int, dest="num_qubits", required=True)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=2,
        help="Number of disjoint blocks per circuit (use smaller values to keep each block large enough for hybrid/DD splits)",
    )
    parser.add_argument("--tail-depth", type=int, default=24, help="Depth of diagonal tails inside each block")
    parser.add_argument("--angle-scale", type=float, default=0.1, help="Rotation angle scale for diagonal tails")
    parser.add_argument("--sparsity", type=float, default=0.10, help="RZ sparsity inside diagonal tails")
    parser.add_argument("--bandwidth", type=int, default=2, help="CZ bandwidth inside diagonal tails")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for circuit construction")
    parser.add_argument("--conv-factor", type=float, default=64.0, help="Conversion amortisation factor")
    parser.add_argument("--twoq-factor", type=float, default=4.0, help="Statevector two-qubit gate factor")
    parser.add_argument("--max-ram-gb", type=float, default=64.0, help="RAM budget for planning/execution")
    parser.add_argument("--sv-ampops-per-sec", type=float, default=None, help="Override SV baseline speed (optional)")
    parser.add_argument("--out-dir", type=str, default="ablation_runs", help="Directory to store outputs")
    parser.add_argument(
        "--json-name",
        type=str,
        default="hybrid_disjoint_results.json",
        help="Name of the JSON summary file",
    )
    parser.add_argument(
        "--times-fig",
        type=str,
        default="ablation_relative_runtime.png",
        help="Filename for the relative runtime bar chart",
    )
    parser.add_argument(
        "--memory-fig",
        "--relative-fig",
        dest="memory_fig",
        type=str,
        default="ablation_relative_memory.png",
        help="Filename for the relative peak memory bar chart",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exec_cfg = ExecutionConfig(max_ram_gb=float(args.max_ram_gb))
    records: List[Dict[str, Any]] = []

    t_start = time.time()
    combos = list(enumerate(args.num_qubits, start=1))
    total = len(combos)

    for index, n in combos:
        seed = _ensure_seed(args.seed, index=index)
        params = {
            "num_qubits": int(n),
            "num_blocks": int(args.num_blocks),
            "tail_depth": int(args.tail_depth),
            "angle_scale": float(args.angle_scale),
            "sparsity": float(args.sparsity),
            "bandwidth": int(args.bandwidth),
            "seed": seed,
        }
        print(f"[{index}/{total}] Building circuit with params: {params}")

        circ = disjoint_preps_plus_tails(
            num_qubits=params["num_qubits"],
            num_blocks=params["num_blocks"],
            block_prep="ghz",
            tail_kind="hybrid",
            tail_depth=params["tail_depth"],
            angle_scale=params["angle_scale"],
            sparsity=params["sparsity"],
            bandwidth=params["bandwidth"],
            seed=params["seed"],
        )

        analysis = analyze(circ)
        planner_base = dict(
            max_ram_gb=float(args.max_ram_gb),
            conv_amp_ops_factor=float(args.conv_factor),
            sv_twoq_factor=float(args.twoq_factor),
        )

        variants: Dict[str, VariantResult] = {}

        cfg_full = PlannerConfig(**planner_base)
        variants["full"] = _run_variant("full", analysis.ssd, cfg_full, exec_cfg)
        if variants["full"].plan is not None:
            try:
                _validate_plan_features(variants["full"].plan)
            except Exception as exc:
                raise RuntimeError(
                    f"Circuit sanity check failed for params {params}: {exc}"
                ) from exc

        merged_ssd = _collapse_to_single_partition(
            circ,
            analysis.metrics_global,
            total_qubits=params["num_qubits"],
        )
        cfg_nodisjoint = PlannerConfig(**planner_base)
        variants["no_disjoint"] = _run_variant("no_disjoint", merged_ssd, cfg_nodisjoint, exec_cfg)

        cfg_nohybrid = PlannerConfig(**planner_base, hybrid_clifford_tail=False)
        variants["no_hybrid"] = _run_variant("no_hybrid", analysis.ssd, cfg_nohybrid, exec_cfg)

        amp_scale: Optional[float] = None
        for key in ("full", "no_disjoint", "no_hybrid"):
            cand = variants.get(key)
            if not cand or cand.amp_ops in (None, 0.0) or cand.wall_s in (None, 0.0):
                continue
            scale = cand.wall_s / cand.amp_ops
            if scale and math.isfinite(scale) and scale > 0:
                amp_scale = scale
                break
        if amp_scale is None:
            for cand in variants.values():
                if not cand or cand.amp_ops in (None, 0.0) or cand.wall_s in (None, 0.0):
                    continue
                scale = cand.wall_s / cand.amp_ops
                if scale and math.isfinite(scale) and scale > 0:
                    amp_scale = scale
                    break

        for cand in variants.values():
            if cand.wall_s is not None:
                cand.wall_estimate_s = cand.wall_s
            elif cand.amp_ops and amp_scale:
                cand.wall_estimate_s = cand.amp_ops * amp_scale
            else:
                cand.wall_estimate_s = None

        print(
            "    -> times (s):",
            {
                k: (
                    v.wall_s
                    if v.wall_s is not None
                    else v.wall_estimate_s
                )
                for k, v in variants.items()
            },
        )

        try:
            baselines = run_baselines(
                circ,
                which=["tableau", "sv", "dd"],
                per_partition=False,
                max_ram_gb=float(args.max_ram_gb),
                sv_ampops_per_sec=args.sv_ampops_per_sec,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            baselines = {"error": str(exc)}

        records.append(
            {
                "params": params,
                "variants": {name: result.to_json() for name, result in variants.items()},
                "baselines": baselines,
            }
        )

    summary = {
        "meta": {
            "created_at": time.time(),
            "args": vars(args),
            "elapsed_s": time.time() - t_start,
        },
        "cases": records,
    }

    json_path = out_dir / args.json_name
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote JSON summary to {json_path}")

    if records:
        times_path = out_dir / args.times_fig
        _make_runtime_plot(records, out_path=times_path, title="QuASAr ablation study: relative runtime")
        print(f"Wrote runtime bar chart to {times_path}")

        mem_path = out_dir / args.memory_fig
        _make_memory_plot(records, out_path=mem_path, title="QuASAr ablation study: relative peak memory")
        print(f"Wrote memory bar chart to {mem_path}")


if __name__ == "__main__":
    main()
