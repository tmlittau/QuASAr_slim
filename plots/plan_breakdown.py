"""Plot hybrid plan breakdowns for a single benchmark case.

This helper focuses on visualising the staged execution produced by the
hybrid planner for an arbitrary circuit from the benchmarking suites.  The
stacked bar encodes the runtime contribution of the prefix/core/suffix
partitions together with the conversion hand-off costs between them,
mirroring the paper figures regardless of circuit size.

The implementation deliberately reuses the parsing heuristics from the
``bar_hybrid`` plotter so the breakdown works against both the legacy JSON
schema and the more recent one that embeds the execution payload under the
``quasar`` key.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Reuse the conversion-time estimation logic from the hybrid stacked bars.
from .bar_hybrid import _estimate_conversion_time  # type: ignore


# Ordering and colour palette tuned to roughly match the paper figures.
COMPONENT_ORDER = [
    "Clifford prefix",
    "Prefix → core",
    "Non-Clifford core",
    "Core → suffix",
    "Sparse suffix",
]

BASELINE_LABEL = "Full circuit (statevector)"

COMPONENT_COLORS = {
    "Clifford prefix": "#ff9f1c",
    "Non-Clifford core": "#2ec4b6",
    "Sparse suffix": "#9d4edd",
    "Prefix → core": "#ffbf69",
    "Core → suffix": "#c77dff",
    BASELINE_LABEL: "#4361ee",
}


def _pick_elapsed(payload: Dict[str, Any]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None

    prefer_estimated = False
    if payload.get("ok") is False:
        prefer_estimated = True
    elif payload.get("error") and payload.get("wall_s_estimated") is not None:
        prefer_estimated = True

    if prefer_estimated:
        order = ["wall_s_estimated", "time_est_sec", "wall_s_measured", "elapsed_s"]
    else:
        order = ["elapsed_s", "wall_s_measured", "wall_s_estimated", "time_est_sec"]

    for key in order:
        value = payload.get(key)
        if value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(fval):
            continue
        return fval
    return None


def _best_whole_baseline(baselines: Any) -> Optional[Tuple[str, float]]:
    entries: Iterable[Dict[str, Any]]
    if isinstance(baselines, dict):
        entries = baselines.get("entries", []) or []
    elif isinstance(baselines, list):
        entries = baselines
    else:
        entries = []

    best: Optional[Tuple[str, float]] = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        scope = (entry.get("scope") or entry.get("mode") or "").lower()
        if scope and scope not in {"whole", "global", "circuit"}:
            if entry.get("per_partition") is not False:
                continue
        if entry.get("ok") is False:
            continue
        elapsed = _pick_elapsed(entry) or entry.get("time_est_sec")
        if elapsed is None:
            continue
        try:
            elapsed_f = float(elapsed)
        except (TypeError, ValueError):
            continue
        method = (
            entry.get("method")
            or entry.get("which")
            or entry.get("backend")
            or entry.get("name")
            or "sv"
        )
        method = str(method).lower()
        if best is None or elapsed_f < best[1]:
            best = (method, elapsed_f)
    return best


@dataclass
class _PartitionInfo:
    partition_id: int
    chain_id: str
    seq_index: int
    backend: str
    planner_reason: str
    num_qubits: int
    metrics: Dict[str, Any]


def _load_partition_info(case: Dict[str, Any]) -> Dict[int, _PartitionInfo]:
    analysis = (case.get("quasar") or {}).get("analysis") or case.get("analysis") or {}
    plan_info = analysis.get("plan") or analysis.get("ssd") or {}
    partitions = plan_info.get("qusds") or plan_info.get("partitions") or []
    info: Dict[int, _PartitionInfo] = {}
    for part in partitions:
        if not isinstance(part, dict):
            continue
        try:
            pid = int(part.get("id"))
        except (TypeError, ValueError):
            continue
        meta = part.get("meta") or {}
        metrics = part.get("metrics") or {}
        info[pid] = _PartitionInfo(
            partition_id=pid,
            chain_id=str(meta.get("chain_id", f"chain_{pid}")),
            seq_index=int(meta.get("seq_index", 0)),
            backend=str(part.get("backend") or meta.get("backend") or "sv").lower(),
            planner_reason=str(meta.get("planner_reason") or ""),
            num_qubits=int(metrics.get("num_qubits", 0) or 0),
            metrics=metrics,
        )
    return info


def _merge_execution_results(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    info = _load_partition_info(case)
    execution = (case.get("quasar") or {}).get("execution") or case.get("execution") or {}
    merged: List[Dict[str, Any]] = []
    for res in execution.get("results", []) or []:
        if not isinstance(res, dict):
            continue
        r = dict(res)
        pid = r.get("qusd_id", r.get("partition"))
        part_meta = info.get(int(pid)) if pid is not None else None
        if part_meta:
            r.setdefault("chain_id", part_meta.chain_id)
            r.setdefault("seq_index", part_meta.seq_index)
            r.setdefault("backend", part_meta.backend)
            r.setdefault("planner_reason", part_meta.planner_reason)
            r.setdefault("num_qubits", part_meta.num_qubits)
            r.setdefault("metrics", part_meta.metrics)
        else:
            r.setdefault("chain_id", f"chain_{pid}")
            r.setdefault("seq_index", 0)
        merged.append(r)
    merged.sort(key=lambda r: (str(r.get("chain_id")), int(r.get("seq_index", 0))))
    return merged


def _classify_component(res: Dict[str, Any]) -> str:
    backend = str(res.get("backend") or "sv").lower()
    metrics = res.get("metrics") or {}
    if backend == "tableau" or metrics.get("is_clifford"):
        return "Clifford prefix"
    if backend == "dd":
        return "Sparse suffix"
    return "Non-Clifford core"


def _conversion_label(prev_backend: str, curr_backend: str) -> Optional[str]:
    prev_b = str(prev_backend or "sv").lower()
    curr_b = str(curr_backend or "sv").lower()
    if prev_b == "tableau" and curr_b != "tableau":
        return "Prefix → core"
    if curr_b == "dd" and prev_b != "dd":
        return "Core → suffix"
    return None


def _extract_component_times(case: Dict[str, Any]) -> Dict[str, float]:
    results = _merge_execution_results(case)
    component_times = {label: 0.0 for label in COMPONENT_ORDER}

    chains: Dict[str, List[Dict[str, Any]]] = {}
    for res in results:
        cid = str(res.get("chain_id"))
        chains.setdefault(cid, []).append(res)
    for chain_results in chains.values():
        chain_results.sort(key=lambda r: int(r.get("seq_index", 0)))
        prev: Optional[Dict[str, Any]] = None
        for res in chain_results:
            elapsed = _pick_elapsed(res)
            if elapsed is None:
                elapsed = 0.0
            component = _classify_component(res)
            component_times[component] += elapsed
            if prev is not None:
                label = _conversion_label(prev.get("backend"), res.get("backend"))
                if label:
                    n_qubits = res.get("num_qubits") or prev.get("num_qubits") or 0
                    try:
                        n_int = int(n_qubits)
                    except (TypeError, ValueError):
                        n_int = 0
                    conv = _estimate_conversion_time(prev, res, n_int)
                    if conv and conv > 0:
                        component_times[label] += conv
            prev = res
    return component_times


def _load_cases(suite_dir: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(suite_dir))
    except FileNotFoundError:
        raise SystemExit(f"suite_dir not found: {suite_dir}")
    for fn in entries:
        if not fn.endswith(".json") or fn == "index.json":
            continue
        with open(os.path.join(suite_dir, fn), "r", encoding="utf-8") as fh:
            cases.append(json.load(fh))
    return cases


def _select_case(
    cases: List[Dict[str, Any]],
    *,
    case_kind: Optional[str],
    case_index: int,
) -> Dict[str, Any]:
    if case_kind:
        filtered = [c for c in cases if (c.get("case") or {}).get("kind") == case_kind]
        if not filtered:
            raise SystemExit(f"No case with kind '{case_kind}' found in {len(cases)} entries")
        cases = filtered
    if not cases:
        raise SystemExit("No cases found in suite directory")
    if case_index < 0 or case_index >= len(cases):
        case_index = max(0, min(case_index, len(cases) - 1))
    return cases[case_index]


def _default_title(case: Dict[str, Any]) -> str:
    case_meta = case.get("case") or {}
    kind = case_meta.get("kind", "hybrid case")
    params = case_meta.get("params") or {}
    n = params.get("num_qubits")
    if n is not None:
        return f"{kind} (n={n}) plan breakdown"
    return f"{kind} plan breakdown"


def make_plot(
    suite_dir: str,
    *,
    out: Optional[str] = None,
    title: Optional[str] = None,
    case_kind: Optional[str] = None,
    case_index: int = 0,
) -> None:
    cases = _load_cases(suite_dir)
    case = _select_case(cases, case_kind=case_kind, case_index=case_index)

    component_times = _extract_component_times(case)

    best_baseline = _best_whole_baseline((case.get("baselines")))
    baseline_time: Optional[float]
    baseline_label = BASELINE_LABEL
    if best_baseline is None:
        baseline_time = None
    else:
        method, baseline_time = best_baseline
        method = method.lower()
        if method not in {"sv", "statevector", "state_vector"}:
            baseline_label = f"Full circuit ({method})"

    plans = ["Partitioned", "Single backend"]
    x = np.arange(len(plans), dtype=float)
    width = 0.55

    bottoms = np.zeros(len(plans), dtype=float)
    plt.figure(figsize=(7.5, 3.0))

    # Partitioned bar (index 0)
    for component in COMPONENT_ORDER:
        value = component_times.get(component, 0.0)
        if value <= 0:
            continue
        plt.bar(
            x[0],
            value,
            width,
            bottom=bottoms[0],
            color=COMPONENT_COLORS.get(component, "#888888"),
            label=component,
        )
        bottoms[0] += value

    # Single backend bar (index 1)
    if baseline_time is not None:
        plt.bar(
            x[1],
            baseline_time,
            width,
            bottom=bottoms[1],
            color=COMPONENT_COLORS.get(baseline_label, COMPONENT_COLORS[BASELINE_LABEL]),
            label=baseline_label,
        )

    plt.xticks(x, plans)
    plt.ylabel("Estimated time (arb. units)")
    plt.title(title or _default_title(case))

    # Build legend without duplicates while preserving the intended order.
    seen = set()
    handles = []
    labels = []
    for component in COMPONENT_ORDER + [baseline_label]:
        if component in seen:
            continue
        if component == baseline_label and baseline_time is None:
            continue
        color = COMPONENT_COLORS.get(component, "#888888")
        handle = plt.Rectangle((0, 0), 1, 1, color=color)
        handles.append(handle)
        labels.append(component)
        seen.add(component)
    plt.legend(handles, labels, title="Component", loc="upper right", frameon=False)
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=200)
        print(f"Wrote {out}")
    else:
        plt.show()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Plot a single hybrid plan breakdown")
    ap.add_argument("--suite-dir", type=str, required=True, help="Suite directory with JSON artefacts")
    ap.add_argument("--out", type=str, default=None, help="Output image path")
    ap.add_argument("--title", type=str, default=None, help="Title override")
    ap.add_argument("--case-kind", type=str, default=None, help="Focus on a specific case kind")
    ap.add_argument("--case-index", type=int, default=0, help="Index of the case to plot when no kind is given")
    args = ap.parse_args()

    make_plot(
        args.suite_dir,
        out=args.out,
        title=args.title,
        case_kind=args.case_kind,
        case_index=args.case_index,
    )


if __name__ == "__main__":
    main()

