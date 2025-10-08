from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .palette import EDGE_COLOR, FALLBACK_COLOR, PASTEL_COLORS

QUASAR_COLOR = PASTEL_COLORS["tableau"]
BASELINE_COLORS = {
    "sv": PASTEL_COLORS["sv"],
    "statevector": PASTEL_COLORS["sv"],
    "state_vector": PASTEL_COLORS["sv"],
    "dd": PASTEL_COLORS["dd"],
    "decisiondiagram": PASTEL_COLORS["dd"],
    "decision_diagram": PASTEL_COLORS["dd"],
    "tableau": PASTEL_COLORS["tableau"],
}

DEFAULT_CASE_KIND = "disjoint_preps_plus_tails"
BACKEND_ALIGNED_CASE_KIND = "disjoint_preps_plus_tails_backend_aligned"


def _normalize_case_kinds(case_kinds: Optional[Iterable[str]]) -> Iterable[str]:
    if case_kinds is None:
        return (DEFAULT_CASE_KIND,)
    return tuple({str(kind) for kind in case_kinds})


def _iter_case_files(suite_dir: str) -> Iterable[str]:
    try:
        entries = sorted(os.listdir(suite_dir))
    except FileNotFoundError:
        raise SystemExit(f"suite_dir not found: {suite_dir}")
    for fn in entries:
        if not fn.endswith(".json") or fn == "index.json":
            continue
        yield os.path.join(suite_dir, fn)


def _load_cases(
    suite_dir: str, case_kinds: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    allowed_kinds = set(_normalize_case_kinds(case_kinds))
    cases: List[Dict[str, Any]] = []
    for path in _iter_case_files(suite_dir):
        with open(path, "r") as f:
            data = json.load(f)
        case = (data.get("case") or {})
        if case.get("kind") in allowed_kinds:
            cases.append(data)
    def sort_key(d: Dict[str, Any]) -> Tuple[int, int]:
        params = (d.get("case") or {}).get("params", {})
        n = int(params.get("num_qubits", 0) or 0)
        k = int(params.get("num_blocks", 0) or 0)
        return (n, k)
    cases.sort(key=sort_key)
    return cases


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _aggregate_partition_time(execution: Dict[str, Any]) -> Optional[float]:
    results = execution.get("results")
    if not isinstance(results, list):
        return None

    chain_totals: Dict[str, float] = {}
    for res in results:
        if not isinstance(res, dict):
            continue
        elapsed = _safe_float(res.get("elapsed_s"))
        if elapsed is None:
            elapsed = _safe_float(res.get("wall_s_measured"))
        if elapsed is None:
            continue
        chain_id = str(res.get("chain_id") or "__default_chain__")
        chain_totals[chain_id] = chain_totals.get(chain_id, 0.0) + elapsed

    if not chain_totals:
        return None

    return max(chain_totals.values())


def _extract_quasar_time(data: Dict[str, Any]) -> Optional[float]:
    quasar = (data.get("quasar") or {})
    execution = quasar.get("execution") or data.get("execution") or {}
    meta = execution.get("meta") or {}
    max_workers = meta.get("max_workers", 1.0)

    aggregated = _aggregate_partition_time(execution)
    wall_meta = _safe_float(meta.get("wall_elapsed_s"))

    if aggregated is not None:
        if wall_meta is None:
            return aggregated
        return min(aggregated, wall_meta)

    return wall_meta


def _pick_elapsed(payload: Dict[str, Any]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None

    prefer_estimated = False
    if payload.get("ok") is False:
        prefer_estimated = True
    elif payload.get("error") and payload.get("wall_s_estimated") is not None:
        prefer_estimated = True

    key_orders: List[str]
    if prefer_estimated:
        key_orders = ["wall_s_estimated", "time_est_sec", "wall_s_measured", "elapsed_s"]
    else:
        key_orders = ["wall_s_measured", "elapsed_s", "wall_s_estimated", "time_est_sec"]

    for key in key_orders:
        if key not in payload or payload[key] is None:
            continue
        try:
            return float(payload[key])
        except Exception:
            continue
    return None


def _is_whole_baseline(entry: Dict[str, Any]) -> bool:
    if (entry or {}).get("ok") is False and _pick_elapsed(entry) is None:
        return False
    scope = (entry or {}).get("scope")
    if scope in {"whole", "global", "circuit"}:
        return True
    if entry.get("per_partition") is False:
        return True
    if "partition_id" not in entry and "chain_id" not in entry:
        return True
    return False


def _flatten_baseline_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    flat = dict(entry)
    res = flat.pop("result", None)
    if isinstance(res, dict):
        for key, val in res.items():
            flat.setdefault(key, val)
    if "method" not in flat and "which" in flat:
        flat["method"] = flat.get("which")
    if "scope" not in flat and "mode" in flat:
        flat["scope"] = flat.get("mode")
    if "wall_s_measured" not in flat and "elapsed_s" in flat:
        try:
            flat["wall_s_measured"] = float(flat.get("elapsed_s"))
        except Exception:
            pass
    return flat


def _best_whole_baseline(
    baselines: Dict[str, Any]
) -> Optional[Tuple[str, float, Dict[str, Any]]]:
    entries = baselines.get("entries") if isinstance(baselines, dict) else None
    if not isinstance(entries, list):
        entries = []
    entries = [_flatten_baseline_entry(e) for e in entries]
    best: Optional[Tuple[str, float, Dict[str, Any]]] = None
    for entry in entries:
        if not _is_whole_baseline(entry):
            continue
        elapsed = _pick_elapsed(entry)
        if elapsed is None:
            continue
        method = (
            entry.get("method")
            or entry.get("which")
            or entry.get("backend")
            or entry.get("name")
            or "sv"
        ).lower()
        if best is None or elapsed < best[1]:
            best = (method, elapsed, entry)
    if best is not None:
        return best
    for entry in entries:
        if entry.get("ok") is False and _pick_elapsed(entry) is None:
            continue
        elapsed = _pick_elapsed(entry)
        if elapsed is None:
            continue
        method = (
            entry.get("method")
            or entry.get("which")
            or entry.get("backend")
            or entry.get("name")
            or "sv"
        ).lower()
        if best is None or elapsed < best[1]:
            best = (method, elapsed, entry)
    return best


def _bytes_to_gib(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    return float(value) / (1024.0 ** 3)


def _extract_quasar_memory_bytes(data: Dict[str, Any]) -> Optional[float]:
    quasar = (data.get("quasar") or {})
    execution = quasar.get("execution") or data.get("execution") or {}
    results = execution.get("results")
    if isinstance(results, list):
        best: Optional[float] = None
        for res in results:
            if not isinstance(res, dict):
                continue
            mem = _safe_float(res.get("mem_bytes"))
            if mem is None:
                continue
            if best is None or mem > best:
                best = mem
        if best is not None:
            return best
    planner = (quasar.get("analysis") or {}).get("ssd") if isinstance(quasar, dict) else None
    if isinstance(planner, dict):
        parts = planner.get("partitions")
        if isinstance(parts, list):
            best: Optional[float] = None
            for part in parts:
                mem = _safe_float((part or {}).get("mem_bytes"))
                if mem is None:
                    continue
                if best is None or mem > best:
                    best = mem
            if best is not None:
                return best
    return None


def _estimate_sv_bytes(num_qubits: int) -> int:
    if num_qubits <= 0:
        return 0
    return 16 * (1 << int(num_qubits))


def _extract_baseline_memory_bytes(
    entry: Optional[Dict[str, Any]],
    fallback_qubits: int,
) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    mem = entry.get("mem_bytes")
    if mem is None:
        mem = entry.get("mem_bytes_estimated")
    if mem is None:
        estimate = entry.get("estimate")
        if isinstance(estimate, dict):
            mem = estimate.get("mem_bytes")
    if mem is None:
        method = (
            entry.get("method")
            or entry.get("which")
            or entry.get("backend")
            or entry.get("name")
            or "sv"
        ).lower()
        if method in {"sv", "statevector", "state_vector"}:
            mem = _estimate_sv_bytes(fallback_qubits)
    return _safe_float(mem)


def _collect_case_metrics(cases: List[Dict[str, Any]]):
    labels: List[str] = []
    quasar_times: List[float] = []
    baseline_times: List[float] = []
    baseline_methods: List[str] = []
    baseline_entries: List[Optional[Dict[str, Any]]] = []
    quasar_mems: List[float] = []
    baseline_mems: List[float] = []

    for data in cases:
        params = (data.get("case") or {}).get("params", {})
        n = int(params.get("num_qubits", 0) or 0)
        k = int(params.get("num_blocks", 0) or 0)
        labels.append(f"n={n}, blocks={k}")

        qt = _extract_quasar_time(data)
        quasar_times.append(qt if qt is not None else float("nan"))

        best = _best_whole_baseline(data.get("baselines", {}))
        if best is None:
            baseline_methods.append("sv")
            baseline_times.append(float("nan"))
            baseline_entries.append(None)
        else:
            method, elapsed, entry = best
            baseline_methods.append(method)
            baseline_times.append(elapsed)
            baseline_entries.append(entry)

        qm = _extract_quasar_memory_bytes(data)
        quasar_mems.append(qm if qm is not None else float("nan"))

        bm = _extract_baseline_memory_bytes(baseline_entries[-1], n)
        baseline_mems.append(bm if bm is not None else float("nan"))

    return {
        "labels": labels,
        "quasar_times": quasar_times,
        "baseline_times": baseline_times,
        "baseline_methods": baseline_methods,
        "baseline_entries": baseline_entries,
        "quasar_mems": quasar_mems,
        "baseline_mems": baseline_mems,
    }


def make_plot(
    suite_dir: str,
    out: Optional[str] = None,
    title: Optional[str] = None,
    *,
    case_kinds: Optional[Iterable[str]] = None,
) -> None:
    cases = _load_cases(suite_dir, case_kinds)
    if not cases:
        raise SystemExit("No matching cases found in suite_dir")

    metrics = _collect_case_metrics(cases)
    labels = metrics["labels"]
    quasar_times = metrics["quasar_times"]
    baseline_times = metrics["baseline_times"]
    baseline_methods = metrics["baseline_methods"]

    count = len(labels)
    if count == 0:
        raise SystemExit("No cases available for plotting")

    width = 0.6
    fig, axes = plt.subplots(1, count, figsize=(max(6, count * 3.6), 5.5), sharey=True)
    if count == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        quasar_time = quasar_times[idx]
        baseline_time = baseline_times[idx]
        baseline_method = baseline_methods[idx]

        if np.isfinite(quasar_time) and quasar_time > 0:
            ax.bar(
                0,
                quasar_time,
                width,
                color=QUASAR_COLOR,
                edgecolor=EDGE_COLOR,
            )

        if np.isfinite(baseline_time) and baseline_time > 0:
            ax.bar(
                1,
                baseline_time,
                width,
                color=BASELINE_COLORS.get(baseline_method, FALLBACK_COLOR),
                edgecolor=EDGE_COLOR,
                alpha=0.9,
            )

            if np.isfinite(quasar_time) and quasar_time > 0:
                speedup = baseline_time / quasar_time
                y_offset = 0.05 * max(quasar_time, baseline_time)
                ax.text(
                    0,
                    quasar_time + y_offset,
                    f"{speedup:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["QuASAr", "Baseline"])
        ax.set_xlim(-0.75, 1.75)
        ax.set_title(labels[idx])
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Time (s)")

    fig.suptitle(title or "QuASAr (parallel disjoint) vs baseline")

    import matplotlib.patches as mpatches

    baseline_used = {
        method
        for method, t in zip(baseline_methods, baseline_times)
        if np.isfinite(t) and t > 0
    }

    legend_handles = [
        mpatches.Patch(
            color=QUASAR_COLOR,
            edgecolor=EDGE_COLOR,
            label="QuASAr (parallel disjoint)",
        )
    ]

    for method in sorted(baseline_used):
        label_method = method.replace("_", " ").upper()
        legend_handles.append(
            mpatches.Patch(
                color=BASELINE_COLORS.get(method, FALLBACK_COLOR),
                edgecolor=EDGE_COLOR,
                alpha=0.9,
                label=f"Baseline: {label_method}",
            )
        )

    if legend_handles:
        fig.legend(handles=legend_handles, loc="center")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if out:
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    else:
        plt.show()


def make_memory_plot(
    suite_dir: str,
    out: Optional[str] = None,
    title: Optional[str] = None,
    *,
    log_scale: bool = False,
    case_kinds: Optional[Iterable[str]] = None,
) -> None:
    cases = _load_cases(suite_dir, case_kinds)
    if not cases:
        raise SystemExit("No matching cases found in suite_dir")

    metrics = _collect_case_metrics(cases)
    labels = metrics["labels"]
    quasar_mems = metrics["quasar_mems"]
    baseline_mems = metrics["baseline_mems"]
    baseline_methods = metrics["baseline_methods"]

    count = len(labels)
    if count == 0:
        raise SystemExit("No cases available for plotting")

    width = 0.6
    fig, axes = plt.subplots(1, count, figsize=(max(6, count * 3.6), 5.5), sharey=True)
    if count == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        quasar_mem = quasar_mems[idx]
        baseline_mem = baseline_mems[idx]
        baseline_method = baseline_methods[idx]

        quasar_height: Optional[float] = None
        baseline_height: Optional[float] = None

        if np.isfinite(quasar_mem) and quasar_mem > 0:
            quasar_height = _bytes_to_gib(quasar_mem)
            ax.bar(
                0,
                quasar_height,
                width,
                color=QUASAR_COLOR,
                edgecolor=EDGE_COLOR,
            )

        if np.isfinite(baseline_mem) and baseline_mem > 0:
            baseline_height = _bytes_to_gib(baseline_mem)
            ax.bar(
                1,
                baseline_height,
                width,
                color=BASELINE_COLORS.get(baseline_method, FALLBACK_COLOR),
                edgecolor=EDGE_COLOR,
                alpha=0.9,
            )

        if (
            quasar_height is not None
            and baseline_height is not None
            and quasar_mem is not None
            and baseline_mem is not None
            and quasar_mem > 0
        ):
            memory_saving = baseline_mem / quasar_mem
            if memory_saving > 0:
                if log_scale:
                    y_position = quasar_height * 1.2
                else:
                    y_position = quasar_height + 0.05 * max(quasar_height, baseline_height)
                ax.text(
                    0,
                    y_position,
                    f"{memory_saving:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["QuASAr", "Baseline"])
        ax.set_xlim(-0.75, 1.75)
        ax.set_title(labels[idx])
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ylabel = "Memory (GiB)"
            if log_scale:
                ylabel += " (log scale)"
            ax.set_ylabel(ylabel)

        if log_scale:
            ax.set_yscale("log")

    fig.suptitle(title or "QuASAr (parallel disjoint) memory vs baseline")

    import matplotlib.patches as mpatches

    baseline_used = {
        method
        for method, m in zip(baseline_methods, baseline_mems)
        if np.isfinite(m) and m > 0
    }

    legend_handles = [
        mpatches.Patch(
            color=QUASAR_COLOR,
            edgecolor=EDGE_COLOR,
            label="QuASAr (parallel disjoint)",
        )
    ]

    for method in sorted(baseline_used):
        label_method = method.replace("_", " ").upper()
        legend_handles.append(
            mpatches.Patch(
                color=BASELINE_COLORS.get(method, FALLBACK_COLOR),
                edgecolor=EDGE_COLOR,
                alpha=0.9,
                label=f"Baseline: {label_method}",
            )
        )

    if legend_handles:
        fig.legend(handles=legend_handles, loc="center")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if out:
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument(
        "--memory-out",
        type=str,
        default=None,
        help=(
            "Optional output path for the memory comparison plot. Pass 'auto' to "
            "derive the path from --out by appending '_memory' before the "
            "extension."
        ),
    )
    ap.add_argument(
        "--log",
        action="store_true",
        help="Plot memory comparisons on a logarithmic scale.",
    )
    args = ap.parse_args()

    make_plot(args.suite_dir, out=args.out, title=args.title)

    memory_out = args.memory_out
    if memory_out == "auto":
        if not args.out:
            raise SystemExit("--memory-out=auto requires --out to be specified")
        root, ext = os.path.splitext(args.out)
        memory_out = f"{root}_memory{ext}" if ext else f"{args.out}_memory"

    if memory_out is not None:
        make_memory_plot(
            args.suite_dir,
            out=memory_out,
            title=args.title,
            log_scale=args.log,
        )


if __name__ == "__main__":
    main()
