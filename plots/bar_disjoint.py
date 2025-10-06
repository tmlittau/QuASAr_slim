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

CASE_KIND = "disjoint_preps_plus_tails"


def _iter_case_files(suite_dir: str) -> Iterable[str]:
    try:
        entries = sorted(os.listdir(suite_dir))
    except FileNotFoundError:
        raise SystemExit(f"suite_dir not found: {suite_dir}")
    for fn in entries:
        if not fn.endswith(".json") or fn == "index.json":
            continue
        yield os.path.join(suite_dir, fn)


def _load_cases(suite_dir: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for path in _iter_case_files(suite_dir):
        with open(path, "r") as f:
            data = json.load(f)
        case = (data.get("case") or {})
        if case.get("kind") == CASE_KIND:
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


def _extract_quasar_time(data: Dict[str, Any]) -> Optional[float]:
    quasar = (data.get("quasar") or {})
    execution = quasar.get("execution") or data.get("execution") or {}
    meta = execution.get("meta") or {}
    t = _safe_float(meta.get("wall_elapsed_s"))
    if t is not None:
        return t
    results = execution.get("results") or []
    best: Optional[float] = None
    if isinstance(results, list):
        for res in results:
            elapsed = _safe_float((res or {}).get("elapsed_s"))
            if elapsed is None:
                continue
            if best is None or elapsed > best:
                best = elapsed
    return best


def _is_whole_baseline(entry: Dict[str, Any]) -> bool:
    if (entry or {}).get("ok") is False:
        return False
    scope = (entry or {}).get("scope")
    if scope in {"whole", "global", "circuit"}:
        return True
    if entry.get("per_partition") is False:
        return True
    if "partition_id" not in entry and "chain_id" not in entry:
        return True
    return False


def _best_whole_baseline(baselines: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    entries = baselines.get("entries") if isinstance(baselines, dict) else None
    if not isinstance(entries, list):
        entries = []
    best: Optional[Tuple[str, float]] = None
    for entry in entries:
        if not _is_whole_baseline(entry):
            continue
        elapsed = entry.get("wall_s_measured")
        if elapsed is None:
            elapsed = entry.get("wall_s_estimated")
        elapsed_f = _safe_float(elapsed)
        if elapsed_f is None:
            continue
        method = (entry.get("method") or entry.get("name") or "sv").lower()
        if best is None or elapsed_f < best[1]:
            best = (method, elapsed_f)
    return best


def make_plot(suite_dir: str, out: Optional[str] = None, title: Optional[str] = None) -> None:
    cases = _load_cases(suite_dir)
    if not cases:
        raise SystemExit("No matching cases found in suite_dir")

    labels: List[str] = []
    quasar_times: List[float] = []
    baseline_times: List[float] = []
    baseline_methods: List[str] = []

    for data in cases:
        params = (data.get("case") or {}).get("params", {})
        n = int(params.get("num_qubits", 0) or 0)
        k = int(params.get("num_blocks", 0) or 0)
        labels.append(f"n={n}, k={k}")

        qt = _extract_quasar_time(data)
        quasar_times.append(qt if qt is not None else float("nan"))

        best = _best_whole_baseline(data.get("baselines", {}))
        if best is None:
            baseline_methods.append("sv")
            baseline_times.append(float("nan"))
        else:
            baseline_methods.append(best[0])
            baseline_times.append(best[1])

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(8, len(labels) * 1.4), 5.5))
    plt.bar(
        x - width / 2,
        quasar_times,
        width,
        label="QuASAr (parallel disjoint)",
        color=QUASAR_COLOR,
        edgecolor=EDGE_COLOR,
    )

    baseline_colors = [BASELINE_COLORS.get(m, FALLBACK_COLOR) for m in baseline_methods]
    plt.bar(
        x + width / 2,
        baseline_times,
        width,
        color=baseline_colors,
        edgecolor=EDGE_COLOR,
        alpha=0.9,
    )

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Time (s)")
    plt.title(title or "QuASAr (parallel disjoint) vs baseline")

    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(
            color=QUASAR_COLOR,
            edgecolor=EDGE_COLOR,
            label="QuASAr (parallel disjoint)",
        ),
        mpatches.Patch(
            color=BASELINE_COLORS["sv"],
            edgecolor=EDGE_COLOR,
            label="Baseline: SV",
        ),
        mpatches.Patch(
            color=BASELINE_COLORS["dd"],
            edgecolor=EDGE_COLOR,
            label="Baseline: DD",
        ),
        mpatches.Patch(
            color=BASELINE_COLORS["tableau"],
            edgecolor=EDGE_COLOR,
            label="Baseline: Tableau",
        ),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=200)
        print(f"Wrote {out}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    make_plot(args.suite_dir, out=args.out, title=args.title)


if __name__ == "__main__":
    main()
