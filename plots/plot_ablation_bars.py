#!/usr/bin/env python3
"""Plot runtime and memory bars for the ablation study variants.

The script consumes the JSON summary produced by ``run_ablation_study`` and
renders two bar charts – one for wall-clock runtime and another for maximum
memory usage – so the three planner variants can be compared side-by-side.

Typical usage::

    python -m plots.plot_ablation_bars --summary path/to/results.json \
        --output plots/ablation_bars.png

The helper functions are kept lightweight so they can also be imported from
tests to verify the aggregation logic without invoking matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


_PASTEL_PALETTE = [
    "#a6cee3",  # pastel blue
    "#b2df8a",  # pastel green
    "#fb9a99",  # pastel red
    "#fdbf6f",  # pastel orange
    "#cab2d6",  # pastel purple
    "#ffffb3",  # pastel yellow
]

_DEFAULT_TITLE = "QuASAr ablation study"
_BASELINE_VARIANT_NAME = "full"
_BASELINE_LABEL = "baseline"

@dataclass(frozen=True)
class VariantMetrics:
    """Aggregate execution metrics for one planner variant."""

    name: str
    wall_time_s: float
    max_mem_bytes: int
    wall_time_estimated: bool = False
    max_mem_estimated: bool = False

    @property
    def max_mem_gib(self) -> float:
        return self.max_mem_bytes / float(1024**3)


def _as_variant_metrics(record: Dict[str, object]) -> VariantMetrics:
    name = str(record.get("name", "unknown"))

    summary = record.get("summary")
    if isinstance(summary, dict):
        wall = float(summary.get("wall_time_s", 0.0))
        max_mem = int(summary.get("max_mem_bytes", 0))
        wall_est = bool(summary.get("wall_time_estimated", False))
        mem_est = bool(summary.get("max_mem_estimated", False))
        return VariantMetrics(
            name=name,
            wall_time_s=wall,
            max_mem_bytes=max_mem,
            wall_time_estimated=wall_est,
            max_mem_estimated=mem_est,
        )

    execution = record.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"Variant '{name}' is missing execution data")

    meta = execution.get("meta", {})
    wall = float(meta.get("wall_elapsed_s", 0.0))

    results = execution.get("results")
    if not isinstance(results, Iterable):
        raise ValueError(f"Variant '{name}' has no execution results to summarise")

    max_mem_bytes = 0
    mem_estimated = False
    for entry in results:
        if isinstance(entry, dict):
            mem = entry.get("mem_bytes")
            estimated = False
            if mem is None:
                mem = entry.get("mem_bytes_estimated")
                if mem is not None:
                    estimated = True
            if mem is None:
                continue
            mem_int = int(mem)
            if mem_int > max_mem_bytes:
                max_mem_bytes = mem_int
                mem_estimated = estimated

    return VariantMetrics(
        name=name,
        wall_time_s=wall,
        max_mem_bytes=max_mem_bytes,
        max_mem_estimated=mem_estimated,
    )


def _memory_values(metrics: List[VariantMetrics]) -> Tuple[List[float], str]:
    if not metrics:
        return [], "GiB"
    max_bytes = max(m.max_mem_bytes for m in metrics)
    if max_bytes <= 0:
        return [0.0 for _ in metrics], "GiB"
    if max_bytes >= 1024**3:
        divisor = float(1024**3)
        unit = "GiB"
    elif max_bytes >= 1024**2:
        divisor = float(1024**2)
        unit = "MiB"
    else:
        divisor = 1024.0
        unit = "KiB"
    values = [m.max_mem_bytes / divisor for m in metrics]
    return values, unit


def collect_variant_metrics(summary: Dict[str, object]) -> List[VariantMetrics]:
    """Extract :class:`VariantMetrics` from the ablation JSON summary."""

    variants = summary.get("variants")
    if not isinstance(variants, Iterable):
        raise ValueError("Summary is missing the 'variants' list")

    metrics: List[VariantMetrics] = []
    for record in variants:
        if isinstance(record, dict):
            metrics.append(_as_variant_metrics(record))

    if not metrics:
        raise ValueError("No variants with execution data were found in the summary")
    return metrics


def _plot_bars(
    ax,
    labels: List[str],
    values: List[float],
    *,
    title: str,
    ylabel: str,
    value_formatter=None,
) -> None:
    colors = list(islice(cycle(_PASTEL_PALETTE), len(values)))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    formatter = value_formatter if value_formatter is not None else (lambda value: f"{value:.2f}")
    for idx, value in enumerate(values):
        if value == 1.0:
            continue
        ax.text(idx, value, formatter(value), ha="center", va="bottom")


def _baseline_index(metrics: List[VariantMetrics]) -> int:
    for idx, metric in enumerate(metrics):
        if metric.name == _BASELINE_VARIANT_NAME:
            return idx
    for idx, metric in enumerate(metrics):
        if metric.name == _BASELINE_LABEL:
            return idx
    return 0


def _relative_to_baseline(values: List[float], baseline_idx: int) -> List[float]:
    if not values:
        return []
    baseline_value = values[baseline_idx]
    if baseline_value == 0:
        return [1.0 if idx == baseline_idx else 0.0 for idx in range(len(values))]
    return [value / baseline_value for value in values]


def _display_labels(metrics: List[VariantMetrics]) -> List[str]:
    labels: List[str] = []
    for metric in metrics:
        if metric.name == _BASELINE_VARIANT_NAME:
            labels.append(_BASELINE_LABEL)
        else:
            labels.append(metric.name)
    return labels


def plot_metrics(
    metrics: List[VariantMetrics], *, output: Optional[Path] = None, title: Optional[str] = None
) -> None:
    """Render the runtime and memory bar charts for the provided metrics."""

    labels = _display_labels(metrics)
    baseline_idx = _baseline_index(metrics)

    runtimes = [m.wall_time_s for m in metrics]
    runtime_rel = _relative_to_baseline(runtimes, baseline_idx)

    memories_bytes = [m.max_mem_bytes for m in metrics]
    memory_rel = _relative_to_baseline(memories_bytes, baseline_idx)

    fig, (ax_runtime, ax_memory) = plt.subplots(1, 2, figsize=(10, 4))
    suptitle = title if title is not None else _DEFAULT_TITLE
    fig.suptitle(suptitle)

    _plot_bars(
        ax_runtime,
        labels,
        runtime_rel,
        title="Runtime",
        ylabel="Runtime (× baseline)",
        value_formatter=lambda value: f"{value:.2f}x",
    )
    _plot_bars(
        ax_memory,
        labels,
        memory_rel,
        title="Memory",
        ylabel="Memory (× baseline)",
        value_formatter=lambda value: f"{value:.2f}x",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()


def _load_summary(path: Path) -> Dict[str, object]:
    import json

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Summary JSON must describe an object")
    return data


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Path to the JSON summary produced by run_ablation_study")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the generated plot. Defaults to showing it interactively.")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help=f"Optional title displayed above the plots (defaults to '{_DEFAULT_TITLE}')",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = _load_summary(args.summary)
    metrics = collect_variant_metrics(summary)
    plot_metrics(metrics, output=args.output, title=args.title)


if __name__ == "__main__":  # pragma: no cover
    main()
