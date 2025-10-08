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
from typing import Dict, Iterable, List, Optional

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

@dataclass(frozen=True)
class VariantMetrics:
    """Aggregate execution metrics for one planner variant."""

    name: str
    wall_time_s: float
    max_mem_gb: float


def _as_variant_metrics(record: Dict[str, object]) -> VariantMetrics:
    name = str(record.get("name", "unknown"))
    execution = record.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"Variant '{name}' is missing execution data")

    meta = execution.get("meta", {})
    wall = float(meta.get("wall_elapsed_s", 0.0))

    results = execution.get("results")
    if not isinstance(results, Iterable):
        raise ValueError(f"Variant '{name}' has no execution results to summarise")

    max_mem_bytes = 0
    for entry in results:
        if isinstance(entry, dict):
            max_mem_bytes = max(max_mem_bytes, int(entry.get("mem_bytes", 0)))

    return VariantMetrics(name=name, wall_time_s=wall, max_mem_gb=max_mem_bytes / (1024**3))


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


def _plot_bars(ax, labels: List[str], values: List[float], *, title: str, ylabel: str) -> None:
    colors = list(islice(cycle(_PASTEL_PALETTE), len(values)))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")


def plot_metrics(
    metrics: List[VariantMetrics], *, output: Optional[Path] = None, title: Optional[str] = None
) -> None:
    """Render the runtime and memory bar charts for the provided metrics."""

    labels = [m.name for m in metrics]
    runtimes = [m.wall_time_s for m in metrics]
    memories = [m.max_mem_gb for m in metrics]

    fig, (ax_runtime, ax_memory) = plt.subplots(1, 2, figsize=(10, 4))
    suptitle = title if title is not None else _DEFAULT_TITLE
    fig.suptitle(suptitle)

    _plot_bars(ax_runtime, labels, runtimes, title="Runtime", ylabel="Seconds")
    _plot_bars(ax_memory, labels, memories, title="Memory", ylabel="GiB")

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
