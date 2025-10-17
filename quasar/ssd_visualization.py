"""Utilities for visualising :class:`~quasar.qusd.Plan` instances."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .qusd import Plan

try:  # pragma: no cover - optional dependency
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

__all__ = ["visualize_plan"]


def _ensure_networkx_available() -> None:
    if nx is None:  # pragma: no cover - exercised only when dependency missing
        raise RuntimeError(
            "networkx is required for plan visualisation. Install it with 'pip "
            "install networkx' to enable this feature."
        )


def _ensure_matplotlib_available() -> None:
    if plt is None:  # pragma: no cover - exercised only when dependency missing
        raise RuntimeError(
            "matplotlib is required to render plan visualisations. Install it "
            "with 'pip install matplotlib' to enable this feature."
        )


def _format_label(title: str, pairs: Iterable[Tuple[str, object]]) -> str:
    lines = [title]
    for key, value in pairs:
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _pick_layout(graph):
    if graph.number_of_nodes() <= 2:
        # Degenerate cases are easier to read with a circular layout
        return nx.circular_layout(graph)  # type: ignore[union-attr]
    return nx.spring_layout(graph, seed=7)  # type: ignore[union-attr]


def visualize_plan(
    plan: Plan,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
    ax=None,
):
    """Visualise an :class:`~quasar.qusd.Plan` using :mod:`networkx`."""

    _ensure_networkx_available()
    if ax is None:
        _ensure_matplotlib_available()
        fig, ax = plt.subplots(figsize=(max(6, len(plan.qusds) * 2.5), 6))
    else:
        fig = ax.figure

    graph = nx.DiGraph()  # type: ignore[union-attr]

    meta_pairs = sorted(plan.meta.items())
    graph.add_node(  # type: ignore[union-attr]
        "__plan_root__",
        label=_format_label("Plan", meta_pairs),
        kind="plan",
    )

    for qusd in plan.qusds:
        metrics = qusd.metrics or {}
        metric_pairs = sorted(metrics.items())
        info: list[Tuple[str, object]] = [
            ("Qubits", ", ".join(map(str, qusd.qubits)) or "-"),
            ("Backend", qusd.backend),
            ("Has out_state", bool(qusd.out_state)),
        ]
        info.extend(metric_pairs)
        label = _format_label(f"QuSD {qusd.id}", info)
        node_id = f"qusd_{qusd.id}"
        graph.add_node(  # type: ignore[union-attr]
            node_id,
            label=label,
            kind="qusd",
        )
        graph.add_edge("__plan_root__", node_id)  # type: ignore[union-attr]

        chain_id = qusd.meta.get("chain_id") if qusd.meta else None
        if chain_id is not None:
            chain_node = f"chain_{chain_id}"
            if chain_node not in graph:  # type: ignore[union-attr]
                graph.add_node(  # type: ignore[union-attr]
                    chain_node,
                    label=_format_label(f"Chain {chain_id}", []),
                    kind="chain",
                )
                graph.add_edge("__plan_root__", chain_node)  # type: ignore[union-attr]
            graph.add_edge(chain_node, node_id)  # type: ignore[union-attr]
        for successor in qusd.successors:
            succ_id = f"qusd_{successor.id}"
            graph.add_edge(node_id, succ_id)  # type: ignore[union-attr]

    layout = _pick_layout(graph)
    labels = {n: data.get("label", str(n)) for n, data in graph.nodes(data=True)}
    colors = []
    for _, data in graph.nodes(data=True):
        kind = data.get("kind")
        if kind == "plan":
            colors.append("#4477AA")
        elif kind == "chain":
            colors.append("#AA3377")
        else:
            colors.append("#66C2A5")

    nx.draw_networkx_nodes(  # type: ignore[union-attr]
        graph,
        layout,
        node_color=colors,
        ax=ax,
        node_size=3000,
        edgecolors="black",
    )
    nx.draw_networkx_edges(graph, layout, ax=ax, arrows=True, arrowstyle="-|>")  # type: ignore[union-attr]
    nx.draw_networkx_labels(graph, layout, labels=labels, ax=ax, font_size=8)  # type: ignore[union-attr]

    ax.set_axis_off()
    ax.set_title("Plan visualisation")

    if save_path:
        _ensure_matplotlib_available()
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        _ensure_matplotlib_available()
        plt.show()

    return ax
