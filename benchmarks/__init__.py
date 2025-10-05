"""Benchmark circuit registries for QuASAr experiments."""

from __future__ import annotations

from typing import Any, Dict

from . import hybrid, disjoint, dd_friendly

CIRCUIT_REGISTRY: Dict[str, Any] = {}
CIRCUIT_REGISTRY.update(hybrid.CIRCUIT_REGISTRY)
CIRCUIT_REGISTRY.update(disjoint.CIRCUIT_REGISTRY)
CIRCUIT_REGISTRY.update(dd_friendly.CIRCUIT_REGISTRY)

__all__ = [
    "hybrid",
    "disjoint",
    "dd_friendly",
    "CIRCUIT_REGISTRY",
    "build",
]


def build(kind: str, /, **kwargs: Any):
    try:
        builder = CIRCUIT_REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown circuit kind '{kind}'") from exc
    return builder(**kwargs)
