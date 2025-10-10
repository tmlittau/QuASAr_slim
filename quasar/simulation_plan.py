from __future__ import annotations

from typing import Dict, Iterator, Optional

from .SSD import SSD


class SimulationPlanCollection:
    """Maintain a deduplicated set of SSD plans ordered by estimated cost."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, int(max_size))
        self._plans: Dict[str, SSD] = {}

    def __iter__(self) -> Iterator[SSD]:
        return iter(sorted(self._plans.values(), key=lambda plan: plan.estimated_cost))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._plans)

    def add(self, plan: SSD) -> None:
        fingerprint = plan.fingerprint
        existing = self._plans.get(fingerprint)
        if existing is not None:
            if plan.estimated_cost < existing.estimated_cost:
                self._plans[fingerprint] = plan
            return
        self._plans[fingerprint] = plan
        if len(self._plans) > self._max_size:
            # Drop the most expensive plans to keep the search bounded.
            sorted_fps = sorted(
                self._plans,
                key=lambda fp: self._plans[fp].estimated_cost,
                reverse=True,
            )
            for fp in sorted_fps[self._max_size :]:
                self._plans.pop(fp, None)

    def best(self) -> Optional[SSD]:
        if not self._plans:
            return None
        return min(self._plans.values(), key=lambda plan: plan.estimated_cost)
