
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CostParams:
    conv_amp_ops_factor: float = 64.0
    sv_twoq_factor: float = 4.0
    tableau_prefix_unit_cost: float = 0.0

class CostEstimator:
    def __init__(self, params: CostParams | None = None) -> None:
        self.params = params or CostParams()

    @staticmethod
    def _amps(n: int) -> int:
        return 1 << max(0, int(n))

    def sv_cost(self, n: int, oneq: int, twoq: int) -> float:
        return self._amps(n) * (oneq + self.params.sv_twoq_factor * twoq)

    def tableau_prefix_cost(self, n: int, oneq: int, twoq: int) -> float:
        return self.params.tableau_prefix_unit_cost * (oneq + twoq)

    def conversion_cost(self, n: int) -> float:
        return self.params.conv_amp_ops_factor * self._amps(n)

    def compare_clifford_prefix_tail(self, *, n: int, one_pre: int, two_pre: int, one_tail: int, two_tail: int) -> Dict[str, Any]:
        sv_total  = self.sv_cost(n, one_pre + one_tail, two_pre + two_tail)
        sv_tail   = self.sv_cost(n, one_tail, two_tail)
        conv      = self.conversion_cost(n)
        tab_pre   = self.tableau_prefix_cost(n, one_pre, two_pre)
        hybrid    = tab_pre + conv + sv_tail
        return {
            "sv_total": float(sv_total),
            "sv_tail": float(sv_tail),
            "conversion": float(conv),
            "tableau_prefix": float(tab_pre),
            "hybrid_total": float(hybrid),
            "hybrid_better": bool(hybrid < sv_total),
        }

    @classmethod
    def from_planner_config(cls, cfg) -> "CostEstimator":
        return cls(CostParams(
            conv_amp_ops_factor=getattr(cfg, "conv_amp_ops_factor", 64.0),
            sv_twoq_factor=getattr(cfg, "sv_twoq_factor", 4.0),
            tableau_prefix_unit_cost=0.0,
        ))
