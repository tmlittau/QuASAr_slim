
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CostParams:
    conv_amp_ops_factor: float = 64.0
    sv_twoq_factor: float = 4.0
    tableau_prefix_unit_cost: float = 0.0
    dd_gate_node_factor: float = 0.05
    dd_frontier_weight: float = 0.35
    dd_rotation_weight: float = 0.1
    dd_twoq_weight: float = 0.2
    dd_sparsity_discount: float = 0.6
    dd_modifier_floor: float = 0.05
    dd_base_cost: float = 0.0

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

    def _decision_diagram_components(
        self,
        *,
        n: int,
        num_gates: int,
        twoq: int,
        rotation_count: int,
        sparsity: float,
    ) -> tuple[float, float, float, float, float]:
        """Return cost model components for decision diagram estimates."""

        if n <= 0 or num_gates <= 0:
            base_cost = float(self.params.dd_base_cost)
            return base_cost, 0.0, 0.0, 0.0, 1.0

        frontier = max(1, int(n))
        base_nodes = frontier * max(1.0, math.log2(frontier + 1.0))
        gate_factor = max(1.0, math.log2(num_gates + 1.0))
        rotation_density = rotation_count / max(1, num_gates)
        twoq_density = twoq / max(1, num_gates)
        sparsity = min(max(float(sparsity), 0.0), 1.0)

        modifier = 1.0
        modifier += self.params.dd_frontier_weight * math.log2(frontier + 1.0)
        modifier += self.params.dd_rotation_weight * rotation_density
        modifier += self.params.dd_twoq_weight * twoq_density
        modifier -= self.params.dd_sparsity_discount * sparsity
        modifier = max(self.params.dd_modifier_floor, modifier)

        node_factor = num_gates * base_nodes * gate_factor * modifier
        cost = self.params.dd_base_cost + self.params.dd_gate_node_factor * node_factor
        return float(cost), float(node_factor), float(base_nodes), float(gate_factor), float(modifier)

    def decision_diagram_cost(
        self,
        *,
        n: int,
        num_gates: int,
        twoq: int,
        rotation_count: int,
        sparsity: float,
    ) -> float:
        cost, *_ = self._decision_diagram_components(
            n=n,
            num_gates=num_gates,
            twoq=twoq,
            rotation_count=rotation_count,
            sparsity=sparsity,
        )
        return float(cost)

    def decision_diagram_details(
        self,
        *,
        n: int,
        num_gates: int,
        twoq: int,
        rotation_count: int,
        sparsity: float,
    ) -> Dict[str, float]:
        """Return detailed components of the decision diagram cost model."""

        cost, node_factor, base_nodes, gate_factor, modifier = self._decision_diagram_components(
            n=n,
            num_gates=num_gates,
            twoq=twoq,
            rotation_count=rotation_count,
            sparsity=sparsity,
        )
        return {
            "cost": float(cost),
            "estimated_nodes": float(node_factor) if node_factor > 0 else 0.0,
            "base_nodes": float(base_nodes),
            "gate_factor": float(gate_factor),
            "modifier": float(modifier),
        }

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

    def compare_clifford_prefix_dd_tail(
        self,
        *,
        n: int,
        prefix_metrics: Dict[str, Any],
        tail_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        pre_twoq = int(prefix_metrics.get("two_qubit_gates", 0) or 0)
        pre_total = int(prefix_metrics.get("num_gates", 0) or 0)
        pre_oneq = max(pre_total - pre_twoq, 0)
        pre_rot = int(prefix_metrics.get("rotation_count", 0) or 0)
        pre_sparsity = float(prefix_metrics.get("sparsity", 0.0) or 0.0)

        tail_twoq = int(tail_metrics.get("two_qubit_gates", 0) or 0)
        tail_total = int(tail_metrics.get("num_gates", 0) or 0)
        tail_oneq = max(tail_total - tail_twoq, 0)
        tail_rot = int(tail_metrics.get("rotation_count", 0) or 0)
        tail_sparsity = float(tail_metrics.get("sparsity", 0.0) or 0.0)

        total_twoq = pre_twoq + tail_twoq
        total_rot = pre_rot + tail_rot
        total_gates = pre_total + tail_total
        combined_sparsity = min(pre_sparsity, tail_sparsity)

        dd_total = self.decision_diagram_cost(
            n=n,
            num_gates=total_gates,
            twoq=total_twoq,
            rotation_count=total_rot,
            sparsity=combined_sparsity,
        )
        dd_tail = self.decision_diagram_cost(
            n=n,
            num_gates=tail_total,
            twoq=tail_twoq,
            rotation_count=tail_rot,
            sparsity=tail_sparsity,
        )

        tab_pre = self.tableau_prefix_cost(n, pre_oneq, pre_twoq)
        conv = self.conversion_cost(n)
        hybrid = tab_pre + conv + dd_tail

        return {
            "dd_total": float(dd_total),
            "dd_tail": float(dd_tail),
            "conversion": float(conv),
            "tableau_prefix": float(tab_pre),
            "hybrid_total": float(hybrid),
            "hybrid_better": bool(hybrid < dd_total),
            "prefix_sparsity": float(pre_sparsity),
            "tail_sparsity": float(tail_sparsity),
        }

    @classmethod
    def from_planner_config(cls, cfg) -> "CostEstimator":
        return cls(CostParams(
            conv_amp_ops_factor=getattr(cfg, "conv_amp_ops_factor", 64.0),
            sv_twoq_factor=getattr(cfg, "sv_twoq_factor", 4.0),
            tableau_prefix_unit_cost=0.0,
        ))
