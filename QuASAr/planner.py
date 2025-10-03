
from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .SSD import SSD, PartitionNode
from .backends.sv import StatevectorBackend, estimate_sv_bytes
from .backends.dd import DecisionDiagramBackend, ddsim_available
from .backends.tableau import TableauBackend, stim_available

@dataclass
class PlannerConfig:
    max_ram_gb: float = 64.0
    max_concurrency: Optional[int] = None
    prefer_dd: bool = True

def _choose_backend(metrics: Dict[str, Any], cfg: PlannerConfig) -> str:
    n = int(metrics.get("num_qubits", 0))
    is_cliff = bool(metrics.get("is_clifford", False))
    if is_cliff and stim_available():
        return "tableau"
    sv_bytes = estimate_sv_bytes(n)
    cap = int(cfg.max_ram_gb * (1024**3))
    if sv_bytes <= cap:
        if cfg.prefer_dd and ddsim_available() and n >= 20:
            return "dd"
        return "sv"
    if ddsim_available():
        return "dd"
    if stim_available():
        return "tableau"
    return "sv"

def plan(ssd: SSD, cfg: Optional[PlannerConfig] = None) -> SSD:
    cfg = cfg or PlannerConfig()
    annotated = SSD(meta=dict(ssd.meta))
    annotated.meta["planner"] = {"max_ram_gb": cfg.max_ram_gb, "prefer_dd": cfg.prefer_dd}
    for node in ssd.partitions:
        b = _choose_backend(node.metrics, cfg)
        new_node = PartitionNode(
            id=node.id,
            qubits=list(node.qubits),
            circuit=node.circuit,
            metrics=dict(node.metrics),
            meta=dict(node.meta),
        )
        new_node.set_backend(b)
        annotated.add(new_node)
    return annotated

def execute(ssd: SSD) -> Dict[str, Any]:
    from .simulation_engine import execute_ssd, ExecutionConfig
    return execute_ssd(ssd, ExecutionConfig())
