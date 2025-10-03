
from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math
from .SSD import SSD, PartitionNode

# Lazy imports in backends
from .backends.sv import StatevectorBackend, estimate_sv_bytes
from .backends.dd import DecisionDiagramBackend, ddsim_available
from .backends.tableau import TableauBackend, stim_available

@dataclass
class PlannerConfig:
    max_ram_gb: float = 64.0
    max_concurrency: Optional[int] = None  # None = auto
    prefer_dd: bool = True  # prefer DD over SV when sizes are borderline

def _choose_backend(metrics: Dict[str, Any], cfg: PlannerConfig) -> str:
    n = int(metrics.get("num_qubits", 0))
    is_cliff = bool(metrics.get("is_clifford", False))
    # 1) Clifford -> tableau if available
    if is_cliff and stim_available():
        return "tableau"
    # 2) Small-enough for SV?
    sv_bytes = estimate_sv_bytes(n)
    cap = int(cfg.max_ram_gb * (1024**3))
    if sv_bytes <= cap:
        # if prefer_dd and ddsim is available, route to DD for larger n
        if cfg.prefer_dd and ddsim_available() and n >= 20:
            return "dd"
        return "sv"
    # 3) Otherwise DD if available
    if ddsim_available():
        return "dd"
    # 4) Fallback to tableau if maybe not strictly Clifford (will raise if unsupported)
    if stim_available():
        return "tableau"
    # Last resort: SV even if it may blow memory (caller should gate this)
    return "sv"

def plan(ssd: SSD, cfg: Optional[PlannerConfig] = None) -> SSD:
    cfg = cfg or PlannerConfig()
    annotated = SSD(meta=dict(ssd.meta))
    annotated.meta["planner"] = {"max_ram_gb": cfg.max_ram_gb, "prefer_dd": cfg.prefer_dd}
    # First pass: backend selection
    per_bytes: List[int] = []
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
        if b == "sv":
            per_bytes.append(estimate_sv_bytes(int(node.metrics.get("num_qubits", 0))))
        else:
            per_bytes.append(0)

    # Concurrency gating for statevector usage
    cap = int(cfg.max_ram_gb * (1024**3))
    sv_needs = [b for b in per_bytes if b > 0]
    sv_needs.sort(reverse=True)
    k_auto = len(sv_needs)
    while k_auto > 0 and sum(sv_needs[:k_auto]) > cap:
        k_auto -= 1
    if cfg.max_concurrency is not None:
        k_auto = min(k_auto, cfg.max_concurrency)
    annotated.meta["concurrency"] = max(1, k_auto if k_auto > 0 else 1)
    return annotated

def execute(ssd: SSD) -> Dict[str, Any]:
    """Execute the annotated SSD and return a simple result payload.
    For now we run partitions sequentially and return per-partition statevectors (when available).
    """
    results = []
    for node in ssd.partitions:
        backend = node.backend or "sv"
        if backend == "tableau":
            out = TableauBackend().run(node.circuit)
        elif backend == "dd":
            out = DecisionDiagramBackend().run(node.circuit)
        else:
            out = StatevectorBackend().run(node.circuit)
        results.append({
            "partition": node.id,
            "backend": backend,
            "num_qubits": node.metrics.get("num_qubits"),
            "statevector_len": len(out) if out is not None else None,
        })
    return {"results": results, "meta": ssd.meta}
