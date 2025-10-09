
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from .SSD import SSD, PartitionNode
from .backends.sv import estimate_sv_bytes
from .backends.dd import ddsim_available
from .backends.tableau import stim_available
from .cost_estimator import CostEstimator
from .gate_metrics import circuit_metrics, gate_name, CLIFFORD_GATES

@dataclass
class PlannerConfig:
    max_ram_gb: float = 64.0
    max_concurrency: Optional[int] = None
    prefer_dd: bool = True
    hybrid_clifford_tail: bool = True
    conv_amp_ops_factor: float = 64.0
    sv_twoq_factor: float = 4.0

def _choose_backend(metrics: Dict[str, Any], cfg: PlannerConfig) -> Tuple[str, str]:
    n = int(metrics.get("num_qubits", 0))
    is_cliff = bool(metrics.get("is_clifford", False))
    if is_cliff and stim_available():
        return "tableau", "is_clifford=True & stim_available"
    sv_bytes = estimate_sv_bytes(n)
    cap = int(cfg.max_ram_gb * (1024**3))
    if sv_bytes <= cap:
        dd_ready = ddsim_available()
        min_dd_qubits = 12
        if cfg.prefer_dd and dd_ready and n >= min_dd_qubits:
            return "dd", f"prefer_dd & n>={min_dd_qubits} & ddsim_available"
        reasons = ["fits_ram"]
        if n < 20:
            reasons.append("n<20")
        if not cfg.prefer_dd:
            reasons.append("prefer_dd=False")
        if cfg.prefer_dd and not dd_ready:
            reasons.append("ddsim_unavailable")
        return "sv", " & ".join(reasons)
    if ddsim_available():
        return "dd", "sv_exceeds_ram & ddsim_available"
    if stim_available():
        return "tableau", "sv_exceeds_ram & ddsim_unavailable & stim_available"
    return "sv", "fallback: only sv available"

def _split_at_first_nonclifford(node: PartitionNode):
    data = node.circuit.data
    split = None
    for idx, (inst, qargs, cargs) in enumerate(data):
        if gate_name(inst) not in CLIFFORD_GATES:
            split = idx
            break
    if split is None or split == 0:
        return None
    return split, data[:split], data[split:]

def _count_ops(ops: List):
    oneq = twoq = 0
    for inst, qargs, _ in ops:
        if len(qargs) >= 2:
            twoq += 1
        else:
            oneq += 1
    return oneq, twoq

def _build_subcircuit_like(parent, ops: List):
    from qiskit import QuantumCircuit
    sub = QuantumCircuit(parent.num_qubits)
    qubit_map = {parent.qubits[i]: sub.qubits[i] for i in range(parent.num_qubits)}
    for inst, qargs, cargs in ops:
        local_qargs = [qubit_map[q] for q in qargs]
        sub.append(inst, local_qargs, cargs)
    parent_meta = getattr(parent, "metadata", None)
    if parent_meta:
        sub.metadata = dict(parent_meta)
    return sub

def _consider_hybrid(node: PartitionNode, cfg: PlannerConfig):
    if not cfg.hybrid_clifford_tail:
        return None
    split = _split_at_first_nonclifford(node)
    if not split:
        return None
    split_idx, pre_ops, tail_ops = split
    n = int(node.metrics.get("num_qubits", 0))
    one_pre, two_pre = _count_ops(pre_ops)
    one_tail, two_tail = _count_ops(tail_ops)
    est = CostEstimator.from_planner_config(cfg)
    cmp = est.compare_clifford_prefix_tail(n=n, one_pre=one_pre, two_pre=two_pre, one_tail=one_tail, two_tail=two_tail)
    if cmp["hybrid_better"]:
        pre = _build_subcircuit_like(node.circuit, pre_ops)
        tail = _build_subcircuit_like(node.circuit, tail_ops)
        pre_metrics = circuit_metrics(pre)
        tail_metrics = circuit_metrics(tail)
        pre_node = PartitionNode(
            id=int(f"{node.id}0"),
            qubits=list(node.qubits),
            circuit=pre,
            metrics=pre_metrics,
            meta=dict(node.meta),
        )
        tail_node = PartitionNode(
            id=int(f"{node.id}1"),
            qubits=list(node.qubits),
            circuit=tail,
            metrics=tail_metrics,
            meta=dict(node.meta),
        )
        pre_node.set_backend("tableau" if stim_available() else "sv")
        tail_backend, tail_backend_reason = _choose_backend(tail_metrics, cfg)
        if (
            tail_backend != "dd"
            and cfg.prefer_dd
            and ddsim_available()
            and float(tail_metrics.get("sparsity", 0.0)) >= 0.5
        ):
            tail_backend = "dd"
            tail_backend_reason = (
                f"hybrid_tail_sparse(s={float(tail_metrics.get('sparsity', 0.0)):.2f})"
            )
        tail_node.set_backend(tail_backend)
        chain_id = f"p{node.id}_hybrid"
        pre_node.meta.update({"chain_id": chain_id, "seq_index": 0,
                              "planner_reason": f"hybrid split@{split_idx}; conv={cmp['conversion']/(1<<n):.1f}, tail={cmp['sv_tail']/(1<<n):.1f}, hybrid={cmp['hybrid_total']/(1<<n):.1f} < sv={cmp['sv_total']/(1<<n):.1f} (ampops/2^n)"})
        tail_reason = (
            f"hybrid tail -> {tail_backend}; {tail_backend_reason}; "
            f"sv_tail={cmp['sv_tail']/(1<<n):.1f} (ampops/2^n)"
        )
        tail_node.meta.update(
            {
                "chain_id": chain_id,
                "seq_index": 1,
                "planner_reason": tail_reason,
            }
        )
        return [pre_node, tail_node]
    return None

def plan(ssd: SSD, cfg: Optional[PlannerConfig] = None) -> SSD:
    cfg = cfg or PlannerConfig()
    annotated = SSD(meta=dict(ssd.meta))
    seen_fingerprints: Dict[str, int] = {}
    annotated.meta["planner"] = {
        "max_ram_gb": cfg.max_ram_gb, "prefer_dd": cfg.prefer_dd,
        "hybrid_clifford_tail": cfg.hybrid_clifford_tail,
        "conv_amp_ops_factor": cfg.conv_amp_ops_factor,
        "sv_twoq_factor": cfg.sv_twoq_factor,
    }
    for node in ssd.partitions:
        meta = dict(node.meta)
        fp = node.compute_fingerprint()
        meta["cache_key"] = fp
        if fp in seen_fingerprints:
            meta["cache_of"] = seen_fingerprints[fp]
        else:
            seen_fingerprints[fp] = node.id
        if meta.get("collapsed"):
            new_node = PartitionNode(
                id=node.id,
                qubits=list(node.qubits),
                circuit=node.circuit,
                metrics=dict(node.metrics),
                meta=meta,
            )
            forced_backend = meta.get("forced_backend")
            if forced_backend:
                new_node.set_backend(forced_backend)
                reason = meta.get("forced_backend_reason") or f"forced_backend={forced_backend}"
            else:
                forced_backend, reason = _choose_backend(node.metrics, cfg)
                new_node.set_backend(forced_backend)
            new_node.meta["planner_reason"] = reason
            annotated.add(new_node)
            continue

        hybrid = _consider_hybrid(node, cfg)
        if hybrid:
            for hn in hybrid:
                hmeta = dict(hn.meta)
                fp = hn.compute_fingerprint()
                hmeta["cache_key"] = fp
                if fp in seen_fingerprints:
                    hmeta["cache_of"] = seen_fingerprints[fp]
                else:
                    seen_fingerprints[fp] = hn.id
                hn.meta = hmeta
                annotated.add(hn)
            continue
        b, why = _choose_backend(node.metrics, cfg)
        new_node = PartitionNode(
            id=node.id,
            qubits=list(node.qubits),
            circuit=node.circuit,
            metrics=dict(node.metrics),
            meta=meta,
        )
        new_node.set_backend(b)
        new_node.meta["planner_reason"] = why
        annotated.add(new_node)
    return annotated

def execute(ssd: SSD):
    from .simulation_engine import execute_ssd, ExecutionConfig
    return execute_ssd(ssd, ExecutionConfig())
