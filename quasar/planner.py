from __future__ import annotations

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .backends.sv import estimate_sv_bytes
from .backends.dd import ddsim_available
from .backends.tableau import stim_available
from .cost_estimator import CostEstimator
from .gate_metrics import circuit_metrics, gate_name, CLIFFORD_GATES
from .simulation_plan import SimulationPlanCollection
from .qusd import Plan, QuSD


@dataclass
class PlannerConfig:
    max_ram_gb: float = 64.0
    max_concurrency: Optional[int] = None
    prefer_dd: bool = True
    hybrid_clifford_tail: bool = True
    conv_amp_ops_factor: float = 64.0
    sv_twoq_factor: float = 4.0
    max_candidate_plans: int = 4
    quick_path_partition_threshold: int = 1
    quick_path_gate_threshold: int = 128
    quick_path_qubit_threshold: int = 16


def _choose_backend(metrics: Dict[str, Any], cfg: PlannerConfig) -> tuple[str, str]:
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


def _split_at_first_nonclifford(node: QuSD):
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


def _consider_hybrid(node: QuSD, cfg: PlannerConfig):
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
    cmp = est.compare_clifford_prefix_tail(
        n=n,
        one_pre=one_pre,
        two_pre=two_pre,
        one_tail=one_tail,
        two_tail=two_tail,
    )
    if cmp["hybrid_better"]:
        pre = _build_subcircuit_like(node.circuit, pre_ops)
        tail = _build_subcircuit_like(node.circuit, tail_ops)
        pre_metrics = circuit_metrics(pre)
        tail_metrics = circuit_metrics(tail)
        pre_node = QuSD(
            id=int(f"{node.id}0"),
            qubits=list(node.qubits),
            circuit=pre,
            metrics=pre_metrics,
            meta=dict(node.meta),
        )
        tail_node = QuSD(
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
        pre_node.meta.update(
            {
                "chain_id": chain_id,
                "seq_index": 0,
                "planner_reason": (
                    f"hybrid split@{split_idx}; conv={cmp['conversion']/(1<<n):.1f}, "
                    f"tail={cmp['sv_tail']/(1<<n):.1f}, hybrid={cmp['hybrid_total']/(1<<n):.1f} "
                    f"< sv={cmp['sv_total']/(1<<n):.1f} (ampops/2^n)"
                ),
            }
        )
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


def _initialize_planner_meta(plan: Plan, cfg: PlannerConfig) -> Plan:
    annotated = Plan(meta=dict(plan.meta))
    annotated.meta["planner"] = {
        "max_ram_gb": cfg.max_ram_gb,
        "prefer_dd": cfg.prefer_dd,
        "hybrid_clifford_tail": cfg.hybrid_clifford_tail,
        "conv_amp_ops_factor": cfg.conv_amp_ops_factor,
        "sv_twoq_factor": cfg.sv_twoq_factor,
    }
    return annotated


def _make_partition_node(node: QuSD, backend: str, reason: str) -> QuSD:
    new_node = QuSD(
        id=node.id,
        qubits=list(node.qubits),
        circuit=node.circuit,
        metrics=dict(node.metrics),
        meta=dict(node.meta),
    )
    new_node.set_backend(backend)
    new_node.meta["planner_reason"] = reason
    return new_node


def _plan_options_for_partition(node: QuSD, cfg: PlannerConfig) -> List[List[QuSD]]:
    meta = dict(node.meta)
    if meta.get("collapsed"):
        collapsed = QuSD(
            id=node.id,
            qubits=list(node.qubits),
            circuit=node.circuit,
            metrics=dict(node.metrics),
            meta=meta,
        )
        forced_backend = meta.get("forced_backend")
        if forced_backend:
            reason = meta.get("forced_backend_reason") or f"forced_backend={forced_backend}"
            collapsed.set_backend(forced_backend)
        else:
            backend, reason = _choose_backend(collapsed.metrics, cfg)
            collapsed.set_backend(backend)
        collapsed.meta["planner_reason"] = reason
        return [[collapsed]]

    options: List[List[QuSD]] = []
    backend, reason = _choose_backend(node.metrics, cfg)
    options.append([_make_partition_node(node, backend, reason)])

    hybrid = _consider_hybrid(node, cfg)
    if hybrid:
        options.append(hybrid)
    return options


def _linear_plan(plan: Plan, cfg: PlannerConfig) -> Plan:
    annotated = _initialize_planner_meta(plan, cfg)
    estimator = CostEstimator.from_planner_config(cfg)
    for node in plan.qusds:
        options = _plan_options_for_partition(node, cfg)
        if not options:
            continue
        best_option = None
        best_cost = None
        for option in options:
            option_cost = 0.0
            for candidate in option:
                option_cost += Plan._estimate_qusd_cost(candidate, estimator)
            if best_cost is None or option_cost < best_cost:
                best_cost = option_cost
                best_option = option
        if best_option:
            annotated.extend_plan(best_option, estimator)
    return annotated


def _should_use_quick_path(plan: Plan, cfg: PlannerConfig) -> bool:
    if len(plan.qusds) <= max(1, cfg.quick_path_partition_threshold):
        return True
    total_gates = 0
    max_qubits = 0
    for node in plan.qusds:
        total_gates += int(node.metrics.get("num_gates", 0) or 0)
        max_qubits = max(max_qubits, int(node.metrics.get("num_qubits", 0) or 0))
    if total_gates <= cfg.quick_path_gate_threshold and max_qubits <= cfg.quick_path_qubit_threshold:
        return True
    return False


def plan(plan: Plan, cfg: Optional[PlannerConfig] = None) -> Plan:
    cfg = cfg or PlannerConfig()
    if _should_use_quick_path(plan, cfg):
        return _linear_plan(plan, cfg)

    base_plan = _initialize_planner_meta(plan, cfg)
    estimator = CostEstimator.from_planner_config(cfg)
    plans = SimulationPlanCollection(cfg.max_candidate_plans)
    plans.add(base_plan)

    for node in plan.qusds:
        options = _plan_options_for_partition(node, cfg)
        next_plans = SimulationPlanCollection(cfg.max_candidate_plans)
        for current in plans:
            for option in options:
                candidate = current.fork()
                candidate.extend_plan(option, estimator)
                next_plans.add(candidate)
        plans = next_plans

    best_plan = plans.best()
    if best_plan is None:
        return _linear_plan(plan, cfg)
    return best_plan


def execute(plan: Plan):
    from .simulation_engine import execute_plan, ExecutionConfig

    return execute_plan(plan, ExecutionConfig())
