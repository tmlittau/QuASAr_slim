from __future__ import annotations

"""Theoretical runtime and memory estimation utilities for QuASAr and baselines."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional

from qiskit import QuantumCircuit

from .analyzer import analyze
from .backends.sv import estimate_sv_bytes
from .cost_estimator import CostEstimator, CostParams
from .gate_metrics import circuit_metrics
from .planner import PlannerConfig, plan

__all__ = [
    "BackendEstimate",
    "PartitionEstimate",
    "QuasarEstimate",
    "estimate_statevector",
    "estimate_tableau",
    "estimate_decision_diagram",
    "estimate_quasar",
]


@dataclass
class BackendEstimate:
    """Container for theoretical runtime and memory estimates of a backend."""

    backend: str
    ok: bool
    work_units: float
    work_unit_label: str
    time_seconds: Optional[float]
    memory_bytes: int
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartitionEstimate:
    """Describe the theoretical cost of a single QuASAr partition."""

    partition_id: int
    backend: str
    planner_reason: Optional[str]
    metrics: Dict[str, Any]
    estimate: BackendEstimate


@dataclass
class QuasarEstimate:
    """Aggregate theoretical runtime and memory information for a plan."""

    partitions: List[PartitionEstimate]
    work_units_by_label: Dict[str, float]
    total_time_seconds: Optional[float]
    peak_memory_bytes: int
    ok: bool
    planner_metadata: Dict[str, Any]
    plan_cost_units: float
    single_backend: Optional[str] = None
    single_backend_reason: Optional[str] = None


def _extract_gate_counts(metrics: Dict[str, Any]) -> Dict[str, int | float]:
    n = int(metrics.get("num_qubits", 0) or 0)
    total = int(metrics.get("num_gates", 0) or 0)
    twoq = int(metrics.get("two_qubit_gates", 0) or 0)
    oneq = max(total - twoq, 0)
    rotations = int(metrics.get("rotation_count", 0) or 0)
    sparsity = float(metrics.get("sparsity", 0.0) or 0.0)
    return {
        "num_qubits": n,
        "num_gates": total,
        "one_qubit_gates": oneq,
        "two_qubit_gates": twoq,
        "rotation_count": rotations,
        "sparsity": sparsity,
    }


def _build_estimator(cost_params: Optional[CostParams], tableau_unit_cost: float) -> CostEstimator:
    params = cost_params or CostParams()
    if params.tableau_prefix_unit_cost != tableau_unit_cost:
        params = replace(params, tableau_prefix_unit_cost=float(tableau_unit_cost))
    return CostEstimator(params)


def _amp_ops(n_qubits: int, oneq: int, twoq: int) -> float:
    if n_qubits <= 0:
        return 0.0
    amps = 1 << n_qubits
    return float(amps * (oneq + 4 * twoq))


def estimate_statevector_from_metrics(
    metrics: Dict[str, Any],
    *,
    ampops_per_sec: Optional[float] = None,
) -> BackendEstimate:
    counts = _extract_gate_counts(metrics)
    n = counts["num_qubits"]
    oneq = counts["one_qubit_gates"]
    twoq = counts["two_qubit_gates"]
    work = _amp_ops(int(n), int(oneq), int(twoq))
    time_seconds = work / ampops_per_sec if ampops_per_sec else None
    mem_bytes = estimate_sv_bytes(int(n))
    details = {"amp_ops": work, **counts}
    return BackendEstimate(
        backend="sv",
        ok=True,
        work_units=work,
        work_unit_label="amp_ops",
        time_seconds=time_seconds,
        memory_bytes=int(mem_bytes),
        details=details,
    )


def estimate_statevector(
    circuit: QuantumCircuit,
    *,
    ampops_per_sec: Optional[float] = None,
) -> BackendEstimate:
    metrics = circuit_metrics(circuit)
    return estimate_statevector_from_metrics(metrics, ampops_per_sec=ampops_per_sec)


def estimate_tableau_from_metrics(
    metrics: Dict[str, Any],
    *,
    tableau_unit_cost: float = 1.0,
    tableau_ops_per_sec: Optional[float] = None,
    tableau_state_bytes: float = 16.0,
) -> BackendEstimate:
    counts = _extract_gate_counts(metrics)
    n = counts["num_qubits"]
    total = counts["num_gates"]
    is_clifford = bool(metrics.get("is_clifford", False))
    details: Dict[str, Any] = {"num_qubits": n, "num_gates": total}
    if not is_clifford:
        return BackendEstimate(
            backend="tableau",
            ok=False,
            work_units=0.0,
            work_unit_label="tableau_ops",
            time_seconds=None,
            memory_bytes=0,
            reason="Circuit contains non-Clifford operations",
            details=details,
        )

    work = float(total) * float(tableau_unit_cost)
    time_seconds = work / tableau_ops_per_sec if tableau_ops_per_sec else None
    n_int = max(1, int(n))
    mem_bytes = int(n_int * n_int * tableau_state_bytes)
    details.update({"tableau_unit_cost": tableau_unit_cost, "work_units": work})
    return BackendEstimate(
        backend="tableau",
        ok=True,
        work_units=work,
        work_unit_label="tableau_ops",
        time_seconds=time_seconds,
        memory_bytes=mem_bytes,
        details=details,
    )


def estimate_tableau(
    circuit: QuantumCircuit,
    *,
    tableau_unit_cost: float = 1.0,
    tableau_ops_per_sec: Optional[float] = None,
    tableau_state_bytes: float = 16.0,
) -> BackendEstimate:
    metrics = circuit_metrics(circuit)
    return estimate_tableau_from_metrics(
        metrics,
        tableau_unit_cost=tableau_unit_cost,
        tableau_ops_per_sec=tableau_ops_per_sec,
        tableau_state_bytes=tableau_state_bytes,
    )


def estimate_decision_diagram_from_metrics(
    metrics: Dict[str, Any],
    estimator: CostEstimator,
    *,
    dd_ops_per_sec: Optional[float] = None,
    dd_node_bytes: float = 64.0,
) -> BackendEstimate:
    counts = _extract_gate_counts(metrics)
    n = int(counts["num_qubits"])
    total = int(counts["num_gates"])
    twoq = int(counts["two_qubit_gates"])
    rotations = int(counts["rotation_count"])
    sparsity = float(counts["sparsity"])
    details = estimator.decision_diagram_details(
        n=n,
        num_gates=total,
        twoq=twoq,
        rotation_count=rotations,
        sparsity=sparsity,
    )
    work = float(details["cost"])
    nodes = float(details.get("estimated_nodes", 0.0))
    mem_bytes = 0
    if nodes > 0:
        mem_bytes = int(max(1.0, nodes) * float(dd_node_bytes))
    elif total > 0:
        mem_bytes = int(float(dd_node_bytes))
    time_seconds = work / dd_ops_per_sec if dd_ops_per_sec else None
    details.update({"dd_node_bytes": dd_node_bytes})
    return BackendEstimate(
        backend="dd",
        ok=True,
        work_units=work,
        work_unit_label="amp_ops",
        time_seconds=time_seconds,
        memory_bytes=mem_bytes,
        details=details,
    )


def estimate_decision_diagram(
    circuit: QuantumCircuit,
    *,
    cost_params: Optional[CostParams] = None,
    dd_ops_per_sec: Optional[float] = None,
    dd_node_bytes: float = 64.0,
) -> BackendEstimate:
    params = cost_params or CostParams()
    estimator = CostEstimator(params)
    metrics = circuit_metrics(circuit)
    return estimate_decision_diagram_from_metrics(
        metrics,
        estimator,
        dd_ops_per_sec=dd_ops_per_sec,
        dd_node_bytes=dd_node_bytes,
    )


def _estimate_backend_from_metrics(
    backend: str,
    metrics: Dict[str, Any],
    estimator: CostEstimator,
    *,
    sv_ampops_per_sec: Optional[float],
    dd_ops_per_sec: Optional[float],
    tableau_unit_cost: float,
    tableau_ops_per_sec: Optional[float],
    tableau_state_bytes: float,
    dd_node_bytes: float,
) -> BackendEstimate:
    name = backend.lower()
    if name == "sv":
        return estimate_statevector_from_metrics(metrics, ampops_per_sec=sv_ampops_per_sec)
    if name == "dd":
        return estimate_decision_diagram_from_metrics(
            metrics,
            estimator,
            dd_ops_per_sec=dd_ops_per_sec,
            dd_node_bytes=dd_node_bytes,
        )
    if name == "tableau":
        return estimate_tableau_from_metrics(
            metrics,
            tableau_unit_cost=tableau_unit_cost,
            tableau_ops_per_sec=tableau_ops_per_sec,
            tableau_state_bytes=tableau_state_bytes,
        )
    # Fallback to SV model for unknown backends
    estimate = estimate_statevector_from_metrics(metrics, ampops_per_sec=sv_ampops_per_sec)
    estimate.reason = f"Fallback estimate for unsupported backend '{backend}'"
    estimate.backend = name
    return estimate


def estimate_quasar(
    circuit: QuantumCircuit,
    *,
    planner_cfg: Optional[PlannerConfig] = None,
    cost_params: Optional[CostParams] = None,
    sv_ampops_per_sec: Optional[float] = None,
    dd_ampops_per_sec: Optional[float] = None,
    tableau_unit_cost: float = 1.0,
    tableau_ops_per_sec: Optional[float] = None,
    tableau_state_bytes: float = 16.0,
    dd_node_bytes: float = 64.0,
    analysis_fn: Optional[Callable[[QuantumCircuit], Any]] = None,
) -> QuasarEstimate:
    """Return theoretical runtime and memory estimates for a QuASAr plan."""

    if analysis_fn is None:
        analysis = analyze(circuit)
    else:
        analysis = analysis_fn(circuit)
    cfg = planner_cfg or PlannerConfig()
    planned = plan(analysis.plan, cfg)

    estimator = _build_estimator(cost_params, tableau_unit_cost)

    partitions: List[PartitionEstimate] = []
    work_units_by_label: Dict[str, float] = {}
    total_time: Optional[float] = 0.0
    peak_memory = 0
    ok_all = True

    for qusd in planned.qusds:
        metrics = qusd.metrics or circuit_metrics(qusd.circuit)
        backend = (qusd.backend or "sv").lower()
        reason = None
        if isinstance(qusd.meta, dict):
            reason = qusd.meta.get("planner_reason")

        estimate = _estimate_backend_from_metrics(
            backend,
            metrics,
            estimator,
            sv_ampops_per_sec=sv_ampops_per_sec,
            dd_ops_per_sec=dd_ampops_per_sec,
            tableau_unit_cost=tableau_unit_cost,
            tableau_ops_per_sec=tableau_ops_per_sec,
            tableau_state_bytes=tableau_state_bytes,
            dd_node_bytes=dd_node_bytes,
        )

        partitions.append(
            PartitionEstimate(
                partition_id=int(qusd.id),
                backend=backend,
                planner_reason=reason,
                metrics=dict(metrics),
                estimate=estimate,
            )
        )
        work_units_by_label[estimate.work_unit_label] = work_units_by_label.get(
            estimate.work_unit_label, 0.0
        ) + float(estimate.work_units)
        peak_memory = max(peak_memory, int(estimate.memory_bytes))
        ok_all = ok_all and bool(estimate.ok)
        if total_time is not None:
            if estimate.time_seconds is None:
                total_time = None
            else:
                total_time += float(estimate.time_seconds)

    single_backend = None
    single_reason = None
    if partitions:
        backends = {p.backend for p in partitions}
        if len(backends) == 1:
            single_backend = partitions[0].backend
            reasons = {p.planner_reason for p in partitions if p.planner_reason}
            if len(reasons) == 1:
                single_reason = next(iter(reasons))

    planner_meta = {}
    if isinstance(planned.meta, dict):
        planner_meta = dict(planned.meta.get("planner", {}))

    return QuasarEstimate(
        partitions=partitions,
        work_units_by_label=work_units_by_label,
        total_time_seconds=total_time,
        peak_memory_bytes=int(peak_memory),
        ok=ok_all,
        planner_metadata=planner_meta,
        plan_cost_units=float(planned.estimated_cost),
        single_backend=single_backend,
        single_backend_reason=single_reason,
    )
