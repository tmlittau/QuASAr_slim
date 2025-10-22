from __future__ import annotations

import pathlib
import sys

import pytest
from qiskit import QuantumCircuit

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasar import theoretical as theo


def _two_component_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    qc.h(2)
    qc.cx(2, 3)
    qc.rx(0.3, 2)
    return qc


def test_analyze_without_disjoint_marks_partition() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.25, 0)

    analysis = theo.analyze_without_disjoint(qc, force_backend="sv")
    assert len(analysis.plan.qusds) == 1
    node = analysis.plan.qusds[0]
    assert node.meta.get("collapsed") is True
    assert node.meta.get("forced_backend") == "sv"
    assert node.meta.get("forced_backend_reason") == "forced_single_partition"


def test_disable_disjoint_ablation_collapses_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = _two_component_circuit()

    captured_plan = None
    original_plan = theo.plan

    def fake_plan(plan_arg, cfg):
        nonlocal captured_plan
        captured_plan = plan_arg
        return original_plan(plan_arg, cfg)

    monkeypatch.setattr(theo, "plan", fake_plan)

    estimate = theo.estimate_quasar(
        qc,
        ablation=theo.TheoreticalAblationOptions(disable_disjoint_subcircuits=True),
    )

    assert captured_plan is not None
    assert len(captured_plan.qusds) == 1
    node = captured_plan.qusds[0]
    assert node.meta.get("collapsed") is True
    assert node.meta.get("forced_backend") == "sv"
    assert estimate.single_backend == "sv"
    assert estimate.single_backend_reason == "forced_single_partition"


def test_disable_method_partitioning_updates_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 0)

    captured_cfg = None
    original_plan = theo.plan

    def fake_plan(plan_arg, cfg):
        nonlocal captured_cfg
        captured_cfg = cfg
        return original_plan(plan_arg, cfg)

    monkeypatch.setattr(theo, "plan", fake_plan)

    theo.estimate_quasar(
        qc,
        ablation=theo.TheoreticalAblationOptions(disable_method_partitioning=True),
    )

    assert captured_cfg is not None
    assert captured_cfg.hybrid_clifford_tail is False


def test_force_full_search_overrides_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = QuantumCircuit(1)
    qc.h(0)

    captured_cfg = None
    original_plan = theo.plan

    def fake_plan(plan_arg, cfg):
        nonlocal captured_cfg
        captured_cfg = cfg
        return original_plan(plan_arg, cfg)

    monkeypatch.setattr(theo, "plan", fake_plan)

    theo.estimate_quasar(
        qc,
        ablation=theo.TheoreticalAblationOptions(force_full_planner_search=True),
    )

    assert captured_cfg is not None
    assert captured_cfg.quick_path_partition_threshold == -1
    assert captured_cfg.quick_path_gate_threshold == -1
    assert captured_cfg.quick_path_qubit_threshold == -1
