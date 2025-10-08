from __future__ import annotations

import pathlib
import sys

import pytest
from qiskit import QuantumCircuit

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasar.analyzer import analyze
from quasar.SSD import SSD, PartitionNode
from quasar.cost_estimator import CostEstimator
from quasar.planner import PlannerConfig, plan
import quasar.planner as planner_mod
from plots import plot_ablation_bars as pab
from scripts import run_ablation_study as ras


def test_streamlined_hybrid_blocks_split(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit, specs = ras.build_ablation_circuit(
        num_components=2,
        component_size=3,
        clifford_depth=2,
        tail_depth=2,
        tail_sequence=("random", "sparse"),
        seed=7,
    )

    analysis = analyze(circuit)
    assert len(analysis.ssd.partitions) == 2

    monkeypatch.setattr(planner_mod, "stim_available", lambda: True)
    monkeypatch.setattr(planner_mod, "ddsim_available", lambda: True)

    cfg = PlannerConfig(conv_amp_ops_factor=0.0, prefer_dd=True, hybrid_clifford_tail=True)
    planned = plan(analysis.ssd, cfg)

    chains: dict[str, list] = {}
    for node in planned.partitions:
        chain_id = node.meta.get("chain_id")
        if chain_id is None:
            continue
        chains.setdefault(chain_id, []).append(node)

    assert len(chains) == len(specs)

    for spec in specs:
        chain_id = f"p{spec.index}_hybrid"
        assert chain_id in chains
        nodes = sorted(chains[chain_id], key=lambda n: n.meta.get("seq_index", 0))
        assert len(nodes) == 2
        prefix, tail = nodes
        assert prefix.backend == "tableau"
        expected_tail = "sv" if spec.tail_kind == "random" else "dd"
        assert tail.backend == expected_tail


def test_collect_variant_metrics_extracts_wall_and_memory() -> None:
    summary = {
        "variants": [
            {
                "name": "full",
                "summary": {
                    "wall_time_s": 3.2,
                    "max_mem_bytes": 2 * 1024**3,
                    "wall_time_estimated": False,
                    "max_mem_estimated": False,
                },
            },
            {
                "name": "no_disjoint",
                "summary": {
                    "wall_time_s": 4.8,
                    "max_mem_bytes": 3 * 1024**3,
                    "wall_time_estimated": True,
                    "max_mem_estimated": True,
                },
            },
        ]
    }

    metrics = pab.collect_variant_metrics(summary)
    assert [m.name for m in metrics] == ["full", "no_disjoint"]
    assert metrics[0].wall_time_s == pytest.approx(3.2)
    assert metrics[0].max_mem_bytes == 2 * 1024**3
    assert metrics[0].max_mem_gib == pytest.approx(2.0)
    assert metrics[1].max_mem_gib == pytest.approx(3.0)
    assert metrics[1].wall_time_estimated is True


def test_summarise_execution_prefers_peak_rss() -> None:
    ssd = SSD()
    qc = QuantumCircuit(1)
    node = PartitionNode(
        id=0,
        qubits=[0],
        circuit=qc,
        metrics={"num_qubits": 1, "num_gates": 1, "two_qubit_gates": 0},
        backend="sv",
    )
    ssd.add(node)

    execution = {
        "results": [
            {
                "status": "ok",
                "backend": "sv",
                "elapsed_s": 0.1,
                "wall_s_measured": 0.1,
                "mem_bytes": 32,
                "peak_rss_bytes": 10 * 1024**2,
            }
        ],
        "meta": {"wall_elapsed_s": 0.1},
    }

    estimator = CostEstimator.from_planner_config(PlannerConfig())
    summary = ras._summarise_execution(ssd, execution, estimator)
    assert summary is not None
    assert summary["max_mem_bytes"] == 10 * 1024**2
    assert summary["max_mem_estimated"] is False
    assert summary.get("max_rss_bytes") == 10 * 1024**2
