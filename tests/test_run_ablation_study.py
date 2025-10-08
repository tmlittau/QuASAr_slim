from __future__ import annotations

import pathlib
import sys

import pytest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from quasar.SSD import PartitionNode, SSD
from quasar.planner import PlannerConfig
from quasar.simulation_engine import ExecutionConfig
from scripts import run_ablation_study as ras


class _DummyCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits


def test_run_variant_falls_back_to_plan_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    ssd = SSD()
    node = PartitionNode(
        id=0,
        qubits=list(range(5)),
        circuit=_DummyCircuit(5),
        metrics={"num_qubits": 5, "num_gates": 10, "two_qubit_gates": 4},
        backend="sv",
    )
    ssd.add(node)

    exec_payload = {
        "results": [
            {
                "partition": 0,
                "status": "estimated",
                "backend": "sv",
                "elapsed_s": 0.1,
                "wall_s_measured": 0.1,
                "mem_bytes_estimated": 123,
            }
        ],
        "meta": {"wall_elapsed_s": 0.1},
    }

    monkeypatch.setattr(ras, "plan", lambda ssd_in, cfg: ssd_in)
    monkeypatch.setattr(ras, "_estimate_amp_ops", lambda planned, cfg: 0.0)
    monkeypatch.setattr(ras, "execute_ssd", lambda planned, cfg: exec_payload)

    planner_cfg = PlannerConfig(max_ram_gb=1.0)
    exec_cfg = ExecutionConfig(max_ram_gb=1.0)

    result = ras._run_variant("full", ssd, planner_cfg, exec_cfg)

    expected_mem = ras._estimate_mem_for_backend("sv", 5)
    assert result.peak_mem_bytes == expected_mem
    assert result.peak_mem_estimate_bytes == 123
