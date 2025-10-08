from __future__ import annotations

import pathlib
import sys

import pytest


# Ensure the repository root is importable when running tests from source checkouts.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os

import quasar.simulation_engine as sim
from quasar.SSD import PartitionNode, SSD
from quasar.simulation_engine import ExecutionConfig, execute_ssd


class _DummyCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 0


def _make_ssd(num_partitions: int, backend: str = "sv") -> SSD:
    ssd = SSD()
    for idx in range(num_partitions):
        node = PartitionNode(
            id=idx,
            qubits=[idx],
            circuit=_DummyCircuit(1),
            metrics={"num_qubits": 1, "gate_count": 0},
            backend=backend,
        )
        ssd.add(node)
    return ssd


@pytest.fixture(autouse=True)
def _patch_backend_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _runner(name: str, circuit, initial_state, **_: object):
        # Return a small dummy payload to mimic a statevector without heavy work.
        return [0]

    monkeypatch.setattr(sim, "_backend_runner", _runner)


@pytest.mark.parametrize(
    "num_partitions,cpu_count",
    [
        (3, 8),
        (6, 16),
        (12, 4),
    ],
)
def test_execute_ssd_auto_workers_serialises_sensitive_backends(
    monkeypatch: pytest.MonkeyPatch,
    num_partitions: int,
    cpu_count: int,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)

    cfg = ExecutionConfig(max_workers=0, heartbeat_sec=0.001, stuck_warn_sec=0.01)
    ssd = _make_ssd(num_partitions)

    execute_ssd(ssd, cfg)

    assert cfg.max_workers == 1


@pytest.mark.parametrize(
    "num_partitions,cpu_count,expected",
    [
        (3, 8, 3),
        (6, 16, 6),
        (12, 4, 4),
    ],
)
def test_execute_ssd_auto_workers_tracks_available_parallelism_for_tableau(
    monkeypatch: pytest.MonkeyPatch,
    num_partitions: int,
    cpu_count: int,
    expected: int,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)

    cfg = ExecutionConfig(max_workers=0, heartbeat_sec=0.001, stuck_warn_sec=0.01)
    ssd = _make_ssd(num_partitions, backend="tableau")

    execute_ssd(ssd, cfg)

    assert cfg.max_workers == expected
