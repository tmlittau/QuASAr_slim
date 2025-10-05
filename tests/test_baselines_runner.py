from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("qiskit")

from quasar.baselines import runner
from benchmarks.hybrid import clifford_prefix_rot_tail


def test_run_single_baseline_adds_wall_time():
    payload = {"ok": True, "backend": "sv", "elapsed_s": 1.5}
    with mock.patch("quasar.baselines.runner._run_backend", return_value=dict(payload)) as mock_backend:
        res = runner.run_single_baseline(object(), "sv")
    mock_backend.assert_called_once()
    assert pytest.approx(1.5) == res["result"]["elapsed_s"]
    assert pytest.approx(1.5) == res["result"]["wall_s_measured"]


def test_run_single_baseline_per_partition_rollup():
    part_payloads = [
        {"ok": True, "backend": "sv", "elapsed_s": 0.4},
        {"ok": True, "backend": "sv", "elapsed_s": 0.6},
    ]
    fake_partitions = []
    for idx in range(len(part_payloads)):
        node = mock.Mock()
        node.id = idx
        node.circuit = mock.Mock()
        fake_partitions.append(node)

    analyze_mock = mock.Mock()
    analyze_mock.ssd.partitions = fake_partitions

    with mock.patch("quasar.baselines.runner.analyze", return_value=analyze_mock), \
         mock.patch("quasar.baselines.runner._run_backend", side_effect=[dict(p) for p in part_payloads]):
        res = runner.run_single_baseline(object(), "sv", per_partition=True)

    assert res["mode"] == "per_partition"
    assert pytest.approx(1.0) == res["elapsed_s"]
    assert pytest.approx(1.0) == res["wall_s_measured"]
    assert len(res["partitions"]) == 2
    for idx, part in enumerate(res["partitions"]):
        assert pytest.approx(part_payloads[idx]["elapsed_s"]) == part["wall_s_measured"]


def test_run_single_baseline_sv_handles_rotations():
    circ = clifford_prefix_rot_tail(num_qubits=3, depth=4, cutoff=0.5, angle_scale=0.1, seed=123)
    res = runner.run_single_baseline(circ, "sv")
    payload = res["result"]
    assert payload["ok"] is True
    assert payload.get("error") is None
    assert payload.get("statevector_len") == 8
    assert payload.get("wall_s_measured", 0.0) > 0.0
