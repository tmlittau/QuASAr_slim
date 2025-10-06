from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("numpy")

from scripts import bench_from_thresholds as bft


@pytest.fixture
def tmp_thresholds(tmp_path: Path) -> dict:
    thr = {
        "meta": {"params": {"angle_scale": 0.1, "conv_factor": 1.0, "twoq_factor": 2.0}},
        "records": [
            {
                "n": 3,
                "cutoff": 0.5,
                "first_depth": 5,
            }
        ],
    }
    return thr


def _make_fake_analyze() -> mock.Mock:
    fake = mock.Mock()
    fake.metrics_global = {"num_qubits": 3}
    fake_ssd = mock.Mock()
    fake_tail = mock.Mock()
    fake_tail.backend = "sv"
    fake_ssd.partitions = [fake_tail]
    fake.ssd = fake_ssd
    return fake


def test_run_from_thresholds_invokes_baselines(tmp_path: Path, tmp_thresholds: dict) -> None:
    execute_payload = {"meta": {"wall_elapsed_s": 1.23}, "results": []}
    fake_analyze = _make_fake_analyze()
    fake_ssd = mock.Mock()
    fake_ssd.partitions = []
    fake_ssd.to_dict.return_value = {"partitions": []}

    with mock.patch.object(bft, "clifford_prefix_rot_tail", return_value=mock.Mock()) as mock_builder, \
         mock.patch.object(bft, "analyze", return_value=fake_analyze) as mock_analyze, \
         mock.patch.object(bft, "plan", return_value=fake_ssd) as mock_plan, \
         mock.patch.object(bft, "execute_ssd", return_value=execute_payload) as mock_execute, \
         mock.patch.object(bft, "run_baselines", return_value={"entries": []}) as mock_baseline:
        bft.run_from_thresholds(
            tmp_thresholds,
            cutoff=0.5,
            out_dir=str(tmp_path),
            angle_scale=0.1,
            conv_factor=1.0,
            twoq_factor=2.0,
            max_ram_gb=1.0,
            sv_ampops_per_sec=None,
            baseline_timeout_s=None,
            log=logging.getLogger("test"),
        )

    mock_builder.assert_called_once()
    mock_analyze.assert_called()
    mock_plan.assert_called()
    mock_execute.assert_called()
    mock_baseline.assert_called()
    assert mock_baseline.call_args.kwargs["which"] == ["sv"]

    out_files = list(tmp_path.glob("*.json"))
    assert out_files, "expected a record JSON to be written"
    payload = json.loads(out_files[0].read_text())
    assert "baselines" in payload


def test_run_from_thresholds_selects_dd_baseline(tmp_path: Path, tmp_thresholds: dict) -> None:
    execute_payload = {"meta": {"wall_elapsed_s": 1.23}, "results": []}
    fake_analyze = _make_fake_analyze()
    fake_ssd = mock.Mock()
    fake_tail = mock.Mock()
    fake_tail.backend = "dd"
    fake_ssd.partitions = [fake_tail]
    fake_ssd.to_dict.return_value = {"partitions": []}

    with mock.patch.object(bft, "clifford_prefix_rot_tail", return_value=mock.Mock()), \
         mock.patch.object(bft, "analyze", return_value=fake_analyze), \
         mock.patch.object(bft, "plan", return_value=fake_ssd), \
         mock.patch.object(bft, "execute_ssd", return_value=execute_payload), \
         mock.patch.object(bft, "run_baselines", return_value={"entries": []}) as mock_baseline:
        bft.run_from_thresholds(
            tmp_thresholds,
            cutoff=0.5,
            out_dir=str(tmp_path),
            angle_scale=0.1,
            conv_factor=1.0,
            twoq_factor=2.0,
            max_ram_gb=1.0,
            sv_ampops_per_sec=None,
            baseline_timeout_s=None,
            log=logging.getLogger("test"),
        )

    assert mock_baseline.call_args.kwargs["which"] == ["dd"]
