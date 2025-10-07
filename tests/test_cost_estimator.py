from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quasar.cost_estimator import CostEstimator, CostParams


def test_compare_clifford_prefix_dd_tail_basic():
    est = CostEstimator(CostParams())
    prefix = {
        "num_gates": 4,
        "two_qubit_gates": 2,
        "rotation_count": 0,
        "sparsity": 0.75,
    }
    tail = {
        "num_gates": 6,
        "two_qubit_gates": 3,
        "rotation_count": 2,
        "sparsity": 0.5,
    }
    res = est.compare_clifford_prefix_dd_tail(n=5, prefix_metrics=prefix, tail_metrics=tail)
    assert set(res.keys()) >= {
        "dd_total",
        "dd_tail",
        "conversion",
        "tableau_prefix",
        "hybrid_total",
        "hybrid_better",
        "prefix_sparsity",
        "tail_sparsity",
    }
    assert res["dd_total"] >= res["dd_tail"]
    assert res["prefix_sparsity"] == prefix["sparsity"]
    assert res["tail_sparsity"] == tail["sparsity"]
