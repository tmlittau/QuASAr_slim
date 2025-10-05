from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from benchmarks.hybrid import clifford_prefix_rot_tail
from quasar.backends import _partition


def test_extract_operations_coerces_rotation_params_to_float():
    circ = clifford_prefix_rot_tail(num_qubits=3, depth=4, cutoff=0.5, angle_scale=0.1, seed=99)
    n, ops = _partition.extract_operations(circ)
    assert n == 3
    angles = [params for name, params in ((op.name, op.params) for op in ops) if name == "rx"]
    assert angles, "Expected at least one rotation parameter"
    for params in angles:
        assert all(isinstance(val, float) for val in params)


def test_extract_operations_rejects_complex_params():
    class FakeGate:
        name = "rx"
        params = (1 + 2j,)

    class FakeEntry:
        operation = FakeGate()
        qubits = (object(),)

    class FakeCircuit:
        num_qubits = 1
        qubits = (object(),)
        data = [FakeEntry()]

    with pytest.raises(TypeError):
        _partition.extract_operations(FakeCircuit())
