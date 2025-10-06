from __future__ import annotations

import pathlib
import sys
from types import ModuleType
from unittest import mock

import pytest


def _ensure_stub(name: str, module: ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


_numpy_stub = ModuleType("numpy")
setattr(_numpy_stub, "array", lambda x, **_: x)
setattr(_numpy_stub, "asarray", lambda x, **_: x)
setattr(_numpy_stub, "complex128", complex)
_ensure_stub("numpy", _numpy_stub)


_qiskit_stub = ModuleType("qiskit")


class _StubQuantumCircuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def copy(self):
        return self

    def save_statevector(self):
        return None


setattr(_qiskit_stub, "QuantumCircuit", _StubQuantumCircuit)
_ensure_stub("qiskit", _qiskit_stub)


_qiskit_aer_stub = ModuleType("qiskit_aer")


class _StubJob:
    def result(self):  # pragma: no cover - not expected to be called
        raise RuntimeError("stub")


class _StubAerSimulator:
    def __init__(self, **_):
        pass

    def run(self, *_args, **_kwargs):
        return _StubJob()


setattr(_qiskit_aer_stub, "AerSimulator", _StubAerSimulator)
_ensure_stub("qiskit_aer", _qiskit_aer_stub)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quasar.baselines import runner
from quasar.backends import dd as dd_module
from quasar.backends import sv as sv_module


class _FakeCircuit:
    num_qubits = 3


def _fake_analyze() -> mock.Mock:
    fake = mock.Mock()
    fake.metrics_global = {
        "num_qubits": _FakeCircuit.num_qubits,
        "num_gates": 10,
        "two_qubit_gates": 4,
    }
    fake.ssd = mock.Mock()
    fake.ssd.partitions = []
    return fake


@pytest.fixture(autouse=True)
def patch_analyze(monkeypatch):
    monkeypatch.setattr(runner, "analyze", lambda circuit: _fake_analyze())


def test_statevector_backend_propagates_timeout(monkeypatch):
    monkeypatch.setattr(sv_module, "extract_operations", lambda circuit: (_FakeCircuit.num_qubits, []))

    class _TimeoutJob:
        def result(self):
            raise TimeoutError("deadline")

    class _TimeoutAerSimulator:
        def __init__(self, **_):
            pass

        def run(self, *_args, **_kwargs):
            return _TimeoutJob()

    monkeypatch.setattr(sv_module, "AerSimulator", _TimeoutAerSimulator)

    backend = sv_module.StatevectorBackend()

    with pytest.raises(TimeoutError):
        backend.run(_FakeCircuit())


def test_decision_diagram_backend_propagates_timeout(monkeypatch):
    monkeypatch.setattr(dd_module, "extract_operations", lambda circuit: (_FakeCircuit.num_qubits, []))

    class _StubQuantumComputation:
        def __init__(self, _num_qubits: int):
            pass

    class _TimeoutDDSim:
        def __init__(self, _qc):
            pass

        def simulate(self, *_):
            raise TimeoutError("deadline")

        def get_constructed_dd(self):  # pragma: no cover - not reached
            return mock.Mock()

    monkeypatch.setattr(dd_module, "QuantumComputation", _StubQuantumComputation)
    _ddsim_stub = ModuleType("mqt.ddsim")
    setattr(_ddsim_stub, "CircuitSimulator", _TimeoutDDSim)
    monkeypatch.setattr(dd_module, "ddsim", _ddsim_stub)

    backend = dd_module.DecisionDiagramBackend()

    with pytest.raises(TimeoutError):
        backend.run(_FakeCircuit())


def test_run_backend_reports_timeout_estimate(monkeypatch):
    class _TimeoutBackend(runner.StatevectorBackend):
        def run(self, circuit):  # type: ignore[override]
            raise TimeoutError("boom")

    monkeypatch.setattr(runner, "StatevectorBackend", _TimeoutBackend)

    result = runner._run_backend("sv", _FakeCircuit(), timeout_s=1.0, sv_ampops_per_sec=123.0)

    assert result["backend"] == "sv"
    assert result["ok"] is False
    assert result["error"].startswith("Timeout")
    assert "estimate" in result
