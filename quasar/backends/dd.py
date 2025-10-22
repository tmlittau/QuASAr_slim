from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from mqt.core import QuantumComputation
import mqt.ddsim as ddsim


from ._partition import Operation, extract_operations

LOGGER = logging.getLogger(__name__)


_DDSIM_LOCK = threading.Lock()


@dataclass
class _DecisionDiagramResult:
    """Wrapper keeping DDSIM's simulator alive for downstream consumers."""

    simulator: "ddsim.CircuitSimulator"
    decision_diagram: "mqt.core.dd.VectorDD"

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.decision_diagram, name)

        if callable(attribute):
            # Wrap callables so ``self`` stays alive for the duration of the
            # delegated call, preventing the simulator from being destroyed
            # mid-execution when users chain ``run(...).method()``.

            def method(*args: Any, _attribute=attribute, _self=self, **kwargs: Any) -> Any:
                return _attribute(*args, **kwargs)

            return method

        return attribute


def ddsim_available() -> bool:
    return ddsim is not None


def _apply_operation(qc: QuantumComputation, op: Operation) -> None:
    name = op.name
    qubits = op.qubits
    params = op.params

    if name in {"i", "id"}:
        return
    if name in {"u", "u3"}:
        theta, phi, lam = (params + (0.0, 0.0, 0.0))[:3]
        qc.u(theta, phi, lam, *qubits)
        return
    if name == "u2":
        phi, lam = (params + (0.0, 0.0))[:2]
        qc.u2(phi, lam, *qubits)
        return
    if name in {"p", "u1"}:
        qc.p(params[0] if params else 0.0, *qubits)
        return
    if name in {"rx", "ry", "rz"}:
        getattr(qc, name)(params[0] if params else 0.0, *qubits)
        return
    if name == "cp":
        qc.cp(params[0] if params else 0.0, *qubits)
        return
    if name == "rzz":
        qc.rzz(params[0] if params else 0.0, *qubits)
        return

    method = getattr(qc, name, None)
    if method is None:
        raise NotImplementedError(f"Unsupported gate '{op.name}' for decision diagram backend")
    method(*qubits)


class DecisionDiagramBackend:
    """Simulate circuit partitions with MQT's decision diagram backend."""

    def run(
        self,
        circuit: Any,
        *,
        initial_state: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = False,
        batch_size: int = 500,
    ) -> Optional[Any]:
        del batch_size

        if QuantumComputation is None or ddsim is None:
            LOGGER.warning("DDSIM is not available; decision diagram backend cannot execute partition.")
            return None

        num_qubits, ops = extract_operations(circuit)
        if num_qubits == 0:
            return None

        partition_id = getattr(circuit, "partition_id", "unknown")
        LOGGER.debug(
            "Simulating partition %s on decision diagram backend with %d qubits and %d ops.",
            partition_id,
            num_qubits,
            len(ops),
        )

        if initial_state is not None:
            vec = np.asarray(initial_state, dtype=np.complex128).ravel()
            expected = 1 << num_qubits
            if vec.size != expected:
                raise ValueError(
                    f"Initial state has dimension {vec.size}, expected {expected} for {num_qubits} qubits.")
            raise NotImplementedError(
                "Custom initial states are not supported by the decision diagram backend"
            )

        qc = QuantumComputation(num_qubits)
        for op in ops:
            _apply_operation(qc, op)

        with _DDSIM_LOCK:
            simulator = ddsim.CircuitSimulator(qc)
            try:
                simulator.simulate(0)
            except TimeoutError:
                raise
            except Exception:
                LOGGER.exception("DDSIM failed while executing the partition.")
                return None

            if progress_cb is not None:
                progress_cb(max(1, len(ops)))

            dd = simulator.get_constructed_dd()
            if want_statevector:
                try:
                    return np.array(dd.get_vector(), dtype=np.complex128)
                except TimeoutError:
                    raise
                except Exception:
                    LOGGER.exception("DDSIM did not provide a statevector representation.")
                    return None
            return _DecisionDiagramResult(simulator=simulator, decision_diagram=dd)
