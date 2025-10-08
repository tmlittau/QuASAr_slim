from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Iterable, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
try:  # pragma: no cover - fallback for stubbed qiskit in tests
    from qiskit.exceptions import QiskitError
except Exception:  # pragma: no cover - fallback when exceptions module missing
    class QiskitError(Exception):
        """Fallback error used when qiskit.exceptions is unavailable."""

from ._partition import Operation, extract_operations

LOGGER = logging.getLogger(__name__)


_AER_LOCK = threading.Lock()


class StatevectorSimulationError(RuntimeError):
    """Raised when qiskit-aer cannot provide a statevector result."""


def estimate_sv_bytes(n_qubits: int) -> int:
    if n_qubits <= 0:
        return 0
    return 16 * (1 << n_qubits)


def _apply_operation(circuit: QuantumCircuit, op: Operation) -> None:
    name = op.name
    qubits = op.qubits
    params = op.params

    if name in {"i", "id"}:
        return
    if name in {"u", "u3"}:
        theta, phi, lam = (params + (0.0, 0.0, 0.0))[:3]
        circuit.u(theta, phi, lam, qubits[0])
        return
    if name == "u2":
        phi, lam = (params + (0.0, 0.0))[:2]
        circuit.u2(phi, lam, qubits[0])
        return
    if name in {"p", "u1"}:
        circuit.p(params[0] if params else 0.0, qubits[0])
        return
    if name in {"rx", "ry", "rz"}:
        getattr(circuit, name)(params[0] if params else 0.0, qubits[0])
        return
    if name == "cp":
        circuit.cp(params[0] if params else 0.0, qubits[0], qubits[1])
        return
    if name == "crx":
        circuit.crx(params[0] if params else 0.0, qubits[0], qubits[1])
        return
    if name == "rzz":
        circuit.rzz(params[0] if params else 0.0, qubits[0], qubits[1])
        return

    method = getattr(circuit, name, None)
    if method is None:
        raise NotImplementedError(f"Unsupported gate '{op.name}' for statevector backend")
    method(*qubits)


class StatevectorBackend:
    """Simulate partitions by delegating to :mod:`qiskit-aer`."""

    @staticmethod
    def _collect_result_errors(result: Any) -> Iterable[str]:
        status = getattr(result, "status", None)
        if isinstance(status, str) and status and status.lower() not in {"success", "done", "completed"}:
            yield status.strip()
        experiments = getattr(result, "results", None)
        if isinstance(experiments, Iterable):
            for idx, entry in enumerate(experiments):
                entry_status = getattr(entry, "status", None)
                if not isinstance(entry_status, str):
                    continue
                clean = entry_status.strip()
                if not clean:
                    continue
                lowered = clean.lower()
                if lowered.startswith("error") or lowered.startswith("fail"):
                    yield f"[Experiment {idx}] {clean}"

    def run(
        self,
        circuit: Any,
        initial_state: Optional[np.ndarray] = None,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = True,
        batch_size: int = 1000,
    ) -> Optional[np.ndarray]:
        del batch_size

        num_qubits, ops = extract_operations(circuit)
        if num_qubits == 0 and want_statevector:
            return np.array([1.0 + 0.0j], dtype=np.complex128)
        if num_qubits == 0:
            return None

        partition_id = getattr(circuit, "partition_id", "unknown")
        LOGGER.debug(
            "Simulating partition %s on statevector backend with %d qubits and %d ops.",
            partition_id,
            num_qubits,
            len(ops),
        )

        qc = QuantumCircuit(num_qubits)
        for op in ops:
            _apply_operation(qc, op)

        run_args = {"shots": 1}
        if initial_state is not None:
            vec = np.asarray(initial_state, dtype=np.complex128).ravel()
            expected = 1 << num_qubits
            if vec.size != expected:
                raise ValueError(
                    f"Initial state has dimension {vec.size}, expected {expected} for {num_qubits} qubits.")
            run_args["initial_statevector"] = vec

        with _AER_LOCK:
            simulator = AerSimulator(method="statevector")
            executable = qc.copy()
            executable.save_statevector()
            executable = transpile(executable, simulator)

            try:
                result = simulator.run(executable, **run_args).result()
            except TimeoutError:
                # Propagate timeout so the caller can convert it into an estimate
                raise
            except Exception as exc:
                msg = f"qiskit-aer execution failed: {exc}"
                LOGGER.warning("%s", msg)
                raise StatevectorSimulationError(msg) from exc

        errors = list(self._collect_result_errors(result))
        if errors:
            message = "; ".join(errors)
            LOGGER.warning("Statevector simulator reported failure: %s", message)
            raise StatevectorSimulationError(message)

        if progress_cb is not None:
            progress_cb(max(1, len(ops)))

        if not want_statevector:
            return None

        try:
            state = result.get_statevector(executable)
        except TimeoutError:
            raise
        except QiskitError as exc:
            msg = f"Unable to fetch statevector: {exc}"
            LOGGER.warning("%s", msg)
            raise StatevectorSimulationError(msg) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Unable to fetch statevector result from qiskit-aer: %s", exc)
            raise StatevectorSimulationError(str(exc)) from exc
        return np.asarray(state, dtype=np.complex128)
