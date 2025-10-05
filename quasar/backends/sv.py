
from __future__ import annotations
from typing import Optional, Any, Callable

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore

def estimate_sv_bytes(n_qubits: int) -> int:
    if n_qubits <= 0:
        return 0
    return 16 * (1 << n_qubits)  # complex128

class StatevectorBackend:
    def __init__(self) -> None:
        self._aer = None
        try:
            from qiskit_aer import Aer  # type: ignore
            self._aer = Aer
        except Exception:
            self._aer = None

    def run(
        self,
        circuit: Any,
        initial_state: Optional[np.ndarray] = None,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = True,
        batch_size: int = 1000,
    ) -> Optional[np.ndarray]:
        """Simulate *circuit* as a statevector while emitting periodic progress callbacks."""

        if np is None:
            return None

        def _emit_progress(count: int) -> None:
            if progress_cb is not None and count:
                progress_cb(int(count))

        try:
            from qiskit.quantum_info import Statevector

            if batch_size <= 0:
                batch_size = 1000

            qubits = list(getattr(circuit, "qubits", []))
            num_qubits = getattr(circuit, "num_qubits", len(qubits)) or 0
            if not qubits and num_qubits:
                qubits = [None] * num_qubits  # fallback so len works
            qindex = {q: i for i, q in enumerate(qubits)}

            if initial_state is None:
                if num_qubits <= 0:
                    return None if not want_statevector else np.array([1.0 + 0.0j], dtype=np.complex128)
                state = Statevector.from_int(0, dims=(2,) * num_qubits)
            else:
                state = Statevector(initial_state)

            data = list(getattr(circuit, "data", []))
            processed_in_batch = 0
            for inst, qargs, _ in data:
                name = getattr(inst, "name", "")
                if name.lower() in {"barrier", "snapshot"}:
                    continue
                try:
                    indices = [qindex[q] for q in qargs]
                except Exception:
                    indices = None
                try:
                    state = state.evolve(inst, qargs=indices)
                except Exception:
                    state = state.evolve(inst)
                processed_in_batch += 1
                if processed_in_batch >= batch_size:
                    _emit_progress(processed_in_batch)
                    processed_in_batch = 0
            if processed_in_batch:
                _emit_progress(processed_in_batch)

            return np.asarray(state.data, dtype=np.complex128) if want_statevector else None
        except Exception:
            try:
                if self._aer is not None and initial_state is None:
                    from qiskit_aer import Aer
                    from qiskit import transpile

                    backend = Aer.get_backend('aer_simulator_statevector')
                    tqc = transpile(circuit, backend)
                    result = backend.run(tqc).result()
                    sv = result.get_statevector(tqc)
                    _emit_progress(len(getattr(circuit, "data", [])) or 1)
                    return np.asarray(sv, dtype=np.complex128) if want_statevector else None
            except Exception:
                pass
        return None
