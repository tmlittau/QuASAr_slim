
from __future__ import annotations
from typing import Optional, Any, Callable

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore

def ddsim_available() -> bool:
    try:
        import mqt.ddsim  # type: ignore
        return True
    except Exception:
        return False

class DecisionDiagramBackend:
    def run(
        self,
        circuit: Any,
        *,
        initial_state: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = False,
        batch_size: int = 500,
    ) -> Optional[Any]:
        """Simulate *circuit* with DDSIM and return a decision diagram or statevector."""

        del batch_size  # Only kept for backwards compatibility.

        if np is None:
            return None

        def _emit(count: int) -> None:
            if progress_cb is not None and count:
                progress_cb(int(count))

        try:
            import mqt.ddsim as ddsim  # type: ignore
            from mqt.core import load  # type: ignore
            from mqt.core.dd import DDPackage  # type: ignore
        except Exception:
            return None

        prepared_circuit = circuit
        total_ops = 0

        qiskit_qc: Any = None
        try:
            from qiskit import QuantumCircuit  # type: ignore

            if isinstance(circuit, QuantumCircuit):
                qiskit_qc = circuit
        except Exception:
            QuantumCircuit = None  # type: ignore

        if initial_state is not None:
            init_vec = np.asarray(initial_state, dtype=np.complex128).ravel()
            need_fallback = False
            target_qubits = None

            if qiskit_qc is not None:
                target_qubits = int(qiskit_qc.num_qubits)
                if target_qubits <= 0:
                    return None
                expected = 1 << target_qubits
                if init_vec.size != expected:
                    raise ValueError(
                        f"Initial state has dimension {init_vec.size}, expected {expected} for {target_qubits} qubits."
                    )
                try:
                    from qiskit.circuit.library import StatePreparation  # type: ignore

                    prep = QuantumCircuit(target_qubits)
                    prep.append(StatePreparation(init_vec, normalize=True), range(target_qubits))
                    prepared_circuit = prep.compose(qiskit_qc)
                except Exception:
                    need_fallback = True
            else:
                need_fallback = True

            if need_fallback:
                try:
                    from .sv import StatevectorBackend  # type: ignore

                    sv_backend = StatevectorBackend()
                    sv = sv_backend.run(
                        circuit,
                        initial_state=init_vec,
                        progress_cb=progress_cb,
                        want_statevector=True,
                    )
                except Exception:
                    sv = None

                if sv is None:
                    return None

                vector = np.asarray(sv, dtype=np.complex128)
                length = int(vector.size)
                if length <= 0 or length & (length - 1):
                    raise ValueError("Statevector length must be a power of two.")
                num_qubits = length.bit_length() - 1
                pkg = DDPackage(num_qubits)
                dd = pkg.from_vector(vector)
                return vector if want_statevector else dd

        try:
            prepared_qc = load(prepared_circuit)
            try:
                total_ops = int(getattr(prepared_qc, "num_ops", 0))
            except Exception:
                total_ops = len(getattr(prepared_circuit, "data", []))

            simulator = ddsim.CircuitSimulator(prepared_qc)
            simulator.simulate(0)
            dd = simulator.get_constructed_dd()
            _emit(total_ops or 1)

            if want_statevector:
                return np.array(dd.get_vector(), dtype=np.complex128, copy=True)
            return dd
        except Exception:
            return None
