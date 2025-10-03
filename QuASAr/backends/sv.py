
from __future__ import annotations
from typing import Optional, Any
import numpy as np

def estimate_sv_bytes(n_qubits: int) -> int:
    # complex128 amplitudes
    if n_qubits <= 0:
        return 0
    return 16 * (1 << n_qubits)

class StatevectorBackend:
    def __init__(self) -> None:
        self._aer = None
        try:
            from qiskit_aer import Aer  # type: ignore
            self._aer = Aer
        except Exception:
            self._aer = None

    def run(self, circuit: Any) -> Optional[np.ndarray]:
        try:
            if self._aer is not None:
                from qiskit_aer import Aer
                from qiskit import transpile
                backend = Aer.get_backend('aer_simulator_statevector')
                tqc = transpile(circuit, backend)
                result = backend.run(tqc).result()
                sv = result.get_statevector(tqc)
                return np.asarray(sv, dtype=np.complex128)
            else:
                # Fallback to qiskit statevector calculation
                from qiskit.quantum_info import Statevector
                sv = Statevector.from_instruction(circuit)
                return np.asarray(sv.data, dtype=np.complex128)
        except Exception as e:
            # Last resort: return None
            return None
