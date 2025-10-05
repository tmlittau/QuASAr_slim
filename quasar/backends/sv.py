
from __future__ import annotations
from typing import Optional, Any
import numpy as np

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

    def run(self, circuit: Any, initial_state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        try:
            from qiskit.quantum_info import Statevector
            if initial_state is None:
                sv = Statevector.from_instruction(circuit)
            else:
                sv = Statevector(initial_state)
                sv = sv.evolve(circuit)
            return np.asarray(sv.data, dtype=np.complex128)
        except Exception:
            try:
                if self._aer is not None and initial_state is None:
                    from qiskit_aer import Aer
                    from qiskit import transpile
                    backend = Aer.get_backend('aer_simulator_statevector')
                    tqc = transpile(circuit, backend)
                    result = backend.run(tqc).result()
                    sv = result.get_statevector(tqc)
                    return np.asarray(sv, dtype=np.complex128)
            except Exception:
                pass
        return None
