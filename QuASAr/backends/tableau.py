
from __future__ import annotations
from typing import Optional, Any
import numpy as np

CLIFFORD = {"i","x","y","z","h","s","sdg","cx","cz","swap"}

def stim_available() -> bool:
    try:
        import stim  # type: ignore
        return True
    except Exception:
        return False

def _is_supported(inst) -> bool:
    try:
        name = inst.name.lower()
    except Exception:
        return False
    return name in CLIFFORD

def _qiskit_clifford_statevector(circuit: Any) -> np.ndarray:
    # For now: use qiskit's Statevector for Clifford circuits (fast enough and robust).
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(circuit)
    return np.asarray(sv.data, dtype=np.complex128)

class TableauBackend:
    def run(self, circuit: Any) -> Optional[np.ndarray]:
        # If the circuit contains only supported Clifford gates, return its statevector via qiskit
        try:
            for inst, _, _ in circuit.data:
                if not _is_supported(inst):
                    # Not strictly Clifford -> refuse
                    return None
            return _qiskit_clifford_statevector(circuit)
        except Exception:
            return None
