
from __future__ import annotations
from typing import Any, Optional
import numpy as np

def tableau_to_dd(tableau_like: Any) -> Optional[Any]:
    try:
        from qiskit.quantum_info import Statevector
        from qiskit import QuantumCircuit
        sv = Statevector.from_instruction(tableau_like)
        n = sv.num_qubits
        qc = QuantumCircuit(n)
        qc.initialize(np.array(sv.data, dtype=complex), list(range(n)))
        return qc
    except Exception:
        return None
