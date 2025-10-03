
from __future__ import annotations
from typing import Any, Optional

def tableau_to_dd(tableau_like: Any) -> Optional[Any]:
    """Convert tableau -> decision diagram via statevector (stopgap).

    Returns a Qiskit circuit that initializes the computed statevector, which can be fed to ddsim.
    """
    try:
        from qiskit.quantum_info import Statevector
        from qiskit import QuantumCircuit
        import numpy as np
        sv = Statevector.from_instruction(tableau_like)
        n = sv.num_qubits
        qc = QuantumCircuit(n)
        qc.initialize(np.array(sv.data, dtype=complex), list(range(n)))
        return qc
    except Exception:
        return None
