
from __future__ import annotations
from typing import Any
import numpy as np

def tableau_to_statevector(tableau_like: Any) -> np.ndarray:
    """Best-effort conversion from a stabilizer (tableau) circuit to a statevector.

    For the slim version we rely on qiskit's Statevector evaluator on the original Clifford circuit.
    If you have a native tableau object, reconstruct the equivalent qiskit circuit before calling.
    """
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(tableau_like)
    return np.asarray(sv.data)
