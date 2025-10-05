
from __future__ import annotations
from typing import Any
import numpy as np

def tableau_to_statevector(tableau_like: Any) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(tableau_like)
    return np.asarray(sv.data)
