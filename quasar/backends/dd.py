
from __future__ import annotations
from typing import Optional, Any


try:
    from mqt.core.dd import VectorDD  # type: ignore
except Exception:  # pragma: no cover - optional dependency during type checking
    VectorDD = Any  # type: ignore[misc,assignment]

def ddsim_available() -> bool:
    try:
        import mqt.ddsim  # type: ignore
        return True
    except Exception:
        return False

class DecisionDiagramBackend:
    def run(self, circuit: Any) -> Optional[VectorDD]:
        try:
            import mqt.ddsim as ddsim  # type: ignore
            from mqt.core import QuantumComputation  # type: ignore
            from qiskit.qasm3 import dumps as qasm3_dumps
        except Exception:
            return None

        try:
            qasm = qasm3_dumps(circuit)
            qc = QuantumComputation.from_qasm_str(qasm)
            simulator = ddsim.CircuitSimulator(qc)
            simulator.simulate(0)
            return simulator.get_constructed_dd()
        except Exception:
            return None
