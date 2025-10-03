
from __future__ import annotations
from typing import Optional, Any
import numpy as np

def ddsim_available() -> bool:
    try:
        import mqt.ddsim  # type: ignore
        return True
    except Exception:
        return False

class DecisionDiagramBackend:
    def run(self, circuit: Any) -> Optional[np.ndarray]:
        try:
            import mqt.ddsim as ddsim  # type: ignore
            sim = ddsim.DDSIMProvider().get_backend('qasm_simulator')
            job = sim.run(circuit, shots=0)  # no shots -> statevector
            # Some versions return 'statevector' in result.data()
            res = job.result()
            if hasattr(res, "get_statevector"):
                sv = res.get_statevector(circuit)
                return np.asarray(sv, dtype=np.complex128)
            data = res.data(circuit)
            if "statevector" in data:
                return np.asarray(data["statevector"], dtype=np.complex128)
        except Exception:
            pass
        return None
