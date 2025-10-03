
from __future__ import annotations
from typing import Any, Optional
import numpy as np

def dd_to_statevector(dd_source: Any) -> Optional[np.ndarray]:
    """Best-effort: if dd_source is a circuit, simulate with ddsim to extract a statevector."""
    try:
        import mqt.ddsim as ddsim  # type: ignore
        sim = ddsim.DDSIMProvider().get_backend('qasm_simulator')
        job = sim.run(dd_source, shots=0)
        res = job.result()
        if hasattr(res, "get_statevector"):
            sv = res.get_statevector(dd_source)
            return np.asarray(sv, dtype=np.complex128)
        data = res.data(dd_source)
        if "statevector" in data:
            return np.asarray(data["statevector"], dtype=np.complex128)
    except Exception:
        pass
    return None
