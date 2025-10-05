
from __future__ import annotations
from typing import Optional, Any, Callable
import numpy as np

def ddsim_available() -> bool:
    try:
        import mqt.ddsim  # type: ignore
        return True
    except Exception:
        return False

class DecisionDiagramBackend:
    def run(
        self,
        circuit: Any,
        *,
        initial_state: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = False,
        batch_size: int = 500,
    ) -> Optional[np.ndarray]:
        """Run the DDSIM backend while emitting coarse-grained progress callbacks."""

        def _emit(count: int) -> None:
            if progress_cb is not None and count:
                progress_cb(int(count))

        try:
            import mqt.ddsim as ddsim  # type: ignore
            if batch_size <= 0:
                batch_size = 500

            total_ops = 0
            try:
                total_ops = len(getattr(circuit, "data", []))
            except Exception:
                total_ops = 0

            # Try to use a step-wise simulator if available (newer DDSIM releases).
            try:
                sim_cls = getattr(ddsim, "CircuitSimulator", None)
                if sim_cls is not None:
                    step_sim = sim_cls(circuit)
                    step_fn = getattr(step_sim, "simulate", None)
                    get_sv = getattr(step_sim, "get_statevector", None)
                    processed = 0
                    if callable(step_fn):
                        while True:
                            if total_ops and processed >= total_ops:
                                break
                            step = batch_size
                            if total_ops:
                                step = min(batch_size, total_ops - processed)
                                if step <= 0:
                                    break
                            step_fn(step)
                            processed += step
                            _emit(step)
                        if want_statevector and callable(get_sv):
                            sv_data = get_sv()
                            if sv_data is not None:
                                return np.asarray(sv_data, dtype=np.complex128)
                        if not want_statevector:
                            return None
            except Exception:
                # Fall back to provider-based execution.
                pass

            sim = ddsim.DDSIMProvider().get_backend('qasm_simulator')
            run_opts: dict[str, Any] = {"shots": 0}
            if initial_state is not None:
                run_opts["initial_statevector"] = initial_state
            try:
                job = sim.run(circuit, **run_opts)
            except TypeError:
                # Older interfaces might not understand initial_statevector.
                run_opts.pop("initial_statevector", None)
                job = sim.run(circuit, **run_opts)
            res = job.result()
            _emit(total_ops or 1)
            if not want_statevector:
                return None
            if hasattr(res, "get_statevector"):
                sv = res.get_statevector(circuit)
                return np.asarray(sv, dtype=np.complex128)
            data = res.data(circuit)
            if "statevector" in data:
                return np.asarray(data["statevector"], dtype=np.complex128)
        except Exception:
            return None
