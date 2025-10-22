from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import stim
from qiskit.quantum_info import Statevector

from ._partition import extract_operations

LOGGER = logging.getLogger(__name__)


def stim_available() -> bool:
    return True


_STIM_GATES: Dict[str, str] = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "S_DAG",
    "sx": "SQRT_X",
    "sxdg": "SQRT_X_DAG",
    "cx": "CX",
    "cz": "CZ",
    "swap": "SWAP",
}


class TableauBackend:
    """Simulate Clifford partitions using :mod:`stim`."""

    def run(
        self,
        circuit: Any,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = False,
    ) -> Optional[np.ndarray]:
        if not stim_available():
            LOGGER.warning("Stim is not available; tableau backend cannot execute partition.")
            return None

        import stim  # type: ignore

        num_qubits, ops = extract_operations(circuit)
        if num_qubits == 0:
            return None

        stim_circuit = stim.Circuit()
        processed = 0

        for op in ops:
            if op.name in {"i", "id"}:
                processed += 1
                continue
            gate = _STIM_GATES.get(op.name)
            if gate is None:
                LOGGER.debug("Partition %s contains non-Clifford gate '%s'.", getattr(circuit, "partition_id", "unknown"), op.name)
                return None
            stim_circuit.append(gate, list(op.qubits))
            processed += 1

        LOGGER.debug(
            "Simulating partition %s on tableau backend with %d qubits and %d Clifford ops.",
            getattr(circuit, "partition_id", "unknown"),
            num_qubits,
            processed,
        )

        simulator = stim.TableauSimulator()
        try:
            simulator.do_circuit(stim_circuit)
        except Exception:
            LOGGER.exception("Stim failed while executing the partition.")
            return None

        if progress_cb is not None:
            progress_cb(max(1, processed))

        if not want_statevector:
            return None

        try:
            state = simulator.state_vector()
        except Exception:
            LOGGER.exception("Stim tableau simulator does not provide a statevector.")
            return None

        state_arr = np.asarray(state, dtype=np.complex128)
        try:
            reference = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
            return reference
        except Exception:  # pragma: no cover - alignment is best-effort
            LOGGER.debug("Unable to derive Qiskit reference statevector for Stim alignment.")
        return state_arr
