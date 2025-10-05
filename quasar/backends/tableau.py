from __future__ import annotations
from typing import Optional, Any, Callable, Sequence, Dict, Tuple, List

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore

# --------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------

def stim_available() -> bool:
    try:
        import stim  # type: ignore
        return True
    except Exception:
        return False

# --------------------------------------------------------------------------------------
# Gate set and mapping
# --------------------------------------------------------------------------------------

# Gates we treat as Clifford
CLIFFORD = {"i","id","x","y","z","h","s","sdg","cx","cz","swap"}

# Map lowercase names to Stim TableauSimulator method names.
_STIM_ALIASES: Dict[str, str] = {
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "s_dag",
    "cx": "cx",
    "cz": "cz",
    "swap": "swap",
}

def _is_supported(inst) -> bool:
    try:
        name = inst.name.lower()
    except Exception:
        return False
    return name in CLIFFORD

# --------------------------------------------------------------------------------------
# Qiskit helpers (fallback / materialization)
# --------------------------------------------------------------------------------------

def _extract_ops_qiskit(circuit: Any) -> Tuple[int, List[Tuple[str, Tuple[int, ...]]]]:
    """
    Extract (n_qubits, ops) from a Qiskit QuantumCircuit.
    Each op is ('name_lower', (q0, q1?)).
    If a non-Clifford is found, return (n, []) to signal unsupported.
    """
    try:
        qubits = list(circuit.qubits)
        q2i = {q: i for i, q in enumerate(qubits)}
        ops: List[Tuple[str, Tuple[int, ...]]] = []
        for inst, qargs, _ in circuit.data:
            name = inst.name.lower()
            if name == "barrier":
                continue
            if not _is_supported(inst):
                return len(qubits), []  # unsupported
            idxs = tuple(q2i[q] for q in qargs)
            ops.append((name, idxs))
        return len(qubits), ops
    except Exception:
        return 0, []

def _clifford_ops_to_statevector(n: int, ops: List[Tuple[str, Tuple[int, ...]]]) -> Optional[np.ndarray]:
    """
    Convert a Clifford defined by 'ops' on n qubits into a statevector without
    simulating a full statevector time-evolution per gate.
    Strategy: build a Qiskit Clifford and then Clifford->Statevector once.
    """

    if np is None:
        return None
    # We build a pure-Clifford circuit and then call Clifford.from_circuit (or Clifford(circuit))
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, Statevector

    qc = QuantumCircuit(n)
    for name, q in ops:
        if name in ("i", "id"):
            continue
        getattr(qc, name)(*q)  # Qiskit has x,y,z,h,s,sdg,cx,cz,swap
    try:
        clf = Clifford.from_circuit(qc)  # preferred in modern Qiskit
    except Exception:
        clf = Clifford(qc)               # fallback for older versions
    sv = Statevector.from_instruction(clf)
    return np.asarray(sv.data, dtype=np.complex128)

# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------

class TableauBackend:
    def run(
        self,
        circuit: Any,
        *,
        progress_cb: Optional[Callable[[int], None]] = None,
        want_statevector: bool = False,   # <-- default False (only materialize when needed)
    ) -> Optional[np.ndarray]:
        """
        Simulate Clifford circuit as a stabilizer (Stim fast path if available).
        - If a non-Clifford is present: return None (caller should pick another backend).
        - If want_statevector=True: materialize |psi> at the end via Clifford->Statevector.
        - progress_cb: optional; called occasionally with 'inc' processed gates (batched).
        """
        n, ops = _extract_ops_qiskit(circuit)
        if n == 0 or not ops:
            return None  # empty or unsupported

        # progress batching (low overhead)
        BATCH = 1000
        processed = 0

        if stim_available():
            import stim  # type: ignore
            sim = stim.TableauSimulator()
            sim.set_num_qubits(n)
            for name, q in ops:
                lname = name.lower()
                if lname in ("i", "id"):
                    # no-op
                    pass
                else:
                    meth_name = _STIM_ALIASES.get(lname)
                    if meth_name is None:
                        # shouldn't happen (we filtered), safeguard:
                        return None
                    getattr(sim, meth_name)(*q)
                processed += 1
                if progress_cb and (processed % BATCH == 0):
                    progress_cb(BATCH)
            if progress_cb and (processed % BATCH):
                progress_cb(processed % BATCH)

            if not want_statevector:
                # We simulated the prefix quickly; no SV needed.
                return None

            # Need a statevector for the next partition: build once via Clifford->SV
            if np is None:
                return None
            return _clifford_ops_to_statevector(n, ops)

        # Fallback (no Stim): still avoid per-gate statevector simulation.
        # Build a Clifford with Qiskit and materialize once if requested.
        if progress_cb:
            progress_cb(len(ops))  # coarse progress update
        if not want_statevector:
            return None
        if np is None:
            return None
        return _clifford_ops_to_statevector(n, ops)
