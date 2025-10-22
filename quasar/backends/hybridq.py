from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ._partition import Operation, extract_operations

try:  # pragma: no cover - optional dependency
    from hybridq.circuit import Circuit as HybridQCircuit
    from hybridq.circuit.simulation import simulate as hybridq_simulate
    from hybridq.gate import Gate
except Exception:  # pragma: no cover - fallback when hybridq missing
    HybridQCircuit = None
    hybridq_simulate = None
    Gate = None


__all__ = [
    "HybridQBackend",
    "hybridq_available",
    "HybridQConversionError",
    "HybridQResult",
]


class HybridQConversionError(RuntimeError):
    """Raised when a circuit contains operations unsupported by HybridQ."""


def hybridq_available() -> bool:
    """Return ``True`` if the optional :mod:`hybridq` dependency is importable."""

    return HybridQCircuit is not None and hybridq_simulate is not None and Gate is not None


def _rotation_gate(name: str, qubits: Iterable[int], params: Iterable[float]) -> Gate:
    try:
        angle = float(next(iter(params)))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise HybridQConversionError(f"Gate '{name}' requires a rotation angle") from exc
    return Gate(name.upper(), qubits=tuple(qubits), params=(angle,))


def _phase_gate(qubits: Iterable[int], theta: float) -> Gate:
    return Gate("CPHASE", qubits=tuple(qubits), params=(float(theta),))


def _u3_gate(qubits: Iterable[int], params: Iterable[float]) -> Gate:
    cleaned = tuple(float(p) for p in params)
    if len(cleaned) != 3:
        raise HybridQConversionError("Gate 'u' expects three parameters")
    return Gate("U3", qubits=tuple(qubits), params=cleaned)


def _make_gate(op: Operation) -> List[Gate]:
    name = op.name.lower()
    qubits = list(op.qubits)
    params = list(op.params)

    if name in {"h", "x", "y", "z"}:
        return [Gate(name.upper(), qubits=tuple(qubits))]

    if name == "s":
        return [Gate("S", qubits=tuple(qubits))]

    if name == "sdg":
        gate = Gate("S", qubits=tuple(qubits))
        gate._conj()
        return [gate]

    if name == "t":
        return [Gate("T", qubits=tuple(qubits))]

    if name == "tdg":
        gate = Gate("T", qubits=tuple(qubits))
        gate._conj()
        return [gate]

    if name == "sx":
        return [Gate("SQRT_X", qubits=tuple(qubits))]

    if name == "sxdg":
        gate = Gate("SQRT_X", qubits=tuple(qubits))
        gate._conj()
        return [gate]

    if name in {"cx", "cz", "swap"}:
        return [Gate(name.upper(), qubits=tuple(qubits))]

    if name == "cp":
        if not params:
            raise HybridQConversionError("Gate 'cp' requires a rotation angle")
        return [_phase_gate(qubits, params[0])]

    if name == "rx":
        return [_rotation_gate("RX", qubits, params)]

    if name == "ry":
        return [_rotation_gate("RY", qubits, params)]

    if name == "rz":
        return [_rotation_gate("RZ", qubits, params)]

    if name == "u" or name == "u3":
        return [_u3_gate(qubits, params)]

    if name == "id":
        return []

    if name == "rzz":
        if not params:
            raise HybridQConversionError("Gate 'rzz' requires a rotation angle")
        if len(qubits) != 2:
            raise HybridQConversionError("Gate 'rzz' requires exactly two qubits")
        control, target = qubits
        angle = params[0]
        return [
            Gate("CX", qubits=(control, target)),
            Gate("RZ", qubits=(target,), params=(float(angle),)),
            Gate("CX", qubits=(control, target)),
        ]

    raise HybridQConversionError(f"Unsupported gate '{op.name}' for HybridQ backend")


@dataclass
class HybridQResult:
    statevector_len: Optional[int]
    runtime_s: Optional[float]
    info: Dict[str, Any]


class HybridQBackend:
    """Execute a circuit using the HybridQ simulator."""

    def __init__(
        self,
        *,
        optimize: str = "evolution-hybridq",
        backend: Any = "numpy",
        complex_type: str = "complex64",
        simulation_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.optimize = optimize
        self.backend = backend
        self.complex_type = complex_type
        self.simulation_options: Dict[str, Any] = dict(simulation_options or {})

    def run(
        self,
        circuit: Any,
        *,
        initial_state: Optional[str] = None,
        progress_cb: Optional[Any] = None,
        want_statevector: bool = False,
    ) -> Optional[HybridQResult]:
        if not hybridq_available():
            warnings.warn(
                "hybridq is not available; HybridQ backend cannot execute partition.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        num_qubits, operations = extract_operations(circuit)
        if num_qubits == 0:
            return HybridQResult(statevector_len=1, runtime_s=0.0, info={})

        gates: List[Gate] = []
        for op in operations:
            gates.extend(_make_gate(op))

        hybrid_circuit = HybridQCircuit(gates)
        init_state = initial_state or ("0" * num_qubits)

        options = {
            "optimize": self.optimize,
            "backend": self.backend,
            "complex_type": self.complex_type,
            "return_info": True,
            "return_numpy_array": True,
        }
        options.update(self.simulation_options)

        result = hybridq_simulate(hybrid_circuit, initial_state=init_state, **options)
        if isinstance(result, tuple) and len(result) == 2:
            state, info = result
        else:  # pragma: no cover - defensive fallback
            state, info = result, {}

        runtime = info.get("runtime (s)")
        statevector_len: Optional[int]
        if want_statevector:
            try:
                statevector_len = len(state)
            except TypeError:
                statevector_len = None
        else:
            statevector_len = None
            state = None

        if progress_cb is not None:
            try:
                progress_cb(max(1, len(operations)))
            except Exception:  # pragma: no cover - defensive
                pass

        return HybridQResult(statevector_len=statevector_len, runtime_s=runtime, info=info)
