
from __future__ import annotations
from typing import Dict, Any, Literal, Optional, List
from time import perf_counter

from ..analyzer import analyze
from ..backends.sv import StatevectorBackend, estimate_sv_bytes
from ..backends.dd import DecisionDiagramBackend, ddsim_available
from ..backends.tableau import TableauBackend, stim_available

Which = Literal["tableau","sv","dd"]

def _run_backend(which: Which, circuit, *, max_ram_gb: float | None = None) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        if which == "sv":
            if max_ram_gb is not None:
                cap = int(max_ram_gb * (1024**3))
                need = estimate_sv_bytes(getattr(circuit, "num_qubits", 0))
                if need > cap:
                    return {"ok": False, "backend": "sv", "error": f"SV exceeds cap ({need} > {cap} bytes)", "elapsed_s": 0.0}
            out = StatevectorBackend().run(circuit)
            t1 = perf_counter()
            return {"ok": out is not None, "backend": "sv", "statevector_len": None if out is None else len(out),
                    "elapsed_s": t1 - t0, "error": None if out is not None else "SV failed (backend returned None)"}
        elif which == "dd":
            if not ddsim_available():
                return {"ok": False, "backend": "dd", "error": "mqt.ddsim not available", "elapsed_s": 0.0}
            out = DecisionDiagramBackend().run(circuit)
            t1 = perf_counter()
            return {"ok": out is not None, "backend": "dd", "statevector_len": None if out is None else len(out),
                    "elapsed_s": t1 - t0, "error": None if out is not None else "DD failed (backend returned None)"}
        else:  # tableau
            out = TableauBackend().run(circuit)
            t1 = perf_counter()
            return {"ok": out is not None, "backend": "tableau", "statevector_len": None if out is None else len(out),
                    "elapsed_s": t1 - t0, "error": None if out is not None else "Tableau failed (non-Clifford or None)"}
    except Exception as e:
        t1 = perf_counter()
        return {"ok": False, "backend": which, "error": f"{type(e).__name__}: {e}", "elapsed_s": t1 - t0}

def run_single_baseline(circuit, which: Which, *, per_partition: bool = False, max_ram_gb: float | None = None) -> Dict[str, Any]:
    if not per_partition:
        res = _run_backend(which, circuit, max_ram_gb=max_ram_gb)
        return {"mode": "whole", "which": which, "result": res}
    analysis = analyze(circuit)
    parts = analysis.ssd.partitions
    results: List[Dict[str, Any]] = []
    total_s = 0.0
    ok_all = True
    for p in parts:
        sub = p.circuit
        r = _run_backend(which, sub, max_ram_gb=max_ram_gb)
        results.append({"partition": p.id, **r, "num_qubits": getattr(sub, "num_qubits", None)})
        total_s += r.get("elapsed_s", 0.0)
        ok_all = ok_all and bool(r.get("ok", False))
    return {"mode": "per_partition", "which": which, "ok": ok_all, "elapsed_s": total_s, "partitions": results,
            "num_partitions": len(parts)}

def run_baselines(circuit, *, which: Optional[List[Which]] = None, per_partition: bool = False, max_ram_gb: float | None = None) -> Dict[str, Any]:
    which = which or ["tableau", "sv", "dd"]
    payload: Dict[str, Any] = {"per_partition": per_partition, "entries": []}
    for w in which:
        payload["entries"].append(run_single_baseline(circuit, w, per_partition=per_partition, max_ram_gb=max_ram_gb))
    return payload
