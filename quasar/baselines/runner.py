
from __future__ import annotations
from typing import Dict, Any, Literal, Optional, List
from time import perf_counter
from ..analyzer import analyze
from ..backends.sv import StatevectorBackend, estimate_sv_bytes
from ..backends.dd import DecisionDiagramBackend, ddsim_available
from ..backends.tableau import TableauBackend, stim_available

Which = Literal["tableau","sv","dd"]

def _estimate_sv(circuit, *, ampops_per_sec: float | None = None) -> Dict[str, Any]:
    a = analyze(circuit)  # reuse analyzer to get global metrics
    m = a.metrics_global
    n = int(m.get("num_qubits", 0))
    total = int(m.get("num_gates", 0))
    twoq = int(m.get("two_qubit_gates", 0))
    oneq = max(0, total - twoq)
    amps = 1 << n
    # Simple op model: one-qubit ~ 1*amps, two-qubit ~ 4*amps
    amp_ops = amps * (oneq + 4 * twoq)
    mem_bytes = estimate_sv_bytes(n)
    est = {"model": "SV_OPS", "amp_ops": int(amp_ops), "mem_bytes": int(mem_bytes)}
    if ampops_per_sec and ampops_per_sec > 0:
        est["time_est_sec"] = amp_ops / float(ampops_per_sec)
    return est

def _run_backend(which: Which, circuit, *, max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        if which == "sv":
            if max_ram_gb is not None:
                cap = int(max_ram_gb * (1024**3))
                need = estimate_sv_bytes(getattr(circuit, "num_qubits", 0))
                if need > cap:
                    est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                    return {"ok": False, "backend": "sv", "error": f"SV exceeds cap ({need} > {cap} bytes)",
                            "elapsed_s": 0.0, "wall_s_measured": 0.0, "estimate": est}
            out = StatevectorBackend().run(circuit)
            t1 = perf_counter()
            return {"ok": out is not None, "backend": "sv", "statevector_len": None if out is None else len(out),
                    "elapsed_s": t1 - t0, "wall_s_measured": t1 - t0,
                    "error": None if out is not None else "SV failed (backend returned None)"}
        elif which == "dd":
            if not ddsim_available():
                # Worst-case bound falls back to SV estimates
                est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                est["note"] = "DD unavailable; worst-case SV bound"
                return {"ok": False, "backend": "dd", "error": "mqt.ddsim not available", "elapsed_s": 0.0,
                        "wall_s_measured": 0.0, "estimate": est}
            out = DecisionDiagramBackend().run(circuit)
            t1 = perf_counter()
            return {"ok": out is not None, "backend": "dd", "statevector_len": None if out is None else len(out),
                    "elapsed_s": t1 - t0, "wall_s_measured": t1 - t0,
                    "error": None if out is not None else "DD failed (backend returned None)"}
        else:  # tableau
            out = TableauBackend().run(circuit)
            t1 = perf_counter()
            # If non-Clifford, provide no strict estimate; report metrics only
            if out is None:
                est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                est["note"] = "Not Clifford; tableau inapplicable. SV worst-case bound provided."
                return {"ok": False, "backend": "tableau", "elapsed_s": 0.0, "wall_s_measured": 0.0,
                        "error": "Tableau failed (non-Clifford)", "estimate": est}
            return {"ok": True, "backend": "tableau", "statevector_len": len(out),
                    "elapsed_s": t1 - t0, "wall_s_measured": t1 - t0, "error": None}
    except Exception as e:
        t1 = perf_counter()
        return {"ok": False, "backend": which, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": t1 - t0, "wall_s_measured": t1 - t0}

def run_single_baseline(circuit, which: Which, *, per_partition: bool = False,
                        max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None) -> Dict[str, Any]:
    if not per_partition:
        res = _run_backend(which, circuit, max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec)
        if "wall_s_measured" not in res and "elapsed_s" in res:
            res = dict(res)
            res["wall_s_measured"] = res.get("elapsed_s", 0.0)
        return {"mode": "whole", "which": which, "result": res}
    # Per-partition mode retained for experimentation
    a = analyze(circuit)
    parts = a.ssd.partitions
    results: List[Dict[str, Any]] = []
    total_s = 0.0
    ok_all = True
    for p in parts:
        sub = p.circuit
        r = _run_backend(which, sub, max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec)
        if "wall_s_measured" not in r and "elapsed_s" in r:
            r = dict(r)
            r["wall_s_measured"] = r.get("elapsed_s", 0.0)
        results.append({"partition": p.id, **r, "num_qubits": getattr(sub, "num_qubits", None)})
        total_s += r.get("elapsed_s", 0.0)
        ok_all = ok_all and bool(r.get("ok", False))
    return {"mode": "per_partition", "which": which, "ok": ok_all, "elapsed_s": total_s,
            "wall_s_measured": total_s, "partitions": results, "num_partitions": len(parts)}

def run_baselines(circuit, *, which: Optional[List[Which]] = None, per_partition: bool = False,
                  max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None) -> Dict[str, Any]:
    which = which or ["tableau", "sv", "dd"]
    payload: Dict[str, Any] = {"per_partition": per_partition, "entries": []}
    for w in which:
        payload["entries"].append(run_single_baseline(circuit, w, per_partition=per_partition,
                                                      max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec))
    return payload
