
from __future__ import annotations

import logging
import multiprocessing
import queue
import signal
import traceback
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional

from ..analyzer import analyze
from ..backends.dd import DecisionDiagramBackend, ddsim_available
from ..backends.hybridq import (
    HybridQBackend,
    HybridQConversionError,
    hybridq_available,
)
from ..backends.sv import StatevectorBackend, estimate_sv_bytes
from ..backends.tableau import TableauBackend, stim_available

Which = Literal["tableau", "sv", "dd", "hybridq"]

LOG = logging.getLogger(__name__)


@contextmanager
def _deadline(seconds: float | None):
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - asynchronous
        raise TimeoutError(f"operation exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

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

def _run_backend_impl(
    which: Which,
    circuit,
    *,
    max_ram_gb: float | None = None,
    sv_ampops_per_sec: float | None = None,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    t0 = perf_counter()
    with _deadline(timeout_s):
        if which == "dd":
            if not ddsim_available():
                est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                est["note"] = "DD unavailable; worst-case SV bound"
                return {
                    "ok": False,
                    "backend": "dd",
                    "error": "mqt.ddsim not available",
                    "elapsed_s": 0.0,
                    "wall_s_measured": 0.0,
                    "estimate": est,
                }

            backend = DecisionDiagramBackend()
            out = backend.run(circuit)
            t1 = perf_counter()
            if out is None:
                return {
                    "ok": False,
                    "backend": "dd",
                    "statevector_len": None,
                    "elapsed_s": t1 - t0,
                    "wall_s_measured": t1 - t0,
                    "mem_bytes": 256 * 1024 * 1024,
                    "error": "DD failed (backend returned None)",
                }

            try:
                state_len = len(out)  # type: ignore[arg-type]
            except Exception:
                state_len = None
            return {
                "ok": True,
                "backend": "dd",
                "statevector_len": state_len,
                "elapsed_s": t1 - t0,
                "wall_s_measured": t1 - t0,
                "mem_bytes": 256 * 1024 * 1024,
                "error": None,
            }

        if which == "sv":
            if max_ram_gb is not None:
                cap = int(max_ram_gb * (1024**3))
                need = estimate_sv_bytes(getattr(circuit, "num_qubits", 0))
                if need > cap:
                    est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                    result = {
                        "ok": False,
                        "backend": "sv",
                        "error": f"SV exceeds cap ({need} > {cap} bytes)",
                        "elapsed_s": 0.0,
                        "wall_s_measured": 0.0,
                        "estimate": est,
                    }
                    time_est = est.get("time_est_sec")
                    if time_est is not None:
                        result["wall_s_estimated"] = float(time_est)
                    mem_est = est.get("mem_bytes")
                    if mem_est is not None:
                        result["mem_bytes_estimated"] = int(mem_est)
                    return result
            out = StatevectorBackend().run(circuit)
            t1 = perf_counter()
            return {
                "ok": out is not None,
                "backend": "sv",
                "statevector_len": None if out is None else len(out),
                "elapsed_s": t1 - t0,
                "wall_s_measured": t1 - t0,
                "mem_bytes": estimate_sv_bytes(getattr(circuit, "num_qubits", 0)),
                "error": None if out is not None else "SV failed (backend returned None)",
            }

        if which == "tableau":
            out = TableauBackend().run(circuit)
            t1 = perf_counter()
            # If non-Clifford, provide no strict estimate; report metrics only
            if out is None:
                est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                est["note"] = "Not Clifford; tableau inapplicable. SV worst-case bound provided."
                result = {
                    "ok": False,
                    "backend": "tableau",
                    "elapsed_s": 0.0,
                    "wall_s_measured": 0.0,
                    "error": "Tableau failed (non-Clifford)",
                    "estimate": est,
                }
                mem_est = est.get("mem_bytes")
                if mem_est is not None:
                    result["mem_bytes_estimated"] = int(mem_est)
                time_est = est.get("time_est_sec")
                if time_est is not None:
                    result["wall_s_estimated"] = float(time_est)
                return result

            return {
                "ok": True,
                "backend": "tableau",
                "statevector_len": len(out),
                "elapsed_s": t1 - t0,
                "wall_s_measured": t1 - t0,
                "mem_bytes": 64 * 1024 * 1024,
                "error": None,
            }

        if which == "hybridq":
            if not hybridq_available():
                est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
                est["note"] = "HybridQ unavailable; worst-case SV bound"
                return {
                    "ok": False,
                    "backend": "hybridq",
                    "error": "hybridq not available",
                    "elapsed_s": 0.0,
                    "wall_s_measured": 0.0,
                    "estimate": est,
                }

            backend = HybridQBackend()
            try:
                result = backend.run(circuit, want_statevector=False)
            except HybridQConversionError as exc:
                t_fail = perf_counter()
                return {
                    "ok": False,
                    "backend": "hybridq",
                    "statevector_len": None,
                    "elapsed_s": t_fail - t0,
                    "wall_s_measured": t_fail - t0,
                    "error": str(exc),
                }

            t1 = perf_counter()
            if result is None:
                return {
                    "ok": False,
                    "backend": "hybridq",
                    "statevector_len": None,
                    "elapsed_s": t1 - t0,
                    "wall_s_measured": t1 - t0,
                    "error": "HybridQ backend returned no result",
                }

            wall = result.runtime_s if result.runtime_s is not None else t1 - t0
            return {
                "ok": True,
                "backend": "hybridq",
                "statevector_len": result.statevector_len,
                "elapsed_s": t1 - t0,
                "wall_s_measured": float(wall),
                "mem_bytes": None,
                "error": None,
                "info": result.info,
            }


def _backend_worker(
    which: Which,
    circuit,
    max_ram_gb: float | None,
    sv_ampops_per_sec: float | None,
    result_queue,
):  # pragma: no cover - executed in a subprocess
    try:
        result = _run_backend_impl(
            which,
            circuit,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=None,
        )
        result_queue.put({"status": "ok", "result": result})
    except TimeoutError as exc:
        result_queue.put({"status": "timeout", "message": str(exc)})
    except MemoryError as exc:
        result_queue.put({"status": "memory_error", "message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            {
                "status": "exception",
                "exc_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_backend_subprocess(
    which: Which,
    circuit,
    *,
    max_ram_gb: float | None,
    sv_ampops_per_sec: float | None,
    timeout_s: float,
):
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - fallback when fork unavailable
        ctx = multiprocessing.get_context("spawn")

    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_backend_worker,
        args=(which, circuit, max_ram_gb, sv_ampops_per_sec, result_queue),
    )
    proc.daemon = True
    proc.start()

    payload = None
    interrupted = False
    try:
        if timeout_s > 0:
            try:
                payload = result_queue.get(timeout=timeout_s)
            except queue.Empty:
                payload = {
                    "status": "timeout",
                    "message": f"operation exceeded {timeout_s} seconds",
                }
        else:
            payload = result_queue.get()
    except EOFError:  # pragma: no cover - queue closed unexpectedly
        payload = {
            "status": "exception",
            "message": "Baseline subprocess ended without producing a result",
        }
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if proc.is_alive() and (
            interrupted
            or payload is None
            or payload.get("status") in {"timeout", "exception"}
        ):
            proc.terminate()
        proc.join()
        result_queue.close()
        result_queue.cancel_join_thread()

    if payload is None:
        if proc.exitcode not in (0, None):
            raise RuntimeError(
                f"Backend subprocess exited with code {proc.exitcode}"
            )
        raise RuntimeError("Backend subprocess returned no result")

    status = payload.get("status")
    if status == "ok":
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Backend subprocess produced invalid result payload")
        return result

    if status == "timeout":
        message = payload.get(
            "message", f"Backend {which} exceeded timeout"
        )
        raise TimeoutError(message)

    if status == "memory_error":
        message = payload.get(
            "message", "Baseline subprocess ran out of memory"
        )
        raise MemoryError(message)

    if status == "exception":
        message = payload.get(
            "message", "Baseline subprocess raised an exception"
        )
        exc_type = payload.get("exc_type")
        if exc_type:
            message = f"{exc_type}: {message}"
        raise RuntimeError(message)

    raise RuntimeError(f"Unexpected backend subprocess status: {status}")


def _run_backend(
    which: Which,
    circuit,
    *,
    max_ram_gb: float | None = None,
    sv_ampops_per_sec: float | None = None,
    timeout_s: float | None = None,
):
    t0 = perf_counter()
    try:
        if timeout_s is not None and timeout_s > 0:
            return _run_backend_subprocess(
                which,
                circuit,
                max_ram_gb=max_ram_gb,
                sv_ampops_per_sec=sv_ampops_per_sec,
                timeout_s=float(timeout_s),
            )

        return _run_backend_impl(
            which,
            circuit,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=timeout_s,
        )
    except TimeoutError:
        t1 = perf_counter()
        est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
        est["note"] = "Baseline execution timed out; theoretical estimate used."
        LOG.warning("Baseline backend=%s timed out after %ss; returning estimate", which, timeout_s)
        result = {
            "ok": False,
            "backend": which,
            "error": f"Timeout after {timeout_s}s",
            "elapsed_s": t1 - t0,
            "wall_s_measured": t1 - t0,
            "estimate": est,
        }
        time_est = est.get("time_est_sec")
        if time_est is not None:
            result["wall_s_estimated"] = float(time_est)
        mem_est = est.get("mem_bytes")
        if mem_est is not None:
            result["mem_bytes_estimated"] = int(mem_est)
        return result
    except MemoryError as e:
        t1 = perf_counter()
        est = _estimate_sv(circuit, ampops_per_sec=sv_ampops_per_sec)
        est["note"] = "Baseline ran out of memory; theoretical estimate used."
        LOG.warning("Baseline backend=%s raised MemoryError; returning estimate", which)
        result = {
            "ok": False,
            "backend": which,
            "error": f"MemoryError: {e}",
            "elapsed_s": t1 - t0,
            "wall_s_measured": t1 - t0,
            "estimate": est,
        }
        time_est = est.get("time_est_sec")
        if time_est is not None:
            result["wall_s_estimated"] = float(time_est)
        mem_est = est.get("mem_bytes")
        if mem_est is not None:
            result["mem_bytes_estimated"] = int(mem_est)
        return result
    except Exception as e:
        t1 = perf_counter()
        return {
            "ok": False,
            "backend": which,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": t1 - t0,
            "wall_s_measured": t1 - t0,
        }

def run_single_baseline(circuit, which: Which, *, per_partition: bool = False,
                        max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None,
                        timeout_s: float | None = None) -> Dict[str, Any]:
    if not per_partition:
        res = _run_backend(
            which,
            circuit,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=timeout_s,
        )
        if "wall_s_measured" not in res and "elapsed_s" in res:
            res = dict(res)
            res["wall_s_measured"] = res.get("elapsed_s", 0.0)
        return {"mode": "whole", "which": which, "result": res}
    # Per-partition mode retained for experimentation
    a = analyze(circuit)
    parts = a.plan.qusds
    results: List[Dict[str, Any]] = []
    total_s = 0.0
    ok_all = True
    for p in parts:
        sub = p.circuit
        r = _run_backend(
            which,
            sub,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=timeout_s,
        )
        if "wall_s_measured" not in r and "elapsed_s" in r:
            r = dict(r)
            r["wall_s_measured"] = r.get("elapsed_s", 0.0)
        results.append({"partition": p.id, **r, "num_qubits": getattr(sub, "num_qubits", None)})
        total_s += r.get("elapsed_s", 0.0)
        ok_all = ok_all and bool(r.get("ok", False))
    return {"mode": "per_partition", "which": which, "ok": ok_all, "elapsed_s": total_s,
            "wall_s_measured": total_s, "qusds": results, "num_qusds": len(parts)}

def run_baselines(circuit, *, which: Optional[List[Which]] = None, per_partition: bool = False,
                  max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None,
                  timeout_s: float | None = None, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = log or LOG
    which = which or ["tableau", "sv", "dd", "hybridq"]
    payload: Dict[str, Any] = {"per_partition": per_partition, "entries": []}
    for w in which:
        logger.info("Starting baseline backend=%s", w)
        entry = run_single_baseline(
            circuit,
            w,
            per_partition=per_partition,
            max_ram_gb=max_ram_gb,
            sv_ampops_per_sec=sv_ampops_per_sec,
            timeout_s=timeout_s,
        )
        payload["entries"].append(entry)
        result = entry.get("result") if not per_partition else entry
        if isinstance(result, dict):
            logger.info(
                "Finished baseline backend=%s ok=%s wall_s=%s error=%s",
                w,
                result.get("ok"),
                result.get("wall_s_measured"),
                result.get("error"),
            )
    return payload
