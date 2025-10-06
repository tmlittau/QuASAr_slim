
from __future__ import annotations
import logging
import signal
import multiprocessing
import queue
import traceback
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional
from ..analyzer import analyze
from ..backends.sv import StatevectorBackend, estimate_sv_bytes
from ..backends.dd import DecisionDiagramBackend, ddsim_available
from ..backends.tableau import TableauBackend, stim_available

Which = Literal["tableau","sv","dd"]

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

def _dd_worker(circuit, result_queue):  # pragma: no cover - executed in a subprocess
    try:
        backend = DecisionDiagramBackend()
        out = backend.run(circuit)
        if out is None:
            payload = {
                "status": "ok",
                "ok": False,
                "statevector_len": None,
                "error": "DD failed (backend returned None)",
            }
        else:
            try:
                state_len = len(out)  # type: ignore[arg-type]
            except Exception:
                state_len = None
            payload = {
                "status": "ok",
                "ok": True,
                "statevector_len": state_len,
                "error": None,
            }
        result_queue.put(payload)
    except TimeoutError as exc:
        result_queue.put({"status": "timeout", "message": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            {
                "status": "exception",
                "exc_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _dd_subprocess_run(circuit, timeout_s: float | None):
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - fallback when fork unavailable
        ctx = multiprocessing.get_context("spawn")

    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_dd_worker, args=(circuit, result_queue))
    proc.daemon = True
    proc.start()

    payload = None
    interrupted = False
    try:
        if timeout_s is not None and timeout_s > 0:
            try:
                payload = result_queue.get(timeout=timeout_s)
            except queue.Empty:
                payload = {"status": "timeout", "message": f"operation exceeded {timeout_s} seconds"}
        else:
            payload = result_queue.get()
    except EOFError:  # pragma: no cover - queue closed unexpectedly
        payload = {"status": "exception", "message": "DD subprocess ended without producing a result"}
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if proc.is_alive() and (interrupted or payload is None or payload.get("status") == "timeout"):
            proc.terminate()
        proc.join()
        result_queue.close()
        result_queue.cancel_join_thread()

    if payload is None:
        if proc.exitcode not in (0, None):
            return {
                "status": "exception",
                "message": f"DD subprocess exited with code {proc.exitcode}",
            }
        return {"status": "exception", "message": "DD subprocess returned no result"}

    if payload.get("status") == "timeout" and proc.exitcode not in (0, None) and proc.exitcode != -15:
        payload = dict(payload)
        payload.setdefault("message", f"operation exceeded {timeout_s} seconds")

    return payload


def _run_backend(
    which: Which,
    circuit,
    *,
    max_ram_gb: float | None = None,
    sv_ampops_per_sec: float | None = None,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    t0 = perf_counter()
    try:
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

            payload = _dd_subprocess_run(circuit, timeout_s)
            status = payload.get("status")
            if status == "timeout":
                raise TimeoutError(payload.get("message", "DD baseline exceeded timeout"))
            if status == "exception":
                message = payload.get("message", "DD subprocess raised an exception")
                exc_type = payload.get("exc_type")
                if exc_type:
                    message = f"{exc_type}: {message}"
                raise RuntimeError(message)

            t1 = perf_counter()
            return {
                "ok": bool(payload.get("ok")),
                "backend": "dd",
                "statevector_len": payload.get("statevector_len"),
                "elapsed_s": t1 - t0,
                "wall_s_measured": t1 - t0,
                "mem_bytes": 256 * 1024 * 1024,
                "error": payload.get("error"),
            }

        with _deadline(timeout_s):
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
            else:  # tableau
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
    parts = a.ssd.partitions
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
            "wall_s_measured": total_s, "partitions": results, "num_partitions": len(parts)}

def run_baselines(circuit, *, which: Optional[List[Which]] = None, per_partition: bool = False,
                  max_ram_gb: float | None = None, sv_ampops_per_sec: float | None = None,
                  timeout_s: float | None = None, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    logger = log or LOG
    which = which or ["tableau", "sv", "dd"]
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
