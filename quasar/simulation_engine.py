from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import threading, time, logging, sys, os
from collections import defaultdict

from .SSD import SSD, PartitionNode
from .backends.sv import (
    StatevectorBackend,
    StatevectorSimulationError,
    estimate_sv_bytes,
)
from .backends.dd import DecisionDiagramBackend
from .backends.tableau import TableauBackend

try:  # pragma: no cover - platform dependent
    import resource
except Exception:  # pragma: no cover - resource may be unavailable
    resource = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    max_ram_gb: float = 64.0
    max_workers: int = 0
    heartbeat_sec: float = 5.0
    stuck_warn_sec: float = 60.0
    # If True, assume a direct Tableau->DD conversion exists (no SV needed in between).
    # If False, we'll materialize a statevector before DD (tab->sv->dd).
    direct_tab_to_dd: bool = False

class _MemGovernor:
    def __init__(self, cap_bytes: int) -> None:
        self.cap = cap_bytes
        self._avail = cap_bytes
        self._cv = threading.Condition()
    def acquire(self, need: int) -> None:
        with self._cv:
            while need > self._avail:
                self._cv.wait()
            self._avail -= need
    def release(self, amount: int) -> None:
        with self._cv:
            self._avail += amount
            if self._avail > self.cap:
                self._avail = self.cap
            self._cv.notify_all()


_FALLBACK_PEAK_RSS = 0


def _get_peak_rss_bytes() -> Optional[int]:
    """Return best-effort peak resident set size in bytes."""

    global _FALLBACK_PEAK_RSS

    if resource is not None:  # pragma: no branch - prefer ru_maxrss when available
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak = getattr(usage, "ru_maxrss", None)
        except Exception:  # pragma: no cover - defensive guard
            peak = None
        if peak is not None:
            try:
                peak_int = int(peak)
            except Exception:  # pragma: no cover - defensive guard
                peak_int = 0
            if peak_int < 0:
                peak_int = 0
            if peak_int:
                scale = 1024
                if sys.platform == "darwin":
                    scale = 1  # macOS reports bytes already
                return peak_int * scale

    if psutil is not None:  # pragma: no cover - used when resource missing
        try:
            rss = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            rss = 0
        if rss > 0:
            _FALLBACK_PEAK_RSS = max(_FALLBACK_PEAK_RSS, rss)
            return _FALLBACK_PEAK_RSS

    return None


def _attach_peak_rss(status: Dict[str, Any]) -> None:
    peak = _get_peak_rss_bytes()
    if peak is not None and peak > 0:
        status["peak_rss_bytes"] = int(peak)

def _backend_runner(
    name: str,
    circ,
    initial_state,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
    want_statevector: bool = False,
    **kwargs,
):
    """
    Thin adapter so we can pass progress_cb and (for tableau) whether we want an SV materialized.
    Falls back gracefully if the backend signature doesn't accept these kwargs yet.
    """
    bname = (name or "sv").lower()
    if bname == "tableau":
        tb = TableauBackend()
        try:
            return tb.run(circ, progress_cb=progress_cb, want_statevector=want_statevector)
        except TypeError:
            return tb.run(circ)
    if bname == "dd":
        dd = DecisionDiagramBackend()
        try:
            return dd.run(
                circ,
                initial_state=initial_state,
                progress_cb=progress_cb,
                want_statevector=want_statevector,
            )
        except TypeError:
            return dd.run(circ)
    sv = StatevectorBackend()
    try:
        return sv.run(
            circ,
            initial_state=initial_state,
            progress_cb=progress_cb,
            want_statevector=want_statevector,
        )
    except TypeError:
        return sv.run(circ, initial_state=initial_state)

def _group_chains(ssd: SSD):
    chains = {}
    for node in ssd.partitions:
        meta = node.meta or {}
        chain_id = meta.get("chain_id", f"chain_{node.id}")
        seq = int(meta.get("seq_index", 0))
        node.meta["chain_id"] = chain_id
        node.meta["seq_index"] = seq
        chains.setdefault(chain_id, []).append(node)
    for cid in chains:
        chains[cid].sort(key=lambda n: int(n.meta.get("seq_index", 0)))
    return chains

def _count_gates(circ) -> int:
    """Best-effort gate count used for heartbeats/ETA."""
    try:
        return int(circ.size())  # Qiskit QuantumCircuit
    except Exception:
        try:
            return int(len(circ.data))
        except Exception:
            return 0

def _estimate_bytes_for_partition(backend: str, n_qubits: int, want_sv: bool) -> int:
    """
    Be less conservative for non-SV partitions so parallelism isn't throttled.
    - SV or when we need to materialize an SV: use statevector size.
    - Tableau without SV materialization: small constant working set.
    - DD: modest placeholder (adjust when you benchmark a better number).
    """
    b = (backend or "sv").lower()
    if b == "sv" or want_sv:
        return int(estimate_sv_bytes(n_qubits))
    if b == "tableau":
        # Stim tableau memory is tiny; reserve a small cushion (64 MiB).
        return 64 * 1024 * 1024
    if b == "dd":
        # DD usage is input-dependent; reserve a moderate cushion (256 MiB).
        return 256 * 1024 * 1024
    # Fallback
    return 128 * 1024 * 1024

def execute_ssd(ssd: SSD, cfg: Optional[ExecutionConfig] = None) -> Dict[str, Any]:
    cfg = cfg or ExecutionConfig()
    cpu_count = None
    if cfg.max_workers <= 0:
        try:
            import os

            cpu_count = os.cpu_count()
        except Exception:
            cpu_count = None
        if cpu_count is None or cpu_count <= 0:
            cpu_count = 2
    else:
        cfg.max_workers = max(1, int(cfg.max_workers))
    cap_bytes = int(cfg.max_ram_gb * (1024**3))
    memgov = _MemGovernor(cap_bytes)

    statuses: Dict[int, Dict[str, Any]] = {}
    lock = threading.Lock()
    done_evt = threading.Event()

    # per-partition progress counters
    progress = defaultdict(lambda: {"done": 0, "total": 0, "last_ts": None})

    chains = _group_chains(ssd)
    if cfg.max_workers <= 0:
        sensitive_backends = {"sv", "dd"}
        has_sensitive = any(
            str((node.backend or "sv")).lower() in sensitive_backends for node in ssd.partitions
        )
        desired = 1 if has_sensitive else max(1, len(chains))
        if cpu_count is None:
            cfg.max_workers = desired
        else:
            cfg.max_workers = max(1, min(cpu_count, desired))

    def _make_progress_cb(pid: int) -> Callable[[int], None]:
        """Backends call this with the number of newly-processed gates (default 1)."""
        def cb(inc: int = 1) -> None:
            now = time.time()
            with lock:
                pr = progress[pid]
                pr["done"] = pr.get("done", 0) + int(inc)
                pr["last_ts"] = now
        return cb

    def run_chain(cid: str, nodes: List[PartitionNode]):
        init_state = None
        start_chain = time.time()
        for idx, node in enumerate(nodes):
            pid = node.id
            n = int(node.metrics.get("num_qubits", 0))

            # Lookahead: decide whether we must materialize an SV at the end of this node.
            next_backend = None
            if idx + 1 < len(nodes):
                try:
                    next_backend = (nodes[idx + 1].backend or "sv").lower()
                except Exception:
                    next_backend = None
            want_sv = False
            if (node.backend or "sv").lower() == "tableau":
                if next_backend == "sv":
                    want_sv = True
                elif next_backend == "dd" and not cfg.direct_tab_to_dd:
                    want_sv = True

            need = _estimate_bytes_for_partition((node.backend or "sv"), n, want_sv)
            memgov.acquire(need)

            total_gates = node.metrics.get("gate_count")
            if not total_gates:
                total_gates = _count_gates(node.circuit)

            start = time.time()
            with lock:
                progress[pid] = {"done": 0, "total": int(total_gates), "last_ts": start}
                statuses[pid] = {
                    "status": "running",
                    "backend": node.backend,
                    "start_ts": start,
                    "chain_id": cid,
                    "seq_index": node.meta.get("seq_index", 0),
                    "total_gates": int(total_gates),
                    "num_qubits": n,
                    "want_statevector": want_sv,
                    "mem_bytes": int(need),
                }

            try:
                out = _backend_runner(
                    node.backend or "sv",
                    node.circuit,
                    init_state if (node.backend or "sv") == "sv" else None,
                    progress_cb=_make_progress_cb(pid),
                    want_statevector=want_sv,
                )
                elapsed = time.time() - start
                with lock:
                    # mark all gates as done on success (if backend never reported)
                    pr = progress.get(pid, {})
                    if pr and pr.get("done", 0) < total_gates:
                        pr["done"] = int(total_gates)
                        pr["last_ts"] = time.time()
                    success = (out is not None) or (not want_sv)
                    statuses[pid] = {
                        "status": "ok" if success else "failed",
                        "backend": node.backend,
                        "elapsed_s": elapsed,
                        "wall_s_measured": elapsed,
                        "statevector_len": None if out is None else len(out),
                        "chain_id": cid,
                        "seq_index": node.meta.get("seq_index", 0),
                        "total_gates": int(total_gates),
                        "num_qubits": n,
                        "want_statevector": want_sv,
                        "mem_bytes": int(need),
                    }
                    _attach_peak_rss(statuses[pid])
                init_state = out
            except StatevectorSimulationError as exc:
                elapsed = time.time() - start
                with lock:
                    pr = progress.get(pid, {})
                    if pr and pr.get("done", 0) < total_gates:
                        pr["done"] = int(total_gates)
                        pr["last_ts"] = time.time()
                    statuses[pid] = {
                        "status": "estimated",
                        "backend": node.backend,
                        "elapsed_s": elapsed,
                        "wall_s_measured": elapsed,
                        "statevector_len": None,
                        "error": str(exc),
                        "chain_id": cid,
                        "seq_index": node.meta.get("seq_index", 0),
                        "total_gates": int(total_gates),
                        "num_qubits": n,
                        "want_statevector": want_sv,
                        "mem_bytes": int(need),
                        "mem_bytes_estimated": int(need),
                    }
                    _attach_peak_rss(statuses[pid])
                LOGGER.warning(
                    "Partition %s on backend %s fell back to estimate: %s",
                    pid,
                    node.backend,
                    exc,
                )
                init_state = None
            except Exception as e:
                elapsed = time.time() - start
                with lock:
                    statuses[pid] = {
                        "status": "error",
                        "backend": node.backend,
                        "elapsed_s": elapsed,
                        "wall_s_measured": elapsed,
                        "error": f"{type(e).__name__}: {e}",
                        "chain_id": cid,
                        "seq_index": node.meta.get("seq_index", 0),
                        "total_gates": int(total_gates),
                        "num_qubits": n,
                        "want_statevector": want_sv,
                        "mem_bytes": int(need),
                    }
                    _attach_peak_rss(statuses[pid])
                init_state = None
            finally:
                memgov.release(need)

        with lock:
            statuses[f"chain_{cid}"] = {"status": "done", "chain_elapsed_s": time.time() - start_chain}

    def heartbeat():
        while not done_evt.is_set():
            time.sleep(getattr(cfg, "heartbeat_sec", 5.0))
            with lock:
                running_items = [(pid, s) for pid, s in statuses.items() if isinstance(pid, int) and s.get("status") == "running"]
            if not running_items:
                continue
            now = time.time()
            msgs = []
            for pid, s in running_items:
                start_ts = float(s.get("start_ts", 0.0))
                elapsed = max(0.0, now - start_ts)
                tot = int(s.get("total_gates", 0))
                pr = progress.get(pid, {})
                done_g = int(pr.get("done", 0))
                pct = (100.0 * done_g / tot) if tot else 0.0
                kgps = (done_g / elapsed / 1e3) if elapsed > 0 and done_g > 0 else 0.0
                last = float(pr.get("last_ts") or start_ts)
                no_prog = now - last
                stuck = (cfg.stuck_warn_sec > 0 and no_prog >= cfg.stuck_warn_sec)
                msgs.append(
                    f"p{pid}-{s.get('backend')} {int(elapsed)}s "
                    f"{done_g}/{tot} ({pct:.1f}%) @ {kgps:.1f} kG/s "
                    f"wantSV={bool(s.get('want_statevector'))} "
                    f"(cid={s.get('chain_id')},seq={s.get('seq_index')})"
                    + (" [STUCK?]" if stuck else "")
                )
            LOGGER.info("[heartbeat] %s", " | ".join(msgs))

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(run_chain, cid, nodes) for cid, nodes in chains.items()]
        for f in futures:
            f.result()
    done_evt.set()
    hb.join(timeout=1.0)

    wall = time.time() - t0
    with lock:
        results = [
            {"partition": pid, **st, **({"done_gates": progress.get(pid, {}).get("done", 0)} if isinstance(pid, int) else {})}
            for pid, st in sorted(((k, v) for k, v in statuses.items() if isinstance(k, int)), key=lambda kv: kv[0])
        ]
    peak_rss = 0
    for entry in results:
        try:
            val = int(entry.get("peak_rss_bytes", 0))
        except Exception:
            val = 0
        if val > peak_rss:
            peak_rss = val
    return {
        "results": results,
        "meta": {
            "max_ram_gb": cfg.max_ram_gb,
            "max_workers": cfg.max_workers,
            "wall_elapsed_s": wall,
            **({"peak_rss_bytes": peak_rss} if peak_rss > 0 else {}),
        },
    }
