from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import threading, time, logging
from collections import defaultdict

from .SSD import SSD, PartitionNode
from .backends.sv import StatevectorBackend, estimate_sv_bytes
from .backends.dd import DecisionDiagramBackend
from .backends.tableau import TableauBackend

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
    try:
        import os
        if cfg.max_workers <= 0:
            cfg.max_workers = max(1, min(4, (os.cpu_count() or 2)))
    except Exception:
        cfg.max_workers = 2
    cap_bytes = int(cfg.max_ram_gb * (1024**3))
    memgov = _MemGovernor(cap_bytes)

    statuses: Dict[int, Dict[str, Any]] = {}
    lock = threading.Lock()
    done_evt = threading.Event()

    # per-partition progress counters
    progress = defaultdict(lambda: {"done": 0, "total": 0, "last_ts": None})

    chains = _group_chains(ssd)

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
                        "statevector_len": None if out is None else len(out),
                        "chain_id": cid,
                        "seq_index": node.meta.get("seq_index", 0),
                        "total_gates": int(total_gates),
                        "num_qubits": n,
                        "want_statevector": want_sv,
                        "mem_bytes": int(need),
                    }
                init_state = out
            except Exception as e:
                elapsed = time.time() - start
                with lock:
                    statuses[pid] = {
                        "status": "error",
                        "backend": node.backend,
                        "elapsed_s": elapsed,
                        "error": f"{type(e).__name__}: {e}",
                        "chain_id": cid,
                        "seq_index": node.meta.get("seq_index", 0),
                        "total_gates": int(total_gates),
                        "num_qubits": n,
                        "want_statevector": want_sv,
                        "mem_bytes": int(need),
                    }
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
    return {
        "results": results,
        "meta": {"max_ram_gb": cfg.max_ram_gb, "max_workers": cfg.max_workers, "wall_elapsed_s": wall},
    }
