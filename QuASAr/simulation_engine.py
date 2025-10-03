
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import threading, time, logging
from .SSD import SSD, PartitionNode
from .backends.sv import StatevectorBackend, estimate_sv_bytes
from .backends.dd import DecisionDiagramBackend
from .backends.tableau import TableauBackend

LOGGER = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    max_ram_gb: float = 64.0
    max_workers: int = 0           # 0 => auto
    heartbeat_sec: float = 5.0
    stuck_warn_sec: float = 60.0

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

def _backend_runner(name: str, circ):
    if name == "tableau":
        return TableauBackend().run(circ)
    if name == "dd":
        return DecisionDiagramBackend().run(circ)
    return StatevectorBackend().run(circ)

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

    statuses = {}
    lock = threading.Lock()
    done = threading.Event()

    def run_node(node: PartitionNode):
        pid = node.id
        need = 0
        if (node.backend or "sv") == "sv":
            need = estimate_sv_bytes(int(node.metrics.get("num_qubits", 0)))
            if need > cap_bytes:
                with lock:
                    statuses[pid] = {"status":"skipped","reason":"sv_exceeds_cap","backend":node.backend,"elapsed_s":0.0}
                return
            memgov.acquire(need)
        start = time.time()
        with lock:
            statuses[pid] = {"status":"running","backend":node.backend,"start_ts":start}
        try:
            out = _backend_runner(node.backend or "sv", node.circuit)
            elapsed = time.time()-start
            with lock:
                statuses[pid] = {"status":"ok" if out is not None else "failed",
                                 "backend":node.backend,"elapsed_s":elapsed,
                                 "statevector_len": None if out is None else len(out)}
        except Exception as e:
            elapsed = time.time()-start
            with lock:
                statuses[pid] = {"status":"error","backend":node.backend,"elapsed_s":elapsed,"error":f"{type(e).__name__}: {e}"}
        finally:
            if need > 0:
                memgov.release(need)

    # heartbeat
    def heartbeat():
        last_warn = {}
        while not done.is_set():
            time.sleep(cfg.heartbeat_sec)
            with lock:
                running = [(pid, s) for pid, s in statuses.items() if s.get("status")=="running"]
            if running:
                msg = " | ".join([f"p{pid}-{s.get('backend')} {int(time.time()-s.get('start_ts',0))}s" for pid,s in running])
                LOGGER.info("[heartbeat] %s", msg)
                now = time.time()
                for pid, s in running:
                    if now - s.get("start_ts", now) > cfg.stuck_warn_sec and last_warn.get(pid, 0) + cfg.stuck_warn_sec <= now:
                        LOGGER.warning("partition %d running > %.0fs", pid, cfg.stuck_warn_sec)
                        last_warn[pid] = now

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    futures = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        for node in ssd.partitions:
            futures.append(ex.submit(run_node, node))
        for f in as_completed(futures):
            _ = f.result()
    done.set()
    hb.join(timeout=1.0)

    with lock:
        results = [{"partition": pid, **st} for pid, st in sorted(statuses.items(), key=lambda kv: kv[0])]
    return {"results": results, "meta": {"max_ram_gb": cfg.max_ram_gb, "max_workers": cfg.max_workers}}
