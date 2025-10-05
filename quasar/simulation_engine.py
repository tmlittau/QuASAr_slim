
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
    max_workers: int = 0
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

def _backend_runner(name: str, circ, initial_state):
    if name == "tableau":
        return TableauBackend().run(circ)
    if name == "dd":
        return DecisionDiagramBackend().run(circ)
    return StatevectorBackend().run(circ, initial_state=initial_state)

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

    chains = _group_chains(ssd)

    def run_chain(cid: str, nodes: List[PartitionNode]):
        init_state = None
        start_chain = time.time()
        for node in nodes:
            pid = node.id
            n = int(node.metrics.get("num_qubits", 0))
            need = estimate_sv_bytes(n)
            memgov.acquire(need)
            start = time.time()
            with lock:
                statuses[pid] = {"status":"running","backend":node.backend,"start_ts":start,"chain_id":cid,"seq_index":node.meta.get("seq_index",0)}
            try:
                out = _backend_runner(node.backend or "sv", node.circuit, init_state if (node.backend or "sv")=="sv" else None)
                elapsed = time.time()-start
                with lock:
                    statuses[pid] = {"status":"ok" if out is not None else "failed",
                                     "backend":node.backend,"elapsed_s":elapsed,
                                     "statevector_len": None if out is None else len(out),
                                     "chain_id": cid, "seq_index": node.meta.get("seq_index",0)}
                init_state = out
            except Exception as e:
                elapsed = time.time()-start
                with lock:
                    statuses[pid] = {"status":"error","backend":node.backend,"elapsed_s":elapsed,"error":f"{type(e).__name__}: {e}",
                                     "chain_id": cid, "seq_index": node.meta.get("seq_index",0)}
                init_state = None
            finally:
                memgov.release(need)
        statuses[f"chain_{cid}"] = {"status":"done","chain_elapsed_s": time.time()-start_chain}

    def heartbeat():
        while not done.is_set():
            time.sleep(cfg.heartbeat_sec)
            with lock:
                running = [(pid, s) for pid, s in statuses.items() if isinstance(pid, int) and s.get("status")=="running"]
            if running:
                msg = " | ".join([f"p{pid}-{s.get('backend')} {int(time.time()-s.get('start_ts',0))}s (cid={s.get('chain_id')},seq={s.get('seq_index')})" for pid,s in running])
                LOGGER.info("[heartbeat] %s", msg)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(run_chain, cid, nodes) for cid, nodes in chains.items()]
        for f in futures:
            f.result()
    done.set()
    hb.join(timeout=1.0)

    wall = time.time() - t0
    with lock:
        results = [{"partition": pid, **st} for pid, st in sorted(((k,v) for k,v in statuses.items() if isinstance(k,int)), key=lambda kv: kv[0])]
    return {"results": results, "meta": {"max_ram_gb": cfg.max_ram_gb, "max_workers": cfg.max_workers, "wall_elapsed_s": wall}}
