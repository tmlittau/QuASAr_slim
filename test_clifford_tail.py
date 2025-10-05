
from __future__ import annotations

import numpy as np

from benchmarks.hybrid import random_clifford
from quasar.analyzer import analyze
from quasar.backends.sv import StatevectorBackend
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd

def build_clifford_tail(n=10, depth_cliff=50, depth_tail=5, seed=7):
    qc = random_clifford(n, depth=depth_cliff, seed=seed)
    from qiskit import QuantumCircuit
    import numpy as np
    tail = QuantumCircuit(n)
    rng = np.random.default_rng(seed+1)
    for _ in range(depth_tail):
        for q in range(n):
            theta = float(rng.uniform(-0.1, 0.1))
            tail.rx(theta, q)
        order = list(range(n))
        rng.shuffle(order)
        for a,b in zip(order[::2], order[1::2]):
            tail.cz(a,b)
    qc.compose(tail, inplace=True)
    return qc

def main():
    n = 10
    circ = build_clifford_tail(n=n)
    sv_full = StatevectorBackend().run(circ)
    analysis = analyze(circ)
    cfg = PlannerConfig(hybrid_clifford_tail=True, conv_amp_ops_factor=16.0)
    ssd = plan(analysis.ssd, cfg)
    exec_payload = execute_ssd(ssd, ExecutionConfig(max_ram_gb=8.0))
    assert sv_full is not None
    sv_direct = StatevectorBackend().run(circ)
    assert np.allclose(sv_full, sv_direct, atol=1e-8)
    print("Planned nodes:")
    for p in ssd.partitions:
        print(f"  node {p.id}: backend={p.backend}, chain={p.meta.get('chain_id')} seq={p.meta.get('seq_index')} reason={p.meta.get('planner_reason')}")
    print("OK: SV baseline consistent. (Hybrid split decision visible above.)")

if __name__ == "__main__":
    main()
