
from __future__ import annotations
import argparse, time, json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit
except Exception as e:
    raise SystemExit("qiskit is required for calibration. Install with: pip install qiskit") from e

# Import slim backends (Tableau + SV)
from .backends.tableau import TableauBackend, stim_available
from .backends.sv import StatevectorBackend
# We'll reuse the same gate classification as the planner
CLIFFORD = {"i","x","y","z","h","s","sdg","cx","cz","swap"}

def _gate_name(inst) -> str:
    try:
        return inst.name.lower()
    except Exception:
        return str(inst).lower()

def _split_at_first_nonclifford(qc: QuantumCircuit) -> Optional[Tuple[int, list, list]]:
    split = None
    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        if _gate_name(inst) not in CLIFFORD:
            split = idx
            break
    if split is None or split == 0:
        return None
    return split, qc.data[:split], qc.data[split:]

def _count_ops(ops: list) -> Tuple[int,int]:
    oneq = twoq = 0
    for inst, qargs, _ in ops:
        if len(qargs) >= 2:
            twoq += 1
        else:
            oneq += 1
    return oneq, twoq

def _build_subcircuit_like(parent: QuantumCircuit, ops: list) -> QuantumCircuit:
    sub = QuantumCircuit(parent.num_qubits)
    for inst, qargs, cargs in ops:
        sub.append(inst, qargs, cargs)
    return sub

def build_clifford_tail(n=12, depth_cliff=150, tail_layers=10, angle_scale=0.1, seed=7) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    # Random Clifford layers
    cliff1 = ["h","s","sdg","x","z"]
    cliff2 = ["cx","cz","swap"]
    for _ in range(depth_cliff):
        for q in range(n):
            getattr(qc, rng.choice(cliff1))(q)
        order = list(range(n))
        rng.shuffle(order)
        for a,b in zip(order[::2], order[1::2]):
            getattr(qc, rng.choice(cliff2))(a,b)
    # Non-Clifford tail (rx + cz)
    for _ in range(tail_layers):
        for q in range(n):
            theta = float(rng.uniform(-angle_scale, angle_scale))
            qc.rx(theta, q)
        order = list(range(n))
        rng.shuffle(order)
        for a,b in zip(order[::2], order[1::2]):
            qc.cz(a,b)
    return qc

@dataclass
class CalibSpec:
    n: int
    depth_cliff: int
    tail_layers: int
    angle_scale: float
    seed: int

def calibrate(specs: List[CalibSpec], *, twoq_factor: float, out_path: str | None = None) -> Dict[str, Any]:
    if not stim_available():
        raise SystemExit("Stim (Tableau) not available. Install with: pip install stim")

    sv_backend = StatevectorBackend()
    tab_backend = TableauBackend()

    rows = []
    for i, sp in enumerate(specs, 1):
        qc = build_clifford_tail(n=sp.n, depth_cliff=sp.depth_cliff, tail_layers=sp.tail_layers, angle_scale=sp.angle_scale, seed=sp.seed)
        split = _split_at_first_nonclifford(qc)
        if not split:
            # no non-Clifford, skip
            continue
        idx, pre_ops, tail_ops = split
        pre = _build_subcircuit_like(qc, pre_ops)
        tail = _build_subcircuit_like(qc, tail_ops)
        one_tail, two_tail = _count_ops(tail_ops)
        tail_norm = one_tail + twoq_factor * two_tail

        t0 = time.perf_counter()
        pre_state = tab_backend.run(pre)  # returns statevector (conversion included inside)
        t1 = time.perf_counter()
        tail_state = sv_backend.run(tail, initial_state=pre_state)
        t2 = time.perf_counter()

        pre_elapsed = float(t1 - t0)
        tail_elapsed = float(t2 - t1)
        if tail_elapsed <= 0:
            continue

        # conv_factor estimate (no 2^n term needed):
        # conv_factor ≈ (prefix_time / tail_time) * tail_norm
        conv_factor_est = (pre_elapsed / tail_elapsed) * float(tail_norm)

        rows.append({
            "n": sp.n,
            "depth_cliff": sp.depth_cliff,
            "tail_layers": sp.tail_layers,
            "angle_scale": sp.angle_scale,
            "seed": sp.seed,
            "one_tail": int(one_tail),
            "two_tail": int(two_tail),
            "tail_norm": float(tail_norm),
            "prefix_elapsed_s": pre_elapsed,
            "tail_elapsed_s": tail_elapsed,
            "conv_factor_est": conv_factor_est,
        })

    if not rows:
        raise SystemExit("No calibration rows produced (check Stim installation and circuit parameters).")

    # Aggregate
    vals = sorted(r["conv_factor_est"] for r in rows)
    median = float(np.median(vals))
    p25 = float(np.percentile(vals, 25))
    p75 = float(np.percentile(vals, 75))
    mean = float(np.mean(vals))

    report = {
        "twoq_factor_used": twoq_factor,
        "conv_factor": {"median": median, "p25": p25, "p75": p75, "mean": mean},
        "samples": rows,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[8,10,12,14])
    ap.add_argument("--depth-cliff", type=int, default=150)
    ap.add_argument("--tail-layers", type=int, default=10)
    ap.add_argument("--angle-scale", type=float, default=0.1)
    ap.add_argument("--samples-per-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--twoq-factor", type=float, default=4.0)
    ap.add_argument("--out", type=str, default="calibration_conv_factor.json")
    args = ap.parse_args()

    specs: List[CalibSpec] = []
    base_seed = int(args.seed)
    for n in args.n:
        for k in range(args.samples_per_n):
            specs.append(CalibSpec(n=n, depth_cliff=args.depth_cliff, tail_layers=args.tail_layers,
                                   angle_scale=args.angle_scale, seed=base_seed + 17*k))

    rep = calibrate(specs, twoq_factor=float(args.twoq_factor), out_path=args.out)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
