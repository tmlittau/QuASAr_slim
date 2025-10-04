
from __future__ import annotations
import argparse, time, json
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from QuASAr.cost_estimator import CostEstimator, CostParams
import benchmark_circuits as bench

CLIFFORD = {"i","x","y","z","h","s","sdg","cx","cz","swap"}

def _gate_name(inst) -> str:
    try:
        return inst.name.lower()
    except Exception:
        return str(inst).lower()

def _split_at_first_nonclifford(qc):
    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        if _gate_name(inst) not in CLIFFORD:
            return idx
    return None

def _count_ops(ops) -> Tuple[int,int]:
    oneq = twoq = 0
    for inst, qargs, _ in ops:
        if len(qargs) >= 2:
            twoq += 1
        else:
            oneq += 1
    return oneq, twoq

def _counts_for_cutoff(qc, cutoff_idx: Optional[int]) -> Tuple[int,int,int,int]:
    pre_ops = qc.data[:cutoff_idx] if cutoff_idx is not None else qc.data
    tail_ops = qc.data[cutoff_idx:] if cutoff_idx is not None else []
    one_pre, two_pre = _count_ops(pre_ops)
    one_tail, two_tail = _count_ops(tail_ops)
    return one_pre, two_pre, one_tail, two_tail

def analyze_case(n: int, depth: int, cutoff: float, angle_scale: float,
                 est: CostEstimator) -> Dict[str, Any]:
    qc = bench.clifford_prefix_rot_tail(
        num_qubits=n, depth=depth, cutoff=cutoff, angle_scale=angle_scale, seed=42
    )
    split_idx = _split_at_first_nonclifford(qc)
    one_pre, two_pre, one_tail, two_tail = _counts_for_cutoff(qc, split_idx)
    cmp = est.compare_clifford_prefix_tail(
        n=n, one_pre=one_pre, two_pre=two_pre, one_tail=one_tail, two_tail=two_tail
    )
    norm = 1 << n
    speedup = (cmp["sv_total"]/cmp["hybrid_total"]) if cmp["hybrid_total"] > 0 else float("inf")
    return {
        "feasible": bool(speedup >= 1.0),
        "sv_total_norm": cmp["sv_total"]/norm,
        "hybrid_norm": cmp["hybrid_total"]/norm,
        "sv_tail_norm": cmp["sv_tail"]/norm,
        "conv_norm": cmp["conversion"]/norm,
        "speedup_vs_sv": speedup
    }

def find_threshold(n: int, depths: List[int], cutoff: float, angle_scale: float,
                   est: CostEstimator, target_speedup: float) -> Dict[str, Any]:
    for d in depths:
        res = analyze_case(n, d, cutoff, angle_scale, est)
        if res["speedup_vs_sv"] >= target_speedup:
            return {"n": n, "cutoff": cutoff, "first_depth": d, **res}
    return {"n": n, "cutoff": cutoff, "first_depth": None, "feasible": False}

def _is_interactive_backend() -> bool:
    return matplotlib.get_backend() in {
        "TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "GTK3Agg", "GTK4Agg", "wxAgg", "WebAgg", "nbAgg"
    }

def main():
    ap = argparse.ArgumentParser(
        description="Cutoff playground: sweep depths in [min,max] with step; "
                    "plot first depth where SV/Hybrid >= target_speedup and save thresholds if requested."
    )
    ap.add_argument("--n", type=int, nargs="+", default=[8, 10, 12, 14, 16],
                    help="List of qubit counts to sweep.")
    ap.add_argument("--min-depth", type=int, default=50, help="Minimum total depth (inclusive).")
    ap.add_argument("--max-depth", type=int, default=400, help="Maximum total depth (inclusive).")
    ap.add_argument("--step", type=int, default=1, help="Depth step size.")
    ap.add_argument("--cutoff", type=float, nargs="+", default=[0.8],
                    help="Cutoff fractions (e.g., 0.8 means 80%% Clifford prefix). Multiple values allowed.")
    ap.add_argument("--angle-scale", type=float, default=0.1)
    ap.add_argument("--conv-factor", type=float, default=64.0)
    ap.add_argument("--twoq-factor", type=float, default=4.0)
    ap.add_argument("--target-speedup", type=float, default=1.0,
                    help="Desired speedup threshold (SV_cost / Hybrid_cost). Example: 1.25 for 25%% faster.")
    ap.add_argument("--out", type=str, default=None,
                    help="If set, save figure to this path. Otherwise show if interactive; else auto-save.")
    ap.add_argument("--save-json", type=str, default=None,
                    help="If set, save threshold results to this JSON file for downstream benchmarking.")
    args = ap.parse_args()

    if args.min_depth < 1 or args.max_depth < args.min_depth:
        raise SystemExit("--max-depth must be >= --min-depth and both >= 1.")
    if args.step < 1:
        raise SystemExit("--step must be >= 1")
    if args.target_speedup <= 0:
        raise SystemExit("--target-speedup must be > 0")

    depths = list(range(int(args.min_depth), int(args.max_depth) + 1, int(args.step)))
    est = CostEstimator(CostParams(conv_amp_ops_factor=args.conv_factor,
                                   sv_twoq_factor=args.twoq_factor))

    all_lines = []  # (cutoff, ns, first_depths)
    records = []
    print("n, cutoff, target_speedup, first_depth, sv_total_norm@first, hybrid_norm@first, speedup@first")
    for c in args.cutoff:
        ns = []
        first_depths = []
        for n in args.n:
            t = find_threshold(n, depths, c, args.angle_scale, est, args.target_speedup)
            ns.append(n)
            dstar = t["first_depth"]
            rec = {"n": n, "cutoff": c, "target_speedup": args.target_speedup,
                   "first_depth": dstar,
                   "sv_total_norm_first": t.get("sv_total_norm"),
                   "hybrid_norm_first": t.get("hybrid_norm"),
                   "speedup_first": t.get("speedup_vs_sv")}
            records.append(rec)
            if dstar is not None:
                print(f"{n},{c},{args.target_speedup},{dstar},{t['sv_total_norm']:.1f},{t['hybrid_norm']:.1f},{t['speedup_vs_sv']:.3f}")
                first_depths.append(dstar)
            else:
                print(f"{n},{c},{args.target_speedup},None,,,")
                first_depths.append(np.nan)
        all_lines.append((c, ns, first_depths))

    meta = {
        "timestamp": int(time.time()),
        "params": {
            "n_list": args.n,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "step": args.step,
            "cutoff_list": args.cutoff,
            "angle_scale": args.angle_scale,
            "conv_factor": args.conv_factor,
            "twoq_factor": args.twoq_factor,
            "target_speedup": args.target_speedup,
        }
    }
    thresholds = {"meta": meta, "records": records}

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Wrote thresholds JSON: {args.save_json}")

    # Plot: x = n (qubits), y = first depth achieving target speedup
    plt.figure(figsize=(8, 5))
    for c, ns, depths_line in all_lines:
        label = f"cutoff={c:.2f}"
        plt.plot(ns, depths_line, marker="o", linewidth=2, label=label)
    plt.xlabel("Qubits (n)")
    plt.ylabel(f"First depth achieving speedup ≥ {args.target_speedup:.2f}")
    plt.title(f"Thresholds vs qubits | target speedup={args.target_speedup:.2f}\n"
              f"(conv={args.conv_factor}, twoq={args.twoq_factor}, angle_scale={args.angle_scale})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Wrote figure: {args.out}")
    else:
        plt.show()
if __name__ == "__main__":
    main()
