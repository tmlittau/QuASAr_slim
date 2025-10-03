
from __future__ import annotations
import argparse, json, os, sys
from typing import Any
from QuASAr.analyzer import analyze
from QuASAr.planner import plan, execute, PlannerConfig
import benchmark_circuits as bench

def load_circuit(kind: str, **kwargs: Any):
    if kind == "ghz_clusters_random":
        return bench.ghz_clusters_random(**kwargs)
    if kind == "random_clifford":
        return bench.random_clifford(**kwargs)
    raise ValueError(f"unknown circuit kind '{kind}'")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, default="ghz_clusters_random")
    p.add_argument("--num-qubits", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--depth", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-ram-gb", type=float, default=64.0)
    p.add_argument("--prefer-dd", action="store_true")
    p.add_argument("--out", type=str, default="result.json")
    args = p.parse_args()

    circ = load_circuit(args.kind, num_qubits=args.num_qubits, block_size=args.block_size, depth=args.depth, seed=args.seed)
    analysis = analyze(circ)
    cfg = PlannerConfig(max_ram_gb=args.max_ram_gb, prefer_dd=args.prefer_dd)
    ssd = plan(analysis.ssd, cfg)
    payload = {
        "analysis": {
            "global": analysis.metrics_global,
            "ssd": ssd.to_dict(),
        },
        "execution": execute(ssd),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
