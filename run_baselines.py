
from __future__ import annotations
import argparse, json
from typing import List
from QuASAr.baselines import run_baselines
import benchmark_circuits as bench

def load_circuit(kind: str, **kwargs):
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
    p.add_argument("--which", type=str, default="all", help="sv,dd,tableau or comma-separated")
    p.add_argument("--per-partition", action="store_true", help="run backend on each independent partition separately")
    p.add_argument("--max-ram-gb", type=float, default=None, help="cap SV; skip if exceeds")
    p.add_argument("--out", type=str, default="baselines.json")
    args = p.parse_args()

    circ = load_circuit(args.kind, num_qubits=args.num_qubits, block_size=args.block_size, depth=args.depth, seed=args.seed)

    which_list: List[str]
    if args.which == "all":
        which_list = ["tableau","sv","dd"]
    else:
        which_list = [w.strip() for w in args.which.split(",") if w.strip() in {"sv","dd","tableau"}]
        if not which_list:
            raise SystemExit("No valid --which provided (use sv, dd, tableau)")

    res = run_baselines(circ, which=which_list, per_partition=args.per_partition, max_ram_gb=args.max_ram_gb)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
