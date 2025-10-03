
from __future__ import annotations
import argparse, json
from QuASAr.analyzer import analyze
from QuASAr.planner import plan, execute, PlannerConfig
import benchmark_circuits as bench

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, default="ghz_clusters_random",
                   choices=list(bench.CIRCUIT_REGISTRY.keys()))
    p.add_argument("--num-qubits", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--depth", type=int, default=200)
    p.add_argument("--depth-pre", type=int, default=100)
    p.add_argument("--depth-post", type=int, default=100)
    p.add_argument("--qft-bandwidth", type=int, default=3)
    p.add_argument("--neighbor-bridge-layers", type=int, default=0)
    p.add_argument("--rot-prob", type=float, default=0.2)
    p.add_argument("--angle-scale", type=float, default=0.1)
    p.add_argument("--pair-scope", type=str, default="global", choices=["global","block"],
                   help="Two-qubit pairing scope for clifford_plus_rot")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-ram-gb", type=float, default=64.0)
    p.add_argument("--prefer-dd", action="store_true")
    p.add_argument("--out", type=str, default="result.json")
    args = p.parse_args()

    kw = dict(num_qubits=args.num_qubits, seed=args.seed)
    if "stitched" in args.kind:
        kw.update(block_size=args.block_size,
                  depth_pre=args.depth_pre, depth_post=args.depth_post,
                  qft_bandwidth=args.qft_bandwidth, neighbor_bridge_layers=args.neighbor_bridge_layers)
    elif args.kind == "ghz_clusters_random":
        kw.update(block_size=args.block_size, depth=args.depth)
    elif args.kind == "random_clifford":
        kw.update(depth=args.depth)
    elif args.kind == "clifford_plus_rot":
        kw.update(depth=args.depth, rot_prob=args.rot_prob, angle_scale=args.angle_scale,
                  pair_scope=args.pair_scope, block_size=args.block_size)

    circ = bench.build(args.kind, **kw)
    analysis = analyze(circ)
    cfg = PlannerConfig(max_ram_gb=args.max_ram_gb, prefer_dd=args.prefer_dd)
    ssd = plan(analysis.ssd, cfg)
    from QuASAr.simulation_engine import execute_ssd, ExecutionConfig
    exec_payload = execute_ssd(ssd, ExecutionConfig(max_ram_gb=args.max_ram_gb))
    payload = {"analysis": {"global": analysis.metrics_global, "ssd": ssd.to_dict()}, "execution": exec_payload}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
