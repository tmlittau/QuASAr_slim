
from __future__ import annotations
import argparse, json, logging
from QuASAr.analyzer import analyze
from QuASAr.planner import plan
from QuASAr.simulation_engine import execute_ssd, ExecutionConfig
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
    p.add_argument("--max-ram-gb", type=float, default=64.0)
    p.add_argument("--max-workers", type=int, default=0)
    p.add_argument("--heartbeat-sec", type=float, default=5.0)
    p.add_argument("--stuck-warn-sec", type=float, default=60.0)
    p.add_argument("--out", type=str, default="result.json")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logging.info("Building circuit: %s", args.kind)
    circ = load_circuit(args.kind, num_qubits=args.num_qubits, block_size=args.block_size, depth=args.depth, seed=args.seed)

    logging.info("Analyzing circuit")
    analysis = analyze(circ)
    logging.info("Found %d partitions", len(analysis.ssd))

    logging.info("Planning")
    ssd = plan(analysis.ssd)

    logging.info("Executing SSD with multithreading")
    cfg = ExecutionConfig(max_ram_gb=args.max_ram_gb, max_workers=args.max_workers,
                          heartbeat_sec=args.heartbeat_sec, stuck_warn_sec=args.stuck_warn_sec)
    exec_payload = execute_ssd(ssd, cfg)

    payload = {"analysis": {"global": analysis.metrics_global, "ssd": ssd.to_dict()}, "execution": exec_payload}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
