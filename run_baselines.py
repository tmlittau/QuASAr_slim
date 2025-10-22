
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from typing import List

from benchmarks import build as build_circuit
from quasar.baselines import run_baselines


def load_circuit(kind: str, **kwargs):
    return build_circuit(kind, **kwargs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", type=str, default="ghz_clusters_random")
    p.add_argument("--num-qubits", type=int, default=64)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--depth", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--which",
        type=str,
        default="all",
        help="sv,dd,tableau,hybridq or comma-separated",
    )
    p.add_argument("--per-partition", action="store_true", help="run backend on each independent partition separately")
    p.add_argument("--max-ram-gb", type=float, default=None, help="cap SV; if exceeded returns estimate only")
    p.add_argument("--sv-ampops-per-sec", type=float, default=None, help="SV throughput to convert amp_ops to seconds (optional)")
    p.add_argument("--out", type=str, default="baselines.json")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--heartbeat-sec", type=float, default=5.0)
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logging.info("Building %s circuit", args.kind)
    circ = load_circuit(args.kind, num_qubits=args.num_qubits, block_size=args.block_size, depth=args.depth, seed=args.seed)

    which_list: List[str]
    if args.which == "all":
        which_list = ["tableau", "sv", "dd", "hybridq"]
    else:
        which_list = [
            w.strip()
            for w in args.which.split(",")
            if w.strip() in {"sv", "dd", "tableau", "hybridq"}
        ]
        if not which_list:
            raise SystemExit("No valid --which provided (use sv, dd, tableau, hybridq)")

    # Heartbeat thread (coarse; baselines may block in backend)
    stop = threading.Event()
    def hb():
        t0 = time.time()
        while not stop.is_set():
            time.sleep(args.heartbeat_sec)
            dt = int(time.time() - t0)
            logging.info("[heartbeat] running baselines for %ds", dt)
    threading.Thread(target=hb, daemon=True).start()

    res = run_baselines(circ, which=which_list, per_partition=args.per_partition,
                        max_ram_gb=args.max_ram_gb, sv_ampops_per_sec=args.sv_ampops_per_sec)
    stop.set()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    logging.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
