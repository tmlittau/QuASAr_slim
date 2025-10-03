
from __future__ import annotations
import argparse, json, os, logging, time
from dataclasses import dataclass
from typing import Dict, Any, List

import benchmark_circuits as bench
from QuASAr.analyzer import analyze
from QuASAr.planner import plan
from QuASAr.simulation_engine import execute_ssd, ExecutionConfig
from QuASAr.baselines import run_baselines

@dataclass
class CaseSpec:
    kind: str
    params: Dict[str, Any]

def default_suite(num_qubits: int, block_size: int) -> List[CaseSpec]:
    return [
        CaseSpec("stitched_rand_bandedqft_rand", {"num_qubits": num_qubits, "block_size": block_size,
                                                  "depth_pre": 100, "depth_post": 100, "qft_bandwidth": 3,
                                                  "neighbor_bridge_layers": 0, "seed": 1}),
        CaseSpec("stitched_diag_bandedqft_diag", {"num_qubits": num_qubits, "block_size": block_size,
                                                  "depth_pre": 100, "depth_post": 100, "qft_bandwidth": 3,
                                                  "neighbor_bridge_layers": 0, "seed": 2}),
        # FAST: restrict two-qubit Cliffords to blocks for partitionability
        CaseSpec("clifford_plus_rot", {"num_qubits": num_qubits, "depth": 200, "rot_prob": 0.2,
                                       "angle_scale": 0.1, "seed": 3,
                                       "pair_scope": "block", "block_size": block_size}),
        CaseSpec("ghz_clusters_random", {"num_qubits": num_qubits, "block_size": block_size, "depth": 200, "seed": 4}),
        CaseSpec("random_clifford", {"num_qubits": num_qubits, "depth": 200, "seed": 5}),
    ]

def run_case(case: CaseSpec, *, max_ram_gb: float, sv_ampops_per_sec: float | None, out_dir: str) -> Dict[str, Any]:
    name = case.kind
    params = dict(case.params)
    circ = bench.build(name, **params)

    a = analyze(circ)
    ssd = plan(a.ssd)
    exec_payload = execute_ssd(ssd, ExecutionConfig(max_ram_gb=max_ram_gb))
    quasar_wall = exec_payload.get("meta", {}).get("wall_elapsed_s", None)

    bl = run_baselines(circ, which=["tableau","sv","dd"], per_partition=False,
                       max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec)

    rec = {
        "case": {"kind": name, "params": params},
        "quasar": {"wall_elapsed_s": quasar_wall, "execution": exec_payload, "analysis": {"global": a.metrics_global}},
        "baselines": bl,
    }
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{name}_" + "_".join([f"{k}-{params[k]}" for k in ("num_qubits","block_size","depth","depth_pre","depth_post","qft_bandwidth") if k in params])
    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="suite_out")
    ap.add_argument("--num-qubits", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--block-size", type=int, nargs="+", default=[8])
    ap.add_argument("--max-ram-gb", type=float, default=64.0)
    ap.add_argument("--sv-ampops-per-sec", type=float, default=None)
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--heartbeat-sec", type=float, default=10.0)
    ap.add_argument("--non-disjoint-qubits", type=int, default=None,
                    help="If set, cap num_qubits for circuits that are non-disjoint (bridges/global).")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    specs: List[CaseSpec] = []
    for nq in args.num_qubits:
        for bs in args.block_size:
            specs.extend(default_suite(nq, bs))

    # Optionally reduce qubits for non-disjoint circuits
    if args.non_disjoint_qubits is not None:
        reduced_specs: List[CaseSpec] = []
        for spec in specs:
            params = dict(spec.params)
            kind = spec.kind
            non_disjoint = False
            if kind.startswith("stitched_") and params.get("neighbor_bridge_layers", 0) > 0:
                non_disjoint = True
            if kind == "clifford_plus_rot" and params.get("pair_scope", "global") != "block":
                non_disjoint = True
            if kind == "random_clifford":
                non_disjoint = True
            if non_disjoint and "num_qubits" in params:
                params["num_qubits"] = min(int(params["num_qubits"]), int(args.non_disjoint_qubits))
            reduced_specs.append(CaseSpec(kind, params))
        specs = reduced_specs

    t0 = time.time()
    last = t0
    for i, spec in enumerate(specs, 1):
        logging.info("Running case %d/%d: %s %s", i, len(specs), spec.kind, spec.params)
        _ = run_case(spec, max_ram_gb=args.max_ram_gb, sv_ampops_per_sec=args.sv_ampops_per_sec, out_dir=args.out_dir)
        now = time.time()
        if now - last >= args.heartbeat_sec:
            logging.info("[heartbeat] progress %d/%d, elapsed %ds", i, len(specs), int(now - t0))
            last = now

    # Write summary index
    index = []
    for fn in sorted(os.listdir(args.out_dir)):
        if not fn.endswith(".json"): continue
        with open(os.path.join(args.out_dir, fn), "r") as f:
            data = json.load(f)
        q_wall = float(data.get("quasar", {}).get("wall_elapsed_s", 0.0) or 0.0)
        entry = {
            "file": fn,
            "kind": data.get("case", {}).get("kind"),
            "params": data.get("case", {}).get("params", {}),
            "quasar_wall_s": q_wall,
            "baselines": data.get("baselines", {}).get("entries", []),
        }
        index.append(entry)
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logging.info("Suite done. Wrote %s", os.path.join(args.out_dir, "index.json"))

if __name__ == "__main__":
    main()
