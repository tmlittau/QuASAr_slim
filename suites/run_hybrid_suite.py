from __future__ import annotations
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from benchmarks import build as build_circuit
from quasar.analyzer import analyze
from quasar.baselines import run_baselines
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_plan

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
        CaseSpec("clifford_plus_rot", {"num_qubits": num_qubits, "depth": 200, "rot_prob": 0.2,
                                       "angle_scale": 0.1, "seed": 3,
                                       "pair_scope": "block", "block_size": block_size}),
        CaseSpec("ghz_clusters_random", {"num_qubits": num_qubits, "block_size": block_size, "depth": 200, "seed": 4}),
        CaseSpec("random_clifford", {"num_qubits": num_qubits, "depth": 200, "seed": 5}),
    ]

def run_case(case: CaseSpec, *, max_ram_gb: float, sv_ampops_per_sec: float | None, out_dir: str,
             conv_factor: float, twoq_factor: float) -> Dict[str, Any]:
    name = case.kind
    params = dict(case.params)
    circ = build_circuit(name, **params)

    a = analyze(circ)
    cfg = PlannerConfig(max_ram_gb=max_ram_gb, conv_amp_ops_factor=conv_factor, sv_twoq_factor=twoq_factor)
    planned = plan(a.plan, cfg)
    exec_payload = execute_plan(planned, ExecutionConfig(max_ram_gb=max_ram_gb))
    quasar_wall = exec_payload.get("meta", {}).get("wall_elapsed_s", None)

    bl = run_baselines(circ, which=["tableau","sv","dd"], per_partition=False,
                       max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec)

    rec = {
        "case": {"kind": name, "params": params},
        "planner": {"conv_factor": conv_factor, "twoq_factor": twoq_factor},
        "quasar": {"wall_elapsed_s": quasar_wall,
                   "execution": exec_payload,
                   "analysis": {"global": a.metrics_global, "plan": planned.to_dict()}},
        "baselines": bl,
    }
    os.makedirs(out_dir, exist_ok=True)

    # Safer stem builder (no weird quotes)
    keys = ("num_qubits","block_size","depth","depth_pre","depth_post","qft_bandwidth")
    stem = f"{name}_" + "_".join([f"{k}-{params[k]}" for k in keys if k in params])

    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec

def _extract_quasar_wall(d: Any) -> float:
    """Be tolerant of schema (old/new) and types."""
    try:
        if not isinstance(d, dict):
            return 0.0
        q = d.get("quasar")
        if isinstance(q, dict):
            w = q.get("wall_elapsed_s")
            if w is None:
                w = ((q.get("execution") or {}).get("meta") or {}).get("wall_elapsed_s")
            return float(w) if w is not None else 0.0
        ex = d.get("execution")
        if isinstance(ex, dict):
            w = ((ex.get("meta") or {}).get("wall_elapsed_s"))
            return float(w) if w is not None else 0.0
    except Exception:
        return 0.0
    return 0.0

def _extract_baseline_entries(d: Any) -> List[Dict[str, Any]]:
    b = None if not isinstance(d, dict) else d.get("baselines")
    if isinstance(b, dict):
        return b.get("entries", []) or []
    if isinstance(b, list):
        return b
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="suite_out")
    ap.add_argument("--num-qubits", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--block-size", type=int, nargs="+", default=[8])
    ap.add_argument("--max-ram-gb", type=float, default=64.0)
    ap.add_argument("--sv-ampops-per-sec", type=float, default=None)
    ap.add_argument("--conv-factor", type=float, default=64.0)
    ap.add_argument("--twoq-factor", type=float, default=4.0)
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--heartbeat-sec", type=float, default=10.0)
    ap.add_argument("--non-disjoint-qubits", type=int, default=None,
                    help="If set, cap num_qubits for non-disjoint circuits.")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    specs: List[CaseSpec] = []
    for nq in args.num_qubits:
        for bs in args.block_size:
            specs.extend(default_suite(nq, bs))

    # Optionally shrink non-disjoint problems
    if args.non_disjoint_qubits is not None:
        reduced: List[CaseSpec] = []
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
            reduced.append(CaseSpec(kind, params))
        specs = reduced

    t0 = time.time()
    last = t0
    for i, spec in enumerate(specs, 1):
        logging.info("Running case %d/%d: %s %s", i, len(specs), spec.kind, spec.params)
        _ = run_case(spec, max_ram_gb=args.max_ram_gb, sv_ampops_per_sec=args.sv_ampops_per_sec, out_dir=args.out_dir,
                     conv_factor=args.conv_factor, twoq_factor=args.twoq_factor)
        now = time.time()
        if now - last >= args.heartbeat_sec:
            logging.info("[heartbeat] progress %d/%d, elapsed %ds", i, len(specs), int(now - t0))
            last = now

    # Build index: skip index.json; be schema-tolerant
    index = []
    for fn in sorted(os.listdir(args.out_dir)):
        if not fn.endswith(".json") or fn == "index.json":
            continue
        with open(os.path.join(args.out_dir, fn), "r") as f:
            data = json.load(f)

        q_wall = _extract_quasar_wall(data)
        bl_entries = _extract_baseline_entries(data)

        entry = {
            "file": fn,
            "kind": (data.get("case") or {}).get("kind") if isinstance(data, dict) else None,
            "params": (data.get("case") or {}).get("params", {}) if isinstance(data, dict) else {},
            "quasar_wall_s": q_wall,
            "baselines": bl_entries,
            "planner": data.get("planner", {}) if isinstance(data, dict) else {},
        }
        index.append(entry)

    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logging.info("Suite done. Wrote %s", os.path.join(args.out_dir, "index.json"))

if __name__ == "__main__":
    main()
