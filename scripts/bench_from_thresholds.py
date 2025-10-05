
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

from benchmarks.hybrid import clifford_prefix_rot_tail
from quasar.analyzer import analyze
from quasar.baselines import run_baselines
from quasar.planner import PlannerConfig, plan
from quasar.simulation_engine import ExecutionConfig, execute_ssd

def load_thresholds(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "records" not in data:
        raise SystemExit("Invalid thresholds JSON: missing 'records'")
    return data

def pick_meta_or_default(meta: Dict[str, Any], key: str, default):
    params = meta.get("params", {}) if isinstance(meta, dict) else {}
    val = params.get(key)
    return val if val is not None else default

def run_from_thresholds(thr_json: Dict[str, Any], *, cutoff: Optional[float], out_dir: str,
                        angle_scale: Optional[float], conv_factor: Optional[float], twoq_factor: Optional[float],
                        max_ram_gb: float, sv_ampops_per_sec: Optional[float], log: logging.Logger) -> None:
    meta = thr_json.get("meta", {})
    if cutoff is None:
        cutoffs = sorted({float(r["cutoff"]) for r in thr_json["records"] if r.get("cutoff") is not None})
        if len(cutoffs) != 1:
            raise SystemExit(f"--cutoff not provided and thresholds include {len(cutoffs)} cutoffs; please specify one of {cutoffs}")
        cutoff = cutoffs[0]

    angle_scale = float(angle_scale if angle_scale is not None else pick_meta_or_default(meta, "angle_scale", 0.1))
    conv_factor = float(conv_factor if conv_factor is not None else pick_meta_or_default(meta, "conv_factor", 64.0))
    twoq_factor = float(twoq_factor if twoq_factor is not None else pick_meta_or_default(meta, "twoq_factor", 4.0))

    recs = [r for r in thr_json["records"] if float(r.get("cutoff", -1)) == float(cutoff) and r.get("first_depth")]
    if not recs:
        raise SystemExit("No records with first_depth available for the selected cutoff")

    os.makedirs(out_dir, exist_ok=True)

    for r in sorted(recs, key=lambda x: int(x["n"])):
        n = int(r["n"])
        depth = int(r["first_depth"])
        log.info("Running threshold case: n=%d depth=%d cutoff=%.2f", n, depth, cutoff)

        circ = clifford_prefix_rot_tail(
            num_qubits=n,
            depth=depth,
            cutoff=float(cutoff),
            angle_scale=angle_scale,
            seed=42,
        )

        a = analyze(circ)
        cfg = PlannerConfig(max_ram_gb=max_ram_gb, conv_amp_ops_factor=conv_factor, sv_twoq_factor=twoq_factor)
        ssd = plan(a.ssd, cfg)
        exec_payload = execute_ssd(ssd, ExecutionConfig(max_ram_gb=max_ram_gb))

        bl = run_baselines(circ, which=["tableau","sv","dd"], per_partition=False,
                           max_ram_gb=max_ram_gb, sv_ampops_per_sec=sv_ampops_per_sec)

        rec_out = {
            "case": {"kind": "clifford_prefix_rot_tail", "params": {"num_qubits": n, "depth": depth, "cutoff": cutoff, "angle_scale": angle_scale}},
            "planner": {"conv_factor": conv_factor, "twoq_factor": twoq_factor},
            "quasar": {"wall_elapsed_s": exec_payload.get("meta", {}).get("wall_elapsed_s", None),
                       "execution": exec_payload,
                       "analysis": {"global": a.metrics_global, "ssd": ssd.to_dict()}},
            "baselines": bl,
        }
        stem = f"clifford_prefix_rot_tail_n-{n}_d-{depth}_cut-{cutoff}"
        with open(os.path.join(out_dir, stem + ".json"), "w") as f:
            json.dump(rec_out, f, indent=2)

    try:
        from plots.bar_clifford_tail import make_plot

        plot_path = os.path.join(out_dir, "bars_from_thresholds.png")
        make_plot(
            out_dir,
            out=plot_path,
            title=f"Threshold bars (cutoff={cutoff}, conv={conv_factor}, twoq={twoq_factor})",
        )
        log.info("Wrote bar chart: %s", plot_path)
    except Exception as exc:
        log.warning(
            "Plot generation failed: %s. You can run: python plots/bar_clifford_tail.py --suite-dir %s --out bars_from_thresholds.png",
            exc,
            out_dir,
        )

def main():
    ap = argparse.ArgumentParser(description="Run benchmark cases derived from saved thresholds and plot bars.")
    ap.add_argument("--thresholds", type=str, required=True, help="Path to thresholds JSON saved by playground_cutoff.py --save-json")
    ap.add_argument("--cutoff", type=float, default=None, help="Cutoff to use from the thresholds JSON (needed if multiple cutoffs present).")
    ap.add_argument("--out-dir", type=str, default="suite_from_thresholds")
    ap.add_argument("--angle-scale", type=float, default=None, help="Override angle_scale (default: use value from thresholds meta or 0.1)")
    ap.add_argument("--conv-factor", type=float, default=None, help="Override conv_factor (default: use thresholds meta or 64.0)")
    ap.add_argument("--twoq-factor", type=float, default=None, help="Override twoq_factor (default: use thresholds meta or 4.0)")
    ap.add_argument("--max-ram-gb", type=float, default=64.0)
    ap.add_argument("--sv-ampops-per-sec", type=float, default=None)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("bench_from_thresholds")

    thr = load_thresholds(args.thresholds)
    run_from_thresholds(thr, cutoff=args.cutoff, out_dir=args.out_dir,
                        angle_scale=args.angle_scale, conv_factor=args.conv_factor, twoq_factor=args.twoq_factor,
                        max_ram_gb=args.max_ram_gb, sv_ampops_per_sec=args.sv_ampops_per_sec, log=log)

if __name__ == "__main__":
    main()
