
from __future__ import annotations
import argparse, os, json
from QuASAr.analyzer import analyze
from QuASAr.planner import plan, PlannerConfig
from QuASAr.simulation_engine import execute_ssd, ExecutionConfig
from QuASAr.baselines import run_baselines

try:
    from benchmark_circuits import dd_friendly_prefix_diag_tail
except Exception:
    from benchmark_circuits_dd_friendly import dd_friendly_prefix_diag_tail

def main():
    ap = argparse.ArgumentParser(description="Run DD-friendly prefix+diag-tail circuits and write JSON results + bar plot")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--depth", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--cutoff", type=float, default=0.8)
    ap.add_argument("--angle-scale", type=float, default=0.1)
    ap.add_argument("--tail-sparsity", type=float, default=0.05)
    ap.add_argument("--tail-bandwidth", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default="suite_dd_friendly")
    ap.add_argument("--conv-factor", type=float, default=64.0)
    ap.add_argument("--twoq-factor", type=float, default=4.0)
    ap.add_argument("--max-ram-gb", type=float, default=64.0)
    ap.add_argument("--sv-ampops-per-sec", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for n in args.n:
        for d in args.depth:
            circ = dd_friendly_prefix_diag_tail(num_qubits=n, depth=d, cutoff=args.cutoff,
                                                angle_scale=args.angle_scale, tail_sparsity=args.tail_sparsity,
                                                tail_bandwidth=args.tail_bandwidth, seed=42)
            a = analyze(circ)
            cfg = PlannerConfig(max_ram_gb=args.max_ram_gb, conv_amp_ops_factor=args.conv_factor, sv_twoq_factor=args.twoq_factor)
            ssd = plan(a.ssd, cfg)
            exec_payload = execute_ssd(ssd, ExecutionConfig(max_ram_gb=args.max_ram_gb))
            bl = run_baselines(circ, which=["tableau","sv","dd"], per_partition=False,
                               max_ram_gb=args.max_ram_gb, sv_ampops_per_sec=args.sv_ampops_per_sec)

            rec = {
                "case": {"kind": "dd_friendly_prefix_diag_tail",
                         "params": {"num_qubits": n, "depth": d, "cutoff": args.cutoff,
                                    "angle_scale": args.angle_scale, "tail_sparsity": args.tail_sparsity,
                                    "tail_bandwidth": args.tail_bandwidth}},
                "planner": {"conv_factor": args.conv_factor, "twoq_factor": args.twoq_factor},
                "quasar": {"execution": exec_payload, "analysis": {"global": a.metrics_global, "ssd": ssd.to_dict()}},
                "baselines": bl,
            }
            fn = os.path.join(args.out_dir, f"dd_friendly_n-{n}_d-{d}.json")
            with open(fn, "w") as f:
                json.dump(rec, f, indent=2)

    try:
        from plot_hybrid_bars import make_plot
        make_plot(args.out_dir, out=os.path.join(args.out_dir, "bars_dd_friendly.png"),
                  title="DD-friendly prefix + diagonal tail: QuASAr vs whole-circuit baseline")
        print(f"Wrote {os.path.join(args.out_dir, 'bars_dd_friendly.png')}")
    except Exception as e:
        print("Plotting failed:", e)
        print("You can plot manually with: python plot_hybrid_bars.py --suite-dir", args.out_dir, "--out bars_dd_friendly.png")

if __name__ == "__main__":
    main()
