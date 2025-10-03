
from __future__ import annotations
import argparse, json
import matplotlib.pyplot as plt

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def plot_baselines(infile: str, outfile: str) -> None:
    with open(infile, "r") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    labels, times = [], []
    for e in entries:
        label = e.get("which")
        if e.get("mode") == "whole":
            res = e.get("result", {})
            if res.get("ok"):
                labels.append(label)
                times.append(_safe_float(res.get("elapsed_s", 0.0)))
            else:
                est = res.get("estimate", {})
                t_est = est.get("time_est_sec")
                if t_est is not None:
                    labels.append(label + " (est)")
                    times.append(_safe_float(t_est, 0.0))
                else:
                    # no time estimate -> plot 0 and annotate via label
                    labels.append(label + " (fail)")
                    times.append(0.0)
        else:
            # per-partition: still allowed but de-emphasize by labeling
            labels.append(label + " (per-part)")
            times.append(_safe_float(e.get("elapsed_s", 0.0)))
    plt.figure()
    plt.bar(labels, times)
    plt.xlabel("Baseline")
    plt.ylabel("Time (s) [estimated when marked '(est)']")
    plt.title("Baseline runtimes")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_ssd(infile: str, outfile: str) -> None:
    with open(infile, "r") as f:
        data = json.load(f)
    execp = data.get("execution", data)
    parts = execp.get("results", [])
    labels = [f"p{p.get('partition')}-{p.get('backend')}" for p in parts]
    times = [_safe_float(p.get("elapsed_s", 0.0)) for p in parts]
    plt.figure()
    plt.bar(labels, times)
    plt.xlabel("Partition-Backend")
    plt.ylabel("Elapsed time (s)")
    plt.title("SSD partition runtimes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ssd","baselines"], required=True)
    args = ap.parse_args()
    if args.mode == "ssd":
        plot_ssd(args.input, args.out)
    else:
        plot_baselines(args.input, args.out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
