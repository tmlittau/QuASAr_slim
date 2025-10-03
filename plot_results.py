
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
        if e.get("mode") == "whole":
            res = e.get("result", {})
            labels.append(e.get("which"))
            times.append(_safe_float(res.get("elapsed_s", 0.0)))
        else:
            labels.append(e.get("which") + " (per-part)")
            times.append(_safe_float(e.get("elapsed_s", 0.0)))
    plt.figure()
    plt.bar(labels, times)
    plt.xlabel("Baseline")
    plt.ylabel("Elapsed time (s)")
    plt.title("Baseline runtimes")
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
