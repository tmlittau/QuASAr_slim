
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

def plot_compare(ssd_file: str, baselines_file: str, outfile: str) -> None:
    # Load SSD (QuASAr) wall-clock time
    with open(ssd_file, "r") as f:
        ssd = json.load(f)
    execp = ssd.get("execution", ssd)
    wall = execp.get("meta", {}).get("wall_elapsed_s", None)
    if wall is None:
        # Fallback: max partition elapsed (upper bound on wall)
        parts = execp.get("results", [])
        wall = max([float(p.get("elapsed_s", 0.0)) for p in parts] + [0.0])

    # Load baselines (measured or estimated)
    with open(baselines_file, "r") as f:
        bl = json.load(f)
    entries = bl.get("entries", [])

    labels = ["QuASAr"]
    times = [float(wall)]

    for e in entries:
        if e.get("mode") != "whole":
            continue  # compare against whole-circuit baselines only
        label = e.get("which")
        res = e.get("result", {})
        if res.get("ok"):
            labels.append(label)
            times.append(float(res.get("elapsed_s", 0.0)))
        else:
            est = res.get("estimate", {})
            t_est = est.get("time_est_sec")
            if t_est is not None:
                labels.append(label + " (est)")
                times.append(float(t_est))
            else:
                labels.append(label + " (fail)")
                times.append(0.0)

    plt.figure()
    plt.bar(labels, times)
    plt.ylabel("Time (s)")
    plt.title("QuASAr vs Baselines (wall clock)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ssd","baselines","compare"], required=True)
    args = ap.parse_args()
    if args.mode == "ssd":
        plot_ssd(args.input, args.out)
    elif args.mode == "baselines":
        plot_baselines(args.input, args.out)
    else:
        # compare requires two inputs; reuse --input for baselines for backward compat
        import sys
        # Expect --ssd and --baselines via env-like extra args? We'll parse from known flags
        # Simpler: overload --input to accept 'SSD_FILE|BASELINES_FILE'
        if '|' in args.input:
            ssd_file, bl_file = args.input.split('|', 1)
            plot_compare(ssd_file.strip(), bl_file.strip(), args.out)
        else:
            print("compare mode expects --input 'SSD_FILE|BASELINES_FILE'", file=sys.stderr)
            sys.exit(2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
