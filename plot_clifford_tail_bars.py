
from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "tableau": "#4CAF50",
    "sv": "#1E88E5",
    "dd": "#F4511E",
    "conversion": "#9E9E9E"
}

KINDS = {"clifford_plus_rot", "clifford_prefix_rot_tail"}

def _load_cases(suite_dir: str) -> List[Dict[str, Any]]:
    cases = []
    for fn in sorted(os.listdir(suite_dir)):
        if not fn.endswith(".json") or fn == "index.json":
            continue
        with open(os.path.join(suite_dir, fn), "r") as f:
            data = json.load(f)
        case = (data.get("case") or {})
        if case.get("kind") in KINDS:
            cases.append(data)
    def _key(d):
        params = (d.get("case") or {}).get("params", {})
        n = int(params.get("num_qubits", 0))
        depth = int(params.get("depth", 0))
        return (n, depth)
    cases.sort(key=_key)
    return cases

def _extract_hybrid_segments(exec_payload: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    results = exec_payload.get("results", [])
    chains = {}
    for r in results:
        cid = r.get("chain_id")
        if cid is None:
            continue
        chains.setdefault(cid, []).append(r)
    for cid, nodes in chains.items():
        if len(nodes) == 2:
            a, b = sorted(nodes, key=lambda d: int(d.get("seq_index", 0)))
            return a, b
    return None

def _best_whole_baseline(baselines: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    entries = baselines.get("entries", []) if isinstance(baselines, dict) else []
    if not isinstance(entries, list):
        entries = []
    def is_whole(e: Dict[str, Any]) -> bool:
        if e.get("ok") is False:
            return False
        if e.get("scope") in ("whole", "global", "circuit"):
            return True
        if e.get("per_partition") is False:
            return True
        if "partition_id" not in e and "chain_id" not in e:
            return True
        return False
    best = None
    for e in entries:
        if not is_whole(e):
            continue
        t = e.get("wall_s_measured")
        if t is None:
            t = e.get("wall_s_estimated")
        if t is None:
            continue
        m = (e.get("method") or e.get("name") or "sv").lower()
        if best is None or float(t) < best[1]:
            best = (m, float(t))
    if best is not None:
        return best
    for e in entries:
        if e.get("ok") is False:
            continue
        t = e.get("wall_s_measured")
        if t is None:
            t = e.get("wall_s_estimated")
        if t is None:
            continue
        m = (e.get("method") or e.get("name") or "sv").lower()
        if best is None or float(t) < best[1]:
            best = (m, float(t))
    return best

def _parse_reason_norms(reason: str) -> Dict[str, float]:
    out = {}
    if not isinstance(reason, str):
        return out
    try:
        parts = reason.split(";")[1].strip()
        fields = parts.split(",")
        for field in fields:
            if "=" not in field:
                continue
            k, v = field.strip().split("=", 1)
            v = v.strip().split()[0].strip("<")
            try:
                out[k.strip()] = float(v)
            except:
                pass
    except Exception:
        pass
    return out

def _estimate_conversion_time(prefix_node: Dict[str, Any], tail_node: Dict[str, Any], n_qubits: int) -> float:
    norms = _parse_reason_norms(prefix_node.get("planner_reason", ""))
    sv_tail_norm = norms.get("tail")
    conv_norm = norms.get("conv")
    tail_elapsed = float(tail_node.get("elapsed_s", 0.0) or 0.0)
    if sv_tail_norm is None or conv_norm is None or tail_elapsed <= 0.0:
        return 0.0
    amps = float(1 << int(n_qubits))
    sec_per_amp = tail_elapsed / (sv_tail_norm * amps)
    return conv_norm * amps * sec_per_amp

def make_plot(suite_dir: str, out: Optional[str] = None, title: Optional[str] = None):
    cases = _load_cases(suite_dir)
    if not cases:
        raise SystemExit("No matching cases found in suite_dir")

    labels = []
    t_tab, t_conv, t_tail = [], [], []
    tail_methods, base_times, base_methods = [], [], []
    totals = []

    for data in cases:
        params = (data.get("case") or {}).get("params", {})
        n_qubits = int(params.get("num_qubits", 0))
        depth = int(params.get("depth", 0))
        labels.append(f"n={n_qubits}, d={depth}")

        exec_payload = (data.get("quasar", {}) or {}).get("execution") or data.get("execution") or {}
        analysis = (data.get("quasar", {}) or {}).get("analysis") or data.get("analysis") or {}
        metrics = (analysis or {}).get("global", {}) or {}
        nq_from_metrics = int(metrics.get("num_qubits", n_qubits))
        n_qubits = nq_from_metrics

        segs = _extract_hybrid_segments(exec_payload)
        if segs is None:
            res = exec_payload.get("results", [])
            if res:
                r = min(res, key=lambda x: x.get("seq_index", 0))
                b = (r.get("backend") or "sv").lower()
                t = float(r.get("elapsed_s", 0.0) or 0.0)
                if b == "tableau":
                    t_tab.append(t); t_conv.append(0.0); t_tail.append(0.0); tail_methods.append("sv")
                    totals.append(t)
                else:
                    t_tab.append(0.0); t_conv.append(0.0); t_tail.append(t); tail_methods.append(b if b in ("sv","dd") else "sv")
                    totals.append(t)
            else:
                t_tab.append(0.0); t_conv.append(0.0); t_tail.append(0.0); tail_methods.append("sv")
                totals.append(0.0)
        else:
            pre, tail = segs
            pre_t = float(pre.get("elapsed_s", 0.0) or 0.0)
            tail_t = float(tail.get("elapsed_s", 0.0) or 0.0)
            tail_b = (tail.get("backend") or "sv").lower()
            conv_t = _estimate_conversion_time(pre, tail, n_qubits)
            tab_t = max(pre_t - conv_t, 0.0)
            t_tab.append(tab_t); t_conv.append(conv_t); t_tail.append(tail_t); tail_methods.append(tail_b if tail_b in ("sv","dd") else "sv")
            totals.append(tab_t + conv_t + tail_t)

        best = _best_whole_baseline(data.get("baselines", {}))
        if best is None:
            base_methods.append("sv"); base_times.append(float("nan"))
        else:
            base_methods.append(best[0]); base_times.append(best[1])

    N = len(labels)
    x = np.arange(N)
    width = 0.42

    plt.figure(figsize=(max(8, N*1.3), 5))

    # QuASAr stacked bar
    plt.bar(x - width/2, t_tab, width, label="QuASAr: tableau", color=COLORS["tableau"])
    plt.bar(x - width/2, t_conv, width, bottom=t_tab, label="QuASAr: conversion", color=COLORS["conversion"], hatch='//', edgecolor='black')
    bottoms = np.array(t_tab) + np.array(t_conv)
    tail_colors = [COLORS.get(m, COLORS["sv"]) for m in tail_methods]
    plt.bar(x - width/2, t_tail, width, bottom=bottoms, label="QuASAr: tail", color=tail_colors)

    # Baseline (whole-circuit) bar next to it
    base_colors = [COLORS.get(m, COLORS["sv"]) for m in base_methods]
    plt.bar(x + width/2, base_times, width, label="Baseline (whole circuit)", color=base_colors, alpha=0.9)

    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel("Time (s)")
    plt.title(title or "QuASAr (stacked) vs whole-circuit baseline")
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=COLORS["tableau"], label="QuASAr: tableau"),
        mpatches.Patch(color=COLORS["conversion"], label="QuASAr: conversion", hatch='//', edgecolor='black'),
        mpatches.Patch(color=COLORS["sv"], label="Tail/Baseline: SV"),
        mpatches.Patch(color=COLORS["dd"], label="Tail/Baseline: DD"),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=200)
        print(f"Wrote {out}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=str, default="suite_out")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()
    make_plot(args.suite_dir, out=args.out, title=args.title)

if __name__ == "__main__":
    main()
