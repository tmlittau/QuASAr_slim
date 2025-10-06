
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "tableau": "#A5D6A7",  # pastel green
    "sv": "#90CAF9",       # pastel blue
    "dd": "#FFCCBC",       # pastel orange
    "conversion": "#CFD8DC"  # pastel grey
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

def _extract_hybrid_segments(results: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
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

def _flatten_baseline_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Bring nested baseline result payloads into a flat dictionary."""
    if not isinstance(entry, dict):
        return {}
    flat = dict(entry)
    res = flat.pop("result", None)
    if isinstance(res, dict):
        # Only copy keys we don't already have (prefer explicit fields such as mode/which).
        for key, val in res.items():
            flat.setdefault(key, val)
    if "method" not in flat and "which" in flat:
        flat["method"] = flat.get("which")
    if "scope" not in flat and "mode" in flat:
        flat["scope"] = flat.get("mode")
    if "wall_s_measured" not in flat and "elapsed_s" in flat:
        try:
            flat["wall_s_measured"] = float(flat.get("elapsed_s"))
        except Exception:
            pass
    return flat


def _pick_elapsed(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("wall_s_measured", "wall_s_estimated", "elapsed_s", "time_est_sec"):
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except Exception:
                continue
    return None


def _best_whole_baseline(baselines: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    entries = baselines.get("entries", []) if isinstance(baselines, dict) else []
    if not isinstance(entries, list):
        entries = []
    entries = [_flatten_baseline_entry(e) for e in entries]
    if not isinstance(entries, list):
        entries = []
    def is_whole(e: Dict[str, Any]) -> bool:
        if e.get("scope") in ("whole", "global", "circuit"):
            return True
        if (e.get("mode") or e.get("scope")) == "whole":
            return True
        if e.get("ok") is False:
            return False
        if e.get("per_partition") is False:
            return True
        if "partition_id" not in e and "chain_id" not in e:
            return True
        return False
    best = None
    for e in entries:
        if not is_whole(e):
            continue
        t = _pick_elapsed(e)
        if t is None:
            continue
        m = (e.get("method") or e.get("which") or e.get("backend") or e.get("name") or "sv").lower()
        if best is None or float(t) < best[1]:
            best = (m, float(t))
    if best is not None:
        return best
    for e in entries:
        if e.get("ok") is False:
            continue
        t = _pick_elapsed(e)
        if t is None:
            continue
        m = (e.get("method") or e.get("which") or e.get("backend") or e.get("name") or "sv").lower()
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
    tail_elapsed = _pick_elapsed(tail_node) or 0.0
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

        # Merge planner metadata into execution results so we can recover reasons.
        part_info: Dict[int, Dict[str, Any]] = {}
        ssd_info = (analysis or {}).get("ssd", {}) or {}
        for part in ssd_info.get("partitions", []) or []:
            try:
                pid = int(part.get("id"))
            except Exception:
                continue
            info: Dict[str, Any] = {}
            meta = part.get("meta") or {}
            if isinstance(meta, dict):
                info.update(meta)
            info.setdefault("backend", part.get("backend"))
            info.setdefault("num_qubits", part.get("num_qubits"))
            part_info[pid] = info

        exec_results: List[Dict[str, Any]] = []
        for res in exec_payload.get("results", []) or []:
            r = dict(res)
            pid = r.get("partition")
            if pid is not None:
                try:
                    extra = part_info.get(int(pid))
                except Exception:
                    extra = None
                if extra:
                    for key, val in extra.items():
                        r.setdefault(key, val)
            exec_results.append(r)

        segs = _extract_hybrid_segments(exec_results)
        if segs is None:
            res = exec_results
            if res:
                r = min(res, key=lambda x: x.get("seq_index", 0))
                b = (r.get("backend") or "sv").lower()
                t = _pick_elapsed(r) or 0.0
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
            pre_t = _pick_elapsed(pre) or 0.0
            tail_t = _pick_elapsed(tail) or 0.0
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
    if N == 0:
        raise SystemExit("No cases available for plotting")

    width = 0.6

    fig, axes = plt.subplots(1, N, figsize=(max(6, N * 4.2), 5), sharey=True)
    if N == 1:
        axes = [axes]

    tableau_used = any(t > 0 for t in t_tab)
    conversion_used = any(t > 0 for t in t_conv)
    tail_methods_used = {m for m, t in zip(tail_methods, t_tail) if t > 0}
    base_methods_used = {m for m, t in zip(base_methods, base_times) if np.isfinite(t) and t > 0}

    for idx, ax in enumerate(axes):
        tab = t_tab[idx]
        conv = t_conv[idx]
        tail = t_tail[idx]
        base_time = base_times[idx]
        tail_method = tail_methods[idx]
        base_method = base_methods[idx]

        bottom = 0.0
        if tab > 0:
            ax.bar(0, tab, width, color=COLORS["tableau"], edgecolor="black")
            bottom += tab
        if conv > 0:
            ax.bar(0, conv, width, bottom=bottom, color=COLORS["conversion"], edgecolor="black", hatch="//")
            bottom += conv
        if tail > 0:
            ax.bar(0, tail, width, bottom=bottom, color=COLORS.get(tail_method, COLORS["sv"]), edgecolor="black")

        total_quasar = tab + conv + tail
        if total_quasar > 0 and np.isfinite(base_time) and base_time > 0:
            speedup = base_time / total_quasar
            y_offset = 0.05 * max(total_quasar, base_time)
            ax.text(
                0,
                total_quasar + y_offset,
                f"{speedup:.1f}×",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        if np.isfinite(base_time) and base_time > 0:
            ax.bar(1, base_time, width, color=COLORS.get(base_method, COLORS["sv"]), edgecolor="black", alpha=0.9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["QuASAr", "Baseline"])
        ax.set_xlim(-0.75, 1.75)
        ax.set_title(labels[idx])
        if idx == 0:
            ax.set_ylabel("Time (s)")

    case_info = cases[0].get("case", {}) if cases else {}
    params = case_info.get("params", {}) if isinstance(case_info, dict) else {}
    kind = case_info.get("kind") if isinstance(case_info, dict) else None
    default_title = None
    if kind:
        try:
            n0 = int(params.get("num_qubits", labels[0]))
            d0 = int(params.get("depth", labels[0]))
            default_title = f"{kind} (n={n0}, d={d0})"
        except Exception:
            default_title = kind
    fig.suptitle(title or default_title or "QuASAr vs baseline")

    import matplotlib.patches as mpatches

    legend_handles = []
    if tableau_used:
        legend_handles.append(mpatches.Patch(color=COLORS["tableau"], edgecolor="black", label="QuASAr: tableau"))
    if conversion_used:
        legend_handles.append(mpatches.Patch(color=COLORS["conversion"], edgecolor="black", hatch="//", label="QuASAr: conversion"))
    for method in sorted(tail_methods_used):
        method_label = str(method).upper()
        label = f"QuASAr tail: {method_label}"
        legend_handles.append(mpatches.Patch(color=COLORS.get(method, COLORS["sv"]), edgecolor="black", label=label))
    for method in sorted(base_methods_used):
        method_label = str(method).upper()
        label = f"Baseline: {method_label}"
        legend_handles.append(mpatches.Patch(color=COLORS.get(method, COLORS["sv"]), edgecolor="black", alpha=0.9, label=label))

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out:
        fig.savefig(out, dpi=200)
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
