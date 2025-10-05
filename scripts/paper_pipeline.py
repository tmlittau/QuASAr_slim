#!/usr/bin/env python3
"""
End-to-end pipeline for paper figures/tables.

- Auto-calibrate conv-factor (imports scripts.calibrate_conv_factor)
- Run hybrid & disjoint suites and generate bar charts (imports scripts.make_figures_and_tables)
- Generate summary table from results
- Logs to console and to paper_runs/pipeline.log

Run from repo root:
    python -m scripts.paper_pipeline
"""

from __future__ import annotations
import sys
import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# =========================================================================================
# EDIT THESE PARAMETERS
# =========================================================================================

WORKDIR = Path("paper_runs")       # all outputs go here (figures, tables, logs, JSONs)

# --- Calibration settings ---
CALIB = {
    "n_list":         [12, 16, 20],
    "depth_cliff":    200,     # modest; tail_layers stabilizes timing
    "tail_layers":    12,
    "angle_scale":    0.10,
    "samples_per_n":  3,
    "twoq_factor":    4.0,
    "repeat_blocks":  2,       # repeat full sweep to gather more samples
    "policy":         "safe",  # "balanced" (median*safety) | "safe" (P95) | "ultra" (max)
    "safety":         1.5,     # used only for policy == "balanced"
    "pin_threads":    "",     # OMP_NUM_THREADS; set "" to leave environment untouched
    "out_file":       "auto_conv_factor.json",
    "extra_args":     [],      # forwarded to calibrate_conv_factor (e.g., circuit family flags)
}

# --- Hybrid figure settings (Clifford + rotation tail) ---
HYBRID = {
    "n":            [16, 24, 32],
    "block_size":   8,
    "twoq_factor":  4.0,   # keep consistent with calibration
    "max_ram_gb":   60,
    "title":        "QuASAr vs baseline on Clifford+tail circuits",
    "out":          "bar_hybrid.png",
}

# --- Disjoint figure settings (GHZ/W blocks + local tails) ---
DISJOINT = {
    "n":           [16, 24, 32, 40],
    "blocks":      [2, 4, 8],
    "prep":        "mixed",   # "ghz" | "w" | "mixed"
    "tail_kind":   "mixed",   # "clifford" | "diag" | "mixed" | "none"
    "tail_depth":  20,
    "angle_scale": 0.10,
    "sparsity":    0.1,
    "bandwidth":   2,
    "twoq_factor": 4.0,       # keep consistent with calibration
    "max_ram_gb":  60,
    "title":       "Parallel disjoint circuits: QuASAr vs baseline",
    "out":         "bar_disjoint.png",
}

# --- Summary table settings (reads from the suite output dirs that subcommands create) ---
TABLE = {
    "suite_dirs": ["suite_hybrid", "suite_disjoint"],
    "out":        "performance_table.csv",  # use .md to emit a Markdown table instead
}

# =========================================================================================
# LOGGING
# =========================================================================================

def setup_logging(workdir: Path) -> logging.Logger:
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "pipeline.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # File
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Logging to %s", log_file)
    return logger

# =========================================================================================
# HELPERS: call existing scripts programmatically
# =========================================================================================

def _call_argparse_module(mod, argv: List[str], logger: logging.Logger) -> None:
    """
    Call a module that has a main() with argparse, simulating sys.argv.
    Catches SystemExit from argparse with code 0.
    """
    argv_saved = sys.argv
    sys.argv = argv
    try:
        logger.info("Running: %s %s", mod.__name__, " ".join(argv[1:]))
        if hasattr(mod, "main") and callable(mod.main):
            try:
                mod.main()
            except SystemExit as se:
                if se.code not in (0, None):
                    logger.error("%s exited with code %s", mod.__name__, se.code)
                    raise
        else:
            raise RuntimeError(f"{mod.__name__} has no callable main()")
    finally:
        sys.argv = argv_saved

def _flatten_conv_values(payload: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    if not isinstance(payload, dict):
        return vals
    if isinstance(payload.get("conv_factor"), list):
        vals += [float(x) for x in payload["conv_factor"] if isinstance(x, (int, float))]

    by_n = payload.get("by_n") or payload.get("per_n") or {}
    if isinstance(by_n, dict):
        for rec in by_n.values():
            arr = rec.get("conv_factor") or rec.get("values") or []
            if isinstance(arr, list):
                vals += [float(x) for x in arr if isinstance(x, (int, float))]

    samples = payload.get("samples")
    if isinstance(samples, list):
        for s in samples:
            cf = s.get("conv_factor")
            if isinstance(cf, (int, float)):
                vals.append(float(cf))
    return vals

def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)

# =========================================================================================
# STEPS
# =========================================================================================

def step_calibrate(workdir: Path, logger: logging.Logger) -> float:
    """
    Run the existing calibrate_conv_factor as an imported module, using the
    parameters in CALIB. Write a consolidated JSON with stats and return the
    recommended conv-factor according to CALIB['policy'].
    """
    if CALIB.get("pin_threads"):
        os.environ.setdefault("OMP_NUM_THREADS", str(CALIB["pin_threads"]))
        logger.info("Pinned OMP_NUM_THREADS=%s", os.environ["OMP_NUM_THREADS"])

    # Build argv for scripts.calibrate_conv_factor
    out_path = workdir / CALIB["out_file"]
    argv = ["calibrate_conv_factor.py"]
    for n in CALIB["n_list"]:
        argv += ["--n", str(int(n))]
    argv += [
        "--depth-cliff", str(int(CALIB["depth_cliff"])),
        "--tail-layers", str(int(CALIB["tail_layers"])),
        "--angle-scale",  str(float(CALIB["angle_scale"])),
        "--samples-per-n", str(int(CALIB["samples_per_n"])),
        "--twoq-factor",  str(float(CALIB["twoq_factor"])),
        "--out", str(out_path),
    ]
    argv += CALIB.get("extra_args", [])

    # Import and call
    import scripts.calibrate_conv_factor as ccf
    t0 = time.time()
    _call_argparse_module(ccf, argv, logger)
    elapsed = time.time() - t0
    logger.info("Calibration finished in %.2fs, output: %s", elapsed, out_path)

    # Aggregate over repeat_blocks by re-running if requested
    all_vals: List[float] = _flatten_conv_values(json.loads(out_path.read_text()))
    for rep in range(max(0, int(CALIB["repeat_blocks"]) - 1)):
        logger.info("Calibration repeat %d/%d ...", rep + 2, CALIB["repeat_blocks"])
        _call_argparse_module(ccf, argv, logger)
        all_vals += _flatten_conv_values(json.loads(out_path.read_text()))

    if not all_vals:
        raise RuntimeError("Calibration produced zero conv_factor samples.")

    # Stats
    import statistics
    med = statistics.median(all_vals)
    p90 = _percentile(all_vals, 0.90)
    p95 = _percentile(all_vals, 0.95)
    mx  = max(all_vals)

    policy = CALIB["policy"]
    if policy == "balanced":
        recommended = float(med * float(CALIB["safety"]))
        rule = f"median * {CALIB['safety']}"
    elif policy == "safe":
        recommended = float(p95)
        rule = "P95"
    elif policy == "ultra":
        recommended = float(mx)
        rule = "max"
    else:
        raise ValueError(f"Unknown policy: {policy}")

    # Write consolidated auto JSON
    auto_payload = {
        "meta": {
            **{k: v for k, v in CALIB.items() if k not in ("extra_args",)},
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "PYTHON": sys.executable,
            },
            "extra_forwarded_args": CALIB.get("extra_args", []),
        },
        "stats": {"count": len(all_vals), "median": med, "p90": p90, "p95": p95, "max": mx},
        "recommended_conv_factor": recommended,
        "recommendation_rule": rule,
    }
    auto_out = workdir / "auto_conv_factor.json"
    auto_out.write_text(json.dumps(auto_payload, indent=2))
    logger.info("Auto calibration summary written to %s", auto_out)
    logger.info("Recommended conv-factor = %.2f (rule: %s)", recommended, rule)
    return recommended

def step_hybrid_fig(workdir: Path, conv_factor: float, logger: logging.Logger) -> Path:
    """
    Run make_figures_and_tables 'hybrid' subcommand with parameters from HYBRID.
    It will build the suite if missing and save the figure to HYBRID['out'].
    """
    import scripts.make_figures_and_tables as mft

    out_path = workdir / HYBRID["out"]
    argv = [
        "make_figures_and_tables.py", "hybrid",
        "--out", str(out_path),
        "--title", HYBRID["title"],
        "--twoq-factor", str(HYBRID["twoq_factor"]),
        "--max-ram-gb", str(HYBRID["max_ram_gb"]),
        "--conv-factor", str(conv_factor),
    ]
    # n and block-size
    for n in HYBRID["n"]:
        argv += ["--n", str(int(n))]
    argv += ["--block-size", str(int(HYBRID["block_size"]))]

    _call_argparse_module(mft, argv, logger)
    logger.info("Hybrid bar chart written to %s", out_path)
    return out_path

def step_disjoint_fig(workdir: Path, conv_factor: float, logger: logging.Logger) -> Path:
    """
    Run make_figures_and_tables 'disjoint' subcommand with parameters from DISJOINT.
    """
    import scripts.make_figures_and_tables as mft

    out_path = workdir / DISJOINT["out"]
    argv = [
        "make_figures_and_tables.py", "disjoint",
        "--out", str(out_path),
        "--title", DISJOINT["title"],
        "--twoq-factor", str(DISJOINT["twoq_factor"]),
        "--max-ram-gb", str(DISJOINT["max_ram_gb"]),
        "--conv-factor", str(conv_factor),
        "--prep", DISJOINT["prep"],
        "--tail-kind", DISJOINT["tail_kind"],
        "--tail-depth", str(int(DISJOINT["tail_depth"])),
        "--angle-scale", str(float(DISJOINT["angle_scale"])),
        "--sparsity", str(float(DISJOINT["sparsity"])),
        "--bandwidth", str(int(DISJOINT["bandwidth"])),
    ]
    for n in DISJOINT["n"]:
        argv += ["--n", str(int(n))]
    for b in DISJOINT["blocks"]:
        argv += ["--blocks", str(int(b))]

    _call_argparse_module(mft, argv, logger)
    logger.info("Disjoint bar chart written to %s", out_path)
    return out_path

def step_table(workdir: Path, logger: logging.Logger) -> Path:
    """
    Run make_figures_and_tables 'table' subcommand to aggregate suite results into a CSV/MD table.
    """
    import scripts.make_figures_and_tables as mft

    out_path = workdir / TABLE["out"]
    argv = ["make_figures_and_tables.py", "table", "--out", str(out_path)]
    for sd in TABLE["suite_dirs"]:
        argv += ["--suite-dir", sd]

    _call_argparse_module(mft, argv, logger)
    logger.info("Summary table written to %s", out_path)
    return out_path

# =========================================================================================
# MAIN
# =========================================================================================

def main() -> None:
    # Ensure we run from repo root for absolute imports to resolve
    here = Path.cwd()
    if not (here / "quasar").exists() or not (here / "scripts").exists():
        print("[ERROR] Please run this script from the repository root.", file=sys.stderr)
        sys.exit(2)

    logger = setup_logging(WORKDIR)
    logger.info("Starting paper pipeline...")
    t0 = time.time()

    # (Optional) log which backends are available
    try:
        from quasar.backends import stim_available, ddsim_available
        logger.info("stim available: %s | ddsim available: %s",
                    bool(stim_available()), bool(ddsim_available()))
    except Exception as e:
        logger.warning("Backend availability check failed: %r", e)

    # 1) Calibration
    try:
        conv = step_calibrate(WORKDIR, logger)
    except Exception as e:
        logger.exception("Calibration failed: %r", e)
        sys.exit(1)

    # 2) Hybrid figure
    try:
        step_hybrid_fig(WORKDIR, conv, logger)
    except Exception as e:
        logger.exception("Hybrid figure step failed: %r", e)

    # 3) Disjoint figure
    try:
        step_disjoint_fig(WORKDIR, conv, logger)
    except Exception as e:
        logger.exception("Disjoint figure step failed: %r", e)

    # 4) Summary table
    try:
        step_table(WORKDIR, logger)
    except Exception as e:
        logger.exception("Table step failed: %r", e)

    logger.info("Pipeline finished in %.2fs", time.time() - t0)

if __name__ == "__main__":
    main()
