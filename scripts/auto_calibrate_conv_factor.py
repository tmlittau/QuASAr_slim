#!/usr/bin/env python3
# scripts/auto_calibrate_conv_factor.py
from __future__ import annotations
import os
import sys
import json
import time
import tempfile
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- TUNE HERE: defaults for your environment ----------
N_LIST: List[int]      = [12, 14, 16]   # qubit sizes to include
DEPTH_CLIFF: int       = 200            # modest prefix; tail-layers drives timing stability
TAIL_LAYERS: int       = 12             # increase if tail timing is noisy
ANGLE_SCALE: float     = 0.10
SAMPLES_PER_N: int     = 3
TWOQ_FACTOR: float     = 4.0
REPEAT_BLOCKS: int     = 2              # repeat full sweep to gather more samples
POLICY: str            = "safe"         # "balanced" (median*safety), "safe" (P95), "ultra" (max)
SAFETY: float          = 1.5            # only used when POLICY == "balanced"
OUT_PATH: str          = "auto_conv_factor.json"
PIN_THREADS: str       = ""            # OMP_NUM_THREADS; set "" to leave as-is
EXTRA_ARGS: List[str]  = []             # extra args forwarded to calibrator (e.g., circuit family)
# --------------------------------------------------------------

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

def _flatten_conv_values(payload: Dict[str, Any]) -> List[float]:
    """
    Be tolerant to schema variants:
    - payload["conv_factor"] -> list
    - payload["by_n"][n]["conv_factor"] -> list
    - payload["samples"][i]["conv_factor"] -> scalar
    """
    vals: List[float] = []
    if not isinstance(payload, dict):
        return vals

    # direct list
    if isinstance(payload.get("conv_factor"), list):
        vals += [float(x) for x in payload["conv_factor"] if isinstance(x, (int, float))]

    # nested by n
    by_n = payload.get("n") or payload.get("per_n") or {}
    if isinstance(by_n, dict):
        for rec in by_n.values():
            arr = rec.get("conv_factor") or rec.get("values") or []
            if isinstance(arr, list):
                vals += [float(x) for x in arr if isinstance(x, (int, float))]

    # top-level samples
    samples = payload.get("samples")
    if isinstance(samples, list):
        for s in samples:
            cf = s.get("conv_factor_est")
            if isinstance(cf, (int, float)):
                vals.append(float(cf))

    return vals

def _build_argv(out_file: Path) -> List[str]:
    """
    Build a fake argv for scripts.calibrate_conv_factor.main() so we can call it programmatically.
    """
    argv: List[str] = ["calibrate_conv_factor.py"]
    for n in N_LIST:
        argv += ["--n", str(int(n))]
    argv += [
        "--depth-cliff", str(int(DEPTH_CLIFF)),
        "--tail-layers", str(int(TAIL_LAYERS)),
        "--angle-scale", str(float(ANGLE_SCALE)),
        "--samples-per-n", str(int(SAMPLES_PER_N)),
        "--twoq-factor", str(float(TWOQ_FACTOR)),
        "--out", str(out_file),
    ]
    # pass through any custom flags your calibrator supports
    argv += EXTRA_ARGS
    return argv

def _run_one_block() -> Dict[str, Any]:
    """
    Import and invoke the existing calibrator's main() with a synthetic argv, then read its JSON.
    We *do not* spawn a subprocess; we import and call the function directly.
    """
    # import here so this file is importable even if calibrator has heavy imports
    try:
        import scripts.calibrate_conv_factor as ccf
    except Exception as e:
        raise RuntimeError(f"Failed to import scripts.calibrate_conv_factor: {e!r}")

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "calibration.json"

        # Prepare argv for the calibrator
        argv_new = _build_argv(out_path)

        # Save/patch sys.argv and call ccf.main(); absorb SystemExit if it uses argparse
        argv_saved = sys.argv
        sys.argv = argv_new
        try:
            if hasattr(ccf, "main") and callable(ccf.main):
                try:
                    ccf.main()
                except SystemExit as se:
                    if se.code not in (0, None):
                        raise
            else:
                # fallback: if calibrator exposes a different entry, try 'run' or 'cli'
                for cand in ("run", "cli"):
                    if hasattr(ccf, cand) and callable(getattr(ccf, cand)):
                        getattr(ccf, cand)()
                        break
                else:
                    raise RuntimeError("calibrate_conv_factor has no callable 'main', 'run', or 'cli'")
        finally:
            sys.argv = argv_saved

        # load calibrator output
        if not out_path.exists():
            raise RuntimeError(f"Calibration output not found at {out_path}")
        with open(out_path, "r") as f:
            return json.load(f)

def main() -> None:
    # Optional thread pin for reproducibility
    if PIN_THREADS:
        os.environ.setdefault("OMP_NUM_THREADS", PIN_THREADS)

    all_values: List[float] = []
    runs: List[Dict[str, Any]] = []

    t0 = time.time()
    for rep in range(int(REPEAT_BLOCKS)):
        payload = _run_one_block()
        vals = _flatten_conv_values(payload)
        if not vals:
            print("[auto] WARNING: no conv_factor values detected in calibrator output", file=sys.stderr)
        runs.append({"rep": rep, "values": vals})
        all_values.extend(vals)

    if not all_values:
        raise SystemExit("[auto] ERROR: collected zero conv_factor samples; aborting.")

    med = statistics.median(all_values)
    p90 = _percentile(all_values, 0.90)
    p95 = _percentile(all_values, 0.95)
    mx  = max(all_values)

    if POLICY == "balanced":
        recommended = float(med * SAFETY)
        rule = f"median * safety (safety={SAFETY})"
    elif POLICY == "safe":
        recommended = float(p95)
        rule = "P95"
    elif POLICY == "ultra":
        recommended = float(mx)
        rule = "max"
    else:
        raise ValueError(f"Unknown POLICY='{POLICY}' (use 'balanced'|'safe'|'ultra')")

    result = {
        "meta": {
            "n_list": N_LIST,
            "depth_cliff": DEPTH_CLIFF,
            "tail_layers": TAIL_LAYERS,
            "angle_scale": ANGLE_SCALE,
            "samples_per_n": SAMPLES_PER_N,
            "twoq_factor": TWOQ_FACTOR,
            "repeat_blocks": REPEAT_BLOCKS,
            "policy": POLICY,
            "safety": SAFETY,
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "PYTHON": sys.executable,
            },
            "extra_forwarded_args": EXTRA_ARGS,
        },
        "runs": runs,
        "stats": {
            "count": len(all_values),
            "median": med,
            "p90": p90,
            "p95": p95,
            "max": mx,
        },
        "recommended_conv_factor": recommended,
        "recommendation_rule": rule,
        "wall_s": time.time() - t0,
    }

    out_path = Path(OUT_PATH)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[auto] Wrote {out_path} with {len(all_values)} samples")
    print(f"[auto] Recommended conv-factor: {recommended:.2f} "
          f"(median={med:.2f}, p95={p95:.2f}, max={mx:.2f})")

if __name__ == "__main__":
    main()
