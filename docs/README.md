# QuASAr slim

QuASAr slim is a lightweight harness for building, planning, and executing
partitioned quantum simulation experiments. This document consolidates the
project overview, the runnable entry points, and the plotting / playground
tools that ship with the repository.

## Repository layout

```
quasar/        # Core library: analyzer, planner, execution engine, backends
benchmarks/    # Circuit generators grouped by use-case
suites/        # End-to-end runners that produce JSON artefacts
plots/         # Plotting utilities for suite outputs
scripts/       # Calibration, playground, and orchestration helpers
docs/          # Project documentation (this file)
```

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # installs qiskit, stim, mqt.ddsim, numpy, matplotlib
```

`stim` and `mqt.ddsim` are optional but unlock the tableau and decision-diagram
backends. Without them the planner automatically falls back to statevector
simulation.

## Benchmarks

Circuit generators live under `benchmarks/`:

- `benchmarks.hybrid` — GHZ clusters, stitched banded-QFT circuits, and
  Clifford+rotation hybrids. Exposes `CIRCUIT_REGISTRY` and `build(kind, **kw)`.
- `benchmarks.dd_friendly` — line-graph Clifford prefixes with sparse diagonal
tails optimised for decision-diagram simulators.
- `benchmarks.disjoint` — block-disjoint preparation + tail families with
  configurable GHZ/W prep and Clifford/diagonal tails. Blocks are constructed
  without cross-couplings so they can be simulated in parallel.
- `benchmarks.__init__` — registry aggregator. Import
  `from benchmarks import build` to fetch any circuit by name.

## Suite runners

The `suites/` directory contains ready-to-run experiments that write per-case
JSON files and an `index.json` summary. Example usage:

```bash
python suites/run_hybrid_suite.py --out-dir suite_hybrid --num-qubits 64 96 --block-size 8
python suites/run_dd_friendly_suite.py --out-dir suite_dd --n 16 24 32 --depth 100 200
python suites/run_disjoint_suite.py --out-dir suite_disjoint --n 32 48 --blocks 2 4 \
    --prep mixed --tail-kind mixed --tail-depth 20
```

All suites accept planner controls (`--conv-factor`, `--twoq-factor`,
`--max-ram-gb`) and emit the QuASAr analysis, plan, execution payload, and
baseline measurements per circuit. The disjoint sweep further allows tuning the
block preparation (`--prep`), tail type (`--tail-kind`), and per-block tail
depth/angles so you can probe GHZ or W structures with optional Clifford or
diagonal tails.

To orchestrate multiple suites and reproduce the paper figures in one shot, use
`scripts/make_figures_and_tables.py`:

```bash
python scripts/make_figures_and_tables.py --workspace paper_artifacts
```

The script runs the hybrid, DD-friendly, and disjoint suites (unless skipped via
`--skip-*` flags), produces stacked bar plots in the workspace directory, and
writes a Markdown summary table aggregating the index files. Use `--*-extra` to
forward additional CLI arguments to individual suites. For the disjoint command,
the helper also invokes `plots/bar_disjoint.py` so you receive a side-by-side bar
chart comparing the parallel QuASAr wall time against the fastest whole-circuit
baseline (statevector, decision diagram, or tableau, colour-coded blue/orange/
green).

## Plotting utilities

All plotters now live under `plots/`:

- `plots/bar_hybrid.py` — stacked bars for QuASAr vs whole-circuit baselines for
  hybrid-style circuits (also used by the DD-friendly suite).
- `plots/bar_clifford_tail.py` — stacked bars focused on Clifford-prefix vs
  rotation-tail experiments.
- `plots/bar_disjoint.py` — parallel disjoint benchmark comparison. Produces a
  two-bar chart per `(n, blocks)` pair showing QuASAr's parallel runtime next to
  the best whole-circuit baseline with the simulator type encoded in the bar
  colour.
- `plots/compare_total.py` — convenience helpers to compare total QuASAr vs
  baseline runtimes or visualise SSD partition timings.

Each plotter accepts `--suite-dir` (directory with suite JSON files) and `--out`
for the output image path.

## Calibration and playground scripts

`scripts/` bundles calibration and exploratory tooling:

- `calibrate_conv_factor.py` — fit the conversion cost factor used by the
  planner's hybrid split heuristics.
- `playground_cutoff.py` — sweep Clifford-prefix cutoffs to locate the depth at
  which the hybrid plan outperforms direct statevector simulation. Supports
  saving thresholds to JSON for later benchmarking.
- `playground_speedup_depth.py` — plot SV/Hybrid speedup versus total depth for
  a grid of qubit counts and cutoff fractions.
- `bench_from_thresholds.py` — consume a thresholds JSON (emitted by the cutoff
  playground) and automatically run QuASAr + baselines at the selected points,
  writing stacked bar plots to the output directory.
- `make_figures_and_tables.py` — orchestrate suite execution and figure
  generation for the paper pipeline (see above).

All scripts provide `--help` for full CLI details.

## Testing helpers

`suites/` and `scripts/` cover the full pipeline, but you can also run smaller
checkpoints:

```bash
python run_bench.py --kind clifford_plus_rot --num-qubits 64 --block-size 8 \
    --depth 200 --rot-prob 0.2 --angle-scale 0.1 --seed 3 --out result.json
python run_baselines.py --kind ghz_clusters_random --num-qubits 64 --depth 200
python test_clifford_tail.py  # smoke test for hybrid splitting
```

Each command writes JSON outputs containing the analysis metrics, SSD
assignment, execution timings, and baseline results.
