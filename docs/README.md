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
python suites/run_hybrid_suite.py --out-dir suite_hybrid --num-qubits 24 28 --block-size 8
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

### Figure and table pipeline

`scripts/make_figures_and_tables.py` orchestrates the full benchmarking loop:
it checks for existing suite outputs (unless `--force` is provided), launches
the runners when necessary, and invokes the matching plotters. The helper
exposes independent sub-commands so you can re-run individual figures without
touching the others.

Run `--help` to inspect the available knobs:

```bash
python scripts/make_figures_and_tables.py --help
```

#### Hybrid benchmark (stacked runtime breakdown)

The `hybrid` sub-command sweeps the stitched and hybrid-style circuits and
builds the stacked bar chart comparing the QuASAr split against the fastest
whole-circuit baseline:

```bash
# Generate hybrid benchmark results and plot
python scripts/make_figures_and_tables.py hybrid \
    --n 24 28 \
    --block-size 8 \
    --out plots/hybrid_stacked.png
```

When you only need a quick sanity check, shrink the sweep so it finishes in a
couple of minutes on a laptop. Dropping the conversion factor to `2000` keeps
the planner from over-estimating costs on small hosts (it will be less
accurate, but still surfaces regressions quickly):

```bash
python scripts/make_figures_and_tables.py hybrid \
    --n 12 16 \
    --block-size 4 \
    --max-ram-gb 32 \
    --conv-factor 2000 \
    --out plots/hybrid_sanity.png
```

Useful flags:

- `--results-dir` to reuse an existing suite output directory (defaults to
  `suite_hybrid`).
- `--force` to re-run the suite even when JSON results are already present.
- `--timeout` to abort the suite runner if it exceeds the specified number of
  seconds (useful in CI to avoid stalls).
- Planner controls (`--max-ram-gb`, `--conv-factor`, `--twoq-factor`) and
  baseline calibration (`--sv-ampops-per-sec`) are forwarded to
  `suites/run_hybrid_suite.py`.

#### Disjoint benchmark (parallel sub-circuits)

The `disjoint` sub-command covers the multi-block disjoint circuits with optional
Clifford/diagonal (random-rotation) tails and emits the QuASAr vs baseline bar
chart:

```bash
# Parallel disjoint benchmark with mixed Clifford + random rotation tails
python scripts/make_figures_and_tables.py disjoint \
    --n 32 48 \
    --blocks 2 4 \
    --prep mixed \
    --tail-kind mixed \
    --tail-depth 20 \
    --angle-scale 0.1 \
    --sparsity 0.05 \
    --bandwidth 2 \
    --out plots/disjoint_runtime.png
```

To iterate faster on workstation-constrained hardware, narrow the sweep while
retaining the mixed-tail structure. Again we lower the conversion factor to
`2000` so that planning stays responsive even on smaller machines:

```bash
python scripts/make_figures_and_tables.py disjoint \
    --n 16 \
    --blocks 2 \
    --prep mixed \
    --tail-kind mixed \
    --tail-depth 10 \
    --angle-scale 0.1 \
    --sparsity 0.05 \
    --bandwidth 2 \
    --max-ram-gb 32 \
    --conv-factor 2000 \
    --out plots/disjoint_sanity.png
```

Key options:

- `--prep` selects the per-block preparation routine (`ghz`, `w`, or `mixed` to
  alternate them).
- `--tail-kind` chooses the tail circuit (`clifford`, `diag`, `mixed`, or
  `none`). The default `mixed` alternates Clifford layers with random diagonal
  rotations so you capture the mixed Clifford + rotation-tail experiment.
- `--tail-depth`, `--angle-scale`, `--sparsity`, and `--bandwidth` refine the
  diagonal tail shape when present.
- `--timeout` mirrors the hybrid workflow and stops the suite if it runs longer
  than the configured limit.
- As with the hybrid workflow, planner and baseline tuning flags are passed
  through to `suites/run_disjoint_suite.py`.

#### Consolidated tables

Once the suites have finished you can summarise them into CSV or Markdown using
the `table` sub-command:

```bash
python scripts/make_figures_and_tables.py table \
    --suite-dir suite_hybrid suite_disjoint \
    --out docs/summary.md
```

The helper inspects each suite directory, extracts the recorded wall-clock
times, and writes a tidy table containing the QuASAr runtime alongside the best
whole-circuit baseline for every case.

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

## SSD visualisation CLI

`scripts/visualize_ssd.py` offers a lightweight way to render the spatial
structure diagrams (SSDs) produced by the circuit analyser. The helper is
particularly convenient when exploring the benchmark generators because it can
instantiate any registered circuit family and immediately draw the associated
dependency graph.

The script depends on the optional `networkx` and `matplotlib` packages. Install
them alongside the core requirements to enable plotting:

```bash
pip install networkx matplotlib
```

Inspect the available circuit families and their canonical names:

```bash
python scripts/visualize_ssd.py --list
```

Once you have a target, generate it (optionally passing builder arguments) and
either show the plot interactively or export it to disk:

```bash
# Render a stitched hybrid circuit and pop up the plot window
python scripts/visualize_ssd.py hybrid_stitched --param n=16 --param block_size=4

# Save the SSD of a disjoint circuit without opening a window
python scripts/visualize_ssd.py disjoint_blocks --param n=24 --param blocks=3 \
    --param prep=ghz --save disjoint_ssd.png --no-show
```

Each `--param` flag accepts `key=value` pairs that are forwarded verbatim to the
underlying `benchmarks.build` helper. String literals can be given with or
without quotes, while more complex types (lists, tuples, numbers) are parsed via
`ast.literal_eval`.

## Calibration and playground scripts

`scripts/` bundles calibration and exploratory tooling:

- `calibrate_conv_factor.py` — fit the conversion cost factor used by the
  planner's hybrid split heuristics.
- `playground_cutoff.py` — sweep Clifford-prefix cutoffs to locate the depth at
  which the hybrid plan outperforms direct statevector simulation. Supports
  saving thresholds to JSON for later benchmarking.
- `playground_speedup_depth.py` — plot SV/Hybrid speedup versus total depth for
  a grid of qubit counts and cutoff fractions.
- `playground_disjoint_speedup.py` — explore statevector vs per-block cost
  speedups for the disjoint prep+tail circuits, sweeping block counts and
  per-block tail depths.
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
