
# QuASAr (slim) — Partitioned & Hybrid Quantum Simulation

This repository is a **slim** re-implementation to generate figures quickly:
- **Analyzer**: finds **disjoint subcircuits** and records metrics.
- **Planner**: assigns simulators per partition and (optionally) applies a **hybrid split** for **Clifford‑prefix + rotation tail** using a **cost estimator**.
- **Simulation Engine**: executes partitions with **parallel chains** and **sequential segments** with **state handoff**.
- **Baselines**: run full‑circuit **tableau**, **statevector**, **decision‑diagram** (best available).
- **Plotting**: total runtime vs baselines + a **stacked bar** view for hybrid runs.
- **Calibration**: estimate the **conversion cost factor** (`conv-factor`) from empirical data.

---

## 1) Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install qiskit qiskit-aer stim mqt.ddsim numpy matplotlib
```

> If `stim` is missing, tableau segments will fall back to SV. Calibration **requires** `stim`.

---

## 2) Key structure

```
QuASAr/
  analyzer.py
  planner.py            # uses CostEstimator for hybrid split
  simulation_engine.py  # chains, heartbeat, state handoff
  SSD.py
  cost_estimator.py     # NEW: amplitude-ops model (conv-factor & twoq-factor)
  calibrate_conv_factor.py  # NEW: automatic conv-factor calibration
  backends/
    sv.py               # supports initial_state
    tableau.py          # Stim-based
    dd.py               # MQT ddsim
  conversion/
    tab2sv.py, tab2dd.py, dd2sv.py
benchmark_circuits.py
run_bench.py
run_suite.py
plot_clifford_tail_bars.py
```

---

## 3) Quick start

### Single run
```bash
python run_bench.py --kind clifford_plus_rot   --num-qubits 64 --block-size 8 --pair-scope block   --depth 200 --rot-prob 0.2 --angle-scale 0.1 --seed 3   --max-ram-gb 64   --conv-factor 64 --twoq-factor 4   --out result.json
```
Inspect `result.json`:
- `analysis.ssd` — partitions and backends with `planner_reason`.
- `execution.results` — timings per partition/segment.
- `execution.meta.wall_elapsed_s` — total runtime.

### Suite
```bash
python run_suite.py --out-dir suite_out --num-qubits 64 96 --block-size 8   --max-ram-gb 64 --sv-ampops-per-sec 5e9   --conv-factor 64 --twoq-factor 4   --log-level INFO --non-disjoint-qubits 32
```
Outputs JSON per case + `index.json` summary.

---

## 4) Hybrid planning & cost estimator

When a partition has a **Clifford-only prefix + non-Clifford tail**, the planner compares:

- **Direct SV**: `SV_total = 2^n * ( #1q_total + twoq_factor * #2q_total )`
- **Hybrid**: `Hybrid = conv_factor * 2^n + 2^n * ( #1q_tail + twoq_factor * #2q_tail )`

If `Hybrid < SV_total`, the partition is split (two segments with the same `chain_id` and `seq_index` 0/1).  
Tune via:
- `--conv-factor` (Tableau→SV conversion cost scale)
- `--twoq-factor` (weight for 2-qubit gates in SV)

The planner writes normalized numbers `(ampops/2^n)` in `planner_reason` for auditing.

---

## 5) Calibrate the conversion factor (conv-factor)

Use empirical runs to compute `conv-factor` automatically:
```bash
python -m QuASAr.calibrate_conv_factor   --n 8 10 12 14   --depth-cliff 150 --tail-layers 10 --angle-scale 0.1   --samples-per-n 3   --twoq-factor 4.0   --out calibration_conv_factor.json
```

How it works:
1. Builds circuits with **Clifford prefix + rotation tail**.
2. Splits at the first non-Clifford.
3. Measures:
   - **prefix time** via **Tableau** (`stim`) → includes conversion to statevector,
   - **tail time** via **SV** seeded with the prefix output.
4. Computes per-sample estimate (note 2^n cancels out):
```
conv_factor_est  ≈  (prefix_elapsed / tail_elapsed) * ( #1q_tail + twoq_factor * #2q_tail )
```
5. Reports **median**, **p25/p75**, and **mean**; writes all samples to JSON.

**Next:** pick the **median** as your `--conv-factor` for the suite runs.

---

## 6) Explore hybrid thresholds (playground)

```bash
python playground_cutoff.py \
  --n 8 10 12 14 16 \
  --min-depth 50 --max-depth 400 --step 25 \
  --cutoff 0.7 0.8 0.9 \
  --conv-factor 64 --twoq-factor 4 --angle-scale 0.1
```
Prints the first prefix length where **Hybrid < SV** for each `n` (using the same estimator as the planner). To save the figure instead use the `--out` flag with the specified location.

---

## 7) Plotting

### Total runtimes (existing suite plotter)
Use your existing script to compare **QuASAr total** vs baselines (not shown here).

### Hybrid stacked bars for `clifford_plus_rot`
```bash
python plot_clifford_tail_bars.py --suite-dir suite_out --out clifford_tail_bars.png
```
- QuASAr bar: **Tableau** (green) + **Conversion** (grey hatched) + **Tail** (blue/orange).  
- Baseline bar: single color of best method.

Conversion time is estimated from the planner’s normalized costs and the measured tail segment.

---

## 8) Tips
- Ensure `stim` and `mqt.ddsim` are installed for Tableau and DD backends.
- For non‑disjoint circuits, cap qubits (e.g., `--non-disjoint-qubits 32`) to avoid SV blow‑ups.
- Use `--pair-scope block` for `clifford_plus_rot` to encourage partitioning.

---

## 9) Reproducibility
- Runner JSON files include:
  - planner parameters (`conv_factor`, `twoq_factor`),
  - partition metrics and `planner_reason`,
  - wall-clock timings.
