# Experiment notebooks

Three notebooks in the repository root drive the experiments described in the
QuASAr evaluation plan. Each notebook generates intermediate CSV artefacts in
`final_results/` and plots in `plots/`.

## `conversion_microbenchmarks.ipynb`

* Recreates and extends the conversion benchmarking sweep.
* Adds fidelity-aware statevector truncation to quantify the runtime/accuracy
  trade-off.
* Saves results to `final_results/conversion_microbenchmarks.csv` and exports
  summary plots.

## `estimator_calibration.ipynb`

* Collects runtime measurements for tableau, statevector, decision diagram, and
  conversion paths.
* Fits the parameters used by `quasar.cost_estimator.CostEstimator`.
* Writes calibration results to `final_results/cost_estimator_calibration.json`
  and `final_results/estimator_calibration_predictions.csv`, alongside a fit
  diagnostic figure.

## `planner_optimality.ipynb`

* Benchmarks QuASAr's planner against exhaustive enumeration on small circuits.
* Reports the relative optimality gap for each circuit.
* Persists a CSV (`final_results/planner_optimality_gap.csv`) and a summary bar
  chart.

All notebooks assume dependencies from `requirements.txt` are available. The
calibration and planner studies require `pandas` for tabular reporting; install
it via `pip install pandas` if needed.
