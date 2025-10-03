
# QuASAr (slim)

A minimal, pluggable pipeline to analyze a Qiskit circuit, partition it into independent subcircuits,
plan per-partition backends (Tableau / Decision Diagram / Statevector), and execute.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_bench.py --kind ghz_clusters_random --num-qubits 64 --block-size 8 --depth 200 --out result.json
```

Flags:
- `--prefer-dd` prefers DD over SV for larger partitions when both feasible.
- `--max-ram-gb` sets the statevector cap for planning.

Results are stored in `result.json` containing analysis metrics, SSD, and simple execution outputs.
