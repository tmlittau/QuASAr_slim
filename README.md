
# QuASAr (slim)

Minimal pipeline: analyze a Qiskit circuit, partition into independent subcircuits, plan per-partition backends,
and execute with a multithreaded simulation engine. Includes baseline runners and plotting.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Unified run (status logging + multithreading)

```bash
python run_all.py --kind ghz_clusters_random --num-qubits 64 --block-size 8 --depth 200   --max-ram-gb 64 --max-workers 0 --heartbeat-sec 5 --stuck-warn-sec 60 --out result.json --log-level INFO
```

## Baselines

```bash
# Whole-circuit baselines (all three)
python run_baselines.py --kind ghz_clusters_random --num-qubits 64 --block-size 8 --depth 200 --which all --out baselines.json

# Per-partition baselines with an SV cap
python run_baselines.py --kind ghz_clusters_random --num-qubits 64 --block-size 8 --depth 200 --which sv,dd --per-partition --max-ram-gb 64
```

## Plotting

```bash
# From SSD run (result.json)
python plot_results.py --mode ssd --input result.json --out ssd_runtimes.png

# From baselines (baselines.json)
python plot_results.py --mode baselines --input baselines.json --out baseline_runtimes.png
```
