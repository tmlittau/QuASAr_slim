

### Fast `clifford_plus_rot`

```bash
python run_bench.py --kind clifford_plus_rot   --num-qubits 64 --depth 200 --rot-prob 0.2 --angle-scale 0.1   --block-size 8 --pair-scope block --out result.json
```

### Suite with optional qubit cap for non-disjoint circuits

```bash
python run_suite.py --out-dir suite_out --num-qubits 64 96 --block-size 8   --max-ram-gb 64 --sv-ampops-per-sec 5e9 --log-level INFO   --non-disjoint-qubits 32
```
