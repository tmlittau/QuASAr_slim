
# DD-friendly sparse circuit (prefix + diagonal tail)

This patch adds a **sparse, DD-friendly circuit generator** and wires it into the stacked **QuASAr vs whole‑circuit baseline** bar chart.

## Why DD-friendly?
Decision-diagram (DD) simulators compress circuits with **regular** or **diagonal/sparse** structure exceptionally well. Examples include **GHZ/BV** and **adder**/arithmetic circuits; in general, DD excels when it can exploit **regularity** and **repetition** in the state/unitary.

## References
- DD compression leverages **circuit regularity**; DDSIM performs extremely well on **Adder** and **GHZ** circuits. (FlatDD paper) 
- DD approaches efficiently handle **GHZ** and **BV** circuits; IQP-style linear network circuits are commonly considered. (FeynmanDD chapter)

## Generator
`dd_friendly_prefix_diag_tail(num_qubits, depth, cutoff, angle_scale, tail_sparsity, tail_bandwidth, seed)`

- **Prefix (Clifford):** repeated **line-graph state** layers (H on all + nearest‑neighbor CZ).
- **Tail (diagonal, sparse):** random **sparse RZ** rotations + **sparse CZ** within small bandwidth—an **IQP‑like** diagonal layer that remains favorable for DDs while introducing non‑Clifford gates.

## Run and plot
```bash
python run_dd_friendly_suite.py   --n 16 24 32   --depth 100 200   --cutoff 0.8   --angle-scale 0.1 --tail-sparsity 0.05 --tail-bandwidth 2   --out-dir suite_dd_friendly   --conv-factor 64 --twoq-factor 4 --max-ram-gb 64
```
This writes JSONs and `suite_dd_friendly/bars_dd_friendly.png`.
