
# Depth Sweep Playground — Speedup vs Depth

This script sweeps total depth in `[min_depth, max_depth]` (step `--step`) and plots
**speedup = SV_cost / Hybrid_cost** vs **depth** for each `(n, cutoff)` pair.

## Usage
```bash
python playground_speedup_depth.py   --n 8 12 16   --min-depth 50 --max-depth 400 --step 10   --cutoff 0.8   --angle-scale 0.1 --conv-factor 64 --twoq-factor 4   --target-speedup 1.25   --out speedup_vs_depth.png   --save-json speedup_vs_depth.json
```
- Multiple `--cutoff` and `--n` values are supported; the plot shows one line per `(n, cutoff)`.
- A horizontal dashed line marks `--target-speedup`.
- If your backend is non-interactive, the script automatically saves the figure.
