
# Threshold → Benchmark → Bars Pipeline

This patch adds:
1. `playground_cutoff.py` **saving thresholds** with `--save-json`.
2. `bench_from_thresholds.py` which **runs benchmarks** at those thresholds and **plots** the stacked bar chart.

## Save thresholds
```bash
python playground_cutoff.py   --n 8 10 12 14 16   --min-depth 50 --max-depth 400 --step 10   --cutoff 0.8   --angle-scale 0.1 --conv-factor 64 --twoq-factor 4   --target-speedup 1.25   --save-json thresholds_0p8.json   --out thresholds_0p8.png
```

## Run benchmarks + plot
```bash
python bench_from_thresholds.py   --thresholds thresholds_0p8.json   --out-dir suite_from_thresholds   --max-ram-gb 64
```
This creates JSON case files and the figure:
`suite_from_thresholds/bars_from_thresholds.png`.

If your thresholds JSON contains multiple cutoffs, add `--cutoff 0.8` (or whichever you want).

The plotter now supports both `clifford_plus_rot` and `clifford_prefix_rot_tail` cases.
