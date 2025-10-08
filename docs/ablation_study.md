# Streamlined ablation study workflow

This guide walks through the lightweight ablation workflow that exercises
QuASAr's disjoint planning and hybrid Clifford-tail splitting on a single,
configurable circuit. Follow the steps below to generate the JSON artefacts,
validate the theoretical expectations, and produce the comparison plots.

## 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Installing `stim` and `mqt.ddsim` (both shipped in `requirements.txt`) ensures
that the tableau and decision-diagram simulators are available. The ablation
workflow still runs without them, but the hybrid split will fall back to the
statevector backend when a simulator is missing.

## 2. Create the hybrid test circuit

The streamlined helper builds a single circuit that contains several disjoint
hybrid blocks. Each block applies a configurable Clifford prefix followed by a
tail that alternates between dense random rotations and sparse diagonal layers.
The metadata that describes every block is attached to the circuit so you can
inspect it later.

```python
from scripts import run_ablation_study as ras

circuit, specs = ras.build_ablation_circuit(
    num_components=3,
    component_size=4,
    clifford_depth=3,
    tail_depth=3,
    seed=11,
)

print(len(specs))              # number of disjoint hybrid blocks
print(circuit.metadata)
```

Key knobs:

- `num_components` – number of disjoint blocks.
- `component_size` – number of qubits per block.
- `clifford_depth` – number of Clifford-only layers in the prefix.
- `tail_depth` – number of layers in the sparse or random rotation tail.
- `tail_sequence` – optional explicit tail pattern (defaults to alternating
  `"random"` and `"sparse"`).
- `seed` – RNG seed used for reproducible Clifford and tail choices.

## 3. Run the three-way ablation study

The CLI wraps the builder and planner helpers and records the analysis, planned
partitions, and execution metrics for all three variants: the full planner, a
run with disjoint partitioning disabled, and a run with method-based hybrid
splitting disabled. The output is a JSON document that downstream tools can
consume directly.

```bash
python -m scripts.run_ablation_study \
    --num-components 3 \
    --component-size 4 \
    --clifford-depth 3 \
    --tail-depth 3 \
    --seed 11 \
    --out summary.json
```

The script prints the output path on success. To inspect the planner behaviour,
open `summary.json` and look at the `variants[*].partitions` payload.

### Optional: Execute the circuit

By default the CLI only plans the circuit. Add `--execute` to run all three
variants with the configured execution backend and capture wall-clock and memory
usage.

```bash
python -m scripts.run_ablation_study ... --execute
```

The execution results are stored under `variants[*].execution`.

## 4. Validate the theoretical expectations

A regression test ships with the repository to verify that, when conversion
costs are ignored, each disjoint block is split into a tableau prefix and a tail
that lands on the expected backend (statevector for random rotations, decision
diagram for sparse diagonals). Run the test with:

```bash
pytest tests/test_run_ablation_study.py::test_streamlined_hybrid_blocks_split
```

The test also confirms that the disjoint partitions are planned independently so
they can execute in parallel.

## 5. Plot runtime and memory comparisons

The plotting helper loads the JSON summary produced by the CLI and renders
side-by-side bar charts that compare runtime and peak memory for the three
planner variants.

```bash
python -m plots.plot_ablation_bars \
    --summary summary.json \
    --out plots/ablation_bars.png
```

When `--out` is omitted the figures are displayed in an interactive window
instead of being written to disk. Use `--dpi` to control the image resolution and
`--width`/`--height` to customise the figure size (in inches).

## 6. Next steps

Once the single-circuit workflow matches your expectations, scale up the study
by increasing the number of components, block sizes, and depths. Repeat the CLI
run for each configuration you would like to benchmark and archive the resulting
JSON files. The plotting helper can combine multiple summaries to build
side-by-side comparisons across several circuit shapes by invoking it once per
summary and saving each image separately.
