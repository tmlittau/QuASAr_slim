from __future__ import annotations

import argparse
import ast
import sys
from typing import Dict, Any

from benchmarks import CIRCUIT_REGISTRY, build as build_circuit
from quasar.analyzer import analyze
from quasar.ssd_visualization import visualize_ssd


def _parse_param(values: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"Invalid --param '{item}'. Expected format key=value.")
        key, raw_value = item.split("=", 1)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        params[key] = value
    return params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a benchmark circuit and visualise the resulting SSD graph. "
            "This requires networkx and matplotlib to be installed."
        )
    )
    ap.add_argument(
        "kind",
        nargs="?",
        choices=sorted(CIRCUIT_REGISTRY.keys()),
        help="Circuit kind to generate.",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional circuit builder parameters in key=value form.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="If provided, save the visualisation to this path.",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Disable showing the plot (useful when --save is specified).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List available circuit kinds and exit.",
    )

    args = ap.parse_args(argv)

    if args.list:
        for name in sorted(CIRCUIT_REGISTRY.keys()):
            print(name)
        return 0

    if not args.kind:
        ap.error("circuit kind is required unless --list is provided")

    params = _parse_param(args.param)
    circuit = build_circuit(args.kind, **params)

    analysis = analyze(circuit)
    visualize_ssd(analysis.ssd, show=not args.no_show, save_path=args.save)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
