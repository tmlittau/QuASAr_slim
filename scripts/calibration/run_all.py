from __future__ import annotations

import argparse
import json
import shlex
from typing import Callable

from scripts.calibration import conv_amp_ops_factor, sv_twoq_factor, tableau_prefix_unit_cost
from scripts.calibration.dd_cost_model import build_parser as build_dd_parser, run_from_args as run_dd


ParserBuilder = Callable[[str | None], argparse.ArgumentParser]


def _parse_subargs(builder: ParserBuilder, raw: str | None) -> argparse.Namespace:
    parser = builder(None)
    defaults = parser.parse_args([])
    if raw:
        namespace = parser.parse_args(shlex.split(raw), namespace=defaults)
    else:
        namespace = defaults
    if hasattr(namespace, "out"):
        setattr(namespace, "out", None)
    return namespace


def _conv_namespace(raw: str | None) -> argparse.Namespace:
    return _parse_subargs(conv_amp_ops_factor.build_parser, raw)


def _sv_namespace(raw: str | None) -> argparse.Namespace:
    return _parse_subargs(sv_twoq_factor.build_parser, raw)


def _tableau_namespace(raw: str | None) -> argparse.Namespace:
    return _parse_subargs(tableau_prefix_unit_cost.build_parser, raw)


def _dd_namespace(raw: str | None) -> argparse.Namespace:
    return _parse_subargs(build_dd_parser, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all cost-parameter calibrations and aggregate the results.")
    parser.add_argument("--conv-args", type=str, default=None, help="Additional CLI arguments for the conversion-factor calibrator.")
    parser.add_argument("--sv-args", type=str, default=None, help="Additional CLI arguments for the statevector two-qubit calibrator.")
    parser.add_argument("--tableau-args", type=str, default=None, help="Additional CLI arguments for the tableau unit-cost calibrator.")
    parser.add_argument("--dd-args", type=str, default=None, help="Additional CLI arguments for the decision diagram calibrator.")
    parser.add_argument("--include-dd-report", action="store_true", help="Include the full decision diagram calibration report in the aggregated output.")
    parser.add_argument("--out", type=str, default=None, help="Path to store the aggregated calibration report.")
    args = parser.parse_args()

    conv_ns = _conv_namespace(args.conv_args)
    sv_ns = _sv_namespace(args.sv_args)
    tab_ns = _tableau_namespace(args.tableau_args)
    dd_ns = _dd_namespace(args.dd_args)

    conv_report = conv_amp_ops_factor.run_from_args(conv_ns)
    sv_report = sv_twoq_factor.run_from_args(sv_ns)
    tableau_report = tableau_prefix_unit_cost.run_from_args(tab_ns)
    dd_report = run_dd(dd_ns)

    parameters = {
        "conv_amp_ops_factor": float(conv_report["conv_factor"]["median"]),
        "sv_twoq_factor": float(sv_report["twoq_factor"]),
        "tableau_prefix_unit_cost": float(tableau_report["tableau_prefix_unit_cost"]),
        "dd_base_cost": float(dd_report["dd_base_cost"]),
        "dd_gate_node_factor": float(dd_report["dd_gate_node_factor"]),
        "dd_frontier_weight": float(dd_report["dd_frontier_weight"]),
        "dd_rotation_weight": float(dd_report["dd_rotation_weight"]),
        "dd_twoq_weight": float(dd_report["dd_twoq_weight"]),
        "dd_sparsity_discount": float(dd_report["dd_sparsity_discount"]),
        "dd_modifier_floor": float(dd_report["dd_modifier_floor"]),
    }

    reports = {
        "conv_amp_ops_factor": conv_report,
        "sv_twoq_factor": sv_report,
        "tableau_prefix_unit_cost": tableau_report,
    }
    if args.include_dd_report:
        reports["decision_diagram"] = dd_report

    aggregated = {
        "parameters": parameters,
        "reports": reports,
    }

    output = json.dumps(aggregated, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf8") as handle:
            handle.write(output)
            handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
