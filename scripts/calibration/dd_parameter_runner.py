from __future__ import annotations

import json
from typing import Any

from scripts.calibration.dd_cost_model import build_parser, run_from_args


def _pack_output(parameter: str, report: dict, include_full: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        parameter: float(report.get(parameter, 0.0)),
        "num_samples": int(report.get("num_samples", 0)),
        "r2": float(report.get("r2", 0.0)),
        "residual_sum_sqr": float(report.get("residual_sum_sqr", 0.0)),
    }
    if include_full:
        payload["full_report"] = report
    return payload


def run_single(parameter: str, description: str) -> None:
    parser = build_parser(description)
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Include the complete decision diagram calibration report in the JSON output.",
    )
    args = parser.parse_args()

    report = run_from_args(args)
    payload = _pack_output(parameter, report, args.full_report)
    output = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf8") as handle:
            handle.write(output)
            handle.write("\n")
    print(output)
