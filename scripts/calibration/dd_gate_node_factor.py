from __future__ import annotations

from scripts.calibration.dd_parameter_runner import run_single


def main() -> None:
    run_single("dd_gate_node_factor", "Calibrate the decision diagram gate-node factor.")


if __name__ == "__main__":
    main()
