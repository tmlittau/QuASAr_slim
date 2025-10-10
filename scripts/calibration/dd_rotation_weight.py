from __future__ import annotations

from scripts.calibration.dd_parameter_runner import run_single


def main() -> None:
    run_single("dd_rotation_weight", "Calibrate the decision diagram rotation density weight.")


if __name__ == "__main__":
    main()
