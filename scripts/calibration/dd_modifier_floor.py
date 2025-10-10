from __future__ import annotations

from scripts.calibration.dd_parameter_runner import run_single


def main() -> None:
    run_single("dd_modifier_floor", "Calibrate the decision diagram modifier floor.")


if __name__ == "__main__":
    main()
