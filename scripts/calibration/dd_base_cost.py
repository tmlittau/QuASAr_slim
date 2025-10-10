from __future__ import annotations

from scripts.calibration.dd_parameter_runner import run_single


def main() -> None:
    run_single("dd_base_cost", "Calibrate the decision diagram base cost parameter.")


if __name__ == "__main__":
    main()
