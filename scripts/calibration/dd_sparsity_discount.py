from __future__ import annotations

from scripts.calibration.dd_parameter_runner import run_single


def main() -> None:
    run_single("dd_sparsity_discount", "Calibrate the decision diagram sparsity discount parameter.")


if __name__ == "__main__":
    main()
