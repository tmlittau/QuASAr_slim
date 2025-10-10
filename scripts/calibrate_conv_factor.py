"""Backward-compatible wrapper for the new calibration entry-point."""

from __future__ import annotations

from scripts.calibration.conv_amp_ops_factor import main


if __name__ == "__main__":
    main()
