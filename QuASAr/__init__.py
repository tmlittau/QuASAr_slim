"""Backward-compatible shim for the legacy ``QuASAr`` package name."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "Importing from 'QuASAr' is deprecated; use 'quasar' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public symbols from the new package root.
from quasar import *  # noqa: F401,F403

# Mirror key submodules so legacy ``from QuASAr.foo import …`` statements keep
# working. ``importlib`` ensures we only pay the import cost when the modules
# exist in the new layout.
for _name in [
    "SSD",
    "analyzer",
    "planner",
    "simulation_engine",
    "cost_estimator",
    "backends",
    "baselines",
    "conversion",
]:
    try:
        _module = importlib.import_module(f"quasar.{_name}")
    except ModuleNotFoundError:
        continue
    sys.modules[f"{__name__}.{_name}"] = _module
del _name, _module
