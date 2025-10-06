"""Shared colour palette and Matplotlib styling for QuASAr plots."""

from __future__ import annotations

from typing import Dict


# Pastel palette matching the paper figures.
PASTEL_COLORS: Dict[str, str] = {
    "tableau": "#A5D6A7",  # pastel green
    "sv": "#90CAF9",  # pastel blue
    "dd": "#FFCCBC",  # pastel orange
    "conversion": "#CFD8DC",  # pastel grey
}

# Fallback colour when a backend/method-specific colour is not available.
FALLBACK_COLOR = "#B0BEC5"

# Default outline colour for bar plots.
EDGE_COLOR = "#37474F"


def apply_paper_style() -> None:
    """Apply a Matplotlib style roughly matching the paper figures."""

    try:  # Import lazily so the module can be used without Matplotlib installed.
        import matplotlib as mpl
        from cycler import cycler
    except Exception:
        return

    colour_cycle = [
        PASTEL_COLORS["tableau"],
        PASTEL_COLORS["conversion"],
        PASTEL_COLORS["sv"],
        PASTEL_COLORS["dd"],
    ]

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": "#1F2933",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.prop_cycle": cycler(color=colour_cycle),
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "grid.color": "#E0E0E0",
        }
    )

