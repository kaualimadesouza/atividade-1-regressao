"""One look for every figure, the report's and the guide's: the palette and the rc settings that apply it."""

from collections.abc import Callable
from functools import wraps

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

# Validated categorical pair (dataviz reference palette, slots 1 and 2) plus neutral chrome
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-4, 9),  # concentrations up to millions stay plain, tiny densities go scientific
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
}


def styled[**P](build: Callable[P, Figure]) -> Callable[P, Figure]:
    """Decorator: build the figure with RC active, so every artist gets the shared look."""

    @wraps(build)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Figure:
        with mpl.rc_context(RC):
            return build(*args, **kwargs)

    return wrapper


def scatter(ax: Axes, x: ArrayLike, y: ArrayLike, label: str | None = None) -> None:
    """Data dots, same look in every figure."""
    ax.scatter(x, y, s=36, color=SERIES_1, edgecolor=SURFACE, linewidth=1, zorder=3, label=label)


def zero_line(ax: Axes) -> None:
    """Horizontal line at y = 0, the reference residuals should scatter around."""
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=2)
