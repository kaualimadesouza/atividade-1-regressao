"""Shared look for the report figures, same validated palette as the guide's docs/img."""

from matplotlib.axes import Axes

SERIES_1 = "#2a78d6"
SURFACE = "#fcfcfb"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(color=MUTED, labelcolor=INK_SECONDARY)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
