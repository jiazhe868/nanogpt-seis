"""Shared style for all figures — one coherent visual system.

Palette is the dataviz reference categorical set (CVD-safe, validated):
blue / aqua / yellow / green / violet / red / magenta / orange, on a warm
off-white surface with ink-toned text and a recessive grid.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical hues (fixed order)
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

# ink / surface tokens
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a897f"
GRID = "#e6e6e3"
SURFACE = "#ffffff"      # white background
PANEL = "#f2f1ec"        # soft fill for boxes
AXIS = "#D3D3D3"         # thin light-gray axes/spines

# --- graphviz diagrams: soft neutral bg, dark-gray borders/edges, pastel fills ---
GV_BG = "#f7f7f5"        # soft neutral background
GV_BORDER = "#424242"    # thin dark-gray node border
GV_EDGE = "#616161"      # charcoal-gray data-flow edges
GV_CLUSTER = "#BDBDBD"   # cluster border
GV_BLUE = "#CFE2F3"      # embeddings / crawl
GV_YELLOW = "#FFF2CC"    # attention / tokenize
GV_GREEN = "#D9EAD3"     # feed-forward / process
GV_PINK = "#F4CCCC"      # normalization / train
GV_PURPLE = "#E1D5E7"    # model
GV_TEAL = "#D0E8E4"      # general sources / inference
GV_NEUTRAL = "#ECECEC"   # I/O, residual-add nodes


def save(fig, stem):
    """Save a matplotlib figure as both PNG and PDF (stem = path without extension)."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")


def gv_render(graph, stem):
    """Render a graphviz graph to both PNG and PDF (stem = path without extension).

    PNG at 300 dpi (raster resolution); PDF at 72 dpi so the vector page is its
    natural point size — otherwise a high dpi inflates the PDF media box ~4x and
    the page overflows normal paper.
    """
    for ext, dpi in (("png", "300"), ("pdf", "72")):
        graph.attr(dpi=dpi)
        graph.render(outfile=f"{stem}.{ext}", cleanup=True)


def apply_base():
    """Minimalist house style: white background, Arial (Liberation Sans fallback),
    thin light-gray axes, no top/right frame, 300-dpi tight saves."""
    plt.rcParams.update({
        # white surfaces
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.dpi": 300, "savefig.bbox": "tight",
        # Arial (falls back to the metric-compatible Liberation/Nimbus Sans)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # thin light-gray axes; readable ink tick labels
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
