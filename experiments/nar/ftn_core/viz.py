"""Shared visualization tokens (dataviz-method reference palette, light mode)
and small helpers used by every nar/ experiment's figure code."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE = "#c3c2b7"
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
       "#4a3aa7", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

RCPARAMS = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 10, "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
}


def apply_style():
    plt.rcParams.update(RCPARAMS)


def save(fig, out, name, dpi=160):
    fig.savefig(out / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
