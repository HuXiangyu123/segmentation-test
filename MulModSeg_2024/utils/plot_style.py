#!/usr/bin/env python
"""
Shared publication-style plotting helpers for segmentation experiments.
"""

from pathlib import Path

import matplotlib.pyplot as plt


BASE_PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#9D9D9D",
    "black": "#222222",
    "light_gray": "#D8D8D8",
    "background": "#FFFFFF",
}


FIGURE_DEFAULTS = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.facecolor": BASE_PALETTE["background"],
    "figure.facecolor": BASE_PALETTE["background"],
    "savefig.facecolor": BASE_PALETTE["background"],
    "axes.edgecolor": "#5C5C5C",
    "axes.linewidth": 0.8,
    "grid.color": BASE_PALETTE["light_gray"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.55,
    "legend.frameon": False,
}


def apply_publication_style():
    plt.rcParams.update(FIGURE_DEFAULTS)


def style_axis(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, output_path, save_pdf=True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if save_pdf and output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

