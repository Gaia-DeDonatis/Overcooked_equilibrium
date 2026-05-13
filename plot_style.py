#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared plotting style for the Overcooked Human-AI co-adaptation analyses.

Use this file in every analysis script so that all figures use the same:
    - font size
    - figure size
    - color palette
    - line widths
    - marker sizes
    - grid style
    - output settings

Example
-------
    import matplotlib.pyplot as plt
    from plot_style import apply_plot_style, save_figure, COLORS, FIGSIZE_WIDE

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot([1, 2, 3], [2, 4, 3], color=COLORS["best_policy"])
    save_figure(fig, "figures/example.png")

Notes
-----
    - Base font size is 12 pt to match the Overleaf document.
    - Seaborn is used if installed.
    - If seaborn is not installed, the file falls back to Matplotlib rcParams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


# ============================================================
# CORE STYLE SETTINGS
# ============================================================

BASE_FONT_SIZE = 12
FONT_FAMILY = "Roboto"

# Standard figure sizes in inches.
FIGSIZE_SMALL = (5.2, 3.8)
FIGSIZE_MEDIUM = (6.4, 4.4)
FIGSIZE_WIDE = (8.0, 4.8)
FIGSIZE_TALL = (6.4, 5.8)
FIGSIZE_RADAR = (6.2, 6.2)

DPI = 300

LINE_WIDTH = 2.0
MEAN_LINE_WIDTH = 2.5
GRID_LINE_WIDTH = 1.0

MARKER_SIZE = 4.8
MARKER_EDGE_WIDTH = 1.15
MARKER_EDGE_COLOR = "#FFFFFF"

SCATTER_SIZE = 60
MEAN_MARKER_SIZE = MARKER_SIZE ** 2
SKIPPED_MARKER_SIZE = 130

ALPHA_LINE = 0.92
ALPHA_FILL = 0.18
ALPHA_GRID = 0.30

LINE_MARKER_KWS = {
    "marker": "o",
    "markersize": MARKER_SIZE,
    "markeredgecolor": MARKER_EDGE_COLOR,
    "markeredgewidth": MARKER_EDGE_WIDTH,
    "linewidth": LINE_WIDTH,
}

MEAN_LINE_MARKER_KWS = {
    "marker": "o",
    "markersize": MARKER_SIZE,
    "markeredgecolor": MARKER_EDGE_COLOR,
    "markeredgewidth": MARKER_EDGE_WIDTH,
    "linewidth": MEAN_LINE_WIDTH,
}

# ============================================================
# COLOR PALETTE
# ============================================================

SEABORN_STYLE = "darkgrid"
SEABORN_PALETTE_NAME = "deep"
N_PALETTE_COLORS = 10

SEABORN_AXES_FACE = "#F7F7FB"
SEABORN_GRID_COLOR = "#FFFFFF"

try:
    import seaborn as sns
    MAIN_PALETTE = sns.color_palette(SEABORN_PALETTE_NAME, N_PALETTE_COLORS).as_hex()
except ImportError:
    MAIN_PALETTE = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]

# Neutral and warning colors.
NEUTRAL_DARK = "#222222"
NEUTRAL_MID = "#666666"
NEUTRAL_LIGHT = "#D8D8E0"
BACKGROUND = "#FFFFFF"
WARNING = "#FFB000"

# Semantic colors. Use these names in scripts instead of hard-coding colors.
COLORS = {
    # Agents
    "human": MAIN_PALETTE[9],
    "ai": MAIN_PALETTE[0],

    # Main story conditions
    "best_policy": NEUTRAL_DARK,
    "replay_changed_behavior": MAIN_PALETTE[0],
    "closest_policy": MAIN_PALETTE[4],
    "stress_neighbor": MAIN_PALETTE[1],

    # Experiment phases
    "seed": MAIN_PALETTE[7],
    "bo": MAIN_PALETTE[0],
    "bo_replay_best": NEUTRAL_DARK,
    "stress": MAIN_PALETTE[1],
    "replay_optimal": MAIN_PALETTE[0],
    "solo": NEUTRAL_MID,

    # General plotting
    "group_mean": NEUTRAL_DARK,
    "skipped": WARNING,
    "text": NEUTRAL_DARK,
    "grid": SEABORN_GRID_COLOR,
    "background": BACKGROUND,
}

CONDITION_COLORS = {
    "Best-policy": COLORS["best_policy"],
    "Replay changed human behavior": COLORS["replay_changed_behavior"],
    "Replay-different-human-behavior": COLORS["replay_changed_behavior"],
    "Changed human behavior": COLORS["replay_changed_behavior"],
    "Closest policy": COLORS["closest_policy"],
    "Stress neighbor": COLORS["stress_neighbor"],
    "Solo": COLORS["solo"],
}

PHASE_COLORS = {
    "seed": COLORS["seed"],
    "bo": COLORS["bo"],
    "bo_replay_best": COLORS["bo_replay_best"],
    "stress": COLORS["stress"],
    "replay_optimal": COLORS["replay_optimal"],
    "solo": COLORS["solo"],
}

PARTICIPANT_COLORS = {
    f"Thinpath_P{i:02d}": MAIN_PALETTE[(i - 1) % len(MAIN_PALETTE)]
    for i in range(1, 11)
}


# ============================================================
# CONSISTENT LABELS
# ============================================================

PHASE_LABELS = {
    "seed": "Seed",
    "bo": "BO",
    "bo_replay_best": "Best-policy",
    "stress": "Stress / neighbor",
    "replay_optimal": "Replay\nchanged behavior",
    "solo": "Solo",
}

CONDITION_LABELS = {
    "bo_replay_best": "Best-policy",
    "replay_optimal": "Replay changed behavior",
    "solo": "Solo",
    "stress": "Stress / neighbor",
}

METRIC_LABELS = {
    "dishes_served": "Dishes served",
    "mean_dishes_per_round": "Mean dishes per round",
    "team_reward_score": "Team reward",
    "human_reward_score": "Human reward",
    "ai_reward_score": "AI reward",
    "human_steps": "Human steps",
    "ai_steps": "AI steps",
    "human_steps_per_dish": "Human steps per dish",
    "mental_demand": "Mental demand",
    "performance_score": "Subjective performance",
    "episode_coadaptation_score": "Co-adaptation score",
    "policy_distance": "Policy-space distance",
    "role_distance": "Role distance from Best-policy",
}

# Raw task columns saved in task_division_summary.csv can be grouped into
# simpler task labels for radar plots.
TASK_GROUPS = {
    "pickup_lettuce": ["pickup_lettuce_1", "pickup_lettuce_2"],
    "pickup_plate": ["pickup_plate_1", "pickup_plate_2"],
    "pickup_from_knife": ["pickup_from_knife"],
    "place_counter": ["place_on_counter"],
    "place_knife": ["place_on_knife"],
    "chop": ["chop"],
    "assemble": ["assemble_salad"],
    "deliver": ["deliver_correct"],
}

TASK_LABELS = {
    "pickup_lettuce": "Pick up\nlettuce",
    "pickup_plate": "Pick up\nplate",
    "pickup_from_knife": "Pick up\nfrom knife",
    "place_counter": "Place on\ncounter",
    "place_knife": "Place on\nknife",
    "chop": "Chop",
    "assemble": "Assemble\nsalad",
    "deliver": "Deliver",
}


# ============================================================
# STYLE FUNCTIONS
# ============================================================

def apply_plot_style(
    context: str = "paper",
    font_scale: float = 1.0,
    use_seaborn: bool = True,
) -> None:
    """
    Apply the shared plotting style.

    Parameters
    ----------
    context:
        Seaborn plotting context. Use "paper" for manuscript figures.

    font_scale:
        Multiplier applied to the base 12 pt font size.

    use_seaborn:
        If True, use seaborn.set_theme when seaborn is installed.
        If False, or if seaborn is unavailable, use Matplotlib rcParams only.
    """
    rc = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
        "font.size": BASE_FONT_SIZE * font_scale,
        "axes.titlesize": BASE_FONT_SIZE * 1.08 * font_scale,
        "axes.labelsize": BASE_FONT_SIZE * font_scale,
        "xtick.labelsize": BASE_FONT_SIZE * 0.86 * font_scale,
        "ytick.labelsize": BASE_FONT_SIZE * 0.86 * font_scale,
        "legend.fontsize": BASE_FONT_SIZE * 0.86 * font_scale,
        "figure.titlesize": BASE_FONT_SIZE * 1.12 * font_scale,

        # Lines and markers
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "patch.linewidth": 0.8,

        # Axes
        "axes.edgecolor": NEUTRAL_DARK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": NEUTRAL_DARK,
        "axes.titlecolor": NEUTRAL_DARK,
        "axes.facecolor": SEABORN_AXES_FACE,
        "axes.xmargin": 0.04,
        "axes.ymargin": 0.06,

        # Grid
        "grid.color": SEABORN_GRID_COLOR,
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,

        # Figure and output
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,

        # Better vector text in PDF/Illustrator
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Ticks
        "xtick.color": NEUTRAL_DARK,
        "ytick.color": NEUTRAL_DARK,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }

    if use_seaborn:
        try:
            import seaborn as sns

            sns.set_theme(
                context=context,
                style=SEABORN_STYLE,
                palette=MAIN_PALETTE,
                font=FONT_FAMILY,
                rc=rc,
            )
            return
        except ImportError:
            print("[WARN] seaborn is not installed. Falling back to Matplotlib style.")

    plt.rcParams.update(rc)


def clean_axis(ax, grid_axis: str = "y") -> None:
    """
    Apply final axis cleanup.

    Parameters
    ----------
    ax:
        Matplotlib axis.

    grid_axis:
        "x", "y", or "both".
    """
    ax.grid(True, axis=grid_axis, color=SEABORN_GRID_COLOR, linewidth=GRID_LINE_WIDTH, alpha=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(
    fig,
    output_path: str | Path,
    formats: Iterable[str] | None = None,
    dpi: int = DPI,
    close: bool = False,
) -> None:
    """
    Save a figure consistently.

    Parameters
    ----------
    fig:
        Matplotlib figure.

    output_path:
        Target path. If formats is None, the suffix decides the format.
        Example: save_figure(fig, "plot.png")

    formats:
        Optional list of formats. If provided, saves multiple versions.
        Example: save_figure(fig, "plot", formats=["png", "pdf"])

    dpi:
        Output DPI for raster formats.

    close:
        If True, close the figure after saving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    else:
        stem = output_path.with_suffix("")
        for fmt in formats:
            fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)


def get_participant_color(participant_id: str, fallback_index: int = 0) -> str:
    """
    Return a stable participant color.

    Participants named Thinpath_P01 ... Thinpath_P10 get fixed colors.
    Other IDs cycle through the same Seaborn palette.
    """
    if participant_id in PARTICIPANT_COLORS:
        return PARTICIPANT_COLORS[participant_id]

    return MAIN_PALETTE[fallback_index % len(MAIN_PALETTE)]


def phase_label(phase: str) -> str:
    """Return a clean display label for an experiment phase."""
    return PHASE_LABELS.get(str(phase), str(phase))


def condition_label(condition: str) -> str:
    """Return a clean display label for a condition or phase."""
    return CONDITION_LABELS.get(str(condition), str(condition))


def metric_label(metric: str) -> str:
    """Return a clean display label for a metric column."""
    return METRIC_LABELS.get(str(metric), str(metric).replace("_", " ").title())


def task_label(task: str) -> str:
    """Return a clean display label for a radar-plot task group."""
    return TASK_LABELS.get(str(task), str(task).replace("_", " ").title())


def add_panel_label(
    ax,
    label: str,
    x: float = -0.10,
    y: float = 1.08,
) -> None:
    """
    Add a panel label such as 'A', 'B', or 'C' to a subplot.
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=BASE_FONT_SIZE * 1.2,
        fontweight="bold",
        va="top",
        ha="left",
        color=NEUTRAL_DARK,
    )
