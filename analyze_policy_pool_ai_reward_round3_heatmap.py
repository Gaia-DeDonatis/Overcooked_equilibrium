#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph 4 variant: Policy-pool heatmap using AI reward score from round 3.

For each participant, this script creates a map of the policy pool:

    - full policy pool in the background, shown in light grey
    - smooth/interpolated heatmap showing where AI reward score is high/low
    - played policies shown as dots
    - every played policy dot labelled with the episode number, e.g. E1, E2, E13
    - episode 13 highlighted as a black dot
    - worst episode highlighted as a red dot

Important:
    This version does NOT average AI reward score across the full episode.
    Instead, for each episode it uses only:

        ai_reward_score from round_in_episode == 3

    By default, this script only includes episodes up to E16.
    This excludes the final episode if the final episode is E17.

    The black dot is episode 13 by default.

    The heatmap is based on the observed participant episodes. It cannot know
    the true AI reward score of unplayed policies; it interpolates from the
    played policy outcomes in the embedding space.

Example
-------
PowerShell single-line version:

python analyze_policy_pool_ai_reward_round3_heatmap.py --data-root submissions --embedding-csv tsnt_thinpath.csv --output-dir figures/policy_pool_heatmap_ai_reward_round3_until_E16 --formats png pdf

PowerShell multi-line version:

python analyze_policy_pool_ai_reward_round3_heatmap.py `
  --data-root submissions `
  --embedding-csv tsnt_thinpath.csv `
  --output-dir figures/policy_pool_heatmap_ai_reward_round3_until_E16 `
  --formats png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from plot_style import (
    apply_plot_style,
    save_figure,
    metric_label,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_POLICY_PREFIX = "[coplay][flexible][thinpath]agent0_"

# This script is specifically for AI reward score from round 3.
DEFAULT_METRIC = "ai_reward_score_round3"
DEFAULT_REWARD_ROUND = 3

# Episode 13 is the Best-policy replay episode in the current design.
DEFAULT_BEST_EPISODE_INDEX = 13

# Only show data up to this episode.
# This excludes the final episode if the final episode is E17.
DEFAULT_MAX_EPISODE_INDEX = 16

LOWER_IS_BETTER_METRICS = {
    "human_steps_per_dish",
    "mental_demand",
}

LABEL_OVERRIDES = {
    "ai_reward_score_round3": "AI reward score, round 3",
    "ai_reward_score": "AI reward score",
    "mean_dishes_per_round": "Mean dishes per round",
    "human_steps_per_dish": "Human steps per dish",
    "team_reward_score": "Team reward score",
    "human_reward_score": "Human reward score",
    "mental_demand": "Mental demand",
    "performance_score": "Performance score",
}

# Cubehelix palette.
CUBEHELIX_START = 2.4
CUBEHELIX_ROT = -0.25
CUBEHELIX_DARK = 0.18
CUBEHELIX_LIGHT = 0.96

HEATMAP_LEVELS = 18

# Smaller, cleaner font sizes.
TITLE_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
EPISODE_LABEL_FONT_SIZE = 7.5
COLORBAR_FONT_SIZE = 8.5


# ============================================================
# STYLE
# ============================================================

def apply_heatmap_text_style() -> None:
    """Use smaller text for this heatmap figure."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
        }
    )


def build_cubehelix_cmap():
    return sns.cubehelix_palette(
        start=CUBEHELIX_START,
        rot=CUBEHELIX_ROT,
        dark=CUBEHELIX_DARK,
        light=CUBEHELIX_LIGHT,
        as_cmap=True,
    )


def label_for_metric(metric: str, reward_round: int | None = None) -> str:
    """Return a readable label for the metric."""
    if metric == "ai_reward_score_round3":
        if reward_round is None:
            return LABEL_OVERRIDES[metric]
        return f"AI reward score, round {reward_round}"

    if metric in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[metric]

    try:
        return metric_label(metric)
    except Exception:
        return str(metric).replace("_", " ").title()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def participant_id_from_file(csv_path: Path, df: pd.DataFrame) -> str:
    folder_name = csv_path.parent.name
    if folder_name:
        return folder_name

    if "prolific_id" in df.columns and df["prolific_id"].notna().any():
        return str(df["prolific_id"].dropna().iloc[0])

    return "unknown_participant"


def find_round_summary_files(data_root: Path) -> list[Path]:
    """Find round_summary.csv files under a submissions-like root."""
    data_root = data_root.expanduser().resolve()

    if data_root.is_file() and data_root.name == "round_summary.csv":
        return [data_root]

    if (data_root / "round_summary.csv").exists():
        return [data_root / "round_summary.csv"]

    files = sorted(data_root.glob("*/round_summary.csv"))
    if not files:
        files = sorted(data_root.rglob("round_summary.csv"))

    return files


def strip_policy_prefix(policy_name: str, policy_prefix: str | None = None) -> str:
    """
    Normalize policy names so matching works whether policy_id is stored as:

        a0sp_0_helping0_True_gamma0.8

    or:

        [coplay][flexible][thinpath]agent0_a0sp_0_helping0_True_gamma0.8
    """
    p = str(policy_name or "").strip()

    if not p or p.lower() in {"nan", "none", "null", "solo", "no_ai"}:
        return ""

    if policy_prefix and p.startswith(policy_prefix):
        return p[len(policy_prefix):]

    if "agent0_" in p:
        return p.split("agent0_", 1)[1]

    return p


def is_valid_policy(policy_short: str) -> bool:
    p = str(policy_short or "").strip().lower()
    return p not in {"", "nan", "none", "null", "solo", "no_ai"}


def episode_label(values: Iterable[int | float]) -> str:
    """
    Create compact labels for one policy used in one or more episodes.

    Examples:
        [13] -> E13
        [13, 16] -> E13/E16
    """
    cleaned = []
    for v in values:
        if pd.notna(v):
            cleaned.append(int(v))

    cleaned = sorted(set(cleaned))

    if not cleaned:
        return ""

    return "/".join(f"E{v}" for v in cleaned)


def filter_episodes_up_to(
    episode_summary: pd.DataFrame,
    max_episode_index: int | None,
) -> pd.DataFrame:
    """
    Keep only episodes up to max_episode_index.

    By default this keeps E1 through E16 and excludes the final episode if
    the final episode is E17.
    """
    if max_episode_index is None:
        return episode_summary.copy()

    out = episode_summary.loc[
        episode_summary["episode_index"].notna()
        & (episode_summary["episode_index"] <= int(max_episode_index))
    ].copy()

    if out.empty:
        raise ValueError(
            f"No episodes remain after filtering to episode_index <= {max_episode_index}."
        )

    return out


def get_round_value(g: pd.DataFrame, value_col: str, round_number: int) -> float:
    """
    Return value_col from a specific round inside one episode.

    If multiple matching rows exist, their mean is used.
    If the requested round is missing, return NaN.
    """
    if "round_in_episode" not in g.columns:
        return np.nan

    if value_col not in g.columns:
        return np.nan

    round_df = g.loc[g["round_in_episode"] == int(round_number)].copy()

    if round_df.empty:
        return np.nan

    values = pd.to_numeric(round_df[value_col], errors="coerce")

    if values.dropna().empty:
        return np.nan

    return float(values.mean(skipna=True))


# ============================================================
# LOAD ROUND SUMMARY DATA
# ============================================================

def load_round_summaries(
    data_root: Path,
    include_skipped: bool = False,
) -> pd.DataFrame:
    """Load and concatenate all participant round_summary.csv files."""
    files = find_round_summary_files(data_root)

    if not files:
        raise FileNotFoundError(f"No round_summary.csv files found under: {data_root}")

    frames = []

    required = {
        "episode_index",
        "episode_phase",
        "round_in_episode",
        "dishes_served",
        "human_steps",
        "policy_id",
        "ai_reward_score",
    }

    for csv_path in files:
        df = pd.read_csv(csv_path)
        df["participant_id"] = participant_id_from_file(csv_path, df)

        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        numeric_cols = [
            "episode_index",
            "round_in_episode",
            "dishes_served",
            "human_steps",
            "ai_steps",
            "team_reward_score",
            "human_reward_score",
            "ai_reward_score",
            "mental_demand",
            "performance_score",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = safe_numeric(df[col])

        if not include_skipped and "skipped_episode" in df.columns:
            skipped = (
                df["skipped_episode"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            df = df.loc[~skipped].copy()

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# LOAD POLICY EMBEDDING
# ============================================================

def infer_embedding_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Infer:
        - policy name column
        - x coordinate column
        - y coordinate column
    """
    name_candidates = [
        "policy",
        "policy_id",
        "policy_name",
        "name",
        "model_id",
        "checkpoint",
        "arm_name",
    ]

    name_col = next((c for c in name_candidates if c in df.columns), None)

    if name_col is None:
        non_numeric = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not non_numeric:
            raise ValueError(
                "Could not infer policy-name column in embedding CSV. "
                f"Columns found: {list(df.columns)}"
            )
        name_col = non_numeric[0]

    xy_candidates = [
        ("emb_x", "emb_y"),
        ("x", "y"),
        ("tsne_x", "tsne_y"),
        ("tsne_0", "tsne_1"),
        ("dim0", "dim1"),
        ("dim_1", "dim_2"),
        ("0", "1"),
    ]

    for x_col, y_col in xy_candidates:
        if x_col in df.columns and y_col in df.columns:
            return name_col, x_col, y_col

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            "Could not infer two numeric embedding columns in embedding CSV. "
            f"Columns found: {list(df.columns)}"
        )

    return name_col, numeric_cols[0], numeric_cols[1]


def load_embedding(
    embedding_csv: Path,
    policy_prefix: str | None,
) -> pd.DataFrame:
    """Load policy embedding CSV and normalize policy names."""
    embedding_csv = embedding_csv.expanduser().resolve()

    if not embedding_csv.exists():
        raise FileNotFoundError(f"Embedding CSV not found: {embedding_csv}")

    df = pd.read_csv(embedding_csv)
    name_col, x_col, y_col = infer_embedding_columns(df)

    out = df[[name_col, x_col, y_col]].copy()
    out = out.rename(
        columns={
            name_col: "policy_id_raw",
            x_col: "emb_x",
            y_col: "emb_y",
        }
    )

    out["policy_id_raw"] = out["policy_id_raw"].astype(str).str.strip()
    out["policy_short"] = out["policy_id_raw"].apply(
        lambda p: strip_policy_prefix(p, policy_prefix)
    )

    out["emb_x"] = safe_numeric(out["emb_x"])
    out["emb_y"] = safe_numeric(out["emb_y"])

    out = out.loc[
        out["policy_short"].apply(is_valid_policy)
        & out["emb_x"].notna()
        & out["emb_y"].notna()
    ].copy()

    out = out.drop_duplicates(subset=["policy_short"], keep="first").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid policies found in embedding CSV: {embedding_csv}")

    return out


# ============================================================
# EPISODE SUMMARY
# ============================================================

def summarize_episodes(
    rounds: pd.DataFrame,
    metric: str,
    policy_prefix: str | None,
    reward_round: int,
) -> pd.DataFrame:
    """
    Produce one row per participant and episode.

    Each episode corresponds to one played policy.

    For metric == ai_reward_score_round3, metric_value is the AI reward score
    from round_in_episode == reward_round, not the episode mean.
    """
    df = rounds.copy()

    df["policy_id"] = df["policy_id"].fillna("").astype(str)
    df["policy_short"] = df["policy_id"].apply(
        lambda p: strip_policy_prefix(p, policy_prefix)
    )

    df = df.loc[df["policy_short"].apply(is_valid_policy)].copy()

    if df.empty:
        raise ValueError("No valid policy_id values found in round_summary.csv files.")

    rows = []

    group_cols = [
        "participant_id",
        "episode_index",
        "episode_phase",
        "policy_id",
        "policy_short",
    ]

    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        participant_id, episode_index, episode_phase, policy_id, policy_short = keys

        total_dishes = float(g["dishes_served"].sum(skipna=True))
        total_human_steps = float(g["human_steps"].sum(skipna=True))

        if metric == "mean_dishes_per_round":
            metric_value = float(g["dishes_served"].mean(skipna=True))

        elif metric == "human_steps_per_dish":
            metric_value = (
                total_human_steps / total_dishes
                if total_dishes > 0
                else np.nan
            )

        elif metric == "ai_reward_score_round3":
            metric_value = get_round_value(
                g=g,
                value_col="ai_reward_score",
                round_number=reward_round,
            )

        elif metric in g.columns:
            # Generic fallback for numeric columns: episode mean.
            metric_value = float(g[metric].mean(skipna=True))

        else:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                "Use ai_reward_score_round3, mean_dishes_per_round, "
                "human_steps_per_dish, or a numeric column present in round_summary.csv."
            )

        row = {
            "participant_id": participant_id,
            "episode_index": int(episode_index) if pd.notna(episode_index) else np.nan,
            "episode_phase": episode_phase,
            "policy_id": policy_id,
            "policy_short": policy_short,
            "metric": metric,
            "reward_round_used": int(reward_round) if metric == "ai_reward_score_round3" else np.nan,
            "metric_value": metric_value,
            "total_dishes": total_dishes,
            "total_human_steps": total_human_steps,
            "n_rounds": int(len(g)),
        }

        # Keep these as reference episode means in the CSV.
        # Note: ai_reward_score here remains the episode mean, while metric_value
        # is round-3 AI reward when metric == ai_reward_score_round3.
        for optional_col in [
            "team_reward_score",
            "human_reward_score",
            "ai_reward_score",
            "mental_demand",
            "performance_score",
            "ai_steps",
        ]:
            if optional_col in g.columns:
                row[f"mean_{optional_col}"] = float(g[optional_col].mean(skipna=True))

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["metric_value"])
    out = out.sort_values(["participant_id", "episode_index"]).reset_index(drop=True)

    return out


# ============================================================
# PLOT DATA HELPERS
# ============================================================

def make_policy_point_summary(plot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple episodes using the same policy/coordinate into one dot.

    Example:
        If E13 and E16 use the same policy, the label becomes E13/E16.
    """
    rows = []

    group_cols = ["policy_short", "emb_x", "emb_y"]

    for keys, g in plot_df.groupby(group_cols, sort=False):
        policy_short, emb_x, emb_y = keys

        episodes = sorted(g["episode_index"].dropna().astype(int).unique().tolist())
        phases = sorted(g["episode_phase"].dropna().astype(str).unique().tolist())

        rows.append(
            {
                "policy_short": policy_short,
                "emb_x": float(emb_x),
                "emb_y": float(emb_y),
                "metric_value": float(g["metric_value"].mean(skipna=True)),
                "episode_indices": episodes,
                "episode_label": episode_label(episodes),
                "episode_phases": phases,
            }
        )

    return pd.DataFrame(rows)


def find_best_episode_row(
    plot_df: pd.DataFrame,
    best_episode_index: int,
) -> pd.Series | None:
    """Find the row corresponding to the requested Best-policy episode, usually E13."""
    rows = plot_df.loc[plot_df["episode_index"] == int(best_episode_index)].copy()

    if rows.empty:
        return None

    return rows.sort_values("episode_index").iloc[0]


def find_worst_episode_row(
    plot_df: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """Find the worst episode according to the selected metric."""
    higher_is_better = metric not in LOWER_IS_BETTER_METRICS

    if higher_is_better:
        worst_idx = plot_df["metric_value"].idxmin()
    else:
        worst_idx = plot_df["metric_value"].idxmax()

    return plot_df.loc[worst_idx]


def build_legend_handles(best_episode_index: int) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#D6D6D6",
            markeredgecolor="#D6D6D6",
            markersize=5.5,
            label="Policy pool",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#666666",
            markeredgewidth=0.9,
            markersize=6,
            label="Played policy",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#222222",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7,
            label=f"Episode {best_episode_index}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#B00020",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7,
            label="Worst episode",
        ),
    ]


# ============================================================
# PLOTTING
# ============================================================

def draw_metric_heatmap(
    ax,
    policy_points: pd.DataFrame,
    cmap,
):
    """
    Draw interpolated heatmap from observed policy points.

    This uses matplotlib tricontourf, so the color represents the metric value
    rather than just point density.
    """
    surface_df = policy_points.dropna(subset=["emb_x", "emb_y", "metric_value"]).copy()

    can_draw_surface = (
        len(surface_df) >= 3
        and surface_df["emb_x"].nunique() > 1
        and surface_df["emb_y"].nunique() > 1
        and surface_df["metric_value"].nunique() > 1
    )

    if not can_draw_surface:
        return None

    try:
        contour = ax.tricontourf(
            surface_df["emb_x"].to_numpy(dtype=float),
            surface_df["emb_y"].to_numpy(dtype=float),
            surface_df["metric_value"].to_numpy(dtype=float),
            levels=HEATMAP_LEVELS,
            cmap=cmap,
            alpha=0.92,
            zorder=2,
        )
        return contour

    except Exception as e:
        print(f"[WARN] Could not draw tricontourf heatmap: {e}")
        return None


def plot_participant_policy_pool_map(
    participant_episodes: pd.DataFrame,
    pool_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    formats: Iterable[str],
    best_episode_index: int,
    reward_round: int,
) -> None:
    """Create one policy-pool heatmap for one participant."""
    participant_id = str(participant_episodes["participant_id"].iloc[0])
    cmap = build_cubehelix_cmap()

    plot_df = participant_episodes.dropna(
        subset=["emb_x", "emb_y", "metric_value", "episode_index"]
    ).copy()

    if plot_df.empty:
        raise ValueError(f"No valid plotted episodes for {participant_id}")

    policy_points = make_policy_point_summary(plot_df)

    best_row = find_best_episode_row(
        plot_df=plot_df,
        best_episode_index=best_episode_index,
    )

    if best_row is None:
        print(
            f"[WARN] {participant_id}: episode {best_episode_index} not found. "
            f"No black E{best_episode_index} dot will be drawn for this participant."
        )

    worst_row = find_worst_episode_row(
        plot_df=plot_df,
        metric=metric,
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # --------------------------------------------------------
    # Background: full policy pool
    # --------------------------------------------------------
    ax.scatter(
        pool_df["emb_x"],
        pool_df["emb_y"],
        s=14,
        color="#D6D6D6",
        alpha=0.30,
        linewidth=0.0,
        zorder=1,
    )

    # --------------------------------------------------------
    # Heatmap: where the metric is high/low
    # --------------------------------------------------------
    mappable = draw_metric_heatmap(
        ax=ax,
        policy_points=policy_points,
        cmap=cmap,
    )

    # Fallback mappable for colorbar if contour cannot be drawn.
    if mappable is None:
        mappable = ax.scatter(
            policy_points["emb_x"],
            policy_points["emb_y"],
            c=policy_points["metric_value"],
            cmap=cmap,
            s=0.1,
            alpha=0.0,
            zorder=0,
        )

    # --------------------------------------------------------
    # Played policy dots
    # --------------------------------------------------------
    ax.scatter(
        policy_points["emb_x"],
        policy_points["emb_y"],
        s=54,
        facecolor="#FFFFFF",
        edgecolor="#666666",
        linewidth=0.9,
        alpha=0.95,
        zorder=4,
    )

    # Worst episode = red dot.
    ax.scatter(
        [worst_row["emb_x"]],
        [worst_row["emb_y"]],
        s=92,
        marker="o",
        color="#B00020",
        edgecolor="white",
        linewidth=1.0,
        zorder=8,
    )

    # Best-policy episode = black dot.
    if best_row is not None:
        # If the best-policy episode is also the worst episode, keep it black but give it a red edge.
        same_as_worst = (
            float(best_row["emb_x"]) == float(worst_row["emb_x"])
            and float(best_row["emb_y"]) == float(worst_row["emb_y"])
        )

        ax.scatter(
            [best_row["emb_x"]],
            [best_row["emb_y"]],
            s=96,
            marker="o",
            color="#222222",
            edgecolor="#B00020" if same_as_worst else "white",
            linewidth=1.4 if same_as_worst else 1.0,
            zorder=9,
        )

    # --------------------------------------------------------
    # Episode labels next to every played policy dot
    # --------------------------------------------------------
    for _, row in policy_points.iterrows():
        label = row["episode_label"]

        if not label:
            continue

        is_best_policy = int(best_episode_index) in row["episode_indices"]
        is_worst_policy = int(worst_row["episode_index"]) in row["episode_indices"]

        if is_best_policy:
            text_color = "#222222"
            dx, dy = 6, 6
            z = 10
        elif is_worst_policy:
            text_color = "#B00020"
            dx, dy = 6, -6
            z = 10
        else:
            text_color = "#333333"
            dx, dy = 5, 5
            z = 7

        ax.annotate(
            label,
            xy=(row["emb_x"], row["emb_y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=EPISODE_LABEL_FONT_SIZE,
            color=text_color,
            ha="left",
            va="bottom" if dy >= 0 else "top",
            zorder=z,
        )

    # --------------------------------------------------------
    # Axes and limits
    # --------------------------------------------------------
    x_pad = 0.05 * (pool_df["emb_x"].max() - pool_df["emb_x"].min())
    y_pad = 0.05 * (pool_df["emb_y"].max() - pool_df["emb_y"].min())

    ax.set_xlim(pool_df["emb_x"].min() - x_pad, pool_df["emb_x"].max() + x_pad)
    ax.set_ylim(pool_df["emb_y"].min() - y_pad, pool_df["emb_y"].max() + y_pad)

    ax.set_xlabel("Policy embedding x", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Policy embedding y", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(
        f"{participant_id} — {label_for_metric(metric, reward_round)}",
        fontsize=TITLE_FONT_SIZE,
        pad=8,
    )

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    ax.set_facecolor("#FFFFFF")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Colorbar and legend
    # --------------------------------------------------------
    cbar = fig.colorbar(mappable, ax=ax, pad=0.025)
    cbar.set_label(label_for_metric(metric, reward_round), fontsize=COLORBAR_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.legend(
        handles=build_legend_handles(best_episode_index),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.4,
        handletextpad=0.45,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_figure(fig, output_path, formats=formats, close=True)


def make_all_plots(
    episode_summary_with_coords: pd.DataFrame,
    pool_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    formats: list[str],
    best_episode_index: int,
    max_episode_index: int | None,
    reward_round: int,
) -> list[Path]:
    """Create one policy-pool heatmap per participant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []

    if max_episode_index is None:
        episode_suffix = "all_episodes"
    else:
        episode_suffix = f"until_E{max_episode_index}"

    round_suffix = f"round{reward_round}" if metric == "ai_reward_score_round3" else ""

    for participant_id, p_df in episode_summary_with_coords.groupby("participant_id", sort=True):
        safe_id = str(participant_id).replace("/", "_").replace("\\", "_")

        stem_parts = ["policy_pool_heatmap", safe_id, metric]
        if round_suffix:
            stem_parts.append(round_suffix)
        stem_parts.append(episode_suffix)
        stem = "_".join(stem_parts)

        plot_participant_policy_pool_map(
            participant_episodes=p_df.copy(),
            pool_df=pool_df,
            metric=metric,
            output_path=output_dir / stem,
            formats=formats,
            best_episode_index=best_episode_index,
            reward_round=reward_round,
        )

        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    return created


# ============================================================
# SUMMARY CSV
# ============================================================

def build_best_worst_summary(
    episode_summary_with_coords: pd.DataFrame,
    metric: str,
    best_episode_index: int,
    reward_round: int,
) -> pd.DataFrame:
    rows = []

    for participant_id, p_df in episode_summary_with_coords.groupby("participant_id", sort=True):
        p_df = p_df.dropna(subset=["metric_value", "emb_x", "emb_y"]).copy()

        if p_df.empty:
            continue

        best_row = find_best_episode_row(
            plot_df=p_df,
            best_episode_index=best_episode_index,
        )

        worst_row = find_worst_episode_row(
            plot_df=p_df,
            metric=metric,
        )

        out = {
            "participant_id": participant_id,
            "metric": metric,
            "reward_round_used": reward_round if metric == "ai_reward_score_round3" else np.nan,
            "best_episode_index_requested": best_episode_index,
            "worst_episode_index": worst_row["episode_index"],
            "worst_episode_phase": worst_row["episode_phase"],
            "worst_policy_id": worst_row["policy_id"],
            "worst_policy_short": worst_row["policy_short"],
            "worst_metric_value": worst_row["metric_value"],
            "worst_emb_x": worst_row["emb_x"],
            "worst_emb_y": worst_row["emb_y"],
        }

        if best_row is not None:
            out.update(
                {
                    "best_episode_found": True,
                    "best_episode_index": best_row["episode_index"],
                    "best_episode_phase": best_row["episode_phase"],
                    "best_policy_id": best_row["policy_id"],
                    "best_policy_short": best_row["policy_short"],
                    "best_metric_value": best_row["metric_value"],
                    "best_emb_x": best_row["emb_x"],
                    "best_emb_y": best_row["emb_y"],
                }
            )
        else:
            out.update(
                {
                    "best_episode_found": False,
                    "best_episode_index": np.nan,
                    "best_episode_phase": "",
                    "best_policy_id": "",
                    "best_policy_short": "",
                    "best_metric_value": np.nan,
                    "best_emb_x": np.nan,
                    "best_emb_y": np.nan,
                }
            )

        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create policy-pool heatmaps using AI reward score from round 3, "
            "with episode labels, Episode 13 highlighted, and worst episode highlighted."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("submissions"),
        help="Folder containing participant folders, each with round_summary.csv.",
    )

    parser.add_argument(
        "--embedding-csv",
        type=Path,
        default=Path("tsnt_thinpath.csv"),
        help="Policy embedding CSV, e.g. tsnt_thinpath.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/policy_pool_heatmap_ai_reward_round3_until_E16"),
        help="Where to save figures and summary CSVs.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help=(
            "Metric to visualize. Default: ai_reward_score_round3. "
            "This uses ai_reward_score from round_in_episode == 3."
        ),
    )

    parser.add_argument(
        "--reward-round",
        type=int,
        default=DEFAULT_REWARD_ROUND,
        help="Round within each episode to use for AI reward score. Default: 3.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf.",
    )

    parser.add_argument(
        "--policy-prefix",
        type=str,
        default=DEFAULT_POLICY_PREFIX,
        help="Policy prefix to strip when matching policy IDs.",
    )

    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include skipped episodes. By default they are excluded.",
    )

    parser.add_argument(
        "--best-episode-index",
        type=int,
        default=DEFAULT_BEST_EPISODE_INDEX,
        help="Episode to highlight in black. Default: 13.",
    )

    parser.add_argument(
        "--max-episode-index",
        type=int,
        default=DEFAULT_MAX_EPISODE_INDEX,
        help=(
            "Only include episodes up to and including this index. "
            "Default: 16, so the final episode is excluded if it is E17."
        ),
    )

    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.75,
        help="Font scale passed to plot_style.apply_plot_style(). Default: 0.75.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    apply_plot_style(font_scale=args.font_scale)
    apply_heatmap_text_style()

    rounds = load_round_summaries(
        data_root=args.data_root,
        include_skipped=args.include_skipped,
    )

    valid_synthetic_metrics = {
        "mean_dishes_per_round",
        "human_steps_per_dish",
        "ai_reward_score_round3",
    }

    if args.metric not in rounds.columns and args.metric not in valid_synthetic_metrics:
        raise ValueError(
            f"Metric '{args.metric}' is not available. "
            f"Columns found in round_summary data include: {sorted(rounds.columns.tolist())}"
        )

    if args.metric == "ai_reward_score_round3":
        required_for_round_metric = {"round_in_episode", "ai_reward_score"}
        missing_for_round_metric = sorted(required_for_round_metric - set(rounds.columns))
        if missing_for_round_metric:
            raise ValueError(
                "ai_reward_score_round3 requires these columns: "
                f"{missing_for_round_metric}"
            )

    pool_df = load_embedding(
        embedding_csv=args.embedding_csv,
        policy_prefix=args.policy_prefix,
    )

    episode_summary = summarize_episodes(
        rounds=rounds,
        metric=args.metric,
        policy_prefix=args.policy_prefix,
        reward_round=args.reward_round,
    )

    episode_summary = filter_episodes_up_to(
        episode_summary=episode_summary,
        max_episode_index=args.max_episode_index,
    )

    merged = episode_summary.merge(
        pool_df[["policy_short", "emb_x", "emb_y"]],
        how="left",
        on="policy_short",
        validate="many_to_one",
    )

    missing_coords = merged["emb_x"].isna() | merged["emb_y"].isna()

    if missing_coords.any():
        missing_policies = sorted(
            merged.loc[missing_coords, "policy_short"]
            .astype(str)
            .unique()
            .tolist()
        )

        print(
            "[WARN] Some played policies were not found in the embedding CSV. "
            "They will be skipped."
        )
        print(f"[WARN] Missing policy_short values: {missing_policies}")

        merged = merged.loc[~missing_coords].copy()

    if merged.empty:
        raise ValueError("No participant episodes could be matched to embedding coordinates.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_episode_index is None:
        episode_suffix = "all_episodes"
    else:
        episode_suffix = f"until_E{args.max_episode_index}"

    round_suffix = f"round{args.reward_round}" if args.metric == "ai_reward_score_round3" else ""

    csv_stem_parts = [args.metric]
    if round_suffix:
        csv_stem_parts.append(round_suffix)
    csv_stem_parts.append(episode_suffix)
    csv_suffix = "_".join(csv_stem_parts)

    episode_csv = args.output_dir / f"episode_policy_pool_summary_{csv_suffix}.csv"
    best_worst_csv = args.output_dir / f"episode{args.best_episode_index}_and_worst_summary_{csv_suffix}.csv"

    merged.to_csv(episode_csv, index=False)

    best_worst_summary = build_best_worst_summary(
        episode_summary_with_coords=merged,
        metric=args.metric,
        best_episode_index=args.best_episode_index,
        reward_round=args.reward_round,
    )
    best_worst_summary.to_csv(best_worst_csv, index=False)

    created = make_all_plots(
        episode_summary_with_coords=merged,
        pool_df=pool_df,
        metric=args.metric,
        output_dir=args.output_dir,
        formats=list(args.formats),
        best_episode_index=args.best_episode_index,
        max_episode_index=args.max_episode_index,
        reward_round=args.reward_round,
    )

    print(f"Loaded participants: {merged['participant_id'].nunique()}")
    print(f"Embedding policies: {pool_df['policy_short'].nunique()}")
    print(f"Metric: {args.metric}")
    if args.metric == "ai_reward_score_round3":
        print(f"AI reward source: round_in_episode == {args.reward_round}")
    print(f"Included episodes: E1 to E{args.max_episode_index}")
    print(f"Saved episode summary: {episode_csv}")
    print(f"Saved E{args.best_episode_index}/Worst summary: {best_worst_csv}")

    for path in created:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
