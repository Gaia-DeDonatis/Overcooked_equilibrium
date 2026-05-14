#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy-pool heatmap with episode labels.

For each participant, this script creates a policy-space map where:

    - grey dots = full policy pool
    - white dots = policies actually played by the participant
    - text next to each white dot = episode(s) where that policy was used
    - black dot = episode 13
    - red dot = worst episode for the selected metric
    - heatmap background = interpolated metric value in policy space

Important:
    The black dot is ALWAYS episode 13 by default.
    It is not selected using the maximum score.

Example
-------
python analyze_policy_pool_episode_heatmap.py ^
    --data-root submissions ^
    --embedding-csv tsnt_thinpath.csv ^
    --output-dir figures/policy_pool_episode_heatmap_dishes ^
    --metric mean_dishes_per_round ^
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
DEFAULT_BEST_EPISODE_INDEX = 13

LOWER_IS_BETTER_METRICS = {
    "human_steps_per_dish",
    "mental_demand",
}

# Cubehelix palette settings.
CUBEHELIX_START = 2.4
CUBEHELIX_ROT = -0.25
CUBEHELIX_DARK = 0.18
CUBEHELIX_LIGHT = 0.96

# Heatmap interpolation settings.
GRID_SIZE = 220
IDW_POWER = 2.0
HEATMAP_LEVELS = 18
HEATMAP_ALPHA = 0.92
MAX_HEATMAP_DISTANCE_FRACTION = 0.42

# Smaller font sizes.
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


def make_episode_label(episodes: Iterable[int | float]) -> str:
    cleaned = []

    for ep in episodes:
        if pd.notna(ep):
            cleaned.append(int(ep))

    cleaned = sorted(set(cleaned))

    if not cleaned:
        return ""

    return "/".join(f"E{ep}" for ep in cleaned)


# ============================================================
# LOAD ROUND SUMMARY DATA
# ============================================================

def load_round_summaries(
    data_root: Path,
    include_skipped: bool = False,
) -> pd.DataFrame:
    files = find_round_summary_files(data_root)

    if not files:
        raise FileNotFoundError(f"No round_summary.csv files found under: {data_root}")

    frames = []

    required = {
        "episode_index",
        "episode_phase",
        "dishes_served",
        "human_steps",
        "policy_id",
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

    out = out.drop_duplicates(
        subset=["policy_short"],
        keep="first",
    ).reset_index(drop=True)

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
) -> pd.DataFrame:
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

        elif metric in g.columns:
            metric_value = float(g[metric].mean(skipna=True))

        else:
            raise ValueError(
                f"Unsupported metric: {metric}. "
                "Use mean_dishes_per_round, human_steps_per_dish, "
                "or a numeric column present in round_summary.csv."
            )

        rows.append(
            {
                "participant_id": participant_id,
                "episode_index": int(episode_index) if pd.notna(episode_index) else np.nan,
                "episode_phase": episode_phase,
                "policy_id": policy_id,
                "policy_short": policy_short,
                "metric_value": metric_value,
                "total_dishes": total_dishes,
                "total_human_steps": total_human_steps,
                "n_rounds": int(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["metric_value"])
    out = out.sort_values(["participant_id", "episode_index"]).reset_index(drop=True)

    return out


# ============================================================
# PLOT DATA
# ============================================================

def make_policy_point_summary(plot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse same-policy repeated episodes into one dot.

    Example:
        E13 and E17 use the same policy -> label becomes E13/E17.
    """
    rows = []

    group_cols = ["policy_short", "emb_x", "emb_y"]

    for keys, g in plot_df.groupby(group_cols, sort=False):
        policy_short, emb_x, emb_y = keys

        episodes = sorted(
            g["episode_index"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        rows.append(
            {
                "policy_short": policy_short,
                "emb_x": float(emb_x),
                "emb_y": float(emb_y),
                "metric_value": float(g["metric_value"].mean(skipna=True)),
                "episode_indices": episodes,
                "episode_label": make_episode_label(episodes),
            }
        )

    return pd.DataFrame(rows)


def find_episode_row(
    plot_df: pd.DataFrame,
    episode_index: int,
):
    rows = plot_df.loc[plot_df["episode_index"] == int(episode_index)].copy()

    if rows.empty:
        return None

    return rows.sort_values("episode_index").iloc[0]


def find_worst_episode_row(
    plot_df: pd.DataFrame,
    metric: str,
):
    higher_is_better = metric not in LOWER_IS_BETTER_METRICS

    if higher_is_better:
        worst_idx = plot_df["metric_value"].idxmin()
    else:
        worst_idx = plot_df["metric_value"].idxmax()

    return plot_df.loc[worst_idx]


# ============================================================
# HEATMAP INTERPOLATION
# ============================================================

def compute_idw_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = GRID_SIZE,
    power: float = IDW_POWER,
):
    """
    Create an inverse-distance weighted surface.

    This means the heatmap represents an interpolation of the observed
    metric values from played policies.
    """
    xi = np.linspace(xlim[0], xlim[1], grid_size)
    yi = np.linspace(ylim[0], ylim[1], grid_size)

    X, Y = np.meshgrid(xi, yi)

    dx = X[None, :, :] - x[:, None, None]
    dy = Y[None, :, :] - y[:, None, None]

    dist = np.sqrt(dx * dx + dy * dy)

    eps = 1e-9
    weights = 1.0 / np.power(dist + eps, power)

    Z = np.sum(weights * z[:, None, None], axis=0) / np.sum(weights, axis=0)

    nearest_dist = np.min(dist, axis=0)

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    max_dist = MAX_HEATMAP_DISTANCE_FRACTION * max(x_range, y_range)

    Z = np.ma.masked_where(nearest_dist > max_dist, Z)

    return X, Y, Z


def draw_metric_heatmap(
    ax,
    policy_points: pd.DataFrame,
    pool_df: pd.DataFrame,
    cmap,
):
    surface_df = policy_points.dropna(
        subset=["emb_x", "emb_y", "metric_value"]
    ).copy()

    if len(surface_df) < 2:
        return None

    x_pad = 0.05 * (pool_df["emb_x"].max() - pool_df["emb_x"].min())
    y_pad = 0.05 * (pool_df["emb_y"].max() - pool_df["emb_y"].min())

    xlim = (
        float(pool_df["emb_x"].min() - x_pad),
        float(pool_df["emb_x"].max() + x_pad),
    )
    ylim = (
        float(pool_df["emb_y"].min() - y_pad),
        float(pool_df["emb_y"].max() + y_pad),
    )

    x = surface_df["emb_x"].to_numpy(dtype=float)
    y = surface_df["emb_y"].to_numpy(dtype=float)
    z = surface_df["metric_value"].to_numpy(dtype=float)

    if np.nanmin(z) == np.nanmax(z):
        return None

    X, Y, Z = compute_idw_surface(
        x=x,
        y=y,
        z=z,
        xlim=xlim,
        ylim=ylim,
    )

    levels = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), HEATMAP_LEVELS)

    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        alpha=HEATMAP_ALPHA,
        zorder=2,
    )

    return contour


# ============================================================
# LEGEND
# ============================================================

def build_legend_handles() -> list[Line2D]:
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
            label="E13",
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

def plot_participant_policy_pool_map(
    participant_episodes: pd.DataFrame,
    pool_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    formats: Iterable[str],
    best_episode_index: int,
) -> None:
    participant_id = str(participant_episodes["participant_id"].iloc[0])
    cmap = build_cubehelix_cmap()

    plot_df = participant_episodes.dropna(
        subset=["emb_x", "emb_y", "metric_value", "episode_index"]
    ).copy()

    if plot_df.empty:
        raise ValueError(f"No valid plotted episodes for {participant_id}")

    policy_points = make_policy_point_summary(plot_df)

    best_row = find_episode_row(
        plot_df=plot_df,
        episode_index=best_episode_index,
    )

    worst_row = find_worst_episode_row(
        plot_df=plot_df,
        metric=metric,
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # Full policy pool.
    ax.scatter(
        pool_df["emb_x"],
        pool_df["emb_y"],
        s=14,
        color="#D6D6D6",
        alpha=0.30,
        linewidth=0.0,
        zorder=1,
    )

    # Heatmap = interpolated metric values.
    mappable = draw_metric_heatmap(
        ax=ax,
        policy_points=policy_points,
        pool_df=pool_df,
        cmap=cmap,
    )

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

    # Played policies: white dots.
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

    # Worst episode: red dot.
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

    # Episode 13: black dot.
    if best_row is not None:
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
    else:
        print(f"[WARN] {participant_id}: E{best_episode_index} not found.")

    # Episode labels next to every played policy dot.
    for _, row in policy_points.iterrows():
        label = row["episode_label"]

        if not label:
            continue

        episodes = set(row["episode_indices"])

        contains_e13 = int(best_episode_index) in episodes
        contains_worst = int(worst_row["episode_index"]) in episodes

        if contains_e13:
            text_color = "#222222"
            dx, dy = 6, 6
            zorder = 10
        elif contains_worst:
            text_color = "#B00020"
            dx, dy = 6, -6
            zorder = 10
        else:
            text_color = "#333333"
            dx, dy = 5, 5
            zorder = 7

        ax.annotate(
            label,
            xy=(row["emb_x"], row["emb_y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=EPISODE_LABEL_FONT_SIZE,
            color=text_color,
            ha="left",
            va="bottom" if dy >= 0 else "top",
            zorder=zorder,
        )

    # Limits.
    x_pad = 0.05 * (pool_df["emb_x"].max() - pool_df["emb_x"].min())
    y_pad = 0.05 * (pool_df["emb_y"].max() - pool_df["emb_y"].min())

    ax.set_xlim(pool_df["emb_x"].min() - x_pad, pool_df["emb_x"].max() + x_pad)
    ax.set_ylim(pool_df["emb_y"].min() - y_pad, pool_df["emb_y"].max() + y_pad)

    ax.set_xlabel("Policy embedding x", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Policy embedding y", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(
        f"{participant_id} — {metric_label(metric)}",
        fontsize=TITLE_FONT_SIZE,
        pad=8,
    )

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.set_facecolor("#FFFFFF")
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(mappable, ax=ax, pad=0.025)
    cbar.set_label(metric_label(metric), fontsize=COLORBAR_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.legend(
        handles=build_legend_handles(),
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
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []

    for participant_id, p_df in episode_summary_with_coords.groupby("participant_id", sort=True):
        safe_id = str(participant_id).replace("/", "_").replace("\\", "_")
        stem = f"policy_pool_episode_heatmap_{safe_id}_{metric}"

        plot_participant_policy_pool_map(
            participant_episodes=p_df.copy(),
            pool_df=pool_df,
            metric=metric,
            output_path=output_dir / stem,
            formats=formats,
            best_episode_index=best_episode_index,
        )

        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    return created


# ============================================================
# SUMMARY CSV
# ============================================================

def build_episode13_and_worst_summary(
    episode_summary_with_coords: pd.DataFrame,
    metric: str,
    best_episode_index: int,
) -> pd.DataFrame:
    rows = []

    for participant_id, p_df in episode_summary_with_coords.groupby("participant_id", sort=True):
        p_df = p_df.dropna(subset=["metric_value", "emb_x", "emb_y"]).copy()

        if p_df.empty:
            continue

        best_row = find_episode_row(
            plot_df=p_df,
            episode_index=best_episode_index,
        )

        worst_row = find_worst_episode_row(
            plot_df=p_df,
            metric=metric,
        )

        out = {
            "participant_id": participant_id,
            "highlight_episode_index": best_episode_index,
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
                    "highlight_episode_found": True,
                    "highlight_episode_phase": best_row["episode_phase"],
                    "highlight_policy_id": best_row["policy_id"],
                    "highlight_policy_short": best_row["policy_short"],
                    "highlight_metric_value": best_row["metric_value"],
                    "highlight_emb_x": best_row["emb_x"],
                    "highlight_emb_y": best_row["emb_y"],
                }
            )
        else:
            out.update(
                {
                    "highlight_episode_found": False,
                    "highlight_episode_phase": "",
                    "highlight_policy_id": "",
                    "highlight_policy_short": "",
                    "highlight_metric_value": np.nan,
                    "highlight_emb_x": np.nan,
                    "highlight_emb_y": np.nan,
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
            "Create policy-pool episode heatmaps with all episode labels, "
            "E13 highlighted in black, and worst episode highlighted in red."
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
        default=Path("figures/policy_pool_episode_heatmap"),
        help="Where to save figures and summary CSVs.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mean_dishes_per_round",
        help=(
            "Metric to visualize. Examples: mean_dishes_per_round, "
            "human_steps_per_dish, team_reward_score, mental_demand."
        ),
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

    pool_df = load_embedding(
        embedding_csv=args.embedding_csv,
        policy_prefix=args.policy_prefix,
    )

    episode_summary = summarize_episodes(
        rounds=rounds,
        metric=args.metric,
        policy_prefix=args.policy_prefix,
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

    episode_csv = args.output_dir / f"episode_policy_pool_summary_{args.metric}.csv"
    e13_worst_csv = args.output_dir / f"episode13_and_worst_summary_{args.metric}.csv"

    merged.to_csv(episode_csv, index=False)

    e13_worst_summary = build_episode13_and_worst_summary(
        episode_summary_with_coords=merged,
        metric=args.metric,
        best_episode_index=args.best_episode_index,
    )
    e13_worst_summary.to_csv(e13_worst_csv, index=False)

    created = make_all_plots(
        episode_summary_with_coords=merged,
        pool_df=pool_df,
        metric=args.metric,
        output_dir=args.output_dir,
        formats=list(args.formats),
        best_episode_index=args.best_episode_index,
    )

    print(f"Loaded participants: {merged['participant_id'].nunique()}")
    print(f"Embedding policies: {pool_df['policy_short'].nunique()}")
    print(f"Saved episode summary: {episode_csv}")
    print(f"Saved E13/Worst summary: {e13_worst_csv}")

    for path in created:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()