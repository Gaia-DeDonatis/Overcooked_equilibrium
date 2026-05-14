#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph 2: Task-division spider/radar plots.

For each participant, this script creates one figure with two radar plots:

    left  -> Best-policy episode, usually episode 13
    right -> Replay changed human behavior episode, usually episode 17

Each radar plot contains two lines:

    Human
    AI

The radial value is the share of that sub-task performed by each agent:

    human_share = human_count / (human_count + ai_count)
    ai_share    = ai_count / (human_count + ai_count)

Interpretation:
    1.0 = that agent performed all of that sub-task
    0.5 = task was equally shared
    0.0 = that agent did not perform that sub-task

Example
-------
    python analyze_task_division_spider_best_vs_replay.py \
        --data-root submissions \
        --output-dir figures/task_division_spider \
        --formats png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (
    apply_plot_style,
    save_figure,
    COLORS,
    TASK_GROUPS,
    task_label,
    ALPHA_FILL,
    LINE_WIDTH,
    MARKER_SIZE,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    SEABORN_AXES_FACE,
    SEABORN_GRID_COLOR,
    GRID_LINE_WIDTH,
)


BEST_PHASE = "bo_replay_best"
REPLAY_PHASE = "replay_optimal"
PHASE_ORDER = [BEST_PHASE, REPLAY_PHASE]

PHASE_DISPLAY_LABELS = {
    BEST_PHASE: "Best-policy",
    REPLAY_PHASE: "Replay changed behavior",
}


def find_task_division_files(data_root: Path) -> list[Path]:
    """Find task_division_summary.csv files under a submissions-like root."""
    data_root = data_root.expanduser().resolve()

    if data_root.is_file() and data_root.name == "task_division_summary.csv":
        return [data_root]

    if (data_root / "task_division_summary.csv").exists():
        return [data_root / "task_division_summary.csv"]

    files = sorted(data_root.glob("*/task_division_summary.csv"))
    if not files:
        files = sorted(data_root.rglob("task_division_summary.csv"))

    return files


def participant_id_from_file(csv_path: Path, df: pd.DataFrame) -> str:
    """Prefer folder name, because it is usually cleaner than prolific_id."""
    folder_name = csv_path.parent.name
    if folder_name:
        return folder_name

    if "prolific_id" in df.columns and df["prolific_id"].notna().any():
        return str(df["prolific_id"].dropna().iloc[0])

    return "unknown_participant"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def load_task_division(data_root: Path) -> pd.DataFrame:
    """Load and concatenate all participant task_division_summary.csv files."""
    files = find_task_division_files(data_root)
    if not files:
        raise FileNotFoundError(
            f"No task_division_summary.csv files found under: {data_root}"
        )

    frames = []

    required_base_cols = {
        "episode_index",
        "episode_phase",
        "agent",
    }

    required_task_cols = {
        col
        for cols in TASK_GROUPS.values()
        for col in cols
    }

    for csv_path in files:
        df = pd.read_csv(csv_path)
        df["participant_id"] = participant_id_from_file(csv_path, df)

        missing_base = sorted(required_base_cols - set(df.columns))
        if missing_base:
            raise ValueError(f"{csv_path} is missing required columns: {missing_base}")

        missing_task_cols = sorted(required_task_cols - set(df.columns))
        if missing_task_cols:
            raise ValueError(f"{csv_path} is missing task columns: {missing_task_cols}")

        df["episode_index"] = pd.to_numeric(df["episode_index"], errors="coerce")
        df["agent"] = df["agent"].astype(str).str.lower().str.strip()
        df["episode_phase"] = df["episode_phase"].astype(str).str.strip()

        for col in required_task_cols:
            df[col] = safe_numeric(df[col])

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def summarize_task_shares(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one row per participant, phase, and task, with:
        - human_count
        - ai_count
        - human_share
        - ai_share

    Shares are in [0, 1].
    """
    target = task_df.loc[task_df["episode_phase"].isin(PHASE_ORDER)].copy()

    if target.empty:
        raise ValueError(
            "No rows found for episode_phase in "
            f"{PHASE_ORDER}. Check task_division_summary.csv."
        )

    rows = []

    for participant_id, p_df in target.groupby("participant_id", sort=True):
        for phase in PHASE_ORDER:
            phase_df = p_df.loc[p_df["episode_phase"] == phase].copy()

            if phase_df.empty:
                continue

            human_df = phase_df.loc[phase_df["agent"] == "human"]
            ai_df = phase_df.loc[phase_df["agent"] == "ai"]

            for task_group, task_cols in TASK_GROUPS.items():
                human_count = float(human_df[task_cols].sum().sum()) if not human_df.empty else 0.0
                ai_count = float(ai_df[task_cols].sum().sum()) if not ai_df.empty else 0.0
                total_count = human_count + ai_count

                if total_count > 0:
                    human_share = human_count / total_count
                    ai_share = ai_count / total_count
                else:
                    human_share = 0.0
                    ai_share = 0.0

                rows.append(
                    {
                        "participant_id": participant_id,
                        "episode_phase": phase,
                        "condition": PHASE_DISPLAY_LABELS.get(phase, phase),
                        "task_group": task_group,
                        "task_label": task_label(task_group).replace("\n", " "),
                        "human_count": human_count,
                        "ai_count": ai_count,
                        "total_count": total_count,
                        "human_share": human_share,
                        "ai_share": ai_share,
                    }
                )

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError("Task-share summary is empty.")

    return summary.sort_values(
        ["participant_id", "episode_phase", "task_group"]
    ).reset_index(drop=True)


def close_loop(values: np.ndarray) -> np.ndarray:
    """Repeat the first value at the end so radar lines close."""
    return np.concatenate([values, values[:1]])


def make_angles(n_axes: int) -> np.ndarray:
    """Return polar angles and close the loop."""
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    return close_loop(angles)


def style_radar_axis(ax, task_groups: list[str], angles: np.ndarray) -> None:
    """Apply shared styling to the radar axis."""
    label_angles = angles[:-1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(label_angles)
    ax.set_xticklabels(
        [task_label(t) for t in task_groups],
        fontsize=9,
    )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=8)
    ax.set_rlabel_position(90)

    ax.grid(
        True,
        color=SEABORN_GRID_COLOR,
        linewidth=GRID_LINE_WIDTH,
        alpha=1.0,
    )

    ax.spines["polar"].set_color("#DDDDDD")
    ax.spines["polar"].set_linewidth(0.9)
    ax.set_facecolor(SEABORN_AXES_FACE)


def plot_participant_spider(
    summary: pd.DataFrame,
    participant_id: str,
    output_path: Path,
    formats: Iterable[str],
) -> None:
    """
    Create one figure per participant with 2 radar plots:
        - left: Best-policy
        - right: Replay changed behavior

    Each subplot has:
        - Human line
        - AI line
    """
    p_df = summary.loc[summary["participant_id"] == participant_id].copy()

    task_groups = list(TASK_GROUPS.keys())
    n_axes = len(task_groups)
    angles = make_angles(n_axes)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 6.2),
        subplot_kw={"projection": "polar"},
    )

    phase_titles = {
        BEST_PHASE: "Best-policy (ep 13)",
        REPLAY_PHASE: "Replay changed behavior (ep 17)",
    }

    for ax, phase in zip(axes, PHASE_ORDER):
        phase_df = p_df.loc[p_df["episode_phase"] == phase].copy()
        style_radar_axis(ax, task_groups=task_groups, angles=angles)

        human_values_by_task = {
            row["task_group"]: row["human_share"]
            for _, row in phase_df.iterrows()
        }
        ai_values_by_task = {
            row["task_group"]: row["ai_share"]
            for _, row in phase_df.iterrows()
        }

        human_values = np.array(
            [human_values_by_task.get(task, 0.0) for task in task_groups],
            dtype=float,
        )
        ai_values = np.array(
            [ai_values_by_task.get(task, 0.0) for task in task_groups],
            dtype=float,
        )

        human_values = close_loop(human_values)
        ai_values = close_loop(ai_values)

        # Human
        ax.plot(
            angles,
            human_values,
            color=COLORS["human"],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Human",
        )
        ax.fill(
            angles,
            human_values,
            color=COLORS["human"],
            alpha=ALPHA_FILL,
        )

        # AI
        ax.plot(
            angles,
            ai_values,
            color=COLORS["ai"],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="AI",
        )
        ax.fill(
            angles,
            ai_values,
            color=COLORS["ai"],
            alpha=ALPHA_FILL,
        )

        ax.set_title(phase_titles.get(phase, phase), pad=22, fontsize=12)

    fig.suptitle(participant_id, fontsize=16, y=1.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    save_figure(fig, output_path, formats=formats, close=True)


def make_all_plots(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    """Create one spider plot per participant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    participants = sorted(summary["participant_id"].dropna().unique())

    for participant_id in participants:
        safe_id = str(participant_id).replace("/", "_").replace("\\", "_")
        stem = f"task_division_spider_{safe_id}"

        plot_participant_spider(
            summary=summary,
            participant_id=participant_id,
            output_path=output_dir / stem,
            formats=formats,
        )

        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Graph 2: task-division spider plots comparing "
            "Best-policy vs Replay changed human behavior."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("submissions"),
        help="Folder containing participant folders, each with task_division_summary.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/task_division_spider_best_vs_replay"),
        help="Where to save figures and the summary CSV.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()

    task_df = load_task_division(args.data_root)
    summary = summarize_task_shares(task_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "task_division_shares_best_vs_replay_summary.csv"
    summary.to_csv(summary_csv, index=False)

    created = make_all_plots(
        summary=summary,
        output_dir=args.output_dir,
        formats=list(args.formats),
    )

    print(f"Loaded participants: {summary['participant_id'].nunique()}")
    print(f"Saved summary: {summary_csv}")

    for path in created:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()