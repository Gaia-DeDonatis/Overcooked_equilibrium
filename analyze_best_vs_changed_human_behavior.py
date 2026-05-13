#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph 1: Best-policy vs replay with changed human behavior.

This script reads participant `round_summary.csv` files and creates paired
plots comparing:

    bo_replay_best  -> Best-policy episode, usually episode 13
    replay_optimal  -> Replay with changed human behavior, usually episode 17

The script is intentionally independent from the game backend. It only needs
saved participant folders such as:

    submissions/
        Thinpath_P01/
            round_summary.csv
        Thinpath_P02/
            round_summary.csv
        ...

Outputs
-------
    paired_best_vs_replay_dishes_per_round.png
    paired_best_vs_replay_human_steps_per_dish.png
    paired_best_vs_replay_mental_demand.png
    paired_best_vs_replay_subjective_performance.png
    best_vs_changed_human_behavior_summary.csv

Example
-------
    python analyze_best_vs_changed_human_behavior.py \
        --data-root submissions \
        --output-dir figures/best_vs_changed_human_behavior

If your data is inside a zip file, extract it first, or pass the extracted root.
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
    clean_axis,
    save_figure,
    get_participant_color,
    metric_label,
    condition_label,
    COLORS,
    FIGSIZE_WIDE,
    ALPHA_LINE,
    LINE_MARKER_KWS,
)


BEST_PHASE = "bo_replay_best"
REPLAY_PHASE = "replay_optimal"
PHASE_ORDER = [BEST_PHASE, REPLAY_PHASE]

# Output name, internal metric column, ylabel-friendly label source.
PLOT_SPECS = [
    ("paired_best_vs_replay_dishes_per_round", "mean_dishes_per_round"),
    ("paired_best_vs_replay_human_steps_per_dish", "human_steps_per_dish"),
    ("paired_best_vs_replay_mental_demand", "mental_demand"),
    ("paired_best_vs_replay_subjective_performance", "performance_score"),
]


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


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _participant_id_from_file(csv_path: Path, df: pd.DataFrame) -> str:
    """Prefer folder name, because it is usually cleaner than free-text IDs."""
    folder_name = csv_path.parent.name
    if folder_name:
        return folder_name

    if "prolific_id" in df.columns and df["prolific_id"].notna().any():
        return str(df["prolific_id"].dropna().iloc[0])

    return "unknown_participant"


def load_round_summaries(data_root: Path, include_skipped: bool = False) -> pd.DataFrame:
    """Load and concatenate all participant round_summary.csv files."""
    files = find_round_summary_files(data_root)
    if not files:
        raise FileNotFoundError(f"No round_summary.csv files found under: {data_root}")

    frames = []
    for csv_path in files:
        df = pd.read_csv(csv_path)
        df["participant_id"] = _participant_id_from_file(csv_path, df)

        required = {
            "episode_index",
            "episode_phase",
            "dishes_served",
            "human_steps",
            "mental_demand",
            "performance_score",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        # Normalize numeric columns used in this script.
        for col in [
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
        ]:
            if col in df.columns:
                df[col] = _safe_numeric(df[col])

        if not include_skipped and "skipped_episode" in df.columns:
            # The column may be saved as bools or strings.
            skipped = df["skipped_episode"].astype(str).str.lower().isin(["true", "1", "yes"])
            df = df.loc[~skipped].copy()

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def summarize_best_vs_replay(rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one row per participant and phase.

    human_steps_per_dish is computed as total human steps / total dishes,
    because that reflects effort per successful dish across the whole episode.
    """
    target = rounds.loc[rounds["episode_phase"].isin(PHASE_ORDER)].copy()

    if target.empty:
        raise ValueError(
            "No rows found for episode_phase in "
            f"{PHASE_ORDER}. Check the round_summary.csv files."
        )

    rows = []
    group_cols = ["participant_id", "episode_phase"]
    for (participant_id, phase), g in target.groupby(group_cols, sort=False):
        total_dishes = float(g["dishes_served"].sum(skipna=True))
        total_human_steps = float(g["human_steps"].sum(skipna=True))
        n_rounds = int(len(g))

        row = {
            "participant_id": participant_id,
            "episode_phase": phase,
            "condition": condition_label(phase),
            "episode_index": int(g["episode_index"].dropna().mode().iloc[0])
            if g["episode_index"].notna().any()
            else np.nan,
            "n_rounds": n_rounds,
            "total_dishes": total_dishes,
            "mean_dishes_per_round": float(g["dishes_served"].mean(skipna=True)),
            "total_human_steps": total_human_steps,
            "human_steps_per_dish": (total_human_steps / total_dishes)
            if total_dishes > 0
            else np.nan,
            "mental_demand": float(g["mental_demand"].mean(skipna=True)),
            "performance_score": float(g["performance_score"].mean(skipna=True)),
        }

        for optional_col in [
            "team_reward_score",
            "human_reward_score",
            "ai_reward_score",
            "ai_steps",
        ]:
            if optional_col in g.columns:
                row[optional_col] = float(g[optional_col].mean(skipna=True))

        if "policy_id" in g.columns and g["policy_id"].notna().any():
            row["policy_id"] = str(g["policy_id"].dropna().mode().iloc[0])

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Add paired differences for convenience.
    for metric in ["mean_dishes_per_round", "human_steps_per_dish", "mental_demand", "performance_score"]:
        wide = summary.pivot(index="participant_id", columns="episode_phase", values=metric)
        if BEST_PHASE in wide.columns and REPLAY_PHASE in wide.columns:
            diff = wide[REPLAY_PHASE] - wide[BEST_PHASE]
            summary[f"delta_replay_minus_best_{metric}"] = summary["participant_id"].map(diff)

    return summary.sort_values(["participant_id", "episode_phase"]).reset_index(drop=True)


def plot_paired_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    formats: Iterable[str],
) -> None:
    """Create one paired participant plot for a given metric."""
    plot_df = summary.loc[summary["episode_phase"].isin(PHASE_ORDER)].copy()
    wide = plot_df.pivot(index="participant_id", columns="episode_phase", values=metric)
    wide = wide.reindex(columns=PHASE_ORDER)

    if wide.dropna(how="all").empty:
        raise ValueError(f"No valid values available for metric: {metric}")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(PHASE_ORDER))

    # Individual participants.
    for i, (participant_id, row) in enumerate(wide.iterrows()):
        y = row.to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            continue

        color = get_participant_color(participant_id, fallback_index=i)
        ax.plot(
            x,
            y,
            color=color,
            alpha=ALPHA_LINE,
            label="_nolegend_",
            **LINE_MARKER_KWS,
        )

        # Small participant label at the replay side, useful with n <= 10.
        if not np.isnan(y[-1]):
            ax.text(
                x[-1] + 0.035,
                y[-1],
                participant_id.replace("Thinpath_", ""),
                va="center",
                ha="left",
                fontsize=8.5,
                color=color,
            )

    # Group mean overlay.
    means = wide.mean(axis=0, skipna=True).to_numpy(dtype=float)
    ax.plot(
        x,
        means,
        color=COLORS["group_mean"],
        label="Group mean",
        zorder=10,
        **LINE_MARKER_KWS,
    )



    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(p) for p in PHASE_ORDER])
    ax.set_xlim(-0.18, len(PHASE_ORDER) - 1 + 0.45)
    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel("")
    ax.set_title(metric_label(metric))

    # For effort and subjective workload, lower is better. Add unobtrusive cue.
    if metric in {"human_steps_per_dish", "mental_demand"}:
        ax.text(
            0.01,
            0.98,
            "lower is better",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=COLORS["text"],
        )

    ax.margins(x=0.05, y=0.08)
    clean_axis(ax, grid_axis="y")

    # Keep the legend compact. Participant labels are already written near the lines.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    save_figure(fig, output_path, formats=formats, close=True)


def make_all_plots(summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    for stem, metric in PLOT_SPECS:
        plot_paired_metric(
            summary=summary,
            metric=metric,
            output_path=output_dir / stem,
            formats=formats,
        )
        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Graph 1: Best-policy vs replay with changed human behavior."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("submissions"),
        help="Folder containing participant folders, each with round_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/best_vs_changed_human_behavior"),
        help="Where to save figures and the summary CSV.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include rows marked as skipped_episode. By default they are excluded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()

    rounds = load_round_summaries(args.data_root, include_skipped=args.include_skipped)
    summary = summarize_best_vs_replay(rounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "best_vs_changed_human_behavior_summary.csv"
    summary.to_csv(summary_csv, index=False)

    created = make_all_plots(summary, args.output_dir, formats=list(args.formats))

    print(f"Loaded participants: {summary['participant_id'].nunique()}")
    print(f"Saved summary: {summary_csv}")
    for path in created:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
