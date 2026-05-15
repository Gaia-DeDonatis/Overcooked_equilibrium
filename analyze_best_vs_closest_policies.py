#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph 3: Best-policy vs closest played policies.

For each participant, this script compares:

    left column  -> Best-policy
    right column -> closest policies in policy space that the participant actually played

Visual design:
    - one color per participant
    - one starting dot at Best-policy
    - three ending dots at closest policies C1, C2, C3
    - all C1/C2/C3 points are on the same right-side column
    - participant colors are shown in the legend
    - episode numbers are written near the closest-policy endpoints

It also creates a cleaner version with only:

    Best-policy -> C1

Outputs
-------
    best_vs_closest_policies_dishes_per_round.png
    best_vs_closest_policies_human_steps_per_dish.png
    best_vs_closest_policy_dishes_per_round.png
    best_vs_closest_policy_human_steps_per_dish.png
    episode_policy_metrics_summary.csv
    best_vs_closest_policies_summary.csv

Example
-------
python analyze_best_vs_closest_policies.py \
    --data-root submissions \
    --embedding-csv tsnt_thinpath.csv \
    --output-dir figures/best_vs_closest_policies \
    --formats png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import (
    apply_plot_style,
    clean_axis,
    save_figure,
    get_participant_color,
    metric_label,
    COLORS,
    FIGSIZE_WIDE,
    ALPHA_LINE,
    LINE_MARKER_KWS,
)


# ============================================================
# CONFIG
# ============================================================

BEST_PHASE = "bo_replay_best"

DEFAULT_POLICY_PREFIX = "[coplay][flexible][thinpath]agent0_"

PLOT_SPECS = [
    ("best_vs_closest_policies_dishes_per_round", "mean_dishes_per_round"),
    ("best_vs_closest_policies_human_steps_per_dish", "human_steps_per_dish"),
]

SINGLE_CLOSEST_SPECS = [
    ("best_vs_closest_policy_dishes_per_round", "mean_dishes_per_round"),
    ("best_vs_closest_policy_human_steps_per_dish", "human_steps_per_dish"),
]


# ============================================================
# BASIC LOADING UTILITIES
# ============================================================

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


def participant_id_from_file(csv_path: Path, df: pd.DataFrame) -> str:
    """Prefer folder name because it is usually cleaner than prolific_id."""
    folder_name = csv_path.parent.name

    if folder_name:
        return folder_name

    if "prolific_id" in df.columns and df["prolific_id"].notna().any():
        return str(df["prolific_id"].dropna().iloc[0])

    return "unknown_participant"


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_round_summaries(
    data_root: Path,
    include_skipped: bool = False,
) -> pd.DataFrame:
    """Load and concatenate all participant round_summary.csv files."""
    files = find_round_summary_files(data_root)

    if not files:
        raise FileNotFoundError(f"No round_summary.csv files found under: {data_root}")

    frames = []

    required_cols = {
        "episode_index",
        "episode_phase",
        "policy_id",
        "dishes_served",
        "human_steps",
    }

    for csv_path in files:
        df = pd.read_csv(csv_path)
        df["participant_id"] = participant_id_from_file(csv_path, df)

        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

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
# POLICY NAME NORMALIZATION
# ============================================================

def strip_policy_prefix(
    policy_name: str,
    policy_prefix: str | None = None,
) -> str:
    """
    Normalize policy names so comparisons work whether the name is stored as:

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


# ============================================================
# EMBEDDING LOADING AND NEAREST-POLICY SEARCH
# ============================================================

def infer_embedding_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Infer:
        - policy-name column
        - x embedding column
        - y embedding column
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

    for candidate_x, candidate_y in xy_candidates:
        if candidate_x in df.columns and candidate_y in df.columns:
            return name_col, candidate_x, candidate_y

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


def load_policy_embedding(
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
            x_col: "embedding_x",
            y_col: "embedding_y",
        }
    )

    out["policy_id_raw"] = out["policy_id_raw"].astype(str)
    out["policy_short"] = out["policy_id_raw"].apply(
        lambda p: strip_policy_prefix(p, policy_prefix)
    )

    out["embedding_x"] = pd.to_numeric(out["embedding_x"], errors="coerce")
    out["embedding_y"] = pd.to_numeric(out["embedding_y"], errors="coerce")

    out = out.loc[
        out["policy_short"].apply(is_valid_policy)
        & out["embedding_x"].notna()
        & out["embedding_y"].notna()
    ].copy()

    out = out.drop_duplicates(
        subset=["policy_short"],
        keep="first",
    ).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid policies found in embedding CSV: {embedding_csv}")

    return out


def policy_distances_from_best(
    embedding: pd.DataFrame,
    best_policy_short: str,
) -> pd.DataFrame:
    """Compute Euclidean distance from the Best-policy to every other policy."""
    best_policy_short = str(best_policy_short).strip()

    best_rows = embedding.loc[embedding["policy_short"] == best_policy_short]

    if best_rows.empty:
        sample = embedding["policy_short"].head(10).tolist()
        raise ValueError(
            f"Best-policy '{best_policy_short}' was not found in embedding CSV. "
            f"Sample available policies: {sample}"
        )

    best_x = float(best_rows.iloc[0]["embedding_x"])
    best_y = float(best_rows.iloc[0]["embedding_y"])

    distances = embedding.copy()
    distances["policy_distance"] = np.sqrt(
        (distances["embedding_x"].astype(float) - best_x) ** 2
        + (distances["embedding_y"].astype(float) - best_y) ** 2
    )

    return distances[
        ["policy_short", "policy_distance", "embedding_x", "embedding_y"]
    ].copy()


# ============================================================
# EPISODE-LEVEL METRICS
# ============================================================

def summarize_episodes(
    rounds: pd.DataFrame,
    policy_prefix: str | None,
) -> pd.DataFrame:
    """
    Produce one row per participant, episode, phase, and policy.

    human_steps_per_dish is computed as:
        total human steps / total dishes
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

        row = {
            "participant_id": participant_id,
            "episode_index": episode_index,
            "episode_phase": episode_phase,
            "policy_id": policy_id,
            "policy_short": policy_short,
            "n_rounds": int(len(g)),
            "total_dishes": total_dishes,
            "mean_dishes_per_round": float(g["dishes_served"].mean(skipna=True)),
            "total_human_steps": total_human_steps,
            "human_steps_per_dish": (
                total_human_steps / total_dishes
                if total_dishes > 0
                else np.nan
            ),
        }

        for optional_col in [
            "ai_steps",
            "team_reward_score",
            "human_reward_score",
            "ai_reward_score",
            "mental_demand",
            "performance_score",
        ]:
            if optional_col in g.columns:
                row[optional_col] = float(g[optional_col].mean(skipna=True))

        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        ["participant_id", "episode_index", "episode_phase"]
    ).reset_index(drop=True)


def select_best_and_closest_played_policies(
    episode_summary: pd.DataFrame,
    embedding: pd.DataFrame,
    n_closest: int = 3,
    unique_policies: bool = True,
) -> pd.DataFrame:
    """
    For each participant:
        1. find Best-policy episode from bo_replay_best
        2. compute distances from this policy in embedding space
        3. restrict to policies actually played by the participant
        4. select the nearest n_closest policies, excluding the Best-policy itself
    """
    rows = []

    for participant_id, p_df in episode_summary.groupby("participant_id", sort=True):
        best_rows = p_df.loc[p_df["episode_phase"] == BEST_PHASE].copy()

        if best_rows.empty:
            print(f"[WARN] {participant_id}: no {BEST_PHASE} episode found. Skipping.")
            continue

        best_rows = best_rows.sort_values("episode_index")
        best = best_rows.iloc[0]

        best_policy_short = str(best["policy_short"])

        try:
            distances = policy_distances_from_best(
                embedding=embedding,
                best_policy_short=best_policy_short,
            )
        except ValueError as e:
            print(f"[WARN] {participant_id}: {e}. Skipping.")
            continue

        candidates = p_df.copy()

        candidates = candidates.loc[
            candidates["policy_short"].apply(is_valid_policy)
            & (candidates["policy_short"] != best_policy_short)
        ].copy()

        if candidates.empty:
            print(f"[WARN] {participant_id}: no non-best played policies found.")
            continue

        candidates = candidates.merge(
            distances,
            on="policy_short",
            how="left",
        )

        candidates = candidates.loc[candidates["policy_distance"].notna()].copy()

        if candidates.empty:
            print(
                f"[WARN] {participant_id}: none of the played non-best policies "
                "were found in the embedding CSV."
            )
            continue

        candidates = candidates.sort_values(
            ["policy_distance", "episode_index"],
            ascending=[True, True],
        ).reset_index(drop=True)

        if unique_policies:
            candidates = candidates.drop_duplicates(
                subset=["policy_short"],
                keep="first",
            ).reset_index(drop=True)

        closest = candidates.head(int(n_closest)).copy()

        if len(closest) < int(n_closest):
            print(
                f"[WARN] {participant_id}: only found {len(closest)} closest "
                f"played policies, requested {n_closest}."
            )

        for rank, (_, candidate) in enumerate(closest.iterrows(), start=1):
            out = {
                "participant_id": participant_id,

                "best_episode_index": best["episode_index"],
                "best_episode_phase": best["episode_phase"],
                "best_policy_id": best["policy_id"],
                "best_policy_short": best["policy_short"],
                "best_n_rounds": best["n_rounds"],

                "closest_rank": rank,
                "closest_episode_index": candidate["episode_index"],
                "closest_episode_phase": candidate["episode_phase"],
                "closest_policy_id": candidate["policy_id"],
                "closest_policy_short": candidate["policy_short"],
                "closest_policy_distance": candidate["policy_distance"],
                "closest_n_rounds": candidate["n_rounds"],
            }

            for metric in [
                "mean_dishes_per_round",
                "human_steps_per_dish",
                "total_dishes",
                "total_human_steps",
                "team_reward_score",
                "mental_demand",
                "performance_score",
            ]:
                if metric in best.index:
                    out[f"best_{metric}"] = best[metric]
                if metric in candidate.index:
                    out[f"closest_{metric}"] = candidate[metric]

            rows.append(out)

    selected = pd.DataFrame(rows)

    if selected.empty:
        raise ValueError(
            "No Best-policy vs closest-policy rows were created. "
            "Check that policy IDs in round_summary.csv match the embedding CSV."
        )

    return selected.sort_values(
        ["participant_id", "closest_rank"]
    ).reset_index(drop=True)


# ============================================================
# PLOTTING — FAN-OUT: BEST VS C1/C2/C3
# ============================================================

def plot_best_vs_closest_metric(
    selected: pd.DataFrame,
    metric: str,
    output_path: Path,
    formats: Iterable[str],
    n_closest: int,
) -> None:
    """
    Create fan-out paired plot:
        Best-policy -> three closest policies

    Visual rules:
        - one color per participant
        - all closest policies are on the same x-column
        - participant colors are shown in the legend
        - closest-policy episode numbers are written near endpoints
    """
    best_col = f"best_{metric}"
    closest_col = f"closest_{metric}"

    if best_col not in selected.columns or closest_col not in selected.columns:
        raise ValueError(f"Metric '{metric}' is not available in selected summary.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    x_best = 0.0
    x_closest = 1.0

    x_closest_by_rank = {
        rank: x_closest
        for rank in range(1, int(n_closest) + 1)
    }

    participant_handles = []

    for i, (participant_id, p_df) in enumerate(selected.groupby("participant_id", sort=True)):
        color = get_participant_color(participant_id, fallback_index=i)
        participant_label = str(participant_id).replace("Thinpath_", "")

        participant_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                label=participant_label,
                **LINE_MARKER_KWS,
            )
        )

        best_values = p_df[best_col].dropna()

        if best_values.empty:
            continue

        best_y = float(best_values.iloc[0])

        for _, row in p_df.iterrows():
            closest_y = row[closest_col]

            if pd.isna(closest_y):
                continue

            rank = int(row["closest_rank"])
            x_end = x_closest_by_rank.get(rank, x_closest)

            ax.plot(
                [x_best, x_end],
                [best_y, float(closest_y)],
                color=color,
                alpha=ALPHA_LINE,
                label="_nolegend_",
                **LINE_MARKER_KWS,
            )

            episode_index = row.get("closest_episode_index", np.nan)

            if pd.notna(episode_index):
                episode_label = f"E{int(episode_index)}"
            else:
                episode_label = f"C{rank}"

            ax.text(
                x_end + 0.035,
                float(closest_y),
                episode_label,
                va="center",
                ha="left",
                fontsize=8.0,
                color=color,
            )

    # Group mean overlay.
    best_mean_by_participant = (
        selected.groupby("participant_id")[best_col]
        .first()
        .dropna()
    )

    group_mean_plotted = False

    if not best_mean_by_participant.empty:
        group_best_y = float(best_mean_by_participant.mean())

        for rank in range(1, int(n_closest) + 1):
            rank_values = selected.loc[
                selected["closest_rank"] == rank,
                closest_col,
            ].dropna()

            if rank_values.empty:
                continue

            group_closest_y = float(rank_values.mean())

            ax.plot(
                [x_best, x_closest],
                [group_best_y, group_closest_y],
                color=COLORS["group_mean"],
                label="_nolegend_",
                zorder=20,
                **LINE_MARKER_KWS,
            )

            group_mean_plotted = True

    ax.set_xticks([x_best, x_closest])
    ax.set_xticklabels(["Best-policy", "Closest policies\nC1 / C2 / C3"])
    ax.set_xlim(-0.25, 1.45)

    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel("")
    ax.set_title(metric_label(metric))

    if metric == "human_steps_per_dish":
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

    legend_handles = participant_handles

    if group_mean_plotted:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["group_mean"],
                label="Group mean",
                **LINE_MARKER_KWS,
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Participant",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    save_figure(fig, output_path, formats=formats, close=True)


# ============================================================
# PLOTTING — CLEAN VERSION: BEST VS C1 ONLY
# ============================================================

def plot_best_vs_single_closest_metric(
    selected: pd.DataFrame,
    metric: str,
    output_path: Path,
    formats: Iterable[str],
) -> None:
    """
    Create cleaner paired plot:
        Best-policy -> closest played policy only, C1
    """
    best_col = f"best_{metric}"
    closest_col = f"closest_{metric}"

    if best_col not in selected.columns or closest_col not in selected.columns:
        raise ValueError(f"Metric '{metric}' is not available in selected summary.")

    plot_df = selected.loc[selected["closest_rank"] == 1].copy()

    if plot_df.empty:
        raise ValueError("No closest_rank == 1 rows found. Cannot create single-closest plot.")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    x_best = 0.0
    x_closest = 1.0

    participant_handles = []

    for i, (participant_id, p_df) in enumerate(plot_df.groupby("participant_id", sort=True)):
        color = get_participant_color(participant_id, fallback_index=i)
        participant_label = str(participant_id).replace("Thinpath_", "")

        participant_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                label=participant_label,
                **LINE_MARKER_KWS,
            )
        )

        row = p_df.iloc[0]

        best_y = row[best_col]
        closest_y = row[closest_col]

        if pd.isna(best_y) or pd.isna(closest_y):
            continue

        ax.plot(
            [x_best, x_closest],
            [float(best_y), float(closest_y)],
            color=color,
            alpha=ALPHA_LINE,
            label="_nolegend_",
            **LINE_MARKER_KWS,
        )

        episode_index = row.get("closest_episode_index", np.nan)

        if pd.notna(episode_index):
            episode_label = f"E{int(episode_index)}"
        else:
            episode_label = "C1"

        ax.text(
            x_closest + 0.035,
            float(closest_y),
            episode_label,
            va="center",
            ha="left",
            fontsize=8.0,
            color=color,
        )

    # Group mean overlay.
    paired = plot_df[[best_col, closest_col]].dropna()

    group_mean_plotted = False

    if not paired.empty:
        group_best_y = float(paired[best_col].mean())
        group_closest_y = float(paired[closest_col].mean())

        ax.plot(
            [x_best, x_closest],
            [group_best_y, group_closest_y],
            color=COLORS["group_mean"],
            label="_nolegend_",
            zorder=20,
            **LINE_MARKER_KWS,
        )

        group_mean_plotted = True

    ax.set_xticks([x_best, x_closest])
    ax.set_xticklabels(["Best-policy", "Closest policy\nC1"])
    ax.set_xlim(-0.18, 1.35)

    ax.set_ylabel(metric_label(metric))
    ax.set_xlabel("")
    ax.set_title(metric_label(metric))

    if metric == "human_steps_per_dish":
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

    legend_handles = participant_handles

    if group_mean_plotted:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["group_mean"],
                label="Group mean",
                **LINE_MARKER_KWS,
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Participant",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    save_figure(fig, output_path, formats=formats, close=True)


def make_all_plots(
    selected: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    n_closest: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # Fan-out plots: Best-policy vs C1/C2/C3.
    for stem, metric in PLOT_SPECS:
        plot_best_vs_closest_metric(
            selected=selected,
            metric=metric,
            output_path=output_dir / stem,
            formats=formats,
            n_closest=n_closest,
        )

        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    # Cleaner plots: Best-policy vs C1 only.
    for stem, metric in SINGLE_CLOSEST_SPECS:
        plot_best_vs_single_closest_metric(
            selected=selected,
            metric=metric,
            output_path=output_dir / stem,
            formats=formats,
        )

        created.extend(output_dir / f"{stem}.{fmt}" for fmt in formats)

    return created


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Graph 3: Best-policy vs closest played policies "
            "in policy space."
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
        default=Path("figures/best_vs_closest_policies"),
        help="Where to save figures and summary CSV.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf.",
    )

    parser.add_argument(
        "--n-closest",
        type=int,
        default=3,
        help="Number of closest played policies to compare against Best-policy.",
    )

    parser.add_argument(
        "--policy-prefix",
        type=str,
        default=DEFAULT_POLICY_PREFIX,
        help=(
            "Policy prefix to strip when matching policy IDs. "
            "Default is the thinpath coplay prefix."
        ),
    )

    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include rows marked as skipped_episode. By default they are excluded.",
    )

    parser.add_argument(
        "--allow-duplicate-policies",
        action="store_true",
        help=(
            "Allow the same non-best policy to appear more than once if played "
            "in multiple episodes. By default, each closest policy is unique."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()

    rounds = load_round_summaries(
        data_root=args.data_root,
        include_skipped=args.include_skipped,
    )

    embedding = load_policy_embedding(
        embedding_csv=args.embedding_csv,
        policy_prefix=args.policy_prefix,
    )

    episode_summary = summarize_episodes(
        rounds=rounds,
        policy_prefix=args.policy_prefix,
    )

    selected = select_best_and_closest_played_policies(
        episode_summary=episode_summary,
        embedding=embedding,
        n_closest=args.n_closest,
        unique_policies=not args.allow_duplicate_policies,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_summary_csv = args.output_dir / "episode_policy_metrics_summary.csv"
    selected_csv = args.output_dir / "best_vs_closest_policies_summary.csv"

    episode_summary.to_csv(episode_summary_csv, index=False)
    selected.to_csv(selected_csv, index=False)

    created = make_all_plots(
        selected=selected,
        output_dir=args.output_dir,
        formats=list(args.formats),
        n_closest=args.n_closest,
    )

    print(f"Loaded participants: {episode_summary['participant_id'].nunique()}")
    print(f"Embedding policies: {embedding['policy_short'].nunique()}")
    print(f"Saved episode summary: {episode_summary_csv}")
    print(f"Saved selected closest-policy summary: {selected_csv}")

    for path in created:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()