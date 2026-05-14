#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph: Best-policy map across participants.

This script creates ONE policy-map figure showing:
    - the full policy pool in the background
    - the policy assigned as Best-policy for each participant
    - labels next to each Best-policy point indicating which participant(s)
      received that policy as best

Expected input
--------------
A CSV like:
    episode_policy_pool_summary_mean_dishes_per_round.csv

with at least these columns:
    participant_id
    episode_index   OR episode_phase
    emb_x / emb_y   (or equivalent x/y embedding columns)

Optional:
---------
You can also pass an embedding CSV (for example tsnt_thinpath.csv)
to display the full policy pool, even if some policies were never played
by the participants in the summary file.

Default logic for Best-policy
-----------------------------
The script looks for:
    episode_phase == "bo_replay_best"
If that column is not available, it falls back to:
    episode_index == 13

Outputs
-------
    best_policy_map_across_participants.png
    best_policy_map_across_participants.pdf
    best_policy_map_across_participants_summary.csv

Example
-------
python analyze_best_policy_map_across_participants.py ^
    --summary-csv episode_policy_pool_summary_mean_dishes_per_round.csv ^
    --embedding-csv tsnt_thinpath.csv ^
    --output-dir figures/best_policy_map ^
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

# ---------------------------------------------------------------------
# Optional use of your local plot_style.py
# ---------------------------------------------------------------------
try:
    from plot_style import apply_plot_style, save_figure
except Exception:
    def apply_plot_style() -> None:
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "figure.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        })

    def save_figure(fig, output_path: Path, formats: Iterable[str], close: bool = True) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(output_path.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=300)
        if close:
            plt.close(fig)


BEST_PHASE = "bo_replay_best"
BEST_EPISODE_INDEX = 13


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a global map of policies and label which participant received each Best-policy."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("episode_policy_pool_summary_mean_dishes_per_round.csv"),
        help="CSV with participant episode summaries and embedding coordinates.",
    )
    parser.add_argument(
        "--embedding-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing the full policy pool embeddings "
            "(e.g. tsnt_thinpath.csv). If not provided, the pool is built "
            "from unique coordinates found in the summary CSV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/best_policy_map_across_participants"),
        help="Folder where outputs will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Figure formats to save, e.g. --formats png pdf",
    )
    return parser.parse_args()


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _detect_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    x_col = _find_first_existing(
        list(df.columns),
        ["emb_x", "x", "tsne_x", "policy_embedding_x", "embed_x"]
    )
    y_col = _find_first_existing(
        list(df.columns),
        ["emb_y", "y", "tsne_y", "policy_embedding_y", "embed_y"]
    )

    if x_col is None or y_col is None:
        raise ValueError(
            "Could not detect embedding columns. "
            "Expected something like emb_x/emb_y, x/y, or tsne_x/tsne_y."
        )
    return x_col, y_col


def load_summary(summary_csv: Path) -> tuple[pd.DataFrame, str, str]:
    summary_csv = summary_csv.expanduser().resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    if "participant_id" not in df.columns:
        raise ValueError("The summary CSV must contain a 'participant_id' column.")

    x_col, y_col = _detect_xy_columns(df)

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    if "episode_index" in df.columns:
        df["episode_index"] = pd.to_numeric(df["episode_index"], errors="coerce")

    if "episode_phase" in df.columns:
        df["episode_phase"] = df["episode_phase"].astype(str).str.strip()

    df = df.dropna(subset=[x_col, y_col]).copy()
    return df, x_col, y_col


def load_policy_pool(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    embedding_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Load full policy pool from an embedding CSV if available.
    Otherwise fallback to unique coordinates from the summary CSV.
    """
    if embedding_csv is None:
        pool = summary_df[[x_col, y_col]].drop_duplicates().copy()
        pool = pool.rename(columns={x_col: "x", y_col: "y"})
        return pool

    embedding_csv = embedding_csv.expanduser().resolve()
    if not embedding_csv.exists():
        raise FileNotFoundError(f"Embedding CSV not found: {embedding_csv}")

    emb = pd.read_csv(embedding_csv)
    ex_col, ey_col = _detect_xy_columns(emb)

    emb[ex_col] = pd.to_numeric(emb[ex_col], errors="coerce")
    emb[ey_col] = pd.to_numeric(emb[ey_col], errors="coerce")

    pool = emb[[ex_col, ey_col]].dropna().drop_duplicates().copy()
    pool = pool.rename(columns={ex_col: "x", ey_col: "y"})
    return pool


def select_best_policy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select one best-policy row per participant.

    Priority:
        1) episode_phase == bo_replay_best
        2) episode_index == 13
    """
    if "episode_phase" in df.columns and (df["episode_phase"] == BEST_PHASE).any():
        best = df.loc[df["episode_phase"] == BEST_PHASE].copy()
    elif "episode_index" in df.columns:
        best = df.loc[df["episode_index"] == BEST_EPISODE_INDEX].copy()
    else:
        raise ValueError(
            "Could not identify Best-policy rows. Need either "
            "'episode_phase' with bo_replay_best or 'episode_index' with episode 13."
        )

    if best.empty:
        raise ValueError("No Best-policy rows found.")

    # Keep only one row per participant
    best = (
        best.sort_values(["participant_id"])
            .groupby("participant_id", as_index=False)
            .first()
    )
    return best


def shorten_participant_id(pid: str) -> str:
    pid = str(pid)
    return pid.replace("Thinpath_", "").replace("thinpath_", "")


def aggregate_best_points(best_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Aggregate participants who share the same Best-policy coordinates.
    """
    tmp = best_df.copy()
    tmp["x_round"] = tmp[x_col].round(6)
    tmp["y_round"] = tmp[y_col].round(6)
    tmp["participant_short"] = tmp["participant_id"].map(shorten_participant_id)

    rows = []
    for (xr, yr), g in tmp.groupby(["x_round", "y_round"], sort=False):
        participants = sorted(g["participant_short"].tolist())
        label = "\n".join(participants)

        rows.append({
            "x": float(g[x_col].iloc[0]),
            "y": float(g[y_col].iloc[0]),
            "participants": participants,
            "label": label,
            "n_participants": len(participants),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["x", "y"]).reset_index(drop=True)


def build_summary_table(best_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = best_df[["participant_id"]].copy()
    out["participant_short"] = out["participant_id"].map(shorten_participant_id)
    out["best_episode_index"] = (
        best_df["episode_index"] if "episode_index" in best_df.columns else np.nan
    )
    out["best_policy_x"] = best_df[x_col].to_numpy()
    out["best_policy_y"] = best_df[y_col].to_numpy()

    if "episode_phase" in best_df.columns:
        out["episode_phase"] = best_df["episode_phase"].to_numpy()

    return out


def annotate_points(ax, points_df: pd.DataFrame) -> None:
    """
    Add participant labels next to each best-policy point.
    """
    offsets = [
        (10, 8),
        (10, -10),
        (-10, 8),
        (-10, -10),
        (12, 0),
        (-12, 0),
        (0, 12),
        (0, -12),
    ]

    for i, row in points_df.iterrows():
        dx, dy = offsets[i % len(offsets)]
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        ax.annotate(
            row["label"],
            xy=(row["x"], row["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="none",
                alpha=0.85,
            ),
        )


def plot_best_policy_map(
    pool_df: pd.DataFrame,
    best_points_df: pd.DataFrame,
    output_path: Path,
    formats: Iterable[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    # Background policy pool
    ax.scatter(
        pool_df["x"],
        pool_df["y"],
        s=80,
        facecolor="#D8D8D8",
        edgecolor="none",
        alpha=0.85,
        label="Policy pool",
        zorder=1,
    )

    # Best-policy points
    ax.scatter(
        best_points_df["x"],
        best_points_df["y"],
        s=180,
        facecolor="#1F1F1F",
        edgecolor="white",
        linewidth=1.5,
        label="Best-policy",
        zorder=3,
    )

    annotate_points(ax, best_points_df)

    ax.set_title("Best-policy map across participants", pad=10)
    ax.set_xlabel("Policy embedding x")
    ax.set_ylabel("Policy embedding y")

    # Cleaner look
    ax.grid(False)
    sns.despine(ax=ax)

    # Legend
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    # Small note
    ax.text(
        0.01,
        0.01,
        "Labels indicate which participant received that policy as Best-policy",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    save_figure(fig, output_path, formats=formats, close=True)


def main() -> None:
    args = parse_args()
    apply_plot_style()

    summary_df, x_col, y_col = load_summary(args.summary_csv)
    pool_df = load_policy_pool(summary_df, x_col, y_col, args.embedding_csv)

    best_df = select_best_policy_rows(summary_df)
    best_points_df = aggregate_best_points(best_df, x_col, y_col)
    summary_table = build_summary_table(best_df, x_col, y_col)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV summary
    summary_csv = args.output_dir / "best_policy_map_across_participants_summary.csv"
    summary_table.to_csv(summary_csv, index=False)

    # Save figure
    output_stem = args.output_dir / "best_policy_map_across_participants"
    plot_best_policy_map(
        pool_df=pool_df,
        best_points_df=best_points_df,
        output_path=output_stem,
        formats=list(args.formats),
    )

    print(f"Loaded participants: {best_df['participant_id'].nunique()}")
    print(f"Unique best-policy locations: {len(best_points_df)}")
    print(f"Saved summary: {summary_csv}")
    for fmt in args.formats:
        print(f"Saved figure: {output_stem.with_suffix(f'.{fmt}')}")

if __name__ == "__main__":
    main()