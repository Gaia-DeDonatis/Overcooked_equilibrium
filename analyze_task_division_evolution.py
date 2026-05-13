#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis 2: Task division evolution heatmaps
--------------------------------------------

Goal:
    Visualize how Human and AI divide work over time.

Main question:
    During Seed/BO/Best-policy phases, do Human and AI start to specialize
    into clearer roles?

Input expected:

    Overcooked_equilibrium/
        analyze_task_division_evolution.py
        submissions/
            Thinpath_P01/
                task_division_summary.csv
                round_summary.csv          optional, used only to mark skipped episodes
            Thinpath_P02/
                task_division_summary.csv
                round_summary.csv
            ...

Output:
    analysis_outputs/task_division_evolution/
        task_division_evolution_long.csv
        task_division_heatmap_Thinpath_P01.png
        task_division_heatmap_Thinpath_P02.png
        ...

How to read the heatmap:
    Each cell is the Human share for that task in that episode.

        0.0 = mostly AI
        0.5 = shared
        1.0 = mostly Human
        gray = task not performed in that episode

    A star (*) in an episode label means that episode was skipped/incomplete
    according to round_summary.csv, if available.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_SOURCES = [
    PROJECT_ROOT / "submissions",
]

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "task_division_evolution"

# Only folders like Thinpath_P01, Thinpath_P02, ..., Thinpath_P10
PARTICIPANT_FOLDER_PATTERN = re.compile(r"^Thinpath_P\d+$", re.IGNORECASE)

# Phases to show in the evolution heatmap.
# Solo is excluded by default because task division is trivial without AI.
INCLUDE_PHASES = [
    "seed",
    "bo",
    "bo_replay_best",
    "stress",
    "replay_optimal",
]

# Palette: Gradient Blues
# Dark/purple = AI-dominated tasks
# Light/turquoise = Human-dominated tasks
HEATMAP_COLORS = [
    "#7400B8",
    "#6930C3",
    "#5E60CE",
    "#5390D9",
    "#4EA8DE",
    "#48BFE3",
    "#56CFE1",
    "#64DFDF",
    "#72EFDD",
    "#80FFDB",
]

GROUP_LINE_COLOR = "#222222"

# Collapse raw columns into clearer task labels.
# The script automatically ignores columns that are not present.
TASK_DEFINITIONS = {
    "Pickup lettuce": [
        "pickup_lettuce_1",
        "pickup_lettuce_2",
    ],
    "Pickup plate": [
        "pickup_plate_1",
        "pickup_plate_2",
    ],
    "Place on counter": [
        "place_on_counter",
    ],
    "Place on knife": [
        "place_on_knife",
    ],
    "Pick up from knife": [
        "pickup_from_knife",
    ],
    "Chop": [
        "chop",
    ],
    "Assemble salad": [
        "assemble_salad",
    ],
    "Deliver correct": [
        "deliver_correct",
    ],
    "Deliver wrong": [
        "deliver_wrong",
    ],
}

# Optional: hide rare/less informative tasks if they are never used.
DROP_TASKS_WITH_ZERO_TOTAL = True


# ============================================================
# FILE LOADING
# ============================================================

def is_valid_participant_folder(folder_name: str) -> bool:
    return bool(PARTICIPANT_FOLDER_PATTERN.match(folder_name))


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", participant_id)
    if match:
        return int(match.group(1)), participant_id
    return 9999, participant_id


def find_task_division_files(data_sources: Iterable[Path]) -> list[Path]:
    """Find task_division_summary.csv files in valid participant folders."""
    files: list[Path] = []

    for src in data_sources:
        src = Path(src).expanduser()

        if not src.exists():
            print(f"[WARN] Path not found: {src}")
            continue

        if src.is_file() and src.name == "task_division_summary.csv":
            if is_valid_participant_folder(src.parent.name):
                files.append(src)

        elif src.is_dir():
            found = src.rglob("task_division_summary.csv")
            files.extend(
                f for f in found
                if is_valid_participant_folder(f.parent.name)
            )

    unique_files = []
    seen = set()

    for f in sorted(files):
        key = f.resolve()
        if key not in seen:
            unique_files.append(f)
            seen.add(key)

    return unique_files


def load_task_division(task_files: list[Path]) -> pd.DataFrame:
    """Load and combine all task_division_summary.csv files."""
    frames = []

    for f in task_files:
        df = pd.read_csv(f)
        df["participant_id"] = f.parent.name
        df["source_file"] = str(f)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No task_division_summary.csv files found. "
            "Check that participant folders are named Thinpath_P01, Thinpath_P02, etc."
        )

    data = pd.concat(frames, ignore_index=True)

    required_cols = [
        "participant_id",
        "episode_index",
        "episode_phase",
        "agent",
    ]

    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column in task_division_summary.csv: {col}")

    data["episode_phase"] = data["episode_phase"].astype(str).str.strip()
    data["agent"] = data["agent"].astype(str).str.strip().str.lower()
    data["episode_index"] = pd.to_numeric(data["episode_index"], errors="coerce")

    for raw_cols in TASK_DEFINITIONS.values():
        for col in raw_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    return data


def load_skipped_episode_flags(task_files: list[Path]) -> pd.DataFrame:
    """
    Optional helper:
    load round_summary.csv files next to each task_division_summary.csv
    and extract which episodes were skipped/incomplete.

    Returns columns:
        participant_id, episode_index, was_skipped
    """
    rows = []

    for task_file in task_files:
        participant_id = task_file.parent.name
        round_file = task_file.parent / "round_summary.csv"

        if not round_file.exists():
            continue

        try:
            df = pd.read_csv(round_file)
        except Exception as exc:
            print(f"[WARN] Could not read {round_file}: {exc}")
            continue

        if "episode_index" not in df.columns or "skipped_episode" not in df.columns:
            continue

        df["episode_index"] = pd.to_numeric(df["episode_index"], errors="coerce")
        skipped = df["skipped_episode"].astype(str).str.lower().isin(["true", "1", "yes"])

        flag = (
            pd.DataFrame({
                "participant_id": participant_id,
                "episode_index": df["episode_index"],
                "was_skipped": skipped,
            })
            .groupby(["participant_id", "episode_index"], as_index=False)["was_skipped"]
            .any()
        )

        rows.append(flag)

    if not rows:
        return pd.DataFrame(columns=["participant_id", "episode_index", "was_skipped"])

    return pd.concat(rows, ignore_index=True)


# ============================================================
# TASK DIVISION COMPUTATION
# ============================================================

def collapse_task_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create clear task columns from raw task columns.
    Example:
        pickup_lettuce_1 + pickup_lettuce_2 -> Pickup lettuce
    """
    out = data.copy()

    for task_label, raw_cols in TASK_DEFINITIONS.items():
        existing_cols = [c for c in raw_cols if c in out.columns]

        if existing_cols:
            out[task_label] = out[existing_cols].sum(axis=1)
        else:
            out[task_label] = 0

    return out


def compute_human_share_by_episode(
    task_data: pd.DataFrame,
    skipped_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute Human share for every participant x episode x task.

    Human share:
        human_count / (human_count + ai_count)
    """
    data = collapse_task_columns(task_data)

    if INCLUDE_PHASES:
        data = data.loc[data["episode_phase"].isin(INCLUDE_PHASES)].copy()

    task_labels = list(TASK_DEFINITIONS.keys())

    grouped = (
        data
        .groupby(
            ["participant_id", "episode_index", "episode_phase", "agent"],
            as_index=False,
        )[task_labels]
        .sum()
    )

    rows = []

    episode_keys = (
        grouped[["participant_id", "episode_index", "episode_phase"]]
        .drop_duplicates()
        .sort_values(["participant_id", "episode_index"])
    )

    for _, ep in episode_keys.iterrows():
        pid = ep["participant_id"]
        ep_idx = ep["episode_index"]
        ep_phase = ep["episode_phase"]

        ep_rows = grouped.loc[
            (grouped["participant_id"] == pid)
            & (grouped["episode_index"] == ep_idx)
            & (grouped["episode_phase"] == ep_phase)
        ]

        human_row = ep_rows.loc[ep_rows["agent"] == "human"]
        ai_row = ep_rows.loc[ep_rows["agent"] == "ai"]

        for task in task_labels:
            human_count = float(human_row[task].sum()) if not human_row.empty else 0.0
            ai_count = float(ai_row[task].sum()) if not ai_row.empty else 0.0
            total_count = human_count + ai_count

            if total_count > 0:
                human_share = human_count / total_count
            else:
                human_share = np.nan

            rows.append({
                "participant_id": pid,
                "episode_index": int(ep_idx) if pd.notna(ep_idx) else np.nan,
                "episode_phase": ep_phase,
                "task": task,
                "human_count": human_count,
                "ai_count": ai_count,
                "total_count": total_count,
                "human_share": human_share,
            })

    result = pd.DataFrame(rows)

    if skipped_flags is not None and not skipped_flags.empty:
        result = result.merge(
            skipped_flags,
            on=["participant_id", "episode_index"],
            how="left",
        )
        result["was_skipped"] = result["was_skipped"].fillna(False).astype(bool)
    else:
        result["was_skipped"] = False

    if DROP_TASKS_WITH_ZERO_TOTAL:
        task_totals = result.groupby("task")["total_count"].sum()
        keep_tasks = task_totals[task_totals > 0].index.tolist()
        result = result.loc[result["task"].isin(keep_tasks)].copy()

    return result.sort_values(["participant_id", "episode_index", "task"])


# ============================================================
# PLOTTING
# ============================================================

def phase_short_label(phase: str) -> str:
    mapping = {
        "seed": "Seed",
        "bo": "BO",
        "bo_replay_best": "Best",
        "stress": "Stress",
        "replay_optimal": "Final",
        "solo": "Solo",
    }
    return mapping.get(str(phase), str(phase))


def make_episode_labels(participant_data: pd.DataFrame) -> list[str]:
    episodes = (
        participant_data[["episode_index", "episode_phase", "was_skipped"]]
        .drop_duplicates()
        .sort_values("episode_index")
    )

    labels = []

    for _, row in episodes.iterrows():
        ep = int(row["episode_index"])
        phase = phase_short_label(row["episode_phase"])
        star = "*" if bool(row["was_skipped"]) else ""

        labels.append(f"E{ep}{star}\n{phase}")

    return labels


def plot_task_division_heatmap(
    participant_data: pd.DataFrame,
    participant_id: str,
    output_path: Path,
) -> None:
    """
    Plot one task-division evolution heatmap for one participant.
    """
    episodes = (
        participant_data[["episode_index", "episode_phase", "was_skipped"]]
        .drop_duplicates()
        .sort_values("episode_index")
    )

    tasks = list(TASK_DEFINITIONS.keys())
    tasks = [t for t in tasks if t in participant_data["task"].unique()]

    if len(episodes) == 0 or len(tasks) == 0:
        print(f"[WARN] No heatmap data for {participant_id}")
        return

    matrix = []

    for task in tasks:
        row_values = []
        for ep_idx in episodes["episode_index"]:
            value = participant_data.loc[
                (participant_data["task"] == task)
                & (participant_data["episode_index"] == ep_idx),
                "human_share",
            ]

            if len(value) == 0:
                row_values.append(np.nan)
            else:
                row_values.append(float(value.iloc[0]))

        matrix.append(row_values)

    matrix = np.array(matrix, dtype=float)

    cmap = ListedColormap(HEATMAP_COLORS)
    cmap.set_bad("#eeeeee")

    masked_matrix = np.ma.masked_invalid(matrix)

    fig_width = max(10, len(episodes) * 0.75)
    fig_height = max(5.5, len(tasks) * 0.55)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        masked_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(np.arange(len(episodes)))
    ax.set_xticklabels(make_episode_labels(participant_data), rotation=0, ha="center", fontsize=8)

    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Task")

    ax.set_title(
        f"Task division evolution — {participant_id}\n"
        "Color = human share of each task"
    )

    # Minor grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(episodes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Phase boundary lines
    episode_phases = list(episodes["episode_phase"])
    for i in range(1, len(episode_phases)):
        if episode_phases[i] != episode_phases[i - 1]:
            ax.axvline(i - 0.5, color=GROUP_LINE_COLOR, linewidth=1.5, alpha=0.7)

    # Add values inside cells for readability
    for y in range(len(tasks)):
        for x in range(len(episodes)):
            val = matrix[y, x]
            if np.isfinite(val):
                ax.text(
                    x,
                    y,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if val > 0.55 else "white",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Human share", rotation=90)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["AI mostly", "Shared", "Human mostly"])

    if participant_data["was_skipped"].any():
        fig.text(
            0.01,
            0.01,
            "* skipped/incomplete episode",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_all_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = sorted(
        summary["participant_id"].unique(),
        key=participant_sort_key,
    )

    for participant_id in participant_ids:
        participant_data = summary.loc[summary["participant_id"] == participant_id].copy()

        output_path = output_dir / f"task_division_heatmap_{participant_id}.png"

        plot_task_division_heatmap(
            participant_data=participant_data,
            participant_id=participant_id,
            output_path=output_path,
        )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create task division evolution heatmaps."
    )

    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Folders or task_division_summary.csv files to include. If absent, DATA_SOURCES is used.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. If absent, OUTPUT_DIR is used.",
    )

    parser.add_argument(
        "--include-solo",
        action="store_true",
        help="Include solo episodes in the heatmap. Default: excluded.",
    )

    return parser.parse_args()


def main() -> None:
    global INCLUDE_PHASES

    args = parse_args()

    if args.include_solo and "solo" not in INCLUDE_PHASES:
        INCLUDE_PHASES = ["seed", "bo", "solo", "bo_replay_best", "stress", "replay_optimal"]

    data_sources = [Path(p) for p in args.data] if args.data else DATA_SOURCES
    output_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Task division evolution analysis ===")

    print("\nData sources:")
    for src in data_sources:
        print(f"  - {src}")

    task_files = find_task_division_files(data_sources)

    print(f"\nFound {len(task_files)} task_division_summary.csv files:")
    for f in task_files:
        print(f"  - {f}")

    task_data = load_task_division(task_files)
    skipped_flags = load_skipped_episode_flags(task_files)

    print("\nEpisode phases found:")
    print(task_data["episode_phase"].value_counts(dropna=False).to_string())

    summary = compute_human_share_by_episode(
        task_data=task_data,
        skipped_flags=skipped_flags,
    )

    summary.to_csv(output_dir / "task_division_evolution_long.csv", index=False)

    make_all_heatmaps(summary, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nCreated:")
    print("  - task_division_evolution_long.csv")
    print("  - task_division_heatmap_Thinpath_PXX.png for each participant")

    print("\nHow to read the heatmap:")
    print("  0.0 = AI mostly")
    print("  0.5 = shared")
    print("  1.0 = Human mostly")
    print("  gray = task not performed")
    print("  * = skipped/incomplete episode")

    print("\nDone.")


if __name__ == "__main__":
    main()
