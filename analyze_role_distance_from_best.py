#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis 3: Role distance from Best-policy
------------------------------------------

Goal:
    Test whether the Human-AI task division during the experiment moves toward
    the task division observed in the Best-policy replay.

Main question:
    During BO, does the collaboration become more similar to the final
    Best-policy role split?

Input expected:

    Overcooked_equilibrium/
        analyze_role_distance_from_best.py
        submissions/
            Thinpath_P01/
                task_division_summary.csv
                round_summary.csv          optional, used to mark skipped episodes
            Thinpath_P02/
                task_division_summary.csv
                round_summary.csv
            ...

Output:
    analysis_outputs/role_distance_from_best/
        role_distance_from_best_long.csv
        role_distance_Thinpath_P01.png
        role_distance_Thinpath_P02.png
        ...
        role_distance_all_participants.png

How to read the graph:
    Y-axis = distance from the participant's Best-policy task division.

        0.0 = same task division as Best-policy
        1.0 = very different task division

    Lower values mean the episode's Human-AI role split is more similar
    to the Best-policy role split.

Important:
    If the Best-policy episode was skipped/incomplete, the reference itself
    should be interpreted with caution. The script marks skipped episodes with X.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_SOURCES = [
    PROJECT_ROOT / "submissions",
]

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "role_distance_from_best"

PARTICIPANT_FOLDER_PATTERN = re.compile(r"^Thinpath_P\d+$", re.IGNORECASE)

# We normally exclude solo because there is no Human-AI task division there.
INCLUDE_PHASES = [
    "seed",
    "bo",
    "bo_replay_best",
    "stress",
    "replay_optimal",
]

REFERENCE_PHASE = "bo_replay_best"

# If a participant has no bo_replay_best, use replay_optimal instead.
USE_REPLAY_OPTIMAL_AS_FALLBACK = True
FALLBACK_REFERENCE_PHASE = "replay_optimal"

# Gradient Blues palette
LINE_COLORS = [
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

PHASE_COLORS = {
    "seed": "#7400B8",
    "bo": "#5390D9",
    "bo_replay_best": "#80FFDB",
    "stress": "#48BFE3",
    "replay_optimal": "#6930C3",
    "solo": "#999999",
}

GROUP_LINE_COLOR = "#222222"

# Clear task groups. Raw columns that are not present are ignored.
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

# If a task is present in Best-policy but absent in another episode,
# we set that episode's role share to 0.5 = undefined/shared.
# This avoids treating "not observed" as fully Human or fully AI.
MISSING_TASK_SHARE = 0.5


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
    frames = []

    for f in task_files:
        df = pd.read_csv(f)
        df["participant_id"] = f.parent.name
        df["source_file"] = str(f)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No task_division_summary.csv files found. "
            "Check participant folders and DATA_SOURCES."
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

    data["episode_index"] = pd.to_numeric(data["episode_index"], errors="coerce")
    data["episode_phase"] = data["episode_phase"].astype(str).str.strip()
    data["agent"] = data["agent"].astype(str).str.strip().str.lower()

    for raw_cols in TASK_DEFINITIONS.values():
        for col in raw_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    return data


def load_skipped_episode_flags(task_files: list[Path]) -> pd.DataFrame:
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
# TASK SHARE COMPUTATION
# ============================================================

def collapse_task_columns(data: pd.DataFrame) -> pd.DataFrame:
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
    include_solo: bool = False,
) -> pd.DataFrame:
    """
    Compute Human share for every participant x episode x task.

    Human share:
        human_count / (human_count + ai_count)
    """
    data = collapse_task_columns(task_data)

    include_phases = INCLUDE_PHASES.copy()
    if include_solo and "solo" not in include_phases:
        include_phases.insert(2, "solo")

    if include_phases:
        data = data.loc[data["episode_phase"].isin(include_phases)].copy()

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

    return result.sort_values(["participant_id", "episode_index", "task"])


# ============================================================
# ROLE DISTANCE
# ============================================================

def select_reference_episode(participant_share: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Select the episode used as Best-policy reference.
    Default: bo_replay_best.
    Optional fallback: replay_optimal.
    """
    ref_rows = participant_share.loc[participant_share["episode_phase"] == REFERENCE_PHASE]

    if not ref_rows.empty:
        ref_episode = int(ref_rows["episode_index"].min())
        return ref_episode, REFERENCE_PHASE

    if USE_REPLAY_OPTIMAL_AS_FALLBACK:
        fallback_rows = participant_share.loc[
            participant_share["episode_phase"] == FALLBACK_REFERENCE_PHASE
        ]

        if not fallback_rows.empty:
            ref_episode = int(fallback_rows["episode_index"].min())
            return ref_episode, FALLBACK_REFERENCE_PHASE

    return None, None


def compute_role_distance_from_best(share_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each participant and episode, compute distance from the participant's
    Best-policy task-division vector.

    Distance for each task:
        abs(current_human_share - reference_human_share) * 2

    Range:
        0 = same role split
        1 = maximally different/opposite role split

    Episode distance:
        average across tasks that are observed in either the current episode
        or the reference episode.
    """
    rows = []

    task_labels = list(TASK_DEFINITIONS.keys())

    for participant_id, pdata in share_data.groupby("participant_id"):
        ref_episode, ref_phase = select_reference_episode(pdata)

        if ref_episode is None:
            print(f"[WARN] No reference episode found for {participant_id}; skipping.")
            continue

        ref = pdata.loc[pdata["episode_index"] == ref_episode].set_index("task")

        ref_was_skipped = bool(
            pdata.loc[pdata["episode_index"] == ref_episode, "was_skipped"].any()
        )

        episodes = (
            pdata[["episode_index", "episode_phase", "was_skipped"]]
            .drop_duplicates()
            .sort_values("episode_index")
        )

        for _, ep in episodes.iterrows():
            ep_idx = int(ep["episode_index"])
            ep_phase = ep["episode_phase"]
            was_skipped = bool(ep["was_skipped"])

            current = pdata.loc[pdata["episode_index"] == ep_idx].set_index("task")

            task_distances = []
            task_weights = []
            n_tasks_compared = 0
            n_tasks_current_observed = 0
            n_tasks_reference_observed = 0

            for task in task_labels:
                if task not in current.index and task not in ref.index:
                    continue

                current_total = (
                    float(current.loc[task, "total_count"])
                    if task in current.index
                    else 0.0
                )

                ref_total = (
                    float(ref.loc[task, "total_count"])
                    if task in ref.index
                    else 0.0
                )

                # If the task is absent in both current and reference, ignore it.
                if current_total <= 0 and ref_total <= 0:
                    continue

                if task in current.index and current_total > 0:
                    current_share = float(current.loc[task, "human_share"])
                    n_tasks_current_observed += 1
                else:
                    current_share = MISSING_TASK_SHARE

                if task in ref.index and ref_total > 0:
                    ref_share = float(ref.loc[task, "human_share"])
                    n_tasks_reference_observed += 1
                else:
                    ref_share = MISSING_TASK_SHARE

                dist = abs(current_share - ref_share) * 2
                dist = min(max(dist, 0.0), 1.0)

                weight = current_total + ref_total

                task_distances.append(dist)
                task_weights.append(weight)
                n_tasks_compared += 1

            if n_tasks_compared == 0:
                role_distance = np.nan
            else:
                task_distances = np.array(task_distances, dtype=float)
                task_weights = np.array(task_weights, dtype=float)

                if np.nansum(task_weights) > 0:
                    role_distance = np.average(task_distances, weights=task_weights)
                else:
                    role_distance = np.nanmean(task_distances)

            rows.append({
                "participant_id": participant_id,
                "episode_index": ep_idx,
                "episode_phase": ep_phase,
                "was_skipped": was_skipped,
                "reference_episode_index": ref_episode,
                "reference_phase": ref_phase,
                "reference_was_skipped": ref_was_skipped,
                "role_distance_from_best": role_distance,
                "n_tasks_compared": n_tasks_compared,
                "n_tasks_current_observed": n_tasks_current_observed,
                "n_tasks_reference_observed": n_tasks_reference_observed,
            })

    result = pd.DataFrame(rows)

    return result.sort_values(["participant_id", "episode_index"])


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


def episode_tick_labels(pdata: pd.DataFrame) -> list[str]:
    labels = []

    for _, row in pdata.sort_values("episode_index").iterrows():
        ep = int(row["episode_index"])
        phase = phase_short_label(row["episode_phase"])
        star = "*" if bool(row["was_skipped"]) else ""
        labels.append(f"E{ep}{star}\n{phase}")

    return labels


def plot_participant_role_distance(
    distance_data: pd.DataFrame,
    participant_id: str,
    output_path: Path,
) -> None:
    pdata = (
        distance_data.loc[distance_data["participant_id"] == participant_id]
        .sort_values("episode_index")
        .copy()
    )

    if pdata.empty:
        print(f"[WARN] No role-distance data for {participant_id}")
        return

    x = np.arange(len(pdata))
    y = pdata["role_distance_from_best"].values

    ref_episode = int(pdata["reference_episode_index"].iloc[0])
    ref_phase = str(pdata["reference_phase"].iloc[0])
    ref_was_skipped = bool(pdata["reference_was_skipped"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(9, len(pdata) * 0.65), 5.2))

    # Main line
    ax.plot(
        x,
        y,
        color=GROUP_LINE_COLOR,
        linewidth=2,
        alpha=0.7,
        zorder=2,
    )

    # Points colored by phase
    legend_added = set()

    for i, (_, row) in enumerate(pdata.iterrows()):
        phase = row["episode_phase"]
        color = PHASE_COLORS.get(phase, "#999999")
        marker = "X" if bool(row["was_skipped"]) else "o"
        size = 105 if marker == "X" else 65
        label = phase_short_label(phase)

        if label in legend_added:
            label = None
        else:
            legend_added.add(label)

        ax.scatter(
            x[i],
            row["role_distance_from_best"],
            color=color,
            edgecolor="#111111",
            linewidth=0.8,
            marker=marker,
            s=size,
            zorder=4,
            label=label,
        )

    # Highlight reference episode
    ref_positions = pdata.index[pdata["episode_index"] == ref_episode].tolist()
    if ref_positions:
        ref_pos = list(pdata["episode_index"]).index(ref_episode)
        ax.axvline(
            ref_pos,
            color="#80FFDB",
            linewidth=2.0,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )

    ax.axhline(0, color="#111111", linewidth=1.0, alpha=0.6)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(episode_tick_labels(pdata), fontsize=8)
    ax.set_ylabel("Role distance from Best-policy\n0 = same role split, 1 = very different")
    ax.set_xlabel("Episode")

    title = (
        f"Role distance from Best-policy — {participant_id}\n"
        f"Reference: E{ref_episode} ({phase_short_label(ref_phase)})"
    )

    if ref_was_skipped:
        title += " — reference skipped/incomplete"

    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    if pdata["was_skipped"].any():
        fig.text(
            0.01,
            0.01,
            "* skipped/incomplete episode; X marker = skipped/incomplete",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_all_participants_role_distance(
    distance_data: pd.DataFrame,
    output_path: Path,
) -> None:
    participant_ids = sorted(
        distance_data["participant_id"].unique(),
        key=participant_sort_key,
    )

    if not participant_ids:
        print("[WARN] No participants for combined role-distance plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, pid in enumerate(participant_ids):
        pdata = (
            distance_data.loc[distance_data["participant_id"] == pid]
            .sort_values("episode_index")
            .copy()
        )

        color = LINE_COLORS[i % len(LINE_COLORS)]

        ax.plot(
            pdata["episode_index"],
            pdata["role_distance_from_best"],
            marker="o",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=pid,
        )

        # Mark skipped episodes
        skipped = pdata.loc[pdata["was_skipped"]]
        if not skipped.empty:
            ax.scatter(
                skipped["episode_index"],
                skipped["role_distance_from_best"],
                marker="X",
                s=95,
                color=color,
                edgecolor="#111111",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Role distance from Best-policy\n0 = same role split, 1 = very different")
    ax.set_title(
        "Role distance from Best-policy across participants\n"
        "Lower values mean the task division is closer to the Best-policy role split"
    )

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_all_plots(distance_data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = sorted(
        distance_data["participant_id"].unique(),
        key=participant_sort_key,
    )

    for participant_id in participant_ids:
        plot_participant_role_distance(
            distance_data=distance_data,
            participant_id=participant_id,
            output_path=output_dir / f"role_distance_{participant_id}.png",
        )

    plot_all_participants_role_distance(
        distance_data=distance_data,
        output_path=output_dir / "role_distance_all_participants.png",
    )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze role distance from Best-policy task division."
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
        "--participant",
        default=None,
        help="Optional participant ID to plot only one participant, e.g. Thinpath_P04.",
    )

    parser.add_argument(
        "--include-solo",
        action="store_true",
        help="Include solo episodes. Default: excluded.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_sources = [Path(p) for p in args.data] if args.data else DATA_SOURCES
    output_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Role distance from Best-policy analysis ===")

    print("\nData sources:")
    for src in data_sources:
        print(f"  - {src}")

    task_files = find_task_division_files(data_sources)

    print(f"\nFound {len(task_files)} task_division_summary.csv files:")
    for f in task_files:
        print(f"  - {f}")

    task_data = load_task_division(task_files)

    if args.participant:
        task_data = task_data.loc[task_data["participant_id"] == args.participant].copy()

        if task_data.empty:
            raise ValueError(f"No task_division_summary data found for {args.participant}")

    skipped_flags = load_skipped_episode_flags(task_files)

    if args.participant and not skipped_flags.empty:
        skipped_flags = skipped_flags.loc[
            skipped_flags["participant_id"] == args.participant
        ].copy()

    print("\nEpisode phases found:")
    print(task_data["episode_phase"].value_counts(dropna=False).to_string())

    share_data = compute_human_share_by_episode(
        task_data=task_data,
        skipped_flags=skipped_flags,
        include_solo=args.include_solo,
    )

    distance_data = compute_role_distance_from_best(share_data)

    distance_data.to_csv(output_dir / "role_distance_from_best_long.csv", index=False)

    make_all_plots(distance_data, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nCreated:")
    print("  - role_distance_from_best_long.csv")
    print("  - role_distance_Thinpath_PXX.png")
    print("  - role_distance_all_participants.png")

    print("\nHow to read:")
    print("  0.0 = same task division as Best-policy")
    print("  1.0 = very different task division")
    print("  Lower values during BO suggest convergence toward the Best-policy role split.")
    print("  X marker = skipped/incomplete episode.")

    print("\nDone.")


if __name__ == "__main__":
    main()
