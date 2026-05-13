#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis 4b: Policy distance and Role distance over time
--------------------------------------------------------

Goal:
    Show, episode by episode, whether the AI policy used by BO gets closer
    to the Best-policy and whether the observed Human-AI task division also
    gets closer to the Best-policy role split.

Why this version:
    The scatter plot policy-distance vs role-distance can be hard to read
    because it removes the temporal order. This script keeps the timeline.

Input expected:

    Overcooked_equilibrium/
        analyze_policy_role_distance_over_time.py
        tsnt_thinpath.csv
        submissions/
            Thinpath_P01/
                round_summary.csv
                task_division_summary.csv
            Thinpath_P02/
                round_summary.csv
                task_division_summary.csv
            ...

Output:
    analysis_outputs/policy_role_distance_over_time/
        policy_role_distance_over_time_long.csv
        policy_role_distance_over_time_participant_summary.csv
        policy_role_distance_over_time_Thinpath_P01.png
        policy_role_distance_over_time_Thinpath_P02.png
        ...
        policy_role_distance_over_time_all_participants.png

How to read the graph:

    X-axis:
        Episode timeline.

    Black solid line:
        Role distance from Best-policy.
        0 = same Human-AI task division as Best-policy.
        1 = very different task division.

    Turquoise dashed line:
        Policy distance from Best-policy in normalized t-SNE policy space.
        This is normalized to [0, 1] by the maximum possible distance in [-1, 1]^2.

    If both lines go down during BO:
        the AI policies are getting closer to the Best-policy and the observed
        task division is also becoming more similar to the Best-policy.

    If policy distance is low but role distance is high:
        a nearby policy produces a different collaboration pattern.
        This suggests fragility.

    X marker / * label:
        skipped or incomplete episode.

Notes:
    - Solo episodes are excluded by default because there is no meaningful
      Human-AI task division.
    - Best-policy is distance 0 by definition.
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

EMBEDDING_CSV = PROJECT_ROOT / "tsnt_thinpath.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "policy_role_distance_over_time"

PARTICIPANT_FOLDER_PATTERN = re.compile(r"^Thinpath_P\d+$", re.IGNORECASE)

INCLUDE_PHASES = [
    "seed",
    "bo",
    "bo_replay_best",
    "stress",
    "replay_optimal",
]

REFERENCE_PHASE = "bo_replay_best"

USE_REPLAY_OPTIMAL_AS_FALLBACK = True
FALLBACK_REFERENCE_PHASE = "replay_optimal"

# Maximum possible Euclidean distance in normalized embedding space [-1, 1] x [-1, 1].
MAX_NORMALIZED_EMBEDDING_DISTANCE = float(np.sqrt(8.0))

# Palette, aligned with previous figures.
COLOR_ROLE = "#222222"       # black
COLOR_POLICY = "#48BFE3"     # turquoise
COLOR_BEST = "#80FFDB"
COLOR_SKIPPED = "#FFB000"

PHASE_COLORS = {
    "seed": "#7400B8",
    "bo": "#5390D9",
    "bo_replay_best": "#80FFDB",
    "stress": "#48BFE3",
    "replay_optimal": "#6930C3",
    "solo": "#999999",
}

PARTICIPANT_COLORS = [
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

MISSING_TASK_SHARE = 0.5


# ============================================================
# HELPERS
# ============================================================

def is_valid_participant_folder(folder_name: str) -> bool:
    return bool(PARTICIPANT_FOLDER_PATTERN.match(folder_name))


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(participant_id))
    if match:
        return int(match.group(1)), str(participant_id)
    return 9999, str(participant_id)


def strip_policy_prefix(policy_name: str | None) -> str | None:
    if policy_name is None:
        return None

    p = str(policy_name).strip()

    if p == "" or p.lower() in {"none", "nan", "null"}:
        return None

    p = p.replace("\\", "/").split("/")[-1]

    if "agent0_" in p:
        p = p.split("agent0_", 1)[1]

    return p.strip()


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


def safe_mean(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").mean(skipna=True)


def safe_sum(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").sum(skipna=True)


def most_common_nonempty(series: pd.Series) -> str | None:
    values = [
        str(v).strip()
        for v in series.dropna().tolist()
        if str(v).strip() and str(v).strip().lower() not in {"none", "nan", "null"}
    ]

    if not values:
        return None

    return pd.Series(values).mode().iloc[0]


def find_files(data_sources: Iterable[Path], filename: str) -> list[Path]:
    files: list[Path] = []

    for src in data_sources:
        src = Path(src).expanduser()

        if not src.exists():
            print(f"[WARN] Path not found: {src}")
            continue

        if src.is_file() and src.name == filename:
            if is_valid_participant_folder(src.parent.name):
                files.append(src)

        elif src.is_dir():
            found = src.rglob(filename)
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


# ============================================================
# LOADING
# ============================================================

def load_round_summary(round_files: list[Path]) -> pd.DataFrame:
    frames = []

    for f in round_files:
        df = pd.read_csv(f)
        df["participant_id"] = f.parent.name
        df["source_file"] = str(f)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No round_summary.csv files found.")

    data = pd.concat(frames, ignore_index=True)

    required_cols = [
        "participant_id",
        "episode_index",
        "episode_phase",
        "policy_id",
    ]

    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column in round_summary.csv: {col}")

    data["episode_index"] = pd.to_numeric(data["episode_index"], errors="coerce")
    if "round_in_episode" in data.columns:
        data["round_in_episode"] = pd.to_numeric(data["round_in_episode"], errors="coerce")
    else:
        data["round_in_episode"] = np.nan

    data["episode_phase"] = data["episode_phase"].astype(str).str.strip()
    data["policy_id"] = data["policy_id"].astype(str).str.strip()
    data["policy_short"] = data["policy_id"].apply(strip_policy_prefix)

    numeric_cols = [
        "dishes_served",
        "human_steps",
        "ai_steps",
        "team_reward_score",
        "mental_demand",
        "performance_score",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "skipped_episode" in data.columns:
        data["was_skipped_round"] = (
            data["skipped_episode"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        data["was_skipped_round"] = False

    return data


def load_task_division(task_files: list[Path]) -> pd.DataFrame:
    frames = []

    for f in task_files:
        df = pd.read_csv(f)
        df["participant_id"] = f.parent.name
        df["source_file"] = str(f)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No task_division_summary.csv files found.")

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


def load_embedding(embedding_csv: Path) -> pd.DataFrame:
    embedding_csv = Path(embedding_csv).expanduser()

    if not embedding_csv.exists():
        raise FileNotFoundError(f"Embedding CSV not found: {embedding_csv}")

    df = pd.read_csv(embedding_csv)

    required = ["policy", "x", "y"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Embedding CSV must contain columns {required}. "
                f"Found columns={list(df.columns)}"
            )

    df = df.copy()
    df["policy"] = df["policy"].astype(str).str.strip()
    df["policy_short"] = df["policy"].apply(strip_policy_prefix)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Same normalization used by TSNEBayesOptimizer.
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    df["x_norm"] = 2 * (df["x"] - x_min) / x_range - 1
    df["y_norm"] = 2 * (df["y"] - y_min) / y_range - 1

    df = df.drop_duplicates(subset=["policy_short"], keep="first").reset_index(drop=True)

    return df


# ============================================================
# EPISODE SUMMARY
# ============================================================

def make_episode_summary(rounds: pd.DataFrame, include_solo: bool = False) -> pd.DataFrame:
    include_phases = INCLUDE_PHASES.copy()
    if include_solo and "solo" not in include_phases:
        include_phases.insert(2, "solo")

    selected = rounds.loc[rounds["episode_phase"].isin(include_phases)].copy()

    rows = []

    for (participant_id, episode_index), g in selected.groupby(["participant_id", "episode_index"]):
        policy_id = most_common_nonempty(g["policy_id"])
        policy_short = strip_policy_prefix(policy_id)

        rows.append({
            "participant_id": participant_id,
            "episode_index": int(episode_index),
            "episode_phase": most_common_nonempty(g["episode_phase"]),
            "policy_id": policy_id,
            "policy_short": policy_short,
            "was_skipped": bool(g["was_skipped_round"].any()),
            "n_rounds": g["round_in_episode"].nunique(),
            "total_dishes_served": safe_sum(g["dishes_served"]) if "dishes_served" in g.columns else np.nan,
            "mean_dishes_per_round": safe_mean(g["dishes_served"]) if "dishes_served" in g.columns else np.nan,
            "mean_team_reward_per_round": safe_mean(g["team_reward_score"]) if "team_reward_score" in g.columns else np.nan,
            "total_human_steps": safe_sum(g["human_steps"]) if "human_steps" in g.columns else np.nan,
            "total_ai_steps": safe_sum(g["ai_steps"]) if "ai_steps" in g.columns else np.nan,
            "mean_mental_demand": safe_mean(g["mental_demand"]) if "mental_demand" in g.columns else np.nan,
            "mean_subjective_performance": safe_mean(g["performance_score"]) if "performance_score" in g.columns else np.nan,
        })

    out = pd.DataFrame(rows)

    return out.sort_values(["participant_id", "episode_index"])


# ============================================================
# ROLE DISTANCE
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
    episode_summary: pd.DataFrame,
    include_solo: bool = False,
) -> pd.DataFrame:
    data = collapse_task_columns(task_data)

    include_phases = INCLUDE_PHASES.copy()
    if include_solo and "solo" not in include_phases:
        include_phases.insert(2, "solo")

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
        ep_idx = int(ep["episode_index"])
        ep_phase = ep["episode_phase"]

        ep_rows = grouped.loc[
            (grouped["participant_id"] == pid)
            & (grouped["episode_index"] == ep_idx)
            & (grouped["episode_phase"] == ep_phase)
        ]

        human_row = ep_rows.loc[ep_rows["agent"] == "human"]
        ai_row = ep_rows.loc[ep_rows["agent"] == "ai"]

        was_skipped = bool(
            episode_summary.loc[
                (episode_summary["participant_id"] == pid)
                & (episode_summary["episode_index"] == ep_idx),
                "was_skipped",
            ].any()
        )

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
                "episode_index": ep_idx,
                "episode_phase": ep_phase,
                "task": task,
                "human_count": human_count,
                "ai_count": ai_count,
                "total_count": total_count,
                "human_share": human_share,
                "was_skipped": was_skipped,
            })

    return pd.DataFrame(rows).sort_values(["participant_id", "episode_index", "task"])


def select_reference_episode(participant_share: pd.DataFrame) -> tuple[int | None, str | None]:
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

                if current_total <= 0 and ref_total <= 0:
                    continue

                if task in current.index and current_total > 0:
                    current_share = float(current.loc[task, "human_share"])
                else:
                    current_share = MISSING_TASK_SHARE

                if task in ref.index and ref_total > 0:
                    ref_share = float(ref.loc[task, "human_share"])
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
            })

    return pd.DataFrame(rows).sort_values(["participant_id", "episode_index"])


# ============================================================
# POLICY DISTANCE
# ============================================================

def compute_policy_distances(
    episode_summary: pd.DataFrame,
    role_distance: pd.DataFrame,
    embedding: pd.DataFrame,
) -> pd.DataFrame:
    emb = embedding.set_index("policy_short")

    merged = role_distance.merge(
        episode_summary,
        on=["participant_id", "episode_index", "episode_phase", "was_skipped"],
        how="left",
    )

    rows = []

    for participant_id, pdata in merged.groupby("participant_id"):
        ref_episode = int(pdata["reference_episode_index"].iloc[0])
        ref_row = pdata.loc[pdata["episode_index"] == ref_episode]

        if ref_row.empty:
            print(f"[WARN] No reference row for {participant_id}; skipping.")
            continue

        ref_policy_short = ref_row["policy_short"].iloc[0]

        if ref_policy_short not in emb.index:
            print(f"[WARN] Reference policy not found in embedding for {participant_id}: {ref_policy_short}")
            continue

        ref_x = float(emb.loc[ref_policy_short, "x_norm"])
        ref_y = float(emb.loc[ref_policy_short, "y_norm"])

        for _, row in pdata.iterrows():
            policy_short = row.get("policy_short")

            policy_distance = np.nan
            policy_distance_norm = np.nan
            policy_found = False

            if policy_short in emb.index:
                x = float(emb.loc[policy_short, "x_norm"])
                y = float(emb.loc[policy_short, "y_norm"])
                policy_distance = float(np.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2))
                policy_distance_norm = policy_distance / MAX_NORMALIZED_EMBEDDING_DISTANCE
                policy_found = True

            new_row = row.to_dict()
            new_row.update({
                "reference_policy_id": ref_row["policy_id"].iloc[0],
                "reference_policy_short": ref_policy_short,
                "policy_distance_from_best": policy_distance,
                "policy_distance_from_best_normalized": policy_distance_norm,
                "policy_found_in_embedding": policy_found,
            })

            rows.append(new_row)

    result = pd.DataFrame(rows)

    return result.sort_values(["participant_id", "episode_index"])


# ============================================================
# SUMMARY
# ============================================================

def compute_correlations(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    df = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if len(df) < 3:
        return np.nan, np.nan

    pearson = df["x"].corr(df["y"], method="pearson")
    spearman = df["x"].rank().corr(df["y"].rank(), method="pearson")

    return pearson, spearman


def make_participant_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for participant_id, pdata in long_df.groupby("participant_id"):
        non_ref = pdata.loc[
            pdata["episode_index"] != pdata["reference_episode_index"]
        ].copy()

        pearson, spearman = compute_correlations(
            non_ref["policy_distance_from_best_normalized"],
            non_ref["role_distance_from_best"],
        )

        bo = pdata.loc[pdata["episode_phase"] == "bo"]
        stress = pdata.loc[pdata["episode_phase"] == "stress"]

        rows.append({
            "participant_id": participant_id,
            "reference_episode_index": int(pdata["reference_episode_index"].iloc[0]),
            "reference_phase": pdata["reference_phase"].iloc[0],
            "reference_policy_short": pdata["reference_policy_short"].iloc[0],
            "reference_was_skipped": bool(pdata["reference_was_skipped"].any()),
            "n_episodes": pdata["episode_index"].nunique(),
            "pearson_policy_vs_role_distance_excluding_reference": pearson,
            "spearman_policy_vs_role_distance_excluding_reference": spearman,
            "mean_policy_distance_bo_normalized": safe_mean(bo["policy_distance_from_best_normalized"]) if not bo.empty else np.nan,
            "mean_role_distance_bo": safe_mean(bo["role_distance_from_best"]) if not bo.empty else np.nan,
            "mean_policy_distance_stress_normalized": safe_mean(stress["policy_distance_from_best_normalized"]) if not stress.empty else np.nan,
            "mean_role_distance_stress": safe_mean(stress["role_distance_from_best"]) if not stress.empty else np.nan,
            "mean_dishes_per_round_stress": safe_mean(stress["mean_dishes_per_round"]) if not stress.empty else np.nan,
        })

    return pd.DataFrame(rows).sort_values("participant_id", key=lambda s: s.map(participant_sort_key))


# ============================================================
# PLOTTING
# ============================================================

def episode_tick_labels(pdata: pd.DataFrame) -> list[str]:
    labels = []

    for _, row in pdata.sort_values("episode_index").iterrows():
        ep = int(row["episode_index"])
        phase = phase_short_label(row["episode_phase"])
        star = "*" if bool(row["was_skipped"]) else ""
        labels.append(f"E{ep}{star}\n{phase}")

    return labels


def plot_participant_over_time(
    long_df: pd.DataFrame,
    participant_id: str,
    output_path: Path,
) -> None:
    pdata = (
        long_df.loc[long_df["participant_id"] == participant_id]
        .sort_values("episode_index")
        .copy()
    )

    if pdata.empty:
        print(f"[WARN] No data for {participant_id}")
        return

    pdata = pdata.loc[pdata["policy_found_in_embedding"]].copy()

    if pdata.empty:
        print(f"[WARN] No embedding matches for {participant_id}")
        return

    x = np.arange(len(pdata))

    ref_ep = int(pdata["reference_episode_index"].iloc[0])
    ref_policy = str(pdata["reference_policy_short"].iloc[0])
    ref_skipped = bool(pdata["reference_was_skipped"].iloc[0])

    ref_pos = None
    for idx, ep in enumerate(pdata["episode_index"].tolist()):
        if int(ep) == ref_ep:
            ref_pos = idx
            break

    fig, ax = plt.subplots(figsize=(max(9.5, len(pdata) * 0.72), 5.8))

    # Main lines
    ax.plot(
        x,
        pdata["role_distance_from_best"],
        marker="o",
        linewidth=2.3,
        color=COLOR_ROLE,
        label="Role distance from Best-policy",
        zorder=3,
    )

    ax.plot(
        x,
        pdata["policy_distance_from_best_normalized"],
        marker="s",
        linewidth=2.3,
        linestyle="--",
        color=COLOR_POLICY,
        label="Policy distance from Best-policy",
        zorder=3,
    )

    # Skipped markers
    skipped = pdata.loc[pdata["was_skipped"]]
    if not skipped.empty:
        skipped_positions = [pdata.index.get_loc(idx) for idx in skipped.index]

        ax.scatter(
            skipped_positions,
            skipped["role_distance_from_best"],
            marker="X",
            s=140,
            color=COLOR_SKIPPED,
            edgecolor="#111111",
            linewidth=0.9,
            label="Skipped/incomplete",
            zorder=6,
        )

        ax.scatter(
            skipped_positions,
            skipped["policy_distance_from_best_normalized"],
            marker="X",
            s=140,
            color=COLOR_SKIPPED,
            edgecolor="#111111",
            linewidth=0.9,
            zorder=6,
        )

    # Best-policy vertical line
    if ref_pos is not None:
        ax.axvline(
            ref_pos,
            color=COLOR_BEST,
            linewidth=2,
            linestyle=":",
            alpha=0.9,
            label="Best-policy reference",
            zorder=1,
        )

    # Phase boundary lines
    phases = pdata["episode_phase"].tolist()
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(i - 0.5, color="#BBBBBB", linewidth=1.0, alpha=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(episode_tick_labels(pdata), fontsize=8)

    ax.set_ylim(-0.05, 1.05)

    ax.set_ylabel("Distance from Best-policy\n0 = same/near, 1 = far/different")
    ax.set_xlabel("Episode")

    title = (
        f"Policy and role distance over time — {participant_id}\n"
        f"Best-policy reference: E{ref_ep}, {ref_policy}"
    )

    if ref_skipped:
        title += " (reference skipped/incomplete)"

    ax.set_title(title)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    note = (
        "Black line = observed Human-AI role distance. "
        "Turquoise dashed line = AI policy distance in normalized t-SNE space. "
        "* / X = skipped or incomplete."
    )
    fig.text(0.01, 0.01, note, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_all_participants(long_df: pd.DataFrame, output_path: Path) -> None:
    data = long_df.loc[long_df["policy_found_in_embedding"]].copy()

    if data.empty:
        print("[WARN] No data for combined plot.")
        return

    participant_ids = sorted(data["participant_id"].unique(), key=participant_sort_key)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for i, pid in enumerate(participant_ids):
        pdata = data.loc[data["participant_id"] == pid].sort_values("episode_index").copy()
        color = PARTICIPANT_COLORS[i % len(PARTICIPANT_COLORS)]

        ax.plot(
            pdata["episode_index"],
            pdata["role_distance_from_best"],
            marker="o",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=f"{pid} role",
        )

        ax.plot(
            pdata["episode_index"],
            pdata["policy_distance_from_best_normalized"],
            marker="s",
            linestyle="--",
            linewidth=1.7,
            color=color,
            alpha=0.45,
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Distance from Best-policy\nsolid = role, dashed = policy")
    ax.set_title(
        "Policy and role distance over time across participants\n"
        "Solid lines = role distance; dashed lines = policy distance"
    )

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_all_plots(long_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = sorted(long_df["participant_id"].unique(), key=participant_sort_key)

    for participant_id in participant_ids:
        plot_participant_over_time(
            long_df=long_df,
            participant_id=participant_id,
            output_path=output_dir / f"policy_role_distance_over_time_{participant_id}.png",
        )

    plot_all_participants(
        long_df=long_df,
        output_path=output_dir / "policy_role_distance_over_time_all_participants.png",
    )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot policy distance and role distance from Best-policy over time."
    )

    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Folders to include. If absent, DATA_SOURCES is used.",
    )

    parser.add_argument(
        "--embedding",
        default=None,
        help="Path to t-SNE embedding CSV. Default: tsnt_thinpath.csv in project root.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. If absent, OUTPUT_DIR is used.",
    )

    parser.add_argument(
        "--participant",
        default=None,
        help="Optional participant ID to analyze only one participant, e.g. Thinpath_P04.",
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
    embedding_csv = Path(args.embedding).expanduser() if args.embedding else EMBEDDING_CSV
    output_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Policy and Role distance over time analysis ===")

    print("\nData sources:")
    for src in data_sources:
        print(f"  - {src}")

    print(f"\nEmbedding CSV:\n  - {embedding_csv}")

    round_files = find_files(data_sources, "round_summary.csv")
    task_files = find_files(data_sources, "task_division_summary.csv")

    print(f"\nFound {len(round_files)} round_summary.csv files")
    print(f"Found {len(task_files)} task_division_summary.csv files")

    rounds = load_round_summary(round_files)
    task_data = load_task_division(task_files)
    embedding = load_embedding(embedding_csv)

    if args.participant:
        rounds = rounds.loc[rounds["participant_id"] == args.participant].copy()
        task_data = task_data.loc[task_data["participant_id"] == args.participant].copy()

        if rounds.empty or task_data.empty:
            raise ValueError(f"No data found for participant {args.participant}")

    print("\nEpisode phases found:")
    print(rounds["episode_phase"].value_counts(dropna=False).to_string())

    episode_summary = make_episode_summary(
        rounds=rounds,
        include_solo=args.include_solo,
    )

    share_data = compute_human_share_by_episode(
        task_data=task_data,
        episode_summary=episode_summary,
        include_solo=args.include_solo,
    )

    role_distance = compute_role_distance_from_best(share_data)

    long_df = compute_policy_distances(
        episode_summary=episode_summary,
        role_distance=role_distance,
        embedding=embedding,
    )

    participant_summary = make_participant_summary(long_df)

    long_df.to_csv(output_dir / "policy_role_distance_over_time_long.csv", index=False)
    participant_summary.to_csv(
        output_dir / "policy_role_distance_over_time_participant_summary.csv",
        index=False,
    )

    make_all_plots(long_df, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nCreated:")
    print("  - policy_role_distance_over_time_long.csv")
    print("  - policy_role_distance_over_time_participant_summary.csv")
    print("  - policy_role_distance_over_time_Thinpath_PXX.png")
    print("  - policy_role_distance_over_time_all_participants.png")

    print("\nParticipant summary:")
    display_cols = [
        "participant_id",
        "reference_episode_index",
        "reference_policy_short",
        "reference_was_skipped",
        "spearman_policy_vs_role_distance_excluding_reference",
        "mean_policy_distance_bo_normalized",
        "mean_role_distance_bo",
        "mean_policy_distance_stress_normalized",
        "mean_role_distance_stress",
    ]
    existing = [c for c in display_cols if c in participant_summary.columns]
    print(participant_summary[existing].to_string(index=False))

    missing = long_df.loc[~long_df["policy_found_in_embedding"]]
    if not missing.empty:
        print("\n[WARN] Some policies were not found in the embedding CSV:")
        print(
            missing[
                ["participant_id", "episode_index", "episode_phase", "policy_id", "policy_short"]
            ].drop_duplicates().to_string(index=False)
        )

    print("\nHow to read:")
    print("  Black solid line = role distance from Best-policy.")
    print("  Turquoise dashed line = policy distance from Best-policy.")
    print("  Both are on 0-1 scale.")
    print("  If both go down during BO, policy search and role division are converging.")
    print("  If policy distance is low but role distance is high, nearby policies produce different roles.")
    print("  X marker / * label = skipped/incomplete episode.")

    print("\nDone.")


if __name__ == "__main__":
    main()
