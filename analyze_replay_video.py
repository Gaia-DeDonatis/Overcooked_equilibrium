#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay animation from Overcooked experiment logs
------------------------------------------------

Goal:
    Create a simple replay animation for one participant, one episode, and one round.

What it does:
    - Reads a participant JSON log containing `rounds` and `action_log`.
    - Selects a specific episode and round.
    - Animates Human and AI positions over time.
    - Shows each agent's current action and held object.
    - Marks skipped/incomplete episodes if that information is available.
    - Saves the animation as MP4 if ffmpeg is available, otherwise GIF.

Important:
    This does NOT recreate the exact browser/game video.
    It creates a clean analytical replay from the logged positions and actions.

Expected folder structure:

    Overcooked_equilibrium/
        analyze_replay_video.py.py
        submissions/
            Thinpath_P06/
                some_log_file.json
                round_summary.csv       optional

Example usage:

    python analyze_replay_video.py.py --participant Thinpath_P06 --episode 13 --round 1

    python analyze_replay_video.py.py --participant Thinpath_P04 --episode 13 --round 1 --fps 8

    python analyze_replay_video.py.py \
        --participant-folder submissions/Thinpath_P06 \
        --episode 13 \
        --round 1 \
        --out analysis_outputs/replay_videos

If the script finds more than one JSON file in the participant folder, it tries to pick
the one that contains a top-level `rounds` list.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle, FancyArrowPatch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "replay_videos"

# Same palette family used in the other analysis.
PALETTE = {
    "human": "#80FFDB",
    "ai": "#7400B8",
    "human_trail": "#48BFE3",
    "ai_trail": "#6930C3",
    "grid": "#DDDDDD",
    "wall": "#222222",
    "text": "#111111",
    "background": "#FFFFFF",
    "event": "#FFB000",
}

# If the map is unknown, the script estimates the grid from observed positions.
# These are extra cells around the min/max observed positions.
GRID_PADDING = 1

# Show the last N positions as a trail.
TRAIL_LENGTH = 12


# ============================================================
# LOG LOADING
# ============================================================

def find_participant_folder(participant: str | None, participant_folder: str | None) -> Path:
    if participant_folder:
        folder = Path(participant_folder).expanduser()
        if not folder.exists():
            raise FileNotFoundError(f"Participant folder not found: {folder}")
        return folder

    if not participant:
        raise ValueError("Provide either --participant or --participant-folder.")

    folder = SUBMISSIONS_DIR / participant
    if not folder.exists():
        raise FileNotFoundError(
            f"Participant folder not found: {folder}\n"
            "Use --participant-folder if the data is somewhere else."
        )

    return folder


def looks_like_experiment_log(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    return isinstance(data, dict) and isinstance(data.get("rounds"), list)


def find_json_log(participant_folder: Path, json_file: str | None = None) -> Path:
    if json_file:
        path = Path(json_file).expanduser()
        if not path.exists():
            # Try relative to participant folder
            path = participant_folder / json_file
        if not path.exists():
            raise FileNotFoundError(f"JSON log not found: {json_file}")
        return path

    json_files = sorted(participant_folder.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {participant_folder}")

    valid = [p for p in json_files if looks_like_experiment_log(p)]

    if not valid:
        raise FileNotFoundError(
            f"No JSON file with a top-level `rounds` list found in {participant_folder}."
        )

    if len(valid) > 1:
        print("[WARN] Multiple candidate JSON logs found. Using the first:")
        for p in valid:
            print(f"  - {p}")

    return valid[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_round_summary(participant_folder: Path) -> pd.DataFrame | None:
    path = participant_folder / "round_summary.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read round_summary.csv: {exc}")
        return None

    return df


# ============================================================
# ROUND SELECTION
# ============================================================

def select_round(
    log_data: dict[str, Any],
    episode_index: int,
    round_in_episode: int | None = None,
    global_round: int | None = None,
) -> dict[str, Any]:
    rounds = log_data.get("rounds", [])

    if not isinstance(rounds, list) or not rounds:
        raise ValueError("The JSON log does not contain a non-empty `rounds` list.")

    candidates = []

    for r in rounds:
        if not isinstance(r, dict):
            continue

        if global_round is not None:
            if int_or_none(r.get("round_index_global")) == global_round:
                candidates.append(r)
        else:
            if int_or_none(r.get("episode_index")) != episode_index:
                continue

            if round_in_episode is not None:
                if int_or_none(r.get("round_in_episode")) != round_in_episode:
                    continue

            candidates.append(r)

    if not candidates:
        if global_round is not None:
            raise ValueError(f"No round found with round_index_global={global_round}")
        raise ValueError(
            f"No round found for episode_index={episode_index}, "
            f"round_in_episode={round_in_episode}"
        )

    if len(candidates) > 1:
        print("[WARN] More than one matching round found. Using the first.")

    return candidates[0]


def int_or_none(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except Exception:
        return None


def get_episode_metadata(log_data: dict[str, Any], episode_index: int) -> dict[str, Any]:
    episodes = log_data.get("episodes", [])
    if not isinstance(episodes, list):
        return {}

    for ep in episodes:
        if int_or_none(ep.get("episode_index")) == episode_index:
            return ep

    return {}


# ============================================================
# ACTION LOG PROCESSING
# ============================================================

def normalize_holding(holding: Any) -> str:
    if holding is None or (isinstance(holding, float) and math.isnan(holding)):
        return "empty"

    if isinstance(holding, str):
        return holding

    if isinstance(holding, dict):
        item = holding.get("item")
        containing = holding.get("containing")

        if item is None:
            return "empty"

        if containing:
            return f"{item}({containing})"

        return str(item)

    return str(holding)


def normalize_action_entry(entry: dict[str, Any], agent: str) -> dict[str, Any]:
    pos = entry.get("pos", [np.nan, np.nan])

    if not isinstance(pos, list) or len(pos) < 2:
        pos = [np.nan, np.nan]

    # In the logs, pos appears as [row, col].
    row = float(pos[0])
    col = float(pos[1])

    if agent == "human":
        action = entry.get("action") or entry.get("key") or ""
    else:
        action = entry.get("arrow") or entry.get("low") or ""

    return {
        "t": int_or_none(entry.get("t")),
        "wall_ms": float(entry.get("wall_ms", np.nan)),
        "row": row,
        "col": col,
        "action": str(action),
        "holding": normalize_holding(entry.get("holding")),
        "agent": agent,
        "raw": entry,
    }


def round_to_timeline(round_data: dict[str, Any]) -> pd.DataFrame:
    action_log = round_data.get("action_log", {})

    if not isinstance(action_log, dict):
        raise ValueError("Selected round does not contain an `action_log` dictionary.")

    human_log = action_log.get("human", [])
    ai_log = action_log.get("ai", [])

    if not isinstance(human_log, list):
        human_log = []
    if not isinstance(ai_log, list):
        ai_log = []

    human = [normalize_action_entry(e, "human") for e in human_log if isinstance(e, dict)]
    ai = [normalize_action_entry(e, "ai") for e in ai_log if isinstance(e, dict)]

    if not human and not ai:
        raise ValueError("Selected round has empty human and AI action logs.")

    df = pd.DataFrame(human + ai)

    if "t" not in df.columns or df["t"].isna().all():
        # fallback: frame index by order
        df["t"] = df.groupby("agent").cumcount() + 1

    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
    df["wall_ms"] = pd.to_numeric(df["wall_ms"], errors="coerce")

    # Create one row per timestep with separate human/AI fields.
    timesteps = sorted(df["t"].dropna().unique())

    rows = []

    last_state = {
        "human": None,
        "ai": None,
    }

    for t in timesteps:
        step = {"t": int(t)}

        for agent in ["human", "ai"]:
            agent_rows = df.loc[(df["agent"] == agent) & (df["t"] == t)]

            if not agent_rows.empty:
                state = agent_rows.iloc[-1].to_dict()
                last_state[agent] = state
            else:
                state = last_state[agent]

            if state is None:
                step[f"{agent}_row"] = np.nan
                step[f"{agent}_col"] = np.nan
                step[f"{agent}_action"] = ""
                step[f"{agent}_holding"] = "empty"
                step[f"{agent}_wall_ms"] = np.nan
            else:
                step[f"{agent}_row"] = state["row"]
                step[f"{agent}_col"] = state["col"]
                step[f"{agent}_action"] = state["action"]
                step[f"{agent}_holding"] = state["holding"]
                step[f"{agent}_wall_ms"] = state["wall_ms"]

        rows.append(step)

    timeline = pd.DataFrame(rows)

    return timeline


def infer_grid_bounds(timeline: pd.DataFrame) -> tuple[int, int, int, int]:
    rows = pd.concat([
        timeline["human_row"],
        timeline["ai_row"],
    ], ignore_index=True).dropna()

    cols = pd.concat([
        timeline["human_col"],
        timeline["ai_col"],
    ], ignore_index=True).dropna()

    if rows.empty or cols.empty:
        return 0, 4, 0, 6

    min_row = int(np.floor(rows.min())) - GRID_PADDING
    max_row = int(np.ceil(rows.max())) + GRID_PADDING
    min_col = int(np.floor(cols.min())) - GRID_PADDING
    max_col = int(np.ceil(cols.max())) + GRID_PADDING

    min_row = min(min_row, 0)
    min_col = min(min_col, 0)

    return min_row, max_row, min_col, max_col


# ============================================================
# DRAWING
# ============================================================

def draw_grid(ax, bounds: tuple[int, int, int, int]) -> None:
    min_row, max_row, min_col, max_col = bounds

    # Draw cells
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=PALETTE["background"],
                edgecolor=PALETTE["grid"],
                linewidth=0.8,
            )
            ax.add_patch(rect)

    ax.set_xlim(min_col - 0.5, max_col + 0.5)
    ax.set_ylim(max_row + 0.5, min_row - 0.5)  # invert y-axis: row 0 at top
    ax.set_aspect("equal")

    ax.set_xticks(range(min_col, max_col + 1))
    ax.set_yticks(range(min_row, max_row + 1))
    ax.tick_params(labelsize=8)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def make_round_title(
    participant_id: str,
    round_data: dict[str, Any],
    episode_meta: dict[str, Any],
) -> str:
    ep = round_data.get("episode_index")
    rin = round_data.get("round_in_episode")
    glob = round_data.get("round_index_global")
    phase = round_data.get("episode_phase", episode_meta.get("episode_phase", ""))
    policy = round_data.get("policy_id", episode_meta.get("policy_id", ""))

    skipped = episode_meta.get("skipped", None)
    skipped_text = " | SKIPPED/INCOMPLETE" if skipped is True else ""

    title = (
        f"{participant_id} — Episode {ep}, Round {rin} "
        f"(global {glob}) — {phase}{skipped_text}\n"
        f"Policy: {policy}"
    )

    return title


def format_summary(round_data: dict[str, Any]) -> str:
    summary = round_data.get("summary", {})
    if not isinstance(summary, dict):
        return ""

    dishes = summary.get("dishes_served", "NA")
    human_steps = summary.get("human_steps", "NA")
    ai_steps = summary.get("ai_steps", "NA")
    team_reward = summary.get("team_reward_score", "NA")

    return (
        f"Dishes: {dishes} | "
        f"Human steps: {human_steps} | "
        f"AI steps: {ai_steps} | "
        f"Team reward: {team_reward}"
    )


def add_direction_arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    if not np.isfinite([x0, y0, x1, y1]).all():
        return

    if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
        return

    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        alpha=0.8,
        zorder=6,
    )
    ax.add_patch(arrow)


def create_animation(
    timeline: pd.DataFrame,
    participant_id: str,
    round_data: dict[str, Any],
    episode_meta: dict[str, Any],
    output_path: Path,
    fps: int = 8,
    show_trails: bool = True,
) -> Path:
    bounds = infer_grid_bounds(timeline)

    fig, ax = plt.subplots(figsize=(8, 6))

    title = make_round_title(participant_id, round_data, episode_meta)
    summary_text = format_summary(round_data)

    # Place fixed title with smaller font.
    fig.suptitle(title, fontsize=10, y=0.98)

    info_text = fig.text(
        0.5,
        0.02,
        "",
        ha="center",
        va="bottom",
        fontsize=10,
        color=PALETTE["text"],
    )

    def update(frame_idx: int):
        ax.clear()
        draw_grid(ax, bounds)

        row = timeline.iloc[frame_idx]

        t = int(row["t"])

        human_x = row["human_col"]
        human_y = row["human_row"]
        ai_x = row["ai_col"]
        ai_y = row["ai_row"]

        # Trails
        if show_trails:
            start = max(0, frame_idx - TRAIL_LENGTH)
            recent = timeline.iloc[start:frame_idx + 1]

            ax.plot(
                recent["human_col"],
                recent["human_row"],
                color=PALETTE["human_trail"],
                linewidth=2,
                alpha=0.5,
                zorder=3,
            )

            ax.plot(
                recent["ai_col"],
                recent["ai_row"],
                color=PALETTE["ai_trail"],
                linewidth=2,
                alpha=0.5,
                zorder=3,
            )

        # Direction arrows from previous frame
        if frame_idx > 0:
            prev = timeline.iloc[frame_idx - 1]
            add_direction_arrow(
                ax,
                prev["human_col"],
                prev["human_row"],
                human_x,
                human_y,
                PALETTE["human"],
            )
            add_direction_arrow(
                ax,
                prev["ai_col"],
                prev["ai_row"],
                ai_x,
                ai_y,
                PALETTE["ai"],
            )

        # Agents
        ax.scatter(
            human_x,
            human_y,
            s=420,
            marker="o",
            color=PALETTE["human"],
            edgecolor="#111111",
            linewidth=1.5,
            zorder=8,
            label="Human",
        )

        ax.scatter(
            ai_x,
            ai_y,
            s=420,
            marker="s",
            color=PALETTE["ai"],
            edgecolor="#111111",
            linewidth=1.5,
            zorder=8,
            label="AI",
        )

        ax.text(
            human_x,
            human_y,
            "H",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#111111",
            zorder=9,
        )

        ax.text(
            ai_x,
            ai_y,
            "AI",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#FFFFFF",
            zorder=9,
        )

        # Labels near agents
        ax.text(
            human_x + 0.15,
            human_y - 0.25,
            f"{row['human_action']}\n{row['human_holding']}",
            fontsize=8,
            ha="left",
            va="top",
            color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["human"], alpha=0.85),
            zorder=10,
        )

        ax.text(
            ai_x + 0.15,
            ai_y + 0.35,
            f"{row['ai_action']}\n{row['ai_holding']}",
            fontsize=8,
            ha="left",
            va="bottom",
            color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["ai"], alpha=0.85),
            zorder=10,
        )

        # Frame title / subtitle
        ax.set_title(f"t = {t}     {summary_text}", fontsize=9)
        ax.legend(loc="upper right", frameon=True)

        info_text.set_text(
            "Replay reconstructed from logged positions/actions, not original screen capture."
        )

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(timeline),
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as mp4 if ffmpeg is available, otherwise gif.
    if output_path.suffix.lower() == ".mp4":
        if shutil.which("ffmpeg") is not None:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path, writer=writer)
            final_path = output_path
        else:
            print("[WARN] ffmpeg not found. Saving GIF instead.")
            final_path = output_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(final_path, writer=writer)
    elif output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        final_path = output_path
    else:
        # Default to mp4.
        final_path = output_path.with_suffix(".mp4")
        if shutil.which("ffmpeg") is not None:
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(final_path, writer=writer)
        else:
            final_path = output_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(final_path, writer=writer)

    plt.close(fig)
    return final_path


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a replay animation for one participant episode/round."
    )

    parser.add_argument(
        "--participant",
        default=None,
        help="Participant folder name, e.g. Thinpath_P06. Looks inside submissions/.",
    )

    parser.add_argument(
        "--participant-folder",
        default=None,
        help="Full path to participant folder. Use this if data is not in submissions/.",
    )

    parser.add_argument(
        "--json",
        default=None,
        help="Optional JSON log path or filename. If omitted, the script searches automatically.",
    )

    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index to replay.",
    )

    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round within the episode to replay. Default: 1.",
    )

    parser.add_argument(
        "--global-round",
        type=int,
        default=None,
        help="Optional global round index. If provided, overrides --episode and --round selection.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. Default: analysis_outputs/replay_videos.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for the output animation. Default: 8.",
    )

    parser.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format. Default: mp4. Falls back to gif if ffmpeg is unavailable.",
    )

    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable movement trails.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    participant_folder = find_participant_folder(args.participant, args.participant_folder)
    participant_id = participant_folder.name

    json_path = find_json_log(participant_folder, args.json)
    log_data = load_json(json_path)

    round_data = select_round(
        log_data=log_data,
        episode_index=args.episode,
        round_in_episode=args.round,
        global_round=args.global_round,
    )

    episode_index = int_or_none(round_data.get("episode_index")) or args.episode
    round_in_episode = int_or_none(round_data.get("round_in_episode")) or args.round
    global_round = int_or_none(round_data.get("round_index_global"))

    episode_meta = get_episode_metadata(log_data, episode_index)

    timeline = round_to_timeline(round_data)

    output_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    phase = str(round_data.get("episode_phase", episode_meta.get("episode_phase", "unknown")))
    safe_phase = re.sub(r"[^A-Za-z0-9_-]+", "_", phase)

    filename = (
        f"{participant_id}_E{episode_index}_R{round_in_episode}"
        f"_G{global_round}_{safe_phase}.{args.format}"
    )

    output_path = output_dir / filename

    print("\n=== Replay animation ===")
    print(f"Participant folder: {participant_folder}")
    print(f"JSON log: {json_path}")
    print(f"Episode: {episode_index}")
    print(f"Round in episode: {round_in_episode}")
    print(f"Global round: {global_round}")
    print(f"Phase: {phase}")
    print(f"Frames: {len(timeline)}")
    print(f"Output requested: {output_path}")

    final_path = create_animation(
        timeline=timeline,
        participant_id=participant_id,
        round_data=round_data,
        episode_meta=episode_meta,
        output_path=output_path,
        fps=args.fps,
        show_trails=not args.no_trails,
    )

    print("\nSaved animation to:")
    print(f"  {final_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
