#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisi 1: Solo vs Best-policy
------------------------------

Questo script confronta, per ogni partecipante:

    Solo episodes
    vs
    Best-policy replay

Produce solo 4 grafici:

    1. paired_dishes_per_round.png
    2. paired_human_steps_per_dish.png
    3. paired_mental_demand.png
    4. paired_subjective_performance.png

Struttura attesa:

    Overcooked_equilibrium/
        analyze_solo_vs_best.py
        submissions/
            Thinpath_P01/
                round_summary.csv
            Thinpath_P02/
                round_summary.csv
            ...
            Thinpath_P10/
                round_summary.csv

Nota:
    Lo script usa il nome della cartella come participant_id.
    Quindi i partecipanti verranno chiamati Thinpath_P01, Thinpath_P02, ecc.
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
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_SOURCES = [
    PROJECT_ROOT / "submissions",
]

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "solo_vs_best"

# Accetta solo cartelle tipo:
# Thinpath_P01, Thinpath_P02, ..., Thinpath_P10
PARTICIPANT_FOLDER_PATTERN = re.compile(r"^Thinpath_P\d+$", re.IGNORECASE)

SOLO_PHASES = {"solo"}
BEST_POLICY_PHASES = {"bo_replay_best"}

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

GROUP_MEAN_COLOR = "#222222"
# ============================================================
# CARICAMENTO DATI
# ============================================================

def is_valid_participant_folder(folder_name: str) -> bool:
    """True se il nome cartella è tipo Thinpath_P01, Thinpath_P02, ecc."""
    return bool(PARTICIPANT_FOLDER_PATTERN.match(folder_name))


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    """
    Ordina Thinpath_P01, Thinpath_P02, ..., Thinpath_P10 in modo corretto.
    """
    match = re.search(r"(\d+)$", participant_id)
    if match:
        return int(match.group(1)), participant_id
    return 9999, participant_id


def find_round_summary_files(data_sources: Iterable[Path]) -> list[Path]:
    """Trova tutti i round_summary.csv dentro cartelle partecipante valide."""
    files: list[Path] = []

    for src in data_sources:
        src = Path(src).expanduser()

        if not src.exists():
            print(f"[WARN] Percorso non trovato: {src}")
            continue

        if src.is_file() and src.name == "round_summary.csv":
            if is_valid_participant_folder(src.parent.name):
                files.append(src)

        elif src.is_dir():
            found = src.rglob("round_summary.csv")
            files.extend(
                f for f in found
                if is_valid_participant_folder(f.parent.name)
            )

    # Rimuove duplicati
    unique_files = []
    seen = set()

    for f in sorted(files):
        key = f.resolve()
        if key not in seen:
            unique_files.append(f)
            seen.add(key)

    return unique_files


def load_rounds(round_files: list[Path]) -> pd.DataFrame:
    """Carica e concatena tutti i round_summary.csv."""
    frames = []

    for f in round_files:
        df = pd.read_csv(f)

        # Usiamo sempre il nome della cartella come participant_id
        df["participant_id"] = f.parent.name
        df["source_file"] = str(f)

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Nessun round_summary.csv trovato. "
            "Controlla che le cartelle siano nominate Thinpath_P01, Thinpath_P02, ecc."
        )

    rounds = pd.concat(frames, ignore_index=True)

    if "episode_phase" not in rounds.columns:
        raise ValueError("Colonna mancante: episode_phase")

    rounds["episode_phase"] = rounds["episode_phase"].astype(str).str.strip()

    numeric_cols = [
        "episode_index",
        "round_in_episode",
        "dishes_served",
        "human_steps",
        "team_reward_score",
        "mental_demand",
        "performance_score",
    ]

    for col in numeric_cols:
        if col in rounds.columns:
            rounds[col] = pd.to_numeric(rounds[col], errors="coerce")

    return rounds


# ============================================================
# ANALISI SOLO VS BEST-POLICY
# ============================================================

def assign_condition(episode_phase: str) -> str | None:
    """Mappa episode_phase in Solo / Best-policy."""
    phase = str(episode_phase).strip()

    if phase in SOLO_PHASES:
        return "Solo"

    if phase in BEST_POLICY_PHASES:
        return "Best-policy"

    return None


def make_phase_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabella con una riga per partecipante e condizione.

    Output:
        participant_id
        condition
        mean_dishes_per_round
        human_steps_per_dish
        mean_mental_demand
        mean_subjective_performance
    """
    rounds = rounds.copy()
    rounds["condition"] = rounds["episode_phase"].apply(assign_condition)

    selected = rounds.loc[rounds["condition"].notna()].copy()

    if selected.empty:
        raise ValueError(
            "Non trovo righe con episode_phase='solo' o episode_phase='bo_replay_best'."
        )

    required_cols = [
        "participant_id",
        "condition",
        "episode_index",
        "dishes_served",
        "human_steps",
    ]

    for col in required_cols:
        if col not in selected.columns:
            raise ValueError(f"Colonna mancante: {col}")

    rows = []

    for (participant_id, condition), g in selected.groupby(["participant_id", "condition"]):
        total_dishes = g["dishes_served"].sum(skipna=True)
        total_human_steps = g["human_steps"].sum(skipna=True)
        n_rounds = g["dishes_served"].notna().sum()

        if total_dishes > 0:
            human_steps_per_dish = total_human_steps / total_dishes
        else:
            human_steps_per_dish = np.nan

        # Rating soggettivi: prendiamo un valore per episodio, poi facciamo la media
        mean_mental_demand = np.nan
        if "mental_demand" in g.columns:
            mean_mental_demand = (
                g[["episode_index", "mental_demand"]]
                .drop_duplicates(subset=["episode_index"])
                ["mental_demand"]
                .mean(skipna=True)
            )

        mean_subjective_performance = np.nan
        if "performance_score" in g.columns:
            mean_subjective_performance = (
                g[["episode_index", "performance_score"]]
                .drop_duplicates(subset=["episode_index"])
                ["performance_score"]
                .mean(skipna=True)
            )

        was_skipped = False
        if "skipped_episode" in g.columns:
            was_skipped = g["skipped_episode"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ).any()    

        rows.append({
            "participant_id": participant_id,
            "condition": condition,
            "n_rounds": n_rounds,
            "total_dishes": total_dishes,
            "total_human_steps": total_human_steps,
            "mean_dishes_per_round": total_dishes / n_rounds if n_rounds else np.nan,
            "human_steps_per_dish": human_steps_per_dish,
            "mean_mental_demand": mean_mental_demand,
            "mean_subjective_performance": mean_subjective_performance,
            "was_skipped": was_skipped,
        })

    summary = pd.DataFrame(rows)

    summary["participant_sort"] = summary["participant_id"].apply(participant_sort_key)
    summary = summary.sort_values(["participant_sort", "condition"])
    summary = summary.drop(columns=["participant_sort"])

    return summary


def make_participant_summary(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabella con Solo, Best-policy e differenza per ogni partecipante.
    """
    metrics = [
        "mean_dishes_per_round",
        "human_steps_per_dish",
        "mean_mental_demand",
        "mean_subjective_performance",
    ]

    solo = phase_summary.loc[phase_summary["condition"] == "Solo"].set_index("participant_id")
    best = phase_summary.loc[phase_summary["condition"] == "Best-policy"].set_index("participant_id")

    participants = sorted(
        set(solo.index).union(set(best.index)),
        key=participant_sort_key,
    )

    rows = []

    for pid in participants:
        row = {"participant_id": pid}

        for metric in metrics:
            solo_value = solo.loc[pid, metric] if pid in solo.index else np.nan
            best_value = best.loc[pid, metric] if pid in best.index else np.nan
            delta = best_value - solo_value

            row[f"solo_{metric}"] = solo_value
            row[f"best_{metric}"] = best_value
            row[f"delta_{metric}"] = delta

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# GRAFICI
# ============================================================

def paired_plot(
    phase_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    lower_is_better: bool = False,
) -> None:
    """
    Crea un paired line plot Solo vs Best-policy.

    Ogni linea = un partecipante.
    I colori seguono la palette Gradient Blues.
    Il diamante nero = media del gruppo.
    La X = Best-policy skippata / incompleta.
    """
    pivot = phase_summary.pivot(
        index="participant_id",
        columns="condition",
        values=metric,
    )

    skipped_pivot = None
    if "was_skipped" in phase_summary.columns:
        skipped_pivot = phase_summary.pivot(
            index="participant_id",
            columns="condition",
            values="was_skipped",
        )

    if "Solo" not in pivot.columns or "Best-policy" not in pivot.columns:
        print(f"[WARN] Non posso creare {output_path.name}: mancano Solo o Best-policy.")
        return

    pivot = pivot.dropna(subset=["Solo", "Best-policy"], how="any")

    if pivot.empty:
        print(f"[WARN] Non posso creare {output_path.name}: mancano dati appaiati.")
        return

    pivot = pivot.loc[sorted(pivot.index, key=participant_sort_key)]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_solo = 0
    x_best = 1

    legend_added_skipped = False

    for i, (participant_id, row) in enumerate(pivot.iterrows()):
        color = LINE_COLORS[i % len(LINE_COLORS)]

        y_solo = row["Solo"]
        y_best = row["Best-policy"]

        best_was_skipped = False
        if skipped_pivot is not None:
            if participant_id in skipped_pivot.index and "Best-policy" in skipped_pivot.columns:
                best_was_skipped = bool(skipped_pivot.loc[participant_id, "Best-policy"])

        # Linea del partecipante
        ax.plot(
            [x_solo, x_best],
            [y_solo, y_best],
            color=color,
            linewidth=2.2,
            alpha=0.95,
        )

        # Punto Solo
        ax.scatter(
            x_solo,
            y_solo,
            color=color,
            marker="o",
            s=55,
            zorder=4,
        )

        # Punto Best-policy
        if best_was_skipped:
            ax.scatter(
                x_best,
                y_best,
                color=color,
                marker="X",
                s=130,
                zorder=5,
                label="Best-policy skipped / incomplete" if not legend_added_skipped else None,
            )
            legend_added_skipped = True
            label_text = f"{participant_id} (skipped)"
        else:
            ax.scatter(
                x_best,
                y_best,
                color=color,
                marker="o",
                s=55,
                zorder=4,
            )
            label_text = participant_id

        ax.text(
            x_best + 0.03,
            y_best,
            label_text,
            va="center",
            fontsize=8,
            color="black",
        )

    # Media del gruppo
    mean_solo = pivot["Solo"].mean()
    mean_best = pivot["Best-policy"].mean()

    ax.plot(
        [x_solo, x_best],
        [mean_solo, mean_best],
        color=GROUP_MEAN_COLOR,
        linewidth=2.5,
        linestyle="--",
        alpha=0.8,
        zorder=5,
    )

    ax.scatter(
        [x_solo, x_best],
        [mean_solo, mean_best],
        s=130,
        marker="D",
        color=GROUP_MEAN_COLOR,
        label="Group mean",
        zorder=6,
    )

    ax.set_xticks([x_solo, x_best])
    ax.set_xticklabels(["Solo", "Best-policy"])
    ax.set_ylabel(ylabel)

    subtitle = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{title}\n({subtitle})")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_plots(phase_summary: pd.DataFrame, output_dir: Path) -> None:
    """Crea solo i 4 grafici scelti."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_plot(
        phase_summary=phase_summary,
        metric="mean_dishes_per_round",
        ylabel="Mean dishes per round",
        title="Solo vs Best-policy: task performance",
        output_path=output_dir / "paired_dishes_per_round.png",
        lower_is_better=False,
    )

    paired_plot(
        phase_summary=phase_summary,
        metric="human_steps_per_dish",
        ylabel="Human steps per dish",
        title="Solo vs Best-policy: human effort efficiency",
        output_path=output_dir / "paired_human_steps_per_dish.png",
        lower_is_better=True,
    )

    paired_plot(
        phase_summary=phase_summary,
        metric="mean_mental_demand",
        ylabel="Mean mental demand rating",
        title="Solo vs Best-policy: subjective effort",
        output_path=output_dir / "paired_mental_demand.png",
        lower_is_better=True,
    )

    paired_plot(
        phase_summary=phase_summary,
        metric="mean_subjective_performance",
        ylabel="Mean subjective performance rating",
        title="Solo vs Best-policy: subjective performance",
        output_path=output_dir / "paired_subjective_performance.png",
        lower_is_better=False,
    )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisi Solo vs Best-policy: produce solo 4 paired plots."
    )

    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Cartelle o file round_summary.csv da includere. Se assente usa DATA_SOURCES.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Cartella di output. Se assente usa OUTPUT_DIR.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_sources = [Path(p) for p in args.data] if args.data else DATA_SOURCES
    output_dir = Path(args.out).expanduser() if args.out else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Solo vs Best-policy analysis ===")

    print("\nData sources:")
    for src in data_sources:
        print(f"  - {src}")

    round_files = find_round_summary_files(data_sources)

    print(f"\nFound {len(round_files)} participant files:")
    for f in round_files:
        print(f"  - {f}")

    rounds = load_rounds(round_files)

    print("\nEpisode phases found:")
    print(rounds["episode_phase"].value_counts(dropna=False).to_string())

    phase_summary = make_phase_summary(rounds)
    participant_summary = make_participant_summary(phase_summary)

    # Salviamo le tabelle perché sono utili per controllare i numeri dietro ai grafici
    phase_summary.to_csv(output_dir / "solo_vs_best_long.csv", index=False)
    participant_summary.to_csv(output_dir / "solo_vs_best_participant_summary.csv", index=False)

    make_plots(phase_summary, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nCreated plots:")
    print("  - paired_dishes_per_round.png")
    print("  - paired_human_steps_per_dish.png")
    print("  - paired_mental_demand.png")
    print("  - paired_subjective_performance.png")

    print("\nParticipant summary:")
    print(participant_summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()