#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi 1: Solo episodes vs Best-policy replay
------------------------------------------------
Scopo:
    Capire se la collaborazione Human-AI con la best-policy funziona meglio
    del gioco in solitaria.

Come usarlo:
    1) Metti questo file nella root del progetto, cioè allo stesso livello della cartella `submissions/`.
    2) Se i partecipanti sono dentro `submissions/`, non devi cambiare nulla.
       La struttura attesa è:

           submissions/
               Thinpath_P01/
                   round_summary.csv
               Thinpath_P02/
                   round_summary.csv
               ...

    3) Se alcuni dati sono in un'altra cartella, aggiungi il percorso in DATA_SOURCES sotto.
    4) Esegui:

           python analyze_solo_vs_best.py

       oppure:

           python analyze_solo_vs_best.py --data submissions --out analysis_outputs/solo_vs_best

Output:
    - solo_vs_best_long.csv
    - solo_vs_best_participant_summary.csv
    - solo_vs_best_group_summary.csv
    - paired_*.png

Nota:
    Di default la best-policy è `bo_replay_best`, non `replay_optimal`.
    `replay_optimal` può includere il test finale in cui l'umano cambia comportamento,
    quindi di solito va analizzato separatamente.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CAMBIA SOLO QUESTA PARTE, SE SERVE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Se tutti i dati nuovi finiscono dentro `submissions/`, non serve modificare nulla.
# Se un partecipante/cartella è altrove, aggiungi un nuovo Path qui.
DATA_SOURCES = [
    PROJECT_ROOT / "submissions",
    # Esempi:
    # PROJECT_ROOT / "submissions" / "Thinpath_P01",
    # Path(r"\\work.org.aalto.fi\T412\T40710\OverCookedHAIC\participant_logs\main_study"),
]

OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "solo_vs_best"

# Fase usata come best-policy principale.
BEST_POLICY_PHASES = {"bo_replay_best"}

# Fase usata come solo baseline.
SOLO_PHASES = {"solo"}

# Se True, quando manca `bo_replay_best` usa `replay_optimal` come fallback.
# Per analisi scientifica pulita consiglio False.
USE_REPLAY_OPTIMAL_AS_FALLBACK = False


# ============================================================
# FUNZIONI DI CARICAMENTO
# ============================================================

def find_round_summary_files(data_sources: Iterable[Path]) -> list[Path]:
    """Trova tutti i file round_summary.csv dentro le cartelle indicate."""
    files: list[Path] = []

    for src in data_sources:
        src = Path(src).expanduser()
        if not src.exists():
            print(f"[WARN] Percorso non trovato: {src}")
            continue

        if src.is_file() and src.name == "round_summary.csv":
            files.append(src)
        elif src.is_dir():
            # Cerca ricorsivamente, così funziona sia con `submissions/`
            # sia con una cartella singola del partecipante.
            files.extend(src.rglob("round_summary.csv"))

    # Rimuove duplicati mantenendo ordine stabile.
    unique = []
    seen = set()
    for f in sorted(files):
        key = f.resolve()
        if key not in seen:
            unique.append(f)
            seen.add(key)

    return unique


def load_rounds(round_files: list[Path]) -> pd.DataFrame:
    """Carica e concatena tutti i round_summary.csv."""
    frames = []

    for f in round_files:
        df = pd.read_csv(f)
        df["source_file"] = str(f)
        df["source_folder"] = f.parent.name

        # Usa prolific_id se esiste; altrimenti usa il nome cartella.
        if "prolific_id" not in df.columns or df["prolific_id"].isna().all():
            df["prolific_id"] = f.parent.name

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Nessun round_summary.csv trovato. Controlla DATA_SOURCES o usa --data."
        )

    rounds = pd.concat(frames, ignore_index=True)

    # Normalizza alcune colonne importanti.
    rounds["participant_id"] = rounds["prolific_id"].fillna(rounds["source_folder"]).astype(str)
    rounds["episode_phase"] = rounds["episode_phase"].astype(str).str.strip()

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
        if col in rounds.columns:
            rounds[col] = pd.to_numeric(rounds[col], errors="coerce")

    # Rimuove round skipped, se la colonna esiste.
    if "skipped_episode" in rounds.columns:
        skipped = rounds["skipped_episode"].astype(str).str.lower().isin(["true", "1", "yes"])
        rounds = rounds.loc[~skipped].copy()

    return rounds


# ============================================================
# ANALISI SOLO VS BEST-POLICY
# ============================================================

def assign_condition(episode_phase: str) -> str | None:
    phase = str(episode_phase).strip()
    if phase in SOLO_PHASES:
        return "Solo"
    if phase in BEST_POLICY_PHASES:
        return "Best-policy"
    return None


def make_phase_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabella participant x condition con le metriche principali."""
    rounds = rounds.copy()
    rounds["condition"] = rounds["episode_phase"].apply(assign_condition)

    selected = rounds.loc[rounds["condition"].notna()].copy()

    if selected.empty and USE_REPLAY_OPTIMAL_AS_FALLBACK:
        rounds["condition"] = np.where(rounds["episode_phase"] == "solo", "Solo", None)
        rounds["condition"] = np.where(
            rounds["episode_phase"] == "replay_optimal",
            "Best-policy",
            rounds["condition"],
        )
        selected = rounds.loc[rounds["condition"].notna()].copy()

    if selected.empty:
        raise ValueError(
            "Non trovo righe con episode_phase='solo' o 'bo_replay_best'. "
            "Controlla i nomi delle fasi nel round_summary.csv."
        )

    rows = []

    for (participant_id, condition), g in selected.groupby(["participant_id", "condition"]):
        total_dishes = g["dishes_served"].sum(skipna=True)
        total_human_steps = g["human_steps"].sum(skipna=True)
        total_team_reward = g["team_reward_score"].sum(skipna=True)
        n_rounds = g["dishes_served"].notna().sum()

        # Subjective ratings: prendiamo un valore per episodio, poi media fra episodi.
        ep_subjective = (
            g[["episode_index", "mental_demand", "performance_score"]]
            .drop_duplicates(subset=["episode_index"])
        )

        human_steps_per_dish = np.nan
        dishes_per_100_human_steps = np.nan
        if total_dishes > 0:
            human_steps_per_dish = total_human_steps / total_dishes
        if total_human_steps > 0:
            dishes_per_100_human_steps = 100 * total_dishes / total_human_steps

        rows.append({
            "participant_id": participant_id,
            "condition": condition,
            "n_episodes": g["episode_index"].nunique(),
            "n_rounds": n_rounds,
            "mean_dishes_per_round": total_dishes / n_rounds if n_rounds else np.nan,
            "mean_team_reward_per_round": total_team_reward / n_rounds if n_rounds else np.nan,
            "mean_human_steps_per_round": total_human_steps / n_rounds if n_rounds else np.nan,
            "human_steps_per_dish": human_steps_per_dish,
            "dishes_per_100_human_steps": dishes_per_100_human_steps,
            "mean_mental_demand": ep_subjective["mental_demand"].mean(skipna=True),
            "mean_subjective_performance": ep_subjective["performance_score"].mean(skipna=True),
        })

    summary = pd.DataFrame(rows)
    return summary.sort_values(["participant_id", "condition"])


def make_participant_delta_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabella con Solo, Best-policy e differenza per ogni partecipante."""
    metrics = [
        "mean_dishes_per_round",
        "mean_team_reward_per_round",
        "mean_human_steps_per_round",
        "human_steps_per_dish",
        "dishes_per_100_human_steps",
        "mean_mental_demand",
        "mean_subjective_performance",
    ]

    solo = phase_summary.loc[phase_summary["condition"] == "Solo"].set_index("participant_id")
    best = phase_summary.loc[phase_summary["condition"] == "Best-policy"].set_index("participant_id")

    participants = sorted(set(solo.index).union(set(best.index)))
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
            row[f"pct_change_{metric}"] = (
                100 * delta / solo_value
                if pd.notna(solo_value) and solo_value != 0
                else np.nan
            )

        # Label interpretativa semplice.
        d_dishes = row.get("delta_mean_dishes_per_round", np.nan)
        d_effort = row.get("delta_human_steps_per_dish", np.nan)  # negativo = meglio
        d_demand = row.get("delta_mean_mental_demand", np.nan)    # negativo = meglio

        if pd.isna(d_dishes):
            label = "missing data"
        elif d_dishes > 0 and pd.notna(d_effort) and d_effort < 0:
            label = "strong improvement: more dishes + less effort per dish"
        elif d_dishes > 0:
            label = "mixed improvement: more dishes, check effort/demand"
        elif d_dishes == 0:
            label = "no performance change"
        else:
            label = "no improvement: solo performed better"

        if pd.notna(d_demand) and d_demand > 0 and label.startswith("strong"):
            label = label + ", but mental demand increased"

        row["interpretation"] = label
        rows.append(row)

    return pd.DataFrame(rows)


def make_group_summary(phase_summary: pd.DataFrame, delta_table: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabella di medie di gruppo."""
    rows = []

    for condition, g in phase_summary.groupby("condition"):
        rows.append({
            "level": "condition_mean",
            "condition_or_metric": condition,
            "n_participants": g["participant_id"].nunique(),
            "mean_dishes_per_round": g["mean_dishes_per_round"].mean(skipna=True),
            "mean_team_reward_per_round": g["mean_team_reward_per_round"].mean(skipna=True),
            "mean_human_steps_per_round": g["mean_human_steps_per_round"].mean(skipna=True),
            "human_steps_per_dish": g["human_steps_per_dish"].mean(skipna=True),
            "dishes_per_100_human_steps": g["dishes_per_100_human_steps"].mean(skipna=True),
            "mean_mental_demand": g["mean_mental_demand"].mean(skipna=True),
            "mean_subjective_performance": g["mean_subjective_performance"].mean(skipna=True),
        })

    delta_metrics = [c for c in delta_table.columns if c.startswith("delta_")]
    for metric in delta_metrics:
        rows.append({
            "level": "delta_mean_best_minus_solo",
            "condition_or_metric": metric.replace("delta_", ""),
            "n_participants": delta_table[metric].notna().sum(),
            "mean_dishes_per_round": delta_table[metric].mean(skipna=True) if metric == "delta_mean_dishes_per_round" else np.nan,
            "mean_team_reward_per_round": delta_table[metric].mean(skipna=True) if metric == "delta_mean_team_reward_per_round" else np.nan,
            "mean_human_steps_per_round": delta_table[metric].mean(skipna=True) if metric == "delta_mean_human_steps_per_round" else np.nan,
            "human_steps_per_dish": delta_table[metric].mean(skipna=True) if metric == "delta_human_steps_per_dish" else np.nan,
            "dishes_per_100_human_steps": delta_table[metric].mean(skipna=True) if metric == "delta_dishes_per_100_human_steps" else np.nan,
            "mean_mental_demand": delta_table[metric].mean(skipna=True) if metric == "delta_mean_mental_demand" else np.nan,
            "mean_subjective_performance": delta_table[metric].mean(skipna=True) if metric == "delta_mean_subjective_performance" else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# VISUALIZZAZIONI
# ============================================================

def paired_plot(
    phase_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    lower_is_better: bool = False,
) -> None:
    """Crea un paired dot plot Solo vs Best-policy."""
    pivot = phase_summary.pivot(index="participant_id", columns="condition", values=metric)

    # Teniamo solo partecipanti con entrambe le condizioni.
    pivot = pivot.dropna(subset=["Solo", "Best-policy"], how="any")

    if pivot.empty:
        print(f"[WARN] Non posso creare {output_path.name}: mancano dati appaiati.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x_solo, x_best = 0, 1

    for pid, row in pivot.iterrows():
        y1 = row["Solo"]
        y2 = row["Best-policy"]
        ax.plot([x_solo, x_best], [y1, y2], marker="o", linewidth=1.5)
        ax.text(x_best + 0.03, y2, str(pid), va="center", fontsize=8)

    # Medie di gruppo
    mean_solo = pivot["Solo"].mean()
    mean_best = pivot["Best-policy"].mean()
    ax.scatter([x_solo, x_best], [mean_solo, mean_best], s=120, marker="D", label="Group mean")

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


def make_all_plots(phase_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_plot(
        phase_summary,
        metric="mean_dishes_per_round",
        ylabel="Mean dishes per round",
        title="Solo vs Best-policy: task performance",
        output_path=output_dir / "paired_dishes_per_round.png",
        lower_is_better=False,
    )

    paired_plot(
        phase_summary,
        metric="mean_team_reward_per_round",
        ylabel="Mean team reward per round",
        title="Solo vs Best-policy: team reward",
        output_path=output_dir / "paired_team_reward.png",
        lower_is_better=False,
    )

    paired_plot(
        phase_summary,
        metric="human_steps_per_dish",
        ylabel="Human steps per dish",
        title="Solo vs Best-policy: human effort efficiency",
        output_path=output_dir / "paired_human_steps_per_dish.png",
        lower_is_better=True,
    )

    paired_plot(
        phase_summary,
        metric="dishes_per_100_human_steps",
        ylabel="Dishes per 100 human steps",
        title="Solo vs Best-policy: human efficiency",
        output_path=output_dir / "paired_dishes_per_100_human_steps.png",
        lower_is_better=False,
    )

    paired_plot(
        phase_summary,
        metric="mean_mental_demand",
        ylabel="Mean mental demand rating",
        title="Solo vs Best-policy: subjective effort",
        output_path=output_dir / "paired_mental_demand.png",
        lower_is_better=True,
    )

    paired_plot(
        phase_summary,
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
    parser = argparse.ArgumentParser(description="Analisi Solo vs Best-policy per esperimento Overcooked.")
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
    print("Data sources:")
    for src in data_sources:
        print(f"  - {src}")

    round_files = find_round_summary_files(data_sources)
    print(f"\nFound {len(round_files)} round_summary.csv files")
    for f in round_files:
        print(f"  - {f}")

    rounds = load_rounds(round_files)
    phase_summary = make_phase_summary(rounds)
    delta_table = make_participant_delta_table(phase_summary)
    group_summary = make_group_summary(phase_summary, delta_table)

    # Salva tabelle.
    phase_summary.to_csv(output_dir / "solo_vs_best_long.csv", index=False)
    delta_table.to_csv(output_dir / "solo_vs_best_participant_summary.csv", index=False)
    group_summary.to_csv(output_dir / "solo_vs_best_group_summary.csv", index=False)

    # Salva grafici.
    make_all_plots(phase_summary, output_dir)

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nParticipant summary:")
    display_cols = [
        "participant_id",
        "solo_mean_dishes_per_round",
        "best_mean_dishes_per_round",
        "delta_mean_dishes_per_round",
        "solo_human_steps_per_dish",
        "best_human_steps_per_dish",
        "delta_human_steps_per_dish",
        "solo_mean_mental_demand",
        "best_mean_mental_demand",
        "delta_mean_mental_demand",
        "interpretation",
    ]
    existing = [c for c in display_cols if c in delta_table.columns]
    print(delta_table[existing].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
