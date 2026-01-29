import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import Counter

# --- CONFIGURATION ---
FILE_PATTERN = "pilot*.csv" 

def calculate_metrics_from_csv(filename):
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

    if 'Score' in df.columns and 'Score_Count' not in df.columns:
        df.rename(columns={'Score': 'Score_Count'}, inplace=True)
        
    required_columns = ['Round', 'Human_X', 'Human_Y', 'Score_Count']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        # Silently skip summary files or bad data
        return []

    processed_data = []
    rounds = sorted(df['Round'].unique())

    for r in rounds:
        round_data = df[df['Round'] == r]
        dishes = round_data['Score_Count'].max()

        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        steps = len(path)

        repetition_score = 0.0
        if steps >= 5:
            SEQUENCE_LEN = 4
            sequences = []
            for i in range(len(path) - SEQUENCE_LEN + 1):
                sequences.append(tuple(path[i : i + SEQUENCE_LEN]))
            
            if sequences:
                counts = Counter(sequences)
                repeats = sum(freq for seq, freq in counts.items() if freq > 1)
                repetition_score = (repeats / len(sequences)) * 100

        processed_data.append({
            'Round': r,
            'Steps': steps,
            'Repetition': repetition_score,
            'Dishes': dishes
        })
        
    return processed_data

def main():
    print("Searching for pilot files...")
    all_files = glob.glob(FILE_PATTERN)
    
    if not all_files:
        print("ERROR: No files found!")
        return

    all_round_data = []
    valid_files_count = 0

    for f in all_files:
        pilot_metrics = calculate_metrics_from_csv(f)
        if pilot_metrics:
            valid_files_count += 1
            for row in pilot_metrics:
                all_round_data.append(row)

    if not all_round_data:
        print("ERROR: No valid data extracted.")
        return

    print(f"Successfully processed {valid_files_count} pilot files.")

    # --- STATISTICS ---
    df = pd.DataFrame(all_round_data)
    
    # Calculate Mean and Std Dev
    grouped = df.groupby('Round').agg(['mean', 'std'])

    # Round the numbers to 2 decimal places for cleaner reading
    grouped = grouped.round(2)

    # 1. PRINT TO TERMINAL
    print("\n" + "="*60)
    print("       PILOT DATA SUMMARY (MEAN +/- STD DEV)")
    print("="*60)
    print(grouped)
    print("="*60 + "\n")

    # 2. SAVE TO CSV (For Copy-Pasting)
    csv_filename = "pilot_data_summary_table.csv"
    grouped.to_csv(csv_filename)
    print(f"-> Table saved as: {csv_filename}")

    # --- PLOTTING ---
    rounds = grouped.index
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = [
        ('Steps', 'Mean Steps', 'blue'),
        ('Repetition', 'Mean Repetition %', 'green'),
        ('Dishes', 'Mean Dishes Served', 'red')
    ]

    for i, (metric, title, color) in enumerate(metrics):
        if metric not in grouped.columns.levels[0]:
            continue
            
        mean_val = grouped[metric]['mean']
        std_val = grouped[metric]['std'].fillna(0)
        
        axs[i].plot(rounds, mean_val, marker='o', label=f'Mean {metric}', color=color, linewidth=2)
        axs[i].fill_between(rounds, mean_val - std_val, mean_val + std_val, alpha=0.2, color=color)
        
        axs[i].set_ylabel(metric)
        axs[i].set_title(title)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].axvline(x=7, color='black', linestyle='--', linewidth=2, label='Intervention (R7)')
        axs[i].legend(loc='upper left')

    axs[2].set_xlabel('Round Number')
    axs[2].set_xticks(range(1, 11))
    
    plt.tight_layout()
    plot_filename = 'pilot_results_automatic.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"-> Plot saved as: {plot_filename}")

if __name__ == "__main__":
    main()