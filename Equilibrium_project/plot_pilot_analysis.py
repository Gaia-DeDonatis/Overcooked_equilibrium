import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import Counter

# --- CONFIGURATION ---
# This will find all files starting with "pilot" and ending in ".csv"
# Example: pilot0.csv, pilot1.csv, pilot_test.csv
FILE_PATTERN = "pilot*.csv" 

def calculate_metrics_from_csv(filename):
    """
    Reads a raw experiment log and calculates:
    - Total Steps per round
    - Repetition % per round
    - Final Score per round
    """
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

    # Check for required columns
    required = ['Round', 'Human_X', 'Human_Y', 'Score_Count']
    if not all(col in df.columns for col in required):
        # Fallback for "Score" vs "Score_Count" naming
        if 'Score' in df.columns:
            df.rename(columns={'Score': 'Score_Count'}, inplace=True)
        else:
            print(f"Skipping {filename}: Missing columns.")
            return []

    processed_data = []
    rounds = sorted(df['Round'].unique())

    for r in rounds:
        round_data = df[df['Round'] == r]
        
        # 1. DISHES SERVED (Max score in the round)
        dishes = round_data['Score_Count'].max()

        # 2. STEPS (Calculate path length)
        # Extract Human X,Y coordinates
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        
        # Filter out "standing still" (consecutive duplicates)
        # This gives us the actual MOVEMENT path
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        steps = len(path)

        # 3. REPETITION % (N-Gram Analysis)
        # We use the logic from our analyzer script
        repetition_score = 0.0
        if steps >= 5:
            SEQUENCE_LEN = 4
            sequences = []
            for i in range(len(path) - SEQUENCE_LEN + 1):
                seq = tuple(path[i : i + SEQUENCE_LEN])
                sequences.append(seq)
            
            if sequences:
                counts = Counter(sequences)
                repeats = sum(freq for seq, freq in counts.items() if freq > 1)
                total_possible = len(sequences)
                repetition_score = (repeats / total_possible) * 100

        processed_data.append({
            'Round': r,
            'Steps': steps,
            'Repetition': repetition_score,
            'Dishes': dishes
        })
        
    return processed_data

def main():
    # --- 1. LOAD AND PROCESS ALL FILES ---
    all_files = glob.glob(FILE_PATTERN)
    
    if not all_files:
        print("No CSV files found starting with 'pilot'!")
        return

    print(f"Found {len(all_files)} files: {all_files}")

    all_round_data = []

    for f in all_files:
        print(f"Processing {f}...")
        pilot_metrics = calculate_metrics_from_csv(f)
        for row in pilot_metrics:
            row['Pilot'] = f # Add filename as ID
            all_round_data.append(row)

    if not all_round_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(all_round_data)

    # --- 2. CALCULATE STATISTICS ---
    grouped = df.groupby('Round').agg({
        'Steps': ['mean', 'std'],
        'Repetition': ['mean', 'std'],
        'Dishes': ['mean', 'std']
    })

    # --- 3. PLOTTING ---
    rounds = grouped.index
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = [
        ('Steps', 'Mean Steps per Round', 'blue'),
        ('Repetition', 'Mean Repetition % per Round', 'green'),
        ('Dishes', 'Mean Dishes Served per Round', 'red')
    ]

    for i, (metric, title, color) in enumerate(metrics):
        mean_val = grouped[metric]['mean']
        std_val = grouped[metric]['std']
        
        # Fill NaN std (happens if only 1 pilot) with 0
        std_val = std_val.fillna(0)

        # Plot Mean
        axs[i].plot(rounds, mean_val, marker='o', label=f'Mean {metric}', color=color, linewidth=2)
        
        # Plot Confidence Band
        axs[i].fill_between(rounds, 
                            mean_val - std_val, 
                            mean_val + std_val, 
                            alpha=0.2, color=color, label='Standard Deviation')
        
        # Aesthetics
        axs[i].set_ylabel(metric)
        axs[i].set_title(title)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Intervention Line
        axs[i].axvline(x=7, color='black', linestyle='--', linewidth=2, label='Proposed Intervention (R7)')
        axs[i].legend(loc='upper left')

    axs[2].set_xlabel('Round Number')
    axs[2].set_xticks(range(1, 11))
    
    plt.tight_layout()
    output_filename = 'pilot_results_automatic.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccess! Plot saved as {output_filename}")
    # plt.show()

if __name__ == "__main__":
    main()