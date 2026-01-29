import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import Counter

# --- CONFIGURATION ---
# This grabs ANY csv file starting with "pilot"
FILE_PATTERN = "pilot*.csv" 

def calculate_metrics_from_csv(filename):
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []

    # 1. HANDLE DIFFERENT COLUMN NAMES
    # Some files might use 'Score', others 'Score_Count'
    if 'Score' in df.columns and 'Score_Count' not in df.columns:
        df.rename(columns={'Score': 'Score_Count'}, inplace=True)
        
    # 2. SAFETY CHECK (This fixes the crash!)
    # We define strictly what columns we NEED to run the math.
    # If a file (like the summary one) is missing these, we skip it.
    required_columns = ['Round', 'Human_X', 'Human_Y', 'Score_Count']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"   -> SKIPPING {filename} (Not a raw log file. Missing: {missing_cols})")
        return []

    processed_data = []
    rounds = sorted(df['Round'].unique())

    for r in rounds:
        round_data = df[df['Round'] == r]
        
        # Max score found in this round
        dishes = round_data['Score_Count'].max()

        # Extract movement path
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        
        # Remove "standing still" moments to count steps
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        steps = len(path)

        # Calculate Repetition %
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
        print("ERROR: No files found starting with 'pilot'!")
        print(f"Current folder: {os.getcwd()}")
        return

    print(f"Found {len(all_files)} files.")

    all_round_data = []
    
    for f in all_files:
        print(f"Processing {f}...")
        pilot_metrics = calculate_metrics_from_csv(f)
        for row in pilot_metrics:
            all_round_data.append(row)

    if not all_round_data:
        print("\nERROR: No valid data extracted from any file.")
        print("Check if your CSVs have 'Round', 'Human_X', 'Human_Y', and 'Score_Count'.")
        return

    # --- PLOTTING ---
    df = pd.DataFrame(all_round_data)
    
    # Group by Round to get Mean and Std Dev
    grouped = df.groupby('Round').agg(['mean', 'std'])

    rounds = grouped.index
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = [
        ('Steps', 'Mean Steps', 'blue'),
        ('Repetition', 'Mean Repetition %', 'green'),
        ('Dishes', 'Mean Dishes Served', 'red')
    ]

    for i, (metric, title, color) in enumerate(metrics):
        # We use .get() to safely grab columns even if they are missing
        if metric not in grouped.columns.levels[0]:
            continue
            
        mean_val = grouped[metric]['mean']
        std_val = grouped[metric]['std'].fillna(0) # If only 1 pilot, std is NaN -> 0
        
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
    output_filename = 'pilot_results_automatic.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nSUCCESS! Plot saved as: {output_filename}")

if __name__ == "__main__":
    main()