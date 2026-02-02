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
        return []

    processed_data = []
    rounds = sorted(df['Round'].unique())

    # Extract clean name from filename (e.g. "pilot0.csv" -> "Pilot 0")
    pilot_name = os.path.basename(filename).replace('.csv', '').capitalize()

    for r in rounds:
        round_data = df[df['Round'] == r]
        dishes = round_data['Score_Count'].max()

        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        steps = len(path)

        repetition_score = np.nan # Default to NaN if path is too short
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
            'Pilot': pilot_name,
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

    all_data = []
    for f in all_files:
        # We process ALL files found (P0, P1, P2, P3)
        pilot_metrics = calculate_metrics_from_csv(f)
        for row in pilot_metrics:
            all_data.append(row)

    if not all_data:
        print("ERROR: No valid data extracted.")
        return

    df = pd.DataFrame(all_data)
    
    # --- PLOTTING ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    metrics = [
        ('Steps', 'Steps per Round'),
        ('Repetition', 'Repetition %'),
        ('Dishes', 'Dishes Served')
    ]

    # Get unique pilots and assign colors
    pilots = sorted(df['Pilot'].unique())
    # Use a colormap to ensure distinct colors for each pilot
    colors = plt.cm.tab10(np.linspace(0, 1, len(pilots)))
    pilot_colors = dict(zip(pilots, colors))

    for i, (metric, title) in enumerate(metrics):
        # Plot a separate line for each pilot
        for pilot in pilots:
            pilot_data = df[df['Pilot'] == pilot]
            
            axs[i].plot(pilot_data['Round'], pilot_data[metric], 
                        marker='o', 
                        label=pilot, 
                        color=pilot_colors[pilot], 
                        linewidth=2,
                        alpha=0.8) # Slight transparency to see overlapping lines
        
        axs[i].set_ylabel(metric)
        axs[i].set_title(title)
        axs[i].grid(True, linestyle='--', alpha=0.5)
        
        # Only put legend on the first plot to avoid clutter
        if i == 0:
            axs[i].legend(loc='upper right', title="Participants")

    axs[2].set_xlabel('Round Number')
    axs[2].set_xticks(range(1, 11))
    
    plt.tight_layout()
    plot_filename = 'pilot_results_individual_lines.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"\nSUCCESS! Plot saved as: {plot_filename}")

if __name__ == "__main__":
    main()