import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from collections import Counter
import numpy as np

# --- CONFIGURATION ---
FILE_PATTERN = "pilot*.csv" 

def get_sequences(path, seq_len=4):
    """Converts a path [(x,y), (x,y)...] into a set of 4-step sequences."""
    if len(path) < seq_len:
        return set()
    sequences = []
    for i in range(len(path) - seq_len + 1):
        sequences.append(tuple(path[i : i + seq_len]))
    return sequences

def calculate_similarity(current_path, prev_path_sequences):
    """Calculates % of current moves that match the previous round."""
    if not current_path or not prev_path_sequences:
        return 0.0
    
    current_sequences = get_sequences(current_path)
    if not current_sequences:
        return 0.0
        
    current_set = set(current_sequences)
    prev_set = set(prev_path_sequences)
    
    if len(current_set) == 0:
        return 0.0

    common = current_set.intersection(prev_set)
    similarity = (len(common) / len(current_set)) * 100
    return similarity

def analyze_pilot(filename):
    try:
        df = pd.read_csv(filename)
    except:
        return None

    # Skip files that are likely summaries
    required = ['Round', 'Human_X', 'Human_Y']
    if not all(col in df.columns for col in required):
        return None

    if 'Score' in df.columns and 'Score_Count' not in df.columns:
        df.rename(columns={'Score': 'Score_Count'}, inplace=True)
    
    pilot_name = os.path.basename(filename).replace('.csv', '').capitalize()
    rounds = sorted(df['Round'].unique())
    
    metrics = []
    prev_path_sequences = None 

    for r in rounds:
        round_data = df[df['Round'] == r]
        
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        # Remove standing still
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        
        # 1. Internal Repetition
        repetition = 0.0
        current_sequences_list = get_sequences(path)
        if current_sequences_list:
            counts = Counter(current_sequences_list)
            repeats = sum(freq for seq, freq in counts.items() if freq > 1)
            repetition = (repeats / len(current_sequences_list)) * 100
            
        # 2. History Similarity
        similarity = 0.0
        if prev_path_sequences is not None:
            similarity = calculate_similarity(path, prev_path_sequences)
        
        prev_path_sequences = set(current_sequences_list)

        metrics.append({
            'Round': r,
            'Repetition': repetition,
            'Similarity': similarity
        })
        
    return pilot_name, pd.DataFrame(metrics)

def main():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print("No pilot files found.")
        return

    # 1. PROCESS DATA
    all_metrics = []
    print("Processing files...")
    
    for f in files:
        # Note: If you want to exclude Pilot 2, uncomment below:
        if "pilot2" in f.lower(): continue
        
        result = analyze_pilot(f)
        if result:
            name, data = result
            data['Pilot'] = name
            all_metrics.append(data)

    if not all_metrics:
        print("No valid pilot data found.")
        return

    combined_df = pd.concat(all_metrics)

    # 2. CALCULATE STATISTICS (Just Mean)
    grouped = combined_df.groupby('Round').agg({
        'Repetition': 'mean',
        'Similarity': 'mean'
    })

    print("Aggregated Data (Mean):")
    print(grouped)

    # 3. PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rounds = grouped.index
    
    # -- Plot Repetition (Blue) --
    rep_mean = grouped['Repetition']
    ax.plot(rounds, rep_mean, marker='o', color='blue', linewidth=2, label='Mean Loopiness (Within-Round)')
    
    # -- Plot Similarity (Orange) --
    sim_mean = grouped['Similarity']
    ax.plot(rounds, sim_mean, marker='x', color='orange', linestyle='--', linewidth=2, label='Mean Strategy Lock (vs Prev Round)')

    # -- Aesthetics --
    ax.set_title("Habit Convergence: Mean Across Pilots", fontsize=14, fontweight='bold')
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Behavior Score (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3)
    
    # Red "Habit Zone"
    ax.axhspan(80, 100, color='red', alpha=0.1, label='Habit Set Zone')
    
    ax.legend(loc='lower right')

    plt.tight_layout()
    output_file = 'habit_convergence_mean_clean.png'
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as '{output_file}'")

if __name__ == "__main__":
    main()