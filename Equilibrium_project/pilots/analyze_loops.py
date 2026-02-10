import pandas as pd
from collections import Counter
import sys

def calculate_repetition(file_path):
    print(f"\nAnalyzing File: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Check the name!")
        return

    # Check if we have the right column names (Score vs Score_Count)
    score_col = 'Score' if 'Score' in df.columns else 'Score_Count'

    # Get all rounds
    rounds = df['Round'].unique()

    print(f"{'ROUND':<8} | {'STEPS':<8} | {'REPETITION %':<15} | {'DISHES SERVED'}")
    print("-" * 55)

    for r in rounds:
        # 1. Get data for this round
        round_data = df[df['Round'] == r]
        
        # 2. Get Final Score (Max value of score column for this round)
        final_score = round_data[score_col].max()
        
        # 3. Extract strictly the Human X, Y coordinates
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        
        # 4. Simplify: Remove consecutive duplicates (standing still)
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]

        if len(path) < 5:
            print(f"{r:<8} | {len(path):<8} | {'N/A':<15} | {final_score}")
            continue

        # 5. N-Gram Analysis (Looking for 4-step sequences)
        SEQUENCE_LEN = 4
        sequences = []

        for i in range(len(path) - SEQUENCE_LEN + 1):
            seq = tuple(path[i : i + SEQUENCE_LEN])
            sequences.append(seq)

        if not sequences:
            score = 0.0
        else:
            # Count Frequencies
            counts = Counter(sequences)
            # Sum of all sequences that appeared more than once
            repeats = sum(freq for seq, freq in counts.items() if freq > 1)
            total_possible = len(sequences)
            score = (repeats / total_possible) * 100 if total_possible > 0 else 0

        # Print Row
        print(f"{r:<8} | {len(path):<8} | {score:>6.1f}%          | {final_score}")

if __name__ == "__main__":
    # REPLACE THIS with your actual csv filename
    filename = "Pilot3.csv" 
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    calculate_repetition(filename)