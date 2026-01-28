import pandas as pd
import sys

def calculate_habits(file_path):
    print(f"\nAnalyzing Habits in: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Handle different column names if needed
    score_col = 'Score' if 'Score' in df.columns else 'Score_Count'
    rounds = sorted(df['Round'].unique())

    print(f"{'ROUND':<8} | {'SIMILARITY TO PREV':<20} | {'DISHES'}")
    print("-" * 50)

    # Store the sequences of the previous round
    prev_sequences = set()

    for r in rounds:
        # 1. Get data for this round
        round_data = df[df['Round'] == r]
        final_score = round_data[score_col].max()
        
        # 2. Extract Path (Human X, Y)
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        
        # 3. Simplify (Remove standing still)
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]

        if len(path) < 5:
            print(f"{r:<8} | {'N/A':<20} | {final_score}")
            continue

        # 4. Generate 4-step Sequences (N-Grams)
        # Example: Up -> Right -> Down -> Left
        SEQUENCE_LEN = 4
        current_sequences = set()
        
        # We collect ALL unique 4-step moves made in this round
        for i in range(len(path) - SEQUENCE_LEN + 1):
            seq = tuple(path[i : i + SEQUENCE_LEN])
            current_sequences.add(seq)

        # 5. Compare with Previous Round
        similarity_score = 0.0
        
        if r == 1:
            # Round 1 has no history to repeat
            similarity_score = 0.0
        else:
            if len(current_sequences) > 0:
                # Find overlapping sequences
                # (Moves present in BOTH this round AND the last one)
                common_moves = current_sequences.intersection(prev_sequences)
                
                # Formula: What % of my CURRENT moves were also in the PREVIOUS round?
                similarity_score = (len(common_moves) / len(current_sequences)) * 100

        # Print Result
        print(f"{r:<8} | {similarity_score:>6.1f}%             | {final_score}")

        # Update "Previous" for the next loop
        prev_sequences = current_sequences

if __name__ == "__main__":
    # REPLACE THIS with your filename
    filename = "experiment_log_20260128_144945.csv"  
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        
    calculate_habits(filename)