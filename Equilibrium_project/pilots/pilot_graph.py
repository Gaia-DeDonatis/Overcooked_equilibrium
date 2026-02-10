import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. CONFIGURAZIONE ---
# Assicurati che questi file siano nella stessa cartella dello script
files = ['Pilot0.csv', 'Pilot1.csv', 'Pilot2.csv', 'Pilot3.csv']

# Piatti serviti inseriti manualmente dalle tue note
manual_dishes = {
    'Pilot0.csv': [9, 11, 11, 11, 8, 12, 10, 12, 12, 13],
    'Pilot1.csv': [7, 12, 12, 13, 13, 14, 5, 16, 16, 13],
    'Pilot2.csv': [1, 3, 1, 4, 7, 0, 1, 5, 2, 4],
    'Pilot3.csv': [3, 17, 14, 19, 15, 16, 15, 6, 1, 15]
}

def analyze_behavior(file_path):
    df = pd.read_csv(file_path)
    rounds = sorted(df['Round'].unique())
    rep_scores, sim_scores = [], []
    prev_sequences_set = set()
    
    for r in rounds:
        round_data = df[df['Round'] == r]
        # Estrazione e semplificazione del percorso (rimuove i momenti da fermo)
        raw_path = list(zip(round_data['Human_X'], round_data['Human_Y']))
        path = [x for i, x in enumerate(raw_path) if i == 0 or x != raw_path[i-1]]
        
        if len(path) < 4:
            rep_scores.append(0.0); sim_scores.append(0.0); prev_sequences_set = set(); continue

        # Generazione sequenze di 4 passi (N-Grams)
        current_sequences_list = [tuple(path[i:i+4]) for i in range(len(path)-3)]
        current_sequences_set = set(current_sequences_list)

        # Calcolo Intra-round Repetition (Ripetizione interna al round)
        counts = Counter(current_sequences_list)
        repeats = sum(freq for seq, freq in counts.items() if freq > 1)
        rep_scores.append((repeats / len(current_sequences_list)) * 100 if current_sequences_list else 0)

        # Calcolo Inter-round Similarity (Similarità con il round precedente)
        if r == 1 or not prev_sequences_set:
            sim_scores.append(0.0)
        else:
            common = current_sequences_set.intersection(prev_sequences_set)
            sim_scores.append((len(common) / len(current_sequences_set)) * 100)
        
        prev_sequences_set = current_sequences_set
    return rep_scores, sim_scores

# --- 2. ELABORAZIONE DATI ---
all_rep, all_sim, all_dish = [], [], []
for f in files:
    rep, sim = analyze_behavior(f)
    all_rep.append(rep[:10]); all_sim.append(sim[:10])
    all_dish.append(manual_dishes[f][:10])

# Calcolo delle medie tra i 4 pilot
avg_rep = np.mean(all_rep, axis=0)
avg_sim = np.mean(all_sim, axis=0)
avg_dish = np.mean(all_dish, axis=0)
rounds_axis = np.arange(1, 11)

# --- 3. CREAZIONE DEL GRAFICO ---
fig, ax1 = plt.subplots(figsize=(13, 8))

# Zona di Stabilizzazione (Grigio chiaro)
ax1.axhspan(80, 100, color='gray', alpha=0.1, label='Stabilization Zone (>80%)')
ax1.text(0.5, 82, 'Stabilization Zone', fontsize=9, color='gray', fontweight='bold')

# Asse Y1: Percentuali (Linee con punti)
line1, = ax1.plot(rounds_axis, avg_sim, color='#4472C4', marker='o', markersize=8, 
                  linewidth=3, label='Inter-round Habit Similarity (%)')
line2, = ax1.plot(rounds_axis, avg_rep, color='#ED7D31', marker='o', markersize=8, 
                  linewidth=3, linestyle='--', label='Intra-round Repetition (%)')

ax1.set_xlabel('Round Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.grid(True, linestyle=':', alpha=0.6)

# Asse Y2: Piatti Serviti (Linea verde con punti)
ax2 = ax1.twinx()
line3, = ax2.plot(rounds_axis, avg_dish, color='darkgreen', marker='o', markersize=8, 
                  linewidth=3, label='Average Dishes Served')
ax2.set_ylabel('Average Dishes Served', color='darkgreen', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='darkgreen')
ax2.set_ylim(0, max(avg_dish) + 5)

# --- 4. LEGEND E STILE ---
plt.title('Pilot Analysis: Behavioral Habits & Performance Flow', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rounds_axis)

# Unione delle legende
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)

plt.tight_layout()
plt.show()