import matplotlib.pyplot as plt

# Data: Round-wise Human, AI Reward Scores, and Dishes Served
rounds = list(range(1, 49))  # Round numbers from 1 to 48

human_rewards = [
    426, 422, 419,   # Ep1
    27, -25, 170,    # Ep2
    -3, 11, 19,      # Ep3
    194, 17, 201,    # Ep4
    44, 47, 20,      # Ep5
    -9, 19, 18,      # Ep6 Control
    401, -34, 229,   # Ep7 Control
    633, 635, 434,   # Ep8 Control solo
    -24, -9, 178,    # Ep9 Control
    639, 439, 649,   # Ep10 Control solo
    -26, -15, -38,   # Ep11 Control
    20, 173, 182,    # Ep12 Control
    -17, -12, -20,   # Ep13 Stress
    212, 7, 185,     # Ep14 Stress
    -11, -20, -19,   # Ep15 Stress
    397, 185, 423    # Ep16 Replay-optimal
]

ai_rewards = [
    -20, -24, -24,   # Ep1
    231, 442, 657,   # Ep2
    218, 178, 214,   # Ep3
    30, 217, 158,    # Ep4
    -15, 0, 111,     # Ep5
    -47, 417, 336,   # Ep6 Control
    -125, 224, 145,  # Ep7 Control
    0, 0, 0,         # Ep8 Control solo
    1098, 1086, 442, # Ep9 Control
    0, 0, 0,         # Ep10 Control solo
    13, 13, 13,      # Ep11 Control
    63, 682, 340,    # Ep12 Control
    335, 102, 226,   # Ep13 Stress
    336, 83, 853,    # Ep14 Stress
    226, 226, 226,   # Ep15 Stress
    665, 260, 213    # Ep16 Replay-optimal
]

dishes_served = [
    2, 2, 2,   # Ep1
    1, 2, 4,   # Ep2
    1, 1, 1,   # Ep3
    2, 1, 2,   # Ep4
    1, 1, 1,   # Ep5
    1, 2, 2,   # Ep6 Control
    2, 1, 2,   # Ep7 Control
    3, 3, 2,   # Ep8 Control solo
    5, 5, 3,   # Ep9 Control
    3, 2, 3,   # Ep10 Control solo
    0, 0, 0,   # Ep11 Control
    2, 5, 3,   # Ep12 Control
    2, 1, 1,   # Ep13 Stress
    3, 1, 5,   # Ep14 Stress
    1, 1, 1,   # Ep15 Stress
    6, 3, 5    # Ep16 Replay-optimal
]

# Phase boundaries:
# Seed = rounds 1-15
# Control = rounds 16-36
# Stress = rounds 37-45
# Replay-optimal = rounds 46-48
phase_boundaries = [15, 36, 45]

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Human and AI reward scores on the left y-axis
ax1.plot(rounds, human_rewards, marker='o', color='green', label='Human Reward Score')
ax1.plot(rounds, ai_rewards, marker='o', color='red', label='AI Reward Score')

ax1.set_xlabel('Rounds')
ax1.set_ylabel('Reward Scores')
ax1.set_title('Pilot_02 Control: Human vs AI Reward Scores with Dishes Served')

# Secondary y-axis for dishes served
ax2 = ax1.twinx()
ax2.plot(rounds, dishes_served, 'o', label='Dishes Served')
ax2.set_ylabel('Dishes Served')
ax2.set_ylim(0, 12)

# Add phase dividers
for phase in phase_boundaries:
    ax1.axvline(x=phase, linestyle='--', linewidth=1)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()