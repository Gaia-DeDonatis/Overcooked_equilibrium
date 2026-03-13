import matplotlib.pyplot as plt

# Pilot_02 Control data
rounds = list(range(1, 49))  # Rounds from 1 to 48

human_steps = [
    64, 68, 71,   # Ep1
    53, 45, 70,   # Ep2
    43, 69, 61,   # Ep3
    66, 43, 59,   # Ep4
    46, 43, 70,   # Ep5
    69, 71, 72,   # Ep6 Control
    79, 34, 61,   # Ep7 Control
    97, 95, 96,   # Ep8 Control solo
    24, 9, 62,    # Ep9 Control
    91, 91, 101,  # Ep10 Control solo
    26, 15, 38,   # Ep11 Control
    60, 67, 78,   # Ep12 Control
    57, 12, 20,   # Ep13 Stress
    68, 43, 55,   # Ep14 Stress
    11, 20, 19,   # Ep15 Stress
 
    83, 75, 97    # Ep16 Replay-optimal
]

ai_steps = [
    20, 24, 24,    # Ep1
    19, 48, 73,    # Ep2
    32, 72, 36,    # Ep3
    70, 33, 62,    # Ep4
    105, 140, 129, # Ep5
    97, 33, 64,    # Ep6 Control
    155, 36, 75,   # Ep7 Control
    0, 0, 0,       # Ep8 Control solo
    152, 164, 88,  # Ep9 Control
    0, 0, 0,       # Ep10 Control solo
    17, 17, 17,    # Ep11 Control
    137, 158, 100, # Ep12 Control
    95, 158, 44,   # Ep13 Stress
    104, 67, 117,  # Ep14 Stress
    44, 44, 44,    # Ep15 Stress
    155, 120, 167  # Ep16 Replay-optimal
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

# Phase boundaries
phase_boundaries = [15, 36, 45]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary axis: human and AI steps
line1, = ax1.plot(
    rounds, human_steps,
    marker='s', color='green',
    label='Human Steps', linestyle='-', markersize=6
)
line2, = ax1.plot(
    rounds, ai_steps,
    marker='^', color='red',
    label='AI Steps', linestyle='-', markersize=6
)

ax1.set_xlabel('Rounds')
ax1.set_ylabel('Steps')
ax1.set_title('Pilot_02 Control: Human and AI Steps, with Dishes Served and Phase Divisions')

# Secondary axis: dishes served
ax2 = ax1.twinx()
scatter = ax2.scatter(
    rounds, dishes_served,
    color='blue', label='Dishes Served', zorder=5
)
ax2.set_ylabel('Dishes Served')
ax2.set_ylim(0, 12)
ax2.set_yticks(range(0, 13))

# Mark phase boundaries
for ep in phase_boundaries:
    ax1.axvline(x=ep, color='black', linestyle='--', linewidth=1)

# Combine legends from both axes
handles = [line1, line2, scatter]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper left')

plt.tight_layout()
plt.show()