import matplotlib.pyplot as plt

# Assuming you have the following data in the form of lists
rounds = list(range(1, 49))  # Rounds from 1 to 48
human_steps = [20, 81, 86, 72, 79, 52, 67, 94, 99, 87, 73, 102, 99, 95, 76, 92, 86, 13, 96, 101, 102, 103, 113, 110, 65, 54, 53, 114, 102, 112, 60, 54, 55, 63, 89, 9, 64, 57, 86, 42, 40, 20, 49, 102, 104, 72, 78, 68]
ai_steps = [29, 149, 95, 59, 59, 61, 44, 85, 96, 87, 57, 105, 4, 3, 5, 45, 65, 60, 15, 16, 17, 0, 0, 0, 50, 52, 53, 0, 0, 0, 47, 47, 49, 57, 73, 160, 60, 52, 76, 45, 41, 22, 10, 13, 4, 62, 65, 59]
dishes_served = [2, 9, 6, 0, 0, 2, 4, 11, 10, 6, 2, 12, 4, 2, 3, 8, 7, 9, 6, 7, 6, 6, 6, 6, 10, 11, 12, 7, 5, 6, 11, 11, 11, 2, 4, 11, 8, 5, 7, 7, 4, 11, 4, 5, 7, 5, 8, 9]

# Phase boundaries
phase_boundaries = [15, 36, 45]

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary axis: human and AI steps
line1, = ax1.plot(rounds, human_steps, marker='s', color='green',
                  label='Human Steps', linestyle='-', markersize=6)
line2, = ax1.plot(rounds, ai_steps, marker='^', color='red',
                  label='AI Steps', linestyle='-', markersize=6)

ax1.set_xlabel('Rounds')
ax1.set_ylabel('Steps')
ax1.set_title('Human and AI Steps, with Dishes Served and Phase Divisions')

# Secondary axis: dishes served
ax2 = ax1.twinx()
scatter = ax2.scatter(rounds, dishes_served, color='blue',
                      label='Dishes Served', zorder=5)
ax2.set_ylabel('Dishes Served')

# Mark phase boundaries
for ep in phase_boundaries:
    ax1.axvline(x=ep, color='black', linestyle='--', linewidth=1)

# Combine legends from both axes
handles = [line1, line2, scatter]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper left')

plt.tight_layout()
plt.show()