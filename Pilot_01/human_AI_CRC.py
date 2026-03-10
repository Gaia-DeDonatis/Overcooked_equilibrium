import matplotlib.pyplot as plt

# Data: Round-wise Human, AI Reward Scores, and Dishes Served
rounds = list(range(1, 49))  # Round numbers from 1 to 48
human_rewards = [
    -20, -81, -86, 18, -29, 238, 463, 1356, 1151, 683, 217, 1368, 871, 365, 604, 1158, 674, -13, 1374, 1589, 1388, 1347, 1377, 1380,
    -65, -54, -53, 1576, 1148, 1378, -60, -54, -55, 67, 121, -9, 466, 423, 674, 228, -50, 220, 876, 1108, 1576, 168, 162, 172
]  # Human reward scores
ai_rewards = [
    461, 2041, 1355, -139, -167, 85, 399, 1069, 1083, 672, 119, 1351, -2, -3, -5, 610, 834, 2032, -85, -81, -92, 0, 0, 0, 2307,
    2538, 2734, 0, 0, 0, 2497, 2537, 2521, 251, 616, 2480, 1299, 673, 874, 1293, 908, 2255, -54, -63, -8, 870, 1597, 1827
]  # AI reward scores
dishes_served = [
    2, 9, 6, 0, 0, 2, 4, 11, 10, 6, 2, 12, 4, 2, 3, 8, 7, 9, 6, 7, 6, 6, 6, 6, 10, 11, 12, 7, 5, 6, 11, 11, 11, 7, 4, 5, 7, 5, 8, 7, 11, 8, 9, 9, 9, 5, 8, 9  # Number of dishes served each round
]

# Define the phases
phase_boundaries = [15, 36, 45]  # The round indices that mark phase changes: Seed (1-5), BO (6-12), Stress (13-15), Replay-optimal (16)

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Human and AI reward scores on the left y-axis
ax1.plot(rounds, human_rewards, marker='o', color='green', label='Human Reward Score')
ax1.plot(rounds, ai_rewards, marker='o', color='red', label='AI Reward Score')

ax1.set_xlabel('Rounds')
ax1.set_ylabel('Reward Scores')
ax1.set_title('Human vs AI Reward Scores with Dishes Served')

# Add a secondary y-axis for dishes served as dots
ax2 = ax1.twinx()
ax2.plot(rounds, dishes_served, 'bo', label='Dishes Served')  # Dishes served as blue dots
ax2.set_ylabel('Dishes Served')

# Add phase dividers (vertical lines to separate phases)
for phase in phase_boundaries:
    ax1.axvline(x=phase, color='black', linestyle='--', linewidth=1)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()