import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pilot01_full_metrics.csv")

# remove solo rounds
df = df[df["ai_reward_score"].notna()].copy()

df["global_round"] = range(1, len(df) + 1)

fig, ax1 = plt.subplots(figsize=(10,6))

# ------------------------
# Axis 1: Reward scores
# ------------------------

ax1.plot(df["global_round"], df["human_reward_score"],
         marker="o", label="Human reward score", color="tab:blue")

ax1.plot(df["global_round"], df["ai_reward_score"],
         marker="o", label="AI reward score", color="tab:orange")

ax1.set_xlabel("Round")
ax1.set_ylabel("Reward Score")

# highlight equilibrium episodes
eq = df[df["episode"].isin([9,11])]

ax1.scatter(eq["global_round"], eq["human_reward_score"], color="red", s=120)
ax1.scatter(eq["global_round"], eq["ai_reward_score"], color="red", s=120)

# ------------------------
# Axis 2: Dishes
# ------------------------

ax2 = ax1.twinx()

ax2.scatter(df["global_round"], df["dishes"],
            color="green", s=80, label="Dishes served")

ax2.set_ylabel("Dishes Served")

# ------------------------

plt.title("Chef Role Contribution and Team Performance")

ax1.grid(True)

# combined legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines + lines2, labels + labels2)

plt.show()