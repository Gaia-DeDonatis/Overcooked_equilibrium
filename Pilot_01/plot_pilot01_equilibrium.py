import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("pilot01_full_metrics.csv").copy()
df["global_round"] = range(1, len(df) + 1)

# Team reward score = 200 * dishes served
df["team_reward_score"] = 200 * df["dishes"]

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# ------------------------
# Phase backgrounds
# ------------------------
ax.axvspan(1, 15, color="lightblue", alpha=0.15)      # Seed
ax.axvspan(16, 36, color="lightgreen", alpha=0.15)    # BO
ax.axvspan(37, 45, color="lightyellow", alpha=0.18)   # Stress
ax.axvspan(46, 48, color="lightpink", alpha=0.18)     # Replay-optimal

# ------------------------
# Episode separators (every 3 rounds)
# ------------------------
for x in [3.5, 6.5, 9.5, 12.5, 15.5, 18.5, 21.5, 24.5, 27.5, 30.5, 33.5, 36.5, 39.5, 42.5, 45.5]:
    ax.axvline(x, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

# ------------------------
# Score lines
# ------------------------
ax.plot(
    df["global_round"],
    df["human_reward_score"],
    marker="o",
    linewidth=2,
    label="Human reward score"
)

ax.plot(
    df["global_round"],
    df["ai_reward_score"],
    marker="o",
    linewidth=2,
    label="AI reward score"
)

# Team reward as dots only
ax.scatter(
    df["global_round"],
    df["team_reward_score"],
    color="darkgrey",
    s=70,
    label="Team reward score"
)

# ------------------------
# Phase labels
# ------------------------
y_top = max(
    df["human_reward_score"].max(skipna=True),
    df["ai_reward_score"].max(skipna=True),
    df["team_reward_score"].max(skipna=True),
)

ax.text(8,  y_top * 1.03, "Seed", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(26, y_top * 1.03, "BO", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(41, y_top * 1.03, "Stress", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(47, y_top * 1.03, "Replay-optimal", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ------------------------
# Formatting
# ------------------------
ax.set_xlabel("Round")
ax.set_ylabel("Score")
ax.set_title("Pilot 01 — Reward Scores Across Rounds")
ax.set_xlim(1, 48)
ax.set_ylim(
    min(
        df["human_reward_score"].min(skipna=True),
        df["ai_reward_score"].min(skipna=True),
        df["team_reward_score"].min(skipna=True),
    ) - 100,
    y_top * 1.1
)

ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()