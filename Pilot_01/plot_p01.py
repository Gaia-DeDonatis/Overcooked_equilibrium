import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("pilot01_full_metrics.csv").copy()
df["global_round"] = range(1, len(df) + 1)

# Derived score
df["team_score"] = 200 * df["dishes"]

# Best episodes: 9 and 11
best_mask = df["episode"].isin([9, 11])

# ------------------------
# Figure
# ------------------------
fig, axes = plt.subplots(
    3, 1,
    figsize=(14, 18),
    sharex=True,
    constrained_layout=True
)

episode_boundaries = [3.5, 6.5, 9.5, 12.5, 15.5, 18.5, 21.5, 24.5,
                      27.5, 30.5, 33.5, 36.5, 39.5, 42.5, 45.5]

def decorate_axis(ax, ymax):
    # Phase backgrounds
    ax.axvspan(0.5, 15.5, color="lightblue", alpha=0.12)   # Seed
    ax.axvspan(15.5, 36.5, color="lightgreen", alpha=0.12) # BO
    ax.axvspan(36.5, 45.5, color="khaki", alpha=0.16)      # Stress
    ax.axvspan(45.5, 48.5, color="lightpink", alpha=0.16)  # Replay-optimal

    # Solo episodes
    ax.axvspan(21.5, 24.5, color="red", alpha=0.07)   # Episode 8
    ax.axvspan(27.5, 30.5, color="red", alpha=0.07)   # Episode 10

    # Episode separators
    for sep in episode_boundaries:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Phase labels
    ax.text(8,  ymax * 1, "Seed", ha="center", va="bottom", fontweight="bold")
    ax.text(26, ymax * 1, "BO", ha="center", va="bottom", fontweight="bold")
    ax.text(41, ymax * 1, "Stress", ha="center", va="bottom", fontweight="bold")
    ax.text(47, ymax * 1, "Replay-optimal", ha="center", va="bottom", fontweight="bold")

    # Solo labels
    ax.text(23, ymax * 0.92, "Solo\n(Ep 8)", ha="center", va="top", color="red", fontweight="bold")
    ax.text(29, ymax * 0.92, "Solo\n(Ep 10)", ha="center", va="top", color="red", fontweight="bold")

    ax.grid(True, alpha=0.25)

# ------------------------
# 1) Team score
# ------------------------
ax = axes[0]
ax.scatter(
    df["global_round"],
    df["team_score"],
    s=70,
    color="tab:blue",
    label="Team score (200 × dishes served)"
)

ax.scatter(
    df.loc[best_mask, "global_round"],
    df.loc[best_mask, "team_score"],
    s=180,
    facecolors="none",
    edgecolors="red",
    linewidths=2.2
)

ax.set_ylabel("Team score")
ax.set_title("Pilot 01 — Team Score, AI Reward Score, and Human Steps")
decorate_axis(ax, df["team_score"].max())
ax.legend(loc="upper left")

# ------------------------
# 2) AI reward score
# ------------------------
ax = axes[1]
ai_mask = df["ai_reward_score"].notna()

ax.scatter(
    df.loc[ai_mask, "global_round"],
    df.loc[ai_mask, "ai_reward_score"],
    s=70,
    color="tab:orange",
    label="AI reward score"
)

ai_best_mask = ai_mask & best_mask
ax.scatter(
    df.loc[ai_best_mask, "global_round"],
    df.loc[ai_best_mask, "ai_reward_score"],
    s=180,
    facecolors="none",
    edgecolors="red",
    linewidths=2.2
)

ax.set_ylabel("AI reward score")
decorate_axis(ax, df["ai_reward_score"].max(skipna=True))
ax.legend(loc="upper left")

# ------------------------
# 3) Human steps + subjective ratings
# ------------------------
ax = axes[2]

ax.scatter(
    df["global_round"],
    df["human_steps"],
    s=70,
    color="tab:green",
    label="Human steps",
    zorder=3
)

ax.scatter(
    df.loc[best_mask, "global_round"],
    df.loc[best_mask, "human_steps"],
    s=180,
    facecolors="none",
    edgecolors="red",
    linewidths=2.2,
    zorder=4
)

ax.set_ylabel("Human steps")
decorate_axis(ax, df["human_steps"].max(skipna=True))

# Episode-level ratings repeated across the 3 rounds of each episode
md =  [4, 14, 10, 9, 10, 8, 5, 1, 3, 1, 2, 7, 13, 11, 12, 11]
perf = [17, 4, 14, 15, 10, 15, 17, 17, 19, 16, 20, 18, 12, 15, 13, 12]

md_round = np.repeat(md, 3)
perf_round = np.repeat(perf, 3)

ax_r = ax.twinx()
ax_r.patch.set_alpha(0)

ax_r.step(
    df["global_round"],
    md_round,
    where="mid",
    color="tab:blue",
    linestyle="--",
    linewidth=2,
    alpha=0.45,
    label="Mental demand",
    zorder=1
)

ax_r.step(
    df["global_round"],
    perf_round,
    where="mid",
    color="tab:orange",
    linestyle="--",
    linewidth=2,
    alpha=0.45,
    label="Performance rating",
    zorder=1
)

ax_r.set_ylabel("Episode ratings")
ax_r.set_ylim(0, 21)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# ------------------------
# Final formatting
# ------------------------
axes[-1].set_xlabel("Round")
axes[-1].set_xlim(0.5, 48.5)
axes[-1].set_xticks(np.arange(1, 49))
axes[-1].set_xticklabels(np.arange(1, 49), fontsize=8)

plt.savefig("pilot01_plots_large.png", dpi=300, bbox_inches="tight")
plt.show()