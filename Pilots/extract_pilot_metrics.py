import json
import csv

# =========================
# CONFIG
# =========================
json_path = r"C:\Users\dedong1\work\Overcooked_equilibrium\submissions\Pilot_04_16_03\final_result.json"
output_csv = r"C:\Users\dedong1\work\Overcooked_equilibrium\Pilots\extracted_results.csv"


# =========================
# HELPERS
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    for key in keys:
        if isinstance(key, tuple):
            val = d
            ok = True
            for k in key:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    ok = False
                    break
            if ok:
                return val
        else:
            if isinstance(d, dict) and key in d:
                return d[key]
    return default


def find_first_value(obj, candidate_keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in candidate_keys:
                return v
        for v in obj.values():
            found = find_first_value(v, candidate_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_value(item, candidate_keys)
            if found is not None:
                return found
    return None


def count_dishes(round_obj):
    explicit = find_first_value(
        round_obj,
        {
            "dishes_per_round",
            "dishes",
            "delivered_dishes",
            "num_dishes",
            "dish_count",
        },
    )
    if isinstance(explicit, (int, float)):
        return explicit
    if isinstance(explicit, list):
        return len(explicit)
    return None


def extract_episode_scores(episode_obj):
    mental_demand_score = find_first_value(
        episode_obj,
        {
            "mental_demand_score",
            "mental_demand",
            "mentalDemand",
            "nasa_tlx_mental_demand",
        },
    )

    performance_score = find_first_value(
        episode_obj,
        {
            "performance_score",
            "performance",
            "nasa_tlx_performance",
        },
    )

    return mental_demand_score, performance_score


def extract_round_metrics(round_obj):
    return {
        "human_steps": find_first_value(round_obj, {"human_steps", "player_steps", "steps_human"}),
        "ai_steps": find_first_value(round_obj, {"ai_steps", "agent_steps", "steps_ai"}),
        "human_score": find_first_value(round_obj, {"human_score", "player_score", "score_human"}),
        "ai_score": find_first_value(round_obj, {"ai_score", "agent_score", "score_ai"}),
        "human_reward_score": find_first_value(
            round_obj,
            {"human_reward_score", "player_reward_score", "reward_human"},
        ),
        "ai_reward_score": find_first_value(
            round_obj,
            {"ai_reward_score", "agent_reward_score", "reward_ai"},
        ),
        "dishes_per_round": count_dishes(round_obj),
    }


def extract_rows(data):
    rows = []

    if isinstance(data, dict):
        episodes = data.get("episodes", None)
        if episodes is None:
            if "rounds" in data or "episode" in data:
                episodes = [data]
            else:
                possible = list(data.values())
                if possible and all(isinstance(x, dict) for x in possible):
                    episodes = possible
                else:
                    episodes = []
    elif isinstance(data, list):
        episodes = data
    else:
        episodes = []

    for ep_idx, episode_obj in enumerate(episodes, start=1):
        if not isinstance(episode_obj, dict):
            continue

        episode_id = safe_get(
            episode_obj,
            "episode",
            "episode_id",
            "ep",
            default=ep_idx,
        )

        mental_demand_score, performance_score = extract_episode_scores(episode_obj)

        rounds = safe_get(
            episode_obj,
            "rounds",
            "results",
            "round_data",
            default=[],
        )

        if not isinstance(rounds, list) or len(rounds) == 0:
            rounds = [episode_obj]

        for rd_idx, round_obj in enumerate(rounds, start=1):
            if not isinstance(round_obj, dict):
                continue

            round_id = safe_get(
                round_obj,
                "round",
                "round_id",
                "trial",
                default=rd_idx,
            )

            metrics = extract_round_metrics(round_obj)

            row = {
                "episode": episode_id,
                "round": round_id,
                "human_steps": metrics["human_steps"],
                "ai_steps": metrics["ai_steps"],
                "human_score": metrics["human_score"],
                "ai_score": metrics["ai_score"],
                "human_reward_score": metrics["human_reward_score"],
                "ai_reward_score": metrics["ai_reward_score"],
                "dishes_per_round": metrics["dishes_per_round"],
                "mental_demand_score": mental_demand_score,
                "performance_score": performance_score,
            }
            rows.append(row)

    return rows


def save_csv(rows, path):
    fieldnames = [
        "episode",
        "round",
        "human_steps",
        "ai_steps",
        "human_score",
        "ai_score",
        "human_reward_score",
        "ai_reward_score",
        "dishes_per_round",
        "mental_demand_score",
        "performance_score",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = load_json(json_path)
    rows = extract_rows(data)
    save_csv(rows, output_csv)

    print(f"Done. Extracted {len(rows)} rows.")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()