import os
import sys
import json
import time
import warnings
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import pygame
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# --- compatibility patch ---
try:
    sys.modules["numpy._core"] = np.core
    if hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    import numpy.random._pickle

    original_ctor = np.random._pickle.__bit_generator_ctor

    def patched_ctor(bit_generator_name):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np.random._pickle.__bit_generator_ctor = patched_ctor
except Exception:
    pass


# =========================
# SETTINGS
# =========================
TOTAL_ROUNDS = 1
ROUND_DURATION = 45
SECONDS_PER_STEP = 0.25

WINDOW_SIZE = (1200, 760)
HUD_HEIGHT = 90
BG_COLOR = (20, 20, 20)
HUD_COLOR = (240, 240, 240)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

IMAGE_DIR = os.path.join(CURRENT_DIR, "static", "images")
LOG_DIR = os.path.join(CURRENT_DIR, "logs_counter_test")

POLICY_POOL_DIR = os.path.join(PROJECT_ROOT, "policy_pool_counter_coplay")
CHECKPOINT_FILENAME = "model_1500000.zip"
POLICY_PREFIX = "[coplay][equilibrium][counter]agent0_"

KEYS_ACTIONS = {
    pygame.K_UP: 3,
    pygame.K_RIGHT: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
}
AGENT_KEYS = {
    pygame.K_1: 0,  # control AI agent directly
    pygame.K_2: 1,  # control human agent
}
ACTION_LABELS = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP", 4: "STAY"}

# Same image names used by the frontend
TILE_MAP = {
    0: "space.png",
    1: "counter.png",
    2: "space.png",          # agents drawn separately
    3: "FreshTomato.png",
    4: "FreshLettuce.png",
    5: "plate.png",
    6: "cutboard.png",
    7: "delivery.png",
    8: "FreshOnion.png",
    9: "dirtyplate.png",
    10: "BadLettuce.png",
}

DEFAULT_AGENT_IMAGE_BY_COLOR = {
    "robot": "agent-robot.png",
    "blue": "agent-blue.png",
    "red": "agent-red.png",
    "magenta": "agent-red.png",
    "green": "agent-blue.png",
    "yellow": "agent-blue.png",
}

FRONTEND_IMAGE_NAMES = [
    "space.png",
    "counter.png",
    "FreshTomato.png",
    "ChoppedTomato.png",
    "FreshLettuce.png",
    "ChoppedLettuce.png",
    "plate.png",
    "cutboard.png",
    "delivery.png",
    "FreshOnion.png",
    "ChoppedOnion.png",
    "dirtyplate.png",
    "BadLettuce.png",
    "agent-red.png",
    "agent-blue.png",
    "agent-robot.png",
]


def ensure_pygame_ready():
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Counter policy smoke test")
    return screen


def load_images():
    images = {}
    for name in FRONTEND_IMAGE_NAMES:
        path = os.path.join(IMAGE_DIR, name)
        if os.path.isfile(path):
            images[name] = pygame.image.load(path).convert_alpha()
        else:
            surf = pygame.Surface((64, 64), pygame.SRCALPHA)
            surf.fill((200, 0, 200, 255))
            images[name] = surf
    return images


def _jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def safe_json_dumps(obj) -> str:
    try:
        return json.dumps(_jsonable(obj), ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def get_macro_action_name(env, macro_id: int) -> str:
    try:
        names = getattr(env, "macroActionName", None)
        if names is not None and 0 <= macro_id < len(names):
            return str(names[macro_id])
    except Exception:
        pass
    return ""


def draw_text(screen, text, pos, size=24, color=HUD_COLOR, center=False):
    font = pygame.font.SysFont("arial", size)
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    screen.blit(surf, rect)


def resolve_obj_image_name(obj):
    if obj is None:
        return None

    # Plate content can be a list
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return None
        return resolve_obj_image_name(obj[0])

    cls = obj.__class__.__name__.lower()

    if "badlettuce" in cls:
        return "BadLettuce.png"
    if "lettuce" in cls:
        return "ChoppedLettuce.png" if getattr(obj, "chopped", False) else "FreshLettuce.png"
    if "tomato" in cls:
        return "ChoppedTomato.png" if getattr(obj, "chopped", False) else "FreshTomato.png"
    if "onion" in cls:
        return "ChoppedOnion.png" if getattr(obj, "chopped", False) else "FreshOnion.png"
    if "dirtyplate" in cls:
        return "dirtyplate.png"
    if cls == "plate":
        return "plate.png"
    if "knife" in cls:
        return "cutboard.png"
    if "delivery" in cls:
        return "delivery.png"

    return None


def get_agent_image_name(agent):
    color = getattr(agent, "color", "blue")
    return DEFAULT_AGENT_IMAGE_BY_COLOR.get(color, "agent-blue.png")


def draw_plate_with_content(screen, images, plate_img_name, content_obj, px, py, cell):
    base = pygame.transform.smoothscale(images[plate_img_name], (cell, cell))
    screen.blit(base, (px, py))

    content_name = resolve_obj_image_name(content_obj)
    if content_name:
        content = pygame.transform.smoothscale(
            images[content_name],
            (int(cell * 0.65), int(cell * 0.65))
        )
        rect = content.get_rect(center=(px + cell // 2, py + cell // 2))
        screen.blit(content, rect)


def draw_game(screen, env, images, policy_name, round_num, selected_agent, step_idx, time_left):
    screen.fill(BG_COLOR)

    rows, cols = int(env.xlen), int(env.ylen)

    margin = 24
    avail_w = WINDOW_SIZE[0] - 2 * margin
    avail_h = WINDOW_SIZE[1] - HUD_HEIGHT - 2 * margin
    cell = int(min(avail_w / cols, avail_h / rows))

    board_w = cell * cols
    board_h = cell * rows
    off_x = (WINDOW_SIZE[0] - board_w) // 2
    off_y = HUD_HEIGHT + (avail_h - board_h) // 2

    display_name = policy_name if len(policy_name) <= 52 else policy_name[:49] + "..."
    draw_text(screen, f"Policy: {display_name}", (24, 16), size=22)
    draw_text(screen, f"Round: {round_num}/{TOTAL_ROUNDS}", (24, 48), size=22)
    draw_text(screen, f"Time left: {max(0, int(time_left))}s", (420, 16), size=22)
    draw_text(screen, f"Step: {step_idx}", (420, 48), size=22)
    draw_text(screen, "Arrows = move human | 1/2 = select agent", (760, 28), size=20)

    # 1. Base map exactly like frontend style
    for x in range(rows):
        for y in range(cols):
            code = int(env.map[x][y])
            tile_name = TILE_MAP.get(code, "space.png")
            img = pygame.transform.smoothscale(images[tile_name], (cell, cell))
            screen.blit(img, (off_x + y * cell, off_y + x * cell))

    # 2. Items on counters, like frontend state.items
    holding_cells = {(int(a.x), int(a.y)) for a in getattr(env, "agent", [])}

    item_groups = [
        getattr(env, "tomato", []),
        getattr(env, "lettuce", []),
        getattr(env, "badlettuce", []),
        getattr(env, "onion", []),
        getattr(env, "plate", []),
        getattr(env, "dirtyplate", []),
        getattr(env, "knife", []),
        getattr(env, "delivery", []),
    ]

    for group in item_groups:
        for item in group:
            x, y = int(item.x), int(item.y)
            if (x, y) in holding_cells:
                continue

            px = off_x + y * cell
            py = off_y + x * cell

            # Draw counter base under movable items, matching frontend
            cls = item.__class__.__name__.lower()
            movable_item = any(k in cls for k in ["lettuce", "badlettuce", "tomato", "onion", "plate", "dirtyplate"])
            if movable_item and "counter.png" in images:
                counter_img = pygame.transform.smoothscale(images["counter.png"], (cell, cell))
                screen.blit(counter_img, (px, py))

            item_name = resolve_obj_image_name(item)
            if item_name is None:
                continue

            if item_name in {"plate.png", "dirtyplate.png"}:
                draw_plate_with_content(screen, images, item_name, getattr(item, "containing", None), px, py, cell)
            else:
                img = pygame.transform.smoothscale(images[item_name], (cell, cell))
                screen.blit(img, (px, py))

                containing = getattr(item, "containing", None)
                holding = getattr(item, "holding", None)

                if containing is not None:
                    content_name = resolve_obj_image_name(containing)
                    if content_name:
                        content_img = pygame.transform.smoothscale(images[content_name], (cell, cell))
                        screen.blit(content_img, (px, py))

                if holding is not None:
                    holding_name = resolve_obj_image_name(holding)
                    if holding_name:
                        hold_img = pygame.transform.smoothscale(images[holding_name], (cell, cell))
                        screen.blit(hold_img, (px, py))

    # 3. Agents
    for idx, agent in enumerate(getattr(env, "agent", [])):
        x, y = int(agent.x), int(agent.y)
        px = off_x + y * cell
        py = off_y + x * cell

        agent_img_name = get_agent_image_name(agent)
        agent_img = pygame.transform.smoothscale(images[agent_img_name], (cell, cell))
        screen.blit(agent_img, (px, py))

        if idx == selected_agent:
            pygame.draw.rect(screen, (255, 255, 0), (px, py, cell, cell), 4)

        if getattr(agent, "holding", None) is not None:
            hold_name = resolve_obj_image_name(agent.holding)
            if hold_name:
                hold_img = pygame.transform.smoothscale(images[hold_name], (int(cell * 0.5), int(cell * 0.5)))
                hx = px + cell - hold_img.get_width()
                hy = py + cell - hold_img.get_height()
                screen.blit(hold_img, (hx, hy))

    pygame.display.flip()


class SingleAgentWrapperAcceptKeyboardAction(gym.Wrapper):
    def __init__(self, env, agent_index):
        super().__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.obs = None

        self.last_ai_reward = 0.0
        self.last_human_reward = 0.0
        self.last_team_reward = 0.0

    def reset(self):
        self.obs = self.env.reset()
        self.last_ai_reward = 0.0
        self.last_human_reward = 0.0
        self.last_team_reward = 0.0
        return self.obs[self.agent_index]

    def step(self, action_agent0, action_agent1):
        actions = [action_agent0, action_agent1]
        obs, rewards, dones, info = self.env.step(actions)
        self.obs = self.env._get_macro_obs()

        rr = np.array(rewards).flatten()
        self.last_ai_reward = float(rr[0]) if rr.size > 0 else 0.0
        self.last_human_reward = float(rr[1]) if rr.size > 1 else 0.0
        self.last_team_reward = float(rr[2]) if rr.size > 2 else (self.last_ai_reward + self.last_human_reward)

        return self.obs[self.agent_index], self.last_team_reward, dones, info


def make_env():
    mac_env_id = "Overcooked-MA-equilibrium-v1"

    rewardList = [{
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": -20,
        "pick up bad lettuce": -100
    }, {
        "minitask finished": 0,
        "minitask failed": 0,
        "metatask finished": 0,
        "metatask failed": 0,
        "goodtask finished": 10,
        "goodtask failed": 0,
        "subtask finished": 20,
        "subtask failed": 0,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1,
        "penalize using dirty plate": 0,
        "penalize using bad lettuce": 0,
        "pick up bad lettuce": 0
    }]

    env_params = {
        "grid_dim": [5, 8],
        "task": ["lettuce salad"],
        "rewardList": rewardList,
        "map_type": "counter",
        "n_agent": 2,
        "obs_radius": 0,
        "mode": "vector",
        "debug": False,  # use our own renderer, not the broken built-in one
    }

    import gym_macro_overcooked
    base_env = gym.make(mac_env_id, **env_params)
    wrapped = SingleAgentWrapperAcceptKeyboardAction(base_env, agent_index=0)
    return base_env, wrapped


def list_policy_checkpoints(policy_pool_dir, checkpoint_filename, prefix=None):
    if not os.path.isdir(policy_pool_dir):
        raise FileNotFoundError(f"Policy pool directory not found: {policy_pool_dir}")

    found = []
    for name in sorted(os.listdir(policy_pool_dir)):
        folder = os.path.join(policy_pool_dir, name)
        if not os.path.isdir(folder):
            continue
        if prefix and prefix not in name:
            continue

        ckpt = os.path.join(folder, checkpoint_filename)
        if os.path.isfile(ckpt):
            found.append((name, ckpt))

    return found


def run_one_round(round_num, base_env, env_agent_0, agent_model, policy_name, screen, images):
    screen = ensure_pygame_ready()
    obs = env_agent_0.reset()
    env_draw = base_env.unwrapped

    selected_agent = 1
    current_human_action = 4
    step_idx = 0

    human_steps = 0
    ai_steps = 0
    human_reward_total = 0.0
    ai_reward_total = 0.0
    team_reward_total = 0.0
    dishes_served = 0

    start_time = time.time()
    last_tick = time.time()

    draw_game(
        screen=screen,
        env=env_draw,
        images=images,
        policy_name=policy_name,
        round_num=round_num,
        selected_agent=selected_agent,
        step_idx=step_idx,
        time_left=ROUND_DURATION,
    )

    aborted = False
    running = True

    while running:
        elapsed = time.time() - start_time
        if elapsed >= ROUND_DURATION:
            running = False
            break

        screen = ensure_pygame_ready()

        try:
            events = pygame.event.get()
        except pygame.error:
            events = []

        for event in events:
            if event.type == pygame.QUIT:
                aborted = True
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    aborted = True
                    running = False
                    break

                if event.key in AGENT_KEYS:
                    selected_agent = AGENT_KEYS[event.key]

                if event.key in KEYS_ACTIONS:
                    current_human_action = KEYS_ACTIONS[event.key]

        if not running:
            break

        now = time.time()
        if (now - last_tick) >= SECONDS_PER_STEP:
            last_tick = now
            step_idx += 1

            ai_action_macro, _ = agent_model.predict(obs, deterministic=True)
            if isinstance(ai_action_macro, np.ndarray):
                ai_action_macro = int(ai_action_macro.item())
            else:
                ai_action_macro = int(ai_action_macro)

            _ = get_macro_action_name(env_draw, ai_action_macro)

            prim_pair, _ = env_draw._computeLowLevelActions([ai_action_macro, 0])
            ai_action_primitive = int(prim_pair[0])

            if selected_agent == 1:
                a0 = ai_action_primitive
                a1 = current_human_action
                human_action_used = a1
                ai_action_used = a0
            else:
                a0 = current_human_action
                a1 = 4
                human_action_used = a0
                ai_action_used = ""

            obs_after, reward, dones, info = env_agent_0.step(a0, a1)

            # Count action-steps (simple and clean)
            human_steps += int(human_action_used in [0, 1, 2, 3])
            ai_steps += int(ai_action_used in [0, 1, 2, 3])

            human_reward_total += env_agent_0.last_human_reward
            ai_reward_total += env_agent_0.last_ai_reward
            team_reward_total += env_agent_0.last_team_reward

            # Simple delivery counter:
            # a correct delivery gives a large positive reward (~200) to one agent
            if max(env_agent_0.last_ai_reward, env_agent_0.last_human_reward) >= 100:
                dishes_served += 1

            obs = obs_after
            current_human_action = 4

            draw_game(
                screen=screen,
                env=env_draw,
                images=images,
                policy_name=policy_name,
                round_num=round_num,
                selected_agent=selected_agent,
                step_idx=step_idx,
                time_left=ROUND_DURATION - elapsed,
            )

        time.sleep(0.005)

    round_summary = {
        "policy_name": policy_name,
        "round": round_num,
        "dishes_served": dishes_served,
        "human_steps": human_steps,
        "ai_steps": ai_steps,
        "human_reward": human_reward_total,
        "ai_reward": ai_reward_total,
        "team_reward": team_reward_total,
    }

    return round_summary, aborted


def load_model(model_path, env_agent_0):
    custom_objects = {
        "action_space": env_agent_0.action_space,
        "observation_space": env_agent_0.observation_space,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0
    }
    return PPO.load(model_path, env=env_agent_0, custom_objects=custom_objects)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    policies = list_policy_checkpoints(
        POLICY_POOL_DIR,
        CHECKPOINT_FILENAME,
        prefix=POLICY_PREFIX
    )

    print(f"Found {len(policies)} policies")
    for i, (name, ckpt) in enumerate(policies, start=1):
        print(f"{i:02d}. {name}")
        print(f"    {ckpt}")

    if not policies:
        raise RuntimeError("No policy checkpoints found. Check POLICY_POOL_DIR and CHECKPOINT_FILENAME.")

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Counter policy smoke test")
    images = load_images()

    base_env, env_agent_0 = make_env()
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    global_summary_rows = []
    aborted = False

    for policy_idx, (policy_name, model_path) in enumerate(policies, start=1):
        print("\n" + "=" * 80)
        print(f"Testing policy {policy_idx}/{len(policies)}")
        print(policy_name)
        print("=" * 80)

        agent_model = load_model(model_path, env_agent_0)
        policy_round_rows = []

        for r in range(1, TOTAL_ROUNDS + 1):
            pygame.display.set_caption(
                f"Counter test | policy {policy_idx}/{len(policies)} | round {r}/{TOTAL_ROUNDS}"
            )

            round_summary, aborted = run_one_round(
                round_num=r,
                base_env=base_env,
                env_agent_0=env_agent_0,
                agent_model=agent_model,
                policy_name=policy_name,
                screen=screen,
                images=images,
            )

            policy_round_rows.append(round_summary)
            global_summary_rows.append(round_summary)

            if aborted:
                break

            time.sleep(0.5)

        # one CSV per policy
        policy_csv_path = os.path.join(
            LOG_DIR,
            f"{session_ts}__policy_{policy_idx:02d}__summary.csv"
        )
        pd.DataFrame(policy_round_rows).to_csv(policy_csv_path, index=False)
        print("[Policy summary saved]", policy_csv_path)

        del agent_model

        if aborted:
            print("Aborted by user.")
            break

    # one global CSV
    global_csv_path = os.path.join(LOG_DIR, f"{session_ts}__ALL_POLICIES_SUMMARY.csv")
    pd.DataFrame(global_summary_rows).to_csv(global_csv_path, index=False)
    print("[Global summary saved]", global_csv_path)

    try:
        base_env.close()
    except Exception:
        pass

    pygame.quit()


if __name__ == "__main__":
    main()