# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
from BayesOpt import TSNEBayesOptimizer, create_surrogate_spec
import csv
import pandas as pd

# Equilibrium_project" folder
current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(current_dir, 'Equilibrium_project')

# =========================
# SUBMISSIONS ROOT
# =========================

# DEFAULT (pilot tests): save locally in ./submissions
SUBMISSIONS_ROOT = os.path.join(current_dir, "submissions")

# FUTURE: when pilot tests are finished and the real experiment starts,
# comment the line above and uncomment ONE of the options below.

# --- Option A: TeamWork path on Windows (if backend runs on your Windows PC) ---
# SUBMISSIONS_ROOT = r"\\work.org.aalto.fi\T412\T40710\OverCookedHAIC\participant_logs\main_study"

# --- Option B: TeamWork path on Linux/server ---
# Use this ONLY if TeamWork has first been mounted on the server.
# Example mount point:
# SUBMISSIONS_ROOT = "/mnt/teamwork/OverCookedHAIC/participant_logs/main_study"

# Insert at index 0 so this folder takes priority over everything else
sys.path.insert(0, target_folder) 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["PYTHONHASHSEED"] = "0"

# Prevent pygame/SDL from trying to initialize display on macOS
# This is required because Flask handles requests in background threads,
# and SDL2 cannot set the main menu from a non-main thread on macOS
os.environ["SDL_VIDEODRIVER"] = "dummy"




import json
import uuid
import time
import math
import datetime as dt
import threading
from collections import deque

import gym
import gym as _gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
except Exception:
    pass

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging

# Suppress Flask/Werkzeug request logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

from logging_utils import backend_logger as logger
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
import random
import gym_macro_overcooked

logger.info("[ENV - INIT] loaded gym_macro_overcooked (Environments: Overcooked-equilibrium-v0)")

from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate


STATIC_DIR = os.path.join(target_folder, "static")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)
CORS(app)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def _seed_env_everything(env, seed: int):
    try:
        env.reset(seed=seed)
    except Exception:
        try:
            env.reset()
        except Exception:
            pass

    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    try:
        env.observation_space.seed(seed)
    except Exception:
        pass



KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2,  'Stay': 4}
ACTION_TO_KEY = {v: k for k, v in KEYS_ACTIONS.items()}
ACTION_TO_KEY[4] = "Stay"



MAX_STEPS = 200
AI_STUCK_WINDOW = 5

# =========================
# map registry / policy pools
# =========================
DEFAULT_EXPERIMENT_MAP = "circle"

MAP_CONFIGS = {
    "circle": {
        "map_type": "circle",
        "grid_dim": [5, 5],
        "policy_pool_dir": os.path.join(current_dir, "policy_pool_circle_coplay"),
        "embedding_csv": os.path.join(current_dir, "tsnt_circle.csv"),
        "policy_prefix": "[coplay][flexible][circle]agent0_",
        "checkpoint_filename": "model_5000000.zip",
        "env_id": "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v5",
        "layout_id": "fixed_circle_5x5",
        "aliases": ["circle", "fixed_circle_5x5"],
    },

    "counter": {
        "map_type": "counter",
        "grid_dim": [5, 8],
        "policy_pool_dir": os.path.join(current_dir, "policy_pool_counter_coplay"),
        "embedding_csv": os.path.join(current_dir, "tsnt_counter.csv"),
        "policy_prefix": "[coplay][flexible][counter]agent0_",
        "checkpoint_filename": "model_5000000.zip",
        "env_id": "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v4",
        "layout_id": "fixed_counter_5x8",
        "aliases": ["counter", "fixed_counter_5x8", "counter"],
    },

    "thinpath": {
        "map_type": "thinpath",
        "grid_dim": [5, 7],
        "policy_pool_dir": os.path.join(current_dir, "policy_pool_thinpath_coplay"),
        "embedding_csv": os.path.join(current_dir, "tsnt_thinpath.csv"),
        "policy_prefix": "[coplay][flexible][thinpath]agent0_",
        "checkpoint_filename": "model_5000000.zip",
        "env_id": "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v3",
        "layout_id": "fixed_thinpath_5x7",
        "aliases": ["thinpath", "fixed_thinpath_5x7", "thinpath"],
    },
}

# backward-compatible defaults
FIXED_MAP_TYPE = DEFAULT_EXPERIMENT_MAP
FIXED_GRID_DIM = list(MAP_CONFIGS[DEFAULT_EXPERIMENT_MAP]["grid_dim"])


def _normalize_map_name(map_name):
    if map_name is None:
        return None
    candidate = str(map_name).strip().lower()
    if not candidate:
        return None
    if candidate in MAP_CONFIGS:
        return candidate

    for registered_name, cfg in MAP_CONFIGS.items():
        aliases = {registered_name, str(cfg.get("map_type", "")).lower(), str(cfg.get("layout_id", "")).lower()}
        aliases.update(str(alias).strip().lower() for alias in cfg.get("aliases", []))
        if candidate in aliases:
            return registered_name
    return None


def _infer_map_name(map_name=None, layout_id=None, config_id=None, sess=None):
    if str(config_id).strip().lower() == "layout_practice":
        return "practice"
    if str(layout_id).strip().lower() == "layout_practice":
        return "practice"
    
    for candidate in (map_name, layout_id, config_id):
        normalized = _normalize_map_name(candidate)
        if normalized is not None:
            return normalized

        if candidate is None:
            continue

        lowered = str(candidate).strip().lower()
        for registered_name, cfg in MAP_CONFIGS.items():
            aliases = {registered_name, str(cfg.get("map_type", "")).lower(), str(cfg.get("layout_id", "")).lower()}
            aliases.update(str(alias).strip().lower() for alias in cfg.get("aliases", []))
            if any(alias and alias in lowered for alias in aliases):
                return registered_name

    current_map_name = getattr(sess, "current_map_name", None) if sess is not None else None
    if current_map_name == "practice":
        return "practice"
    if current_map_name in MAP_CONFIGS:
        return current_map_name

    return DEFAULT_EXPERIMENT_MAP


def _get_experiment_map_config(map_name=None, layout_id=None, config_id=None, sess=None):
    resolved_map_name = _infer_map_name(map_name=map_name, layout_id=layout_id, config_id=config_id, sess=sess)
    if resolved_map_name not in MAP_CONFIGS:
        available = ", ".join(sorted(MAP_CONFIGS))
        raise ValueError(f"Unknown map '{map_name}'. Available maps: {available}")
    return resolved_map_name, MAP_CONFIGS[resolved_map_name]


def _get_used_policy_names(sess, map_name: str):
    used_by_map = getattr(sess, "used_policy_names_by_map", None)
    if not isinstance(used_by_map, dict):
        used_by_map = {}
        sess.used_policy_names_by_map = used_by_map
    return list(used_by_map.get(map_name, []))


def _remember_policy_name(sess, map_name: str, policy_name: str):
    used_by_map = getattr(sess, "used_policy_names_by_map", None)
    if not isinstance(used_by_map, dict):
        used_by_map = {}
        sess.used_policy_names_by_map = used_by_map
    used = used_by_map.setdefault(map_name, [])
    if policy_name not in used:
        used.append(policy_name)




def _pick_random_policy_checkpoint(
    policy_pool_dir: str,
    checkpoint_filename: str,
    exclude_policy_names=None,
):
    if not policy_pool_dir:
        raise ValueError("policy_pool_dir must be provided")
    if not os.path.isdir(policy_pool_dir):
        raise FileNotFoundError(f"policy pool directory not found: {policy_pool_dir}")

    subdirs = [
        os.path.join(policy_pool_dir, d)
        for d in os.listdir(policy_pool_dir)
        if os.path.isdir(os.path.join(policy_pool_dir, d))
    ]
    subdirs = [d for d in subdirs if "agent0" in os.path.basename(d)]

    if not subdirs:
        raise FileNotFoundError(f"No policy subfolders found under: {policy_pool_dir}")

    if exclude_policy_names:
        exclude_set = set(exclude_policy_names)
        filtered = [d for d in subdirs if os.path.basename(d) not in exclude_set]
        if filtered:
            subdirs = filtered

    chosen_dir = random.choice(subdirs)
    ckpt_path = os.path.join(chosen_dir, checkpoint_filename)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{checkpoint_filename} not found in: {chosen_dir}")

    return chosen_dir, ckpt_path




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
    "penalize using bad lettuce": 0,
    "pick up bad lettuce": 0
},{
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



# The environment wrapper
class SingleAgentWrapper_accept_keyboard_action(_gym.Wrapper):
    def __init__(self, env, agent_index, reset_step):
        super(SingleAgentWrapper_accept_keyboard_action, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env_reset_step = 0

        self.reset_step = reset_step

        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True

    def reset(self):
        self.obs = self.env.reset()
        self.env_reset_step = 0
       
        self.firsttime_down_go_to_counter = True
        self.firsttime_up_get_counter_lettuce = True
        return self.obs[self.agent_index]

    def step(self, action, keyboard_action):
        actions = [action, keyboard_action]
        self.obs, rewards, dones, info = self.env.step(actions)
        self.obs = self.env._get_macro_obs()

        self.env_reset_step += 1

        if self.env_reset_step % self.reset_step == 0:
            self.env.soft_reset_obs_only()
            self.env.macroAgent[0].cur_macro_action_done = True
            self.env.macroAgent[1].cur_macro_action_done = True
            self.obs = self.env._get_macro_obs()

        # Save per-agent rewards for logging (agent 0 = AI, agent 1 = Human)
        self.last_raw_rewards = rewards
        try:
            rr = np.array(rewards).flatten()
            self.last_ai_reward = float(rr[0]) if rr.size > 0 else None
            self.last_human_reward = float(rr[1]) if rr.size > 1 else None
        except Exception:
            self.last_ai_reward = None
            self.last_human_reward = None

        team_reward = rewards[self.agent_index] + rewards[1 - self.agent_index]
        return self.obs[self.agent_index], team_reward, dones, info



def _as_int_action(a):
    """Robustly cast SB3 action to a Python int."""
    if isinstance(a, (int, np.integer)):
        return int(a)
    a_np = np.asarray(a)
    if a_np.ndim == 0:        # numpy 0-d scalar
        return int(a_np.item())
    return int(a_np.flatten()[0])

def _reset_ai_position_history(sess, env):
    sess.ai_position_history = deque(maxlen=AI_STUCK_WINDOW)
    try:
        agent = env.agent[0]
        sess.ai_position_history.append((int(agent.x), int(agent.y)))
    except Exception:
        pass


def _record_ai_position(sess, env):
    history = getattr(sess, 'ai_position_history', None)
    if not isinstance(history, deque):
        history = deque(maxlen=AI_STUCK_WINDOW)
        sess.ai_position_history = history
    try:
        agent = env.agent[0]
        history.append((int(agent.x), int(agent.y)))
    except Exception:
        pass


def _ai_has_been_stuck(sess) -> bool:
    history = getattr(sess, 'ai_position_history', None)
    if not isinstance(history, deque) or len(history) < AI_STUCK_WINDOW:
        return False
    first = history[0]
    return all(pos == first for pos in history)


def _pick_random_legal_ai_move(env, agent_index: int = 0):
    try:
        agent = env.agent[agent_index]
    except Exception:
        return None

    candidates = []
    directions = [
        (0, 1, 0),   # right
        (1, 0, 1),   # down
        (0, -1, 2),  # left
        (-1, 0, 3),  # up
    ]

    for dx, dy, action_int in directions:
        nx = int(agent.x) + dx
        ny = int(agent.y) + dy
        try:
            tile_name = ITEMNAME[env.map[nx][ny]]
        except Exception:
            continue
        if tile_name == 'space':
            candidates.append(action_int)

    if not candidates:
        return None
    return random.choice(candidates)


# =========================
# AI reward helpers for BO
# =========================
import re

ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "dirtyplate", "badlettuce"]

macroActionDict = {"stay": 0, "get lettuce 1": 1, "get lettuce 2": 2, "get plate 1": 3, "get plate 2": 4, "go to knife 1": 5, "deliver 1": 6, "chop": 7, "go to counter": 8, "right": 9, "down": 10, "left": 11, "up": 12}


def check_action_benevolence_circle(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce):

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    counter1_x = 2
    counter1_y = 2

    counter1 = ITEMNAME[env.map[counter1_x][counter1_y]]

    reward_shaping_bonus = 0
    total_reward_bonus = 0


    reward_bonus_up = 0
    reward_bonus_down = 0

    counters = [counter1]

    if any(counter in ("lettuce") for counter in counters):
        best_action = intelligently_find_item_number(env, agent_up, "get lettuce")

        if firsttime_up_get_counter_lettuce == True:
            reward_shaping_bonus = check_benevolence(env, best_action, action_up)
            if reward_shaping_bonus == 20:
                total_reward_bonus += reward_shaping_bonus
                reward_bonus_up = 1000
                firsttime_up_get_counter_lettuce = False


    if all(counter not in ("lettuce") for counter in counters):

        if agent_down.holding and isinstance(agent_down.holding, Lettuce):
            best_action = "go to counter"

            if firsttime_down_go_to_counter == True:

                reward_shaping_bonus = check_benevolence(env, best_action, action_down)

                if reward_shaping_bonus == 20:
                    total_reward_bonus += reward_shaping_bonus
                    reward_bonus_down = 1000
                    firsttime_down_go_to_counter = False

    return reward_bonus_up, reward_bonus_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce




def check_action_benevolence_counter(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce):

    agent_up = env.agent[0]
    agent_down = env.agent[1]


    counter1_x = 2
    counter1_y = 2

    counter2_x = 2
    counter2_y = 3

    counter3_x = 2
    counter3_y = 4

    counter4_x = 2
    counter4_y = 5



    counter1 = ITEMNAME[env.map[counter1_x][counter1_y]]
    counter2 = ITEMNAME[env.map[counter2_x][counter2_y]]
    counter3 = ITEMNAME[env.map[counter3_x][counter3_y]]
    counter4 = ITEMNAME[env.map[counter4_x][counter4_y]]



    reward_shaping_bonus = 0
    total_reward_bonus = 0


    reward_bonus_up = 0
    reward_bonus_down = 0

    counters = [counter1, counter2, counter3, counter4]



    if any(counter in ("lettuce") for counter in counters):
        best_action = intelligently_find_item_number(env, agent_up, "get lettuce")

        if firsttime_up_get_counter_lettuce == True:
            reward_shaping_bonus = check_benevolence(env, best_action, action_up)
            if reward_shaping_bonus == 20:
                total_reward_bonus += reward_shaping_bonus
                reward_bonus_up = 1000
                firsttime_up_get_counter_lettuce = False


    if all(counter not in ("lettuce") for counter in counters):

        if agent_down.holding and isinstance(agent_down.holding, Lettuce):
            best_action = "go to counter"

            if firsttime_down_go_to_counter == True:

                reward_shaping_bonus = check_benevolence(env, best_action, action_down)

                if reward_shaping_bonus == 20:
                    total_reward_bonus += reward_shaping_bonus
                    reward_bonus_down = 1000
                    firsttime_down_go_to_counter = False


    return reward_bonus_up, reward_bonus_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce




def check_action_benevolence_thinpath(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce):

    agent_up = env.agent[0]
    agent_down = env.agent[1]



    counter1_x = 1
    counter1_y = 3

    counter2_x = 3
    counter2_y = 3



    counter1 = ITEMNAME[env.map[counter1_x][counter1_y]]
    counter2 = ITEMNAME[env.map[counter2_x][counter2_y]]


    reward_shaping_bonus = 0
    total_reward_bonus = 0


    reward_bonus_up = 0
    reward_bonus_down = 0

    counters = [counter1, counter2]



    if any(counter in ("lettuce") for counter in counters):
        best_action = intelligently_find_item_number(env, agent_up, "get lettuce")

        if firsttime_up_get_counter_lettuce == True:
            reward_shaping_bonus = check_benevolence(env, best_action, action_up)
            if reward_shaping_bonus == 20:
                total_reward_bonus += reward_shaping_bonus
                reward_bonus_up = 1000
                firsttime_up_get_counter_lettuce = False


    if all(counter not in ("lettuce") for counter in counters):

        if agent_down.holding and isinstance(agent_down.holding, Lettuce):
            best_action = "go to counter"

            if firsttime_down_go_to_counter == True:

                reward_shaping_bonus = check_benevolence(env, best_action, action_down)

                if reward_shaping_bonus == 20:
                    total_reward_bonus += reward_shaping_bonus
                    reward_bonus_down = 1000
                    firsttime_down_go_to_counter = False


    return reward_bonus_up, reward_bonus_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce


def check_action_benevolence(env, action_up, action_down, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce):
    map_type = getattr(env, "map_type", None)

    if map_type == "circle":
        return check_action_benevolence_circle(
            env, action_up, action_down,
            firsttime_down_go_to_counter,
            firsttime_up_get_counter_lettuce
        )
    elif map_type == "counter":
        return check_action_benevolence_counter(
            env, action_up, action_down,
            firsttime_down_go_to_counter,
            firsttime_up_get_counter_lettuce
        )
    elif map_type == "thinpath":
        return check_action_benevolence_thinpath(
            env, action_up, action_down,
            firsttime_down_go_to_counter,
            firsttime_up_get_counter_lettuce
        )

    return 0.0, 0.0, firsttime_down_go_to_counter, firsttime_up_get_counter_lettuce


def find_best_reachable_index(can_reach_1, can_reach_2, can_reach_3, distance_1, distance_2, distance_3):
    reachable_indices = []
    distances = []
    
    if can_reach_1 != 4:
        reachable_indices.append(0)
        distances.append(distance_1)
    if can_reach_2 != 4:
        reachable_indices.append(1)
        distances.append(distance_2)
    if can_reach_3 != 4:
        reachable_indices.append(2)
        distances.append(distance_3)
    
    if not reachable_indices:
        return False
    
    if len(reachable_indices) == 1:
        return reachable_indices[0]
    
    min_distance_index = reachable_indices[distances.index(min(distances))]
    return min_distance_index



def intelligently_find_item_number(env, agent_item, raw_name):

    if raw_name == "get plate":
        target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
        can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
        distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

        target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
        can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
        distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

        target_x_3, target_y_3 = env._findPOitem(agent_item, macroActionDict["get dirty plate"])
        can_reach_3 = env._navigate(agent_item, target_x_3, target_y_3)
        distance_3 = env._calDistance(target_x_3, target_y_3, agent_item.x, agent_item.y)


        best_action = "stay"

        min_distance_index = find_best_reachable_index(can_reach_1, can_reach_2, can_reach_3, distance_1, distance_2, distance_3)

        if min_distance_index == 0:
            best_action = raw_name + " 1"
        if min_distance_index == 1:
            best_action = raw_name + " 2"
        if min_distance_index == 2:
            best_action = "get dirty plate"

        return best_action

    

    target_x_1, target_y_1 = env._findPOitem(agent_item, macroActionDict[raw_name + " 1"])
    can_reach_1 = env._navigate(agent_item, target_x_1, target_y_1)
    distance_1 = env._calDistance(target_x_1, target_y_1, agent_item.x, agent_item.y)

    target_x_2, target_y_2 = env._findPOitem(agent_item, macroActionDict[raw_name + " 2"])
    can_reach_2 = env._navigate(agent_item, target_x_2, target_y_2)
    distance_2 = env._calDistance(target_x_2, target_y_2, agent_item.x, agent_item.y)

    best_action = "stay"
    if can_reach_1 == 4 and can_reach_2 != 4:
        best_action = raw_name + " 2"

    if can_reach_1 != 4 and can_reach_2 == 4:
        best_action = raw_name + " 1"

    if can_reach_1 != 4 and can_reach_2 != 4:
        if distance_1 <= distance_2:
            best_action = raw_name + " 1"

        else:
            best_action = raw_name + " 2"
    
    return best_action



def check_benevolence(env, best_action, action):
    env.reward = 0
    if action == macroActionDict[best_action] and macroActionDict[best_action] != 0:
        env.reward += 20
    return env.reward



import re

def parse_policy_id(policy_id: str):
    """
    Parse step_penalty0 and helping0 bonus from policy_id

    policy_id format:
    [equilibrium][MAP_NAME]agent0_a0sp_STEPPENALTY0_a1sp_STEPPENALTY1_helping0_BONUS0_helping1_BONUS1_gammaAAA_BBB
    """

    pattern = r"a0sp_([^_]+).*?helping0_([^_]+)"
    match = re.search(pattern, policy_id)

    if not match:
        raise ValueError(f"Invalid policy_id format: {policy_id}")

    step_penalty0 = match.group(1)
    bonus0 = match.group(2)

    return step_penalty0, bonus0

# =========================
# Session management for parallel participants taking the user study at the same time
# =========================
class Session:
    def __init__(self, config_id="layout_practice"):
        self.config_id = config_id
        self.env = None
        self.env_mac = None
        self.wrapper = None
        self.model = None
        self.obs = None
        self.cur_step = 0
        self.cumulative_reward = 0.0
        self.ai_reward_total = 0.0  # AI-only reward used for BayesOpt
        self.current_layout_id = None
        self.current_model_id = None
        self.robot_steps = []       # save each step of the AI agent
        self.last_access = time.time()
        self.lock = threading.RLock()
        self.chosen_policy_dir = None
        self.chosen_ckpt_path = None
        self.ai_tick_counter = 0
        self.last_ai_action_int = 4
        self.ai_position_history = deque(maxlen=AI_STUCK_WINDOW)
        self.ai_blocked_move_streak = 0
       
        # Experiment structure (episodes)
        self.episode_index = None
        self.round_in_episode = None
        self.episode_phase = None
        self.used_policy_names = []  # basenames of chosen policy dirs
        self.solo_episode = False  # if True, freeze/hide AI teammate
        self.bo_best_policy_name = None
        self.used_policy_names = []  # backward-compatible flat cache
        self.used_policy_names_by_map = {}  # map_name -> [policy_dir_basename, ...]
        self.current_map_name = DEFAULT_EXPERIMENT_MAP
        self.current_map_type = MAP_CONFIGS[DEFAULT_EXPERIMENT_MAP]["map_type"]
        self.current_grid_dim = list(MAP_CONFIGS[DEFAULT_EXPERIMENT_MAP]["grid_dim"])


class SessionManager:
    def __init__(self, ttl_seconds=3600):
        self.sessions = {}
        self.ttl = ttl_seconds
        self.lock = threading.RLock()

    def new_session(self, default_config_id="layout_practice"):
        sid = uuid.uuid4().hex
        with self.lock:
            self.sessions[sid] = Session(config_id=default_config_id)
        return sid

    def get(self, sid):
        with self.lock:
            s = self.sessions.get(sid)
        if s:
            s.last_access = time.time()
        return s

    def ensure(self, sid, default_config_id="layout_practice"):
        s = self.get(sid)
        if s is None:
            with self.lock:
                s = Session(config_id=default_config_id)
                self.sessions[sid] = s
        return s

    def cleanup(self):
        now = time.time()
        with self.lock:
            dead = [sid for sid, s in self.sessions.items() if now - s.last_access > self.ttl]
            for sid in dead:
                try:
                    if self.sessions[sid].env is not None:
                        try: self.sessions[sid].env.close()
                        except: pass
                    if self.sessions[sid].env_mac is not None:
                        try: self.sessions[sid].env_mac.close()
                        except: pass
                finally:
                    del self.sessions[sid]

SESSION_MGR = SessionManager(ttl_seconds=3600)

class OptimizerManager:
    def __init__(self) -> None:
        self.optimizers = {}

    def _key(self, prolific_id, map_name):
        return (prolific_id, map_name)

    def optimizer_exists(self, prolific_id, map_name):
        return self._key(prolific_id, map_name) in self.optimizers

    def create_optimizer(self, prolific_id, map_name, embedding_csv, n_init, n_bo, n_knn):
        key = self._key(prolific_id, map_name)
        if key in self.optimizers:
            return

        surr = create_surrogate_spec(noise_variance=0.2)
        self.optimizers[key] = TSNEBayesOptimizer(
            embedding_csv=embedding_csv,
            n_init=n_init,
            n_bo=n_bo,
            n_knn=n_knn,
            n_best=1,
            surrogate_spec=surr,
            verbose=True
        )

    def get_optimizer(self, prolific_id, map_name):
        return self.optimizers[self._key(prolific_id, map_name)]
        
OPTIMIZER_MGR = OptimizerManager()

def _get_best_policy_name_from_optimizer(optimizer: TSNEBayesOptimizer) -> str:
    """Return the current best policy name without creating a new optimizer trial."""
    best_policy, _best_params, _prediction, _index, _name = optimizer.get_best()
    if not best_policy:
        raise ValueError("Could not determine best policy from optimizer.")
    return str(best_policy)



_model_cache_by_path = {}

def _pick_policy_checkpoint(
    policy: str,
    policy_pool_dir: str,
    policy_prefix: str,
    checkpoint_filename: str,
):
    raw_policy = str(policy).strip()

    candidate_names = [
        raw_policy,
        f"{policy_prefix}{raw_policy}",
    ]

    tried = []

    for candidate in candidate_names:
        chosen_dir = os.path.join(policy_pool_dir, candidate)
        ckpt_pth = os.path.join(chosen_dir, checkpoint_filename)
        tried.append(ckpt_pth)

        if os.path.isfile(ckpt_pth):
            logger.info(f"[POLICY - FOUND] {ckpt_pth}")
            return chosen_dir, ckpt_pth

    raise FileNotFoundError(
        f"No checkpoint found for policy '{raw_policy}' in '{policy_pool_dir}'. "
        f"Tried: {tried}"
    )

# This function is used for loading an AI model or getting an existing AI model.
def _load_or_get_model_by_ckpt_path(ckpt_path: str):
    assert "agent0" in ckpt_path, "loading agent1 somwhere"
    if ckpt_path in _model_cache_by_path:
        return _model_cache_by_path[ckpt_path]
    m = PPO.load(ckpt_path, device="cpu")
    try:
        m.policy.set_training_mode(False)
    except Exception:
        pass
    try:
        m.policy.eval()
    except Exception:
        pass
    _model_cache_by_path[ckpt_path] = m
    return m


def _parse_config_id(layout_id: str = None, model_id: str = None, config_id: str = None) -> str:
    if config_id:
        return config_id
    if layout_id and model_id:
        return f"{layout_id}_{model_id}"
    raise ValueError("Either 'config_id' or both 'layout_id' and 'model_id' must be provided.")


def create_envs_for_session(
    sess: Session,
    config_id: str,
    choose_new_policy: bool = True,
    optimizer: TSNEBayesOptimizer | None = None,
    is_solo: bool | None = None,
    map_name: str | None = None,
    layout_id: str | None = None,
    forced_policy_name: str | None = None,
):
    is_practice = (config_id == "layout_practice")
    is_solo = bool(getattr(sess, "solo_episode", False)) if is_solo is None else bool(is_solo)

    if is_practice:
        env_id = "Overcooked-equilibrium-v0"
        mac_env_id = "Overcooked-MA-equilibrium-v0"
        active_map_name = "practice"
        active_layout_id = "layout_practice"
        active_cfg = {
            "grid_dim": [5, 5],
            "task": ["lettuce salad"],
            "rewardList": rewardList,
            "map_type": "A",
            "n_agent": 2,
            "obs_radius": 0,
            "mode": "vector",
            "debug": True,
        }
        policy_pool_dir = None
        policy_prefix = None
        checkpoint_filename = None
    else:
        active_map_name, map_cfg = _get_experiment_map_config(
            map_name=map_name,
            layout_id=layout_id,
            config_id=config_id,
            sess=sess,
        )
        active_cfg = dict(map_cfg)
        env_id = active_cfg.pop("env_id")
        mac_env_id = active_cfg.pop("mac_env_id")
        active_layout_id = active_cfg.pop(
            "layout_id",
            f"fixed_{active_map_name}_{active_cfg['grid_dim'][0]}x{active_cfg['grid_dim'][1]}"
        )
        policy_pool_dir = active_cfg.pop("policy_pool_dir")
        policy_prefix = active_cfg.pop("policy_prefix")
        checkpoint_filename = active_cfg.pop("checkpoint_filename")

    n_agent = 1 if (is_solo and not is_practice) else int(active_cfg.get("n_agent", 2))
    env_params = {
        "grid_dim": list(active_cfg["grid_dim"]),
        "task": list(active_cfg.get("task", ["lettuce salad"])),
        "rewardList": active_cfg.get("rewardList", rewardList),
        "map_type": active_cfg["map_type"],
        "n_agent": n_agent,
        "obs_radius": active_cfg.get("obs_radius", 0),
        "mode": active_cfg.get("mode", "vector"),
        "debug": active_cfg.get("debug", True),
    }

    active_grid_dim = list(env_params["grid_dim"])
    active_map_type = env_params["map_type"]
    map_changed = (getattr(sess, "current_map_name", None) != active_map_name)

    if sess.env is not None:
        try:
            sess.env.close()
        except Exception:
            pass
    if sess.env_mac is not None:
        try:
            sess.env_mac.close()
        except Exception:
            pass

    sess.env = gym.make(env_id, **env_params)
    _seed_env_everything(sess.env, SEED)
    sess.env.reset()

    sess.env_mac = gym.make(mac_env_id, **env_params)
    _seed_env_everything(sess.env_mac, SEED)

    if n_agent == 2:
        reset_step = 10000
        sess.wrapper = SingleAgentWrapper_accept_keyboard_action(
            sess.env_mac, agent_index=0, reset_step=reset_step
        )
    else:
        sess.wrapper = None

    if is_practice or is_solo:
        sess.chosen_policy_dir = None
        sess.chosen_ckpt_path = None
        sess.model = None
    
    else:
        used_policy_names = _get_used_policy_names(sess, active_map_name)
        should_pick = bool(forced_policy_name) or bool(choose_new_policy) or map_changed or (not sess.chosen_ckpt_path)

        if should_pick:
            if forced_policy_name:
                # Replay episodes should not ask the optimizer for a new trial.
                # They should explicitly reload the saved/best policy.
                logger.info(
                    f"[POLICY - REPLAY] forcing policy={forced_policy_name} "
                    f"for phase={getattr(sess, 'episode_phase', None)}"
                )
                chosen_dir, ckpt_path = _pick_policy_checkpoint(
                    forced_policy_name,
                    policy_pool_dir=policy_pool_dir,
                    policy_prefix=policy_prefix,
                    checkpoint_filename=checkpoint_filename,
                )
                sess.chosen_policy_dir = chosen_dir
                sess.chosen_ckpt_path = ckpt_path
                sess.model = _load_or_get_model_by_ckpt_path(ckpt_path)

            elif optimizer is not None:
                logger.info(f"[POLICY - PICK] picking policy using optimization pipeline for map={active_map_name}")
                mapped_trials = optimizer.ask()
                checkpoint = mapped_trials[optimizer._actual_trial_idx]["policy"]
                chosen_dir, ckpt_path = _pick_policy_checkpoint(
                    checkpoint,
                    policy_pool_dir=policy_pool_dir,
                    policy_prefix=policy_prefix,
                    checkpoint_filename=checkpoint_filename,
                )
                sess.chosen_policy_dir = chosen_dir
                sess.chosen_ckpt_path = ckpt_path
                sess.model = _load_or_get_model_by_ckpt_path(ckpt_path)

            else:
                logger.info(f"[POLICY - PICK] picking a random policy for map={active_map_name}")
                chosen_dir, ckpt_path = _pick_random_policy_checkpoint(
                    policy_pool_dir=policy_pool_dir,
                    checkpoint_filename=checkpoint_filename,
                    exclude_policy_names=used_policy_names,
                )
                sess.chosen_policy_dir = chosen_dir
                sess.chosen_ckpt_path = ckpt_path

            policy_name = os.path.basename(sess.chosen_policy_dir)
            _remember_policy_name(sess, active_map_name, policy_name)
            if policy_name not in sess.used_policy_names:
                sess.used_policy_names.append(policy_name)

        sess.model = _load_or_get_model_by_ckpt_path(sess.chosen_ckpt_path)

    if sess.wrapper is not None:
        sess.obs = sess.wrapper.reset()
    else:
        _obs = sess.env_mac.reset()
        try:
            sess.obs = sess.env_mac._get_macro_obs()[0]
        except Exception:
            if isinstance(_obs, (list, tuple)) and len(_obs) > 0:
                sess.obs = _obs[0]
            else:
                sess.obs = _obs

    sess.config_id = "layout_practice" if is_practice else config_id
    sess.current_layout_id = active_layout_id
    sess.current_map_name = active_map_name
    sess.current_map_type = active_map_type
    sess.current_grid_dim = active_grid_dim

    if is_practice:
        sess.current_model_id = "none"
    elif is_solo:
        sess.current_model_id = "solo"
    else:
        sess.current_model_id = os.path.basename(sess.chosen_policy_dir)

    sess.cur_step = 0
    sess.cumulative_reward = 0.0
    sess.ai_reward_total = 0.0
    sess.robot_steps = []

    sess.ai_tick_counter = 0
    sess.last_ai_action_int = 4
    sess.ai_blocked_move_streak = 0

    _reset_ai_position_history(sess, sess.env_mac)

# =========================
# Collect the current state into a json.
# =========================
def extract_state(sess: Session):
    # env = sess.env
    env = sess.env_mac
    state = {
        "cur_step": sess.cur_step,
        "xlen": env.xlen,
        "ylen": env.ylen,
        "map": env.map,
        # "pomap": env.agent[0].pomap if hasattr(env.agent[0], 'pomap') else None,
        "items": [],
        "agents": [],
        # "layout": env.layout_pomap
    }

    def get_contained_name(obj):
        if isinstance(obj, Plate) or isinstance(obj, DirtyPlate):
            try:
                return obj.containedName
            except Exception:
                return None
        return None

    def get_type_name(obj):
        if hasattr(obj, 'name'):
            return obj.name
        elif hasattr(obj, 'rawName'):
            return obj.rawName
        else:
            return "unknown"

    def add_item_list(item_list):
        for item in item_list:
            state["items"].append({
                "x": item.x,
                "y": item.y,
                "type": get_type_name(item),
                "containing": get_contained_name(item),
                "holding": get_type_name(item.holding) if hasattr(item, 'holding') and item.holding else None,
                "holding_containing": get_contained_name(item.holding) if hasattr(item, 'holding') and item.holding else None
            })

    add_item_list(env.tomato)
    add_item_list(env.lettuce)
    add_item_list(env.badlettuce)
    add_item_list(env.onion)
    add_item_list(env.knife)
    add_item_list(env.delivery)
    add_item_list(env.plate)
    add_item_list(env.dirtyplate)

    for agent in env.agent:
        holding = agent.holding
        state["agents"].append({
            "x": agent.x,
            "y": agent.y,
            "color": agent.color if hasattr(agent, 'color') else "red",
            "holding": get_type_name(holding) if holding else None,
            "holding_containing": get_contained_name(holding) if holding else None
        })
    
    if getattr(env, 'n_agent', None) == 1 and len(state['agents']) == 1:
        state['agents'][0]['color'] = "blue"

    return state

# =========================
# Routing
# =========================

# 1. Route for the HTML file
@app.route('/')
def index():
    # Points to Equilibrium_project/equilibrium_frontend.html
    return send_from_directory(target_folder, 'equilibrium_frontend.html')

@app.route('/new_session', methods=['POST'])
def new_session():
    sid = SESSION_MGR.new_session()
    return jsonify(success=True, session_id=sid)

@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id')
    prolific_raw = data.get('prolificId')
    if not prolific_raw:
        return jsonify(success=False, error="prolificId is required"), 400
    prolific = prolific_raw.strip().replace('/', '_')

    if not sid:
        return jsonify(success=False, error="session_id is required"), 400


    # Frontend episode metadata
    layout_id = data.get('layout_id')
    model_id  = data.get('model_id')
    config_id = data.get('config_id')
    map_type = data.get('map_type')

    selection_mode = 'bo'

    # metadata (optional)
    episode_index = data.get('episode_index', None)
    round_in_episode = data.get('round_in_episode', None)
    episode_phase = data.get('episode_phase', None)
    new_episode = data.get('new_episode', None)
    solo_episode = bool(data.get('solo_episode', False)) or (episode_phase == 'solo')

    try:
        episode_index_int = int(episode_index) if episode_index is not None else None
    except Exception:
        episode_index_int = None
    try:
        round_in_episode_int = int(round_in_episode) if round_in_episode is not None else None
    except Exception:
        round_in_episode_int = None

    if new_episode is None:
        # Heuristic default: the first round in an episode is round 1.
        new_episode = (round_in_episode_int == 1)

    is_practice = ((config_id or "layout_practice") == "layout_practice")

    sess = SESSION_MGR.ensure(sid)
    # persist solo flag in the session
    sess.solo_episode = bool(solo_episode)

    if is_practice or sess.solo_episode:
        optimizer = None
    else:
        resolved_map_name, resolved_map_cfg = _get_experiment_map_config(
            map_name=map_type,
            layout_id=layout_id,
            config_id=config_id,
            sess=sess,
        )

        embedding_csv = resolved_map_cfg["embedding_csv"]

        n_init = int(data.get('n_init'))
        n_bo = int(data.get('n_bo'))
        n_knn = int(data.get('n_knn'))

        if not OPTIMIZER_MGR.optimizer_exists(prolific, resolved_map_name):
            OPTIMIZER_MGR.create_optimizer(
                prolific_id=prolific,
                map_name=resolved_map_name,
                embedding_csv=embedding_csv,
                n_init=n_init,
                n_bo=n_bo,
                n_knn=n_knn,
            )
            logger.info(
                f"[BACKEND - RESET] optimizer configured for prolific_id={prolific}, "
                f"map={resolved_map_name}, embedding_csv={embedding_csv}"
            )

        optimizer = OPTIMIZER_MGR.get_optimizer(prolific, resolved_map_name)

    
    with sess.lock:
        try:
            # Force a new policy if the episode index changed.
            episode_changed = (
                episode_index_int is not None
                and sess.episode_index is not None
                and episode_index_int != sess.episode_index
            )

            choose_new_policy = (
                (not is_practice)
                and (not sess.solo_episode)
                and (bool(new_episode) or episode_changed)
            )

           
            forced_policy_name = None

            replay_best_phases = {"bo_replay_best", "replay_optimal"}

            if choose_new_policy and episode_phase in replay_best_phases:
                forced_policy_name = getattr(sess, "bo_best_policy_name", None)

                if not forced_policy_name:
                    if optimizer is None:
                        raise ValueError(
                            f"Cannot replay best policy for phase={episode_phase}: "
                            "no saved best policy and optimizer is not available."
                        )

                    forced_policy_name = _get_best_policy_name_from_optimizer(optimizer)
                    sess.bo_best_policy_name = forced_policy_name

                logger.info(
                    f"[POLICY - FORCE BEST] phase={episode_phase}, "
                    f"policy={forced_policy_name}"
                )

            # Persist metadata in the session.
            if episode_index_int is not None:
                sess.episode_index = episode_index_int
            sess.round_in_episode = round_in_episode_int
            sess.episode_phase = episode_phase
            create_envs_for_session(
                sess,
                config_id=(config_id or "layout_practice"),
                choose_new_policy=choose_new_policy,
                optimizer=optimizer,
                is_solo=sess.solo_episode,
                map_name=map_type,
                layout_id=layout_id,
                forced_policy_name=forced_policy_name,
            )


            #if choose_new_policy and episode_phase == "bo_replay_best":
            #    sess.bo_best_policy_name = sess.current_model_id
            #    logger.info(
            #        f"[POLICY - SAVE BEST] bo_best_policy_name={sess.bo_best_policy_name}"
            #    )
        
        except Exception as e:
            logger.exception(
                f"[BACKEND - RESET - ERROR] sid={sid}, config_id={config_id}, map_type={map_type}, prolific={prolific}"
            )
            return jsonify(success=False, error=str(e)), 400

        steps_left = MAX_STEPS
        logger.info(
            f"[BACKEND - RESET] sid={sid}, episode={sess.episode_index}, round={sess.round_in_episode}, "
            f"phase={sess.episode_phase}, map={sess.current_map_type}, grid={sess.current_grid_dim}, policy={sess.current_model_id}"
        )

        return jsonify(
            success=True,
            state=extract_state(sess),
            steps_left=steps_left,
            cumulative_reward=sess.cumulative_reward,
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id,
            chosen_policy_dir=os.path.basename(sess.chosen_policy_dir) if sess.chosen_policy_dir else None,
            chosen_ckpt=os.path.basename(sess.chosen_ckpt_path) if sess.chosen_ckpt_path else None,

            episode_index=sess.episode_index,
            round_in_episode=sess.round_in_episode,
            episode_phase=sess.episode_phase,

            map_type=sess.current_map_type,
            selected_map=sess.current_map_name,
            grid_dim=sess.current_grid_dim,
            policy_id=sess.current_model_id,

            optimal_policy_id=(
                sess.bo_best_policy_name
                if sess.episode_phase in {"bo_replay_best", "stress", "replay_optimal"}
                else None
            ),
            selection_mode=selection_mode,
        )


@app.route('/key_event', methods=['POST'])
def key_event():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id')
    if not sid:
        return jsonify(success=False, error="session_id is required"), 400

    key = data.get('key')
    if not key:
        return jsonify(success=False, error="key is required"), 400

    layout_id = data.get('layout_id')
    model_id  = data.get('model_id')
    config_id = data.get('config_id')
    map_type = data.get('map_type')

    sess = SESSION_MGR.ensure(sid)
    with sess.lock:
        # When the config id changes, hot switch the env.
        if layout_id or model_id or config_id or map_type:
            try:
                target_cfg_id = sess.config_id or "layout_practice"
                if config_id:
                    target_cfg_id = config_id
                elif layout_id and model_id:
                    target_cfg_id = _parse_config_id(layout_id=layout_id, model_id=model_id)

                if target_cfg_id == "layout_practice":
                    target_map_name = "practice"
                else:
                    target_map_name = _infer_map_name(
                        map_name=map_type,
                        layout_id=layout_id,
                        config_id=target_cfg_id,
                        sess=sess,
                    )

                needs_switch = (
                    target_cfg_id != sess.config_id
                    or target_map_name != getattr(sess, "current_map_name", None)
                )

                if needs_switch:
                    create_envs_for_session(
                        sess,
                        target_cfg_id,
                        is_solo=getattr(sess, 'solo_episode', False),
                        map_name=target_map_name,
                        layout_id=layout_id,
                    )
                    logger.info(f"[BACKEND - HOT_SWITCH] sid={sid}, config={target_cfg_id}, map={target_map_name}")
            except Exception as e:
                return jsonify(success=False, error=f"hot switch failed: {e}"), 400
            
        if not hasattr(sess, 'dishes_served'): sess.dishes_served = 0
        if sess.cur_step == 0: sess.dishes_served = 0

        KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2, 'Stay': 4}

        if key not in KEYS_ACTIONS:
            return jsonify(success=False, error="invalid key"), 400

        t0 = time.time()

        human_low = int(KEYS_ACTIONS[key])
        solo_mode = bool(getattr(sess, 'solo_episode', False)) or (getattr(sess.env, 'n_agent', 2) == 1)

        if solo_mode:
            # Human-alone: no AI model; step env with a single human action.
            ai_action_int = 4
            robot_low = 4
            action = [robot_low, human_low]  # (AI placeholder, Human) for logging/score adjustment

            try:
                obs_all, rewards_list, dones, info = sess.env_mac.step([human_low])
            except Exception as e:
                return jsonify(success=False, error=f'solo step failed: {e}'), 400

            try:
                sess.obs = sess.env_mac._get_macro_obs()[0]
            except Exception:
                if isinstance(obs_all, (list, tuple)) and len(obs_all) > 0:
                    sess.obs = obs_all[0]
                else:
                    sess.obs = obs_all

            # human-alone mode, use only the human agent's reward
            if isinstance(rewards_list, (list, tuple, np.ndarray)):
                rewards = float(np.array(rewards_list).flatten()[0])
            else:
                rewards = float(rewards_list)

            robot_key = ACTION_TO_KEY.get(robot_low, 'Stay')
            sess.robot_steps.append({
                'step': int(sess.cur_step + 1),
                'ai_macro_action': int(ai_action_int),
                'low_level_action': int(robot_low),
                'arrow': robot_key,
                'timestamp': time.time(),
            })

        else:
            # Normal human+AI: run policy -> macro action -> low-level action -> step wrapper
            benev_up = 0.0
            benev_down = 0.0
            ai_prev_loc = [sess.env_mac.agent[0].x, sess.env_mac.agent[0].y]

            AI_DECISION_EVERY = 2  # AI updates every 2 game ticks
            AI_MOVE_EVERY = 2      # AI physically moves only every 2 ticks

            sess.ai_tick_counter += 1

            if sess.model is not None:
                if sess.ai_tick_counter % AI_DECISION_EVERY == 1:
                    with torch.no_grad():
                        ai_action, _ = sess.model.predict(sess.obs, deterministic=True)
                    sess.last_ai_action_int = _as_int_action(ai_action)

                ai_action_int = sess.last_ai_action_int
                print(
                    f"[AI DEBUG] step={sess.cur_step} "
                    f"pos=({sess.env_mac.agent[0].x},{sess.env_mac.agent[0].y}) "
                    f"macro_action={ai_action_int}"
                )
            else:
                # No model loaded (practice), keep a placeholder macro id
                ai_action_int = 4

            # Compute low-level actions from macro action (and get executed macro actions when available)
            if sess.model is not None:
                ll_ret = sess.env_mac._computeLowLevelActions([ai_action_int, 0])

                # handle both possible return formats robustly
                if isinstance(ll_ret, (list, tuple)) and len(ll_ret) == 2:
                    primitive_action, real_execute_macro_actions = ll_ret
                else:
                    primitive_action = ll_ret
                    real_execute_macro_actions = [ai_action_int, 0]

                
                try:
                    benev_up, benev_down, sess.wrapper.firsttime_down_go_to_counter, sess.wrapper.firsttime_up_get_counter_lettuce = \
                        check_action_benevolence(
                            sess.env_mac,
                            real_execute_macro_actions[0],
                            real_execute_macro_actions[1],
                            sess.wrapper.firsttime_down_go_to_counter,
                            sess.wrapper.firsttime_up_get_counter_lettuce
                        )
                except Exception:
                    benev_up, benev_down = 0.0, 0.0
            else:
                primitive_action = [4] * sess.env.n_agent

            action = [4] * sess.env.n_agent
            action[1] = human_low

            if sess.ai_tick_counter % AI_MOVE_EVERY == 1:
                intended_ai_low = int(primitive_action[0])
                action[0] = intended_ai_low
            else:
                intended_ai_low = 4
                action[0] = 4

            unstuck_override = False
            anti_stuck_enabled = (
                sess.model is not None
                and getattr(sess, "current_map_name", None) != "practice"
            )

            if anti_stuck_enabled and getattr(sess, "ai_blocked_move_streak", 0) >= 3:
                forced_move = _pick_random_legal_ai_move(sess.env_mac, agent_index=0)
                if forced_move is not None:
                    action[0] = int(forced_move)
                    unstuck_override = True
                    logger.info(
                        f"[AI UNSTUCK] sid={sid}, step={sess.cur_step}, "
                        f"blocked_streak={sess.ai_blocked_move_streak}, "
                        f"forced_low_level={action[0]}"
                    )

            robot_low = int(action[0])
            robot_key = ACTION_TO_KEY.get(robot_low, 'Unknown')

            print(
                f"[AI DEBUG] step={sess.cur_step} "
                f"pos_before=({ai_prev_loc[0]},{ai_prev_loc[1]}) "
                f"macro_action={ai_action_int} "
                f"low_level={robot_low} "
                f"arrow={robot_key}"
            )

            sess.robot_steps.append({
                'step': int(sess.cur_step + 1),
                'ai_macro_action': int(ai_action_int),
                'low_level_action': robot_low,
                'arrow': robot_key,
                'timestamp': time.time(),
                'unstuck_override': bool(unstuck_override),
            })

            sess.obs, rewards, dones, info = sess.wrapper.step(action[0], action[1])
            _record_ai_position(sess, sess.env_mac)

            # determine if AI moved this step (for step-penalty shaping)
            ai_cur_loc = [sess.env_mac.agent[0].x, sess.env_mac.agent[0].y]
            ai_moved = (ai_prev_loc != ai_cur_loc)
            if anti_stuck_enabled:
                if unstuck_override:
                    sess.ai_blocked_move_streak = 0
                elif intended_ai_low != 4:
                    if not ai_moved:
                        sess.ai_blocked_move_streak += 1
                    else:
                        sess.ai_blocked_move_streak = 0

            print(
                f"[AI DEBUG] step={sess.cur_step} "
                f"pos_after=({ai_cur_loc[0]},{ai_cur_loc[1]}) "
                f"moved={ai_moved}"
            )

        r_env = 0.0
        r_adjusted = 0.0
        ai_reward_raw = None
        human_reward_raw = None
        ai_reward_adjusted = None
        human_reward_adjusted = None
        try:
            if isinstance(rewards, (list, tuple, np.ndarray)):
                r_env = float(np.array(rewards).flatten()[0])
            
            else:
                r_env = float(rewards)

            try:
                step_pen_ai = float(sess.env_mac.rewardList[0].get("step penalty", 0))
                step_pen_hu = float(sess.env_mac.rewardList[1].get("step penalty", 0))
            except Exception:
                step_pen_ai = -1.0
                step_pen_hu = -1.0

            # compensate "Stay" actions so they don't subtract points
            ai_low = int(action[0]) if (not solo_mode) else None
            human_low2 = int(action[1])


            # Per-agent raw rewards (agent 0 = AI, agent 1 = Human)
            if solo_mode:
                ai_reward_raw = 0.0
                human_reward_raw = float(r_env)
            else:
                rr = getattr(sess.wrapper, 'last_raw_rewards', None)
                if isinstance(rr, (list, tuple, np.ndarray)) and len(rr) >= 2:
                    rr = np.array(rr).flatten()
                    ai_reward_raw = float(rr[0])
                    human_reward_raw = float(rr[1])
            r_adjusted = r_env
            if (not solo_mode) and (ai_low == 4):
                r_adjusted += (-step_pen_ai)  # add back +1 if penalty is -1
            if human_low2 == 4:
                r_adjusted += (-step_pen_hu)


            # Per-agent adjusted rewards (compensate Stay actions like adjusted_reward)
            ai_reward_adjusted = ai_reward_raw
            human_reward_adjusted = human_reward_raw
            if (ai_reward_adjusted is not None) and (not solo_mode) and (ai_low == 4):
                ai_reward_adjusted += (-step_pen_ai)
            if (human_reward_adjusted is not None) and (human_low2 == 4):
                human_reward_adjusted += (-step_pen_hu)

            sess.cumulative_reward += r_adjusted

            # accumulate AI-only reward for BayesOpt (server-side)
            if not solo_mode:
                try:
                    step_penalty_str, cooperation_bonus_str = parse_policy_id(sess.current_model_id or "")
                    step_penalty = float(step_penalty_str)
                    cooperation_bonus = (cooperation_bonus_str == "True")
                except Exception:
                    step_penalty = 0.0
                    cooperation_bonus = False

                moved = bool(locals().get('ai_moved', False))
                benev_up_local = float(locals().get('benev_up', 0.0))
                move_cost = step_penalty if moved else 0.0
                team_reward = float(r_env)
                if cooperation_bonus:
                    ai_step_reward = team_reward - move_cost + benev_up_local
                else:
                    ai_step_reward = team_reward - move_cost
                sess.ai_reward_total += float(ai_step_reward)

            # dish served detection
            if r_env >= 150:
                sess.dishes_served += 1
                logger.info(f"[BACKEND - KEY_EVENT] sid={sid}, dish_served! total={sess.dishes_served}")

        except Exception as e:
            logger.info(f"[BACKEND - KEY_EVENT] error updating rewards: {e}")
                
        sess.cur_step += 1

        state = extract_state(sess)
        steps_left = max(0, MAX_STEPS - sess.cur_step)


        return jsonify(
            success=True,
            state=state,
            steps_left=steps_left,
            cumulative_reward=sess.cumulative_reward,
            raw_reward=r_env,
            adjusted_reward=r_adjusted,
            # Explicit reward breakdown for logging
            team_reward_raw=r_env,
            team_reward_adjusted=r_adjusted,
            ai_reward_raw=ai_reward_raw,
            human_reward_raw=human_reward_raw,
            ai_reward_adjusted=ai_reward_adjusted,
            human_reward_adjusted=human_reward_adjusted,
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id,
            dishes_served=sess.dishes_served,
            robot_last_action=(sess.robot_steps[-1] if sess.robot_steps else None)
        )


@app.route('/tell', methods=['POST'])
def tell():
    data = request.get_json(silent=True) or {}
    prolific_raw = data.get('prolificId')
    if not prolific_raw:
        return jsonify(success=False, error="prolific_id is required"), 400
    prolific = prolific_raw.strip().replace('/', '_')

    sid = data.get('session_id')

    if sid:
         sess = SESSION_MGR.ensure(sid)
         with sess.lock:
            if getattr(sess, "episode_phase", None) in {"bo_replay_best", "replay_optimal"}:
                return jsonify(
                    success=True,
                    skipped=True,
                    reason=f"replay episode: {getattr(sess, 'episode_phase', None)}"
                )

    # Preferred: use server-side AI reward (computed in /key_event) for BO.
    score_raw = data.get('score')

    if score_raw is not None:
        try:
            score = float(score_raw)
        except Exception:
            return jsonify(success=False, error="score must be numeric"), 400
    else:
        if not sid:
            return jsonify(success=False, error="session_id is required when score is not provided"), 400
        sess = SESSION_MGR.ensure(sid)
        with sess.lock:
            score = float(getattr(sess, 'ai_reward_total', 0.0))

    map_name = data.get('map_type')
    if not map_name and sid:
        sess = SESSION_MGR.ensure(sid)
        map_name = getattr(sess, "current_map_name", None)

    if not map_name:
        return jsonify(success=False, error="map_type is required for optimizer lookup"), 400

    if not OPTIMIZER_MGR.optimizer_exists(prolific, map_name):
        return jsonify(
            success=False,
            error=f"optimizer not found for prolificId={prolific}, map={map_name}"
        ), 400

    optimizer = OPTIMIZER_MGR.get_optimizer(prolific, map_name)
    trial_idx = optimizer._actual_trial_idx
    optimizer.tell({trial_idx: score})
    return jsonify(success=True)


@app.route('/close_optimizer', methods=['POST'])
def close_optimizer():
    data = request.get_json(silent=True) or {}
    prolific = (data.get('prolificId') or 'anon').strip().replace('/', '_')
    if not prolific:
        return jsonify(success=False, error="prolific_id is required"), 400

    to_delete = []
    for key, optimizer in OPTIMIZER_MGR.optimizers.items():
        pid, _map = key
        if pid == prolific:
            optimizer.close()
            to_delete.append(key)

    for key in to_delete:
        del OPTIMIZER_MGR.optimizers[key]

    return jsonify(success=True)


@app.route('/get_state', methods=['GET', 'POST'])
def get_state():
    sid = None
    if request.method == 'GET':
        sid = request.args.get('session_id')
    else:
        payload = request.get_json(silent=True) or {}
        sid = payload.get('session_id')

    if not sid:
        return jsonify(success=False, error="session_id is required"), 400

    sess = SESSION_MGR.get(sid)
    if not sess or sess.env is None:
        return jsonify(success=False, error="session not initialized; call /reset first"), 400

    with sess.lock:
        return jsonify(success=True, state=extract_state(sess))


def _normalize_action_token(action_value):
    if action_value is None:
        return None
    token = str(action_value).strip().upper()
    if token in ("", "NONE", "NULL"):
        return None
    return token


def _extract_human_action_sequence(round_obj):
    human_log = ((round_obj.get("action_log", {}) or {}).get("human", []) or [])
    seq = []
    for entry in human_log:
        token = _normalize_action_token(entry.get("action"))
        if token is not None:
            seq.append(token)
    return seq


def _extract_ai_action_sequence(round_obj):
    ai_log = ((round_obj.get("action_log", {}) or {}).get("ai", []) or [])
    seq = []
    for entry in ai_log:
        token = _normalize_action_token(entry.get("arrow"))
        if token is not None:
            seq.append(token)
    return seq


def _extract_joint_action_sequence(round_obj):
    human_seq = _extract_human_action_sequence(round_obj)
    ai_seq = _extract_ai_action_sequence(round_obj)
    n = min(len(human_seq), len(ai_seq))
    if n == 0:
        return []
    return list(zip(human_seq[:n], ai_seq[:n]))


def _sequence_similarity(seq_a, seq_b):
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return None
    matches = sum(1 for i in range(n) if seq_a[i] == seq_b[i])
    return matches / n


def _round_has_ai(round_obj):
    policy_id = str(round_obj.get("policy_id", "") or "").strip().lower()
    if policy_id in ("solo", "no_ai"):
        return False

    ai_log = ((round_obj.get("action_log", {}) or {}).get("ai", []) or [])
    return len(ai_log) > 0

def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _minmax_dict(values_by_key):
    numeric = {k: v for k, v in values_by_key.items() if v is not None and not pd.isna(v)}
    if not numeric:
        return {k: None for k in values_by_key}

    vmin = min(numeric.values())
    vmax = max(numeric.values())

    if vmin == vmax:
        return {k: 0.5 if values_by_key[k] is not None and not pd.isna(values_by_key[k]) else None
                for k in values_by_key}

    out = {}
    for k, v in values_by_key.items():
        if v is None or pd.isna(v):
            out[k] = None
        else:
            out[k] = (v - vmin) / (vmax - vmin)
    return out

def write_coadaptation_summary_csv(log_payload, out_csv_path):

    episodes = log_payload.get("episodes", []) or []
    rounds = log_payload.get("rounds", []) or []

    feedback_by_episode = {}
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        fb = ep.get("feedback", {}) or {}
        feedback_by_episode[ep_idx] = {
            "mental_demand": fb.get("mental_demand"),
            "performance_score": fb.get("performance"),
        }

    sorted_rounds = sorted(rounds, key=lambda r: r.get("round_index_global", 0))
    prev_ai_round = None
    rows = []

    # First pass: build round-level rows
    for r in sorted_rounds:
        if not _round_has_ai(r):
            continue

        summary = r.get("summary", {}) or {}
        ep_idx = r.get("episode_index")
        feedback = feedback_by_episode.get(ep_idx, {})

        human_stability = None
        ai_stability = None
        joint_stability = None

        if prev_ai_round is not None:
            curr_human = _extract_human_action_sequence(r)
            prev_human = _extract_human_action_sequence(prev_ai_round)

            curr_ai = _extract_ai_action_sequence(r)
            prev_ai = _extract_ai_action_sequence(prev_ai_round)

            curr_joint = _extract_joint_action_sequence(r)
            prev_joint = _extract_joint_action_sequence(prev_ai_round)

            human_stability = _sequence_similarity(prev_human, curr_human)
            ai_stability = _sequence_similarity(prev_ai, curr_ai)
            joint_stability = _sequence_similarity(prev_joint, curr_joint)

        rows.append({
            "episode_index": r.get("episode_index"),
            "round_in_episode": r.get("round_in_episode"),
            "policy_id": r.get("policy_id"),
            "human_stability_prev_ai_round": human_stability,
            "ai_stability_prev_ai_round": ai_stability,
            "joint_stability_prev_ai_round": joint_stability,
            "mental_demand": feedback.get("mental_demand"),
            "performance_score": feedback.get("performance_score"),
            "dishes_served": summary.get("dishes_served"),
            "team_reward_score": summary.get("team_reward_score"),
            "human_steps": summary.get("human_steps"),
            "ai_steps": summary.get("ai_steps"),
        })

        prev_ai_round = r

    if not rows:
        fieldnames = [
            "episode_index",
            "round_in_episode",
            "policy_id",
            "human_stability_prev_ai_round",
            "ai_stability_prev_ai_round",
            "joint_stability_prev_ai_round",
            "mental_demand",
            "performance_score",
            "dishes_served",
            "team_reward_score",
            "human_steps",
            "ai_steps",
            "episode_coadaptation_score",
        ]
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    # Second pass: compute episode-level score
    df = pd.DataFrame(rows)

    episode_stats = (
        df.groupby("episode_index", dropna=False)
        .agg(
            mean_human_stability=("human_stability_prev_ai_round", "mean"),
            mean_ai_stability=("ai_stability_prev_ai_round", "mean"),
            mean_joint_stability=("joint_stability_prev_ai_round", "mean"),
            mental_demand=("mental_demand", "mean"),
            performance_score=("performance_score", "mean"),
            total_dishes_served=("dishes_served", "sum"),
            total_team_reward_score=("team_reward_score", "sum"),
            mean_ai_steps=("ai_steps", "mean"),
        )
        .reset_index()
    )

    # Normalize subjective + outcome within participant
    md_norm = _minmax_dict(dict(zip(episode_stats["episode_index"], episode_stats["mental_demand"])))
    perf_norm = _minmax_dict(dict(zip(episode_stats["episode_index"], episode_stats["performance_score"])))
    dishes_norm = _minmax_dict(dict(zip(episode_stats["episode_index"], episode_stats["total_dishes_served"])))
    reward_norm = _minmax_dict(dict(zip(episode_stats["episode_index"], episode_stats["total_team_reward_score"])))

    score_by_episode = {}

    for _, ep_row in episode_stats.iterrows():
        ep_idx = ep_row["episode_index"]

        behavior_score = (
            0.2 * (ep_row["mean_human_stability"] if pd.notna(ep_row["mean_human_stability"]) else 0.0) +
            0.2 * (ep_row["mean_ai_stability"] if pd.notna(ep_row["mean_ai_stability"]) else 0.0) +
            0.6 * (ep_row["mean_joint_stability"] if pd.notna(ep_row["mean_joint_stability"]) else 0.0)
        )

        subjective_score = (
            0.5 * (1 - (md_norm.get(ep_idx) if md_norm.get(ep_idx) is not None else 0.5)) +
            0.5 * ((perf_norm.get(ep_idx) if perf_norm.get(ep_idx) is not None else 0.5))
        )

        outcome_score = (
            0.5 * ((dishes_norm.get(ep_idx) if dishes_norm.get(ep_idx) is not None else 0.5)) +
            0.5 * ((reward_norm.get(ep_idx) if reward_norm.get(ep_idx) is not None else 0.5))
        )

        score = 100 * (
            0.45 * behavior_score +
            0.30 * subjective_score +
            0.25 * outcome_score
        )

        # Penalty for almost-frozen AI
        mean_ai_steps = ep_row["mean_ai_steps"] if pd.notna(ep_row["mean_ai_steps"]) else 0
        total_dishes = ep_row["total_dishes_served"] if pd.notna(ep_row["total_dishes_served"]) else 0
        if mean_ai_steps < 5 and total_dishes <= 1:
            score -= 10

        score = max(0, min(100, score))
        score_by_episode[ep_idx] = score

    df["episode_coadaptation_score"] = df["episode_index"].map(score_by_episode)

    fieldnames = [
        "episode_index",
        "round_in_episode",
        "policy_id",
        "human_stability_prev_ai_round",
        "ai_stability_prev_ai_round",
        "joint_stability_prev_ai_round",
        "mental_demand",
        "performance_score",
        "dishes_served",
        "team_reward_score",
        "human_steps",
        "ai_steps",
        "episode_coadaptation_score",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in df.to_dict(orient="records"):
            writer.writerow(row)


def write_round_summary_csv(log_payload, out_csv_path):
    prolific = str(log_payload.get("prolificId", "unknown")).strip().replace("/", "_")
    episodes = log_payload.get("episodes", []) or []
    rounds = log_payload.get("rounds", []) or []

    feedback_by_episode = {}
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        fb = ep.get("feedback", {}) or {}
        feedback_by_episode[ep_idx] = {
            "mental_demand": fb.get("mental_demand"),
            "performance": fb.get("performance"),
        }

    fieldnames = [
        "prolific_id",
        "episode_index",
        "round_in_episode",
        "episode_phase",
        "experiment_phase",
        "map",
        "policy_id",
        "config_id",
        "dishes_served",
        "human_steps",
        "ai_steps",
        "team_reward_score",
        "human_reward_score",
        "ai_reward_score",
        "mental_demand",
        "performance_score",
        "end_time_iso",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(rounds, key=lambda x: x.get("round_index_global", 0)):
            summary = r.get("summary", {}) or {}
            ep_idx = r.get("episode_index")
            feedback = feedback_by_episode.get(ep_idx, {})

            writer.writerow({
                "prolific_id": prolific,
                "episode_index": ep_idx,
                "round_in_episode": r.get("round_in_episode"),
                "episode_phase": r.get("episode_phase"),
                "experiment_phase": r.get("experiment_phase"),
                "map": r.get("map"),
                "policy_id": r.get("policy_id"),
                "config_id": r.get("configId"),
                "dishes_served": summary.get("dishes_served"),
                "human_steps": summary.get("human_steps"),
                "ai_steps": summary.get("ai_steps"),
                "team_reward_score": summary.get("team_reward_score"),
                "human_reward_score": summary.get("human_reward_score"),
                "ai_reward_score": summary.get("ai_reward_score"),
                "mental_demand": feedback.get("mental_demand"),
                "performance_score": feedback.get("performance"),
                "end_time_iso": r.get("endTimeISO"),
            })


def write_participant_summary_csv(log_payload, out_csv_path):
    meta = log_payload.get("meta", {}) or {}
    assignment = meta.get("assignment", {}) or {}
    rounds = log_payload.get("rounds", []) or []
    episodes = log_payload.get("episodes", []) or []

    replayed_best_policy_id = None
    for ep in episodes:
        if ep.get("episode_phase") in ("bo_replay_best", "replay_optimal"):
            replayed_best_policy_id = ep.get("optimal_policy_id") or ep.get("policy_id")
            if replayed_best_policy_id:
                break

    fieldnames = [
        "prolific_id",
        "age",
        "gender",
        "experience",
        "assigned_map",
        "consent_given",
        "start_time_iso",
        "n_episodes_saved",
        "n_rounds_saved",
        "replayed_best_policy_id",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "prolific_id": log_payload.get("prolificId"),
            "age": meta.get("age"),
            "gender": meta.get("gender"),
            "experience": meta.get("experience"),
            "assigned_map": assignment.get("map"),
            "consent_given": meta.get("consentGiven"),
            "start_time_iso": meta.get("startTimeISO"),
            "n_episodes_saved": len(episodes),
            "n_rounds_saved": len(rounds),
            "replayed_best_policy_id": replayed_best_policy_id,
        })


def save_participant_snapshot(log_payload):
    """Save participant data in the final agreed format."""
    prolific = str(log_payload.get('prolificId', 'unknown')).strip().replace('/', '_')

    participant_dir = os.path.join(SUBMISSIONS_ROOT, prolific)
    os.makedirs(participant_dir, exist_ok=True)

    # 1) Full raw JSON
    result_filename = os.path.join(participant_dir, 'final_result.json')
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)

    # 2) Main analysis CSV: one row per round
    round_csv = os.path.join(participant_dir, 'round_summary.csv')
    write_round_summary_csv(log_payload, round_csv)

    # 3) Participant-level summary CSV
    participant_csv = os.path.join(participant_dir, 'participant_summary.csv')
    write_participant_summary_csv(log_payload, participant_csv)

    # 4) Co-adaptation summary CSV
    coadapt_csv = os.path.join(participant_dir, 'coadaptation_summary.csv')
    write_coadaptation_summary_csv(log_payload, coadapt_csv)

    # Optional: keep BayesOpt state if present
    for (pid, map_name), optimizer in OPTIMIZER_MGR.optimizers.items():
        if pid != prolific:
            continue

        optimizer_filename = os.path.join(participant_dir, f'bayesopt_state_{map_name}.json')
        try:
            optimizer.save(optimizer_filename)
        except Exception as e:
            logger.info(
                f"[BACKEND - SNAPSHOT] warning: could not save BayesOpt state for {prolific}, map={map_name}: {e}"
            )

    return participant_dir

@app.route('/save_progress', methods=['POST'])
def save_progress():
    try:
        data = request.get_json(silent=True)
        if data is None:
            raw = request.data.decode('utf-8', errors='ignore')
            data = json.loads(raw) if raw else {}

        log_payload = data.get('log', data)

        if not isinstance(log_payload, dict) or 'rounds' not in log_payload:
            return jsonify(success=False, error="Invalid payload: 'rounds' missing"), 400

        save_participant_snapshot(log_payload)
        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/submit_log', methods=['POST'])
def submit_log():
    """Receive the final logData json from the frontend, save the participant snapshot, then return the completion code."""
    try:
        data = request.get_json(silent=True) or {}
        log_payload = data.get('log', data)

        if not isinstance(log_payload, dict) or 'rounds' not in log_payload:
            return jsonify(success=False, error="Invalid payload: 'rounds' missing"), 400

        save_participant_snapshot(log_payload)

        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
    
''' @app.route('/submit_log', methods=['POST'])
def submit_log():
    """Receive the logData json from the frontend, save the json and csv to the server, then return the Prolific completion code."""
    try:
        data = request.get_json(silent=True) or {}
        log_payload = data.get('log', data)
        if not isinstance(log_payload, dict) or 'rounds' not in log_payload:
            return jsonify(success=False, error="Invalid payload: 'rounds' missing"), 400

        # Prolific completion code.
        completion_code = "CK4KW637"

        prolific = str(log_payload.get('prolificId', 'unknown')).strip().replace('/', '_')

        # Create a folder for this participant
        participant_dir = os.path.join('submissions', prolific)
        os.makedirs(participant_dir, exist_ok=True)

        # Save final_result.json in the participant's folder
        result_filename = os.path.join(participant_dir, 'final_result.json')
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(log_payload, f, ensure_ascii=False, indent=2)

        # Save round_summary.csv in the participant's folder
        csv_filename = os.path.join(participant_dir, 'round_summary.csv')
        write_round_summary_csv(log_payload, csv_filename)

        # Save BayesOpt state(s) for this participant, if any exist
        for (pid, map_name), optimizer in OPTIMIZER_MGR.optimizers.items():
            if pid != prolific:
                continue

            optimizer_filename = os.path.join(participant_dir, f'bayesopt_state_{map_name}.json')
            try:
                optimizer.save(optimizer_filename)
            except Exception as e:
                logger.info(
                    f"[BACKEND - SUBMIT] warning: could not save BayesOpt state for {prolific}, map={map_name}: {e}"
                )

        return jsonify(success=True, completion_code=completion_code)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500 '''

# =========================
# run the server
# =========================

# if __name__ == '__main__':

#     app.run(host='0.0.0.0', port=5000, debug=True)

#if __name__ == '__main__':
#    from waitress import serve
#    serve(app, host='0.0.0.0', port=5000, threads=8)

if __name__ == '__main__':
    print("STARTING BACKEND")
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)