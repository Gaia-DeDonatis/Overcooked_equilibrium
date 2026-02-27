# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
from BayesOpt import TSNEBayesOptimizer

# Equilibrium_project" folder
current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(current_dir, 'Equilibrium_project')

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
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
import random
import gym_macro_overcooked

print("--> Loaded gym_macro_overcooked (Environments: Overcooked-equilibrium-v0)")

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
    except TypeError:
        env.seed(seed)
        env.reset()
    try: env.action_space.seed(seed)
    except: pass
    try: env.observation_space.seed(seed)
    except: pass



KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2,  'Stay': 4}
ACTION_TO_KEY = {v: k for k, v in KEYS_ACTIONS.items()}
ACTION_TO_KEY[4] = "Stay"



MAX_STEPS = 200


# =========================
# fixed map, random AI policy
# =========================
FIXED_MAP_TYPE = "cramped"
FIXED_GRID_DIM = [5, 5]

POLICY_POOL_DIR = os.path.join(current_dir, "policy_pool")


# This function is used for randomly picking one policy from the AI policy pool.

def _pick_random_policy_checkpoint(exclude_policy_names=None):
    """
    pick an model from /policy_pool/
    """
    if not os.path.isdir(POLICY_POOL_DIR):
        raise FileNotFoundError(f"POLICY_POOL_DIR not found: {POLICY_POOL_DIR}")

    subdirs = [
        os.path.join(POLICY_POOL_DIR, d)
        for d in os.listdir(POLICY_POOL_DIR)
        if os.path.isdir(os.path.join(POLICY_POOL_DIR, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found under: {POLICY_POOL_DIR}")

    # Optional: sample without replacement (per-session) when possible.
    if exclude_policy_names:
        exclude_set = set(exclude_policy_names)
        filtered = [d for d in subdirs if os.path.basename(d) not in exclude_set]
        if filtered:
            subdirs = filtered

    chosen_dir = [d for d in subdirs if "agent0" in d]
    chosen_dir = random.choice(subdirs)
    ckpt_path = os.path.join(chosen_dir, "model_500000.zip")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"model_500000.zip not found in: {chosen_dir}")

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

    def reset(self):
        self.obs = self.env.reset()
        self.env_reset_step = 0
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

        return self.obs[self.agent_index], rewards[self.agent_index] + rewards[1 - self.agent_index], dones, info



def _as_int_action(a):
    """Robustly cast SB3 action to a Python int."""
    if isinstance(a, (int, np.integer)):
        return int(a)
    a_np = np.asarray(a)
    if a_np.ndim == 0:        # numpy 0-d scalar
        return int(a_np.item())
    return int(a_np.flatten()[0])



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
        self.current_layout_id = None
        self.current_model_id = None
        self.robot_steps = []       # save each step of the AI agent
        self.last_access = time.time()
        self.lock = threading.RLock()
        self.chosen_policy_dir = None
        self.chosen_ckpt_path = None
       
        # Experiment structure (episodes)
        self.episode_index = None
        self.round_in_episode = None
        self.episode_phase = None
        self.used_policy_names = []  # basenames of chosen policy dirs


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
        
    def optimizer_exists(self, prolofic_id):
        return prolofic_id in self.optimizers
           
    def create_optimizer(self, prolific_id):
        if self.optimizer_exists(prolific_id):
            return jsonify(success=False, error="Optimizer already created for participant"), 400
        self.optimizers[prolific_id] =TSNEBayesOptimizer(embedding_csv="tsnt.csv", n_init=3, verbose=True)
    
        
OPTIMIZER_MGR = OptimizerManager()




_model_cache_by_path = {}

def _pick_policy_checkpoint(policy:str):
    policy = "[equilibrium]agent0_"+policy
    chosen_dir = os.path.join(POLICY_POOL_DIR, policy)
    ckpt_pth = os.path.join(chosen_dir, "model_500000.zip")
    return chosen_dir, ckpt_pth

# This function is used for loading an AI model or getting an existing AI model.
def _load_or_get_model_by_ckpt_path(ckpt_path: str):
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



def create_envs_for_session(sess: Session, config_id: str, choose_new_policy: bool = True, optimizer: TSNEBayesOptimizer = None):

    # Now, I use a fixed map, which is the circle (5*5). And for each time, I randomly load a model from the policy_pool

    # Fix the environment ID
    env_id     = "Overcooked-equilibrium-v0"
    mac_env_id = "Overcooked-MA-equilibrium-v0"

    grid_dim = FIXED_GRID_DIM
    n_agent = 2

    # Judge whether the current phase is the practicing phase
    is_practice = (config_id == "layout_practice")
    if is_practice:
        env_params = {
            'grid_dim': [5, 5],
            'task': ["lettuce salad"],
            'rewardList': rewardList,
            'map_type': "A",
            'n_agent': n_agent,
            'obs_radius': 0,
            'mode': "vector",
            'debug': True
        }
    else:
        # If it is not the practicing phase
        env_params = {
            'grid_dim': grid_dim,
            'task': ["lettuce salad"],
            'rewardList': rewardList,
            'map_type': FIXED_MAP_TYPE,   # <- Using the circle map
            'n_agent': n_agent,
            'obs_radius': 0,
            'mode': "vector",
            'debug': True
        }

    # Close the old env
    if sess.env is not None:
        try: sess.env.close()
        except: pass
    if sess.env_mac is not None:
        try: sess.env_mac.close()
        except: pass

    # Create env
    sess.env = gym.make(env_id, **env_params)
    _seed_env_everything(sess.env, SEED)
    sess.env.reset()

    sess.env_mac = gym.make(mac_env_id, **env_params)
    _seed_env_everything(sess.env_mac, SEED)

    # wrapper the env
    reset_step = 10000
    sess.wrapper = SingleAgentWrapper_accept_keyboard_action(
        sess.env_mac, agent_index=0, reset_step=reset_step
    )

    # load the model
    if is_practice:
        sess.chosen_policy_dir = None
        sess.chosen_ckpt_path = None
        sess.model = None
    else:
        # Pick a new AI policy only when starting a new episode.
        should_pick = bool(choose_new_policy) or (not sess.chosen_ckpt_path)

        if should_pick:
            if optimizer is not None:
                print("Picking policy using BO pipeline")
                mapped_trials = optimizer.ask()
                checkpoint = mapped_trials[optimizer._actual_trial_idx]['policy']
                chosen_dir, ckpt_path = _pick_policy_checkpoint(checkpoint)
                sess.chosen_policy_dir = chosen_dir
                sess.chosen_ckpt_path = ckpt_path
                sess.model = _load_or_get_model_by_ckpt_path(ckpt_path)
                 
            else:
                print("Picking a random policy")
                chosen_dir, ckpt_path = _pick_random_policy_checkpoint(exclude_policy_names=sess.used_policy_names)
                sess.chosen_policy_dir = chosen_dir
                sess.chosen_ckpt_path = ckpt_path

            policy_name = os.path.basename(chosen_dir)
            if policy_name not in sess.used_policy_names:
                sess.used_policy_names.append(policy_name)
        
        assert "agent0" in ckpt_path, "loading agent1 somwhere"
        # Always (re)load the model into the freshly created env.
        sess.model = _load_or_get_model_by_ckpt_path(sess.chosen_ckpt_path)


    sess.obs = sess.wrapper.reset()

    # log the config
    if is_practice:
        sess.config_id = "layout_practice"
        sess.current_layout_id = "layout_practice"
        sess.current_model_id  = "none"
    else:
        sess.config_id = config_id
        sess.current_layout_id = "fixed_cramped_5x5"
        sess.current_model_id  = os.path.basename(sess.chosen_policy_dir)

    # reset the step count and reward
    sess.cur_step = 0
    sess.cumulative_reward = 0.0
    sess.robot_steps = []





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

    # metadata (optional)
    episode_index = data.get('episode_index', None)
    round_in_episode = data.get('round_in_episode', None)
    episode_phase = data.get('episode_phase', None)
    new_episode = data.get('new_episode', None)

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
    
    if not OPTIMIZER_MGR.optimizer_exists(prolific):
        OPTIMIZER_MGR.create_optimizer(prolific)
        print("OPTIMZER CONFIGURED")
        
    optimizer = OPTIMIZER_MGR.optimizers[prolific]

    
    with sess.lock:
        try:
            # Force a new policy if the episode index changed.
            episode_changed = (
                episode_index_int is not None
                and sess.episode_index is not None
                and episode_index_int != sess.episode_index
            )
            choose_new_policy = (not is_practice) and (bool(new_episode) or episode_changed)

            # Persist metadata in the session.
            if episode_index_int is not None:
                sess.episode_index = episode_index_int
            sess.round_in_episode = round_in_episode_int
            sess.episode_phase = episode_phase
            create_envs_for_session(sess, config_id=(config_id or "IGNORED"), choose_new_policy=choose_new_policy, optimizer=optimizer)
        
        except Exception as e:
            return jsonify(success=False, error=str(e)), 400

        steps_left = MAX_STEPS
        print(
            f"[RESET][{sid}] episode={sess.episode_index} round_in_episode={sess.round_in_episode} phase={sess.episode_phase} "
            f"map_type={FIXED_MAP_TYPE} grid_dim={FIXED_GRID_DIM} policy={sess.current_model_id} ckpt={sess.chosen_ckpt_path}"
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

            map_type=FIXED_MAP_TYPE if config_id != "layout_practice" else "A",
            grid_dim=FIXED_GRID_DIM if config_id != "layout_practice" else [5, 5],
            policy_id=sess.current_model_id
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

    sess = SESSION_MGR.ensure(sid)
    with sess.lock:
        # When the config id changes, hot switch the env.
        if layout_id or model_id or config_id:
            try:
                target_cfg_id = _parse_config_id(layout_id=layout_id, model_id=model_id, config_id=config_id)
                if target_cfg_id != sess.config_id:
                    create_envs_for_session(sess, target_cfg_id)
                    print(f"[HOT-SWITCH][{sid}] -> {target_cfg_id}")
            except Exception as e:
                return jsonify(success=False, error=f"hot switch failed: {e}"), 400
            
        if not hasattr(sess, 'dishes_served'): sess.dishes_served = 0
        if sess.cur_step == 0: sess.dishes_served = 0

        KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2, 'Stay': 4}

        if key in KEYS_ACTIONS:
            t0 = time.time()

            if sess.model is not None:
                with torch.no_grad():
                    ai_action, _ = sess.model.predict(sess.obs, deterministic=True)
                ai_action_int = _as_int_action(ai_action)
            else:
                ai_action_int = 4  # stay

            t1 = time.time()

            if sess.model is not None:
                if ai_action_int == 4:
                    primitive_action = [4] * sess.env.n_agent
                else:
                    primitive_action, _ = sess.env_mac._computeLowLevelActions([ai_action_int, 0])
            else:
                primitive_action = [4] * sess.env.n_agent

            action = [4] * sess.env.n_agent
            action[1] = KEYS_ACTIONS[key]           
            action[0] = int(primitive_action[0])

            robot_low = int(action[0])
            robot_key = ACTION_TO_KEY.get(robot_low, "Unknown")
            sess.robot_steps.append({
                "step": int(sess.cur_step + 1),
                "ai_macro_action": int(ai_action_int),
                "low_level_action": robot_low,
                "arrow": robot_key,
                "timestamp": time.time(),
            })


            sess.obs, rewards, dones, info = sess.wrapper.step(action[0], action[1])
            
            r_env = 0.0
            r_adjusted = 0.0

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
                ai_low = int(action[0])
                human_low = int(action[1])

                r_adjusted = r_env
                if ai_low == 4:
                   r_adjusted += (-step_pen_ai) # add back +1 if penalty is -1
                if human_low == 4:
                    r_adjusted += (-step_pen_hu)

                sess.cumulative_reward += r_adjusted

                # dish served detection
                if r_env >= 150:
                    sess.dishes_served += 1
                    print(f"[{sid}] DISH SERVED! Total: {sess.dishes_served}")

            except Exception as e:
                print(f"Error updating rewards: {e}")
                
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
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id,
            dishes_served=sess.dishes_served,
            robot_last_action=(sess.robot_steps[-1] if sess.robot_steps else None)
        )


@app.route('/tell', methods=['POST'])
def tell():
    data = request.get_json(silent=True) or {}

    score_raw = data.get('score')
    if score_raw is None:
        return jsonify(success=False, error="score is required"), 400
    score = float(score_raw)

    prolific_raw = data.get('prolificId')
    if not prolific_raw:
        return jsonify(success=False, error="prolific_id is required"), 400
    prolific = prolific_raw.strip().replace('/', '_')

    
    trial_idx = OPTIMIZER_MGR.optimizers[prolific]._actual_trial_idx
    OPTIMIZER_MGR.optimizers[prolific].tell({trial_idx: score})
    return jsonify(success=True)


@app.route('/close_optimizer', methods=['POST'])
def close_optimizer():
    data = request.get_json(silent=True) or {}
    prolific = (data.get('prolificId') or 'anon').strip().replace('/', '_')
    if not prolific:
        return jsonify(success=False, error="prolific_id is required"), 400
    OPTIMIZER_MGR.optimizers[prolific].close()
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


@app.route('/submit_log', methods=['POST'])
def submit_log():
    """Receive the logData json from the frontend, and save the json to the server. Then return the Prolific completion code."""
    try:
        data = request.get_json(silent=True) or {}
        log_payload = data.get('log', data)
        if not isinstance(log_payload, dict) or 'rounds' not in log_payload:
            return jsonify(success=False, error="Invalid payload: 'rounds' missing"), 400

        # Prolific completion code.
        completion_code = "CK4KW637"

        prolific = log_payload.get('prolificId').strip().replace('/', '_')

        # Create a folder for this participant
        participant_dir = os.path.join('submissions', prolific)
        os.makedirs(participant_dir, exist_ok=True)

        # Save final_result.json in the participant's folder
        result_filename = os.path.join(participant_dir, 'final_result.json')
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(log_payload, f, ensure_ascii=False, indent=2)

        # Save BayesOpt state for this participant if optimizer exists
        if OPTIMIZER_MGR.optimizer_exists(prolific):
            optimizer_filename = os.path.join(participant_dir, 'bayesopt_state.json')
            try:
                OPTIMIZER_MGR.optimizers[prolific].save(optimizer_filename)
            except Exception as e:
                print(f"Warning: Could not save BayesOpt state for {prolific}: {e}")

        return jsonify(success=True, completion_code=completion_code)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# =========================
# run the server
# =========================

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)