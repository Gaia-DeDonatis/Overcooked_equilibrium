# -*- coding: utf-8 -*-
import os
import numpy as np
import sys

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


from flask import Flask, jsonify, request
from flask_cors import CORS
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
import random
import gym_macro_overcooked

print("--> Loaded gym_macro_overcooked (Environments: Overcooked-equilibrium-v0)")

from gym_macro_overcooked.items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food, DirtyPlate


app = Flask(__name__)
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



KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2}
ACTION_TO_KEY = {v: k for k, v in KEYS_ACTIONS.items()}
ACTION_TO_KEY[4] = "Stay"



MAX_STEPS = 200


# =========================
# fixed map, random AI policy
# =========================
FIXED_MAP_TYPE = "circle"
FIXED_GRID_DIM = [5, 5]

POLICY_POOL_DIR = os.path.join(current_dir, "policy_pool")


# This function is used for randomly picking one policy from the AI policy pool.

def _pick_random_policy_checkpoint():
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



_model_cache_by_path = {}



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





def create_envs_for_session(sess: Session, config_id: str):

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
    reset_step = 100
    sess.wrapper = SingleAgentWrapper_accept_keyboard_action(
        sess.env_mac, agent_index=0, reset_step=reset_step
    )

    # load the model
    if is_practice:
        sess.chosen_policy_dir = None
        sess.chosen_ckpt_path = None
        sess.model = None
    else:
        chosen_dir, ckpt_path = _pick_random_policy_checkpoint()
        sess.chosen_policy_dir = chosen_dir
        sess.chosen_ckpt_path = ckpt_path
        sess.model = _load_or_get_model_by_ckpt_path(ckpt_path)


    sess.obs = sess.wrapper.reset()

    # log the config
    if is_practice:
        sess.config_id = "layout_practice"
        sess.current_layout_id = "layout_practice"
        sess.current_model_id  = "none"
    else:
        sess.config_id = config_id
        sess.current_layout_id = "fixed_circle_5x5"
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
@app.route('/new_session', methods=['POST'])
def new_session():
    sid = SESSION_MGR.new_session()
    return jsonify(success=True, session_id=sid)

@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json(silent=True) or {}
    sid = data.get('session_id')
    if not sid:
        return jsonify(success=False, error="session_id is required"), 400


    # Now, although I parse the config from the frontend, I never use them. I just use the fixed circle layout.
    layout_id = data.get('layout_id')
    model_id  = data.get('model_id')
    config_id = data.get('config_id')

    sess = SESSION_MGR.ensure(sid)
    with sess.lock:
        try:
            create_envs_for_session(sess, config_id=(config_id or "IGNORED"))
        except Exception as e:
            return jsonify(success=False, error=str(e)), 400

        steps_left = MAX_STEPS
        print(f"[RESET][{sid}] map_type={FIXED_MAP_TYPE} grid_dim={FIXED_GRID_DIM} "
            f"chosen_policy_dir={sess.chosen_policy_dir} ckpt={sess.chosen_ckpt_path}")


        return jsonify(
            success=True,
            state=extract_state(sess),
            steps_left=steps_left,
            cumulative_reward=sess.cumulative_reward,
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id,
            chosen_policy_dir=sess.current_model_id,
            chosen_ckpt=os.path.basename(sess.chosen_ckpt_path) if sess.chosen_ckpt_path else None
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

        KEYS_ACTIONS = {'ArrowUp':3,'ArrowRight':0,'ArrowDown':1,'ArrowLeft':2}
        if key in KEYS_ACTIONS:
            # index 0 is AI, index 1 is the human user
            if True:
                t0 = time.time()
                if sess.model is not None:
                    with torch.no_grad():
                        ai_action, _ = sess.model.predict(sess.obs, deterministic=True)
                    ai_action_int = _as_int_action(ai_action)   # convert action to integer
                else:
                    ai_action_int = 4  # stay
                t1 = time.time()

                if sess.model is not None:
                    primitive_action, _ = sess.env_mac._computeLowLevelActions([ai_action_int, 0])
                else:
                    primitive_action = [4] * sess.env.n_agent

                action = [4] * sess.env.n_agent
                action[1] = KEYS_ACTIONS[key]
                action[0] = primitive_action[0]

                # record the robot's step
                robot_low = int(action[0])
                robot_key = ACTION_TO_KEY.get(robot_low, "Unknown")
                sess.robot_steps.append({
                    "step": int(sess.cur_step + 1),
                    "ai_macro_action": ai_action_int,
                    "low_level_action": robot_low,
                    "arrow": robot_key,
                    "timestamp": time.time(),
                })



                sess.obs, rewards, dones, info = sess.wrapper.step(action[0], action[1])

                try:
                # Robustly handle rewards (Scalar, List, or Numpy Array)
                    if isinstance(rewards, (list, tuple, np.ndarray)):
                        r_flat = np.array(rewards).flatten()
                        r = float(r_flat[0])
                    else:
                        r = float(rewards)

                    sess.cumulative_reward += r
                
                    if r >= 199:
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
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id,
            dishes_served=sess.dishes_served,
            robot_last_action=(sess.robot_steps[-1] if sess.robot_steps else None)
        )

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


        # Prolific completion code. remember change to your code.
        completion_code = "C108AMXR"


        os.makedirs('submissions', exist_ok=True)
        ts = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        prolific = (log_payload.get('prolificId') or 'anon').strip().replace('/', '_')
        filename = f"submissions/{ts}_{prolific}_{completion_code}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_payload, f, ensure_ascii=False, indent=2)

        return jsonify(success=True, completion_code=completion_code)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# =========================
# run the server
# =========================

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)