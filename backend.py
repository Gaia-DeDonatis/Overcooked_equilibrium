# -*- coding: utf-8 -*-
import os

import sys

# Equilibrium_project" folder
current_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(current_dir, 'Equilibrium_project')

# Insert at index 0 so this folder takes priority over everything else
sys.path.insert(0, target_folder) 

# —— 彻底约束多线程BLAS带来的非确定性
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# —— Python哈希随机化，避免dict/集合遍历顺序引入差异（某些tie-break会受影响）
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


# 关闭自动 Benchmark（会选择不同算法）
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
except Exception:
    pass

# 严格确定性（某些算子不支持会抛异常，可按需降级到 warn=True）
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# 限制线程，避免并行调度带来的微小非确定性
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

# 你的感知模型工具（如未使用可删）
from utils import ABIGatedExtractorWithConf, LightweightTransformerABIModel_adaptivelength

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
        # 兼容旧 Gym
        env.seed(seed)
        env.reset()
    try: env.action_space.seed(seed)
    except: pass
    try: env.observation_space.seed(seed)
    except: pass



KEYS_ACTIONS = {'ArrowUp': 3, 'ArrowRight': 0, 'ArrowDown': 1, 'ArrowLeft': 2}
ACTION_TO_KEY = {v: k for k, v in KEYS_ACTIONS.items()}
ACTION_TO_KEY[4] = "Stay"   # 4 表示不动（环境里通常是“停留”）


# =========================
# 全局常量 & 配置
# =========================
MAX_STEPS = 200

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

# 一个键对应一个 (layout, model) 组合
LAYOUT_CONFIG = {
    "layout_practice": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A",
        "model":      "none",         # 练习局：不加载RL模型，AI驻留
        "grid_dim":   [5, 5],
        "n_agent":    2
    },


    "layout1_model1": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B",
        "model":      "layout1_model1",  # TrustPOMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout1_model2": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B",
        "model":      "layout1_model2",  # FCP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout1_model3": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B",
        "model":      "layout1_model3",  # MEP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout1_model4": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B",
        "model":      "layout1_model4",  # POMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },



    "layout2_model1": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A",
        "model":      "layout2_model1",  # TrustPOMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout2_model2": {
       "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A",
        "model":      "layout2_model2",  # FCP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout2_model3": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A",
        "model":      "layout2_model3",  # MEP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout2_model4": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A",
        "model":      "layout2_model4",  # POMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },



    "layout3_model1": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B_lowuncertainty",
        "model":      "layout3_model1",  # TrustPOMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout3_model2": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B_lowuncertainty",
        "model":      "layout3_model2",  # FCP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout3_model3": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B_lowuncertainty",
        "model":      "layout3_model3",  # MEP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout3_model4": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "B_lowuncertainty",
        "model":      "layout3_model4",  # POMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },



    "layout4_model1": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A_lowuncertainty",
        "model":      "layout4_model1",  # TrustPOMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout4_model2": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A_lowuncertainty",
        "model":      "layout4_model2",  # FCP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout4_model3": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A_lowuncertainty",
        "model":      "layout4_model3",  # MEP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
    "layout4_model4": {
        "env_id":     "Overcooked-equilibrium-v0",
        "mac_env_id": "Overcooked-MA-equilibrium-v0",
        "map_type":   "A_lowuncertainty",
        "model":      "layout4_model4",  # POMDP
        "grid_dim":   [15, 15],
        "n_agent":    2
    },
}

# SB3 模型路径（RL层）
# 注意：不同 key 的模型可能训练在不同的 obs 维度，请保持一致
MODEL_PATHS = {
    "layout1_model1": "userstudy_models/mapB_TrustPOMDP",
    "layout1_model2": "userstudy_models/mapB_FCP",
    "layout1_model3": "userstudy_models/mapB_MEP",
    "layout1_model4": "userstudy_models/mapB_POMDP",

    "layout2_model1": "userstudy_models/mapA_TrustPOMDP",
    "layout2_model2": "userstudy_models/mapA_FCP",
    "layout2_model3": "userstudy_models/mapA_MEP",
    "layout2_model4": "userstudy_models/mapA_POMDP",

    "layout3_model1": "userstudy_models/mapB_lowuncertainty_TrustPOMDP",
    "layout3_model2": "userstudy_models/mapB_lowuncertainty_FCP",
    "layout3_model3": "userstudy_models/mapB_lowuncertainty_MEP",
    "layout3_model4": "userstudy_models/mapB_lowuncertainty_POMDP",

    "layout4_model1": "userstudy_models/mapA_lowuncertainty_TrustPOMDP",
    "layout4_model2": "userstudy_models/mapA_lowuncertainty_FCP",
    "layout4_model3": "userstudy_models/mapA_lowuncertainty_MEP",
    "layout4_model4": "userstudy_models/mapA_lowuncertainty_POMDP",
    # 'none' 不需要
}




# ABI 感知模型的路径（包装器内部用）
# 根据你的说明：layout1 使用 MapB，layout2 使用 MapA
ABI_MODEL_PATHS = {
    "layout1_model1": "[MapB]abi_inference_model/abi_model_step_10000.pt",
    "layout2_model1": "[MapA]abi_inference_model/abi_model_step_10000.pt",
    "layout3_model1": "[MapB_lowuncertainty]abi_inference_model/abi_model_step_10000.pt",
    "layout4_model1": "[MapA_lowuncertainty]abi_inference_model/abi_model_step_10000.pt",
}

# =========================
# 包装器
# =========================

class SingleAgentWrapper_accept_keyboard_action(_gym.Wrapper):
    """基础包装器：不扩展 obs 维度"""
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


        # 返回：obs[agent_index], sum_rewards, dones, info
        return self.obs[self.agent_index], rewards[self.agent_index] + rewards[1 - self.agent_index], dones, info


class SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history30(_gym.Wrapper):
    """扩展 obs 维度，并在关键时刻调用 ABI 感知模型"""
    def __init__(self, env, agent_index, state_dim, reset_step, model_path=None):
        super(SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history30, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.partner_key_event_happen = False
        self.state_history = deque(maxlen=30)

        self.env_reset_step = 0

        self.reset_step = reset_step

        self.model = LightweightTransformerABIModel_adaptivelength(state_dim)
        if model_path:
            self.model.load_model(model_path)
        # —— 新增：关训练态
        self.model.eval()


        # 扩展 obs 维度（增加 6 维）
        if isinstance(env.observation_space, gym.spaces.Box):
            extra = 6
            low = np.append(env.observation_space.low,  [-np.inf] * extra)
            high = np.append(env.observation_space.high, [np.inf] * extra)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
        else:
            raise NotImplementedError("This wrapper only works with Box observation spaces.")

    def reset(self):
        self.obs = self.env.reset()

        self.env_reset_step = 0

        # 初始化 ABI 相关变量
        self.partner_A = 0.5
        self.partner_B = 0.5
        self.partner_I = 0.5
        self.partner_A_confidence = 0.0
        self.partner_B_confidence = 0.0
        self.partner_I_confidence = 0.0

        self.state_history.clear()
        partner_cented_obs = self.env._get_macro_vector_obs_for_ABImodel()
        self.state_history.append(partner_cented_obs[1])

        self.partner_key_event_happen = False

        return np.concatenate([
            self.obs[self.agent_index],
            [self.partner_A, self.partner_B, self.partner_I,
             self.partner_A_confidence, self.partner_B_confidence, self.partner_I_confidence]
        ])

    def step(self, action, keyboard_action):
        actions = [action, keyboard_action]

        prev_holding = self.env.agent[1].holding
        self.obs, rewards, dones, info = self.env.step(actions)
        self.obs = self.env._get_macro_obs()
        after_holding = self.env.agent[1].holding

        partner_cented_obs = self.env._get_macro_vector_obs_for_ABImodel()
        self.state_history.append(partner_cented_obs[1])

        self.partner_key_event_happen = (
            prev_holding is not None and after_holding is None
        )

        # 当轨迹累计到30步且发生关键事件时，调用 ABI 模型预测
        if len(self.state_history) == 30 and self.partner_key_event_happen:
            state_np = np.asarray(self.state_history, dtype=np.float32)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0)  # [1, T, D]
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)
            T_cur = state_np.shape[0]

            # 默认超参
            if not hasattr(self, 'posterior_decay'):  self.posterior_decay = 0.999
            if not hasattr(self, 'kappa_max'):        self.kappa_max = 2.0
            if not hasattr(self, 'conf_mode'):        self.conf_mode = "var"  # "var" | "strength" | "hybrid"
            if not hasattr(self, 'S_max'):            self.S_max = 50.0

            def _conf_from_ab(a: float, b: float):
                S = a + b
                if S <= 0.0:
                    mean = 0.5; var = 1.0/12.0
                else:
                    mean = a / S
                    var  = (a * b) / (S * S * (S + 1.0))
                std = math.sqrt(var)
                conf_var = max(0.0, min(1.0, 1.0 - 2.0 * std))
                conf_S   = S / (S + self.S_max)
                if self.conf_mode == "var":
                    conf = conf_var
                elif self.conf_mode == "strength":
                    conf = conf_S
                else:
                    conf = 0.5 * conf_var + 0.5 * conf_S
                return mean, var, std, conf, S

            with torch.no_grad():
                if hasattr(self.model, "predict_beta_params"):
                    out = self.model.predict_beta_params(state_tensor, lengths_np=[T_cur])
                    aA = float(out['A']['alpha']); bA = float(out['A']['beta'])
                    aB = float(out['B']['alpha']); bB = float(out['B']['beta'])
                    aI = float(out['I']['alpha']); bI = float(out['I']['beta'])
                else:
                    (aA_t,bA_t),(aB_t,bB_t),(aI_t,bI_t) = self.model(state_tensor, lengths=torch.tensor([T_cur]))
                    aA, bA = float(aA_t.item()), float(bA_t.item())
                    aB, bB = float(aB_t.item()), float(bB_t.item())
                    aI, bI = float(aI_t.item()), float(bI_t.item())

            meanA, varA, stdA, confA, SA = _conf_from_ab(aA, bA)
            meanB, varB, stdB, confB, SB = _conf_from_ab(aB, bB)
            meanI, varI, stdI, confI, SI = _conf_from_ab(aI, bI)

            # 即刻估计
            self.partner_A_prob = float(meanA)
            self.partner_B_prob = float(meanB)
            self.partner_I_prob = float(meanI)

            self.partner_A_conf = float(confA)
            self.partner_B_conf = float(confB)
            self.partner_I_conf = float(confI)

            # 维护平滑“后验”仅用于概率平滑（如需）
            def _update_posterior_for_smoothing(name, p, S_model):
                a_name = f'partner_{name}_alpha'
                b_name = f'partner_{name}_beta'
                if not hasattr(self, a_name): setattr(self, a_name, 1.0)
                if not hasattr(self, b_name): setattr(self, b_name, 1.0)
                setattr(self, a_name, getattr(self, a_name) * self.posterior_decay)
                setattr(self, b_name, getattr(self, b_name) * self.posterior_decay)
                kappa = min(S_model, self.kappa_max)
                setattr(self, a_name, getattr(self, a_name) + kappa * p)
                setattr(self, b_name, getattr(self, b_name) + kappa * (1.0 - p))
                A = getattr(self, a_name); B = getattr(self, b_name)
                S_post = A + B
                p_smooth = A / S_post if S_post > 0 else 0.5
                setattr(self, f'partner_{name}_prob_smooth', float(p_smooth))

            _update_posterior_for_smoothing('A', meanA, SA)
            _update_posterior_for_smoothing('B', meanB, SB)
            _update_posterior_for_smoothing('I', meanI, SI)

            # 若需要 ±1 标签
            # self.partner_A = 1 if self.partner_A_prob_smooth >= 0.5 else -1
            # self.partner_B = 1 if self.partner_B_prob_smooth >= 0.5 else -1
            # self.partner_I = 1 if self.partner_I_prob_smooth >= 0.5 else -1
            self.partner_A = self.partner_A_prob_smooth
            self.partner_B = self.partner_B_prob_smooth
            self.partner_I = self.partner_I_prob_smooth

            self.partner_A_confidence = self.partner_A_conf
            self.partner_B_confidence = self.partner_B_conf
            self.partner_I_confidence = self.partner_I_conf


        self.env_reset_step += 1

        if self.env_reset_step % self.reset_step == 0:
            self.env.soft_reset_obs_only()
            self.env.macroAgent[0].cur_macro_action_done = True
            self.env.macroAgent[1].cur_macro_action_done = True
            self.obs = self.env._get_macro_obs()


        return np.concatenate([
            self.obs[self.agent_index],
            [self.partner_A, self.partner_B, self.partner_I,
             self.partner_A_confidence, self.partner_B_confidence, self.partner_I_confidence]
        ]), rewards[self.agent_index] + rewards[1 - self.agent_index], dones, info






class SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history10(_gym.Wrapper):
    """扩展 obs 维度，并在关键时刻调用 ABI 感知模型"""
    def __init__(self, env, agent_index, state_dim, reset_step, model_path=None):
        super(SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history10, self).__init__(env)
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.partner_key_event_happen = False
        self.state_history = deque(maxlen=10)

        self.env_reset_step = 0

        self.reset_step = reset_step

        self.model = LightweightTransformerABIModel_adaptivelength(state_dim)
        if model_path:
            self.model.load_model(model_path)
        # —— 新增：关训练态
        self.model.eval()


        # 扩展 obs 维度（增加 6 维）
        if isinstance(env.observation_space, gym.spaces.Box):
            extra = 6
            low = np.append(env.observation_space.low,  [-np.inf] * extra)
            high = np.append(env.observation_space.high, [np.inf] * extra)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
        else:
            raise NotImplementedError("This wrapper only works with Box observation spaces.")

    def reset(self):
        self.obs = self.env.reset()

        self.env_reset_step = 0

        # 初始化 ABI 相关变量
        self.partner_A = 0.5
        self.partner_B = 0.5
        self.partner_I = 0.5
        self.partner_A_confidence = 0.0
        self.partner_B_confidence = 0.0
        self.partner_I_confidence = 0.0

        self.state_history.clear()
        partner_cented_obs = self.env._get_macro_vector_obs_for_ABImodel()
        self.state_history.append(partner_cented_obs[1])

        self.partner_key_event_happen = False

        return np.concatenate([
            self.obs[self.agent_index],
            [self.partner_A, self.partner_B, self.partner_I,
             self.partner_A_confidence, self.partner_B_confidence, self.partner_I_confidence]
        ])

    def step(self, action, keyboard_action):
        actions = [action, keyboard_action]

        prev_holding = self.env.agent[1].holding
        self.obs, rewards, dones, info = self.env.step(actions)
        self.obs = self.env._get_macro_obs()
        after_holding = self.env.agent[1].holding

        partner_cented_obs = self.env._get_macro_vector_obs_for_ABImodel()
        self.state_history.append(partner_cented_obs[1])


        # Add a new criteria, which is cannot update ABI if the agent puts lettuce on a knife.

        distance_to_knife1 = self.env._calDistance(self.env.agent[1].x, self.env.agent[1].y, self.env.knife[0].x, self.env.knife[0].y)
        distance_to_knife2 = self.env._calDistance(self.env.agent[1].x, self.env.agent[1].y, self.env.knife[1].x, self.env.knife[1].y)
        

        self.partner_key_event_happen = (
            prev_holding is not None and after_holding is None and distance_to_knife1 != 1 and distance_to_knife2 != 1
        )


        # 当轨迹累计到30步且发生关键事件时，调用 ABI 模型预测
        if len(self.state_history) == 10 and self.partner_key_event_happen:
            state_np = np.asarray(self.state_history, dtype=np.float32)
            state_tensor = torch.from_numpy(state_np).unsqueeze(0)  # [1, T, D]
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)
            T_cur = state_np.shape[0]

            # 默认超参
            if not hasattr(self, 'posterior_decay'):  self.posterior_decay = 0.999
            if not hasattr(self, 'kappa_max'):        self.kappa_max = 2.0
            if not hasattr(self, 'conf_mode'):        self.conf_mode = "var"  # "var" | "strength" | "hybrid"
            if not hasattr(self, 'S_max'):            self.S_max = 50.0

            def _conf_from_ab(a: float, b: float):
                S = a + b
                if S <= 0.0:
                    mean = 0.5; var = 1.0/12.0
                else:
                    mean = a / S
                    var  = (a * b) / (S * S * (S + 1.0))
                std = math.sqrt(var)
                conf_var = max(0.0, min(1.0, 1.0 - 2.0 * std))
                conf_S   = S / (S + self.S_max)
                if self.conf_mode == "var":
                    conf = conf_var
                elif self.conf_mode == "strength":
                    conf = conf_S
                else:
                    conf = 0.5 * conf_var + 0.5 * conf_S
                return mean, var, std, conf, S

            with torch.no_grad():
                if hasattr(self.model, "predict_beta_params"):
                    out = self.model.predict_beta_params(state_tensor, lengths_np=[T_cur])
                    aA = float(out['A']['alpha']); bA = float(out['A']['beta'])
                    aB = float(out['B']['alpha']); bB = float(out['B']['beta'])
                    aI = float(out['I']['alpha']); bI = float(out['I']['beta'])
                else:
                    (aA_t,bA_t),(aB_t,bB_t),(aI_t,bI_t) = self.model(state_tensor, lengths=torch.tensor([T_cur]))
                    aA, bA = float(aA_t.item()), float(bA_t.item())
                    aB, bB = float(aB_t.item()), float(bB_t.item())
                    aI, bI = float(aI_t.item()), float(bI_t.item())

            meanA, varA, stdA, confA, SA = _conf_from_ab(aA, bA)
            meanB, varB, stdB, confB, SB = _conf_from_ab(aB, bB)
            meanI, varI, stdI, confI, SI = _conf_from_ab(aI, bI)

            # 即刻估计
            self.partner_A_prob = float(meanA)
            self.partner_B_prob = float(meanB)
            self.partner_I_prob = float(meanI)

            self.partner_A_conf = float(confA)
            self.partner_B_conf = float(confB)
            self.partner_I_conf = float(confI)

            # 维护平滑“后验”仅用于概率平滑（如需）
            def _update_posterior_for_smoothing(name, p, S_model):
                a_name = f'partner_{name}_alpha'
                b_name = f'partner_{name}_beta'
                if not hasattr(self, a_name): setattr(self, a_name, 1.0)
                if not hasattr(self, b_name): setattr(self, b_name, 1.0)
                setattr(self, a_name, getattr(self, a_name) * self.posterior_decay)
                setattr(self, b_name, getattr(self, b_name) * self.posterior_decay)
                kappa = min(S_model, self.kappa_max)
                setattr(self, a_name, getattr(self, a_name) + kappa * p)
                setattr(self, b_name, getattr(self, b_name) + kappa * (1.0 - p))
                A = getattr(self, a_name); B = getattr(self, b_name)
                S_post = A + B
                p_smooth = A / S_post if S_post > 0 else 0.5
                setattr(self, f'partner_{name}_prob_smooth', float(p_smooth))

            _update_posterior_for_smoothing('A', meanA, SA)
            _update_posterior_for_smoothing('B', meanB, SB)
            _update_posterior_for_smoothing('I', meanI, SI)

            # 若需要 ±1 标签
            # self.partner_A = 1 if self.partner_A_prob_smooth >= 0.5 else -1
            # self.partner_B = 1 if self.partner_B_prob_smooth >= 0.5 else -1
            # self.partner_I = 1 if self.partner_I_prob_smooth >= 0.5 else -1
            self.partner_A = self.partner_A_prob_smooth
            self.partner_B = self.partner_B_prob_smooth
            self.partner_I = self.partner_I_prob_smooth

            self.partner_A_confidence = self.partner_A_conf
            self.partner_B_confidence = self.partner_B_conf
            self.partner_I_confidence = self.partner_I_conf


        self.env_reset_step += 1

        if self.env_reset_step % self.reset_step == 0:
            self.env.soft_reset_obs_only()
            self.env.macroAgent[0].cur_macro_action_done = True
            self.env.macroAgent[1].cur_macro_action_done = True
            self.obs = self.env._get_macro_obs()

            
        return np.concatenate([
            self.obs[self.agent_index],
            [self.partner_A, self.partner_B, self.partner_I,
             self.partner_A_confidence, self.partner_B_confidence, self.partner_I_confidence]
        ]), rewards[self.agent_index] + rewards[1 - self.agent_index], dones, info
    






def _as_int_action(a):
    """Robustly cast SB3 action to a Python int."""
    if isinstance(a, (int, np.integer)):
        return int(a)
    a_np = np.asarray(a)
    if a_np.ndim == 0:        # numpy 0-d scalar
        return int(a_np.item())
    return int(a_np.flatten()[0])



# =========================
# 会话管理（并发安全）
# =========================
class Session:
    def __init__(self, config_id="layout_practice"):
        self.config_id = config_id
        self.env = None
        self.env_mac = None
        self.wrapper = None  # env_agent_0
        self.model = None
        self.obs = None
        self.cur_step = 0
        self.cumulative_reward = 0.0
        self.current_layout_id = None
        self.current_model_id = None
        self.robot_steps = []       # ← 新增：保存 robot 每一步
        self.last_access = time.time()
        self.lock = threading.RLock()

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

# =========================
# 工具
# =========================
_model_cache = {}

# 只有这4个 key 需要注入自定义 extractor 
_NEED_CUSTOM_OBJECTS = {"layout1_model1", "layout2_model1", "layout3_model1", "layout4_model1"}

# 模型缓存（只做只读推理，不再在共享模型上 set_env）
_model_cache = {}

def _load_or_get_model(model_key: str, env_for_model=None):
    """
    按模型 key（例如 'layout1_model1'）加载/复用模型。
    - 不再在共享模型上 set_env，也不在加载时绑定 env（避免并发时 env 被互相覆盖）。
    - 若 checkpoint 缺失 policy_kwargs 且该 key 需要自定义特征抽取器，则只注入 policy_kwargs，
      不注入 observation_space / action_space（否则会把结构“固定死”，与其他会话的维度不一致）。
    """
    if not model_key or model_key == "none":
        return None

    # 命中缓存：直接返回（不做 set_env）
    if model_key in _model_cache:
        return _model_cache[model_key]

    if model_key not in MODEL_PATHS:
        raise ValueError(f"Unknown model key '{model_key}'. Known: {list(MODEL_PATHS.keys())}")
    path = MODEL_PATHS[model_key]

    # 仅当需要自定义 extractor 的模型时，检查 checkpoint 是否包含 policy_kwargs
    if model_key in _NEED_CUSTOM_OBJECTS:
        try:
            data, params, pytorch_vars = load_from_zip_file(path, device="cpu")
        except Exception as e:
            print(f"⚠️ load_from_zip_file failed for {path}: {e}")
            data = {}
        saved_policy_kwargs = (data or {}).get("policy_kwargs", None)

        trained_policy_kwargs = dict(
            features_extractor_class=ABIGatedExtractorWithConf,
            features_extractor_kwargs={"base_dim": 64},
            net_arch=[{"pi": [128, 64], "vf": [128, 64]}],
        )

        try:
            if saved_policy_kwargs is None:
                # checkpoint 缺失 policy_kwargs → 只注入 policy_kwargs，且不绑定 env/不覆盖 spaces
                m = PPO.load(
                    path,
                    device="cpu",
                    custom_objects={
                        "policy_kwargs": trained_policy_kwargs,
                    },
                )

                try:
                    m.policy.set_training_mode(False)  # SB3 提供的接口
                except Exception:
                    pass
                try:
                    m.policy.eval()  # 兜底
                except Exception:
                    pass


            else:
                # checkpoint 自带 policy_kwargs → 直接加载（不绑定 env）
                m = PPO.load(path, device="cpu")
                try:
                    m.policy.set_training_mode(False)  # SB3 提供的接口
                except Exception:
                    pass
                try:
                    m.policy.eval()  # 兜底
                except Exception:
                    pass

    
        except Exception as e:
            print("❌ PPO.load failed (custom path):", e)
            print("▶ saved policy_kwargs:", saved_policy_kwargs)
            print("▶ trained policy_kwargs (used when missing):", trained_policy_kwargs)
            raise
    else:

        # 其他模型：默认加载（不绑定 env）
        try:
            m = PPO.load(path, device="cpu")
            try:
                m.policy.set_training_mode(False)  # SB3 提供的接口
            except Exception:
                pass
            try:
                m.policy.eval()  # 兜底
            except Exception:
                pass

        except Exception as e:
            print("❌ PPO.load failed (default path):", e)
            raise

    _model_cache[model_key] = m
    return m



# def _load_or_get_model(model_key: str, env_for_model):
#     """按 key 载入/复用 SB3 模型。model_key 为 None/'none' 时返回 None。"""
#     if not model_key or model_key == "none":
#         return None
#     if model_key in _model_cache:
#         m = _model_cache[model_key]
#         try:
#             m.set_env(env_for_model)   # 仅当 obs space 一致时生效
#         except Exception:
#             pass
#         return m
#     if model_key not in MODEL_PATHS:
#         raise ValueError(f"Unknown model key '{model_key}'. Known: {list(MODEL_PATHS.keys())}")
#     path = MODEL_PATHS[model_key]

#     print('model path: ', path)
#     print('model env: ', env_for_model)
#     m = PPO.load(path, env=env_for_model)
#     _model_cache[model_key] = m
#     return m

def _parse_config_id(layout_id: str = None, model_id: str = None, config_id: str = None) -> str:
    if config_id:
        return config_id
    if layout_id and model_id:
        return f"{layout_id}_{model_id}"
    raise ValueError("Either 'config_id' or both 'layout_id' and 'model_id' must be provided.")

def create_envs_for_session(sess: Session, config_id: str):
    """在会话内创建/切换环境与模型，并重置计数。"""
    if config_id not in LAYOUT_CONFIG:
        raise ValueError(f"Unknown config_id '{config_id}'. Known: {list(LAYOUT_CONFIG.keys())}")
    cfg = LAYOUT_CONFIG[config_id]

    grid_dim = cfg.get('grid_dim', [15, 15])
    n_agent = cfg.get('n_agent', 2)

    env_params = {
        'grid_dim': grid_dim,
        'task': ["lettuce salad"],
        'rewardList': rewardList,
        'map_type': cfg['map_type'],
        'n_agent': n_agent,
        'obs_radius': 0,
        'mode': "vector",
        'debug': True
    }

    # 关闭旧环境（避免资源泄露）
    if sess.env is not None:
        try: sess.env.close()
        except: pass
    if sess.env_mac is not None:
        try: sess.env_mac.close()
        except: pass

    # 1) 单环境
    sess.env = gym.make(cfg['env_id'], **env_params)
    _seed_env_everything(sess.env, SEED)        # ← 加上
    sess.env.reset()

    # 2) 多智能体
    sess.env_mac = gym.make(cfg['mac_env_id'], **env_params)
    _seed_env_everything(sess.env_mac, SEED)    # ← 加上

    # 3) 选择包装器

        
    # 根据 layout 决定 reset_step
    reset_step = 70 if config_id.startswith("layout4") else 100

    if config_id in ("layout2_model1", "layout3_model1", "layout4_model1"):
        abi_path = ABI_MODEL_PATHS.get(config_id, None)

        sess.wrapper = SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history30(
            sess.env_mac, agent_index=0, state_dim=6, reset_step=reset_step, model_path=abi_path
        )

    elif config_id in ("layout1_model1",):
        abi_path = ABI_MODEL_PATHS.get(config_id, None)

        sess.wrapper = SingleAgentWrapper_accept_keyboard_action_TrustPOMDP_history10(
            sess.env_mac, agent_index=0, state_dim=6, reset_step=reset_step, model_path=abi_path
        )

    else:
        sess.wrapper = SingleAgentWrapper_accept_keyboard_action(
            sess.env_mac, agent_index=0, reset_step=reset_step
        )



    # 4) 载入/复用 RL 模型
    model_key = cfg.get('model', None)
    sess.model = _load_or_get_model(model_key, sess.wrapper)  # 允许 None（练习局）

    # PRACTICE ROUND
    if config_id == "layout_practice":
        env = sess.env_mac.unwrapped 

        env.tomato = []
        env.onion = []
        env.badlettuce = []
        
        env.lettuce = [Lettuce(0, 1)]
        env.knife = [Knife(0, 2)]
        env.plate = [Plate(2, 0)]
        env.delivery = [Delivery(3, 4)]

        env.agent[0].x, env.agent[0].y = 0, 4  # top-right
        env.agent[1].x, env.agent[1].y = 4, 0  # bottom-left
        
        # env._updateMap()

    # 5) 初始 obs
    sess.obs = sess.wrapper.reset()

    # 记录
    sess.config_id = config_id
    try:
        parts = config_id.split('_')
        sess.current_layout_id = parts[0]
        sess.current_model_id  = parts[1] if len(parts) > 1 else None
    except:
        sess.current_layout_id = None
        sess.current_model_id = None

    # 重置计数
    sess.cur_step = 0
    sess.cumulative_reward = 0.0
    sess.robot_steps = []   # ← 新增：清空 robot 轨迹

# =========================
# 状态打包
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
# 路由
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

    layout_id = data.get('layout_id')
    model_id  = data.get('model_id')
    config_id = data.get('config_id')

    try:
        cfg_id = _parse_config_id(layout_id=layout_id, model_id=model_id, config_id=config_id)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400

    sess = SESSION_MGR.ensure(sid)
    with sess.lock:
        try:
            create_envs_for_session(sess, cfg_id)
        except Exception as e:
            return jsonify(success=False, error=str(e)), 400

        steps_left = MAX_STEPS
        cfg = LAYOUT_CONFIG[cfg_id]
        print(f"[RESET][{sid}] cfg={cfg_id} map={cfg['map_type']} mac_env={cfg['mac_env_id']} model={cfg.get('model')} "
              f"grid_dim={cfg.get('grid_dim')} n_agent={cfg.get('n_agent')}")

        return jsonify(
            success=True,
            state=extract_state(sess),
            steps_left=steps_left,
            cumulative_reward=sess.cumulative_reward,
            config_id=sess.config_id,
            layout_id=sess.current_layout_id,
            model_id=sess.current_model_id
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
        # 可选：带 config 的请求触发热切换
        if layout_id or model_id or config_id:
            try:
                target_cfg_id = _parse_config_id(layout_id=layout_id, model_id=model_id, config_id=config_id)
                if target_cfg_id != sess.config_id:
                    create_envs_for_session(sess, target_cfg_id)
                    print(f"[HOT-SWITCH][{sid}] -> {target_cfg_id}")
            except Exception as e:
                return jsonify(success=False, error=f"hot switch failed: {e}"), 400

        KEYS_ACTIONS = {'ArrowUp':3,'ArrowRight':0,'ArrowDown':1,'ArrowLeft':2}
        if key in KEYS_ACTIONS:
            # 0 号 AI；1 号为人
            if True:
                t0 = time.time()
                if sess.model is not None:
                    with torch.no_grad():
                        ai_action, _ = sess.model.predict(sess.obs, deterministic=True)
                    ai_action_int = _as_int_action(ai_action)   # ← 规范成 int
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

                # 记录 robot 步骤
                robot_low = int(action[0])
                robot_key = ACTION_TO_KEY.get(robot_low, "Unknown")
                sess.robot_steps.append({
                    "step": int(sess.cur_step + 1),
                    "ai_macro_action": ai_action_int,   # ← 不再索引 [0]
                    "low_level_action": robot_low,
                    "arrow": robot_key,
                    "timestamp": time.time(),
                })



                sess.obs, rewards, dones, info = sess.wrapper.step(action[0], action[1])
                # _, reward_env, done_env, info_env = sess.env.step(action)
                t3 = time.time()

                try:
                    sess.cumulative_reward += float(rewards)
                except Exception:
                    pass

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
            robot_last_action=(sess.robot_steps[-1] if sess.robot_steps else None)  # ← 新增（可选）
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
    """接收前端上传的完整 logData JSON，保存并返回完成码。"""
    try:
        data = request.get_json(silent=True) or {}
        log_payload = data.get('log', data)  # 兼容两种格式
        if not isinstance(log_payload, dict) or 'rounds' not in log_payload:
            return jsonify(success=False, error="Invalid payload: 'rounds' missing"), 400

        # completion_code = uuid.uuid4().hex[:8].upper()
        # 固定完成码
        completion_code = "C108AMXR"
        # completion_code = "CK3UXUQJ"


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

    app.run(host='0.0.0.0', port=5000, debug=False)
